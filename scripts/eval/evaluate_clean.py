import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.common import normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.sr import (
    _binary_sensitive_metrics,
    extract_sensitive_binary_targets,
    load_sensitive_intent_head_checkpoint,
    make_speech_recognition_cfg,
)
from src.eval_metrics import calculate_eer
from src.evaluation.speaker.speaker_recognition_evaluator import EmbeddingSample
from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)
from src.main import construct_data_module, construct_module

load_dotenv()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)


def _cfg_get(cfg: DictConfig, key: str, default):
    if "clean" in cfg and key in cfg.clean:
        return cfg.clean[key]
    return default


def _safe_progress_total(dataloader, max_batches=None):
    if max_batches is not None:
        return int(max_batches)
    try:
        return int(len(dataloader))
    except (TypeError, ValueError, NotImplementedError):
        return None


def _compute_scores_for_pairs(evaluator, pairs, samples):
    sample_map = {sample.sample_id: sample for sample in samples}
    ground_truth = []
    prediction_pairs = []
    for pair in pairs:
        if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
            continue
        ground_truth.append(1 if pair.same_speaker else 0)
        prediction_pairs.append((sample_map[pair.sample1_id], sample_map[pair.sample2_id]))

    if not prediction_pairs:
        raise ValueError(
            "No complete SV pairs were found. Increase clean.sv_max_batches or use null."
        )
    raw_scores = evaluator._compute_prediction_scores(prediction_pairs)
    scores = torch.tensor(raw_scores, dtype=torch.float32)
    return ground_truth, torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0).tolist()


def evaluate_clean_sv(victim_model, evaluator, dataloader, pairs, device, max_batches=None):
    samples = []
    total = _safe_progress_total(dataloader, max_batches)
    victim_model.eval()

    with torch.no_grad():
        pbar = tqdm(total=total, desc="Clean SV embeddings", leave=True)
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            clean_input = normalize_batch_like_pipeline(batch.network_input)
            embeddings = victim_model.compute_speaker_embedding(clean_input)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            for idx, sample_id in enumerate(batch.keys):
                samples.append(
                    EmbeddingSample(
                        sample_id=sample_id,
                        embedding=embeddings[idx].detach().cpu(),
                    )
                )
            pbar.update(1)
        pbar.close()

    ground_truth, scores = _compute_scores_for_pairs(evaluator, pairs, samples)
    eer, threshold = calculate_eer(ground_truth, scores, pos_label=1)
    negative_scores = [score for gt, score in zip(ground_truth, scores) if gt == 0]
    positive_scores = [score for gt, score in zip(ground_truth, scores) if gt == 1]
    far = sum(score >= threshold for score in negative_scores) / max(1, len(negative_scores))
    frr = sum(score < threshold for score in positive_scores) / max(1, len(positive_scores))
    accuracy = sum(
        ((gt == 0) and (score < threshold)) or ((gt == 1) and (score >= threshold))
        for gt, score in zip(ground_truth, scores)
    ) / max(1, len(ground_truth))

    results = {
        "samples": len(samples),
        "pairs": len(ground_truth),
        "eer": float(eer),
        "threshold": float(threshold),
        "far_at_eer_threshold": float(far),
        "frr_at_eer_threshold": float(frr),
        "accuracy_at_eer_threshold": float(accuracy),
    }
    print(
        "[*] Clean SV eval: "
        f"samples={results['samples']}, pairs={results['pairs']}, "
        f"eer={results['eer']:.4f}, threshold={results['threshold']:.4f}, "
        f"far={results['far_at_eer_threshold']:.4f}, "
        f"frr={results['frr_at_eer_threshold']:.4f}, "
        f"acc={results['accuracy_at_eer_threshold']:.4f}"
    )
    return results


def evaluate_clean_sr(
    sr_model,
    intent_head,
    dataloader,
    sensitive_intent_labels,
    sensitive_keywords,
    device,
    decision_threshold,
    max_batches=None,
):
    targets = []
    probabilities = []
    batches_seen = 0
    total = _safe_progress_total(dataloader, max_batches)
    sr_model.eval()
    intent_head.eval()

    with torch.no_grad():
        pbar = tqdm(total=total, desc="Clean SR eval", leave=True)
        for batch in dataloader:
            if max_batches is not None and batches_seen >= max_batches:
                break
            batches_seen += 1
            batch = batch.to(device)
            target = extract_sensitive_binary_targets(
                batch=batch,
                sensitive_intent_labels=sensitive_intent_labels,
                sensitive_keywords=sensitive_keywords,
                device=device,
            )
            embedding, embedding_lengths = sr_model.compute_speech_embedding(
                batch.network_input,
                batch.input_lengths,
            )
            logits = intent_head(embedding, embedding_lengths)
            p_sensitive = torch.softmax(logits, dim=-1)[:, 1]
            targets.append(target.detach().cpu())
            probabilities.append(p_sensitive.detach().cpu())
            pbar.update(1)
        pbar.close()

    if not targets:
        raise ValueError("No samples available for clean SR evaluation.")
    target = torch.cat(targets)
    p_sensitive = torch.cat(probabilities)
    results = _binary_sensitive_metrics(target, p_sensitive, decision_threshold)
    benign_mask = target == 0
    sensitive_mask = target == 1
    results.update(
        {
            "batches": batches_seen,
            "benign_mean_p_sensitive": float(p_sensitive[benign_mask].mean().item()),
            "sensitive_mean_p_sensitive": float(p_sensitive[sensitive_mask].mean().item()),
            "benign_false_sensitive_rate": float(
                (p_sensitive[benign_mask] >= decision_threshold).float().mean().item()
            ),
        }
    )
    print(
        "[*] Clean SR eval: "
        f"batches={results['batches']}, samples={results['samples']}, "
        f"threshold={results['decision_threshold']:.2f}, "
        f"acc={results['acc']:.4f}, balanced_acc={results['balanced_acc']:.4f}, "
        f"neg_acc={results['neg_acc']:.4f}, sens_acc={results['sens_acc']:.4f}, "
        f"sens_precision={results['sens_precision']:.4f}, "
        f"sens_recall={results['sens_recall']:.4f}, sens_f1={results['sens_f1']:.4f}, "
        f"benign_p_sens={results['benign_mean_p_sensitive']:.4f}, "
        f"benign_false_rate={results['benign_false_sensitive_rate']:.4f}, "
        f"neg={results['neg']}, sens={results['sens']}"
    )
    return results


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_sv = bool(_cfg_get(cfg, "eval_sv", True))
    run_sr = bool(_cfg_get(cfg, "eval_sr", True))
    sv_max_batches = _cfg_get(cfg, "sv_max_batches", attack_cfg.sv_eval_max_batches)
    sr_max_batches = _cfg_get(cfg, "sr_max_batches", attack_cfg.sr_eval_max_batches)
    sv_max_batches = None if sv_max_batches is None else int(sv_max_batches)
    sr_max_batches = None if sr_max_batches is None else int(sr_max_batches)
    output_path = Path(
        _cfg_get(
            cfg,
            "output_path",
            str(Path(project_root) / "results" / "main" / "clean_evaluation.json"),
        )
    )
    print(
        "[*] Clean baseline: "
        f"eval_sv={run_sv}, eval_sr={run_sr}, "
        f"sv_max_batches={sv_max_batches}, sr_max_batches={sr_max_batches}"
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")
    evaluator = instantiate(cfg.evaluator)
    all_results = {}

    if run_sv:
        print("[*] Initializing clean SV evaluation...")
        sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        sv_cfg.data.pipeline.test_pipeline = []
        sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
        sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        sv_dm = construct_data_module(sv_cfg)
        victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
        for parameter in victim_model.parameters():
            parameter.requires_grad = False
        all_results["sv"] = evaluate_clean_sv(
            victim_model=victim_model,
            evaluator=evaluator,
            dataloader=sv_dm.test_dataloader(),
            pairs=sv_dm.test_pairs,
            device=device,
            max_batches=sv_max_batches,
        )
        victim_model.to("cpu")
        del victim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if run_sr:
        print("[*] Initializing clean SR evaluation...")
        sr_cfg = make_speech_recognition_cfg(
            cfg,
            train_max_num_samples=attack_cfg.sr_train_max_num_samples,
            train_batch_size=attack_cfg.sr_train_batch_size,
            project_root=project_root,
        )
        sr_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2)
        sr_dm = construct_data_module(sr_cfg)
        sensitive_labels = [
            label
            for label in attack_cfg.sensitive_intent_labels
            if label in sr_dm.intent_label_to_id
        ]
        if not sensitive_labels:
            raise ValueError("No configured sensitive intent labels exist in FSC.")
        sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
        for parameter in sr_model.parameters():
            parameter.requires_grad = False
        intent_head = load_sensitive_intent_head_checkpoint(
            checkpoint_path=attack_cfg.intent_head_path,
            sensitive_intent_labels=sensitive_labels,
            decision_threshold=attack_cfg.intent_decision_threshold,
            device=device,
        )
        all_results["sr"] = evaluate_clean_sr(
            sr_model=sr_model,
            intent_head=intent_head,
            dataloader=sr_dm.test_dataloader(),
            sensitive_intent_labels=sensitive_labels,
            sensitive_keywords=attack_cfg.sensitive_keywords,
            device=device,
            decision_threshold=attack_cfg.intent_decision_threshold,
            max_batches=sr_max_batches,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"[*] Saved clean evaluation results: {output_path}")


if __name__ == "__main__":
    main_evaluation()
