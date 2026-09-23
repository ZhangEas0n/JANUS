import csv
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

from btuap.common import apply_uap, normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
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
    if "sv_sweep" in cfg and key in cfg.sv_sweep:
        return cfg.sv_sweep[key]
    return default


def _safe_progress_total(dataloader, max_batches=None):
    if max_batches is not None:
        return int(max_batches)
    try:
        return int(len(dataloader))
    except (TypeError, ValueError, NotImplementedError):
        return None


def load_uap(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing UAP tensor to evaluate: {path}")
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict):
        for key in ("uap_delta", "delta", "perturbation", "uap"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsupported UAP checkpoint content: {type(payload)}")
    uap = payload.detach().float().to(device)
    if uap.dim() == 1:
        uap = uap.unsqueeze(0)
    if uap.dim() == 3 and uap.shape[1] == 1:
        uap = uap[:, 0, :]
    if uap.dim() != 2:
        raise ValueError(f"Expected UAP shape [1, samples], got {tuple(uap.shape)}")
    print(
        "[*] Loaded UAP: "
        f"path={path}, shape={tuple(uap.shape)}, "
        f"linf={uap.abs().max().item():.5f}, "
        f"rms={uap.pow(2).mean().sqrt().item():.5f}"
    )
    return uap


def collect_embeddings(victim_model, dataloader, device, uap_delta=None, max_batches=None):
    victim_model.eval()
    samples = []
    total = _safe_progress_total(dataloader, max_batches)
    desc = "SV adv embeddings" if uap_delta is not None else "SV clean embeddings"

    with torch.no_grad():
        pbar = tqdm(total=total, desc=desc, leave=True)
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            raw_audio = batch.network_input
            eval_raw = apply_uap(raw_audio, uap_delta) if uap_delta is not None else raw_audio
            eval_input = normalize_batch_like_pipeline(eval_raw)
            embeddings = victim_model.compute_speaker_embedding(eval_input)
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
    return samples


def compute_pair_scores(evaluator, pairs, clean_samples, adv_samples):
    clean_map = {sample.sample_id: sample for sample in clean_samples}
    adv_map = {sample.sample_id: sample for sample in adv_samples}
    kept_pairs = []
    clean_prediction_pairs = []
    adv_prediction_pairs = []

    for pair in pairs:
        if (
            pair.sample1_id not in clean_map
            or pair.sample2_id not in clean_map
            or pair.sample1_id not in adv_map
            or pair.sample2_id not in adv_map
        ):
            continue
        kept_pairs.append(pair)
        clean_prediction_pairs.append((clean_map[pair.sample1_id], clean_map[pair.sample2_id]))
        adv_prediction_pairs.append((adv_map[pair.sample1_id], adv_map[pair.sample2_id]))

    if not kept_pairs:
        raise ValueError("No complete SV pairs were found. Increase sv_sweep.max_batches.")

    clean_scores = torch.tensor(
        evaluator._compute_prediction_scores(clean_prediction_pairs),
        dtype=torch.float32,
    )
    adv_scores = torch.tensor(
        evaluator._compute_prediction_scores(adv_prediction_pairs),
        dtype=torch.float32,
    )
    clean_scores = torch.clamp((clean_scores + 1.0) / 2.0, 0.0, 1.0)
    adv_scores = torch.clamp((adv_scores + 1.0) / 2.0, 0.0, 1.0)

    records = []
    for index, (pair, clean_score, adv_score) in enumerate(
        zip(kept_pairs, clean_scores.tolist(), adv_scores.tolist())
    ):
        records.append(
            {
                "pair_index": index,
                "sample1_id": pair.sample1_id,
                "sample2_id": pair.sample2_id,
                "same_speaker": int(pair.same_speaker),
                "clean_score": float(clean_score),
                "adv_score": float(adv_score),
                "score_delta": float(adv_score - clean_score),
            }
        )
    return records


def metrics_at_threshold(records, threshold, score_key):
    tp = tn = fp = fn = 0
    for record in records:
        gt = int(record["same_speaker"])
        prediction = 1 if float(record[score_key]) >= threshold else 0
        if gt == 1 and prediction == 1:
            tp += 1
        elif gt == 1 and prediction == 0:
            fn += 1
        elif gt == 0 and prediction == 1:
            fp += 1
        else:
            tn += 1

    positives = tp + fn
    negatives = fp + tn
    total = positives + negatives
    tpr = tp / max(1, positives)
    fpr = fp / max(1, negatives)
    far = fpr
    frr = fn / max(1, positives)
    accuracy = (tp + tn) / max(1, total)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "far": far,
        "frr": frr,
        "accuracy": accuracy,
    }


def sweep_thresholds(records, thresholds):
    rows = []
    for threshold in thresholds:
        clean = metrics_at_threshold(records, threshold, "clean_score")
        adv = metrics_at_threshold(records, threshold, "adv_score")
        attack_successes = sum(
            (
                int(record["same_speaker"]) == 0
                and float(record["adv_score"]) >= threshold
            )
            or (
                int(record["same_speaker"]) == 1
                and float(record["adv_score"]) < threshold
            )
            for record in records
        )
        rows.append(
            {
                "threshold": float(threshold),
                "pairs": len(records),
                "clean_far": clean["far"],
                "clean_frr": clean["frr"],
                "clean_tpr": clean["tpr"],
                "clean_fpr": clean["fpr"],
                "clean_accuracy": clean["accuracy"],
                "adv_far": adv["far"],
                "adv_frr": adv["frr"],
                "adv_tpr": adv["tpr"],
                "adv_fpr": adv["fpr"],
                "adv_accuracy": adv["accuracy"],
                "asr_sv": attack_successes / max(1, len(records)),
                "tp": adv["tp"],
                "tn": adv["tn"],
                "fp": adv["fp"],
                "fn": adv["fn"],
            }
        )
    return rows


def make_thresholds(records, fixed_thresholds, sweep_all_scores):
    thresholds = [float(value) for value in fixed_thresholds]
    if sweep_all_scores:
        all_scores = []
        for record in records:
            all_scores.append(float(record["clean_score"]))
            all_scores.append(float(record["adv_score"]))
        thresholds.extend(all_scores)
    thresholds = sorted(set(max(0.0, min(1.0, threshold)) for threshold in thresholds))
    return thresholds


def save_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[*] Saved: {path}")


def print_fixed_threshold_table(rows, fixed_thresholds):
    fixed = {float(value) for value in fixed_thresholds}
    print("\n[*] SV threshold sensitivity")
    print(
        f"{'tau':>6} {'FAR':>8} {'FRR':>8} {'TPR':>8} "
        f"{'FPR':>8} {'ASR-SV':>8} {'clean_FAR':>10} {'clean_FRR':>10}"
    )
    for row in rows:
        if float(row["threshold"]) not in fixed:
            continue
        print(
            f"{row['threshold']:>6.2f} "
            f"{row['adv_far']:>8.4f} {row['adv_frr']:>8.4f} "
            f"{row['adv_tpr']:>8.4f} {row['adv_fpr']:>8.4f} "
            f"{row['asr_sv']:>8.4f} "
            f"{row['clean_far']:>10.4f} {row['clean_frr']:>10.4f}"
        )


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_batches = _cfg_get(cfg, "max_batches", attack_cfg.sv_eval_max_batches)
    max_batches = None if max_batches is None else int(max_batches)
    fixed_thresholds = _cfg_get(cfg, "thresholds", [0.5, 0.6, 0.7, 0.8, 0.9])
    sweep_all_scores = bool(_cfg_get(cfg, "sweep_all_scores", True))
    output_prefix = _cfg_get(
        cfg,
        "output_prefix",
        str(Path(project_root) / "results" / "ablation" / "threshold" / "sv_threshold_sweep"),
    )
    uap_path = _cfg_get(
        cfg,
        "uap_path",
        attack_cfg.evaluation_uap_path or attack_cfg.output_path,
    )

    print(
        "[*] SV threshold sweep: "
        f"max_batches={max_batches}, fixed_thresholds={fixed_thresholds}, "
        f"sweep_all_scores={sweep_all_scores}"
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 backbone: {local_wav2vec2}")

    uap_delta = load_uap(uap_path, device)

    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.data.pipeline.test_pipeline = []
    sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
    sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
    sv_dm = construct_data_module(sv_cfg)
    evaluator = instantiate(sv_cfg.evaluator)
    victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    victim_model.eval()
    for parameter in victim_model.parameters():
        parameter.requires_grad = False

    dataloader = sv_dm.test_dataloader()
    clean_samples = collect_embeddings(
        victim_model=victim_model,
        dataloader=dataloader,
        device=device,
        uap_delta=None,
        max_batches=max_batches,
    )
    adv_samples = collect_embeddings(
        victim_model=victim_model,
        dataloader=sv_dm.test_dataloader(),
        device=device,
        uap_delta=uap_delta,
        max_batches=max_batches,
    )

    pair_records = compute_pair_scores(
        evaluator=evaluator,
        pairs=sv_dm.test_pairs,
        clean_samples=clean_samples,
        adv_samples=adv_samples,
    )
    thresholds = make_thresholds(pair_records, fixed_thresholds, sweep_all_scores)
    sweep_rows = sweep_thresholds(pair_records, thresholds)
    clean_eer, clean_eer_threshold = calculate_eer(
        [record["same_speaker"] for record in pair_records],
        [record["clean_score"] for record in pair_records],
        pos_label=1,
    )
    adv_eer, adv_eer_threshold = calculate_eer(
        [record["same_speaker"] for record in pair_records],
        [record["adv_score"] for record in pair_records],
        pos_label=1,
    )

    output_prefix = Path(output_prefix)
    save_csv(
        output_prefix.with_name(output_prefix.name + "_pair_scores.csv"),
        pair_records,
        [
            "pair_index",
            "sample1_id",
            "sample2_id",
            "same_speaker",
            "clean_score",
            "adv_score",
            "score_delta",
        ],
    )
    save_csv(
        output_prefix.with_name(output_prefix.name + "_metrics.csv"),
        sweep_rows,
        [
            "threshold",
            "pairs",
            "clean_far",
            "clean_frr",
            "clean_tpr",
            "clean_fpr",
            "clean_accuracy",
            "adv_far",
            "adv_frr",
            "adv_tpr",
            "adv_fpr",
            "adv_accuracy",
            "asr_sv",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )

    summary = {
        "num_pairs": len(pair_records),
        "fixed_thresholds": [float(value) for value in fixed_thresholds],
        "sweep_all_scores": sweep_all_scores,
        "clean_eer": float(clean_eer),
        "clean_eer_threshold": float(clean_eer_threshold),
        "adv_eer": float(adv_eer),
        "adv_eer_threshold": float(adv_eer_threshold),
        "uap_path": str(uap_path),
    }
    json_path = output_prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "metrics": sweep_rows}, handle, indent=2)
    print(f"[*] Saved: {json_path}")
    print(
        "[*] EER summary: "
        f"clean_eer={clean_eer:.4f} @ {clean_eer_threshold:.4f}, "
        f"adv_eer={adv_eer:.4f} @ {adv_eer_threshold:.4f}"
    )
    print_fixed_threshold_table(sweep_rows, fixed_thresholds)


if __name__ == "__main__":
    main_evaluation()
