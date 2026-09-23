import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.config import BTUAPAttackConfig
from btuap.common import normalize_batch_like_pipeline
from btuap.sr import (
    extract_raw_audio_batch,
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
    if "muting" in cfg and key in cfg.muting:
        return cfg.muting[key]
    return default


def _lengths_as_int_list(input_lengths, batch_size, num_samples):
    if input_lengths is None:
        return [int(num_samples)] * int(batch_size)
    if torch.is_tensor(input_lengths):
        return [int(length) for length in input_lengths.detach().cpu().tolist()]
    return [int(length) for length in input_lengths]


def load_attack_prefix(path, device, scale=1.0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Muting Whisper attack segment: {path}")

    prefix = torch.from_numpy(np.load(str(path))).float().reshape(-1)
    prefix = torch.clamp(prefix * float(scale), -1.0, 1.0).to(device)
    print(
        "[*] Loaded Muting Whisper prefix: "
        f"path={path}, samples={prefix.numel()}, duration={prefix.numel() / 16000:.3f}s, "
        f"scale={scale}, linf={prefix.abs().max().item():.5f}, "
        f"rms={prefix.pow(2).mean().sqrt().item():.5f}"
    )
    return prefix


def prepend_attack(waveforms, prefix, input_lengths=None):
    if waveforms.dim() == 3:
        if waveforms.shape[1] != 1:
            raise ValueError(f"Expected mono waveform batch, got {waveforms.shape}")
        squeezed = waveforms[:, 0, :]
        restore_channel = True
    elif waveforms.dim() == 2:
        squeezed = waveforms
        restore_channel = False
    else:
        raise ValueError(f"Expected waveform batch with 2D or 3D shape, got {waveforms.shape}")

    lengths = _lengths_as_int_list(
        input_lengths,
        batch_size=squeezed.shape[0],
        num_samples=squeezed.shape[-1],
    )
    attacked_lengths = [length + prefix.numel() for length in lengths]
    attacked = torch.zeros(
        (squeezed.shape[0], max(attacked_lengths)),
        dtype=squeezed.dtype,
        device=squeezed.device,
    )

    for idx, valid_length in enumerate(lengths):
        attacked[idx, : prefix.numel()] = prefix.to(dtype=squeezed.dtype)
        attacked[idx, prefix.numel() : attacked_lengths[idx]] = squeezed[idx, :valid_length]

    if restore_channel:
        attacked = attacked.unsqueeze(1)
    return attacked, attacked_lengths


def _prefix_snr_db(clean, prefix, input_lengths=None):
    if clean.dim() == 3:
        clean = clean[:, 0, :]
    lengths = _lengths_as_int_list(input_lengths, clean.shape[0], clean.shape[-1])
    prefix_energy = prefix.pow(2).sum().clamp_min(1e-12)
    snrs = []
    for idx, valid_length in enumerate(lengths):
        signal_energy = clean[idx, :valid_length].pow(2).sum().clamp_min(1e-12)
        snrs.append(10.0 * torch.log10(signal_energy / prefix_energy))
    return torch.stack(snrs).mean().item()


def _compute_scores_for_pairs(evaluator, pairs, samples):
    sample_map = {sample.sample_id: sample for sample in samples}
    ground_truth = []
    prediction_pairs = []
    for pair in pairs:
        if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
            continue
        ground_truth.append(1 if pair.same_speaker else 0)
        prediction_pairs.append((sample_map[pair.sample1_id], sample_map[pair.sample2_id]))

    raw_scores = evaluator._compute_prediction_scores(prediction_pairs)
    scores = torch.tensor(raw_scores, dtype=torch.float32)
    scores = torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0)
    return ground_truth, scores.tolist()


def evaluate_sv_muting(
    victim_model,
    evaluator,
    dataloader,
    pairs,
    device,
    prefix,
    max_batches=None,
):
    clean_samples = []
    adv_samples = []
    cosine_sum = 0.0
    cosine_count = 0
    snr_sum = 0.0
    stat_batches = 0

    victim_model.eval()
    with torch.no_grad():
        pbar = tqdm(desc="SV Muting Whisper embeddings", leave=False)
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = batch.to(device)
            clean_raw = batch.network_input
            adv_raw, _ = prepend_attack(clean_raw, prefix)
            clean_embedding = victim_model.compute_speaker_embedding(
                normalize_batch_like_pipeline(clean_raw)
            )
            adv_embedding = victim_model.compute_speaker_embedding(
                normalize_batch_like_pipeline(adv_raw)
            )
            if clean_embedding.dim() == 1:
                clean_embedding = clean_embedding.unsqueeze(0)
            if adv_embedding.dim() == 1:
                adv_embedding = adv_embedding.unsqueeze(0)

            cosine = F.cosine_similarity(clean_embedding, adv_embedding, dim=-1)
            cosine_sum += cosine.sum().item()
            cosine_count += cosine.numel()
            snr_sum += _prefix_snr_db(clean_raw, prefix)
            stat_batches += 1

            for idx, sample_id in enumerate(batch.keys):
                clean_samples.append(
                    EmbeddingSample(sample_id=sample_id, embedding=clean_embedding[idx].cpu())
                )
                adv_samples.append(
                    EmbeddingSample(sample_id=sample_id, embedding=adv_embedding[idx].cpu())
                )
            pbar.update(1)
        pbar.close()

    ground_truth, clean_scores = _compute_scores_for_pairs(evaluator, pairs, clean_samples)
    _, adv_scores = _compute_scores_for_pairs(evaluator, pairs, adv_samples)
    clean_eer, clean_threshold = calculate_eer(ground_truth, clean_scores, pos_label=1)
    adv_eer, adv_threshold = calculate_eer(ground_truth, adv_scores, pos_label=1)

    negative_scores = [score for gt, score in zip(ground_truth, adv_scores) if gt == 0]
    positive_scores = [score for gt, score in zip(ground_truth, adv_scores) if gt == 1]
    far = sum(score >= clean_threshold for score in negative_scores) / max(1, len(negative_scores))
    frr = sum(score < clean_threshold for score in positive_scores) / max(1, len(positive_scores))
    asr_sv = sum(
        ((gt == 0) and (score >= clean_threshold)) or ((gt == 1) and (score < clean_threshold))
        for gt, score in zip(ground_truth, adv_scores)
    ) / max(1, len(ground_truth))
    mean_cos = cosine_sum / max(1, cosine_count)

    results = {
        "num_pairs": len(ground_truth),
        "clean_eer": clean_eer,
        "adv_eer": adv_eer,
        "adv_threshold": adv_threshold,
        "eer_delta": adv_eer - clean_eer,
        "far_at_clean_threshold": far,
        "frr_at_clean_threshold": frr,
        "asr_sv_at_clean_threshold": asr_sv,
        "mean_embedding_cosine": mean_cos,
        "cosine_similarity_drop": 1.0 - mean_cos,
        "prefix_linf": prefix.abs().max().item(),
        "prefix_rms": prefix.pow(2).mean().sqrt().item(),
        "prefix_snr_db": snr_sum / max(1, stat_batches),
    }
    print(
        "[*] SV Muting Whisper eval: "
        f"pairs={results['num_pairs']}, clean_eer={clean_eer:.4f}, "
        f"adv_eer={adv_eer:.4f}, eer_delta={results['eer_delta']:+.4f}, "
        f"far={far:.4f}, frr={frr:.4f}, asr_sv={asr_sv:.4f}, "
        f"mean_cos={mean_cos:.4f}, cos_drop={results['cosine_similarity_drop']:.4f}, "
        f"prefix_linf={results['prefix_linf']:.5f}, "
        f"prefix_rms={results['prefix_rms']:.5f}, "
        f"prefix_snr_db={results['prefix_snr_db']:.2f}"
    )
    return results


def evaluate_sr_muting(
    sr_model,
    intent_head,
    dataloader,
    sensitive_intent_labels,
    sensitive_keywords,
    sensitive_class_ids,
    device,
    prefix,
    decision_threshold,
    max_batches=None,
):
    clean_probabilities = []
    adv_probabilities = []
    snr_sum = 0.0
    stat_batches = 0
    batches_seen = 0

    sr_model.eval()
    intent_head.eval()
    with torch.no_grad():
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
            benign_mask = target == 0
            if not benign_mask.any():
                continue

            clean_embedding, clean_lengths = sr_model.compute_speech_embedding(
                batch.network_input,
                batch.input_lengths,
            )
            clean_logits = intent_head(clean_embedding, clean_lengths)
            clean_p_sens = torch.softmax(clean_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            raw_audio = extract_raw_audio_batch(batch, device)
            adv_raw, adv_lengths = prepend_attack(raw_audio, prefix, batch.input_lengths)
            adv_input = normalize_batch_like_pipeline(adv_raw, input_lengths=adv_lengths)
            adv_embedding, adv_embedding_lengths = sr_model.compute_speech_embedding(
                adv_input,
                adv_lengths,
            )
            adv_logits = intent_head(adv_embedding, adv_embedding_lengths)
            adv_p_sens = torch.softmax(adv_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            clean_probabilities.append(clean_p_sens[benign_mask].cpu())
            adv_probabilities.append(adv_p_sens[benign_mask].cpu())
            snr_sum += _prefix_snr_db(raw_audio, prefix, batch.input_lengths)
            stat_batches += 1

    if not clean_probabilities:
        raise ValueError("No benign samples available for Muting Whisper SR evaluation.")

    p_clean = torch.cat(clean_probabilities)
    p_adv = torch.cat(adv_probabilities)
    clean_sensitive = p_clean >= decision_threshold
    adv_sensitive = p_adv >= decision_threshold
    eligible = ~clean_sensitive
    flipped = eligible & adv_sensitive

    results = {
        "batches": batches_seen,
        "benign_samples": int(p_clean.numel()),
        "clean_mean_p_sens": p_clean.mean().item(),
        "adv_mean_p_sens": p_adv.mean().item(),
        "delta_mean_p_sens": (p_adv - p_clean).mean().item(),
        "clean_sensitive_rate": clean_sensitive.float().mean().item(),
        "adv_sensitive_rate": adv_sensitive.float().mean().item(),
        "rate_increase": (
            adv_sensitive.float().mean() - clean_sensitive.float().mean()
        ).item(),
        "flip_rate": flipped.float().mean().item(),
        "conditional_flip_rate": flipped.float().sum().item() / max(1, eligible.sum().item()),
        "prefix_linf": prefix.abs().max().item(),
        "prefix_rms": prefix.pow(2).mean().sqrt().item(),
        "prefix_snr_db": snr_sum / max(1, stat_batches),
    }
    print(
        "[*] SR Muting Whisper eval on true benign: "
        f"batches={batches_seen}, samples={results['benign_samples']}, "
        f"threshold={decision_threshold:.2f}, "
        f"p_clean={results['clean_mean_p_sens']:.4f}, "
        f"p_adv={results['adv_mean_p_sens']:.4f}, "
        f"delta_p={results['delta_mean_p_sens']:+.4f}, "
        f"clean_rate={results['clean_sensitive_rate']:.4f}, "
        f"adv_rate={results['adv_sensitive_rate']:.4f}, "
        f"rate_delta={results['rate_increase']:+.4f}, "
        f"flip={results['flip_rate']:.4f}, "
        f"cond_flip={results['conditional_flip_rate']:.4f}, "
        f"prefix_linf={results['prefix_linf']:.5f}, "
        f"prefix_rms={results['prefix_rms']:.5f}, "
        f"prefix_snr_db={results['prefix_snr_db']:.2f}"
    )
    return results


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix_path = _cfg_get(
        cfg,
        "prefix_path",
        str(Path(project_root) / "prepend_acoustic_attack-main" / "audio_attack_segments" / "base.en.np.npy"),
    )
    prefix_scale = float(_cfg_get(cfg, "scale", 1.0))
    run_sv = bool(_cfg_get(cfg, "eval_sv", True))
    run_sr = bool(_cfg_get(cfg, "eval_sr", True))
    sv_max_batches = _cfg_get(cfg, "sv_max_batches", attack_cfg.sv_eval_max_batches)
    sr_max_batches = _cfg_get(cfg, "sr_max_batches", attack_cfg.sr_eval_max_batches)
    sv_max_batches = None if sv_max_batches is None else int(sv_max_batches)
    sr_max_batches = None if sr_max_batches is None else int(sr_max_batches)
    prefix = load_attack_prefix(prefix_path, device=device, scale=prefix_scale)

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")

    evaluator = instantiate(cfg.evaluator)
    if run_sv:
        print("[*] Initializing SV evaluation...")
        sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        sv_cfg.data.pipeline.test_pipeline = []
        sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
        sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        sv_dm = construct_data_module(sv_cfg)
        victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
        victim_model.eval()
        for param in victim_model.parameters():
            param.requires_grad = False

        evaluate_sv_muting(
            victim_model=victim_model,
            evaluator=evaluator,
            dataloader=sv_dm.test_dataloader(),
            pairs=sv_dm.test_pairs,
            device=device,
            prefix=prefix,
            max_batches=sv_max_batches,
        )
        if run_sr:
            victim_model.to("cpu")
            del victim_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if run_sr:
        print("[*] Initializing SR evaluation...")
        sr_cfg = make_speech_recognition_cfg(
            cfg,
            train_max_num_samples=attack_cfg.sr_train_max_num_samples,
            train_batch_size=attack_cfg.sr_train_batch_size,
            project_root=project_root,
        )
        sr_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2)
        sr_dm = construct_data_module(sr_cfg)
        present_sensitive_labels = [
            label
            for label in attack_cfg.sensitive_intent_labels
            if label in sr_dm.intent_label_to_id
        ]
        if not present_sensitive_labels:
            raise ValueError("No configured sensitive intent labels exist in the FSC dataset.")
        attack_cfg.sensitive_intent_labels = present_sensitive_labels
        attack_cfg.sensitive_class_ids = [1]

        sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
        sr_model.eval()
        for param in sr_model.parameters():
            param.requires_grad = False
        intent_head = load_sensitive_intent_head_checkpoint(
            checkpoint_path=attack_cfg.intent_head_path,
            sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
            decision_threshold=attack_cfg.intent_decision_threshold,
            device=device,
        )
        evaluate_sr_muting(
            sr_model=sr_model,
            intent_head=intent_head,
            dataloader=sr_dm.test_dataloader(),
            sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
            sensitive_keywords=attack_cfg.sensitive_keywords,
            sensitive_class_ids=attack_cfg.sensitive_class_ids,
            device=device,
            prefix=prefix,
            decision_threshold=attack_cfg.intent_decision_threshold,
            max_batches=sr_max_batches,
        )


if __name__ == "__main__":
    main_evaluation()
