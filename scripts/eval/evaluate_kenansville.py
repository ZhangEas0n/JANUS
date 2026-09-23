import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import sys

import hydra
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
    if "kenan" in cfg and key in cfg.kenan:
        return cfg.kenan[key]
    return default


def _lengths_to_device(input_lengths, device):
    if input_lengths is None:
        return None
    if torch.is_tensor(input_lengths):
        return input_lengths.to(device)
    return torch.tensor(input_lengths, device=device)


def _valid_lengths_from_batch(waveforms, input_lengths=None):
    if input_lengths is None:
        return [waveforms.shape[-1]] * waveforms.shape[0]
    if torch.is_tensor(input_lengths):
        lengths = input_lengths.detach().cpu().tolist()
    else:
        lengths = list(input_lengths)
    return [int(length) for length in lengths]


def apply_kenansville_fft(waveforms, input_lengths=None, factor=0.1):
    """Zero low-magnitude FFT bins per utterance.

    factor <= 1 is interpreted as a ratio of each utterance's max FFT magnitude.
    factor > 1 is interpreted as an absolute FFT magnitude threshold.
    """
    outputs = waveforms.clone()
    lengths = _valid_lengths_from_batch(waveforms, input_lengths)

    for idx, valid_len in enumerate(lengths):
        valid_len = max(1, min(valid_len, waveforms.shape[-1]))
        signal = waveforms[idx, :valid_len].detach().cpu()
        spectrum = torch.fft.fft(signal)
        threshold = float(factor)
        if threshold <= 1.0:
            threshold = threshold * spectrum.abs().max().item()
        spectrum = torch.where(
            spectrum.abs() < threshold,
            torch.zeros_like(spectrum),
            spectrum,
        )
        reconstructed = torch.fft.ifft(spectrum).real.to(
            device=waveforms.device,
            dtype=waveforms.dtype,
        )
        outputs[idx, :valid_len] = reconstructed.clamp(-1.0, 1.0)

    return outputs


def apply_kenansville_ssa(waveforms, input_lengths=None, factor=10.0, project_root=None):
    """Apply Kenansville SSA reconstruction per utterance.

    factor is the percent of SSA components kept, matching the original code's
    percent-style parameter.
    """
    import numpy as np

    kenan_dir = Path(project_root or ".") / "kenansville_attack-master"
    if str(kenan_dir) not in sys.path:
        sys.path.insert(0, str(kenan_dir))
    from ssa_core import inv_ssa, ssa

    outputs = waveforms.clone()
    lengths = _valid_lengths_from_batch(waveforms, input_lengths)

    for idx, valid_len in enumerate(lengths):
        valid_len = max(1, min(valid_len, waveforms.shape[-1]))
        data = waveforms[idx, :valid_len].detach().cpu().numpy()
        window = int(len(data) * 0.05)
        window = max(2, min(window, 3000, len(data) - 1))
        keep = int(window * float(factor) / 100.0)
        keep = max(1, min(keep, window))
        pc, _, v = ssa(data, window)
        reconstructed = inv_ssa(pc, v, np.arange(0, keep, 1))
        reconstructed = torch.as_tensor(
            reconstructed,
            dtype=waveforms.dtype,
            device=waveforms.device,
        )
        outputs[idx, :valid_len] = reconstructed[:valid_len].clamp(-1.0, 1.0)

    return outputs


def apply_kenansville(waveforms, input_lengths, attack, fft_factor, ssa_factor, project_root):
    if attack == "fft":
        return apply_kenansville_fft(waveforms, input_lengths=input_lengths, factor=fft_factor)
    if attack == "ssa":
        return apply_kenansville_ssa(
            waveforms,
            input_lengths=input_lengths,
            factor=ssa_factor,
            project_root=project_root,
        )
    raise ValueError(f"Unknown Kenansville attack '{attack}'. Expected 'fft' or 'ssa'.")


def _compute_perturbation_stats(clean, adv, input_lengths=None):
    lengths = _valid_lengths_from_batch(clean, input_lengths)
    linf_values = []
    rms_values = []
    snr_values = []
    for idx, valid_len in enumerate(lengths):
        valid_len = max(1, min(valid_len, clean.shape[-1]))
        delta = adv[idx, :valid_len] - clean[idx, :valid_len]
        signal = clean[idx, :valid_len]
        linf_values.append(delta.abs().max())
        rms_values.append(delta.pow(2).mean().sqrt())
        noise_power = delta.pow(2).mean().clamp_min(1e-12)
        signal_power = signal.pow(2).mean().clamp_min(1e-12)
        snr_values.append(10.0 * torch.log10(signal_power / noise_power))
    return (
        torch.stack(linf_values).mean().item(),
        torch.stack(rms_values).mean().item(),
        torch.stack(snr_values).mean().item(),
    )


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


def _collect_sv_embeddings(
    victim_model,
    dataloader,
    device,
    attack=None,
    fft_factor=0.1,
    ssa_factor=10.0,
    project_root=None,
    max_batches=None,
):
    samples = []
    cosine_sum = 0.0
    cosine_count = 0
    linf_sum = 0.0
    rms_sum = 0.0
    snr_sum = 0.0
    stat_batches = 0

    with torch.no_grad():
        pbar = tqdm(desc="SV eval embeddings", leave=False)
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = batch.to(device)
            x_clean_raw = batch.network_input
            if attack is None:
                x_eval_raw = x_clean_raw
            else:
                x_eval_raw = apply_kenansville(
                    x_clean_raw,
                    input_lengths=None,
                    attack=attack,
                    fft_factor=fft_factor,
                    ssa_factor=ssa_factor,
                    project_root=project_root,
                )
                linf, rms, snr = _compute_perturbation_stats(x_clean_raw, x_eval_raw)
                linf_sum += linf
                rms_sum += rms
                snr_sum += snr
                stat_batches += 1

            x_eval = normalize_batch_like_pipeline(x_eval_raw)
            embedding = victim_model.compute_speaker_embedding(x_eval)
            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)

            if attack is not None:
                x_clean = normalize_batch_like_pipeline(x_clean_raw)
                clean_embedding = victim_model.compute_speaker_embedding(x_clean)
                if len(clean_embedding.shape) == 1:
                    clean_embedding = clean_embedding.unsqueeze(0)
                cosine = F.cosine_similarity(clean_embedding, embedding, dim=-1)
                cosine_sum += cosine.sum().item()
                cosine_count += cosine.numel()

            for idx, sample_id in enumerate(batch.keys):
                samples.append(
                    EmbeddingSample(
                        sample_id=sample_id,
                        embedding=embedding[idx].detach().cpu(),
                    )
                )

            pbar.update(1)
        pbar.close()

    mean_embedding_cosine = cosine_sum / cosine_count if cosine_count > 0 else None
    perturbation_stats = None
    if stat_batches > 0:
        perturbation_stats = {
            "linf": linf_sum / stat_batches,
            "rms": rms_sum / stat_batches,
            "snr_db": snr_sum / stat_batches,
        }
    return samples, mean_embedding_cosine, perturbation_stats


def evaluate_sv_kenansville(
    victim_model,
    evaluator,
    dataloader,
    pairs,
    device,
    attack,
    fft_factor,
    ssa_factor,
    project_root,
    max_batches=None,
):
    victim_model.eval()
    clean_samples, _, _ = _collect_sv_embeddings(
        victim_model=victim_model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )
    adv_samples, mean_embedding_cosine, perturbation_stats = _collect_sv_embeddings(
        victim_model=victim_model,
        dataloader=dataloader,
        device=device,
        attack=attack,
        fft_factor=fft_factor,
        ssa_factor=ssa_factor,
        project_root=project_root,
        max_batches=max_batches,
    )

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
    cosine_drop = 1.0 - mean_embedding_cosine if mean_embedding_cosine is not None else None

    results = {
        "num_pairs": len(ground_truth),
        "clean_eer": clean_eer,
        "clean_threshold": clean_threshold,
        "adv_eer": adv_eer,
        "adv_threshold": adv_threshold,
        "eer_delta": adv_eer - clean_eer,
        "far_at_clean_threshold": far,
        "frr_at_clean_threshold": frr,
        "asr_sv_at_clean_threshold": asr_sv,
        "mean_embedding_cosine": mean_embedding_cosine,
        "cosine_similarity_drop": cosine_drop,
    }
    if perturbation_stats is not None:
        results.update(perturbation_stats)

    print(
        "[*] SV Kenansville eval: "
        f"attack={attack}, pairs={results['num_pairs']}, "
        f"clean_eer={results['clean_eer']:.4f}, "
        f"adv_eer={results['adv_eer']:.4f}, "
        f"eer_delta={results['eer_delta']:+.4f}, "
        f"far={results['far_at_clean_threshold']:.4f}, "
        f"frr={results['frr_at_clean_threshold']:.4f}, "
        f"asr_sv={results['asr_sv_at_clean_threshold']:.4f}, "
        f"mean_cos={results['mean_embedding_cosine']:.4f}, "
        f"cos_drop={results['cosine_similarity_drop']:.4f}, "
        f"linf={results.get('linf', 0.0):.5f}, "
        f"rms={results.get('rms', 0.0):.5f}, "
        f"snr_db={results.get('snr_db', 0.0):.2f}"
    )
    return results


def evaluate_sr_kenansville(
    sr_model,
    intent_head,
    dataloader,
    sensitive_intent_labels,
    sensitive_keywords,
    sensitive_class_ids,
    device,
    attack,
    fft_factor,
    ssa_factor,
    project_root,
    decision_threshold,
    max_batches=None,
):
    sr_model.eval()
    intent_head.eval()
    clean_probabilities = []
    adv_probabilities = []
    linf_sum = 0.0
    rms_sum = 0.0
    snr_sum = 0.0
    stat_batches = 0
    batches_seen = 0

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
            input_lengths = _lengths_to_device(batch.input_lengths, device)
            adv_raw = apply_kenansville(
                raw_audio,
                input_lengths=input_lengths,
                attack=attack,
                fft_factor=fft_factor,
                ssa_factor=ssa_factor,
                project_root=project_root,
            )
            linf, rms, snr = _compute_perturbation_stats(raw_audio, adv_raw, input_lengths)
            linf_sum += linf
            rms_sum += rms
            snr_sum += snr
            stat_batches += 1
            adv_input = normalize_batch_like_pipeline(
                adv_raw,
                input_lengths=batch.input_lengths,
            )
            adv_embedding, adv_lengths = sr_model.compute_speech_embedding(
                adv_input,
                batch.input_lengths,
            )
            adv_logits = intent_head(adv_embedding, adv_lengths)
            adv_p_sens = torch.softmax(adv_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            clean_probabilities.append(clean_p_sens[benign_mask].cpu())
            adv_probabilities.append(adv_p_sens[benign_mask].cpu())

    if not clean_probabilities:
        raise ValueError("No benign samples available for SR Kenansville evaluation.")

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
        "conditional_flip_rate": flipped.float().sum().item() / max(1, eligible.float().sum().item()),
        "eligible_clean_benign": int(eligible.float().sum().item()),
        "linf": linf_sum / max(1, stat_batches),
        "rms": rms_sum / max(1, stat_batches),
        "snr_db": snr_sum / max(1, stat_batches),
    }

    print(
        "[*] SR Kenansville eval on true benign: "
        f"attack={attack}, batches={results['batches']}, "
        f"samples={results['benign_samples']}, "
        f"threshold={decision_threshold:.2f}, "
        f"p_clean={results['clean_mean_p_sens']:.4f}, "
        f"p_adv={results['adv_mean_p_sens']:.4f}, "
        f"delta_p={results['delta_mean_p_sens']:+.4f}, "
        f"clean_rate={results['clean_sensitive_rate']:.4f}, "
        f"adv_rate={results['adv_sensitive_rate']:.4f}, "
        f"rate_delta={results['rate_increase']:+.4f}, "
        f"flip={results['flip_rate']:.4f}, "
        f"cond_flip={results['conditional_flip_rate']:.4f}, "
        f"linf={results['linf']:.5f}, "
        f"rms={results['rms']:.5f}, "
        f"snr_db={results['snr_db']:.2f}"
    )
    return results


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kenan_attack = str(_cfg_get(cfg, "attack", "fft")).lower()
    fft_factor = float(_cfg_get(cfg, "fft_factor", 0.1))
    ssa_factor = float(_cfg_get(cfg, "ssa_factor", 10.0))
    print(
        "[*] Kenansville baseline: "
        f"attack={kenan_attack}, fft_factor={fft_factor}, ssa_factor={ssa_factor}"
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")

    print("[*] Initializing SV evaluation...")
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.data.pipeline.test_pipeline = []
    sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
    sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
    print(f"[*] Using local SV wav2vec2 backbone from {local_wav2vec2}")
    sv_dm = construct_data_module(sv_cfg)
    sv_dataloader = sv_dm.test_dataloader()
    evaluator = instantiate(sv_cfg.evaluator)
    victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False

    print("[*] Running SV Kenansville evaluation...")
    evaluate_sv_kenansville(
        victim_model=victim_model,
        evaluator=evaluator,
        dataloader=sv_dataloader,
        pairs=sv_dm.test_pairs,
        device=device,
        attack=kenan_attack,
        fft_factor=fft_factor,
        ssa_factor=ssa_factor,
        project_root=project_root,
        max_batches=attack_cfg.sv_eval_max_batches,
    )

    print("[*] Initializing SR evaluation...")
    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    sr_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
    sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2)
    print(f"[*] Using local SR wav2vec2 backbone from {local_wav2vec2}")
    print(f"[*] Using local SR tokenizer from {sr_cfg.tokenizer.tokenizer_huggingface_id}")
    sr_dm = construct_data_module(sr_cfg)
    sr_test_dataloader = sr_dm.test_dataloader()

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

    print("[*] Running SR Kenansville evaluation on FSC test benign samples...")
    evaluate_sr_kenansville(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=sr_test_dataloader,
        sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        sensitive_class_ids=attack_cfg.sensitive_class_ids,
        device=device,
        attack=kenan_attack,
        fft_factor=fft_factor,
        ssa_factor=ssa_factor,
        project_root=project_root,
        decision_threshold=attack_cfg.intent_decision_threshold,
        max_batches=attack_cfg.sr_eval_max_batches,
    )


if __name__ == "__main__":
    main_evaluation()
