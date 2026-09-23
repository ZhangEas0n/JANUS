import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    if "pgd" in cfg and key in cfg.pgd:
        return cfg.pgd[key]
    return default


def _safe_progress_total(dataloader, max_batches=None):
    if max_batches is not None:
        return int(max_batches)
    try:
        return int(len(dataloader))
    except (TypeError, ValueError, NotImplementedError):
        return None


def _as_lengths(input_lengths, batch_size, num_samples, device):
    if input_lengths is None:
        return torch.full((batch_size,), num_samples, dtype=torch.long, device=device)
    if torch.is_tensor(input_lengths):
        return input_lengths.to(device=device, dtype=torch.long)
    return torch.tensor(input_lengths, dtype=torch.long, device=device)


def _lengths_as_int_list(input_lengths, batch_size, num_samples):
    if input_lengths is None:
        return [int(num_samples)] * int(batch_size)
    if torch.is_tensor(input_lengths):
        return [int(length) for length in input_lengths.detach().cpu().tolist()]
    return [int(length) for length in input_lengths]


def _valid_mask_like(x, input_lengths=None):
    if x.dim() == 3:
        batch_size, _, num_samples = x.shape
    elif x.dim() == 2:
        batch_size, num_samples = x.shape
    else:
        raise ValueError(f"Expected waveform batch with 2D or 3D shape, got {x.shape}")

    lengths = _as_lengths(input_lengths, batch_size, num_samples, x.device)
    arange = torch.arange(num_samples, device=x.device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    if x.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.to(dtype=x.dtype)


def _normalize_for_sr(x_raw, input_lengths=None):
    if input_lengths is not None and x_raw.dim() == 2:
        return normalize_batch_like_pipeline(x_raw, input_lengths=input_lengths)
    return normalize_batch_like_pipeline(x_raw)


def _pgd_sensitive_attack(
    sr_model,
    intent_head,
    x_clean_raw,
    input_lengths,
    sensitive_class_ids,
    epsilon,
    alpha,
    steps,
    random_start,
    optimize_mask=None,
):
    sr_model.eval()
    intent_head.eval()
    valid_mask = _valid_mask_like(x_clean_raw, input_lengths)
    if random_start:
        delta = torch.empty_like(x_clean_raw).uniform_(-epsilon, epsilon)
        delta = delta * valid_mask
    else:
        delta = torch.zeros_like(x_clean_raw)

    if optimize_mask is not None:
        optimize_mask = optimize_mask.to(device=x_clean_raw.device, dtype=torch.bool)
    model_lengths = _lengths_as_int_list(
        input_lengths,
        batch_size=x_clean_raw.shape[0],
        num_samples=x_clean_raw.shape[-1],
    )

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        adv_raw = torch.clamp(x_clean_raw + delta * valid_mask, -1.0, 1.0)
        adv_input = _normalize_for_sr(adv_raw, input_lengths=model_lengths)
        embedding, embedding_lengths = sr_model.compute_speech_embedding(
            adv_input,
            model_lengths,
        )
        logits = intent_head(embedding, embedding_lengths)
        p_sens = torch.softmax(logits, dim=-1)[:, sensitive_class_ids].sum(dim=-1)
        if optimize_mask is not None and optimize_mask.any():
            p_sens = p_sens[optimize_mask]
        loss = -torch.log(p_sens.clamp_min(1e-8)).mean()

        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta - alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon) * valid_mask
        delta = torch.clamp(x_clean_raw + delta, -1.0, 1.0) - x_clean_raw
        delta = delta * valid_mask

    return torch.clamp(x_clean_raw + delta.detach(), -1.0, 1.0)


def _pgd_sensitive_attack_chunked(
    sr_model,
    intent_head,
    x_clean_raw,
    input_lengths,
    sensitive_class_ids,
    epsilon,
    alpha,
    steps,
    random_start,
    max_attack_samples,
    optimize_mask=None,
    show_chunk_progress=False,
):
    if max_attack_samples is None or x_clean_raw.shape[-1] <= max_attack_samples:
        return _pgd_sensitive_attack(
            sr_model=sr_model,
            intent_head=intent_head,
            x_clean_raw=x_clean_raw,
            input_lengths=input_lengths,
            sensitive_class_ids=sensitive_class_ids,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            optimize_mask=optimize_mask,
        )

    model_lengths = _lengths_as_int_list(
        input_lengths,
        batch_size=x_clean_raw.shape[0],
        num_samples=x_clean_raw.shape[-1],
    )
    adv_raw = x_clean_raw.detach().clone()
    min_chunk_samples = 400

    for sample_idx, valid_length in enumerate(model_lengths):
        if optimize_mask is not None and not bool(optimize_mask[sample_idx]):
            continue

        start = 0
        num_chunks = (valid_length + max_attack_samples - 1) // max_attack_samples
        tail_samples = valid_length % max_attack_samples
        if 0 < tail_samples < min_chunk_samples and num_chunks > 1:
            num_chunks -= 1
        chunk_pbar = tqdm(
            total=num_chunks,
            desc=f"PGD chunks sample {sample_idx + 1}/{len(model_lengths)}",
            leave=False,
            disable=not show_chunk_progress or num_chunks <= 1,
        )
        while start < valid_length:
            end = min(start + max_attack_samples, valid_length)
            remaining = valid_length - end
            if 0 < remaining < min_chunk_samples:
                end = valid_length

            chunk = x_clean_raw[sample_idx : sample_idx + 1, ..., start:end]
            adv_chunk = _pgd_sensitive_attack(
                sr_model=sr_model,
                intent_head=intent_head,
                x_clean_raw=chunk,
                input_lengths=[end - start],
                sensitive_class_ids=sensitive_class_ids,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                random_start=random_start,
            )
            adv_raw[sample_idx : sample_idx + 1, ..., start:end] = adv_chunk
            del chunk, adv_chunk
            start = end
            chunk_pbar.update(1)
            chunk_pbar.set_postfix(samples=f"{start}/{valid_length}")
        chunk_pbar.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return adv_raw


def _perturbation_stats(clean, adv, input_lengths=None):
    mask = _valid_mask_like(clean, input_lengths)
    delta = (adv - clean) * mask
    denom = mask.sum().clamp_min(1.0)
    linf = delta.abs().max().item()
    rms = (delta.pow(2).sum() / denom).sqrt().item()
    signal_power = ((clean * mask).pow(2).sum() / denom).clamp_min(1e-12)
    noise_power = (delta.pow(2).sum() / denom).clamp_min(1e-12)
    snr_db = (10.0 * torch.log10(signal_power / noise_power)).item()
    return linf, rms, snr_db


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
    sr_model,
    intent_head,
    dataloader,
    device,
    sensitive_class_ids,
    epsilon,
    alpha,
    steps,
    random_start,
    max_attack_samples,
    use_pgd,
    max_batches=None,
):
    samples = []
    cosine_sum = 0.0
    cosine_count = 0
    linf_sum = 0.0
    rms_sum = 0.0
    snr_sum = 0.0
    stat_batches = 0

    total_batches = _safe_progress_total(dataloader, max_batches)
    pbar = tqdm(total=total_batches, desc="SV PGD eval", leave=True)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        x_clean_raw = batch.network_input
        if use_pgd:
            input_lengths = _as_lengths(
                None,
                x_clean_raw.shape[0],
                x_clean_raw.shape[-1],
                device,
            )
            adv_raw = _pgd_sensitive_attack_chunked(
                sr_model=sr_model,
                intent_head=intent_head,
                x_clean_raw=x_clean_raw,
                input_lengths=input_lengths,
                sensitive_class_ids=sensitive_class_ids,
                epsilon=epsilon,
                alpha=alpha,
                steps=steps,
                random_start=random_start,
                max_attack_samples=max_attack_samples,
                show_chunk_progress=True,
            )
            linf, rms, snr = _perturbation_stats(x_clean_raw, adv_raw, input_lengths)
            linf_sum += linf
            rms_sum += rms
            snr_sum += snr
            stat_batches += 1
            x_eval_raw = adv_raw
        else:
            x_eval_raw = x_clean_raw

        with torch.no_grad():
            x_eval = normalize_batch_like_pipeline(x_eval_raw)
            embedding = victim_model.compute_speaker_embedding(x_eval)
            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)

            if use_pgd:
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
        pbar.set_postfix(mode="attack" if use_pgd else "clean")
        del batch, x_clean_raw, x_eval_raw, embedding
        if use_pgd:
            del adv_raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    mean_embedding_cosine = cosine_sum / cosine_count if cosine_count > 0 else None
    stats = None
    if stat_batches > 0:
        stats = {
            "linf": linf_sum / stat_batches,
            "rms": rms_sum / stat_batches,
            "snr_db": snr_sum / stat_batches,
        }
    return samples, mean_embedding_cosine, stats


def evaluate_sv_pgd(
    victim_model,
    sr_model,
    intent_head,
    evaluator,
    dataloader,
    pairs,
    device,
    sensitive_class_ids,
    epsilon,
    alpha,
    steps,
    random_start,
    max_attack_samples,
    max_batches=None,
):
    clean_samples, _, _ = _collect_sv_embeddings(
        victim_model=victim_model,
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=dataloader,
        device=device,
        sensitive_class_ids=sensitive_class_ids,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        max_attack_samples=max_attack_samples,
        use_pgd=False,
        max_batches=max_batches,
    )
    adv_samples, mean_cosine, stats = _collect_sv_embeddings(
        victim_model=victim_model,
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=dataloader,
        device=device,
        sensitive_class_ids=sensitive_class_ids,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        max_attack_samples=max_attack_samples,
        use_pgd=True,
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
    cos_drop = 1.0 - mean_cosine if mean_cosine is not None else None

    results = {
        "num_pairs": len(ground_truth),
        "clean_eer": clean_eer,
        "adv_eer": adv_eer,
        "adv_threshold": adv_threshold,
        "eer_delta": adv_eer - clean_eer,
        "far_at_clean_threshold": far,
        "frr_at_clean_threshold": frr,
        "asr_sv_at_clean_threshold": asr_sv,
        "mean_embedding_cosine": mean_cosine,
        "cosine_similarity_drop": cos_drop,
        **(stats or {}),
    }

    print(
        "[*] SV PGD eval: "
        f"pairs={results['num_pairs']}, clean_eer={clean_eer:.4f}, "
        f"adv_eer={adv_eer:.4f}, eer_delta={results['eer_delta']:+.4f}, "
        f"far={far:.4f}, frr={frr:.4f}, asr_sv={asr_sv:.4f}, "
        f"mean_cos={mean_cosine:.4f}, cos_drop={cos_drop:.4f}, "
        f"linf={results.get('linf', 0.0):.5f}, rms={results.get('rms', 0.0):.5f}, "
        f"snr_db={results.get('snr_db', 0.0):.2f}"
    )
    return results


def evaluate_sr_pgd(
    sr_model,
    intent_head,
    dataloader,
    sensitive_intent_labels,
    sensitive_keywords,
    sensitive_class_ids,
    device,
    epsilon,
    alpha,
    steps,
    random_start,
    max_attack_samples,
    decision_threshold,
    max_batches=None,
):
    clean_probabilities = []
    adv_probabilities = []
    linf_sum = 0.0
    rms_sum = 0.0
    snr_sum = 0.0
    stat_batches = 0
    batches_seen = 0

    total_batches = _safe_progress_total(dataloader, max_batches)
    pbar = tqdm(total=total_batches, desc="SR PGD eval", leave=True)

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
            pbar.update(1)
            pbar.set_postfix(benign=0, attacked=stat_batches)
            continue

        with torch.no_grad():
            clean_embedding, clean_lengths = sr_model.compute_speech_embedding(
                batch.network_input,
                batch.input_lengths,
            )
            clean_logits = intent_head(clean_embedding, clean_lengths)
            clean_p_sens = torch.softmax(clean_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            raw_audio = extract_raw_audio_batch(batch, device)
        model_lengths = _lengths_as_int_list(
            batch.input_lengths,
            batch_size=raw_audio.shape[0],
            num_samples=raw_audio.shape[-1],
        )
        adv_raw = _pgd_sensitive_attack_chunked(
            sr_model=sr_model,
            intent_head=intent_head,
            x_clean_raw=raw_audio,
            input_lengths=model_lengths,
            sensitive_class_ids=sensitive_class_ids,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            max_attack_samples=max_attack_samples,
            optimize_mask=benign_mask,
            show_chunk_progress=True,
        )
        linf, rms, snr = _perturbation_stats(raw_audio, adv_raw, model_lengths)
        linf_sum += linf
        rms_sum += rms
        snr_sum += snr
        stat_batches += 1

        with torch.no_grad():
            adv_input = normalize_batch_like_pipeline(adv_raw, input_lengths=model_lengths)
            adv_embedding, adv_lengths = sr_model.compute_speech_embedding(
                adv_input,
                model_lengths,
            )
            adv_logits = intent_head(adv_embedding, adv_lengths)
            adv_p_sens = torch.softmax(adv_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

        clean_probabilities.append(clean_p_sens[benign_mask].cpu())
        adv_probabilities.append(adv_p_sens[benign_mask].cpu())
        del batch, raw_audio, adv_raw, adv_input, adv_embedding, adv_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pbar.update(1)
        pbar.set_postfix(
            benign=int(benign_mask.sum().item()),
            attacked=stat_batches,
            p_adv=f"{adv_p_sens[benign_mask].mean().item():.3f}",
        )

    pbar.close()

    if not clean_probabilities:
        raise ValueError("No benign samples available for SR PGD evaluation.")

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
        "[*] SR PGD eval on true benign: "
        f"batches={results['batches']}, samples={results['benign_samples']}, "
        f"threshold={decision_threshold:.2f}, p_clean={results['clean_mean_p_sens']:.4f}, "
        f"p_adv={results['adv_mean_p_sens']:.4f}, delta_p={results['delta_mean_p_sens']:+.4f}, "
        f"clean_rate={results['clean_sensitive_rate']:.4f}, "
        f"adv_rate={results['adv_sensitive_rate']:.4f}, "
        f"rate_delta={results['rate_increase']:+.4f}, flip={results['flip_rate']:.4f}, "
        f"cond_flip={results['conditional_flip_rate']:.4f}, "
        f"linf={results['linf']:.5f}, rms={results['rms']:.5f}, "
        f"snr_db={results['snr_db']:.2f}"
    )
    return results


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epsilon = float(_cfg_get(cfg, "epsilon", 0.05))
    alpha = float(_cfg_get(cfg, "alpha", 0.005))
    steps = int(_cfg_get(cfg, "steps", 10))
    random_start = bool(_cfg_get(cfg, "random_start", True))
    max_attack_samples = _cfg_get(cfg, "max_attack_samples", 48000)
    max_attack_samples = (
        None if max_attack_samples is None else int(max_attack_samples)
    )
    if max_attack_samples is not None and max_attack_samples <= 0:
        raise ValueError("pgd.max_attack_samples must be positive or null.")
    if max_attack_samples is not None and max_attack_samples < 400:
        raise ValueError("pgd.max_attack_samples must be at least 400 for wav2vec2.")
    run_sv = bool(_cfg_get(cfg, "eval_sv", True))
    run_sr = bool(_cfg_get(cfg, "eval_sr", True))
    sv_max_batches = _cfg_get(cfg, "sv_max_batches", attack_cfg.sv_eval_max_batches)
    sr_max_batches = _cfg_get(cfg, "sr_max_batches", attack_cfg.sr_eval_max_batches)
    sv_max_batches = None if sv_max_batches is None else int(sv_max_batches)
    sr_max_batches = None if sr_max_batches is None else int(sr_max_batches)
    print(
        "[*] PGD baseline: "
        f"epsilon={epsilon}, alpha={alpha}, steps={steps}, random_start={random_start}, "
        f"max_attack_samples={max_attack_samples}, "
        f"eval_sv={run_sv}, eval_sr={run_sr}, "
        f"sv_max_batches={sv_max_batches}, sr_max_batches={sr_max_batches}"
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")

    print("[*] Initializing SR surrogate and intent head...")
    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    sr_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
    sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2)
    evaluator = instantiate(cfg.evaluator)
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

    if run_sv:
        print("[*] Initializing SV evaluation...")
        sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        sv_cfg.data.pipeline.test_pipeline = []
        sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
        sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        sv_dm = construct_data_module(sv_cfg)
        sv_dataloader = sv_dm.test_dataloader()
        victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
        victim_model.eval()
        for param in victim_model.parameters():
            param.requires_grad = False

        print("[*] Running SV PGD transfer evaluation...")
        evaluate_sv_pgd(
            victim_model=victim_model,
            sr_model=sr_model,
            intent_head=intent_head,
            evaluator=evaluator,
            dataloader=sv_dataloader,
            pairs=sv_dm.test_pairs,
            device=device,
            sensitive_class_ids=attack_cfg.sensitive_class_ids,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            max_attack_samples=max_attack_samples,
            max_batches=sv_max_batches,
        )
        if run_sr:
            print("[*] Releasing SV victim model before SR PGD evaluation...")
            victim_model.to("cpu")
            del victim_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if run_sr:
        print("[*] Running SR PGD evaluation on FSC test benign samples...")
        evaluate_sr_pgd(
            sr_model=sr_model,
            intent_head=intent_head,
            dataloader=sr_test_dataloader,
            sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
            sensitive_keywords=attack_cfg.sensitive_keywords,
            sensitive_class_ids=attack_cfg.sensitive_class_ids,
            device=device,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            max_attack_samples=max_attack_samples,
            decision_threshold=attack_cfg.intent_decision_threshold,
            max_batches=sr_max_batches,
        )


if __name__ == "__main__":
    main_evaluation()
