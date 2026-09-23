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
    if "fakebob" in cfg and key in cfg.fakebob:
        return cfg.fakebob[key]
    return default


def _safe_progress_total(dataloader, max_batches=None):
    if max_batches is not None:
        return int(max_batches)
    try:
        return int(len(dataloader))
    except (TypeError, ValueError, NotImplementedError):
        return None


def _speaker_embeddings(victim_model, raw_audio):
    normalized = normalize_batch_like_pipeline(raw_audio)
    embeddings = victim_model.compute_speaker_embedding(normalized)
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    return F.normalize(embeddings, p=2, dim=-1)


def _query_cosine_losses(
    victim_model,
    candidate_audio,
    clean_embedding,
    query_batch_size,
):
    losses = []
    with torch.no_grad():
        for start in range(0, candidate_audio.shape[0], query_batch_size):
            end = min(start + query_batch_size, candidate_audio.shape[0])
            embeddings = _speaker_embeddings(victim_model, candidate_audio[start:end])
            reference = clean_embedding.expand(embeddings.shape[0], -1)
            losses.append(F.cosine_similarity(embeddings, reference, dim=-1))
    return torch.cat(losses, dim=0)


def fakebob_untargeted_attack(
    victim_model,
    clean_raw,
    epsilon,
    max_iter,
    learning_rate,
    min_learning_rate,
    samples_per_draw,
    sigma,
    momentum,
    plateau_length,
    plateau_drop,
    target_cosine,
    query_batch_size,
):
    if clean_raw.shape[0] != 1:
        raise ValueError("FAKEBOB baseline currently expects SV test batch size 1.")
    if samples_per_draw < 2:
        raise ValueError("fakebob.samples_per_draw must be at least 2.")

    if samples_per_draw % 2 != 0:
        samples_per_draw += 1

    with torch.no_grad():
        clean_embedding = _speaker_embeddings(victim_model, clean_raw)

    adv_raw = clean_raw.detach().clone()
    lower = torch.clamp(clean_raw - epsilon, -1.0, 1.0)
    upper = torch.clamp(clean_raw + epsilon, -1.0, 1.0)
    grad_momentum = torch.zeros_like(clean_raw)
    learning_rate_current = learning_rate
    recent_losses = []
    total_queries = 1
    final_cosine = 1.0
    iterations_used = 0

    for iteration in range(max_iter):
        half = samples_per_draw // 2
        noise_pos = torch.randn(
            (half,) + tuple(clean_raw.shape[1:]),
            dtype=clean_raw.dtype,
            device=clean_raw.device,
        )
        noise = torch.cat([noise_pos, -noise_pos], dim=0)
        candidates = torch.clamp(adv_raw + sigma * noise, -1.0, 1.0)
        losses = _query_cosine_losses(
            victim_model=victim_model,
            candidate_audio=candidates,
            clean_embedding=clean_embedding,
            query_batch_size=query_batch_size,
        )
        total_queries += candidates.shape[0]

        estimate_grad = (losses.view(-1, 1, 1) * noise).mean(dim=0, keepdim=True) / sigma
        grad_momentum = momentum * grad_momentum + (1.0 - momentum) * estimate_grad
        adv_raw = adv_raw - learning_rate_current * grad_momentum.sign()
        adv_raw = torch.maximum(torch.minimum(adv_raw, upper), lower)
        adv_raw = torch.clamp(adv_raw, -1.0, 1.0).detach()

        with torch.no_grad():
            adv_embedding = _speaker_embeddings(victim_model, adv_raw)
            final_cosine = F.cosine_similarity(
                adv_embedding,
                clean_embedding,
                dim=-1,
            ).item()
        total_queries += 1
        iterations_used = iteration + 1

        recent_losses.append(final_cosine)
        recent_losses = recent_losses[-plateau_length:]
        if (
            len(recent_losses) == plateau_length
            and recent_losses[-1] >= recent_losses[0]
            and learning_rate_current > min_learning_rate
        ):
            learning_rate_current = max(
                learning_rate_current / plateau_drop,
                min_learning_rate,
            )
            recent_losses = []

        if final_cosine <= target_cosine:
            break

    stats = {
        "queries": total_queries,
        "iterations": iterations_used,
        "final_cosine": final_cosine,
    }
    return adv_raw, stats


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


def evaluate_fakebob_sv(
    victim_model,
    evaluator,
    dataloader,
    pairs,
    device,
    epsilon,
    max_iter,
    learning_rate,
    min_learning_rate,
    samples_per_draw,
    sigma,
    momentum,
    plateau_length,
    plateau_drop,
    target_cosine,
    query_batch_size,
    max_batches=None,
):
    clean_samples = []
    adv_samples = []
    cosine_sum = 0.0
    linf_sum = 0.0
    rms_sum = 0.0
    snr_sum = 0.0
    query_sum = 0
    iteration_sum = 0
    attacked_samples = 0

    total_batches = _safe_progress_total(dataloader, max_batches)
    pbar = tqdm(total=total_batches, desc="FAKEBOB SV eval", leave=True)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        clean_raw = batch.network_input
        adv_raw, attack_stats = fakebob_untargeted_attack(
            victim_model=victim_model,
            clean_raw=clean_raw,
            epsilon=epsilon,
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            samples_per_draw=samples_per_draw,
            sigma=sigma,
            momentum=momentum,
            plateau_length=plateau_length,
            plateau_drop=plateau_drop,
            target_cosine=target_cosine,
            query_batch_size=query_batch_size,
        )

        with torch.no_grad():
            clean_embedding = _speaker_embeddings(victim_model, clean_raw)
            adv_embedding = _speaker_embeddings(victim_model, adv_raw)

        delta = adv_raw - clean_raw
        linf = delta.abs().max().item()
        rms = delta.pow(2).mean().sqrt().item()
        signal_power = clean_raw.pow(2).mean().clamp_min(1e-12)
        noise_power = delta.pow(2).mean().clamp_min(1e-12)
        snr = (10.0 * torch.log10(signal_power / noise_power)).item()

        cosine_sum += F.cosine_similarity(clean_embedding, adv_embedding, dim=-1).sum().item()
        linf_sum += linf
        rms_sum += rms
        snr_sum += snr
        query_sum += attack_stats["queries"]
        iteration_sum += attack_stats["iterations"]
        attacked_samples += clean_raw.shape[0]

        for idx, sample_id in enumerate(batch.keys):
            clean_samples.append(
                EmbeddingSample(sample_id=sample_id, embedding=clean_embedding[idx].cpu())
            )
            adv_samples.append(
                EmbeddingSample(sample_id=sample_id, embedding=adv_embedding[idx].cpu())
            )

        pbar.update(1)
        pbar.set_postfix(
            cos=f"{attack_stats['final_cosine']:.3f}",
            queries=attack_stats["queries"],
            iters=attack_stats["iterations"],
        )
        del batch, clean_raw, adv_raw, clean_embedding, adv_embedding, delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    results = {
        "num_pairs": len(ground_truth),
        "attacked_samples": attacked_samples,
        "clean_eer": clean_eer,
        "adv_eer": adv_eer,
        "adv_threshold": adv_threshold,
        "eer_delta": adv_eer - clean_eer,
        "far_at_clean_threshold": far,
        "frr_at_clean_threshold": frr,
        "asr_sv_at_clean_threshold": asr_sv,
        "mean_embedding_cosine": cosine_sum / max(1, attacked_samples),
        "cosine_similarity_drop": 1.0 - cosine_sum / max(1, attacked_samples),
        "linf": linf_sum / max(1, attacked_samples),
        "rms": rms_sum / max(1, attacked_samples),
        "snr_db": snr_sum / max(1, attacked_samples),
        "mean_queries": query_sum / max(1, attacked_samples),
        "mean_iterations": iteration_sum / max(1, attacked_samples),
    }
    print(
        "[*] FAKEBOB-style SV eval: "
        f"pairs={results['num_pairs']}, samples={results['attacked_samples']}, "
        f"clean_eer={clean_eer:.4f}, adv_eer={adv_eer:.4f}, "
        f"eer_delta={results['eer_delta']:+.4f}, far={far:.4f}, frr={frr:.4f}, "
        f"asr_sv={asr_sv:.4f}, mean_cos={results['mean_embedding_cosine']:.4f}, "
        f"cos_drop={results['cosine_similarity_drop']:.4f}, "
        f"linf={results['linf']:.5f}, rms={results['rms']:.5f}, "
        f"snr_db={results['snr_db']:.2f}, mean_queries={results['mean_queries']:.1f}, "
        f"mean_iters={results['mean_iterations']:.1f}"
    )
    return results


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epsilon = float(_cfg_get(cfg, "epsilon", 0.05))
    max_iter = int(_cfg_get(cfg, "max_iter", 10))
    learning_rate = float(_cfg_get(cfg, "learning_rate", 0.005))
    min_learning_rate = float(_cfg_get(cfg, "min_learning_rate", 1e-5))
    samples_per_draw = int(_cfg_get(cfg, "samples_per_draw", 20))
    sigma = float(_cfg_get(cfg, "sigma", 0.001))
    momentum = float(_cfg_get(cfg, "momentum", 0.9))
    plateau_length = int(_cfg_get(cfg, "plateau_length", 5))
    plateau_drop = float(_cfg_get(cfg, "plateau_drop", 2.0))
    target_cosine = float(_cfg_get(cfg, "target_cosine", 0.5))
    query_batch_size = int(_cfg_get(cfg, "query_batch_size", 4))
    max_batches = _cfg_get(cfg, "max_batches", attack_cfg.sv_eval_max_batches)
    max_batches = None if max_batches is None else int(max_batches)

    print(
        "[*] FAKEBOB-style SV baseline: "
        f"epsilon={epsilon}, max_iter={max_iter}, learning_rate={learning_rate}, "
        f"samples_per_draw={samples_per_draw}, sigma={sigma}, momentum={momentum}, "
        f"target_cosine={target_cosine}, query_batch_size={query_batch_size}, "
        f"max_batches={max_batches}"
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")

    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.data.pipeline.test_pipeline = []
    sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
    sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
    sv_dm = construct_data_module(sv_cfg)
    evaluator = instantiate(sv_cfg.evaluator)
    victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False

    evaluate_fakebob_sv(
        victim_model=victim_model,
        evaluator=evaluator,
        dataloader=sv_dm.test_dataloader(),
        pairs=sv_dm.test_pairs,
        device=device,
        epsilon=epsilon,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        samples_per_draw=samples_per_draw,
        sigma=sigma,
        momentum=momentum,
        plateau_length=plateau_length,
        plateau_drop=plateau_drop,
        target_cosine=target_cosine,
        query_batch_size=query_batch_size,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    main_evaluation()
