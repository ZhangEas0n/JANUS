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
from btuap.sr import (
    extract_raw_audio_batch,
    extract_sensitive_binary_targets,
    load_sensitive_intent_head_checkpoint,
    make_speech_recognition_cfg,
)
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
    if "quality" in cfg and key in cfg.quality:
        return cfg.quality[key]
    return default


def _lengths_as_int_list(input_lengths, batch_size, num_samples):
    if input_lengths is None:
        return [int(num_samples)] * int(batch_size)
    if torch.is_tensor(input_lengths):
        return [int(length) for length in input_lengths.detach().cpu().tolist()]
    return [int(length) for length in input_lengths]


def _normalize_for_sr(x_raw, input_lengths):
    return normalize_batch_like_pipeline(x_raw, input_lengths=input_lengths)


def load_uap(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing BTUAP perturbation: {path}")
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
        f"[*] Loaded BTUAP: path={path}, shape={tuple(uap.shape)}, "
        f"linf={uap.abs().max().item():.5f}, rms={uap.pow(2).mean().sqrt().item():.5f}"
    )
    return uap


def apply_kenansville_fft(waveforms, factor):
    original_device = waveforms.device
    original_dtype = waveforms.dtype
    x_cpu = waveforms.detach().float().cpu()
    spectrum = torch.fft.fft(x_cpu, dim=-1)
    magnitude = spectrum.abs()
    threshold = magnitude.amax(dim=-1, keepdim=True) * float(factor)
    spectrum = torch.where(magnitude >= threshold, spectrum, torch.zeros_like(spectrum))
    return torch.fft.ifft(spectrum, dim=-1).real.to(
        device=original_device,
        dtype=original_dtype,
    ).clamp(-1.0, 1.0)


def apply_random_noise(waveforms, epsilon, generator=None):
    noise = torch.empty_like(waveforms).uniform_(
        -float(epsilon),
        float(epsilon),
        generator=generator,
    )
    return torch.clamp(waveforms + noise, -1.0, 1.0)


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
):
    if random_start:
        delta = torch.empty_like(x_clean_raw).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x_clean_raw)

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        adv_raw = torch.clamp(x_clean_raw + delta, -1.0, 1.0)
        adv_input = _normalize_for_sr(adv_raw, input_lengths)
        embedding, embedding_lengths = sr_model.compute_speech_embedding(
            adv_input,
            input_lengths,
        )
        logits = intent_head(embedding, embedding_lengths)
        p_sensitive = torch.softmax(logits, dim=-1)[:, sensitive_class_ids].sum(dim=-1)
        loss = -torch.log(p_sensitive.clamp_min(1e-8)).mean()
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = delta - alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(x_clean_raw + delta, -1.0, 1.0) - x_clean_raw

    return torch.clamp(x_clean_raw + delta.detach(), -1.0, 1.0)


def apply_pgd_chunked(
    sr_model,
    intent_head,
    waveform,
    sensitive_class_ids,
    epsilon,
    alpha,
    steps,
    random_start,
    max_attack_samples,
):
    valid_length = waveform.shape[-1]
    if max_attack_samples is None or valid_length <= max_attack_samples:
        return _pgd_sensitive_attack(
            sr_model,
            intent_head,
            waveform,
            [valid_length],
            sensitive_class_ids,
            epsilon,
            alpha,
            steps,
            random_start,
        )

    attacked = waveform.detach().clone()
    start = 0
    min_chunk_samples = 400
    while start < valid_length:
        end = min(start + max_attack_samples, valid_length)
        if 0 < valid_length - end < min_chunk_samples:
            end = valid_length
        chunk = waveform[..., start:end]
        attacked[..., start:end] = _pgd_sensitive_attack(
            sr_model,
            intent_head,
            chunk,
            [end - start],
            sensitive_class_ids,
            epsilon,
            alpha,
            steps,
            random_start,
        )
        start = end
    return attacked


def _load_metric_functions():
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError as exc:
        raise ImportError(
            "Audio quality dependencies are missing. Install them with: "
            "pip install pesq pystoi"
        ) from exc
    return pesq, stoi


def rescale_attack_to_snr(
    clean,
    attacked,
    target_snr_db,
    search_steps=40,
    tolerance_db=0.05,
):
    clean = clean.detach()
    delta = attacked.detach() - clean
    signal_power = clean.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    delta_power = delta.pow(2).mean()
    if delta_power <= 1e-20:
        raise ValueError("Attack produced a zero perturbation and cannot be SNR-scaled.")

    initial_scale = torch.sqrt(target_noise_power / delta_power).item()

    def candidate(scale):
        scaled = torch.clamp(clean + delta * scale, -1.0, 1.0)
        noise_power = (scaled - clean).pow(2).mean()
        return scaled, noise_power

    low = 0.0
    high = max(initial_scale, 1e-8)
    best, best_power = candidate(high)
    initial_snr_db = float(
        (10.0 * torch.log10(signal_power / best_power.clamp_min(1e-12))).item()
    )
    if abs(initial_snr_db - float(target_snr_db)) <= float(tolerance_db):
        return best

    for _ in range(20):
        if best_power >= target_noise_power:
            break
        high *= 2.0
        best, best_power = candidate(high)

    for _ in range(search_steps):
        middle = (low + high) / 2.0
        current, current_power = candidate(middle)
        if abs(current_power.item() - target_noise_power.item()) < abs(
            best_power.item() - target_noise_power.item()
        ):
            best, best_power = current, current_power
        if current_power < target_noise_power:
            low = middle
        else:
            high = middle

    achieved_snr_db = float(
        (10.0 * torch.log10(signal_power / best_power.clamp_min(1e-12))).item()
    )
    if abs(achieved_snr_db - float(target_snr_db)) > float(tolerance_db):
        raise ValueError(
            f"Could not match target SNR after clipping: "
            f"target={target_snr_db:g}dB, achieved={achieved_snr_db:.4f}dB"
        )
    return best


def calculate_quality_metrics(clean, attacked, sample_rate, pesq_fn, stoi_fn):
    clean = clean.detach().float().cpu().reshape(-1)
    attacked = attacked.detach().float().cpu().reshape(-1)

    clean_np = clean.numpy()
    attacked_np = attacked.numpy()
    stoi_value = float(stoi_fn(clean_np, attacked_np, sample_rate, extended=False))
    pesq_mode = "wb" if sample_rate == 16000 else "nb"
    pesq_value = float(pesq_fn(sample_rate, clean_np, attacked_np, pesq_mode))
    return {"stoi": stoi_value, "pesq": pesq_value}


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "std": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def summarize_results(results, failures):
    summary = {}
    for method, target_results in results.items():
        summary[method] = {}
        for target_snr, records in target_results.items():
            summary[method][target_snr] = {
                metric: _summary([record[metric] for record in records])
                for metric in ("stoi", "pesq")
            }
            summary[method][target_snr]["failures"] = failures[method][target_snr]
    return summary


def print_summary(summary):
    def format_metric(metric):
        if metric["mean"] is None:
            return "n/a"
        return f"{metric['mean']:.4f}/{metric['std']:.4f}"

    print("\n[*] Audio quality comparison at matched SNR")
    print(
        f"{'method':<14} {'target SNR':>12} {'samples':>8} "
        f"{'STOI mean/std':>22} {'PESQ mean/std':>22} {'failed':>8}"
    )
    for method, target_summaries in summary.items():
        for target_snr, metrics in target_summaries.items():
            count = metrics["stoi"]["count"]
            stoi = format_metric(metrics["stoi"])
            pesq = format_metric(metrics["pesq"])
            print(
                f"{method:<14} {float(target_snr):>10.1f}dB {count:>8} "
                f"{stoi:>22} {pesq:>22} {metrics['failures']:>8}"
            )


def save_results(output_prefix, results, summary, config):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config, "summary": summary, "per_sample": results},
            handle,
            indent=2,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "target_snr_db",
                "sample_index",
                "stoi",
                "pesq",
            ),
        )
        writer.writeheader()
        for method, target_results in results.items():
            for target_snr, records in target_results.items():
                for record in records:
                    writer.writerow(
                        {
                            "method": method,
                            "target_snr_db": target_snr,
                            **record,
                        }
                    )
    print(f"[*] Saved results: {json_path}")
    print(f"[*] Saved per-sample metrics: {csv_path}")


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    pesq_fn, stoi_fn = _load_metric_functions()
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_rate = int(_cfg_get(cfg, "sample_rate", attack_cfg.sample_rate))
    max_batches = _cfg_get(cfg, "max_batches", 100)
    max_batches = None if max_batches is None else int(max_batches)
    target_snrs = [
        float(value) for value in _cfg_get(cfg, "target_snrs", [10.0, 20.0, 30.0])
    ]
    if not target_snrs:
        raise ValueError("quality.target_snrs must contain at least one SNR value.")
    snr_tolerance_db = float(_cfg_get(cfg, "snr_tolerance_db", 0.05))
    kenansville_factor = float(_cfg_get(cfg, "kenansville_factor", 0.1))
    pgd_epsilon = float(_cfg_get(cfg, "pgd_epsilon", attack_cfg.epsilon))
    pgd_alpha = float(_cfg_get(cfg, "pgd_alpha", 0.005))
    pgd_steps = int(_cfg_get(cfg, "pgd_steps", 10))
    pgd_random_start = bool(_cfg_get(cfg, "pgd_random_start", True))
    pgd_max_attack_samples = _cfg_get(cfg, "pgd_max_attack_samples", 48000)
    pgd_max_attack_samples = (
        None if pgd_max_attack_samples is None else int(pgd_max_attack_samples)
    )
    random_epsilon = float(_cfg_get(cfg, "random_epsilon", attack_cfg.epsilon))
    random_seed = int(_cfg_get(cfg, "random_seed", 2026))
    uap_path = _cfg_get(
        cfg,
        "uap_path",
        attack_cfg.evaluation_uap_path or attack_cfg.output_path,
    )
    output_prefix = _cfg_get(
        cfg,
        "output_prefix",
        str(Path(project_root) / "results" / "main" / "audio_quality_comparison"),
    )

    if sample_rate not in (8000, 16000):
        raise ValueError("PESQ supports sample_rate=8000 or 16000.")
    if pgd_max_attack_samples is not None and pgd_max_attack_samples < 400:
        raise ValueError("quality.pgd_max_attack_samples must be at least 400.")

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2 files: {local_wav2vec2}")

    print("[*] Initializing FSC test data, SR surrogate, and sensitive intent head...")
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
    sensitive_labels = [
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not sensitive_labels:
        raise ValueError("No configured sensitive intent labels exist in FSC.")

    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    sr_model.eval()
    for parameter in sr_model.parameters():
        parameter.requires_grad = False
    intent_head = load_sensitive_intent_head_checkpoint(
        checkpoint_path=attack_cfg.intent_head_path,
        sensitive_intent_labels=sensitive_labels,
        decision_threshold=attack_cfg.intent_decision_threshold,
        device=device,
    )
    uap = load_uap(uap_path, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)

    methods = ("btuap", "kenansville", "pgd", "random_noise")
    snr_keys = [str(value) for value in target_snrs]
    results = {
        method: {target_snr: [] for target_snr in snr_keys}
        for method in methods
    }
    failures = {
        method: {target_snr: 0 for target_snr in snr_keys}
        for method in methods
    }
    sample_index = 0
    dataloader = sr_dm.test_dataloader()
    total = max_batches
    if total is None:
        try:
            total = len(dataloader)
        except (TypeError, ValueError):
            total = None
    pbar = tqdm(total=total, desc="Audio quality comparison", leave=True)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch.to(device)
        targets = extract_sensitive_binary_targets(
            batch=batch,
            sensitive_intent_labels=sensitive_labels,
            sensitive_keywords=attack_cfg.sensitive_keywords,
            device=device,
        )
        raw_audio = extract_raw_audio_batch(batch, device)
        lengths = _lengths_as_int_list(
            batch.input_lengths,
            raw_audio.shape[0],
            raw_audio.shape[-1],
        )

        for idx, target in enumerate(targets):
            if int(target.item()) != 0:
                continue
            valid_length = lengths[idx]
            clean = raw_audio[idx : idx + 1, :valid_length].detach()
            attacked_by_method = {
                "btuap": apply_uap(clean, uap),
                "kenansville": apply_kenansville_fft(clean, kenansville_factor),
                "random_noise": apply_random_noise(clean, random_epsilon, generator),
            }
            attacked_by_method["pgd"] = apply_pgd_chunked(
                sr_model=sr_model,
                intent_head=intent_head,
                waveform=clean,
                sensitive_class_ids=[1],
                epsilon=pgd_epsilon,
                alpha=pgd_alpha,
                steps=pgd_steps,
                random_start=pgd_random_start,
                max_attack_samples=pgd_max_attack_samples,
            )

            for method, attacked in attacked_by_method.items():
                for target_snr, target_snr_key in zip(target_snrs, snr_keys):
                    try:
                        matched_attack = rescale_attack_to_snr(
                            clean,
                            attacked,
                            target_snr,
                            tolerance_db=snr_tolerance_db,
                        )
                        metrics = calculate_quality_metrics(
                            clean,
                            matched_attack,
                            sample_rate,
                            pesq_fn,
                            stoi_fn,
                        )
                        results[method][target_snr_key].append(
                            {
                                "sample_index": sample_index,
                                **metrics,
                            }
                        )
                    except Exception as exc:
                        failures[method][target_snr_key] += 1
                        tqdm.write(
                            f"[!] Skipping {method} at {target_snr:g}dB, "
                            f"sample {sample_index}: {type(exc).__name__}: {exc}"
                        )
            sample_index += 1

        pbar.update(1)
        pbar.set_postfix(benign_samples=sample_index)
        del batch, raw_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    summary = summarize_results(results, failures)
    print_summary(summary)
    config = {
        "sample_rate": sample_rate,
        "max_batches": max_batches,
        "target_snrs": target_snrs,
        "snr_tolerance_db": snr_tolerance_db,
        "uap_path": str(uap_path),
        "kenansville_factor": kenansville_factor,
        "pgd_epsilon": pgd_epsilon,
        "pgd_alpha": pgd_alpha,
        "pgd_steps": pgd_steps,
        "pgd_random_start": pgd_random_start,
        "pgd_max_attack_samples": pgd_max_attack_samples,
        "random_epsilon": random_epsilon,
        "random_seed": random_seed,
    }
    save_results(output_prefix, results, summary, config)


if __name__ == "__main__":
    main_evaluation()
