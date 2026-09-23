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
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.common import apply_uap
from btuap.config import BTUAPAttackConfig
from btuap.sr import (
    extract_raw_audio_batch,
    extract_sensitive_binary_targets,
    make_speech_recognition_cfg,
)
from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)
from src.main import construct_data_module

load_dotenv()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)


def _cfg_get(cfg: DictConfig, key: str, default):
    if "btuap_quality" in cfg and key in cfg.btuap_quality:
        return cfg.btuap_quality[key]
    return default


def _lengths_as_int_list(input_lengths, batch_size, num_samples):
    if input_lengths is None:
        return [int(num_samples)] * int(batch_size)
    if torch.is_tensor(input_lengths):
        return [int(length) for length in input_lengths.detach().cpu().tolist()]
    return [int(length) for length in input_lengths]


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
        "[*] Loaded BTUAP: "
        f"path={path}, shape={tuple(uap.shape)}, "
        f"linf={uap.abs().max().item():.5f}, "
        f"rms={uap.pow(2).mean().sqrt().item():.5f}"
    )
    return uap


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


def calculate_snr_db(clean, attacked):
    clean = clean.detach().float().reshape(-1)
    attacked = attacked.detach().float().reshape(-1)
    delta = attacked - clean
    signal_power = clean.pow(2).mean().clamp_min(1e-12)
    noise_power = delta.pow(2).mean().clamp_min(1e-12)
    return float((10.0 * torch.log10(signal_power / noise_power)).item())


def calculate_quality_metrics(clean, attacked, sample_rate, pesq_fn, stoi_fn):
    snr_db = calculate_snr_db(clean, attacked)
    clean = clean.detach().float().cpu().reshape(-1)
    attacked = attacked.detach().float().cpu().reshape(-1)
    clean_np = clean.numpy()
    attacked_np = attacked.numpy()
    pesq_mode = "wb" if sample_rate == 16000 else "nb"
    return {
        "snr_db": snr_db,
        "stoi": float(stoi_fn(clean_np, attacked_np, sample_rate, extended=False)),
        "pesq": float(pesq_fn(sample_rate, clean_np, attacked_np, pesq_mode)),
    }


def rescale_attack_to_snr(clean, attacked, target_snr_db, tolerance_db=0.05, search_steps=40):
    clean = clean.detach()
    delta = attacked.detach() - clean
    signal_power = clean.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
    delta_power = delta.pow(2).mean()
    if delta_power <= 1e-20:
        raise ValueError("BTUAP produced a zero perturbation and cannot be SNR-scaled.")

    initial_scale = torch.sqrt(target_noise_power / delta_power).item()

    def candidate(scale):
        scaled = torch.clamp(clean + delta * scale, -1.0, 1.0)
        noise_power = (scaled - clean).pow(2).mean()
        return scaled, noise_power

    low = 0.0
    high = max(initial_scale, 1e-8)
    best, best_power = candidate(high)
    initial_snr = float(
        (10.0 * torch.log10(signal_power / best_power.clamp_min(1e-12))).item()
    )
    if abs(initial_snr - float(target_snr_db)) <= float(tolerance_db):
        return best

    for _ in range(20):
        if best_power >= target_noise_power:
            break
        high *= 2.0
        best, best_power = candidate(high)

    for _ in range(search_steps):
        mid = (low + high) / 2.0
        current, current_power = candidate(mid)
        if abs(current_power.item() - target_noise_power.item()) < abs(
            best_power.item() - target_noise_power.item()
        ):
            best, best_power = current, current_power
        if current_power < target_noise_power:
            low = mid
        else:
            high = mid

    achieved = float(
        (10.0 * torch.log10(signal_power / best_power.clamp_min(1e-12))).item()
    )
    if abs(achieved - float(target_snr_db)) > float(tolerance_db):
        raise ValueError(
            f"Could not match target SNR after clipping: "
            f"target={target_snr_db:g}dB, achieved={achieved:.4f}dB"
        )
    return best


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "std": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def summarize(records_by_mode, failures):
    summary = {}
    for mode, records in records_by_mode.items():
        summary[mode] = {
            "snr_db": _summary([record["snr_db"] for record in records]),
            "stoi": _summary([record["stoi"] for record in records]),
            "pesq": _summary([record["pesq"] for record in records]),
            "failures": failures[mode],
        }
    return summary


def _format_summary(metric):
    if metric["mean"] is None:
        return "n/a"
    return f"{metric['mean']:.4f}/{metric['std']:.4f}"


def print_summary(summary):
    print("\n[*] BTUAP audio quality")
    print(
        f"{'mode':<14} {'samples':>8} {'SNR mean/std':>22} "
        f"{'STOI mean/std':>22} {'PESQ mean/std':>22} {'failed':>8}"
    )
    for mode, metrics in summary.items():
        print(
            f"{mode:<14} {metrics['snr_db']['count']:>8} "
            f"{_format_summary(metrics['snr_db']):>22} "
            f"{_format_summary(metrics['stoi']):>22} "
            f"{_format_summary(metrics['pesq']):>22} "
            f"{metrics['failures']:>8}"
        )


def save_results(output_prefix, summary, records_by_mode, config):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config, "summary": summary, "per_sample": records_by_mode},
            handle,
            indent=2,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("mode", "sample_index", "snr_db", "stoi", "pesq"),
        )
        writer.writeheader()
        for mode, records in records_by_mode.items():
            for record in records:
                writer.writerow({"mode": mode, **record})

    print(f"[*] Saved summary: {json_path}")
    print(f"[*] Saved per-sample metrics: {csv_path}")


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    pesq_fn, stoi_fn = _load_metric_functions()
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_rate = int(_cfg_get(cfg, "sample_rate", attack_cfg.sample_rate))
    if sample_rate not in (8000, 16000):
        raise ValueError("PESQ supports sample_rate=8000 or 16000.")
    max_batches = _cfg_get(cfg, "max_batches", 100)
    max_batches = None if max_batches is None else int(max_batches)
    only_benign = bool(_cfg_get(cfg, "only_benign", True))
    snr_tolerance_db = float(_cfg_get(cfg, "snr_tolerance_db", 0.05))
    target_snrs = _cfg_get(cfg, "target_snrs", None)
    target_snrs = [] if target_snrs is None else [float(value) for value in target_snrs]
    uap_path = _cfg_get(
        cfg,
        "uap_path",
        attack_cfg.evaluation_uap_path or attack_cfg.output_path,
    )
    output_prefix = _cfg_get(
        cfg,
        "output_prefix",
        str(Path(project_root) / "results" / "main" / "btuap_quality"),
    )

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_wav2vec2.exists():
        raise FileNotFoundError(f"Missing local wav2vec2/tokenizer files: {local_wav2vec2}")

    print("[*] Initializing FSC test data for BTUAP quality evaluation...")
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
    if only_benign and not sensitive_labels:
        raise ValueError("No configured sensitive intent labels exist in FSC.")

    uap = load_uap(uap_path, device)
    modes = ["original"] + [f"snr_{snr:g}dB" for snr in target_snrs]
    records_by_mode = {mode: [] for mode in modes}
    failures = {mode: 0 for mode in modes}

    dataloader = sr_dm.test_dataloader()
    total = max_batches
    if total is None:
        try:
            total = len(dataloader)
        except (TypeError, ValueError):
            total = None
    pbar = tqdm(total=total, desc="BTUAP quality", leave=True)
    sample_index = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch.to(device)
        raw_audio = extract_raw_audio_batch(batch, device)
        lengths = _lengths_as_int_list(
            batch.input_lengths,
            raw_audio.shape[0],
            raw_audio.shape[-1],
        )

        if only_benign:
            targets = extract_sensitive_binary_targets(
                batch=batch,
                sensitive_intent_labels=sensitive_labels,
                sensitive_keywords=attack_cfg.sensitive_keywords,
                device=device,
            )
            selected = targets == 0
        else:
            selected = torch.ones(raw_audio.shape[0], dtype=torch.bool, device=device)

        for idx, should_use in enumerate(selected):
            if not bool(should_use.item()):
                continue
            valid_length = lengths[idx]
            clean = raw_audio[idx : idx + 1, :valid_length].detach()
            attacked = apply_uap(clean, uap)

            try:
                metrics = calculate_quality_metrics(
                    clean,
                    attacked,
                    sample_rate,
                    pesq_fn,
                    stoi_fn,
                )
                records_by_mode["original"].append(
                    {"sample_index": sample_index, **metrics}
                )
            except Exception as exc:
                failures["original"] += 1
                tqdm.write(
                    f"[!] Skipping original sample {sample_index}: "
                    f"{type(exc).__name__}: {exc}"
                )

            for target_snr in target_snrs:
                mode = f"snr_{target_snr:g}dB"
                try:
                    matched = rescale_attack_to_snr(
                        clean,
                        attacked,
                        target_snr,
                        tolerance_db=snr_tolerance_db,
                    )
                    metrics = calculate_quality_metrics(
                        clean,
                        matched,
                        sample_rate,
                        pesq_fn,
                        stoi_fn,
                    )
                    records_by_mode[mode].append(
                        {"sample_index": sample_index, **metrics}
                    )
                except Exception as exc:
                    failures[mode] += 1
                    tqdm.write(
                        f"[!] Skipping {mode} sample {sample_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            sample_index += 1

        pbar.update(1)
        pbar.set_postfix(samples=sample_index)
        del batch, raw_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    summary = summarize(records_by_mode, failures)
    print_summary(summary)
    save_results(
        output_prefix,
        summary,
        records_by_mode,
        {
            "sample_rate": sample_rate,
            "max_batches": max_batches,
            "only_benign": only_benign,
            "target_snrs": target_snrs,
            "snr_tolerance_db": snr_tolerance_db,
            "uap_path": str(uap_path),
        },
    )


if __name__ == "__main__":
    main_evaluation()
