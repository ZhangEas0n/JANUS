#!/usr/bin/env python3
"""Optimize a 3-second bird-like JANUS perturbation on DistilHuBERT heads."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btuap.distilhubert_fsc import load_backbone, load_heads_checkpoint, load_waveform, scaled_cosine
from btuap.mel_mimic import MelMimicLoss, load_environment_sound, rms, waveform_similarity_loss


def load_tensor(path, device):
    payload = torch.load(str(path), map_location=device)
    if torch.is_tensor(payload): return payload.reshape(1, -1).float()
    for key in ("uap", "delta", "perturbation"):
        if key in payload and torch.is_tensor(payload[key]): return payload[key].reshape(1, -1).float()
    raise ValueError("No perturbation tensor in {}".format(path))


def fixed_length(path, length, random_crop=True):
    wave = load_waveform(path)
    if wave.numel() > length:
        start = random.randint(0, wave.numel() - length) if random_crop else 0
        wave = wave[start:start + length]
    elif wave.numel() < length:
        wave = F.pad(wave, (0, length - wave.numel()))
    return wave


def encode_grad(backbone, waves, lengths):
    hidden = backbone(waves).last_hidden_state
    frame_lengths = backbone._get_feat_extract_output_lengths(lengths)
    index = torch.arange(hidden.shape[1], device=waves.device).unsqueeze(0)
    mask = (index < frame_lengths.unsqueeze(1)).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--initial-uap", type=Path, required=True)
    parser.add_argument("--environment-sound", type=Path, required=True)
    parser.add_argument("--enrollment-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--sv-weight", type=float, default=0.3)
    parser.add_argument("--sr-weight", type=float, default=0.7)
    parser.add_argument("--mimic-weight", type=float, default=0.005)
    parser.add_argument("--waveform-weight", type=float, default=0.02)
    parser.add_argument("--power-weight", type=float, default=20.0)
    args = parser.parse_args()
    root = args.project_root.resolve(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(root / "model" / "distilhubert", device)
    heads, _ = load_heads_checkpoint(root / "models/checkpoints/distilhubert_fsc/distilhubert_fsc_heads.pt", device)
    thresholds = json.loads((root / "models/checkpoints/distilhubert_fsc/thresholds.json").read_text())
    with (root / "fluent_speech_commands_dataset/data/train_data.csv").open(encoding="utf-8-sig", newline="") as f:
        train_rows = list(csv.DictReader(f))
    audio_root = root / "fluent_speech_commands_dataset"
    enroll_rows = list(csv.DictReader(args.enrollment_list.open(encoding="utf-8-sig", newline="")))
    grouped = {}
    for row in enroll_rows:
        p = Path(row.get("enrollment_path") or row.get("audio_path") or row.get("path"))
        label = row.get("speaker_id") or row.get("label") or p.stem
        grouped.setdefault(label, []).append(p)
    prototypes = []
    with torch.no_grad():
        for paths in grouped.values():
            waves = [fixed_length(p, 48000, False) for p in paths]
            lengths = torch.tensor([w.numel() for w in waves], device=device)
            pooled = encode_grad(backbone, pad_sequence(waves, batch_first=True).to(device), lengths)
            prototypes.append(F.normalize(heads.speaker_embedding(pooled).mean(0), dim=0))
    gallery = torch.stack(prototypes)
    init = load_tensor(args.initial_uap, device)
    if init.shape[-1] < 48000: init = init.repeat(1, (48000 + init.shape[-1] - 1) // init.shape[-1])
    init = init[:, :48000]
    target_rms = float(rms(init))
    delta = torch.nn.Parameter(init.clone())
    env = load_environment_sound(args.environment_sound, 48000, 16000, device, torch.float32)
    env = env / rms(env) * target_rms
    mel_loss_fn = MelMimicLoss().to(device)
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    log = []
    for step in range(1, args.steps + 1):
        batch_rows = random.sample(train_rows, args.batch_size)
        clean = torch.stack([fixed_length(audio_root / r["path"], 48000) for r in batch_rows]).to(device)
        adv = torch.clamp(clean + delta, -1.0, 1.0)
        lengths = torch.full((args.batch_size,), 48000, dtype=torch.long, device=device)
        pooled = encode_grad(backbone, adv, lengths)
        sv_emb = heads.speaker_embedding(pooled)
        best_scores = scaled_cosine(sv_emb, gallery).max(1).values
        sr_logits = heads.sr_classifier(pooled)
        sv_loss = F.relu(float(thresholds["sv_threshold"]) + 0.01 - best_scores).mean()
        sr_loss = F.cross_entropy(sr_logits, torch.ones(args.batch_size, dtype=torch.long, device=device))
        mimic_loss = mel_loss_fn(delta, env)
        wave_loss = waveform_similarity_loss(delta, init)
        power_loss = (rms(delta) - target_rms).abs()
        loss = (args.sv_weight * sv_loss + args.sr_weight * sr_loss +
                args.mimic_weight * mimic_loss + args.waveform_weight * wave_loss +
                args.power_weight * power_loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            delta.clamp_(-args.epsilon, args.epsilon)
            delta.mul_(target_rms / rms(delta).clamp_min(1e-8))
        if step == 1 or step % 25 == 0 or step == args.steps:
            item = {"step": step, "loss": float(loss), "sv_loss": float(sv_loss), "sr_loss": float(sr_loss),
                    "mimic_loss": float(mimic_loss), "waveform_loss": float(wave_loss),
                    "mean_sv_score": float(best_scores.mean()),
                    "p_sensitive": float(torch.softmax(sr_logits, 1)[:, 1].mean()),
                    "sampled_far": float((best_scores >= float(thresholds["sv_threshold"])).float().mean()),
                    "rms": float(rms(delta)), "linf": float(delta.abs().max())}
            log.append(item); print(json.dumps(item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = delta.detach().cpu()
    torch.save(saved, args.output)
    torchaudio.save(str(args.output.with_suffix(".wav")), saved, 16000)
    report = {"janus_type": "JANUS-DH", "source_model": "DistilHuBERT",
              "target_model": "DistilHuBERT", "reoptimized_on_target": True,
              "initial_uap": str(args.initial_uap.resolve()), "environment_sound": str(args.environment_sound.resolve()),
              "steps": args.steps, "target_rms": target_rms, "epsilon": args.epsilon,
              "weights": {"sv": args.sv_weight, "sr": args.sr_weight,
                          "mimic": args.mimic_weight, "waveform": args.waveform_weight,
                          "power": args.power_weight},
              "history": log}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", args.output, args.output.with_suffix(".wav"))


if __name__ == "__main__":
    main()
