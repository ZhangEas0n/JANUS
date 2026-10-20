#!/usr/bin/env python3
"""Evaluate aligned physical OTA audio with DistilHuBERT FSC heads."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import torchaudio

from btuap.distilhubert_fsc import (
    SAMPLE_RATE, extract_path_embeddings, load_backbone, load_heads_checkpoint,
    scaled_cosine,
)


LOGGER = logging.getLogger("distilhubert_physical_eval")


def truth(value):
    return str(value).lower() in {"1", "true", "yes"}


def setup_logging(path):
    LOGGER.setLevel(logging.INFO); LOGGER.handlers[:] = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(str(path), encoding="utf-8")):
        handler.setFormatter(formatter); LOGGER.addHandler(handler)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_audio(value, project_root, metadata_parent):
    path = Path(value).expanduser()
    if path.is_absolute(): return path.resolve()
    for base in (metadata_parent, project_root):
        candidate = (base / path).resolve()
        if candidate.exists(): return candidate
    return (project_root / path).resolve()


def audio_properties(path):
    waveform, sample_rate = torchaudio.load(str(path))
    mono = waveform.float().mean(0)
    return {"duration": mono.numel() / float(sample_rate), "sample_rate_original": int(sample_rate),
            "sample_rate_model": SAMPLE_RATE, "rms": float(mono.pow(2).mean().sqrt()),
            "peak_abs": float(mono.abs().max()), "clipping_rate": float((mono.abs() >= 0.999).float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--enrollment-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--heads-checkpoint", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    backbone_path = (args.backbone or project_root / "model" / "distilhubert").resolve()
    checkpoint_path = (args.heads_checkpoint or project_root / "models" / "checkpoints" / "distilhubert_fsc" / "distilhubert_fsc_heads.pt").resolve()
    thresholds_path = (args.thresholds or checkpoint_path.parent / "thresholds.json").resolve()
    for path in (args.metadata, args.enrollment_list, backbone_path, checkpoint_path, thresholds_path):
        if not path.exists(): raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True); setup_logging(args.log)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    sv_threshold = float(thresholds["sv_threshold"]); sr_threshold = float(thresholds["sr_threshold"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Backbone=%s", backbone_path); LOGGER.info("Heads=%s", checkpoint_path)
    LOGGER.info("Frozen thresholds: SV=%.9f SR=%.4f", sv_threshold, sr_threshold)
    metadata = read_csv(args.metadata)
    enroll_rows = read_csv(args.enrollment_list)
    enrollments = []
    for row in enroll_rows:
        value = row.get("enrollment_path") or row.get("audio_path") or row.get("path")
        label = row.get("speaker_id") or row.get("label") or Path(value).stem
        enrollments.append((str(label), resolve_audio(value, project_root, args.enrollment_list.parent)))
    trials = []
    all_paths = []
    for row in metadata:
        ota = resolve_audio(row["audio_path"], project_root, args.metadata.parent)
        clean = resolve_audio(row["clean_audio_path"], project_root, args.metadata.parent)
        trials.append((row, ota, clean)); all_paths.extend((ota, clean))
    all_paths.extend(path for _, path in enrollments)
    unique_paths = list(dict.fromkeys(Path(p).resolve() for p in all_paths))
    LOGGER.info("Extracting DistilHuBERT features for %d unique audio files", len(unique_paths))
    backbone = load_backbone(backbone_path, device)
    pooled = extract_path_embeddings(backbone, unique_paths, device, args.batch_size)
    del backbone
    heads, payload = load_heads_checkpoint(checkpoint_path, device)
    with torch.inference_mode():
        sv_embeddings = heads.speaker_embedding(pooled.to(device)).cpu()
        sr_probs = torch.softmax(heads.sr_classifier(pooled.to(device)), 1)[:, 1].cpu()
    index = {str(path): i for i, path in enumerate(unique_paths)}
    grouped = {}
    for label, path in enrollments: grouped.setdefault(label, []).append(path)
    gallery_labels, gallery_paths, prototypes = [], [], []
    for label, paths in grouped.items():
        emb = torch.stack([sv_embeddings[index[str(p.resolve())]] for p in paths])
        prototypes.append(F.normalize(emb.mean(0), dim=0)); gallery_labels.append(label); gallery_paths.append(paths[0])
    gallery = torch.stack(prototypes)

    def infer(path):
        i = index[str(path.resolve())]
        scores = scaled_cosine(sv_embeddings[i:i+1], gallery)[0]
        best = int(scores.argmax()); p_sensitive = float(sr_probs[i])
        return {"sv_score": float(scores[best]), "best_enrollment": str(gallery_paths[best]),
                "best_enrollment_speaker": gallery_labels[best], "sv_accepted": float(scores[best]) >= sv_threshold,
                "p_sensitive": p_sensitive, "is_sensitive": p_sensitive >= sr_threshold,
                "sr_pred": "sensitive" if p_sensitive >= sr_threshold else "benign"}

    output = []
    for row, ota, clean in trials:
        o = infer(ota); c = infer(clean); props = audio_properties(ota)
        sv_eligible = not c["sv_accepted"]; sr_eligible = not c["is_sensitive"]
        joint_eligible = sv_eligible and sr_eligible
        result = dict(row)
        result.update(props)
        result.update({"sv_score": o["sv_score"], "best_sv_score": o["sv_score"],
                       "best_enrollment": o["best_enrollment"], "best_enrollment_speaker": o["best_enrollment_speaker"],
                       "num_enrollments": len(gallery_labels), "sv_threshold": sv_threshold,
                       "sv_accepted": o["sv_accepted"], "sr_pred": o["sr_pred"],
                       "p_sensitive": o["p_sensitive"], "sr_threshold": sr_threshold,
                       "is_sensitive": o["is_sensitive"], "joint_success": o["sv_accepted"] and o["is_sensitive"],
                       "union_success": o["sv_accepted"] or o["is_sensitive"],
                       "sv_score_clean": c["sv_score"], "sv_score_ota": o["sv_score"],
                       "sv_eligible": sv_eligible, "sv_success": sv_eligible and o["sv_accepted"],
                       "sr_pred_clean": c["sr_pred"], "sr_pred_ota": o["sr_pred"],
                       "p_sensitive_clean": c["p_sensitive"], "p_sensitive_ota": o["p_sensitive"],
                       "sr_eligible": sr_eligible, "sr_success": sr_eligible and o["is_sensitive"],
                       "joint_eligible": joint_eligible,
                       "joint_attack_success": joint_eligible and o["sv_accepted"] and o["is_sensitive"],
                       "union_eligible": joint_eligible,
                       "union_attack_success": joint_eligible and (o["sv_accepted"] or o["is_sensitive"])})
        output.append(result)
    fields = list(output[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)

    def mean_bool(key, rows): return sum(bool(r[key]) for r in rows) / len(rows)
    def conditional(eligible, success):
        rows = [r for r in output if r[eligible]]
        return (sum(bool(r[success]) for r in rows) / len(rows) if rows else None, len(rows))
    asr_sv, n_sv = conditional("sv_eligible", "sv_success"); asr_sr, n_sr = conditional("sr_eligible", "sr_success")
    asr_joint, n_joint = conditional("joint_eligible", "joint_attack_success")
    asr_union, n_union = conditional("union_eligible", "union_attack_success")
    summary = {"num_trials": len(output), "backbone": str(backbone_path), "heads_checkpoint": str(checkpoint_path),
               "sv_threshold": sv_threshold, "sr_threshold": sr_threshold,
               "far": mean_bool("sv_accepted", output), "sensitive_rate": mean_bool("is_sensitive", output),
               "joint_success_rate": mean_bool("joint_success", output), "union_success_rate": mean_bool("union_success", output),
               "asr_sv": asr_sv, "asr_sr": asr_sr, "asr_joint": asr_joint, "asr_union": asr_union,
               "eligible_sv_trials": n_sv, "eligible_sr_trials": n_sr,
               "eligible_joint_trials": n_joint, "eligible_union_trials": n_union}
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
