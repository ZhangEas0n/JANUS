#!/usr/bin/env python3
"""Train and validate lightweight FSC heads on frozen DistilHuBERT features."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from btuap.config import BTUAPAttackConfig
from btuap.distilhubert_fsc import (
    DistilHuBERTFSCHeads, extract_path_embeddings, load_backbone, scaled_cosine,
)


def read_split(project_root, name, sensitive_labels):
    csv_path = project_root / "fluent_speech_commands_dataset" / "data" / name
    audio_root = project_root / "fluent_speech_commands_dataset"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["audio_path"] = str((audio_root / row["path"]).resolve())
        row["intent"] = "|".join((row["action"], row["object"], row["location"]))
        row["sensitive"] = int(row["intent"] in sensitive_labels)
    return rows


def extract_split(name, rows, backbone, device, cache_dir, batch_size):
    cache = cache_dir / (name + "_pooled.pt")
    if cache.exists():
        payload = torch.load(str(cache), map_location="cpu")
        if payload.get("paths") == [r["audio_path"] for r in rows]:
            print("Reusing", cache)
            return payload["features"]
    print("Extracting", name, len(rows))
    features = extract_path_embeddings(
        backbone, [Path(r["audio_path"]) for r in rows], device, batch_size
    )
    torch.save({"paths": [r["audio_path"] for r in rows], "features": features}, cache)
    return features


def verification_scores(features, rows, heads, enrollment_k=3):
    speakers = sorted({r["speakerId"] for r in rows})
    enroll_indices, trial_indices = {}, []
    for speaker in speakers:
        indices = [i for i, r in enumerate(rows) if r["speakerId"] == speaker]
        indices.sort(key=lambda i: rows[i]["audio_path"])
        enroll_indices[speaker] = indices[:enrollment_k]
        trial_indices.extend(indices[enrollment_k:])
    with torch.no_grad():
        embeddings = heads.speaker_embedding(features.to(next(heads.parameters()).device)).cpu()
    prototypes = []
    for speaker in speakers:
        prototype = F.normalize(embeddings[enroll_indices[speaker]].mean(0), dim=0)
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)
    score_matrix = scaled_cosine(embeddings[trial_indices], prototypes)
    labels = torch.tensor([speakers.index(rows[i]["speakerId"]) for i in trial_indices])
    genuine = score_matrix[torch.arange(len(trial_indices)), labels]
    mask = torch.ones_like(score_matrix, dtype=torch.bool)
    mask[torch.arange(len(trial_indices)), labels] = False
    impostor = score_matrix[mask]
    return genuine, impostor


def select_eer_threshold(genuine, impostor):
    candidates = torch.unique(torch.cat((genuine, impostor))).sort().values
    best = None
    for threshold in candidates:
        far = float((impostor >= threshold).float().mean())
        frr = float((genuine < threshold).float().mean())
        item = (abs(far - frr), (far + frr) / 2.0, float(threshold), far, frr)
        if best is None or item < best:
            best = item
    return {"threshold": best[2], "eer": best[1], "far": best[3], "frr": best[4]}


def binary_metrics(probabilities, targets, threshold):
    pred = probabilities >= threshold
    target = targets.bool()
    tp = int((pred & target).sum()); tn = int((~pred & ~target).sum())
    fp = int((pred & ~target).sum()); fn = int((~pred & target).sum())
    recall = tp / max(1, tp + fn); specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"balanced_accuracy": (recall + specificity) / 2.0,
            "sensitive_precision": precision, "sensitive_recall": recall,
            "sensitive_f1": f1, "sensitive_rate": float(pred.float().mean()),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def select_sr_threshold(probabilities, targets):
    best = None
    for i in range(5, 96):
        threshold = i / 100.0
        metrics = binary_metrics(probabilities, targets, threshold)
        item = (metrics["sensitive_f1"], metrics["balanced_accuracy"], -abs(threshold - 0.5))
        if best is None or item > best[0]:
            best = (item, threshold, metrics)
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=48)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    backbone_path = (args.backbone or project_root / "model" / "distilhubert").resolve()
    output_dir = (args.output_dir or project_root / "models" / "checkpoints" / "distilhubert_fsc").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"; cache_dir.mkdir(exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sensitive_labels = set(BTUAPAttackConfig().sensitive_intent_labels)
    splits = {name: read_split(project_root, file_name, sensitive_labels) for name, file_name in
              (("train", "train_data.csv"), ("valid", "valid_data.csv"), ("test", "test_data.csv"))}
    backbone = load_backbone(backbone_path, device)
    features = {name: extract_split(name, rows, backbone, device, cache_dir, args.feature_batch_size)
                for name, rows in splits.items()}
    del backbone
    if device.type == "cuda": torch.cuda.empty_cache()
    train_rows = splits["train"]
    speaker_names = sorted({r["speakerId"] for r in train_rows})
    speaker_to_id = {s: i for i, s in enumerate(speaker_names)}
    speaker_targets = torch.tensor([speaker_to_id[r["speakerId"]] for r in train_rows])
    sr_targets = torch.tensor([r["sensitive"] for r in train_rows])
    head_config = dict(input_dim=features["train"].shape[1], sv_dim=256,
                       sr_hidden_dim=256, num_train_speakers=len(speaker_names), dropout=0.1)
    heads = DistilHuBERTFSCHeads(**head_config).to(device)
    counts = torch.bincount(sr_targets, minlength=2).float()
    sr_weights = (counts.sum() / (2 * counts)).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        heads.train(); order = torch.randperm(len(train_rows), generator=generator)
        total_loss = total_sv = total_sr = correct_sv = correct_sr = seen = 0
        for start in range(0, len(order), args.head_batch_size):
            index = order[start:start + args.head_batch_size]
            x = features["train"][index].to(device)
            y_sv = speaker_targets[index].to(device); y_sr = sr_targets[index].to(device)
            _, logits_sv, logits_sr = heads(x)
            loss_sv = F.cross_entropy(logits_sv, y_sv)
            loss_sr = F.cross_entropy(logits_sr, y_sr, weight=sr_weights)
            loss = loss_sv + loss_sr
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n = len(index); seen += n
            total_loss += float(loss) * n; total_sv += float(loss_sv) * n; total_sr += float(loss_sr) * n
            correct_sv += int((logits_sv.argmax(1) == y_sv).sum())
            correct_sr += int((logits_sr.argmax(1) == y_sr).sum())
        print("epoch=%d loss=%.4f sv_loss=%.4f sr_loss=%.4f sv_acc=%.4f sr_acc=%.4f" %
              (epoch, total_loss/seen, total_sv/seen, total_sr/seen, correct_sv/seen, correct_sr/seen))
    heads.eval()
    valid_genuine, valid_impostor = verification_scores(features["valid"], splits["valid"], heads)
    sv_cal = select_eer_threshold(valid_genuine, valid_impostor)
    with torch.no_grad():
        valid_sr_prob = torch.softmax(heads.sr_classifier(features["valid"].to(device)), 1)[:, 1].cpu()
    valid_sr_targets = torch.tensor([r["sensitive"] for r in splits["valid"]])
    sr_threshold, valid_sr_metrics = select_sr_threshold(valid_sr_prob, valid_sr_targets)
    test_genuine, test_impostor = verification_scores(features["test"], splits["test"], heads)
    test_sv = {"far": float((test_impostor >= sv_cal["threshold"]).float().mean()),
               "frr": float((test_genuine < sv_cal["threshold"]).float().mean())}
    with torch.no_grad():
        test_sr_prob = torch.softmax(heads.sr_classifier(features["test"].to(device)), 1)[:, 1].cpu()
    test_sr_metrics = binary_metrics(test_sr_prob, torch.tensor([r["sensitive"] for r in splits["test"]]), sr_threshold)
    checkpoint = {"format": "janus_distilhubert_fsc_heads_v1", "backbone_path": str(backbone_path),
                  "head_config": head_config, "speaker_to_id": speaker_to_id,
                  "sensitive_intent_labels": sorted(sensitive_labels), "model_state_dict": heads.state_dict()}
    torch.save(checkpoint, output_dir / "distilhubert_fsc_heads.pt")
    thresholds = {"sv_threshold": sv_cal["threshold"], "sr_threshold": sr_threshold,
                  "sv_score_space": "scaled_cosine_[0,1]", "sv_selection_rule": "validation_EER_K3",
                  "sr_selection_rule": "validation_sensitive_F1", "backbone": str(backbone_path),
                  "heads_checkpoint": str(output_dir / "distilhubert_fsc_heads.pt")}
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    report = {"validation_sv": sv_cal, "validation_sr": valid_sr_metrics,
              "test_sv_fixed_validation_threshold": test_sv, "test_sr_fixed_validation_threshold": test_sr_metrics,
              "num_train": len(splits["train"]), "num_valid": len(splits["valid"]), "num_test": len(splits["test"]),
              "num_train_speakers": len(speaker_names), "head_config": head_config}
    (output_dir / "clean_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "report": report}, indent=2))


if __name__ == "__main__":
    main()
