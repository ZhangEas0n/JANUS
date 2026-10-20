from __future__ import print_function

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from btuap.config import BTUAPAttackConfig


REQUIRED_COLUMNS = (
    "path",
    "speakerId",
    "transcription",
    "action",
    "object",
    "location",
)
SPLIT_FILES = {
    "train": "train_data.csv",
    "validation": "valid_data.csv",
    "test": "test_data.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the fixed FSC protocol for same-input joint SV-SR evaluation."
    )
    parser.add_argument("--fsc_root", required=True, type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: FSC_ROOT/data_processed/fsc_joint_protocol",
    )
    parser.add_argument("--enrollment_k", type=int, default=3)
    parser.add_argument("--enrollment_seed", type=int, default=0)
    parser.add_argument(
        "--pairing_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for test random-target impersonation trials.",
    )
    parser.add_argument(
        "--validation_pairing_seed",
        type=int,
        default=0,
        help="Seed used for validation non-target trials.",
    )
    return parser.parse_args()


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensitive_type(intent, is_sensitive):
    if not is_sensitive:
        return "benign"
    tokens = set(intent.split("|"))
    if "heat" in tokens:
        return "temperature"
    if tokens.intersection({"light", "lights", "lamp"}):
        return "device_control"
    return "other_sensitive"


def load_split(fsc_root, split, csv_name, sensitive_intents):
    csv_path = fsc_root / "data" / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError("Missing FSC CSV: {}".format(csv_path))

    frame = pd.read_csv(csv_path)
    if "transcription" not in frame.columns and "transcript" in frame.columns:
        frame = frame.rename(columns={"transcript": "transcription"})

    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError("{} is missing columns: {}".format(csv_path, missing))

    for name in REQUIRED_COLUMNS:
        blank = frame[name].isna() | frame[name].astype(str).str.strip().eq("")
        if blank.any():
            rows = (frame.index[blank] + 2).tolist()[:10]
            raise ValueError(
                "{} has blank values in '{}', CSV rows: {}".format(
                    csv_path, name, rows
                )
            )

    frame = frame.copy().reset_index(drop=True)
    frame["dataset_index"] = frame.index
    frame["sample_id"] = ["fsc-{}".format(index) for index in frame.index]
    frame["split"] = split
    for name in ("action", "object", "location"):
        frame[name] = frame[name].map(normalize_text)
    frame["speakerId"] = frame["speakerId"].map(lambda value: str(value).strip())
    frame["transcription"] = frame["transcription"].map(
        lambda value: str(value).strip()
    )
    frame["intent"] = frame[["action", "object", "location"]].agg("|".join, axis=1)
    frame["is_sensitive_label"] = frame["intent"].isin(sensitive_intents).astype(int)
    frame["sensitive_type"] = [
        sensitive_type(intent, bool(label))
        for intent, label in zip(frame["intent"], frame["is_sensitive_label"])
    ]

    def resolve_audio(relative_path):
        candidate = Path(str(relative_path))
        if not candidate.is_absolute():
            candidate = fsc_root / candidate
        return candidate.resolve()

    full_paths = frame["path"].map(resolve_audio)
    frame["full_path"] = full_paths.map(str)
    frame["audio_exists"] = full_paths.map(lambda path: int(path.is_file()))
    return frame, csv_path


def validate_splits(frames, enrollment_k):
    speaker_sets = {
        split: set(frame["speakerId"].tolist()) for split, frame in frames.items()
    }
    names = list(speaker_sets)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlap = speaker_sets[first].intersection(speaker_sets[second])
            if overlap:
                raise ValueError(
                    "FSC speaker leakage between {} and {}: {}".format(
                        first, second, sorted(overlap)[:10]
                    )
                )

    for split in ("validation", "test"):
        counts = frames[split].groupby("speakerId").size()
        too_small = counts[counts <= enrollment_k]
        if not too_small.empty:
            raise ValueError(
                "{} speakers need more than K={} utterances: {}".format(
                    split, enrollment_k, too_small.to_dict()
                )
            )

    missing_audio = sum(
        int((frame["audio_exists"] == 0).sum()) for frame in frames.values()
    )
    if missing_audio:
        raise FileNotFoundError(
            "FSC protocol contains {} missing audio files.".format(missing_audio)
        )


def select_enrollment(frame, split, enrollment_k, seed):
    rng = random.Random(int(seed))
    enrollment_rows = []
    enrollment_ids = set()
    for speaker_id, speaker_frame in frame.groupby("speakerId", sort=True):
        candidates = speaker_frame["sample_id"].tolist()
        selected = rng.sample(candidates, enrollment_k)
        selected_rows = frame[frame["sample_id"].isin(selected)].copy()
        rank_by_id = {sample_id: rank for rank, sample_id in enumerate(selected, 1)}
        selected_rows["enrollment_rank"] = selected_rows["sample_id"].map(rank_by_id)
        enrollment_rows.append(selected_rows)
        enrollment_ids.update(selected)

    gallery = pd.concat(enrollment_rows, ignore_index=True)
    gallery = gallery.sort_values(["speakerId", "enrollment_rank"]).reset_index(drop=True)
    gallery["split"] = split
    queries = frame[~frame["sample_id"].isin(enrollment_ids)].copy()
    return gallery, queries.reset_index(drop=True)


def gallery_lookup(gallery):
    lookup = {}
    for speaker_id, rows in gallery.groupby("speakerId", sort=True):
        rows = rows.sort_values("enrollment_rank")
        lookup[str(speaker_id)] = {
            "sample_ids": rows["sample_id"].tolist(),
            "paths": rows["path"].astype(str).tolist(),
            "full_paths": rows["full_path"].astype(str).tolist(),
        }
    return lookup


def trial_row(query, split, trial_type, seed, target_speaker, gallery_by_speaker):
    enrollment = gallery_by_speaker[str(target_speaker)]
    return {
        "split": split,
        "trial_type": trial_type,
        "seed": int(seed),
        "query_sample_id": query["sample_id"],
        "query_path": query["path"],
        "query_full_path": query["full_path"],
        "query_speaker": query["speakerId"],
        "enroll_speaker": str(target_speaker),
        "enroll_sample_ids": json.dumps(enrollment["sample_ids"]),
        "enroll_paths": json.dumps(enrollment["paths"]),
        "enroll_full_paths": json.dumps(enrollment["full_paths"]),
        "transcription": query["transcription"],
        "intent": query["intent"],
        "is_sensitive_label": int(query["is_sensitive_label"]),
    }


def build_validation_trials(queries, gallery, seed):
    gallery_by_speaker = gallery_lookup(gallery)
    speakers = sorted(gallery_by_speaker)
    rng = random.Random(int(seed))
    rows = []
    for _, query in queries.sort_values("sample_id").iterrows():
        query_speaker = str(query["speakerId"])
        rows.append(
            trial_row(
                query,
                "validation",
                "genuine",
                seed,
                query_speaker,
                gallery_by_speaker,
            )
        )
        non_targets = [speaker for speaker in speakers if speaker != query_speaker]
        rows.append(
            trial_row(
                query,
                "validation",
                "impostor",
                seed,
                rng.choice(non_targets),
                gallery_by_speaker,
            )
        )
    trials = pd.DataFrame(rows)
    trials.insert(0, "trial_id", ["val-{:07d}".format(i) for i in range(len(trials))])
    return trials


def build_test_impostor_trials(queries, gallery, pairing_seeds):
    gallery_by_speaker = gallery_lookup(gallery)
    speakers = sorted(gallery_by_speaker)
    benign_queries = queries[queries["is_sensitive_label"] == 0].copy()
    rows = []
    for seed in pairing_seeds:
        rng = random.Random(int(seed))
        for _, query in benign_queries.sort_values("sample_id").iterrows():
            query_speaker = str(query["speakerId"])
            non_targets = [speaker for speaker in speakers if speaker != query_speaker]
            rows.append(
                trial_row(
                    query,
                    "test",
                    "impostor",
                    seed,
                    rng.choice(non_targets),
                    gallery_by_speaker,
                )
            )
    trials = pd.DataFrame(rows)
    trials.insert(
        0, "trial_id", ["test-impostor-{:07d}".format(i) for i in range(len(trials))]
    )
    return trials


def build_test_genuine_trials(queries, gallery, seed):
    gallery_by_speaker = gallery_lookup(gallery)
    rows = [
        trial_row(
            query,
            "test",
            "genuine",
            seed,
            str(query["speakerId"]),
            gallery_by_speaker,
        )
        for _, query in queries.sort_values("sample_id").iterrows()
    ]
    trials = pd.DataFrame(rows)
    trials.insert(
        0, "trial_id", ["test-genuine-{:07d}".format(i) for i in range(len(trials))]
    )
    return trials


def split_stats(frame):
    return {
        "utterances": int(len(frame)),
        "speakers": int(frame["speakerId"].nunique()),
        "intents": int(frame["intent"].nunique()),
        "sensitive": int(frame["is_sensitive_label"].sum()),
        "benign": int((frame["is_sensitive_label"] == 0).sum()),
        "sensitive_ratio": float(frame["is_sensitive_label"].mean()),
        "missing_audio": int((frame["audio_exists"] == 0).sum()),
        "intent_counts": dict(Counter(frame["intent"].tolist())),
        "speaker_counts": dict(Counter(frame["speakerId"].tolist())),
    }


def main():
    args = parse_args()
    fsc_root = args.fsc_root.expanduser().resolve()
    if args.enrollment_k <= 0:
        raise ValueError("--enrollment_k must be positive")
    if len(set(args.pairing_seeds)) != len(args.pairing_seeds):
        raise ValueError("--pairing_seeds must not contain duplicates")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fsc_root / "data_processed" / "fsc_joint_protocol"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_cfg = BTUAPAttackConfig()
    sensitive_intents = {
        normalize_text(label).replace(" ", "|")
        for label in attack_cfg.sensitive_intent_labels
    }

    frames = {}
    source_paths = {}
    print("[*] Loading FSC splits from {}".format(fsc_root))
    for split, csv_name in SPLIT_FILES.items():
        frame, source_path = load_split(
            fsc_root, split, csv_name, sensitive_intents
        )
        frames[split] = frame
        source_paths[split] = source_path
        print(
            "[*] {}: utterances={}, speakers={}, intents={}, sensitive={}".format(
                split,
                len(frame),
                frame["speakerId"].nunique(),
                frame["intent"].nunique(),
                int(frame["is_sensitive_label"].sum()),
            )
        )

    validate_splits(frames, args.enrollment_k)

    processed_paths = {}
    for split, frame in frames.items():
        destination = output_dir / "{}_fsc_joint.csv".format(split)
        frame.to_csv(destination, index=False)
        processed_paths[split] = destination

    val_gallery, val_queries = select_enrollment(
        frames["validation"],
        "validation",
        args.enrollment_k,
        args.enrollment_seed,
    )
    test_gallery, test_queries = select_enrollment(
        frames["test"], "test", args.enrollment_k, args.enrollment_seed
    )
    gallery = pd.concat([val_gallery, test_gallery], ignore_index=True)
    gallery_path = output_dir / "fsc_joint_enrollment_gallery.csv"
    gallery.to_csv(gallery_path, index=False)

    validation_trials = build_validation_trials(
        val_queries, val_gallery, args.validation_pairing_seed
    )
    validation_path = output_dir / "fsc_joint_validation_trials.csv"
    validation_trials.to_csv(validation_path, index=False)

    test_trials = build_test_impostor_trials(
        test_queries, test_gallery, args.pairing_seeds
    )
    test_trials_path = output_dir / "fsc_joint_trials_all.csv"
    test_trials.to_csv(test_trials_path, index=False)

    genuine_trials = build_test_genuine_trials(
        test_queries, test_gallery, args.enrollment_seed
    )
    genuine_trials_path = output_dir / "fsc_joint_test_genuine_trials.csv"
    genuine_trials.to_csv(genuine_trials_path, index=False)

    stats = {
        "protocol_version": 1,
        "fsc_root": str(fsc_root),
        "sensitive_intents": sorted(sensitive_intents),
        "enrollment_k": int(args.enrollment_k),
        "enrollment_seed": int(args.enrollment_seed),
        "validation_pairing_seed": int(args.validation_pairing_seed),
        "test_pairing_seeds": [int(seed) for seed in args.pairing_seeds],
        "splits": {split: split_stats(frame) for split, frame in frames.items()},
        "protocol_counts": {
            "validation_enrollment_utterances": int(len(val_gallery)),
            "validation_queries": int(len(val_queries)),
            "validation_genuine_trials": int(
                (validation_trials["trial_type"] == "genuine").sum()
            ),
            "validation_impostor_trials": int(
                (validation_trials["trial_type"] == "impostor").sum()
            ),
            "test_enrollment_utterances": int(len(test_gallery)),
            "test_queries_all": int(len(test_queries)),
            "test_queries_benign": int(
                (test_queries["is_sensitive_label"] == 0).sum()
            ),
            "test_impostor_trials": int(len(test_trials)),
            "test_genuine_trials": int(len(genuine_trials)),
        },
        "source_sha256": {
            split: file_sha256(path) for split, path in source_paths.items()
        },
        "outputs": {
            "processed_splits": {
                split: str(path) for split, path in processed_paths.items()
            },
            "enrollment_gallery": str(gallery_path),
            "validation_trials": str(validation_path),
            "test_impostor_trials": str(test_trials_path),
            "test_genuine_trials": str(genuine_trials_path),
        },
    }
    stats_path = output_dir / "fsc_joint_protocol_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print("[+] FSC joint protocol written to {}".format(output_dir))
    print(
        "[+] validation: genuine={}, impostor={}; test: impostor={}, genuine={}".format(
            stats["protocol_counts"]["validation_genuine_trials"],
            stats["protocol_counts"]["validation_impostor_trials"],
            stats["protocol_counts"]["test_impostor_trials"],
            stats["protocol_counts"]["test_genuine_trials"],
        )
    )
    print(
        "[*] Genuine trials are auxiliary FRR diagnostics and are not part of "
        "ASR-SV-joint or ASR-Joint."
    )


if __name__ == "__main__":
    main()
