import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


SPLIT_FILES = {
    "train": "train_data.csv",
    "valid": "valid_data.csv",
    "test": "test_data.csv",
}

REQUIRED_COLUMNS = [
    "path",
    "speakerId",
    "transcription",
    "action",
    "object",
    "location",
]

SENSITIVE_INTENTS = {
    "increase heat kitchen",
    "increase heat bedroom",
    "increase heat washroom",
    "decrease heat kitchen",
    "decrease heat bedroom",
    "decrease heat washroom",
    "activate lights kitchen",
    "activate lights bedroom",
    "activate lamp bedroom",
}

TEMPERATURE_KEYWORDS = {"heat", "temperature", "thermostat"}
DEVICE_CONTROL_KEYWORDS = {"light", "lights", "lamp", "fan", "music", "volume"}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_path(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", os.sep).replace("/", os.sep)


def load_sensitive_intents(path: Path = None) -> set:
    if path is None:
        return {normalize_text(intent) for intent in SENSITIVE_INTENTS}

    if not path.exists():
        raise FileNotFoundError(f"Sensitive intent file does not exist: {path}")

    intents = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            intents.add(normalize_text(line))

    if not intents:
        raise ValueError(f"No sensitive intents found in {path}")
    return intents


def read_split_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing FSC split CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    if "transcription" not in df.columns and "transcript" in df.columns:
        df = df.rename(columns={"transcript": "transcription"})

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def build_intent(row: pd.Series) -> str:
    parts = [
        normalize_text(row["action"]),
        normalize_text(row["object"]),
        normalize_text(row["location"]),
    ]
    return " ".join(parts)


def build_intent_pipe(row: pd.Series) -> str:
    parts = [
        normalize_text(row["action"]),
        normalize_text(row["object"]),
        normalize_text(row["location"]),
    ]
    return "|".join(parts)


def infer_sensitive_type(intent: str) -> str:
    tokens = set(intent.split())
    if tokens & TEMPERATURE_KEYWORDS:
        return "temperature"
    if tokens & DEVICE_CONTROL_KEYWORDS:
        return "device_control"
    return "benign"


def add_sr_labels(
    df: pd.DataFrame,
    fsc_root: Path,
    sensitive_intents: set,
) -> Tuple[pd.DataFrame, int]:
    processed = df.copy()

    for column in ["action", "object", "location"]:
        processed[column] = processed[column].map(normalize_text)
    processed["path"] = processed["path"].map(normalize_path)

    empty_required_mask = processed[REQUIRED_COLUMNS].apply(
        lambda col: col.map(lambda value: str(value).strip() == "")
    ).any(axis=1)
    skipped_rows = int(empty_required_mask.sum())
    if skipped_rows:
        processed = processed.loc[~empty_required_mask].copy()

    processed["intent"] = (
        processed["action"] + " " + processed["object"] + " " + processed["location"]
    )
    processed["intent_pipe"] = (
        processed["action"] + "|" + processed["object"] + "|" + processed["location"]
    )
    processed["is_sensitive"] = processed["intent"].map(
        lambda intent: 1 if intent in sensitive_intents else 0
    )
    processed["sensitive_type"] = processed["intent"].map(
        lambda intent: infer_sensitive_type(intent)
        if intent in sensitive_intents
        else "benign"
    )
    processed["full_path"] = processed["path"].map(
        lambda rel_path: str(fsc_root / rel_path)
    )
    processed["audio_exists"] = processed["full_path"].map(
        lambda full_path: 1 if Path(full_path).is_file() else 0
    )

    return processed, skipped_rows


def make_split_stats(df: pd.DataFrame, skipped_rows: int) -> Dict:
    total = int(len(df))
    sensitive = int(df["is_sensitive"].sum())
    benign = total - sensitive
    missing_audio = int((df["audio_exists"] == 0).sum())

    return {
        "total_samples": total,
        "skipped_malformed_rows": skipped_rows,
        "sensitive_samples": sensitive,
        "benign_samples": benign,
        "sensitive_ratio": sensitive / total if total else 0.0,
        "num_intents": int(df["intent"].nunique()),
        "intent_counts": {
            str(intent): int(count)
            for intent, count in df["intent"].value_counts().sort_index().items()
        },
        "speaker_counts": {
            str(speaker): int(count)
            for speaker, count in df["speakerId"].value_counts().sort_index().items()
        },
        "missing_audio_samples": missing_audio,
    }


def process_split(
    split: str,
    csv_path: Path,
    output_path: Path,
    fsc_root: Path,
    sensitive_intents: set,
) -> Dict:
    print(f"[*] Processing {split}: {csv_path}")
    raw_df = read_split_csv(csv_path)
    processed_df, skipped_rows = add_sr_labels(raw_df, fsc_root, sensitive_intents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False, encoding="utf-8")

    stats = make_split_stats(processed_df, skipped_rows)
    print(
        "[+] {split}: total={total}, sensitive={sensitive}, benign={benign}, "
        "ratio={ratio:.4f}, intents={intents}, missing_audio={missing_audio}, "
        "skipped={skipped}".format(
            split=split,
            total=stats["total_samples"],
            sensitive=stats["sensitive_samples"],
            benign=stats["benign_samples"],
            ratio=stats["sensitive_ratio"],
            intents=stats["num_intents"],
            missing_audio=stats["missing_audio_samples"],
            skipped=stats["skipped_malformed_rows"],
        )
    )
    print(f"    saved: {output_path}")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build SR sensitive-intent labels for Fluent Speech Commands CSV splits."
        )
    )
    parser.add_argument(
        "--fsc_root",
        required=True,
        type=Path,
        help="Root directory of the Fluent Speech Commands dataset.",
    )
    parser.add_argument(
        "--sensitive_intents_file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one sensitive intent per line. "
            "If omitted, the script uses the built-in SENSITIVE_INTENTS set."
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    fsc_root = args.fsc_root.expanduser().resolve()
    data_dir = fsc_root / "data"
    output_dir = fsc_root / "data_processed"

    if not fsc_root.exists():
        raise FileNotFoundError(f"FSC root does not exist: {fsc_root}")
    if not data_dir.exists():
        raise FileNotFoundError(f"FSC data directory does not exist: {data_dir}")

    sensitive_intents = load_sensitive_intents(args.sensitive_intents_file)
    print(f"[*] FSC root: {fsc_root}")
    print(f"[*] Output directory: {output_dir}")
    print(f"[*] Sensitive intents: {len(sensitive_intents)}")

    all_stats = {}
    for split, file_name in SPLIT_FILES.items():
        csv_path = data_dir / file_name
        output_path = output_dir / f"{split}_sr_labels.csv"
        all_stats[split] = process_split(
            split=split,
            csv_path=csv_path,
            output_path=output_path,
            fsc_root=fsc_root,
            sensitive_intents=sensitive_intents,
        )

    stats_path = output_dir / "sr_label_stats.json"
    stats_payload = {
        "fsc_root": str(fsc_root),
        "sensitive_intents": sorted(sensitive_intents),
        "splits": all_stats,
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2, ensure_ascii=False)

    print(f"[+] Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
