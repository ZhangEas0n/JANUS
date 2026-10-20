"""Align local physical recordings and build strict FSC Any-10 eval metadata."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "fluent_speech_commands_dataset"
SOURCE_ROOT = (
    PROJECT_ROOT
    / "recordings"
    / "physical_ota"
    / "local_fsc50_bird_snr_sweep_20260921"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "physical_ota"
    / "local_fsc50_bird_snr_sweep_20260921"
)
GROUPS = {
    1: "recordings_1db",
    2: "recordings_2db",
    5: "recordings_5db",
    10: "recordings_10db",
    20: "recordings_20db",
}
SAMPLE_RATE = 16000
TRIM_SECONDS = 3.0


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1), rate


def envelope(audio: np.ndarray, frame: int = 320, hop: int = 80) -> np.ndarray:
    squared = np.square(audio, dtype=np.float64)
    energy = np.convolve(squared, np.ones(frame) / frame, mode="valid")
    return np.sqrt(np.maximum(energy[::hop], 1e-12))


def estimate_offset(recording: np.ndarray, clean: np.ndarray) -> tuple[int, float]:
    rec_env = envelope(recording)
    clean_env = envelope(clean)
    clean_env = (clean_env - clean_env.mean()) / max(clean_env.std(), 1e-12)
    hop = 80
    # Recording starts 0.75 s before ffplay.  Keep the search close to this
    # known timing so a loud word inside the utterance cannot become a false
    # alignment peak.
    minimum = int(0.55 * SAMPLE_RATE / hop)
    maximum = int(0.90 * SAMPLE_RATE / hop)
    best_lag, best_score = minimum, -float("inf")
    for lag in range(minimum, maximum + 1):
        segment = rec_env[lag : lag + len(clean_env)]
        if len(segment) != len(clean_env):
            continue
        segment = (segment - segment.mean()) / max(segment.std(), 1e-12)
        score = float(np.mean(segment * clean_env))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag * hop, best_score


def build_enrollment_list(path: Path) -> None:
    valid_csv = DATASET_ROOT / "data" / "valid_data.csv"
    with valid_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speakerId"]].append(row)
    speakers = sorted(speaker for speaker, values in by_speaker.items() if len(values) >= 3)[:10]
    enrollment = []
    for speaker in speakers:
        for row in by_speaker[speaker][:3]:
            audio = (DATASET_ROOT / row["path"]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            enrollment.append(
                {"speaker_id": speaker, "enrollment_path": str(audio)}
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("speaker_id", "enrollment_path"))
        writer.writeheader()
        writer.writerows(enrollment)
    print("Enrollment speakers:", speakers)


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (SOURCE_ROOT / "selection_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        selected = {row["trial_id"]: row for row in csv.DictReader(handle)}

    alignment_rows = []
    for snr, directory_name in GROUPS.items():
        source_dir = SOURCE_ROOT / directory_name
        aligned_dir = RESULT_ROOT / "aligned" / f"{snr}db"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        metadata_rows = []
        for trial_id, selection in selected.items():
            recording_path = source_dir / f"{trial_id}_recorded.wav"
            clean_path = (DATASET_ROOT / selection["dataset_path"]).resolve()
            recording, recording_rate = load_mono(recording_path)
            clean, clean_rate = load_mono(clean_path)
            if recording_rate != SAMPLE_RATE or clean_rate != SAMPLE_RATE:
                raise ValueError(
                    f"Unexpected sample rate: {recording_path}={recording_rate}, "
                    f"{clean_path}={clean_rate}"
                )
            offset, score = estimate_offset(recording, clean)
            wanted = int(TRIM_SECONDS * SAMPLE_RATE)
            aligned = recording[offset : offset + wanted]
            if len(aligned) < wanted:
                aligned = np.pad(aligned, (0, wanted - len(aligned)))
            aligned_path = aligned_dir / f"{trial_id}_aligned.wav"
            sf.write(str(aligned_path), aligned.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
            offset_ms = 1000.0 * offset / SAMPLE_RATE
            alignment_rows.append(
                {
                    "snr_db": snr,
                    "trial_id": trial_id,
                    "source_recording": str(recording_path.resolve()),
                    "aligned_audio": str(aligned_path.resolve()),
                    "offset_ms": offset_ms,
                    "alignment_correlation": score,
                }
            )
            metadata_rows.append(
                {
                    "trial_id": f"{snr}db_{trial_id}",
                    "audio_path": str(aligned_path.resolve()),
                    "speaker_id": selection["speaker_id"],
                    "intent": "|".join(
                        (selection["action"], selection["object"], selection["location"])
                    ),
                    "enrollment_path": "",
                    "method": "JANUS-bird-physical-OTA",
                    "environment": "laptop-speaker-to-internal-microphone",
                    "distance_m": "",
                    "angle_deg": "",
                    "repeat_id": 1,
                    "offset_ms": f"{offset_ms:.3f}",
                    "clean_audio_path": str(clean_path),
                }
            )
        metadata_path = RESULT_ROOT / f"metadata_{snr}db.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"Prepared {snr} dB: {len(metadata_rows)} trials -> {metadata_path}")

    with (RESULT_ROOT / "alignment_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alignment_rows[0]))
        writer.writeheader()
        writer.writerows(alignment_rows)
    build_enrollment_list(RESULT_ROOT / "fsc_any10_k3_enrollment.csv")
    correlations = [row["alignment_correlation"] for row in alignment_rows]
    offsets = [row["offset_ms"] for row in alignment_rows]
    print(
        "Alignment: n={}, offset_ms=[{:.1f}, {:.1f}], median={:.1f}, "
        "corr=[{:.3f}, {:.3f}], median={:.3f}".format(
            len(alignment_rows), min(offsets), max(offsets), statistics.median(offsets),
            min(correlations), max(correlations), statistics.median(correlations),
        )
    )
    return 0


if __name__ == "__main__":
    import statistics

    raise SystemExit(main())
