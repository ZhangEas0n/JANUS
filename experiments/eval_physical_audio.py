"""Evaluate physical re-recordings with the existing JANUS SV and sensitive-intent models."""

from __future__ import print_function

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("physical_audio_eval")
METADATA_FIELDS = (
    "trial_id", "audio_path", "speaker_id", "intent", "enrollment_path",
    "method", "environment", "distance_m", "angle_deg", "repeat_id",
    "offset_ms", "clean_audio_path",
)


@dataclass(frozen=True)
class EnrollmentRef:
    label: str
    path: Path


def _resolve_path(value, bases: Sequence[Path]) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers[:] = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(str(path), encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _read_enrollment_list(path: Path) -> List[EnrollmentRef]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    refs = []
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            if isinstance(row, str):
                value, label = row, Path(row).stem
            else:
                value = row.get("enrollment_path") or row.get("audio_path") or row.get("path")
                label = row.get("speaker_id") or row.get("label") or Path(str(value)).stem
            refs.append(EnrollmentRef(str(label), _resolve_path(value, (path.parent, PROJECT_ROOT))))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_no, row in enumerate(csv.DictReader(handle), 2):
                value = row.get("enrollment_path") or row.get("audio_path") or row.get("path")
                if not value:
                    raise ValueError("Enrollment CSV row {} has no path".format(line_no))
                label = row.get("speaker_id") or row.get("label") or Path(value).stem
                refs.append(EnrollmentRef(str(label), _resolve_path(value, (path.parent, PROJECT_ROOT))))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            label, value = parts if len(parts) == 2 else (Path(line).stem, line)
            refs.append(EnrollmentRef(str(label), _resolve_path(value, (path.parent, PROJECT_ROOT))))
    return _deduplicate(refs)


def _deduplicate(refs: Iterable[EnrollmentRef]) -> List[EnrollmentRef]:
    output, seen = [], set()
    for ref in refs:
        path = ref.path.resolve()
        if not path.is_file():
            raise FileNotFoundError("Missing enrollment audio: {}".format(path))
        key = (ref.label, str(path))
        if key not in seen:
            seen.add(key)
            output.append(EnrollmentRef(ref.label, path))
    if not output:
        raise ValueError("At least one enrollment audio is required")
    return output


def _global_enrollments(args) -> List[EnrollmentRef]:
    refs = []
    for value in args.enrollment or []:
        path = _resolve_path(value, (Path.cwd(), args.project_root))
        refs.append(EnrollmentRef(path.stem, path))
    if args.enrollment_list:
        refs.extend(_read_enrollment_list(args.enrollment_list))
    return _deduplicate(refs) if refs else []


def _row_enrollments(row: Dict[str, str], args, global_refs: Sequence[EnrollmentRef]) -> List[EnrollmentRef]:
    value = (row.get("enrollment_path") or "").strip()
    if not value:
        return list(global_refs)
    bases = (args.metadata.parent if args.metadata else args.audio_dir, args.audio_dir, args.project_root)
    refs = []
    for item in value.split(";"):
        path = _resolve_path(item, bases)
        refs.append(EnrollmentRef(row.get("speaker_id") or path.stem, path))
    return _deduplicate(refs)


def _audio_files(directory: Path) -> List[Path]:
    extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return sorted(p.resolve() for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _metadata_rows(
    metadata: Optional[Path],
    audio_dir: Path,
    max_files: Optional[int] = None,
) -> List[Dict[str, str]]:
    if metadata:
        with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Metadata CSV is empty: {}".format(metadata))
    else:
        rows = [
            {"trial_id": path.stem, "audio_path": str(path)}
            for path in _audio_files(audio_dir)
        ]
    return rows[:max_files] if max_files is not None else rows


def _run_worker(stage: str, manifest: Path, output: Path, args, model_paths: Dict[str, Path]) -> None:
    command = [
        sys.executable, str(args.project_root / "experiments" / "physical_eval_worker.py"),
        "--worker-stage", stage, "--manifest", str(manifest), "--output", str(output),
        "--device", args.device, "--project-root", str(args.project_root),
        "--sv-checkpoint", str(model_paths["sv_checkpoint"]),
        "--sr-checkpoint", str(model_paths["sr_checkpoint"]),
        "--backbone-path", str(model_paths["backbone_path"]),
        "--thresholds", str(model_paths["thresholds_path"]),
    ]
    LOGGER.info("Starting %s stage in an isolated process.", stage.upper())
    worker_env = os.environ.copy()
    worker_env["OMP_NUM_THREADS"] = "1"
    worker_env["MKL_NUM_THREADS"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=worker_env,
    )
    for line in process.stdout:
        LOGGER.info("[%s] %s", stage.upper(), line.rstrip())
    return_code = process.wait()
    if return_code:
        raise RuntimeError("{} worker failed with exit code {}".format(stage.upper(), return_code))
    LOGGER.info("%s stage: PASS", stage.upper())


def _merge_stage_results(requests: Sequence[Dict], sv_path: Path, sr_path: Path) -> Dict[str, Dict]:
    sv = {row["request_id"]: row for row in json.loads(sv_path.read_text(encoding="utf-8"))}
    sr = {row["request_id"]: row for row in json.loads(sr_path.read_text(encoding="utf-8"))}
    merged = {}
    for request in requests:
        request_id = request["request_id"]
        if request_id not in sv or request_id not in sr:
            raise ValueError("Missing staged output for {}".format(request_id))
        result = {
            "audio_path": sv[request_id]["audio_path"],
            "sample_rate": 16000,
            "audio_properties": sv[request_id]["audio_properties"],
            "sv": sv[request_id]["sv"],
            "sr": sr[request_id]["sr"],
        }
        result["joint"] = {
            "success": bool(
                result["sv"]["accepted"] and result["sr"]["is_sensitive"]
            )
        }
        result["union"] = {
            "success": bool(
                result["sv"]["accepted"] or result["sr"]["is_sensitive"]
            )
        }
        merged[request_id] = result
        LOGGER.info("Audio path: %s", result["audio_path"])
        LOGGER.info("SV score=%.6f threshold=%.6f accepted=%s", result["sv"]["best_score"], result["sv"]["threshold"], result["sv"]["accepted"])
        LOGGER.info("SR prediction=%s p_sensitive=%.6f threshold=%.3f sensitive=%s", result["sr"]["predicted_class"], result["sr"]["sensitive_probability"], result["sr"]["threshold"], result["sr"]["is_sensitive"])
        LOGGER.info("Joint success=%s", result["joint"]["success"])
        LOGGER.info("Union success=%s", result["union"]["success"])
    return merged


def _evaluate_requests(requests: Sequence[Dict], args, model_paths: Dict[str, Path]) -> Dict[str, Dict]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="physical_eval_", dir=str(args.output.parent)) as temp:
        temp = Path(temp)
        manifest = temp / "manifest.json"
        sv_output, sr_output = temp / "sv.json", temp / "sr.json"
        manifest.write_text(json.dumps({"requests": list(requests)}, indent=2), encoding="utf-8")
        _run_worker("sv", manifest, sv_output, args, model_paths)
        _run_worker("sr", manifest, sr_output, args, model_paths)
        return _merge_stage_results(requests, sv_output, sr_output)


def _flatten(metadata: Dict[str, str], ota: Dict, clean: Optional[Dict]) -> Dict:
    row = {key: metadata.get(key, "") for key in METADATA_FIELDS}
    props = ota["audio_properties"]
    row.update({
        "audio_path": ota["audio_path"], "duration": props["duration"],
        "sample_rate_original": props["sample_rate_original"], "sample_rate_model": props["sample_rate_model"],
        "rms": props["rms"], "peak_abs": props["peak_abs"], "clipping_rate": props["clipping_rate"],
        "sv_score": ota["sv"]["best_score"], "best_sv_score": ota["sv"]["best_score"],
        "best_enrollment": ota["sv"]["best_enrollment"], "num_enrollments": ota["sv"]["num_enrollments"],
        "sv_threshold": ota["sv"]["threshold"], "sv_accepted": ota["sv"]["accepted"],
        "sr_pred": ota["sr"]["predicted_class"], "p_sensitive": ota["sr"]["sensitive_probability"],
        "sr_threshold": ota["sr"]["threshold"], "is_sensitive": ota["sr"]["is_sensitive"],
        "joint_success": ota["joint"]["success"],
        "union_success": ota["union"]["success"],
    })
    if clean:
        eligible_sv = not clean["sv"]["accepted"]
        eligible_sr = not clean["sr"]["is_sensitive"]
        eligible_joint = eligible_sv and eligible_sr
        row.update({
            "sv_score_clean": clean["sv"]["best_score"], "sv_score_ota": ota["sv"]["best_score"],
            "sv_eligible": eligible_sv, "sv_success": bool(eligible_sv and ota["sv"]["accepted"]),
            "sr_pred_clean": clean["sr"]["predicted_class"], "sr_pred_ota": ota["sr"]["predicted_class"],
            "p_sensitive_clean": clean["sr"]["sensitive_probability"], "p_sensitive_ota": ota["sr"]["sensitive_probability"],
            "sr_eligible": eligible_sr, "sr_success": bool(eligible_sr and ota["sr"]["is_sensitive"]),
            "joint_eligible": eligible_joint,
            "joint_attack_success": bool(
                eligible_joint and ota["joint"]["success"]
            ),
            "union_eligible": eligible_joint,
            "union_attack_success": bool(
                eligible_joint and ota["union"]["success"]
            ),
        })
    return row


def _rate(rows: Sequence[Dict], key: str, eligible: Optional[str] = None) -> Optional[float]:
    selected = [row for row in rows if eligible is None or bool(row.get(eligible))]
    return None if not selected else sum(bool(row.get(key)) for row in selected) / float(len(selected))


def _summary(rows: Sequence[Dict]) -> Dict:
    clean = [row for row in rows if row.get("sv_score_clean", "") != ""]
    output = {
        "num_trials": len(rows), "conditional_asr_available": bool(clean),
        "far": _rate(rows, "sv_accepted"), "sensitive_rate": _rate(rows, "is_sensitive"),
        "joint_success_rate": _rate(rows, "joint_success"),
        "union_success_rate": _rate(rows, "union_success"),
    }
    if clean:
        output.update({
            "num_trials_with_clean_reference": len(clean),
            "asr_sv": _rate(clean, "sv_success", "sv_eligible"),
            "asr_sr": _rate(clean, "sr_success", "sr_eligible"),
            "asr_joint": _rate(clean, "joint_attack_success", "joint_eligible"),
            "asr_union": _rate(clean, "union_attack_success", "union_eligible"),
            "eligible_sv_trials": sum(bool(r.get("sv_eligible")) for r in clean),
            "eligible_sr_trials": sum(bool(r.get("sr_eligible")) for r in clean),
            "eligible_joint_trials": sum(bool(r.get("joint_eligible")) for r in clean),
            "eligible_union_trials": sum(bool(r.get("union_eligible")) for r in clean),
        })
    return output


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=Path)
    mode.add_argument("--audio-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--max-files",
        type=int,
        help="Evaluate at most this many sorted audio files or metadata rows.",
    )
    parser.add_argument("--enrollment", action="append")
    parser.add_argument("--enrollment-list", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sv-checkpoint", type=Path)
    parser.add_argument("--sr-checkpoint", type=Path)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--thresholds", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.audio = args.audio.resolve() if args.audio else None
    args.audio_dir = args.audio_dir.resolve() if args.audio_dir else None
    args.metadata = args.metadata.resolve() if args.metadata else None
    args.enrollment_list = args.enrollment_list.resolve() if args.enrollment_list else None
    args.output = args.output.resolve()
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be a positive integer")
    args.summary_output = args.summary_output.resolve() if args.summary_output else args.output.with_name(args.output.stem + "_summary.json")
    log_path = args.log.resolve() if args.log else args.output.with_suffix(".log")
    _setup_logging(log_path)
    model_paths = {
        "backbone_path": (args.backbone_path or args.project_root / "model" / "wav2vec2-base-960h").resolve(),
        "sv_checkpoint": (args.sv_checkpoint or args.project_root / "models" / "checkpoints" / "sv" / "wav2vec_base_inference_shards" / "index.json").resolve(),
        "sr_checkpoint": (args.sr_checkpoint or args.project_root / "models" / "checkpoints" / "sr" / "intent_head_fsc.pt").resolve(),
        "thresholds_path": (args.thresholds or args.project_root / "results" / "joint" / "evaluation" / "fsc_joint_thresholds.json").resolve(),
    }
    for label, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError("Missing {}: {}".format(label, path))
    global_refs = _global_enrollments(args)
    if args.audio:
        if not global_refs:
            raise ValueError("Single-file mode requires --enrollment or --enrollment-list")
        requests = [{"request_id": "single", "audio_path": str(args.audio), "enrollments": [{"label": r.label, "path": str(r.path)} for r in global_refs]}]
        result = _evaluate_requests(requests, args, model_paths)["single"]
        result["models"] = {key: str(value) for key, value in model_paths.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("JSON output path: %s", args.output)
        return 0

    if not args.audio_dir.is_dir():
        raise NotADirectoryError(args.audio_dir)
    prepared, requests = [], []
    metadata_rows = _metadata_rows(
        args.metadata, args.audio_dir, max_files=args.max_files
    )
    if not metadata_rows:
        raise ValueError("No supported audio files found in {}".format(args.audio_dir))
    LOGGER.info(
        "Batch input: %d audio trial(s) selected from %s",
        len(metadata_rows),
        args.audio_dir,
    )
    for index, metadata in enumerate(metadata_rows):
        trial = dict(metadata); trial.setdefault("trial_id", str(index))
        bases = (args.audio_dir, args.metadata.parent if args.metadata else args.audio_dir, args.project_root)
        audio = _resolve_path(trial.get("audio_path"), bases)
        if audio is None:
            raise ValueError("Trial {} has no audio_path".format(trial["trial_id"]))
        enrollments = _row_enrollments(trial, args, global_refs)
        if not enrollments:
            raise ValueError("Trial {} has no enrollment audio".format(trial["trial_id"]))
        enrollment_payload = [{"label": r.label, "path": str(r.path)} for r in enrollments]
        ota_id = "{}:ota".format(trial["trial_id"])
        requests.append({"request_id": ota_id, "audio_path": str(audio), "enrollments": enrollment_payload})
        clean = _resolve_path(trial.get("clean_audio_path"), bases)
        clean_id = None
        if clean:
            clean_id = "{}:clean".format(trial["trial_id"])
            requests.append({"request_id": clean_id, "audio_path": str(clean), "enrollments": enrollment_payload})
        prepared.append((trial, ota_id, clean_id))
    evaluated = _evaluate_requests(requests, args, model_paths)
    rows = [_flatten(trial, evaluated[ota_id], evaluated[clean_id] if clean_id else None) for trial, ota_id, clean_id in prepared]
    _write_csv(args.output, rows)
    summary = _summary(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("CSV output path: %s", args.output)
    LOGGER.info("Summary output path: %s", args.summary_output)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
