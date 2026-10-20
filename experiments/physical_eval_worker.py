"""Evaluate re-recorded physical audio with the existing FSC SV and sensitive-intent models."""

from __future__ import print_function

import argparse
import csv
import gc
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import torch.nn.functional as F
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from btuap.common import normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.fsc_sv import scaled_cosine_scores
from btuap.sr import (
    load_sensitive_intent_head_checkpoint,
    make_speech_recognition_cfg,
)
from scripts.eval.evaluate_fsc_joint import FSCVerificationMetadata
from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)
from src.main import construct_data_module, construct_module

LOGGER = logging.getLogger("physical_audio_eval")
MODEL_SAMPLE_RATE = 16000
METADATA_FIELDS = (
    "trial_id", "audio_path", "speaker_id", "intent", "enrollment_path",
    "method", "environment", "distance_m", "angle_deg", "repeat_id",
    "offset_ms", "clean_audio_path",
)


@dataclass(frozen=True)
class EnrollmentRef:
    label: str
    path: Path


def _register_resolvers() -> None:
    for name, resolver in (
        ("divide", division_resolver),
        ("idivide", integer_division_resolver),
        ("random_uuid", random_uuid),
    ):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def _compose_project_config(project_root: Path):
    _register_resolvers()
    with initialize_config_dir(config_dir=str((project_root / "config").resolve())):
        return compose(config_name="train_eval")


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


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers[:] = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _load_thresholds(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError("Missing frozen threshold file: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = ("sv_threshold", "sr_threshold")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("Threshold file is missing fields: {}".format(missing))
    if payload.get("sv_score_space") != "scaled_cosine_[0,1]":
        raise ValueError("Unexpected SV score space in {}".format(path))
    return payload


def _audio_files(directory: Path) -> List[Path]:
    suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return sorted(p.resolve() for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _read_enrollment_list(path: Path) -> List[EnrollmentRef]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("Missing enrollment list: {}".format(path))
    refs = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        for index, row in enumerate(rows):
            if isinstance(row, str):
                value, label = row, Path(row).stem
            else:
                value = row.get("enrollment_path") or row.get("audio_path") or row.get("path")
                label = row.get("speaker_id") or row.get("label") or Path(str(value)).stem
            resolved = _resolve_path(value, (path.parent, PROJECT_ROOT))
            refs.append(EnrollmentRef(str(label), resolved))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                value = row.get("enrollment_path") or row.get("audio_path") or row.get("path")
                if not value:
                    raise ValueError("Enrollment CSV row {} has no path".format(index + 2))
                label = row.get("speaker_id") or row.get("label") or Path(value).stem
                refs.append(EnrollmentRef(str(label), _resolve_path(value, (path.parent, PROJECT_ROOT))))
    else:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                label, value = parts
            else:
                value, label = line, Path(line).stem
            refs.append(EnrollmentRef(str(label), _resolve_path(value, (path.parent, PROJECT_ROOT))))
    if not refs:
        raise ValueError("Enrollment list is empty: {}".format(path))
    return refs


def _deduplicate_enrollments(refs: Iterable[EnrollmentRef]) -> List[EnrollmentRef]:
    output, seen = [], set()
    for ref in refs:
        path = ref.path.resolve()
        if not path.is_file():
            raise FileNotFoundError("Missing enrollment audio: {}".format(path))
        key = (ref.label, str(path))
        if key not in seen:
            output.append(EnrollmentRef(ref.label, path))
            seen.add(key)
    if not output:
        raise ValueError("At least one enrollment audio file is required")
    return output


def _cli_enrollments(args) -> List[EnrollmentRef]:
    refs = []
    for value in args.enrollment or []:
        path = _resolve_path(value, (Path.cwd(), args.project_root))
        refs.append(EnrollmentRef(path.stem, path))
    if args.enrollment_list:
        refs.extend(_read_enrollment_list(args.enrollment_list))
    return _deduplicate_enrollments(refs) if refs else []


class PhysicalAudioEvaluator:
    """Run SV and SR in separate stages to fit the 2 GiB inference host."""

    def __init__(
        self,
        stage: str,
        project_root: Path,
        device: torch.device,
        backbone_path: Path,
        sv_checkpoint: Path,
        sr_checkpoint: Path,
        thresholds_path: Path,
    ):
        self.project_root = project_root
        self.device = device
        self.backbone_path = backbone_path
        self.sv_checkpoint = sv_checkpoint
        self.sr_checkpoint = sr_checkpoint
        self.thresholds_path = thresholds_path
        self.stage = stage
        self.thresholds = _load_thresholds(thresholds_path)
        self.sv_threshold = float(self.thresholds["sv_threshold"])
        self.sr_threshold = float(self.thresholds["sr_threshold"])
        self._wave_cache = {}
        self._prepare_configs()

    def _prepare_configs(self) -> None:
        for path, label in (
            (self.backbone_path, "backbone"),
            (self.sv_checkpoint, "SV checkpoint"),
            (self.sr_checkpoint, "SR checkpoint"),
        ):
            if not path.exists():
                raise FileNotFoundError("Missing {}: {}".format(label, path))
        self.cfg = _compose_project_config(self.project_root)
        attack_cfg = BTUAPAttackConfig()
        attack_cfg.resolve_paths(self.project_root)
        self.model_evaluator = instantiate(self.cfg.evaluator)
        if self.stage == "sv":
            train_csv = (
                self.project_root / "fluent_speech_commands_dataset"
                / "data" / "train_data.csv"
            )
            with train_csv.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                train_speakers = len({
                    str(row["speakerId"]) for row in csv.DictReader(handle)
                })
            self.speaker_metadata = FSCVerificationMetadata(
                num_speakers=train_speakers
            )
            self.sv_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg, resolve=False)
            )
            self.sv_cfg.network.wav2vec_hunggingface_id = str(
                self.backbone_path
            )
            self.sv_cfg.load_network_from_checkpoint = (
                None if self.sv_checkpoint.suffix.lower() == ".json"
                else str(self.sv_checkpoint)
            )
        else:
            self.sr_cfg = make_speech_recognition_cfg(
                self.cfg,
                train_max_num_samples=attack_cfg.sr_train_max_num_samples,
                train_batch_size=attack_cfg.sr_train_batch_size,
                project_root=self.project_root,
            )
            self.sr_cfg.network.wav2vec_hunggingface_id = str(
                self.backbone_path
            )
            self.sr_cfg.tokenizer.tokenizer_huggingface_id = str(
                self.backbone_path
            )
            self.sr_dm = construct_data_module(self.sr_cfg)
            self.sensitive_labels = [
                label for label in attack_cfg.sensitive_intent_labels
                if label in self.sr_dm.intent_label_to_id
            ]
            if not self.sensitive_labels:
                raise ValueError(
                    "No configured sensitive FSC intents are present"
                )
        LOGGER.info(
            "Frozen thresholds: SV=%.9f, SR=%.3f",
            self.sv_threshold,
            self.sr_threshold,
        )

    def _load_audio(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        path = path.resolve()
        key = str(path)
        if key in self._wave_cache:
            return self._wave_cache[key]
        if not path.is_file():
            raise FileNotFoundError("Missing audio: {}".format(path))
        waveform, original_sr = torchaudio.load(str(path))
        if waveform.numel() == 0:
            raise ValueError("Empty audio: {}".format(path))
        mono = waveform.float().mean(dim=0)
        properties = {
            "duration": float(mono.numel()) / float(original_sr),
            "sample_rate_original": int(original_sr),
            "sample_rate_model": MODEL_SAMPLE_RATE,
            "rms": float(mono.pow(2).mean().sqrt()),
            "peak_abs": float(mono.abs().max()),
            "clipping_rate": float((mono.abs() >= 0.999).float().mean()),
        }
        mono_model = (
            torchaudio.functional.resample(mono, original_sr, MODEL_SAMPLE_RATE)
            if original_sr != MODEL_SAMPLE_RATE else mono
        )
        raw = mono_model.unsqueeze(0)
        lengths = torch.tensor([raw.shape[-1]], dtype=torch.long)
        normalized = normalize_batch_like_pipeline(raw, input_lengths=lengths)
        self._wave_cache[key] = (normalized, lengths, properties)
        return self._wave_cache[key]

    def _load_sv_model(self):
        model = construct_module(
            self.sv_cfg,
            self.model_evaluator,
            self.speaker_metadata,
            load_optim=False,
        )
        if self.sv_checkpoint.suffix.lower() == ".json":
            with self.sv_checkpoint.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            if index.get("format") != "janus_sharded_state_dict_v1":
                raise ValueError("Unsupported sharded checkpoint index")
            state = model.state_dict()
            loaded, unexpected = 0, []
            with torch.no_grad():
                for entry in index["tensors"]:
                    key = entry["key"]
                    tensor_path = self.sv_checkpoint.parent / entry["file"]
                    tensor = torch.load(str(tensor_path), map_location="cpu")
                    if key not in state:
                        unexpected.append(key)
                        del tensor
                        continue
                    if tuple(tensor.shape) != tuple(state[key].shape):
                        raise ValueError(
                            "SV tensor shape mismatch for {}: {} != {}".format(
                                key, tuple(tensor.shape), tuple(state[key].shape)
                            )
                        )
                    state[key].copy_(tensor)
                    loaded += 1
                    del tensor
            if loaded == 0:
                raise ValueError("No SV checkpoint tensors were loaded")
            LOGGER.info(
                "Loaded %d SV tensors from low-memory shards (%d task-head tensors ignored).",
                loaded, len(unexpected),
            )
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        LOGGER.info(
            "SV model loaded: Wav2Vec2 speaker encoder, checkpoint=%s",
            self.sv_checkpoint,
        )
        return model

    def _load_sr_model(self):
        model = construct_module(
            self.sr_cfg,
            self.model_evaluator,
            self.sr_dm,
            load_optim=False,
        ).to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        head = load_sensitive_intent_head_checkpoint(
            checkpoint_path=self.sr_checkpoint,
            sensitive_intent_labels=self.sensitive_labels,
            decision_threshold=self.sr_threshold,
            device=self.device,
        )
        LOGGER.info(
            "SR model loaded: SimpleSensitiveIntentHead (binary), checkpoint=%s",
            self.sr_checkpoint,
        )
        return model, head

    def evaluate_sv_many(
        self,
        requests: Sequence[Tuple[Path, Sequence[EnrollmentRef]]],
    ) -> List[Dict]:
        normalized_requests = [
            (Path(audio).resolve(), _deduplicate_enrollments(enrollments))
            for audio, enrollments in requests
        ]
        sv_model = self._load_sv_model()
        embeddings = {}
        all_paths = []
        for audio, enrollments in normalized_requests:
            all_paths.append(audio)
            all_paths.extend(ref.path for ref in enrollments)
        with torch.no_grad():
            for path in dict.fromkeys(p.resolve() for p in all_paths):
                normalized, _, _ = self._load_audio(path)
                embedding = sv_model.compute_speaker_embedding(normalized.to(self.device))
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                embeddings[str(path)] = F.normalize(embedding, p=2, dim=-1).detach().cpu()
        output = []
        for audio, enrollments in normalized_requests:
            grouped = {}
            for ref in enrollments:
                grouped.setdefault(ref.label, []).append(ref)
            gallery_refs, prototypes = [], []
            for label, refs in grouped.items():
                stacked = torch.cat(
                    [embeddings[str(ref.path.resolve())] for ref in refs], dim=0
                )
                stacked = F.normalize(stacked, p=2, dim=-1)
                prototypes.append(
                    F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=-1)
                )
                gallery_refs.append(refs[0])
            gallery = torch.cat(prototypes, dim=0)
            scores = [float(v) for v in scaled_cosine_scores(embeddings[str(audio)], gallery)[0].tolist()]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            best_score = scores[best_index]
            protocol = "single-enrollment" if len(gallery_refs) == 1 else ("Any-10" if len(gallery_refs) == 10 else "Any-{}".format(len(gallery_refs)))
            _, _, properties = self._load_audio(audio)
            output.append({
                "audio_path": str(audio),
                "audio_properties": properties,
                "sv": {
                    "protocol": protocol,
                    "scores": [{"enrollment": str(ref.path), "label": ref.label, "score": score} for ref, score in zip(gallery_refs, scores)],
                    "score": best_score,
                    "best_score": best_score,
                    "best_enrollment": str(gallery_refs[best_index].path),
                    "best_enrollment_label": gallery_refs[best_index].label,
                    "num_enrollments": len(gallery_refs),
                    "enrollment_k": max(len(refs) for refs in grouped.values()),
                    "threshold": self.sv_threshold,
                    "score_space": "scaled_cosine_[0,1]",
                    "accepted": bool(best_score >= self.sv_threshold),
                },
            })
        return output

    def evaluate_sr_many(self, audio_paths: Sequence[Path]) -> List[Dict]:
        sr_model, intent_head = self._load_sr_model()
        output = []
        probabilities_cache = {}
        with torch.no_grad():
            for audio in [Path(p).resolve() for p in audio_paths]:
                key = str(audio)
                if key not in probabilities_cache:
                    normalized, lengths, _ = self._load_audio(audio)
                    embedding, embedding_lengths = sr_model.compute_speech_embedding(
                        normalized.to(self.device), lengths.to(self.device)
                    )
                    probabilities_cache[key] = torch.softmax(
                        intent_head(embedding, embedding_lengths), dim=-1
                    )[0].detach().cpu()
                probabilities = probabilities_cache[key]
                p_sensitive = float(probabilities[1])
                is_sensitive = bool(p_sensitive >= self.sr_threshold)
                _, _, properties = self._load_audio(audio)
                output.append({
                    "audio_path": key,
                    "audio_properties": properties,
                    "sr": {
                        "mode": "binary_sensitive_intent",
                        "predicted_intent": None,
                        "predicted_class": "sensitive" if is_sensitive else "benign",
                        "class_probabilities": {"benign": float(probabilities[0]), "sensitive": p_sensitive},
                        "sensitive_probability": p_sensitive,
                        "threshold": self.sr_threshold,
                        "is_sensitive": is_sensitive,
                    },
                })
        return output

    def evaluate_many(
        self,
        requests: Sequence[Tuple[Path, Sequence[EnrollmentRef]]],
    ) -> List[Dict]:
        if not requests:
            return []
        normalized_requests = [
            (Path(audio).resolve(), _deduplicate_enrollments(enrollments))
            for audio, enrollments in requests
        ]

        sv_model = self._load_sv_model()
        sv_embeddings = {}
        sv_paths = []
        for audio, enrollments in normalized_requests:
            sv_paths.append(audio)
            sv_paths.extend(ref.path for ref in enrollments)
        with torch.no_grad():
            for path in dict.fromkeys(p.resolve() for p in sv_paths):
                normalized, _, _ = self._load_audio(path)
                embedding = sv_model.compute_speaker_embedding(
                    normalized.to(self.device)
                )
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                sv_embeddings[str(path)] = F.normalize(
                    embedding, p=2, dim=-1
                ).detach().cpu()
        sv_parts = []
        for audio, enrollments in normalized_requests:
            query = sv_embeddings[str(audio)]
            gallery = torch.cat(
                [sv_embeddings[str(ref.path.resolve())] for ref in enrollments],
                dim=0,
            )
            scores = scaled_cosine_scores(query, gallery)[0].tolist()
            scores = [float(value) for value in scores]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            sv_parts.append((scores, best_index))
        del sv_model
        del sv_embeddings
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info("SV stage complete; SV model released before SR loading.")

        sr_model, intent_head = self._load_sr_model()
        sr_probabilities = {}
        with torch.no_grad():
            for audio, _ in normalized_requests:
                key = str(audio)
                if key in sr_probabilities:
                    continue
                normalized, lengths, _ = self._load_audio(audio)
                speech_embedding, speech_lengths = sr_model.compute_speech_embedding(
                    normalized.to(self.device), lengths.to(self.device)
                )
                logits = intent_head(speech_embedding, speech_lengths)
                sr_probabilities[key] = torch.softmax(
                    logits, dim=-1
                )[0].detach().cpu()
        del sr_model
        del intent_head
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info("SR stage complete; SR model released.")

        results = []
        for (audio, enrollments), (scores, best_index) in zip(
            normalized_requests, sv_parts
        ):
            LOGGER.info("Audio path: %s", audio)
            probabilities = sr_probabilities[str(audio)]
            best_score = scores[best_index]
            p_sensitive = float(probabilities[1])
            accepted = bool(best_score >= self.sv_threshold)
            is_sensitive = bool(p_sensitive >= self.sr_threshold)
            protocol = "single-enrollment" if len(enrollments) == 1 else (
                "Any-10" if len(enrollments) == 10
                else "Any-{}".format(len(enrollments))
            )
            _, _, properties = self._load_audio(audio)
            result = {
                "audio_path": str(audio),
                "sample_rate": MODEL_SAMPLE_RATE,
                "audio_properties": properties,
                "sv": {
                    "protocol": protocol,
                    "scores": [
                        {
                            "enrollment": str(ref.path),
                            "label": ref.label,
                            "score": score,
                        }
                        for ref, score in zip(enrollments, scores)
                    ],
                    "score": best_score,
                    "best_score": best_score,
                    "best_enrollment": str(enrollments[best_index].path),
                    "best_enrollment_label": enrollments[best_index].label,
                    "num_enrollments": len(enrollments),
                    "threshold": self.sv_threshold,
                    "score_space": "scaled_cosine_[0,1]",
                    "accepted": accepted,
                },
                "sr": {
                    "mode": "binary_sensitive_intent",
                    "predicted_intent": None,
                    "predicted_class": "sensitive" if is_sensitive else "benign",
                    "class_probabilities": {
                        "benign": float(probabilities[0]),
                        "sensitive": p_sensitive,
                    },
                    "sensitive_probability": p_sensitive,
                    "threshold": self.sr_threshold,
                    "is_sensitive": is_sensitive,
                },
                "joint": {"success": bool(accepted and is_sensitive)},
            }
            LOGGER.info(
                "SV score=%.6f threshold=%.6f accepted=%s",
                best_score, self.sv_threshold, accepted,
            )
            LOGGER.info(
                "SR prediction=%s p_sensitive=%.6f threshold=%.3f sensitive=%s",
                result["sr"]["predicted_class"], p_sensitive,
                self.sr_threshold, is_sensitive,
            )
            LOGGER.info("Joint success=%s", result["joint"]["success"])
            results.append(result)
        return results

def _metadata_rows(metadata: Optional[Path], audio_dir: Path) -> List[Dict[str, str]]:
    if metadata:
        with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Metadata CSV is empty: {}".format(metadata))
        return rows
    return [
        {"trial_id": p.stem, "audio_path": str(p)} for p in _audio_files(audio_dir)
    ]


def _row_enrollments(row, args, global_refs: Sequence[EnrollmentRef]) -> List[EnrollmentRef]:
    value = row.get("enrollment_path", "").strip()
    if not value:
        return list(global_refs)
    refs = []
    for item in value.split(";"):
        path = _resolve_path(item, (args.metadata.parent if args.metadata else args.audio_dir, args.audio_dir, args.project_root))
        refs.append(EnrollmentRef(row.get("speaker_id") or path.stem, path))
    return _deduplicate_enrollments(refs)


def _flatten_batch_result(metadata: Dict[str, str], ota: Dict, clean: Optional[Dict]) -> Dict:
    row = {key: metadata.get(key, "") for key in METADATA_FIELDS}
    row.update({
        "audio_path": ota["audio_path"],
        "duration": ota["audio_properties"]["duration"],
        "sample_rate_original": ota["audio_properties"]["sample_rate_original"],
        "sample_rate_model": ota["audio_properties"]["sample_rate_model"],
        "rms": ota["audio_properties"]["rms"],
        "peak_abs": ota["audio_properties"]["peak_abs"],
        "clipping_rate": ota["audio_properties"]["clipping_rate"],
        "sv_score": ota["sv"]["best_score"],
        "best_sv_score": ota["sv"]["best_score"],
        "best_enrollment": ota["sv"]["best_enrollment"],
        "num_enrollments": ota["sv"]["num_enrollments"],
        "sv_threshold": ota["sv"]["threshold"],
        "sv_accepted": ota["sv"]["accepted"],
        "sr_pred": ota["sr"]["predicted_class"],
        "p_sensitive": ota["sr"]["sensitive_probability"],
        "sr_threshold": ota["sr"]["threshold"],
        "is_sensitive": ota["sr"]["is_sensitive"],
        "joint_success": ota["joint"]["success"],
    })
    if clean is not None:
        clean_sv = clean["sv"]["accepted"]
        clean_sr = clean["sr"]["is_sensitive"]
        eligible_sv = not clean_sv
        eligible_sr = not clean_sr
        eligible_joint = eligible_sv and eligible_sr
        row.update({
            "sv_score_clean": clean["sv"]["best_score"],
            "sv_score_ota": ota["sv"]["best_score"],
            "sv_eligible": eligible_sv,
            "sv_success": bool(eligible_sv and ota["sv"]["accepted"]),
            "sr_pred_clean": clean["sr"]["predicted_class"],
            "sr_pred_ota": ota["sr"]["predicted_class"],
            "p_sensitive_clean": clean["sr"]["sensitive_probability"],
            "p_sensitive_ota": ota["sr"]["sensitive_probability"],
            "sr_eligible": eligible_sr,
            "sr_success": bool(eligible_sr and ota["sr"]["is_sensitive"]),
            "joint_eligible": eligible_joint,
            "joint_attack_success": bool(eligible_joint and ota["joint"]["success"]),
        })
    return row


def _mean_bool(rows: Sequence[Dict], key: str, eligible_key: Optional[str] = None) -> Optional[float]:
    selected = [row for row in rows if eligible_key is None or row.get(eligible_key) in (True, "True", "true", 1, "1")]
    if not selected:
        return None
    return sum(bool(row.get(key)) for row in selected) / float(len(selected))


def _summary(rows: Sequence[Dict]) -> Dict:
    has_clean = [row for row in rows if row.get("sv_score_clean", "") != ""]
    summary = {
        "num_trials": len(rows),
        "conditional_asr_available": bool(has_clean),
        "far": _mean_bool(rows, "sv_accepted"),
        "sensitive_rate": _mean_bool(rows, "is_sensitive"),
        "joint_success_rate": _mean_bool(rows, "joint_success"),
    }
    if has_clean:
        summary.update({
            "num_trials_with_clean_reference": len(has_clean),
            "asr_sv": _mean_bool(has_clean, "sv_success", "sv_eligible"),
            "asr_sr": _mean_bool(has_clean, "sr_success", "sr_eligible"),
            "asr_joint": _mean_bool(has_clean, "joint_attack_success", "joint_eligible"),
            "eligible_sv_trials": sum(bool(row.get("sv_eligible")) for row in has_clean),
            "eligible_sr_trials": sum(bool(row.get("sr_eligible")) for row in has_clean),
            "eligible_joint_trials": sum(bool(row.get("joint_eligible")) for row in has_clean),
        })
    return summary


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=Path, help="One recorded physical audio file")
    mode.add_argument("--audio-dir", type=Path, help="Directory of recorded audio files")
    parser.add_argument("--metadata", type=Path, help="Batch trial metadata CSV")
    parser.add_argument("--enrollment", action="append", help="Enrollment audio; repeat for multiple enrollments")
    parser.add_argument("--enrollment-list", type=Path, help="TXT, CSV, or JSON enrollment list")
    parser.add_argument("--output", required=True, type=Path, help="Single JSON or batch CSV output")
    parser.add_argument("--summary-output", type=Path, help="Batch summary JSON path")
    parser.add_argument("--log", type=Path, help="Log path")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
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
    args.summary_output = args.summary_output.resolve() if args.summary_output else args.output.with_name(args.output.stem + "_summary.json")
    log_path = args.log.resolve() if args.log else args.output.with_suffix(".log")
    _setup_logging(log_path)
    device = _select_device(args.device)
    LOGGER.info("Device: %s", device)

    backbone = (args.backbone_path or args.project_root / "model" / "wav2vec2-base-960h").resolve()
    sv_checkpoint = (args.sv_checkpoint or args.project_root / "models" / "checkpoints" / "sv" / "wav2vec_base_inference_shards" / "index.json").resolve()
    sr_checkpoint = (args.sr_checkpoint or args.project_root / "models" / "checkpoints" / "sr" / "intent_head_fsc.pt").resolve()
    thresholds = (args.thresholds or args.project_root / "results" / "joint" / "evaluation" / "fsc_joint_thresholds.json").resolve()
    evaluator = PhysicalAudioEvaluator(args.project_root, device, backbone, sv_checkpoint, sr_checkpoint, thresholds)
    global_refs = _cli_enrollments(args)

    if args.audio:
        if not global_refs:
            raise ValueError("Single-file mode requires --enrollment or --enrollment-list")
        result = evaluator.evaluate_many([(args.audio, global_refs)])[0]
        result["models"] = {
            "sv_checkpoint": str(sv_checkpoint),
            "sr_checkpoint": str(sr_checkpoint),
            "backbone_path": str(backbone),
            "thresholds_path": str(thresholds),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        LOGGER.info("Output path: %s", args.output)
        return 0

    if not args.audio_dir.is_dir():
        raise NotADirectoryError(args.audio_dir)
    prepared = []
    requests = []
    for index, metadata in enumerate(_metadata_rows(args.metadata, args.audio_dir)):
        trial = dict(metadata)
        trial.setdefault("trial_id", str(index))
        bases = (
            args.audio_dir,
            args.metadata.parent if args.metadata else args.audio_dir,
            args.project_root,
        )
        audio_path = _resolve_path(trial.get("audio_path"), bases)
        if audio_path is None:
            raise ValueError("Trial {} has no audio_path".format(trial["trial_id"]))
        enrollments = _row_enrollments(trial, args, global_refs)
        if not enrollments:
            raise ValueError("Trial {} has no enrollment audio".format(trial["trial_id"]))
        clean_path = _resolve_path(trial.get("clean_audio_path"), bases)
        ota_index = len(requests)
        requests.append((audio_path, enrollments))
        clean_index = None
        if clean_path is not None:
            clean_index = len(requests)
            requests.append((clean_path, enrollments))
        prepared.append((trial, ota_index, clean_index))
    evaluated = evaluator.evaluate_many(requests)
    rows = [
        _flatten_batch_result(
            trial,
            evaluated[ota_index],
            evaluated[clean_index] if clean_index is not None else None,
        )
        for trial, ota_index, clean_index in prepared
    ]
    if not rows:
        raise ValueError("No audio trials found")
    _write_csv(args.output, rows)
    summary = _summary(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    LOGGER.info("CSV output path: %s", args.output)
    LOGGER.info("Summary output path: %s", args.summary_output)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


def worker_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-stage", required=True, choices=("sv", "sr"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sv-checkpoint", required=True, type=Path)
    parser.add_argument("--sr-checkpoint", required=True, type=Path)
    parser.add_argument("--backbone-path", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    device = _select_device(args.device)
    evaluator = PhysicalAudioEvaluator(
        args.worker_stage, args.project_root.resolve(), device,
        args.backbone_path.resolve(),
        args.sv_checkpoint.resolve(), args.sr_checkpoint.resolve(),
        args.thresholds.resolve(),
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requests = []
    for row in manifest["requests"]:
        enrollments = [EnrollmentRef(item["label"], Path(item["path"])) for item in row["enrollments"]]
        requests.append((Path(row["audio_path"]), enrollments))
    if args.worker_stage == "sv":
        results = evaluator.evaluate_sv_many(requests)
    else:
        results = evaluator.evaluate_sr_many([audio for audio, _ in requests])
    payload = [dict(request_id=row["request_id"], **result) for row, result in zip(manifest["requests"], results)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
