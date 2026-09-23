"""Evaluate a fixed Any-10 FSC gallery under a joint SV-SR protocol.

The ten clean enrollment identities come from the FSC validation split.  Test
speakers are disjoint source identities.  Results report both pair-level rates
and query-level success where acceptance by any of the ten identities counts.
"""

from __future__ import print_function

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from btuap.common import apply_async_perturbation, normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.fsc_sv import build_fsc_enrollment_gallery, scaled_cosine_scores
from btuap.kenansville import perturbation_statistics, rescale_to_snr
from btuap.sr import (
    extract_raw_audio_batch,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from evaluate_fsc_joint import (
    FSCVerificationMetadata,
    load_uap,
)
from src.main import construct_data_module, construct_module
from tqdm import tqdm


load_dotenv()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _cfg_get(cfg, key, default):
    if "fsc_any10_eval" in cfg and key in cfg.fsc_any10_eval:
        return cfg.fsc_any10_eval[key]
    return default


def _safe_rate(mask):
    values = np.asarray(mask, dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


def _optional_float(value):
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None
    return float(value)


def _summarize_distortion(rows):
    summary = {"num_utterances": int(len(rows))}
    for key in ("linf", "rms", "snr_db"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key + "_mean"] = float(values.mean()) if values.size else None
        summary[key + "_std"] = float(values.std()) if values.size else None
    return summary


def collect_uap_outputs(
    dataloader,
    sv_model,
    sr_model,
    intent_head,
    uap_delta,
    device,
    target_snr_db,
    sample_rate,
    offset_seconds,
    epsilon,
    compute_sr,
    description,
):
    """Collect clean/UAP outputs, optionally matching every sample to one SNR."""
    outputs = {}
    distortion_rows = []
    sv_model.eval()
    sr_model.eval()
    intent_head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            batch = batch.to(device)
            raw_audio = extract_raw_audio_batch(batch, device)
            attacked_raw, _ = apply_async_perturbation(
                raw_audio,
                uap_delta,
                sample_rate=sample_rate,
                offset_seconds=offset_seconds,
                epsilon=epsilon,
            )
            if target_snr_db is not None:
                attacked_raw = rescale_to_snr(
                    raw_audio,
                    attacked_raw,
                    target_snr_db=target_snr_db,
                    input_lengths=batch.input_lengths,
                )
            distortion_rows.extend(
                perturbation_statistics(
                    raw_audio, attacked_raw, input_lengths=batch.input_lengths
                )
            )

            clean_input = normalize_batch_like_pipeline(
                raw_audio, input_lengths=batch.input_lengths
            )
            attacked_input = normalize_batch_like_pipeline(
                attacked_raw, input_lengths=batch.input_lengths
            )
            clean_sv = sv_model.compute_speaker_embedding(clean_input)
            attacked_sv = sv_model.compute_speaker_embedding(attacked_input)
            if clean_sv.dim() == 1:
                clean_sv = clean_sv.unsqueeze(0)
                attacked_sv = attacked_sv.unsqueeze(0)
            clean_sv = F.normalize(clean_sv, p=2, dim=-1)
            attacked_sv = F.normalize(attacked_sv, p=2, dim=-1)

            clean_probabilities = [None] * len(batch.keys)
            attacked_probabilities = [None] * len(batch.keys)
            if compute_sr:
                clean_sr, clean_lengths = sr_model.compute_speech_embedding(
                    clean_input, batch.input_lengths
                )
                attacked_sr, attacked_lengths = sr_model.compute_speech_embedding(
                    attacked_input, batch.input_lengths
                )
                clean_probabilities = torch.softmax(
                    intent_head(clean_sr, clean_lengths), dim=-1
                )[:, 1].detach().cpu().tolist()
                attacked_probabilities = torch.softmax(
                    intent_head(attacked_sr, attacked_lengths), dim=-1
                )[:, 1].detach().cpu().tolist()

            for index, sample_id in enumerate(batch.keys):
                outputs[str(sample_id)] = {
                    "clean_sv": clean_sv[index].detach().cpu(),
                    "adv_sv": attacked_sv[index].detach().cpu(),
                    "clean_p_sens": clean_probabilities[index],
                    "adv_p_sens": attacked_probabilities[index],
                }
    return outputs, distortion_rows


def _load_sv_threshold(path, fallback):
    path = Path(path)
    if not path.is_file():
        print(
            "[!] Missing threshold file {}; using fallback {:.6f}".format(
                path, fallback
            )
        )
        return float(fallback), None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["sv_threshold"]), payload.get("validation_eer")


def _intent_label(row):
    return "|".join(
        str(row[name]).strip().lower() for name in ("action", "object", "location")
    )


def _stack_output_embeddings(outputs, sample_ids, key):
    missing = [sample_id for sample_id in sample_ids if sample_id not in outputs]
    if missing:
        raise KeyError("Missing inference outputs: {}".format(missing[:10]))
    return torch.stack([outputs[sample_id][key] for sample_id in sample_ids])


def _make_pair_frame(
    sample_ids,
    source_speakers,
    transcriptions,
    target_speakers,
    clean_scores,
    adv_scores,
    clean_p_sensitive,
    adv_p_sensitive,
    sv_threshold,
    sr_threshold,
):
    query_count = len(sample_ids)
    target_count = len(target_speakers)
    query_indices = np.repeat(np.arange(query_count), target_count)
    target_indices = np.tile(np.arange(target_count), query_count)

    frame = pd.DataFrame(
        {
            "query_sample_id": np.asarray(sample_ids)[query_indices],
            "query_speaker": np.asarray(source_speakers)[query_indices],
            "transcription": np.asarray(transcriptions)[query_indices],
            "target_index": target_indices,
            "target_speaker": np.asarray(target_speakers)[target_indices],
            "clean_sv_score": clean_scores.reshape(-1),
            "adv_sv_score": adv_scores.reshape(-1),
            "clean_p_sens": np.repeat(clean_p_sensitive, target_count),
            "adv_p_sens": np.repeat(adv_p_sensitive, target_count),
        }
    )
    frame["eligible_pair"] = (
        (frame["clean_sv_score"] < sv_threshold)
        & (frame["clean_p_sens"] < sr_threshold)
    )
    frame["sv_success"] = frame["eligible_pair"] & (
        frame["adv_sv_score"] >= sv_threshold
    )
    frame["sr_success"] = frame["eligible_pair"] & (
        frame["adv_p_sens"] >= sr_threshold
    )
    frame["joint_success"] = frame["sv_success"] & frame["sr_success"]
    frame["union_success"] = frame["sv_success"] | frame["sr_success"]
    return frame


def _summarize_pairs(frame, sv_threshold):
    eligible = frame[frame["eligible_pair"]]
    return {
        "num_pairs": int(len(frame)),
        "eligible_pairs": int(len(eligible)),
        "eligibility_retention_rate": float(len(eligible) / max(1, len(frame))),
        "clean_pair_far": _safe_rate(frame["clean_sv_score"] >= sv_threshold),
        "adv_pair_far": _safe_rate(frame["adv_sv_score"] >= sv_threshold),
        "asr_sv_pair": _safe_rate(eligible["sv_success"]),
        "asr_sr_pair": _safe_rate(eligible["sr_success"]),
        "asr_joint_pair": _safe_rate(eligible["joint_success"]),
        "asr_union_pair": _safe_rate(eligible["union_success"]),
        "mean_clean_sv_score": float(frame["clean_sv_score"].mean()),
        "mean_adv_sv_score": float(frame["adv_sv_score"].mean()),
    }


def _summarize_queries(
    sample_ids,
    source_speakers,
    target_speakers,
    clean_scores,
    adv_scores,
    clean_p_sensitive,
    adv_p_sensitive,
    sv_threshold,
    sr_threshold,
):
    clean_best = clean_scores.max(axis=1)
    adv_best = adv_scores.max(axis=1)
    adv_target_index = adv_scores.argmax(axis=1)
    eligible = (clean_best < sv_threshold) & (clean_p_sensitive < sr_threshold)
    sv_success = eligible & (adv_best >= sv_threshold)
    sr_success = eligible & (adv_p_sensitive >= sr_threshold)
    joint_success = sv_success & sr_success
    union_success = sv_success | sr_success

    frame = pd.DataFrame(
        {
            "query_sample_id": sample_ids,
            "query_speaker": source_speakers,
            "clean_best_sv_score": clean_best,
            "adv_best_sv_score": adv_best,
            "adv_best_target_index": adv_target_index,
            "adv_best_target_speaker": [
                target_speakers[index] for index in adv_target_index
            ],
            "clean_p_sens": clean_p_sensitive,
            "adv_p_sens": adv_p_sensitive,
            "eligible_any10": eligible,
            "sv_success_any10": sv_success,
            "sr_success": sr_success,
            "joint_success_any10": joint_success,
            "union_success_any10": union_success,
        }
    )
    eligible_frame = frame[frame["eligible_any10"]]
    summary = {
        "num_queries": int(len(frame)),
        "eligible_queries": int(len(eligible_frame)),
        "eligibility_retention_rate": float(
            len(eligible_frame) / max(1, len(frame))
        ),
        "clean_any10_acceptance_rate": _safe_rate(clean_best >= sv_threshold),
        "adv_any10_acceptance_rate": _safe_rate(adv_best >= sv_threshold),
        "clean_sensitive_rate": _safe_rate(clean_p_sensitive >= sr_threshold),
        "adv_sensitive_rate": _safe_rate(adv_p_sensitive >= sr_threshold),
        "asr_sv_any10": _safe_rate(eligible_frame["sv_success_any10"]),
        "asr_sr": _safe_rate(eligible_frame["sr_success"]),
        "asr_joint_any10": _safe_rate(eligible_frame["joint_success_any10"]),
        "asr_union_any10": _safe_rate(eligible_frame["union_success_any10"]),
        "mean_clean_best_score": float(clean_best.mean()),
        "mean_adv_best_score": float(adv_best.mean()),
    }
    return frame, summary


def _genuine_diagnostics(
    val_rows,
    val_outputs,
    gallery,
    sv_threshold,
):
    speaker_to_index = {
        speaker_id: index for index, speaker_id in enumerate(gallery.speaker_ids)
    }
    sample_ids = []
    speaker_ids = []
    target_indices = []
    enrollment_ids = set(gallery.sample_ids)
    for index, row in enumerate(val_rows):
        sample_id = "fsc-{}".format(index)
        speaker_id = str(row["speakerId"]).strip()
        if speaker_id not in speaker_to_index or sample_id in enrollment_ids:
            continue
        sample_ids.append(sample_id)
        speaker_ids.append(speaker_id)
        target_indices.append(speaker_to_index[speaker_id])

    clean_embeddings = _stack_output_embeddings(val_outputs, sample_ids, "clean_sv")
    adv_embeddings = _stack_output_embeddings(val_outputs, sample_ids, "adv_sv")
    gallery_embeddings = gallery.embeddings.detach().cpu()
    clean_matrix = scaled_cosine_scores(clean_embeddings, gallery_embeddings)
    adv_matrix = scaled_cosine_scores(adv_embeddings, gallery_embeddings)
    row_indices = torch.arange(len(sample_ids))
    target_tensor = torch.tensor(target_indices, dtype=torch.long)
    clean_scores = clean_matrix[row_indices, target_tensor].numpy()
    adv_scores = adv_matrix[row_indices, target_tensor].numpy()
    clean_accepted = clean_scores >= sv_threshold
    adv_rejected = adv_scores < sv_threshold

    frame = pd.DataFrame(
        {
            "query_sample_id": sample_ids,
            "query_speaker": speaker_ids,
            "clean_sv_score": clean_scores,
            "adv_sv_score": adv_scores,
            "clean_accepted": clean_accepted,
            "adv_rejected": adv_rejected,
            "attack_induced_rejection": clean_accepted & adv_rejected,
        }
    )
    summary = {
        "num_genuine_trials": int(len(frame)),
        "clean_frr": _safe_rate(~clean_accepted),
        "adv_frr": _safe_rate(adv_rejected),
        "clean_accepted_n": int(clean_accepted.sum()),
        "conditional_genuine_rejection_rate": _safe_rate(
            adv_rejected[clean_accepted]
        ),
        "note": "Auxiliary diagnostic; excluded from Any-10 attack success.",
    }
    return frame, summary


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    attack_cfg.resolve_paths(project_root)
    requested_device = str(_cfg_get(cfg, "device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    fsc_root = Path(
        str(_cfg_get(cfg, "fsc_root", project_root / "fluent_speech_commands_dataset"))
    )
    uap_path = Path(
        str(
            _cfg_get(
                cfg,
                "uap_path",
                project_root / "perturbations" / "original" / "janus_fsc_any10_3s.pt",
            )
        )
    )
    output_dir = Path(
        str(
            _cfg_get(
                cfg,
                "output_dir",
                project_root / "results" / "main" / "fsc_any10_evaluation",
            )
        )
    )
    backbone = Path(
        str(_cfg_get(cfg, "backbone", project_root / "model" / "wav2vec2-base-960h"))
    )
    sv_ckpt = Path(str(_cfg_get(cfg, "sv_ckpt", attack_cfg.ckpt_path)))
    intent_head_path = Path(
        str(_cfg_get(cfg, "intent_head_path", attack_cfg.intent_head_path))
    )
    threshold_path = Path(
        str(
            _cfg_get(
                cfg,
                "sv_threshold_path",
                project_root
                / "results"
                / "joint"
                / "evaluation"
                / "fsc_joint_thresholds.json",
            )
        )
    )
    target_count = int(_cfg_get(cfg, "target_count", 10))
    enrollment_k = int(_cfg_get(cfg, "enrollment_k", 3))
    sr_threshold = float(
        _cfg_get(cfg, "sr_threshold", attack_cfg.intent_decision_threshold)
    )
    target_snr_db = _optional_float(_cfg_get(cfg, "target_snr_db", None))
    epsilon = _optional_float(_cfg_get(cfg, "epsilon", None))
    if target_snr_db is not None and epsilon is not None:
        raise ValueError("target_snr_db and epsilon are mutually exclusive")
    sample_rate = int(_cfg_get(cfg, "sample_rate", attack_cfg.sample_rate))
    async_offset_seconds = float(_cfg_get(cfg, "async_offset_seconds", 0.0))
    joint_trials_csv = _cfg_get(cfg, "joint_trials_csv", None)
    max_trials = _cfg_get(cfg, "max_trials", None)
    max_trials = None if max_trials is None else int(max_trials)
    intent_eval_max_batches = _cfg_get(
        cfg, "intent_eval_max_batches", attack_cfg.intent_eval_max_batches
    )
    if intent_eval_max_batches is not None:
        intent_eval_max_batches = int(intent_eval_max_batches)

    for required in (fsc_root, uap_path, backbone, sv_ckpt, intent_head_path):
        if not required.exists():
            raise FileNotFoundError("Missing Any-10 evaluation asset: {}".format(required))
    configured_sv_threshold = _optional_float(_cfg_get(cfg, "sv_threshold", None))
    if configured_sv_threshold is None:
        sv_threshold, validation_eer = _load_sv_threshold(
            threshold_path, attack_cfg.tau_eval
        )
    else:
        sv_threshold, validation_eer = configured_sv_threshold, None
    uap_delta = load_uap(uap_path, device)

    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    sr_cfg.data.module.dataset_folder = str(fsc_root)
    sr_cfg.data.module.train_csv = str(fsc_root / "data" / "train_data.csv")
    sr_cfg.data.module.val_csv = str(fsc_root / "data" / "valid_data.csv")
    sr_cfg.data.module.test_csv = str(fsc_root / "data" / "test_data.csv")
    sr_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sr_cfg.tokenizer.tokenizer_huggingface_id = str(backbone)
    sr_dm = construct_data_module(sr_cfg)
    train_dataloader = sr_dm.train_dataloader()
    val_dataloader = sr_dm.val_dataloader()
    test_dataloader = sr_dm.test_dataloader()

    sensitive_labels = [
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    sensitive_set = set(sensitive_labels)
    if not sensitive_labels:
        raise ValueError("No configured sensitive intents exist in FSC.")

    evaluator = instantiate(cfg.evaluator)
    train_speakers = len(
        {str(row["speakerId"]) for row in sr_dm.train_ds.rows}
    )
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sv_cfg.load_network_from_checkpoint = str(sv_ckpt)
    sv_model = construct_module(
        sv_cfg,
        evaluator,
        FSCVerificationMetadata(num_speakers=train_speakers),
        load_optim=False,
    ).to(device)
    sv_model.eval()
    for parameter in sv_model.parameters():
        parameter.requires_grad = False

    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    sr_model.eval()
    for parameter in sr_model.parameters():
        parameter.requires_grad = False
    intent_head = initialize_intent_head(
        sr_model=sr_model,
        sr_train_dataloader=train_dataloader,
        sr_eval_dataloaders=val_dataloader,
        sensitive_intent_labels=sensitive_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        warmup_steps=0,
        device=device,
        checkpoint_path=intent_head_path,
        lr=attack_cfg.intent_head_lr,
        sensitive_class_weight=attack_cfg.sensitive_class_weight,
        eval_interval_steps=attack_cfg.intent_eval_interval_steps,
        eval_max_batches=intent_eval_max_batches,
        decision_threshold=sr_threshold,
        num_intents=2,
    )

    gallery = build_fsc_enrollment_gallery(
        victim_model=sv_model,
        dataloader=val_dataloader,
        enrollment_k=enrollment_k,
        num_speakers=target_count,
        device=device,
    )
    print("[*] Fixed Any-{} gallery: {}".format(target_count, gallery.speaker_ids))

    val_outputs, val_distortion = collect_uap_outputs(
        val_dataloader,
        sv_model,
        sr_model,
        intent_head,
        uap_delta,
        device,
        target_snr_db,
        sample_rate,
        async_offset_seconds,
        epsilon,
        compute_sr=False,
        description="FSC Any-10 validation inference",
    )
    test_outputs, test_distortion = collect_uap_outputs(
        test_dataloader,
        sv_model,
        sr_model,
        intent_head,
        uap_delta,
        device,
        target_snr_db,
        sample_rate,
        async_offset_seconds,
        epsilon,
        compute_sr=True,
        description="FSC Any-10 test inference",
    )

    sample_ids = []
    source_speakers = []
    transcriptions = []
    sample_paths = []
    for index, row in enumerate(sr_dm.test_ds.rows):
        if _intent_label(row) in sensitive_set:
            continue
        sample_ids.append("fsc-{}".format(index))
        source_speakers.append(str(row["speakerId"]).strip())
        transcriptions.append(str(row["transcription"]).strip())
        sample_paths.append(str(row["path"]).replace("\\", "/"))

    if joint_trials_csv is not None:
        trial_path = Path(str(joint_trials_csv))
        if not trial_path.is_file():
            raise FileNotFoundError("Missing joint trial CSV: {}".format(trial_path))
        trial_frame = pd.read_csv(trial_path)
        if "enroll_speaker" in trial_frame.columns:
            csv_targets = set(trial_frame["enroll_speaker"].astype(str))
            gallery_targets = set(str(value) for value in gallery.speaker_ids)
            if csv_targets != gallery_targets:
                raise ValueError(
                    "joint_trials_csv enrollment speakers do not match the fixed "
                    "gallery: csv={}, gallery={}".format(
                        sorted(csv_targets), sorted(gallery_targets)
                    )
                )
        if "query_sample_id" in trial_frame.columns:
            allowed_ids = set(trial_frame["query_sample_id"].astype(str))
        elif "test_sample_id" in trial_frame.columns:
            allowed_ids = set(trial_frame["test_sample_id"].astype(str))
        elif "query_path" in trial_frame.columns or "test_path" in trial_frame.columns:
            path_column = (
                "query_path" if "query_path" in trial_frame.columns else "test_path"
            )
            allowed_paths = set(
                trial_frame[path_column]
                .astype(str)
                .str.replace("\\", "/", regex=False)
            )
            allowed_ids = {
                sample_id
                for sample_id, sample_path in zip(sample_ids, sample_paths)
                if sample_path in allowed_paths
            }
        else:
            raise ValueError(
                "joint_trials_csv must contain a sample ID or query/test path column"
            )
        selected = [index for index, value in enumerate(sample_ids) if value in allowed_ids]
        sample_ids = [sample_ids[index] for index in selected]
        source_speakers = [source_speakers[index] for index in selected]
        transcriptions = [transcriptions[index] for index in selected]
        sample_paths = [sample_paths[index] for index in selected]

    if max_trials is not None:
        sample_ids = sample_ids[:max_trials]
        source_speakers = source_speakers[:max_trials]
        transcriptions = transcriptions[:max_trials]
        sample_paths = sample_paths[:max_trials]
    if not sample_ids:
        raise ValueError("No benign FSC queries remain after trial filtering")
    overlap = set(source_speakers).intersection(gallery.speaker_ids)
    if overlap:
        raise ValueError("Test sources overlap Any-10 targets: {}".format(sorted(overlap)))

    clean_embeddings = _stack_output_embeddings(
        test_outputs, sample_ids, "clean_sv"
    )
    adv_embeddings = _stack_output_embeddings(test_outputs, sample_ids, "adv_sv")
    gallery_embeddings = gallery.embeddings.detach().cpu()
    clean_scores = scaled_cosine_scores(
        clean_embeddings, gallery_embeddings
    ).numpy()
    adv_scores = scaled_cosine_scores(adv_embeddings, gallery_embeddings).numpy()
    clean_p_sensitive = np.asarray(
        [test_outputs[sample_id]["clean_p_sens"] for sample_id in sample_ids],
        dtype=np.float64,
    )
    adv_p_sensitive = np.asarray(
        [test_outputs[sample_id]["adv_p_sens"] for sample_id in sample_ids],
        dtype=np.float64,
    )

    pair_frame = _make_pair_frame(
        sample_ids,
        source_speakers,
        transcriptions,
        gallery.speaker_ids,
        clean_scores,
        adv_scores,
        clean_p_sensitive,
        adv_p_sensitive,
        sv_threshold,
        sr_threshold,
    )
    pair_summary = _summarize_pairs(pair_frame, sv_threshold)
    per_target = []
    for target_speaker, target_frame in pair_frame.groupby(
        "target_speaker", sort=True
    ):
        row = _summarize_pairs(target_frame, sv_threshold)
        row["target_speaker"] = str(target_speaker)
        per_target.append(row)

    query_frame, any10_summary = _summarize_queries(
        sample_ids,
        source_speakers,
        gallery.speaker_ids,
        clean_scores,
        adv_scores,
        clean_p_sensitive,
        adv_p_sensitive,
        sv_threshold,
        sr_threshold,
    )
    genuine_frame, genuine_summary = _genuine_diagnostics(
        sr_dm.val_ds.rows, val_outputs, gallery, sv_threshold
    )

    per_target_frame = pd.DataFrame(per_target)
    macro_target = {}
    for metric in (
        "asr_sv_pair",
        "asr_sr_pair",
        "asr_joint_pair",
        "asr_union_pair",
        "adv_pair_far",
    ):
        macro_target[metric + "_mean"] = float(per_target_frame[metric].mean())
        macro_target[metric + "_std"] = float(
            per_target_frame[metric].std(ddof=0)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(output_dir / "fsc_any10_pair_results.csv", index=False)
    query_frame.to_csv(output_dir / "fsc_any10_query_results.csv", index=False)
    per_target_frame.to_csv(output_dir / "fsc_any10_per_target.csv", index=False)
    genuine_frame.to_csv(output_dir / "fsc_any10_genuine_diagnostics.csv", index=False)

    results = {
        "method": str(_cfg_get(cfg, "method", "JANUS-FSC-Any10")),
        "protocol": "fixed_validation_gallery_test_source_any_enrollment",
        "uap_path": str(uap_path),
        "parameters": {
            "target_snr_db": target_snr_db,
            "epsilon": epsilon,
            "sample_rate": sample_rate,
            "async_offset_seconds": async_offset_seconds,
            "joint_trials_csv": None
            if joint_trials_csv is None
            else str(joint_trials_csv),
            "max_trials": max_trials,
        },
        "target_speakers": list(gallery.speaker_ids),
        "thresholds": {
            "sv_threshold": sv_threshold,
            "validation_eer": validation_eer,
            "sv_score_space": "scaled_cosine_[0,1]",
            "sr_threshold": sr_threshold,
            "enrollment_k": enrollment_k,
        },
        "distortion": {
            "validation": _summarize_distortion(val_distortion),
            "test": _summarize_distortion(test_distortion),
        },
        "pair_level": pair_summary,
        "query_level_any10": any10_summary,
        "macro_over_targets": macro_target,
        "per_target": per_target,
        "genuine_rejection_diagnostics": genuine_summary,
    }
    with (output_dir / "fsc_any10_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)

    print("[*] Any-10 pair-level: {}".format(pair_summary))
    print("[*] Any-10 query-level: {}".format(any10_summary))
    print("[*] Any-10 macro targets: {}".format(macro_target))
    print("[+] Saved FSC Any-10 evaluation to {}".format(output_dir))


if __name__ == "__main__":
    main_evaluation()
