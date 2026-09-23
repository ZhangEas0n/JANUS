"""Evaluate Kenansville under the fixed FSC Any-N SV-SR protocol."""

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
from tqdm import tqdm

from btuap.common import normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.fsc_sv import build_fsc_enrollment_gallery, scaled_cosine_scores
from btuap.kenansville import (
    apply_kenansville_fft,
    perturbation_statistics,
    rescale_to_snr,
)
from btuap.sr import (
    extract_raw_audio_batch,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from evaluate_fsc_any10 import (
    _genuine_diagnostics,
    _intent_label,
    _load_sv_threshold,
    _make_pair_frame,
    _stack_output_embeddings,
    _summarize_pairs,
    _summarize_queries,
)
from evaluate_fsc_joint import FSCVerificationMetadata
from src.main import construct_data_module, construct_module


load_dotenv()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _cfg_get(cfg, key, default):
    if "kenansville_any10" in cfg and key in cfg.kenansville_any10:
        return cfg.kenansville_any10[key]
    return default


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


def collect_kenansville_outputs(
    dataloader,
    sv_model,
    sr_model,
    intent_head,
    device,
    factor,
    target_snr_db,
    compute_sr,
    description,
):
    outputs = {}
    distortion_rows = []
    sv_model.eval()
    sr_model.eval()
    intent_head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            batch = batch.to(device)
            raw_audio = extract_raw_audio_batch(batch, device)
            attacked_raw = apply_kenansville_fft(
                raw_audio,
                input_lengths=batch.input_lengths,
                factor=factor,
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
            clean_sv = F.normalize(
                sv_model.compute_speaker_embedding(clean_input), p=2, dim=-1
            )
            attacked_sv = F.normalize(
                sv_model.compute_speaker_embedding(attacked_input), p=2, dim=-1
            )
            if clean_sv.dim() == 1:
                clean_sv = clean_sv.unsqueeze(0)
                attacked_sv = attacked_sv.unsqueeze(0)

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


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fsc_root = Path(str(_cfg_get(
        cfg, "fsc_root", project_root / "fluent_speech_commands_dataset"
    )))
    output_dir = Path(str(_cfg_get(
        cfg,
        "output_dir",
        project_root / "results" / "main" / "fsc_any10_kenansville_fft",
    )))
    backbone = Path(str(_cfg_get(
        cfg, "backbone", project_root / "model" / "wav2vec2-base-960h"
    )))
    sv_ckpt = Path(str(_cfg_get(cfg, "sv_ckpt", attack_cfg.ckpt_path)))
    intent_head_path = Path(str(_cfg_get(
        cfg, "intent_head_path", attack_cfg.intent_head_path
    )))
    threshold_path = Path(str(_cfg_get(
        cfg,
        "sv_threshold_path",
        project_root / "results" / "joint" / "evaluation"
        / "fsc_joint_thresholds.json",
    )))
    factor = float(_cfg_get(cfg, "factor", 0.1))
    target_snr_db = _optional_float(_cfg_get(cfg, "target_snr_db", None))
    target_count = int(_cfg_get(cfg, "target_count", 10))
    enrollment_k = int(_cfg_get(cfg, "enrollment_k", 3))
    sr_threshold = float(_cfg_get(
        cfg, "sr_threshold", attack_cfg.intent_decision_threshold
    ))

    for required in (fsc_root, backbone, sv_ckpt, intent_head_path):
        if not required.exists():
            raise FileNotFoundError("Missing Kenansville asset: {}".format(required))
    sv_threshold, validation_eer = _load_sv_threshold(
        threshold_path, attack_cfg.tau_eval
    )
    print(
        "[*] Kenansville Any-{}: factor={}, target_snr_db={}, device={}".format(
            target_count, factor, target_snr_db, device
        )
    )

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
        label for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not sensitive_labels:
        raise ValueError("No configured sensitive intents exist in FSC.")
    sensitive_set = set(sensitive_labels)

    evaluator = instantiate(cfg.evaluator)
    train_speakers = len({str(row["speakerId"]) for row in sr_dm.train_ds.rows})
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sv_cfg.load_network_from_checkpoint = str(sv_ckpt)
    sv_model = construct_module(
        sv_cfg,
        evaluator,
        FSCVerificationMetadata(num_speakers=train_speakers),
        load_optim=False,
    ).to(device)
    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    for model in (sv_model, sr_model):
        model.eval()
        for parameter in model.parameters():
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
        eval_max_batches=attack_cfg.intent_eval_max_batches,
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

    val_outputs, val_distortion = collect_kenansville_outputs(
        val_dataloader, sv_model, sr_model, intent_head, device, factor,
        target_snr_db, False, "Kenansville validation inference"
    )
    test_outputs, test_distortion = collect_kenansville_outputs(
        test_dataloader, sv_model, sr_model, intent_head, device, factor,
        target_snr_db, True, "Kenansville test inference"
    )

    sample_ids, source_speakers, transcriptions = [], [], []
    for index, row in enumerate(sr_dm.test_ds.rows):
        if _intent_label(row) in sensitive_set:
            continue
        sample_ids.append("fsc-{}".format(index))
        source_speakers.append(str(row["speakerId"]).strip())
        transcriptions.append(str(row["transcription"]).strip())
    overlap = set(source_speakers).intersection(gallery.speaker_ids)
    if overlap:
        raise ValueError("Test sources overlap Any-N targets: {}".format(sorted(overlap)))

    clean_embeddings = _stack_output_embeddings(test_outputs, sample_ids, "clean_sv")
    attacked_embeddings = _stack_output_embeddings(test_outputs, sample_ids, "adv_sv")
    gallery_embeddings = gallery.embeddings.detach().cpu()
    clean_scores = scaled_cosine_scores(clean_embeddings, gallery_embeddings).numpy()
    attacked_scores = scaled_cosine_scores(
        attacked_embeddings, gallery_embeddings
    ).numpy()
    clean_p_sensitive = np.asarray([
        test_outputs[sample_id]["clean_p_sens"] for sample_id in sample_ids
    ])
    attacked_p_sensitive = np.asarray([
        test_outputs[sample_id]["adv_p_sens"] for sample_id in sample_ids
    ])

    pair_frame = _make_pair_frame(
        sample_ids, source_speakers, transcriptions, gallery.speaker_ids,
        clean_scores, attacked_scores, clean_p_sensitive,
        attacked_p_sensitive, sv_threshold, sr_threshold
    )
    pair_summary = _summarize_pairs(pair_frame, sv_threshold)
    per_target = []
    for target_speaker, target_frame in pair_frame.groupby("target_speaker", sort=True):
        row = _summarize_pairs(target_frame, sv_threshold)
        row["target_speaker"] = str(target_speaker)
        per_target.append(row)
    per_target_frame = pd.DataFrame(per_target)
    macro_target = {}
    for metric in ("asr_sv_pair", "asr_sr_pair", "asr_joint_pair", "adv_pair_far"):
        macro_target[metric + "_mean"] = float(per_target_frame[metric].mean())
        macro_target[metric + "_std"] = float(per_target_frame[metric].std(ddof=0))

    query_frame, query_summary = _summarize_queries(
        sample_ids, source_speakers, gallery.speaker_ids, clean_scores,
        attacked_scores, clean_p_sensitive, attacked_p_sensitive,
        sv_threshold, sr_threshold
    )
    genuine_frame, genuine_summary = _genuine_diagnostics(
        sr_dm.val_ds.rows, val_outputs, gallery, sv_threshold
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(output_dir / "fsc_any10_pair_results.csv", index=False)
    query_frame.to_csv(output_dir / "fsc_any10_query_results.csv", index=False)
    per_target_frame.to_csv(output_dir / "fsc_any10_per_target.csv", index=False)
    genuine_frame.to_csv(output_dir / "fsc_any10_genuine_diagnostics.csv", index=False)
    result = {
        "method": "Kenansville-FFT-FSC-Any{}".format(target_count),
        "protocol": "fixed_validation_gallery_test_source_any_enrollment",
        "parameters": {"factor": factor, "target_snr_db": target_snr_db},
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
        "query_level_any10": query_summary,
        "macro_over_targets": macro_target,
        "per_target": per_target,
        "genuine_rejection_diagnostics": genuine_summary,
    }
    with (output_dir / "fsc_any10_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("[*] Kenansville pair-level: {}".format(pair_summary))
    print("[*] Kenansville query-level: {}".format(query_summary))
    print("[*] Distortion: {}".format(result["distortion"]["test"]))
    print("[+] Saved Kenansville Any-N evaluation to {}".format(output_dir))


if __name__ == "__main__":
    main_evaluation()
