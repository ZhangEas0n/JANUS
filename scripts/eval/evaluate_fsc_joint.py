from __future__ import print_function

import json
import os
from collections import Counter
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

from btuap.common import apply_uap, normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.sr import (
    extract_raw_audio_batch,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from src.data.modules.speaker.speaker_data_module import SpeakerLightningDataModule
from src.eval_metrics import calculate_eer
from src.evaluation.speaker.speaker_recognition_evaluator import EmbeddingSample
from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)
from src.main import construct_data_module, construct_module


load_dotenv()
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)


class FSCVerificationMetadata(SpeakerLightningDataModule):
    """Minimal FSC speaker metadata required when restoring the SV model."""

    def __init__(self, num_speakers):
        super().__init__()
        self._num_speakers = int(num_speakers)

    @property
    def num_speakers(self):
        return self._num_speakers

    @property
    def val_pairs(self):
        return []

    @property
    def test_pairs(self):
        return []


def _cfg_get(cfg, key, default):
    if "fsc_joint_eval" in cfg and key in cfg.fsc_joint_eval:
        return cfg.fsc_joint_eval[key]
    return default


def load_uap(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing UAP tensor: {}".format(path))
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict):
        for key in ("uap_delta", "delta", "perturbation", "uap"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError("Unsupported UAP checkpoint content: {}".format(type(payload)))
    uap = payload.detach().float().to(device)
    if uap.dim() == 1:
        uap = uap.unsqueeze(0)
    if uap.dim() == 3 and uap.shape[1] == 1:
        uap = uap[:, 0, :]
    if uap.dim() != 2 or uap.shape[0] != 1:
        raise ValueError("Expected UAP shape [1, samples], got {}".format(tuple(uap.shape)))
    print(
        "[*] Loaded UAP: path={}, shape={}, linf={:.6f}, rms={:.6f}".format(
            path,
            tuple(uap.shape),
            float(uap.abs().max()),
            float(uap.pow(2).mean().sqrt()),
        )
    )
    return uap


def load_protocol(protocol_dir):
    protocol_dir = Path(protocol_dir)
    paths = {
        "gallery": protocol_dir / "fsc_joint_enrollment_gallery.csv",
        "validation": protocol_dir / "fsc_joint_validation_trials.csv",
        "test": protocol_dir / "fsc_joint_trials_all.csv",
        "genuine": protocol_dir / "fsc_joint_test_genuine_trials.csv",
        "stats": protocol_dir / "fsc_joint_protocol_stats.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing FSC joint protocol files: {}. Run scripts/data/build_fsc_joint_protocol.py first.".format(
                missing
            )
        )

    gallery = pd.read_csv(paths["gallery"], dtype={"speakerId": str})
    validation = pd.read_csv(
        paths["validation"], dtype={"query_speaker": str, "enroll_speaker": str}
    )
    test = pd.read_csv(
        paths["test"], dtype={"query_speaker": str, "enroll_speaker": str}
    )
    genuine = pd.read_csv(
        paths["genuine"], dtype={"query_speaker": str, "enroll_speaker": str}
    )
    with paths["stats"].open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    if (test["query_speaker"] == test["enroll_speaker"]).any():
        raise ValueError("Test impostor protocol contains same-speaker trials.")
    if (genuine["query_speaker"] != genuine["enroll_speaker"]).any():
        raise ValueError("Test genuine protocol contains different-speaker trials.")
    if (test["is_sensitive_label"] != 0).any():
        raise ValueError("Joint test protocol must contain ground-truth benign queries only.")

    print(
        "[*] Loaded FSC protocol: gallery={}, validation={}, test={}, genuine={}".format(
            len(gallery), len(validation), len(test), len(genuine)
        )
    )
    return gallery, validation, test, genuine, stats


def collect_split_outputs(
    dataloader,
    sv_model,
    sr_model,
    intent_head,
    uap_delta,
    device,
    compute_sr,
    description,
):
    outputs = {}
    sv_model.eval()
    sr_model.eval()
    intent_head.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            batch = batch.to(device)
            raw_audio = extract_raw_audio_batch(batch, device)
            clean_input = normalize_batch_like_pipeline(
                raw_audio, input_lengths=batch.input_lengths
            )
            adv_raw = apply_uap(raw_audio, uap_delta)
            adv_input = normalize_batch_like_pipeline(
                adv_raw, input_lengths=batch.input_lengths
            )

            clean_sv = sv_model.compute_speaker_embedding(clean_input)
            adv_sv = sv_model.compute_speaker_embedding(adv_input)
            if clean_sv.dim() == 1:
                clean_sv = clean_sv.unsqueeze(0)
                adv_sv = adv_sv.unsqueeze(0)
            clean_sv = F.normalize(clean_sv, p=2, dim=-1)
            adv_sv = F.normalize(adv_sv, p=2, dim=-1)

            clean_probabilities = [None] * len(batch.keys)
            adv_probabilities = [None] * len(batch.keys)
            if compute_sr:
                clean_sr, clean_lengths = sr_model.compute_speech_embedding(
                    clean_input, batch.input_lengths
                )
                adv_sr, adv_lengths = sr_model.compute_speech_embedding(
                    adv_input, batch.input_lengths
                )
                clean_logits = intent_head(clean_sr, clean_lengths)
                adv_logits = intent_head(adv_sr, adv_lengths)
                clean_probabilities = (
                    torch.softmax(clean_logits, dim=-1)[:, 1].detach().cpu().tolist()
                )
                adv_probabilities = (
                    torch.softmax(adv_logits, dim=-1)[:, 1].detach().cpu().tolist()
                )

            for index, sample_id in enumerate(batch.keys):
                outputs[str(sample_id)] = {
                    "clean_sv": clean_sv[index].detach().cpu(),
                    "adv_sv": adv_sv[index].detach().cpu(),
                    "clean_p_sens": clean_probabilities[index],
                    "adv_p_sens": adv_probabilities[index],
                }
    return outputs


def make_enrollment_embeddings(gallery, split_outputs):
    embeddings = {}
    for (split, speaker_id), rows in gallery.groupby(["split", "speakerId"]):
        sample_embeddings = []
        for sample_id in rows.sort_values("enrollment_rank")["sample_id"]:
            sample_id = str(sample_id)
            if sample_id not in split_outputs[split]:
                raise KeyError("Enrollment sample not found: {}/{}".format(split, sample_id))
            sample_embeddings.append(split_outputs[split][sample_id]["clean_sv"])
        stacked = F.normalize(torch.stack(sample_embeddings), p=2, dim=-1)
        mean_embedding = F.normalize(stacked.mean(dim=0), p=2, dim=-1)
        embeddings[(str(split), str(speaker_id))] = mean_embedding
    return embeddings


def score_trials(frame, split_outputs, enrollment_embeddings, evaluator, use_adv):
    scores = []
    batch_size = 1024
    prediction_pairs = []

    def flush():
        if not prediction_pairs:
            return
        raw = evaluator._compute_prediction_scores(prediction_pairs)
        scores.extend(np.clip((np.asarray(raw) + 1.0) / 2.0, 0.0, 1.0).tolist())
        prediction_pairs[:] = []

    embedding_key = "adv_sv" if use_adv else "clean_sv"
    for row in frame.itertuples(index=False):
        split = str(row.split)
        sample_id = str(row.query_sample_id)
        enrollment_key = (split, str(row.enroll_speaker))
        if sample_id not in split_outputs[split]:
            raise KeyError("Query sample not found: {}/{}".format(split, sample_id))
        if enrollment_key not in enrollment_embeddings:
            raise KeyError("Enrollment speaker not found: {}".format(enrollment_key))
        prediction_pairs.append(
            (
                EmbeddingSample(
                    "enroll-{}-{}".format(split, row.enroll_speaker),
                    enrollment_embeddings[enrollment_key],
                ),
                EmbeddingSample(
                    "query-{}-{}".format(split, sample_id),
                    split_outputs[split][sample_id][embedding_key],
                ),
            )
        )
        if len(prediction_pairs) >= batch_size:
            flush()
    flush()
    return np.asarray(scores, dtype=np.float64)


def add_sensitive_probabilities(frame, split_outputs):
    frame = frame.copy()
    frame["clean_p_sens"] = [
        split_outputs[str(row.split)][str(row.query_sample_id)]["clean_p_sens"]
        for row in frame.itertuples(index=False)
    ]
    frame["adv_p_sens"] = [
        split_outputs[str(row.split)][str(row.query_sample_id)]["adv_p_sens"]
        for row in frame.itertuples(index=False)
    ]
    return frame


def safe_rate(mask):
    return float(np.asarray(mask, dtype=np.float64).mean()) if len(mask) else 0.0


def summarize_seed(frame, sv_threshold, sr_threshold):
    eligible = frame["eligible_joint"].astype(bool)
    eligible_frame = frame[eligible]
    union_success = eligible_frame["sv_success"] | eligible_frame["sr_success"]
    return {
        "seed": int(frame["seed"].iloc[0]),
        "raw_n": int(len(frame)),
        "eligible_n": int(eligible.sum()),
        "eligibility_retention_rate": safe_rate(eligible),
        "asr_sv_joint": safe_rate(eligible_frame["sv_success"]),
        "asr_sr_joint": safe_rate(eligible_frame["sr_success"]),
        "asr_joint": safe_rate(eligible_frame["joint_success"]),
        "asr_union": safe_rate(union_success),
        "clean_far": safe_rate(frame["clean_sv_score"] >= sv_threshold),
        "adv_far": safe_rate(frame["adv_sv_score"] >= sv_threshold),
        "clean_sensitive_rate": safe_rate(frame["clean_p_sens"] >= sr_threshold),
        "adv_sensitive_rate": safe_rate(frame["adv_p_sens"] >= sr_threshold),
        "mean_clean_sv_score": float(frame["clean_sv_score"].mean()),
        "mean_adv_sv_score": float(frame["adv_sv_score"].mean()),
        "mean_clean_p_sens": float(frame["clean_p_sens"].mean()),
        "mean_adv_p_sens": float(frame["adv_p_sens"].mean()),
    }


def summarize_genuine(frame, sv_threshold):
    clean_accepted = frame["clean_sv_score"] >= sv_threshold
    adv_rejected = frame["adv_sv_score"] < sv_threshold
    eligible_n = int(clean_accepted.sum())
    attack_induced = clean_accepted & adv_rejected
    return {
        "num_genuine_trials": int(len(frame)),
        "clean_rejected_n": int((~clean_accepted).sum()),
        "clean_frr": safe_rate(~clean_accepted),
        "adv_rejected_n": int(adv_rejected.sum()),
        "adv_frr": safe_rate(adv_rejected),
        "clean_accepted_n": eligible_n,
        "attack_induced_rejection_n": int(attack_induced.sum()),
        "conditional_genuine_rejection_rate": safe_rate(attack_induced[clean_accepted]),
        "note": "Auxiliary diagnostic only; excluded from ASR-SV-joint and ASR-Joint.",
    }


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fsc_root = Path(
        str(
            _cfg_get(
                cfg,
                "fsc_root",
                project_root / "fluent_speech_commands_dataset",
            )
        )
    )
    protocol_dir = Path(
        str(
            _cfg_get(
                cfg,
                "protocol_dir",
                fsc_root / "data_processed" / "fsc_joint_protocol",
            )
        )
    )
    uap_path = Path(
        str(
            _cfg_get(
                cfg,
                "uap_path",
                attack_cfg.evaluation_uap_path or attack_cfg.output_path,
            )
        )
    )
    backbone = Path(
        str(
            _cfg_get(
                cfg,
                "backbone",
                project_root / "model" / "wav2vec2-base-960h",
            )
        )
    )
    sv_ckpt = Path(str(_cfg_get(cfg, "sv_ckpt", attack_cfg.ckpt_path)))
    intent_head_path = Path(
        str(_cfg_get(cfg, "intent_head_path", attack_cfg.intent_head_path))
    )
    output_dir = Path(
        str(
            _cfg_get(
                cfg,
                "output_dir",
                project_root / "results" / "joint" / "evaluation",
            )
        )
    )
    sr_threshold = float(
        _cfg_get(cfg, "sr_threshold", attack_cfg.intent_decision_threshold)
    )
    for required in (fsc_root, protocol_dir, uap_path, backbone, sv_ckpt, intent_head_path):
        if not required.exists():
            raise FileNotFoundError("Missing FSC joint evaluation asset: {}".format(required))

    gallery, validation, test, genuine, protocol_stats = load_protocol(protocol_dir)
    uap_delta = load_uap(uap_path, device)
    print(
        "[*] FSC joint evaluation: device={}, sr_threshold={}, output={}".format(
            device, sr_threshold, output_dir
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
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not sensitive_labels:
        raise ValueError("No configured sensitive intents exist in FSC.")

    evaluator = instantiate(cfg.evaluator)
    train_speakers = len(
        {str(row["speakerId"]) for row in sr_dm.train_ds.rows}
    )
    speaker_metadata = FSCVerificationMetadata(num_speakers=train_speakers)
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sv_cfg.load_network_from_checkpoint = str(sv_ckpt)
    sv_model = construct_module(
        sv_cfg, evaluator, speaker_metadata, load_optim=False
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
        warmup_steps=attack_cfg.intent_warmup_steps,
        device=device,
        checkpoint_path=intent_head_path,
        lr=attack_cfg.intent_head_lr,
        sensitive_class_weight=attack_cfg.sensitive_class_weight,
        eval_interval_steps=attack_cfg.intent_eval_interval_steps,
        eval_max_batches=attack_cfg.intent_eval_max_batches,
        decision_threshold=sr_threshold,
        num_intents=2,
    )

    split_outputs = {
        "validation": collect_split_outputs(
            val_dataloader,
            sv_model,
            sr_model,
            intent_head,
            uap_delta,
            device,
            compute_sr=False,
            description="FSC validation inference",
        ),
        "test": collect_split_outputs(
            test_dataloader,
            sv_model,
            sr_model,
            intent_head,
            uap_delta,
            device,
            compute_sr=True,
            description="FSC test joint inference",
        ),
    }
    enrollment_embeddings = make_enrollment_embeddings(gallery, split_outputs)

    validation = validation.copy()
    validation["clean_sv_score"] = score_trials(
        validation, split_outputs, enrollment_embeddings, evaluator, use_adv=False
    )
    validation_labels = (validation["trial_type"] == "genuine").astype(int).tolist()
    val_eer, sv_threshold = calculate_eer(
        validation_labels, validation["clean_sv_score"].tolist(), pos_label=1
    )
    sv_threshold = float(sv_threshold)
    print(
        "[*] Frozen validation SV threshold: EER={:.6f}, threshold={:.6f}".format(
            float(val_eer), sv_threshold
        )
    )

    test = add_sensitive_probabilities(test, split_outputs)
    test["clean_sv_score"] = score_trials(
        test, split_outputs, enrollment_embeddings, evaluator, use_adv=False
    )
    test["adv_sv_score"] = score_trials(
        test, split_outputs, enrollment_embeddings, evaluator, use_adv=True
    )
    test["eligible_joint"] = (
        (test["is_sensitive_label"] == 0)
        & (test["clean_sv_score"] < sv_threshold)
        & (test["clean_p_sens"] < sr_threshold)
    )
    test["sv_success"] = test["eligible_joint"] & (
        test["adv_sv_score"] >= sv_threshold
    )
    test["sr_success"] = test["eligible_joint"] & (
        test["adv_p_sens"] >= sr_threshold
    )
    test["joint_success"] = test["sv_success"] & test["sr_success"]

    def risk_type(row):
        if not row.eligible_joint:
            return "ineligible"
        if row.sv_success and row.sr_success:
            return "both"
        if row.sv_success:
            return "sv_only"
        if row.sr_success:
            return "sr_only"
        return "neither"

    test["risk_type"] = [risk_type(row) for row in test.itertuples(index=False)]
    summaries = [
        summarize_seed(seed_frame, sv_threshold, sr_threshold)
        for _, seed_frame in test.groupby("seed", sort=True)
    ]
    summary_frame = pd.DataFrame(summaries)
    metric_columns = [
        "eligibility_retention_rate",
        "asr_sv_joint",
        "asr_sr_joint",
        "asr_joint",
        "asr_union",
        "clean_far",
        "adv_far",
        "clean_sensitive_rate",
        "adv_sensitive_rate",
        "mean_clean_sv_score",
        "mean_adv_sv_score",
        "mean_clean_p_sens",
        "mean_adv_p_sens",
    ]
    aggregate = {
        "num_seeds": int(len(summary_frame)),
        "raw_n_per_seed": int(summary_frame["raw_n"].iloc[0]),
    }
    for column in metric_columns:
        aggregate[column + "_mean"] = float(summary_frame[column].mean())
        aggregate[column + "_std"] = float(summary_frame[column].std(ddof=0))

    eligible = test[test["eligible_joint"]].copy()
    risk_rows = []
    for seed, seed_frame in eligible.groupby("seed", sort=True):
        counts = Counter(seed_frame["risk_type"].tolist())
        total = len(seed_frame)
        for name in ("neither", "sv_only", "sr_only", "both"):
            risk_rows.append(
                {
                    "seed": int(seed),
                    "risk_type": name,
                    "count": int(counts.get(name, 0)),
                    "ratio": float(counts.get(name, 0) / max(1, total)),
                }
            )
    risk_frame = pd.DataFrame(risk_rows)

    genuine = genuine.copy()
    genuine["clean_sv_score"] = score_trials(
        genuine, split_outputs, enrollment_embeddings, evaluator, use_adv=False
    )
    genuine["adv_sv_score"] = score_trials(
        genuine, split_outputs, enrollment_embeddings, evaluator, use_adv=True
    )
    genuine["clean_accepted"] = genuine["clean_sv_score"] >= sv_threshold
    genuine["adv_rejected"] = genuine["adv_sv_score"] < sv_threshold
    genuine["attack_induced_rejection"] = (
        genuine["clean_accepted"] & genuine["adv_rejected"]
    )
    genuine_summary = summarize_genuine(genuine, sv_threshold)

    if not (
        summary_frame["asr_joint"]
        <= summary_frame[["asr_sv_joint", "asr_sr_joint"]].min(axis=1) + 1e-8
    ).all():
        raise AssertionError("ASR-Joint consistency check failed.")
    if not (
        np.abs(
            summary_frame["asr_union"]
            - summary_frame["asr_sv_joint"]
            - summary_frame["asr_sr_joint"]
            + summary_frame["asr_joint"]
        )
        <= 1e-8
    ).all():
        raise AssertionError("ASR-Union consistency check failed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "fsc_joint_validation_trials.csv", index=False)
    test.to_csv(output_dir / "fsc_joint_eval_results.csv", index=False)
    eligible.to_csv(output_dir / "fsc_joint_trials_eligible.csv", index=False)
    summary_frame.to_csv(output_dir / "fsc_joint_summary.csv", index=False)
    risk_frame.to_csv(output_dir / "fsc_joint_risk_composition.csv", index=False)
    genuine.to_csv(output_dir / "fsc_joint_genuine_diagnostics.csv", index=False)
    pd.DataFrame([genuine_summary]).to_csv(
        output_dir / "fsc_joint_genuine_summary.csv", index=False
    )

    thresholds = {
        "sv_threshold": sv_threshold,
        "validation_eer": float(val_eer),
        "sv_score_space": "scaled_cosine_[0,1]",
        "sr_threshold": sr_threshold,
        "enrollment_k": int(protocol_stats["enrollment_k"]),
        "enrollment_seed": int(protocol_stats["enrollment_seed"]),
    }
    with (output_dir / "fsc_joint_thresholds.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(thresholds, handle, indent=2)

    results = {
        "method": str(_cfg_get(cfg, "method", "JANUS")),
        "uap_path": str(uap_path),
        "protocol_dir": str(protocol_dir),
        "thresholds": thresholds,
        "per_seed": summaries,
        "aggregate": aggregate,
        "genuine_rejection_diagnostics": genuine_summary,
    }
    with (output_dir / "fsc_joint_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)

    print("[*] FSC same-input joint results")
    for row in summaries:
        print(
            "    seed={seed}: eligible={eligible_n}/{raw_n}, "
            "ASR-SV={asr_sv_joint:.4f}, ASR-SR={asr_sr_joint:.4f}, "
            "ASR-Joint={asr_joint:.4f}, ASR-Union={asr_union:.4f}".format(**row)
        )
    print(
        "[*] Aggregate: ASR-SV={:.4f}+/-{:.4f}, ASR-SR={:.4f}+/-{:.4f}, "
        "ASR-Joint={:.4f}+/-{:.4f}, ASR-Union={:.4f}+/-{:.4f}".format(
            aggregate["asr_sv_joint_mean"],
            aggregate["asr_sv_joint_std"],
            aggregate["asr_sr_joint_mean"],
            aggregate["asr_sr_joint_std"],
            aggregate["asr_joint_mean"],
            aggregate["asr_joint_std"],
            aggregate["asr_union_mean"],
            aggregate["asr_union_std"],
        )
    )
    print(
        "[*] Auxiliary genuine rejection: clean_FRR={:.4f}, adv_FRR={:.4f}, "
        "conditional_rejection={:.4f}".format(
            genuine_summary["clean_frr"],
            genuine_summary["adv_frr"],
            genuine_summary["conditional_genuine_rejection_rate"],
        )
    )
    print("[+] Saved FSC joint evaluation to {}".format(output_dir))


if __name__ == "__main__":
    main_evaluation()
