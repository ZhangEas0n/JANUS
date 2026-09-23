import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from btuap.config import BTUAPAttackConfig
from btuap.sr import (
    evaluate_sr_attack,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from btuap.sv import evaluate_sv_attack
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


def _cfg_get(cfg: DictConfig, key: str, default):
    if "wavlm_eval" in cfg and key in cfg.wavlm_eval:
        return cfg.wavlm_eval[key]
    return default


def load_uap(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing UAP tensor: {path}")
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict):
        for key in ("uap_delta", "delta", "perturbation", "uap"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsupported UAP checkpoint content: {type(payload)}")
    uap = payload.detach().float().to(device)
    if uap.dim() == 1:
        uap = uap.unsqueeze(0)
    if uap.dim() == 3 and uap.shape[1] == 1:
        uap = uap[:, 0, :]
    if uap.dim() != 2:
        raise ValueError(f"Expected UAP shape [1, samples], got {tuple(uap.shape)}")
    print(
        "[*] Loaded BTUAP: "
        f"path={path}, shape={tuple(uap.shape)}, "
        f"linf={uap.abs().max().item():.5f}, "
        f"rms={uap.pow(2).mean().sqrt().item():.5f}"
    )
    return uap


def _optional_path(value):
    if value is None:
        return None
    return str(Path(str(value)))


def evaluate_wavlm_sv(
    cfg,
    evaluator,
    wavlm_backbone,
    sv_ckpt,
    uap_delta,
    device,
    max_batches,
):
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.data.pipeline.test_pipeline = []
    sv_cfg.network.wav2vec_hunggingface_id = str(wavlm_backbone)
    sv_cfg.load_network_from_checkpoint = sv_ckpt

    print(
        "[*] Initializing WavLM SV evaluation: "
        f"backbone={wavlm_backbone}, sv_ckpt={sv_ckpt}"
    )
    sv_dm = construct_data_module(sv_cfg)
    sv_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    sv_model.eval()
    for parameter in sv_model.parameters():
        parameter.requires_grad = False

    results = evaluate_sv_attack(
        victim_model=sv_model,
        evaluator=evaluator,
        dataloader=sv_dm.test_dataloader(),
        pairs=sv_dm.test_pairs,
        device=device,
        uap_delta=uap_delta,
        max_batches=max_batches,
    )
    sv_model.to("cpu")
    del sv_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def evaluate_wavlm_sr(
    cfg,
    attack_cfg,
    evaluator,
    wavlm_backbone,
    intent_head_path,
    uap_delta,
    device,
    project_root,
    max_batches,
):
    print(
        "[*] Initializing WavLM SR evaluation: "
        f"backbone={wavlm_backbone}, intent_head={intent_head_path}"
    )
    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    sr_cfg.network.wav2vec_hunggingface_id = str(wavlm_backbone)

    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if local_wav2vec2.exists():
        sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2)
        sr_cfg.network.speech_head_huggingface_id = str(local_wav2vec2)

    sr_dm = construct_data_module(sr_cfg)
    sr_train_dataloader = sr_dm.train_dataloader()
    sr_eval_dataloaders = sr_dm.val_dataloader()
    sr_test_dataloader = sr_dm.test_dataloader()

    present_sensitive_labels = [
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not present_sensitive_labels:
        raise ValueError("No configured sensitive FSC intent labels exist in the dataset.")

    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    sr_model.eval()
    for parameter in sr_model.parameters():
        parameter.requires_grad = False

    intent_head = initialize_intent_head(
        sr_model=sr_model,
        sr_train_dataloader=sr_train_dataloader,
        sr_eval_dataloaders=sr_eval_dataloaders,
        sensitive_intent_labels=present_sensitive_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        warmup_steps=attack_cfg.intent_warmup_steps,
        device=device,
        checkpoint_path=intent_head_path,
        lr=attack_cfg.intent_head_lr,
        sensitive_class_weight=attack_cfg.sensitive_class_weight,
        eval_interval_steps=attack_cfg.intent_eval_interval_steps,
        eval_max_batches=attack_cfg.intent_eval_max_batches,
        decision_threshold=attack_cfg.intent_decision_threshold,
        num_intents=2,
    )

    results = evaluate_sr_attack(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=sr_test_dataloader,
        sensitive_intent_labels=present_sensitive_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        sensitive_class_ids=[1],
        device=device,
        uap_delta=uap_delta,
        decision_threshold=attack_cfg.intent_decision_threshold,
        max_batches=max_batches,
    )
    sr_model.to("cpu")
    del sr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def compute_risk_overall(sv_results, sr_results):
    asr_sv = None if sv_results is None else float(sv_results["asr_sv_at_clean_threshold"])
    asr_sr = None if sr_results is None else float(sr_results["conditional_flip_rate"])
    if asr_sv is None or asr_sr is None:
        risk_overall = None
    else:
        risk_overall = 0.5 * asr_sv + 0.5 * asr_sr
    return {
        "asr_sv": asr_sv,
        "asr_sr": asr_sr,
        "risk_overall": risk_overall,
    }


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wavlm_backbone = Path(
        _cfg_get(cfg, "wavlm_backbone", project_root / "model" / "wavlm-base")
    )
    if not wavlm_backbone.exists():
        raise FileNotFoundError(f"Missing local WavLM files: {wavlm_backbone}")

    uap_path = _cfg_get(
        cfg,
        "uap_path",
        attack_cfg.evaluation_uap_path or attack_cfg.output_path,
    )
    intent_head_path = Path(
        _cfg_get(
            cfg,
            "intent_head_path",
            project_root / "intent" / "wavlm-base-intent.pt",
        )
    )
    sv_ckpt = _optional_path(
        _cfg_get(cfg, "sv_ckpt", project_root / "model" / "wavlm_sv.ckpt")
    )
    run_sv = bool(_cfg_get(cfg, "eval_sv", True))
    run_sr = bool(_cfg_get(cfg, "eval_sr", True))
    sv_max_batches = _cfg_get(cfg, "sv_max_batches", attack_cfg.sv_eval_max_batches)
    sr_max_batches = _cfg_get(cfg, "sr_max_batches", attack_cfg.sr_eval_max_batches)
    sv_max_batches = None if sv_max_batches is None else int(sv_max_batches)
    sr_max_batches = None if sr_max_batches is None else int(sr_max_batches)
    output_path = Path(
        _cfg_get(
            cfg,
            "output_path",
            project_root / "results" / "main" / "wavlm_btuap_eval.json",
        )
    )

    print(
        "[*] WavLM BTUAP evaluation config: "
        f"wavlm_backbone={wavlm_backbone}, uap_path={uap_path}, "
        f"intent_head_path={intent_head_path}, sv_ckpt={sv_ckpt}, "
        f"eval_sv={run_sv}, eval_sr={run_sr}, "
        f"sv_max_batches={sv_max_batches}, sr_max_batches={sr_max_batches}"
    )

    uap_delta = load_uap(uap_path, device)
    evaluator = instantiate(cfg.evaluator)

    sv_results = None
    sr_results = None
    if run_sv:
        sv_results = evaluate_wavlm_sv(
            cfg=cfg,
            evaluator=evaluator,
            wavlm_backbone=wavlm_backbone,
            sv_ckpt=sv_ckpt,
            uap_delta=uap_delta,
            device=device,
            max_batches=sv_max_batches,
        )

    if run_sr:
        sr_results = evaluate_wavlm_sr(
            cfg=cfg,
            attack_cfg=attack_cfg,
            evaluator=evaluator,
            wavlm_backbone=wavlm_backbone,
            intent_head_path=intent_head_path,
            uap_delta=uap_delta,
            device=device,
            project_root=project_root,
            max_batches=sr_max_batches,
        )

    combined = compute_risk_overall(sv_results, sr_results)
    print(
        "[*] WavLM BTUAP summary: "
        f"ASR-SV={combined['asr_sv']}, "
        f"ASR-SR={combined['asr_sr']}, "
        f"Risk_overall={combined['risk_overall']}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "wavlm_backbone": str(wavlm_backbone),
                    "uap_path": str(uap_path),
                    "intent_head_path": str(intent_head_path),
                    "sv_ckpt": sv_ckpt,
                    "sv_max_batches": sv_max_batches,
                    "sr_max_batches": sr_max_batches,
                },
                "sv": sv_results,
                "sr": sr_results,
                "combined": combined,
            },
            handle,
            indent=2,
        )
    print(f"[*] Saved WavLM BTUAP evaluation: {output_path}")


if __name__ == "__main__":
    main_evaluation()
