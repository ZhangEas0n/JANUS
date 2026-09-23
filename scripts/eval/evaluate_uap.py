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
    load_sensitive_intent_head_checkpoint,
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


@hydra.main(config_path="../../config", config_name="train_eval")
def main_evaluation(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uap_path = Path(attack_cfg.evaluation_uap_path or attack_cfg.output_path)
    if not uap_path.exists():
        raise FileNotFoundError(f"Missing UAP tensor to evaluate: {uap_path}")
    uap_delta = torch.load(str(uap_path), map_location=device).to(device)
    if uap_delta.dim() == 1:
        uap_delta = uap_delta.unsqueeze(0)
    print(f"[*] Loaded UAP from {uap_path}, shape={tuple(uap_delta.shape)}")

    print("[*] Initializing SV evaluation...")
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.data.pipeline.test_pipeline = []
    sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
    local_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if local_wav2vec2.exists():
        sv_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2)
        print(f"[*] Using local SV wav2vec2 backbone from {local_wav2vec2}")
    else:
        raise FileNotFoundError(
            "SV evaluation requires the offline wav2vec2 backbone at "
            f"{local_wav2vec2}; it cannot download facebook/wav2vec2-base "
            "without network access."
        )
    sv_dm = construct_data_module(sv_cfg)
    sv_dataloader = sv_dm.test_dataloader()
    evaluator = instantiate(sv_cfg.evaluator)
    victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False

    print("[*] Running SV UAP evaluation...")
    evaluate_sv_attack(
        victim_model=victim_model,
        evaluator=evaluator,
        dataloader=sv_dataloader,
        pairs=sv_dm.test_pairs,
        device=device,
        uap_delta=uap_delta,
        max_batches=attack_cfg.sv_eval_max_batches,
    )

    print("[*] Initializing SR evaluation...")
    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    local_sr_wav2vec2 = Path(project_root) / "model" / "wav2vec2-base-960h"
    if not local_sr_wav2vec2.exists():
        raise FileNotFoundError(
            "SR evaluation requires the offline wav2vec2/tokenizer files at "
            f"{local_sr_wav2vec2}; it cannot download facebook/wav2vec2-base-960h "
            "without network access."
        )
    sr_cfg.network.wav2vec_hunggingface_id = str(local_sr_wav2vec2)
    sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_sr_wav2vec2)
    print(f"[*] Using local SR wav2vec2 backbone from {local_sr_wav2vec2}")
    print(f"[*] Using local SR tokenizer from {sr_cfg.tokenizer.tokenizer_huggingface_id}")
    sr_dm = construct_data_module(sr_cfg)
    sr_test_dataloader = sr_dm.test_dataloader()
    present_sensitive_labels = [
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not present_sensitive_labels:
        raise ValueError("No configured sensitive intent labels exist in the FSC dataset.")
    attack_cfg.sensitive_intent_labels = present_sensitive_labels
    attack_cfg.sensitive_class_ids = [1]

    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    sr_model.eval()
    for param in sr_model.parameters():
        param.requires_grad = False
    intent_head = load_sensitive_intent_head_checkpoint(
        checkpoint_path=attack_cfg.intent_head_path,
        sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
        decision_threshold=attack_cfg.intent_decision_threshold,
        device=device,
    )

    print("[*] Running SR UAP evaluation on FSC test benign samples...")
    evaluate_sr_attack(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=sr_test_dataloader,
        sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        sensitive_class_ids=attack_cfg.sensitive_class_ids,
        device=device,
        uap_delta=uap_delta,
        decision_threshold=attack_cfg.intent_decision_threshold,
        max_batches=attack_cfg.sr_eval_max_batches,
    )


if __name__ == "__main__":
    main_evaluation()
