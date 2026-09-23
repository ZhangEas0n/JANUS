from collections import deque

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.config import BTUAPAttackConfig
from btuap.common import apply_uap, normalize_batch_like_pipeline
from btuap.sr import (
    compute_sr_sensitive_intent_loss,
    evaluate_sr_attack,
    extract_raw_audio_batch,
    extract_sensitive_binary_targets,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from btuap.sv import compute_sv_risk_loss, generate_pseudo_enrollment_set
from btuap.sv import evaluate_sv_attack
from src.hydra_resolvers import (
    division_resolver,
    integer_division_resolver,
    random_uuid,
)
from src.main import construct_data_module, construct_module

load_dotenv()

OmegaConf.register_new_resolver("divide", division_resolver)
OmegaConf.register_new_resolver("idivide", integer_division_resolver)
OmegaConf.register_new_resolver("random_uuid", random_uuid)

@hydra.main(config_path="config", config_name="train_eval")
def main_attack(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = get_original_cwd()
    attack_cfg.resolve_paths(project_root)

    print("[*] Initializing SV DataModule...")
    sv_cfg = cfg
    sv_cfg.data.pipeline.train_pipeline = ["selector_train"]
    sv_dm = construct_data_module(sv_cfg)
    sv_train_dataloader = sv_dm.train_dataloader()
    sv_eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_eval_cfg.data.pipeline.test_pipeline = []
    sv_eval_dm = construct_data_module(sv_eval_cfg)
    sv_eval_dataloader = sv_eval_dm.test_dataloader()

    print("[*] Initializing SR DataModule...")
    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=attack_cfg.sr_train_batch_size,
        project_root=project_root,
    )
    sr_dm = construct_data_module(sr_cfg)
    sr_train_dataloader = sr_dm.train_dataloader()
    sr_eval_dataloaders = sr_dm.val_dataloader()
    sr_test_dataloader = sr_dm.test_dataloader()
    if hasattr(sr_dm, "intent_label_to_id"):
        present_sensitive_labels = []
        missing_sensitive_labels = []
        for label in attack_cfg.sensitive_intent_labels:
            if label in sr_dm.intent_label_to_id:
                present_sensitive_labels.append(label)
            else:
                missing_sensitive_labels.append(label)

        if len(present_sensitive_labels) == 0:
            raise ValueError(
                "None of the configured sensitive FSC intents exist in the dataset: "
                f"{attack_cfg.sensitive_intent_labels}"
            )

        attack_cfg.sensitive_intent_labels = present_sensitive_labels
        attack_cfg.sensitive_class_ids = [1]
        print(
            "[*] Binary sensitive intent head: "
            f"benign=0, sensitive=1, sensitive_intents={present_sensitive_labels}"
        )
        if len(missing_sensitive_labels) > 0:
            print(f"[!] Missing sensitive intent labels: {missing_sensitive_labels}")

    print(f"[*] Loading victim model from {attack_cfg.ckpt_path}...")
    sv_cfg.load_network_from_checkpoint = attack_cfg.ckpt_path
    evaluator = instantiate(sv_cfg.evaluator)
    victim_model = construct_module(sv_cfg, evaluator, sv_dm, load_optim=False).to(device)
    victim_model.eval()

    for param in victim_model.parameters():
        param.requires_grad = False

    print("[*] Loading SR surrogate model...")
    sr_model = construct_module(sr_cfg, evaluator, sr_dm, load_optim=False).to(device)
    sr_model.eval()

    for param in sr_model.parameters():
        param.requires_grad = False

    print("[*] Initializing sensitive intent head...")
    attack_cfg.ensure_intent_head_parent()
    intent_head = initialize_intent_head(
        sr_model=sr_model,
        sr_train_dataloader=sr_train_dataloader,
        sr_eval_dataloaders=sr_eval_dataloaders,
        sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        warmup_steps=attack_cfg.intent_warmup_steps,
        device=device,
        checkpoint_path=attack_cfg.intent_head_path,
        lr=attack_cfg.intent_head_lr,
        sensitive_class_weight=attack_cfg.sensitive_class_weight,
        eval_interval_steps=attack_cfg.intent_eval_interval_steps,
        eval_max_batches=attack_cfg.intent_eval_max_batches,
        decision_threshold=attack_cfg.intent_decision_threshold,
        num_intents=2,
    )

    if attack_cfg.intent_head_only:
        print("[+] Intent head evaluation finished. Skipping UAP optimization.")
        return

    audio_length_samples = attack_cfg.audio_length_samples
    print(
        "[*] UAP length: "
        f"{audio_length_samples} samples "
        f"({audio_length_samples / attack_cfg.sample_rate:.3f}s @ {attack_cfg.sample_rate}Hz)"
    )
    uap_delta = torch.zeros((1, audio_length_samples), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([uap_delta], lr=0.01)
    pseudo_embs = generate_pseudo_enrollment_set(
        victim_model,
        sv_train_dataloader,
        num_samples=attack_cfg.pseudo_enrollment_samples,
        device=device,
    )
    print("\n? Starting BTUAP-SV/SR Joint Perturbation Training...")
    sr_loss_window = deque(maxlen=attack_cfg.sr_loss_moving_average_window)
    global_step = 0
    evaluate_sr_attack(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=sr_eval_dataloaders,
        sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
        sensitive_keywords=attack_cfg.sensitive_keywords,
        sensitive_class_ids=attack_cfg.sensitive_class_ids,
        device=device,
        uap_delta=uap_delta.detach(),
        decision_threshold=attack_cfg.intent_decision_threshold,
        max_batches=attack_cfg.sr_attack_monitor_max_batches,
    )

    for epoch in range(attack_cfg.max_epochs):
        sv_iter = iter(sv_train_dataloader)
        sr_iter = iter(sr_train_dataloader)
        pbar = tqdm(
            range(attack_cfg.steps_per_epoch),
            total=attack_cfg.steps_per_epoch,
            desc=f"Epoch {epoch+1}/{attack_cfg.max_epochs}",
        )

        for _step in pbar:
            try:
                sv_batch = next(sv_iter)
            except StopIteration:
                sv_iter = iter(sv_train_dataloader)
                sv_batch = next(sv_iter)

            sv_batch = sv_batch.to(device)
            sv_clean_raw = sv_batch.network_input

            try:
                sr_batch = next(sr_iter)
            except StopIteration:
                sr_iter = iter(sr_train_dataloader)
                sr_batch = next(sr_iter)

            sr_batch = sr_batch.to(device)

            x_adv_raw = apply_uap(sv_clean_raw, uap_delta)
            x_adv = normalize_batch_like_pipeline(x_adv_raw)
            adv_embs = victim_model.compute_speaker_embedding(x_adv)
            sv_loss, current_prob, tau_cosine = compute_sv_risk_loss(
                adv_embs,
                pseudo_embs,
                attack_cfg.tau_eval,
                attack_cfg.beta,
            )

            sr_clean_raw = extract_raw_audio_batch(sr_batch, device)
            sr_adv_raw = apply_uap(sr_clean_raw, uap_delta)
            sr_adv = normalize_batch_like_pipeline(
                sr_adv_raw,
                input_lengths=sr_batch.input_lengths,
            )
            sr_target = extract_sensitive_binary_targets(
                batch=sr_batch,
                sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
                sensitive_keywords=attack_cfg.sensitive_keywords,
                device=device,
            )
            benign_mask = sr_target == 0
            sr_loss, sr_psens = compute_sr_sensitive_intent_loss(
                sr_model=sr_model,
                intent_head=intent_head,
                x_adv=sr_adv,
                input_lengths=sr_batch.input_lengths,
                sensitive_class_ids=attack_cfg.sensitive_class_ids,
                benign_mask=benign_mask,
            )
            energy_loss = uap_delta.pow(2).mean()
            if uap_delta.shape[-1] > 1:
                smooth_loss = (uap_delta[..., 1:] - uap_delta[..., :-1]).pow(2).mean()
            else:
                smooth_loss = torch.zeros((), device=device)
            total_loss = (
                attack_cfg.alpha * sv_loss
                + attack_cfg.gamma * sr_loss
                + attack_cfg.lambda_reg * energy_loss
                + attack_cfg.lambda_smooth * smooth_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                uap_delta.clamp_(-attack_cfg.epsilon, attack_cfg.epsilon)

            if benign_mask.any():
                sr_loss_window.append(sr_loss.item())
            sr_loss_avg = (
                sum(sr_loss_window) / len(sr_loss_window)
                if len(sr_loss_window) > 0
                else float("nan")
            )
            global_step += 1
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "SV": f"{sv_loss.item():.4f}",
                "SR": f"{sr_loss.item():.4f}",
                "Smooth": f"{smooth_loss.item():.6f}",
                "SR_Avg": f"{sr_loss_avg:.4f}",
                "SR_Benign": int(benign_mask.sum().item()),
                "FAR_Prob": f"{current_prob * 100:.1f}%",
                "P_sens": f"{sr_psens:.3f}",
                "TauEval": f"{attack_cfg.tau_eval:.3f}",
                "TauCos": f"{tau_cosine:.3f}", 
            })

            if (
                attack_cfg.sr_attack_eval_interval_steps > 0
                and global_step % attack_cfg.sr_attack_eval_interval_steps == 0
            ):
                evaluate_sr_attack(
                    sr_model=sr_model,
                    intent_head=intent_head,
                    dataloader=sr_eval_dataloaders,
                    sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
                    sensitive_keywords=attack_cfg.sensitive_keywords,
                    sensitive_class_ids=attack_cfg.sensitive_class_ids,
                    device=device,
                    uap_delta=uap_delta.detach(),
                    decision_threshold=attack_cfg.intent_decision_threshold,
                    max_batches=attack_cfg.sr_attack_monitor_max_batches,
                )

    print("\n[+] SV/SR UAP optimization completed!")
    attack_cfg.ensure_output_parent()
    torch.save(uap_delta.detach().cpu(), attack_cfg.output_path)
    print(f"[+] UAP tensor saved to '{attack_cfg.output_path}'")

    if attack_cfg.run_sr_eval_after_training:
        evaluate_sr_attack(
            sr_model=sr_model,
            intent_head=intent_head,
            dataloader=sr_test_dataloader,
            sensitive_intent_labels=attack_cfg.sensitive_intent_labels,
            sensitive_keywords=attack_cfg.sensitive_keywords,
            sensitive_class_ids=attack_cfg.sensitive_class_ids,
            device=device,
            uap_delta=uap_delta.detach(),
            decision_threshold=attack_cfg.intent_decision_threshold,
            max_batches=attack_cfg.sr_eval_max_batches,
        )

    if attack_cfg.run_sv_eval_after_training:
        evaluate_sv_attack(
            victim_model=victim_model,
            evaluator=evaluator,
            dataloader=sv_eval_dataloader,
            pairs=sv_eval_dm.test_pairs,
            device=device,
            uap_delta=uap_delta.detach(),
            max_batches=attack_cfg.sv_eval_max_batches,
        )

if __name__ == "__main__":
    main_attack()
