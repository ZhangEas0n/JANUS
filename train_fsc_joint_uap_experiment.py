from __future__ import print_function

import json
import os
import random
from collections import deque
from pathlib import Path

import hydra
import torch
import torchaudio
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.common import apply_uap, normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.fsc_sv import (
    build_fsc_enrollment_gallery,
    compute_fsc_any_enrollment_loss,
    compute_fsc_impostor_loss,
    extract_fsc_speaker_ids,
)
from btuap.mel_mimic import (
    MelMimicLoss,
    MultiResolutionSTFTLoss,
    load_environment_sound,
    normalize_to_rms,
    rms,
    waveform_similarity_loss,
)
from btuap.sr import (
    compute_sr_sensitive_intent_loss,
    extract_raw_audio_batch,
    extract_sensitive_binary_targets,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from src.data.modules.speaker.speaker_data_module import SpeakerLightningDataModule
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


class FSCSpeakerMetadata(SpeakerLightningDataModule):
    """Minimal metadata needed to restore the pretrained SV checkpoint."""

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
    if "fsc_joint_train" in cfg and key in cfg.fsc_joint_train:
        return cfg.fsc_joint_train[key]
    return default


def count_fsc_speakers(dataset):
    return len({str(row["speakerId"]) for row in dataset.rows})


def load_sv_threshold(path, fallback):
    path = Path(path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload["sv_threshold"])
        print("[*] Loaded validation SV threshold {:.6f} from {}".format(threshold, path))
        return threshold
    print("[!] Missing threshold file {}; using fallback {:.6f}".format(path, fallback))
    return float(fallback)


def save_uap(path, uap_delta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(uap_delta.detach().cpu(), str(path))
    print("[+] Saved UAP tensor to {}".format(path))


def save_uap_wav(path, uap_delta, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = uap_delta.detach().cpu().clamp(-1.0, 1.0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)
    print("[+] Saved UAP waveform to {}".format(path))


@hydra.main(config_path="config", config_name="train_eval")
def main_train(cfg: DictConfig):
    attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(_cfg_get(cfg, "seed", cfg.seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    fsc_root = Path(
        str(_cfg_get(cfg, "fsc_root", project_root / "fluent_speech_commands_dataset"))
    )
    backbone = Path(
        str(_cfg_get(cfg, "backbone", project_root / "model" / "wav2vec2-base-960h"))
    )
    sv_ckpt = Path(str(_cfg_get(cfg, "sv_ckpt", attack_cfg.ckpt_path)))
    intent_head_path = Path(
        str(_cfg_get(cfg, "intent_head_path", attack_cfg.intent_head_path))
    )
    output_path = Path(
        str(
            _cfg_get(
                cfg,
                "output_path",
                project_root / "perturbations" / "mel_mimic" / "fan" / "janus_melmimic_fan_mu001.pt",
            )
        )
    )
    output_wav_path = Path(
        str(_cfg_get(cfg, "output_wav_path", output_path.with_suffix(".wav")))
    )
    train_log_path = Path(
        str(
            _cfg_get(
                cfg,
                "train_log_path",
                project_root
                / "results"
                / "joint"
                / "training"
                / "janus_melmimic_fan_mu001_train_log.json",
            )
        )
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

    for required in (fsc_root, backbone, sv_ckpt, intent_head_path):
        if not required.exists():
            raise FileNotFoundError("Missing required training asset: {}".format(required))

    audio_length = float(_cfg_get(cfg, "audio_length", 3.0))
    sample_rate = int(_cfg_get(cfg, "sample_rate", attack_cfg.sample_rate))
    uap_samples = int(round(audio_length * sample_rate))
    if uap_samples <= 0:
        raise ValueError("fsc_joint_train.audio_length must be positive.")

    max_epochs = int(_cfg_get(cfg, "max_epochs", attack_cfg.max_epochs))
    steps_per_epoch = int(_cfg_get(cfg, "steps_per_epoch", attack_cfg.steps_per_epoch))
    train_batch_size = int(_cfg_get(cfg, "train_batch_size", attack_cfg.sr_train_batch_size))
    learning_rate = float(_cfg_get(cfg, "learning_rate", 0.01))
    alpha = float(_cfg_get(cfg, "alpha", attack_cfg.alpha))
    gamma = float(_cfg_get(cfg, "gamma", attack_cfg.gamma))
    epsilon = float(_cfg_get(cfg, "epsilon", attack_cfg.epsilon))
    beta = float(_cfg_get(cfg, "beta", attack_cfg.beta))
    lambda_reg = float(_cfg_get(cfg, "lambda_reg", attack_cfg.lambda_reg))
    lambda_smooth = float(_cfg_get(cfg, "lambda_smooth", attack_cfg.lambda_smooth))
    env_sound_path = Path(
        str(
            _cfg_get(
                cfg,
                "env_sound_path",
                project_root / "data" / "environmental_sounds" / "fan_16k_mono.wav",
            )
        )
    )
    mu_mimic = float(_cfg_get(cfg, "mu_mimic", 0.01))
    mel_n_fft = int(_cfg_get(cfg, "mel_n_fft", 512))
    mel_hop_length = int(_cfg_get(cfg, "mel_hop_length", 160))
    mel_n_mels = int(_cfg_get(cfg, "mel_n_mels", 80))
    mel_eps = float(_cfg_get(cfg, "mel_eps", 1e-6))
    carrier_enabled = bool(_cfg_get(cfg, "carrier_enabled", False))
    carrier_fraction = float(_cfg_get(cfg, "carrier_fraction", 0.7))
    configured_residual_epsilon = _cfg_get(cfg, "residual_epsilon", None)
    residual_epsilon = (
        epsilon * (1.0 - carrier_fraction)
        if configured_residual_epsilon is None
        else float(configured_residual_epsilon)
    )
    carrier_to_residual_rms = float(
        _cfg_get(cfg, "carrier_to_residual_rms", 0.0)
    )
    mu_stft = float(_cfg_get(cfg, "mu_stft", 0.1))
    mu_waveform = float(_cfg_get(cfg, "mu_waveform", 0.05))
    if mu_mimic <= 0:
        raise ValueError("fsc_joint_train.mu_mimic must be positive for Mel-Mimic training.")
    if not env_sound_path.is_file():
        raise FileNotFoundError(
            "Missing fsc_joint_train.env_sound_path: {}".format(env_sound_path)
        )
    if carrier_enabled:
        if not 0.0 < carrier_fraction < 1.0:
            raise ValueError("carrier_fraction must be between zero and one.")
        carrier_peak = epsilon * carrier_fraction
        if residual_epsilon <= 0.0:
            raise ValueError("residual_epsilon must be positive.")
        if carrier_peak + residual_epsilon > epsilon + 1e-8:
            raise ValueError(
                "carrier peak plus residual_epsilon must not exceed epsilon."
            )
        if carrier_to_residual_rms < 0.0:
            raise ValueError("carrier_to_residual_rms must be non-negative.")
    enrollment_k = int(_cfg_get(cfg, "enrollment_k", 3))
    sv_objective = str(_cfg_get(cfg, "sv_objective", "pair_aligned")).lower()
    if sv_objective not in ("pair_aligned", "any_enrollment"):
        raise ValueError(
            "fsc_joint_train.sv_objective must be pair_aligned or any_enrollment."
        )
    configured_speakers = _cfg_get(cfg, "pseudo_enrollment_speakers", None)
    any_enrollment_speakers = int(_cfg_get(cfg, "any_enrollment_speakers", 10))
    any_enrollment_top_k = int(_cfg_get(cfg, "any_enrollment_top_k", 1))
    num_random_targets = int(_cfg_get(cfg, "num_random_targets", 4))
    num_hard_targets = int(_cfg_get(cfg, "num_hard_targets", 4))
    fallback_sv_threshold = float(_cfg_get(cfg, "fallback_sv_threshold", attack_cfg.tau_eval))
    sv_threshold = load_sv_threshold(threshold_path, fallback_sv_threshold)
    intent_warmup_steps = int(_cfg_get(cfg, "intent_warmup_steps", attack_cfg.intent_warmup_steps))
    intent_eval_max_batches = _cfg_get(
        cfg, "intent_eval_max_batches", attack_cfg.intent_eval_max_batches
    )
    if intent_eval_max_batches is not None:
        intent_eval_max_batches = int(intent_eval_max_batches)

    print(
        "[*] FSC-only joint UAP training: device={}, uap={:.2f}s/{} samples, "
        "epochs={}, steps_per_epoch={}, batch_size={}, alpha={}, gamma={}, "
        "epsilon={}, sv_threshold={:.6f}, objective={}, random_targets={}, "
        "hard_targets={}, env_sound={}, mu_mimic={}".format(
            device,
            audio_length,
            uap_samples,
            max_epochs,
            steps_per_epoch,
            train_batch_size,
            alpha,
            gamma,
            epsilon,
            sv_threshold,
            sv_objective,
            num_random_targets,
            num_hard_targets,
            env_sound_path,
            mu_mimic,
        )
    )

    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=attack_cfg.sr_train_max_num_samples,
        train_batch_size=train_batch_size,
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
    train_speakers = count_fsc_speakers(sr_dm.train_ds)
    num_enrollment_speakers = (
        None if configured_speakers is None else int(configured_speakers)
    )
    if num_enrollment_speakers is not None and (
        num_enrollment_speakers < 2 or num_enrollment_speakers > train_speakers
    ):
        raise ValueError(
            "pseudo_enrollment_speakers must be in [2, {}], got {}".format(
                train_speakers, num_enrollment_speakers
            )
        )

    sensitive_labels = [
        label
        for label in attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not sensitive_labels:
        raise ValueError("No configured sensitive intents exist in FSC.")

    evaluator = instantiate(cfg.evaluator)
    sv_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    sv_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sv_cfg.load_network_from_checkpoint = str(sv_ckpt)
    sv_model = construct_module(
        sv_cfg,
        evaluator,
        FSCSpeakerMetadata(num_speakers=train_speakers),
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
        warmup_steps=intent_warmup_steps,
        device=device,
        checkpoint_path=intent_head_path,
        lr=attack_cfg.intent_head_lr,
        sensitive_class_weight=attack_cfg.sensitive_class_weight,
        eval_interval_steps=attack_cfg.intent_eval_interval_steps,
        eval_max_batches=intent_eval_max_batches,
        decision_threshold=attack_cfg.intent_decision_threshold,
        num_intents=2,
    )

    if sv_objective == "any_enrollment":
        gallery = build_fsc_enrollment_gallery(
            victim_model=sv_model,
            dataloader=val_dataloader,
            num_speakers=any_enrollment_speakers,
            enrollment_k=enrollment_k,
            device=device,
        )
        source_speakers = {
            str(row["speakerId"]).strip() for row in sr_dm.train_ds.rows
        }
        overlap = source_speakers.intersection(gallery.speaker_ids)
        if overlap:
            raise ValueError(
                "Any-enrollment source and target speakers overlap: {}".format(
                    sorted(overlap)
                )
            )
        print(
            "[*] Fixed Any-{} target speakers from FSC validation: {}".format(
                len(gallery.speaker_ids), list(gallery.speaker_ids)
            )
        )
    else:
        gallery = build_fsc_enrollment_gallery(
            victim_model=sv_model,
            dataloader=train_dataloader,
            num_speakers=num_enrollment_speakers,
            enrollment_k=enrollment_k,
            device=device,
        )
    target_generator = torch.Generator().manual_seed(seed + 1701)

    env_template = load_environment_sound(
        path=env_sound_path,
        target_len=uap_samples,
        sample_rate=sample_rate,
        device=device,
        dtype=torch.float32,
    )
    if carrier_enabled:
        carrier_template = (
            env_template * (epsilon * carrier_fraction)
        ).detach()
        carrier_rms_reference = rms(carrier_template).detach()
        uap_delta = carrier_template.clone().detach().requires_grad_(True)
        fixed_mimic_target = carrier_template
    else:
        carrier_template = None
        uap_delta = torch.zeros(
            (1, uap_samples),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        fixed_mimic_target = None
    mel_mimic_loss_fn = MelMimicLoss(
        sample_rate=sample_rate,
        n_fft=mel_n_fft,
        hop_length=mel_hop_length,
        n_mels=mel_n_mels,
        eps=mel_eps,
    ).to(device)
    stft_mimic_loss_fn = MultiResolutionSTFTLoss(eps=mel_eps).to(device)
    for parameter in mel_mimic_loss_fn.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam([uap_delta], lr=learning_rate)
    sr_loss_window = deque(maxlen=attack_cfg.sr_loss_moving_average_window)
    history = []
    global_step = 0

    for epoch in range(max_epochs):
        train_iterator = iter(train_dataloader)
        pbar = tqdm(
            range(steps_per_epoch),
            desc="FSC-only joint epoch {}/{}".format(epoch + 1, max_epochs),
        )
        for _ in pbar:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            batch = batch.to(device)

            raw_audio = extract_raw_audio_batch(batch, device)
            adv_raw = apply_uap(raw_audio, uap_delta)
            adv_input = normalize_batch_like_pipeline(
                adv_raw, input_lengths=batch.input_lengths
            )

            batch_speaker_ids = extract_fsc_speaker_ids(batch)
            sv_embeddings = sv_model.compute_speaker_embedding(adv_input)
            if sv_objective == "any_enrollment":
                sv_result = compute_fsc_any_enrollment_loss(
                    adv_embeddings=sv_embeddings,
                    query_speaker_ids=batch_speaker_ids,
                    gallery=gallery,
                    sv_threshold=sv_threshold,
                    beta=beta,
                    top_k=any_enrollment_top_k,
                )
            else:
                sv_result = compute_fsc_impostor_loss(
                    adv_embeddings=sv_embeddings,
                    query_speaker_ids=batch_speaker_ids,
                    gallery=gallery,
                    sv_threshold=sv_threshold,
                    beta=beta,
                    num_random_targets=num_random_targets,
                    num_hard_targets=num_hard_targets,
                    generator=target_generator,
                )
            sv_loss = sv_result.loss

            sr_targets = extract_sensitive_binary_targets(
                batch=batch,
                sensitive_intent_labels=sensitive_labels,
                sensitive_keywords=attack_cfg.sensitive_keywords,
                device=device,
            )
            benign_mask = sr_targets == 0
            sr_loss, p_sens = compute_sr_sensitive_intent_loss(
                sr_model=sr_model,
                intent_head=intent_head,
                x_adv=adv_input,
                input_lengths=batch.input_lengths,
                sensitive_class_ids=[1],
                benign_mask=benign_mask,
            )

            energy_loss = uap_delta.pow(2).mean()
            smooth_loss = (uap_delta[:, 1:] - uap_delta[:, :-1]).pow(2).mean()
            env_for_loss = (
                fixed_mimic_target
                if carrier_enabled
                else normalize_to_rms(env_template, rms(uap_delta.detach()))
            )
            mimic_loss = mel_mimic_loss_fn(uap_delta, env_for_loss)
            if carrier_enabled:
                stft_loss = stft_mimic_loss_fn(uap_delta, env_for_loss)
                waveform_loss = waveform_similarity_loss(
                    uap_delta, env_for_loss
                )
            else:
                stft_loss = uap_delta.new_zeros(())
                waveform_loss = uap_delta.new_zeros(())
            total_loss = (
                alpha * sv_loss
                + gamma * sr_loss
                + mu_mimic * mimic_loss
                + mu_stft * stft_loss
                + mu_waveform * waveform_loss
                + lambda_reg * energy_loss
                + lambda_smooth * smooth_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                if carrier_enabled:
                    residual = (uap_delta - carrier_template).clamp(
                        -residual_epsilon, residual_epsilon
                    )
                    if carrier_to_residual_rms > 0.0:
                        residual_rms = rms(residual)
                        maximum_residual_rms = (
                            carrier_rms_reference / carrier_to_residual_rms
                        )
                        if residual_rms > maximum_residual_rms:
                            residual = residual * (
                                maximum_residual_rms / residual_rms
                            )
                    uap_delta.copy_(
                        (carrier_template + residual).clamp(-epsilon, epsilon)
                    )
                else:
                    uap_delta.clamp_(-epsilon, epsilon)

            if benign_mask.any():
                sr_loss_window.append(float(sr_loss.detach()))
            sr_loss_avg = (
                sum(sr_loss_window) / len(sr_loss_window)
                if sr_loss_window
                else float("nan")
            )
            global_step += 1
            step_record = {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": float(total_loss.detach()),
                "sv_loss": float(sv_loss.detach()),
                "sr_loss": float(sr_loss.detach()),
                "mimic_loss": float(mimic_loss.detach()),
                "stft_loss": float(stft_loss.detach()),
                "waveform_loss": float(waveform_loss.detach()),
                "reg_loss": float(energy_loss.detach()),
                "smooth_loss": float(smooth_loss.detach()),
                "sampled_far": sv_result.metrics["sampled_far"],
                "any_enrollment_asr": sv_result.metrics.get(
                    "any_enrollment_asr"
                ),
                "random_pair_far": sv_result.metrics.get("random_pair_far"),
                "hard_pair_far": sv_result.metrics.get("hard_pair_far"),
                "query_any_far": sv_result.metrics["query_any_far"],
                "p_sensitive": float(p_sens),
                "mean_impostor_score": sv_result.metrics["mean_impostor_score"],
                "mean_threshold_margin": sv_result.metrics["mean_threshold_margin"],
                "best_impostor_score": sv_result.metrics["mean_best_impostor_score"],
                "sr_loss_avg": float(sr_loss_avg),
                "uap_linf": float(uap_delta.detach().abs().max()),
                "uap_rms": float(uap_delta.detach().pow(2).mean().sqrt()),
                "residual_linf": (
                    float((uap_delta.detach() - carrier_template).abs().max())
                    if carrier_enabled
                    else None
                ),
            }
            history.append(step_record)
            pbar.set_postfix(
                {
                    "loss": "{:.4f}".format(step_record["loss"]),
                    "sv": "{:.4f}".format(step_record["sv_loss"]),
                    "sr": "{:.4f}".format(step_record["sr_loss"]),
                    "mimic": "{:.4f}".format(step_record["mimic_loss"]),
                    "far": "{:.3f}".format(step_record["sampled_far"]),
                    "rand_far": "{:.3f}".format(
                        step_record["random_pair_far"]
                        if step_record["random_pair_far"] is not None
                        else float("nan")
                    ),
                    "p_sens": "{:.3f}".format(p_sens),
                    "sv_score": "{:.3f}".format(
                        step_record["mean_impostor_score"]
                    ),
                }
            )

        epoch_path = output_path.with_name(
            "{}.epoch{}{}".format(output_path.stem, epoch + 1, output_path.suffix)
        )
        save_uap(epoch_path, uap_delta)

    save_uap(output_path, uap_delta)
    save_uap_wav(output_wav_path, uap_delta, sample_rate)
    if carrier_enabled:
        save_uap_wav(
            output_wav_path.with_name(output_wav_path.stem + ".carrier.wav"),
            carrier_template,
            sample_rate,
        )
        save_uap_wav(
            output_wav_path.with_name(output_wav_path.stem + ".residual.wav"),
            uap_delta.detach() - carrier_template,
            sample_rate,
        )
    with torch.no_grad():
        final_env_for_loss = (
            fixed_mimic_target
            if carrier_enabled
            else normalize_to_rms(env_template, rms(uap_delta.detach()))
        )
        final_mimic_loss = float(
            mel_mimic_loss_fn(uap_delta.detach(), final_env_for_loss).cpu()
        )
        final_delta_rms = float(rms(uap_delta.detach()).cpu())
        final_delta_linf = float(uap_delta.detach().abs().max().cpu())
        final_stft_loss = float(
            stft_mimic_loss_fn(uap_delta.detach(), final_env_for_loss).cpu()
        ) if carrier_enabled else 0.0
        final_waveform_loss = float(
            waveform_similarity_loss(
                uap_delta.detach(), final_env_for_loss
            ).cpu()
        ) if carrier_enabled else 0.0
        final_residual_linf = float(
            (uap_delta.detach() - carrier_template).abs().max().cpu()
        ) if carrier_enabled else None
    if not torch.isfinite(uap_delta.detach()).all():
        raise RuntimeError("Final Mel-Mimic perturbation contains NaN or Inf.")
    if final_delta_linf > epsilon + 1e-6:
        raise RuntimeError(
            "Final perturbation Linf {:.8f} exceeds epsilon {:.8f}.".format(
                final_delta_linf, epsilon
            )
        )
    if not torch.isfinite(torch.tensor(final_mimic_loss)):
        raise RuntimeError("Final Mel-Mimic loss is NaN or Inf.")
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    with train_log_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "seed": seed,
                    "fsc_root": str(fsc_root),
                    "backbone": str(backbone),
                    "sv_ckpt": str(sv_ckpt),
                    "intent_head_path": str(intent_head_path),
                    "output_path": str(output_path),
                    "output_wav_path": str(output_wav_path),
                    "env_sound_path": str(env_sound_path),
                    "mu_mimic": mu_mimic,
                    "carrier_enabled": carrier_enabled,
                    "carrier_fraction": carrier_fraction,
                    "residual_epsilon": residual_epsilon,
                    "carrier_to_residual_rms": carrier_to_residual_rms,
                    "mu_stft": mu_stft,
                    "mu_waveform": mu_waveform,
                    "mel_n_fft": mel_n_fft,
                    "mel_hop_length": mel_hop_length,
                    "mel_n_mels": mel_n_mels,
                    "mel_eps": mel_eps,
                    "audio_length": audio_length,
                    "sample_rate": sample_rate,
                    "uap_samples": uap_samples,
                    "max_epochs": max_epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "train_batch_size": train_batch_size,
                    "learning_rate": learning_rate,
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "beta": beta,
                    "lambda_reg": lambda_reg,
                    "lambda_smooth": lambda_smooth,
                    "sv_threshold": sv_threshold,
                    "sv_objective": sv_objective,
                    "enrollment_k": enrollment_k,
                    "pseudo_enrollment_speakers": len(gallery.speaker_ids),
                    "enrollment_sample_count": len(gallery.sample_ids),
                    "num_random_targets": num_random_targets,
                    "num_hard_targets": num_hard_targets,
                    "any_enrollment_speakers": len(gallery.speaker_ids),
                    "any_enrollment_top_k": any_enrollment_top_k,
                    "gallery_speaker_ids": list(gallery.speaker_ids),
                    "gallery_sample_ids": sorted(gallery.sample_ids),
                    "sensitive_intent_labels": sensitive_labels,
                },
                "history": history,
                "final": dict(
                    history[-1] if history else {},
                    final_mimic_loss=final_mimic_loss,
                    final_stft_loss=final_stft_loss,
                    final_waveform_loss=final_waveform_loss,
                    final_residual_linf=final_residual_linf,
                    final_delta_rms=final_delta_rms,
                    final_delta_linf=final_delta_linf,
                    delta_shape=list(uap_delta.shape),
                    saved_pt_path=str(output_path),
                    saved_wav_path=str(output_wav_path),
                ),
            },
            handle,
            indent=2,
        )
    print("[+] Saved FSC-only joint training log to {}".format(train_log_path))
    print(
        "[+] Mel-Mimic generation complete: shape={}, RMS={:.8f}, "
        "Linf={:.8f}, final_mimic_loss={:.6f}".format(
            list(uap_delta.shape),
            final_delta_rms,
            final_delta_linf,
            final_mimic_loss,
        )
    )


if __name__ == "__main__":
    main_train()
