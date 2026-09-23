from __future__ import print_function

import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from btuap.common import apply_uap, normalize_batch_like_pipeline
from btuap.config import BTUAPAttackConfig
from btuap.sr import (
    compute_sr_sensitive_intent_loss,
    evaluate_sr_attack,
    extract_raw_audio_batch,
    extract_sensitive_binary_targets,
    initialize_intent_head,
    make_speech_recognition_cfg,
)
from btuap.sv import evaluator_score_to_cosine_threshold
from src.data.modules.speaker.speaker_data_module import SpeakerLightningDataModule
from src.eval_metrics import calculate_eer
from src.evaluation.speaker.speaker_recognition_evaluator import (
    EmbeddingSample,
    EvaluationPair,
)
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
    """Minimal speaker metadata required to restore the pretrained SV model."""

    def __init__(self, num_speakers, val_pairs, test_pairs):
        super().__init__()
        self._num_speakers = int(num_speakers)
        self._val_pairs = list(val_pairs)
        self._test_pairs = list(test_pairs)

    @property
    def num_speakers(self):
        return self._num_speakers

    @property
    def val_pairs(self):
        return self._val_pairs

    @property
    def test_pairs(self):
        return self._test_pairs

    def summary(self):
        print(
            "FSC SV metadata: "
            f"speakers={self.num_speakers}, val_pairs={len(self.val_pairs)}, "
            f"test_pairs={len(self.test_pairs)}"
        )


def _cfg_get(cfg, key, default):
    if "fsc_joint" in cfg and key in cfg.fsc_joint:
        return cfg.fsc_joint[key]
    return default


def _speaker_ids_from_batch(batch):
    speaker_ids = []
    for key in batch.keys:
        side_info = batch.side_info.get(key) if batch.side_info is not None else None
        if side_info is None or side_info.meta is None:
            raise ValueError(f"Missing FSC metadata for sample {key}")
        speaker_id = side_info.meta.get("speaker_id")
        if speaker_id is None:
            raise ValueError(f"Missing FSC speaker_id for sample {key}")
        speaker_ids.append(str(speaker_id))
    return speaker_ids


def _group_fsc_sample_keys(dataset):
    grouped = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        grouped[str(row["speakerId"])].append(f"fsc-{index}")
    return dict(grouped)


def build_fsc_verification_pairs(dataset, max_pairs, seed):
    """Build a deterministic, balanced set of genuine and impostor trials."""
    grouped = _group_fsc_sample_keys(dataset)
    speakers = sorted(grouped)
    if len(speakers) < 2:
        raise ValueError("FSC SV evaluation needs at least two speakers.")

    target_each = max(1, int(max_pairs) // 2)
    rng = random.Random(int(seed))
    positive_pairs = set()
    negative_pairs = set()

    positive_speakers = [speaker for speaker in speakers if len(grouped[speaker]) >= 2]
    attempts = 0
    while len(positive_pairs) < target_each and attempts < target_each * 100:
        speaker = rng.choice(positive_speakers)
        first, second = rng.sample(grouped[speaker], 2)
        positive_pairs.add(tuple(sorted((first, second))))
        attempts += 1

    attempts = 0
    while len(negative_pairs) < target_each and attempts < target_each * 100:
        first_speaker, second_speaker = rng.sample(speakers, 2)
        first = rng.choice(grouped[first_speaker])
        second = rng.choice(grouped[second_speaker])
        negative_pairs.add(tuple(sorted((first, second))))
        attempts += 1

    pairs = [
        EvaluationPair(True, first, second)
        for first, second in sorted(positive_pairs)
    ]
    pairs.extend(
        EvaluationPair(False, first, second)
        for first, second in sorted(negative_pairs)
    )
    rng.shuffle(pairs)
    if not positive_pairs or not negative_pairs:
        raise ValueError("Unable to construct both genuine and impostor FSC trials.")
    print(
        "[*] FSC verification trials: "
        f"speakers={len(speakers)}, genuine={len(positive_pairs)}, "
        f"impostor={len(negative_pairs)}"
    )
    return pairs


def generate_fsc_pseudo_enrollment_set(
    victim_model, dataloader, num_speakers, device
):
    """Select one clean enrollment embedding per unique FSC training speaker."""
    victim_model.eval()
    embeddings = []
    enrollment_speaker_ids = []
    seen_speakers = set()
    pbar = tqdm(total=num_speakers, desc="FSC pseudo-enrollment", leave=False)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            raw_audio = extract_raw_audio_batch(batch, device)
            clean_input = normalize_batch_like_pipeline(
                raw_audio, input_lengths=batch.input_lengths
            )
            batch_embeddings = victim_model.compute_speaker_embedding(clean_input)
            if batch_embeddings.dim() == 1:
                batch_embeddings = batch_embeddings.unsqueeze(0)

            for speaker_id, embedding in zip(
                _speaker_ids_from_batch(batch), batch_embeddings
            ):
                if speaker_id in seen_speakers:
                    continue
                seen_speakers.add(speaker_id)
                embeddings.append(embedding.detach())
                enrollment_speaker_ids.append(speaker_id)
                pbar.update(1)
                if len(embeddings) >= num_speakers:
                    break
            if len(embeddings) >= num_speakers:
                break
    pbar.close()

    if len(embeddings) < num_speakers:
        raise ValueError(
            f"Requested {num_speakers} enrollment speakers, found {len(embeddings)}."
        )
    return (
        F.normalize(torch.stack(embeddings), p=2, dim=-1),
        enrollment_speaker_ids,
    )


def compute_fsc_sv_risk_loss(
    adv_embeddings,
    batch_speaker_ids,
    enrollment_embeddings,
    enrollment_speaker_ids,
    tau_eval,
    beta,
):
    """Maximize impostor acceptance while excluding genuine-speaker targets."""
    adv_embeddings = F.normalize(adv_embeddings, p=2, dim=-1)
    scores = torch.matmul(adv_embeddings, enrollment_embeddings.T)
    tau_cosine = evaluator_score_to_cosine_threshold(tau_eval)
    crossing_probability = torch.sigmoid(beta * (scores - tau_cosine))
    genuine_mask = torch.tensor(
        [
            [batch_id == enrollment_id for enrollment_id in enrollment_speaker_ids]
            for batch_id in batch_speaker_ids
        ],
        dtype=torch.bool,
        device=adv_embeddings.device,
    )
    crossing_probability = crossing_probability.masked_fill(genuine_mask, 0.0)
    probability_any_impostor = 1.0 - torch.prod(
        1.0 - crossing_probability, dim=1
    )
    return (
        -probability_any_impostor.mean(),
        probability_any_impostor.mean().detach().item(),
        tau_cosine,
    )


def _collect_fsc_sv_embeddings(victim_model, dataloader, device, uap_delta=None):
    clean_samples = []
    adv_samples = []
    cosine_sum = 0.0
    cosine_count = 0
    victim_model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="FSC SV embeddings", leave=False):
            batch = batch.to(device)
            raw_audio = extract_raw_audio_batch(batch, device)
            clean_input = normalize_batch_like_pipeline(
                raw_audio, input_lengths=batch.input_lengths
            )
            clean_embeddings = victim_model.compute_speaker_embedding(clean_input)
            if clean_embeddings.dim() == 1:
                clean_embeddings = clean_embeddings.unsqueeze(0)

            if uap_delta is None:
                adv_embeddings = clean_embeddings
            else:
                adv_raw = apply_uap(raw_audio, uap_delta)
                adv_input = normalize_batch_like_pipeline(
                    adv_raw, input_lengths=batch.input_lengths
                )
                adv_embeddings = victim_model.compute_speaker_embedding(adv_input)
                if adv_embeddings.dim() == 1:
                    adv_embeddings = adv_embeddings.unsqueeze(0)
                cosine = F.cosine_similarity(clean_embeddings, adv_embeddings, dim=-1)
                cosine_sum += cosine.sum().item()
                cosine_count += cosine.numel()

            for index, sample_id in enumerate(batch.keys):
                clean_samples.append(
                    EmbeddingSample(sample_id, clean_embeddings[index].detach().cpu())
                )
                adv_samples.append(
                    EmbeddingSample(sample_id, adv_embeddings[index].detach().cpu())
                )

    mean_cosine = cosine_sum / cosine_count if cosine_count else None
    return clean_samples, adv_samples, mean_cosine


def _scores_for_pairs(evaluator, pairs, samples):
    sample_map = {sample.sample_id: sample for sample in samples}
    labels = []
    prediction_pairs = []
    for pair in pairs:
        if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
            continue
        labels.append(1 if pair.same_speaker else 0)
        prediction_pairs.append(
            (sample_map[pair.sample1_id], sample_map[pair.sample2_id])
        )
    if not prediction_pairs:
        raise ValueError("No FSC verification pairs matched the collected embeddings.")
    raw_scores = evaluator._compute_prediction_scores(prediction_pairs)
    scores = torch.as_tensor(raw_scores, dtype=torch.float32)
    scores = torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0).tolist()
    return labels, scores


def _mixed_scores_for_pairs(evaluator, pairs, clean_samples, adv_samples):
    """Score clean enrollment against adversarial query speech."""
    clean_map = {sample.sample_id: sample for sample in clean_samples}
    adv_map = {sample.sample_id: sample for sample in adv_samples}
    labels = []
    prediction_pairs = []
    for pair in pairs:
        if pair.sample1_id not in clean_map or pair.sample2_id not in adv_map:
            continue
        labels.append(1 if pair.same_speaker else 0)
        prediction_pairs.append(
            (clean_map[pair.sample1_id], adv_map[pair.sample2_id])
        )
    if not prediction_pairs:
        raise ValueError("No FSC verification pairs matched clean/attack embeddings.")
    raw_scores = evaluator._compute_prediction_scores(prediction_pairs)
    scores = torch.as_tensor(raw_scores, dtype=torch.float32)
    scores = torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0).tolist()
    return labels, scores


def _rates_at_threshold(labels, scores, threshold):
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    positives = [score for label, score in zip(labels, scores) if label == 1]
    far = sum(score >= threshold for score in negatives) / max(1, len(negatives))
    frr = sum(score < threshold for score in positives) / max(1, len(positives))
    attack_success = sum(
        (label == 0 and score >= threshold)
        or (label == 1 and score < threshold)
        for label, score in zip(labels, scores)
    ) / max(1, len(labels))
    return far, frr, attack_success


def evaluate_fsc_sv_attack(
    victim_model,
    evaluator,
    val_dataloader,
    val_pairs,
    test_dataloader,
    test_pairs,
    device,
    uap_delta,
):
    val_clean, _, _ = _collect_fsc_sv_embeddings(
        victim_model, val_dataloader, device, uap_delta=None
    )
    val_labels, val_scores = _scores_for_pairs(evaluator, val_pairs, val_clean)
    val_eer, clean_threshold = calculate_eer(val_labels, val_scores, pos_label=1)

    clean_samples, adv_samples, mean_cosine = _collect_fsc_sv_embeddings(
        victim_model, test_dataloader, device, uap_delta=uap_delta
    )
    labels, clean_scores = _scores_for_pairs(evaluator, test_pairs, clean_samples)
    _, adv_scores = _mixed_scores_for_pairs(
        evaluator, test_pairs, clean_samples, adv_samples
    )
    clean_eer, _ = calculate_eer(labels, clean_scores, pos_label=1)
    adv_eer, _ = calculate_eer(labels, adv_scores, pos_label=1)
    clean_far, clean_frr, _ = _rates_at_threshold(
        labels, clean_scores, clean_threshold
    )
    far, frr, asr_sv = _rates_at_threshold(labels, adv_scores, clean_threshold)

    results = {
        "validation_clean_eer": float(val_eer),
        "validation_clean_threshold": float(clean_threshold),
        "test_clean_eer": float(clean_eer),
        "test_adv_eer": float(adv_eer),
        "test_eer_delta": float(adv_eer - clean_eer),
        "clean_far_at_validation_threshold": float(clean_far),
        "clean_frr_at_validation_threshold": float(clean_frr),
        "far_at_validation_threshold": float(far),
        "frr_at_validation_threshold": float(frr),
        "asr_sv_at_validation_threshold": float(asr_sv),
        "mean_embedding_cosine": None
        if mean_cosine is None
        else float(mean_cosine),
        "num_pairs": len(labels),
    }
    print(
        "[*] FSC SV attack: "
        f"pairs={results['num_pairs']}, val_threshold={clean_threshold:.4f}, "
        f"clean_eer={clean_eer:.4f}, adv_eer={adv_eer:.4f}, "
        f"FAR={far:.4f}, FRR={frr:.4f}, ASR-SV={asr_sv:.4f}"
    )
    return results


def _save_uap(path, uap_delta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(uap_delta.detach().cpu(), str(path))
    print(f"[+] Saved FSC joint UAP: {path}")


@hydra.main(config_path="config", config_name="train_eval")
def main_attack(cfg: DictConfig):
    base_attack_cfg = BTUAPAttackConfig()
    project_root = Path(get_original_cwd())
    base_attack_cfg.resolve_paths(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(_cfg_get(cfg, "seed", cfg.seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    audio_length = float(_cfg_get(cfg, "audio_length", 3.0))
    sample_rate = int(_cfg_get(cfg, "sample_rate", base_attack_cfg.sample_rate))
    uap_samples = int(round(audio_length * sample_rate))
    if uap_samples <= 0:
        raise ValueError("fsc_joint.audio_length must be positive.")

    output_path = Path(
        str(_cfg_get(cfg, "output_path", project_root / "perturbations" / "original" / "janus_fsc_joint_3s.pt"))
    )
    results_path = Path(
        str(
            _cfg_get(
                cfg,
                "results_path",
                project_root / "results" / "joint" / "training" / "janus_fsc_joint_results.json",
            )
        )
    )
    sv_ckpt = Path(str(_cfg_get(cfg, "sv_ckpt", base_attack_cfg.ckpt_path)))
    intent_head_path = Path(
        str(_cfg_get(cfg, "intent_head_path", base_attack_cfg.intent_head_path))
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
    for required_path in [sv_ckpt, intent_head_path, backbone]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required FSC joint asset: {required_path}")

    train_batch_size = int(
        _cfg_get(cfg, "train_batch_size", base_attack_cfg.sr_train_batch_size)
    )
    max_epochs = int(_cfg_get(cfg, "max_epochs", base_attack_cfg.max_epochs))
    steps_per_epoch = int(
        _cfg_get(cfg, "steps_per_epoch", base_attack_cfg.steps_per_epoch)
    )
    learning_rate = float(_cfg_get(cfg, "learning_rate", 0.01))
    alpha = float(_cfg_get(cfg, "alpha", base_attack_cfg.alpha))
    gamma = float(_cfg_get(cfg, "gamma", base_attack_cfg.gamma))
    epsilon = float(_cfg_get(cfg, "epsilon", base_attack_cfg.epsilon))
    lambda_reg = float(_cfg_get(cfg, "lambda_reg", base_attack_cfg.lambda_reg))
    lambda_smooth = float(
        _cfg_get(cfg, "lambda_smooth", base_attack_cfg.lambda_smooth)
    )
    tau_eval = float(_cfg_get(cfg, "tau_eval", base_attack_cfg.tau_eval))
    beta = float(_cfg_get(cfg, "beta", base_attack_cfg.beta))
    num_enrollment = int(
        _cfg_get(
            cfg, "pseudo_enrollment_speakers", base_attack_cfg.pseudo_enrollment_samples
        )
    )
    num_pairs = int(_cfg_get(cfg, "num_verification_pairs", 5000))
    run_evaluation = bool(_cfg_get(cfg, "run_evaluation", True))
    intent_warmup_steps = int(
        _cfg_get(cfg, "intent_warmup_steps", base_attack_cfg.intent_warmup_steps)
    )
    intent_eval_max_batches = _cfg_get(
        cfg, "intent_eval_max_batches", base_attack_cfg.intent_eval_max_batches
    )
    if intent_eval_max_batches is not None:
        intent_eval_max_batches = int(intent_eval_max_batches)

    print(
        "[*] FSC joint attack config: "
        f"device={device}, uap={audio_length:.2f}s/{uap_samples} samples, "
        f"epochs={max_epochs}, steps_per_epoch={steps_per_epoch}, "
        f"batch_size={train_batch_size}, alpha={alpha}, gamma={gamma}, "
        f"epsilon={epsilon}, output={output_path}"
    )

    sr_cfg = make_speech_recognition_cfg(
        cfg,
        train_max_num_samples=base_attack_cfg.sr_train_max_num_samples,
        train_batch_size=train_batch_size,
        project_root=project_root,
    )
    sr_cfg.network.wav2vec_hunggingface_id = str(backbone)
    sr_cfg.tokenizer.tokenizer_huggingface_id = str(backbone)
    sr_dm = construct_data_module(sr_cfg)
    train_dataloader = sr_dm.train_dataloader()
    val_dataloader = sr_dm.val_dataloader()
    test_dataloader = sr_dm.test_dataloader()

    sensitive_labels = [
        label
        for label in base_attack_cfg.sensitive_intent_labels
        if label in sr_dm.intent_label_to_id
    ]
    if not sensitive_labels:
        raise ValueError("No configured sensitive intents exist in FSC.")

    val_pairs = build_fsc_verification_pairs(sr_dm.val_ds, num_pairs, seed + 1)
    test_pairs = build_fsc_verification_pairs(sr_dm.test_ds, num_pairs, seed + 2)
    train_speakers = len(_group_fsc_sample_keys(sr_dm.train_ds))
    if num_enrollment > train_speakers:
        raise ValueError(
            f"pseudo_enrollment_speakers={num_enrollment}, but FSC train has "
            f"only {train_speakers} speakers."
        )

    evaluator = instantiate(cfg.evaluator)
    speaker_metadata = FSCVerificationMetadata(
        num_speakers=train_speakers,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )
    speaker_metadata.summary()

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
        sensitive_keywords=base_attack_cfg.sensitive_keywords,
        warmup_steps=intent_warmup_steps,
        device=device,
        checkpoint_path=intent_head_path,
        lr=base_attack_cfg.intent_head_lr,
        sensitive_class_weight=base_attack_cfg.sensitive_class_weight,
        eval_interval_steps=base_attack_cfg.intent_eval_interval_steps,
        eval_max_batches=intent_eval_max_batches,
        decision_threshold=base_attack_cfg.intent_decision_threshold,
        num_intents=2,
    )

    pseudo_embeddings, pseudo_speaker_ids = generate_fsc_pseudo_enrollment_set(
        sv_model,
        train_dataloader,
        num_speakers=num_enrollment,
        device=device,
    )
    uap_delta = torch.zeros(
        (1, uap_samples), dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([uap_delta], lr=learning_rate)
    sr_loss_window = deque(maxlen=base_attack_cfg.sr_loss_moving_average_window)
    global_step = 0

    for epoch in range(max_epochs):
        train_iterator = iter(train_dataloader)
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"FSC joint epoch {epoch + 1}/{max_epochs}",
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

            batch_speaker_ids = _speaker_ids_from_batch(batch)
            sv_embeddings = sv_model.compute_speaker_embedding(adv_input)
            sv_loss, crossing_probability, tau_cosine = compute_fsc_sv_risk_loss(
                sv_embeddings,
                batch_speaker_ids,
                pseudo_embeddings,
                pseudo_speaker_ids,
                tau_eval,
                beta,
            )
            sr_targets = extract_sensitive_binary_targets(
                batch,
                sensitive_intent_labels=sensitive_labels,
                sensitive_keywords=base_attack_cfg.sensitive_keywords,
                device=device,
            )
            benign_mask = sr_targets == 0
            sr_loss, sensitive_probability = compute_sr_sensitive_intent_loss(
                sr_model,
                intent_head,
                adv_input,
                batch.input_lengths,
                sensitive_class_ids=[1],
                benign_mask=benign_mask,
            )

            energy_loss = uap_delta.pow(2).mean()
            smooth_loss = (uap_delta[:, 1:] - uap_delta[:, :-1]).pow(2).mean()
            total_loss = (
                alpha * sv_loss
                + gamma * sr_loss
                + lambda_reg * energy_loss
                + lambda_smooth * smooth_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                uap_delta.clamp_(-epsilon, epsilon)

            if benign_mask.any():
                sr_loss_window.append(float(sr_loss.detach()))
            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{float(total_loss.detach()):.4f}",
                    "sv": f"{float(sv_loss.detach()):.4f}",
                    "sr": f"{float(sr_loss.detach()):.4f}",
                    "p_cross": f"{crossing_probability:.3f}",
                    "p_sens": f"{sensitive_probability:.3f}",
                    "tau_cos": f"{tau_cosine:.3f}",
                }
            )

        epoch_path = output_path.with_name(
            f"{output_path.stem}.epoch{epoch + 1}{output_path.suffix}"
        )
        _save_uap(epoch_path, uap_delta)

    _save_uap(output_path, uap_delta)

    results = None
    if run_evaluation:
        sr_results = evaluate_sr_attack(
            sr_model=sr_model,
            intent_head=intent_head,
            dataloader=test_dataloader,
            sensitive_intent_labels=sensitive_labels,
            sensitive_keywords=base_attack_cfg.sensitive_keywords,
            sensitive_class_ids=[1],
            device=device,
            uap_delta=uap_delta.detach(),
            decision_threshold=base_attack_cfg.intent_decision_threshold,
            max_batches=None,
        )
        sv_results = evaluate_fsc_sv_attack(
            victim_model=sv_model,
            evaluator=evaluator,
            val_dataloader=val_dataloader,
            val_pairs=val_pairs,
            test_dataloader=test_dataloader,
            test_pairs=test_pairs,
            device=device,
            uap_delta=uap_delta.detach(),
        )
        risk_overall = 0.5 * sv_results["asr_sv_at_validation_threshold"] + 0.5 * sr_results["conditional_flip_rate"]
        results = {
            "config": {
                "seed": seed,
                "audio_length": audio_length,
                "sample_rate": sample_rate,
                "uap_samples": uap_samples,
                "max_epochs": max_epochs,
                "steps_per_epoch": steps_per_epoch,
                "train_batch_size": train_batch_size,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "num_verification_pairs": num_pairs,
                "pseudo_enrollment_speakers": num_enrollment,
                "sv_ckpt": str(sv_ckpt),
                "intent_head_path": str(intent_head_path),
                "backbone": str(backbone),
                "uap_path": str(output_path),
            },
            "sv": sv_results,
            "sr": sr_results,
            "combined": {"risk_overall": float(risk_overall)},
        }
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"[+] Saved FSC joint results: {results_path}")
        print(f"[*] FSC joint Risk_overall={risk_overall:.4f}")

    return results


if __name__ == "__main__":
    main_attack()
