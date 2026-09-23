from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .common import apply_uap, load_config_fragment, normalize_batch_like_pipeline


def make_speech_recognition_cfg(
    base_cfg: DictConfig,
    train_max_num_samples: int,
    train_batch_size: int,
    project_root: Path,
) -> DictConfig:
    project_root = Path(project_root)
    sr_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    sr_cfg.data.module = load_config_fragment("data/module/fluent_speech_commands.yaml")
    sr_cfg.data.pipeline = load_config_fragment("data/pipeline/wav2vec_full_seq_pipeline.yaml")
    sr_cfg.data.dataloader = load_config_fragment("data/dataloader/speech.yaml")
    sr_cfg.data.shards = load_config_fragment("data/shards/shards_librispeech.yaml")
    sr_cfg.network = load_config_fragment("network/wav2vec2_fc_letter.yaml")
    sr_cfg.optim.loss = load_config_fragment("optim/loss/ctc.yaml")
    sr_cfg.load_network_from_checkpoint = None
    sr_cfg.data.dataloader.train_batch_size = train_batch_size
    sr_cfg.data.dataloader.train_max_num_samples = train_max_num_samples
    sr_cfg.data.module.dataset_folder = str(project_root / "fluent_speech_commands_dataset")
    sr_cfg.data.module.train_csv = str(
        project_root / "fluent_speech_commands_dataset" / "data" / "train_data.csv"
    )
    sr_cfg.data.module.val_csv = str(
        project_root / "fluent_speech_commands_dataset" / "data" / "valid_data.csv"
    )
    sr_cfg.data.module.test_csv = str(
        project_root / "fluent_speech_commands_dataset" / "data" / "test_data.csv"
    )

    local_wav2vec2_ctc = project_root / "model" / "wav2vec2-base-960h"
    if local_wav2vec2_ctc.exists():
        sr_cfg.network.wav2vec_hunggingface_id = str(local_wav2vec2_ctc)
        sr_cfg.tokenizer.tokenizer_huggingface_id = str(local_wav2vec2_ctc)

    return sr_cfg


class SimpleSensitiveIntentHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_intents: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_intents),
        )

    @staticmethod
    def masked_mean_pool(hidden_states: torch.Tensor, lengths) -> torch.Tensor:
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to(hidden_states.device)
        else:
            lengths = torch.as_tensor(lengths, device=hidden_states.device)

        max_len = hidden_states.shape[1]
        time_index = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)

        pooled = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def forward(self, hidden_states: torch.Tensor, lengths) -> torch.Tensor:
        pooled = self.masked_mean_pool(hidden_states, lengths)
        return self.classifier(pooled)


def load_sensitive_intent_head_checkpoint(
    checkpoint_path,
    sensitive_intent_labels,
    decision_threshold,
    device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing sensitive intent head checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Intent head checkpoint predates the binary-head format. "
            "Train and save a binary sensitive intent head before evaluating a UAP."
        )

    expected_selection_mode = "best_validation_sens_f1_fixed_threshold"
    if (
        checkpoint.get("num_intents") != 2
        or checkpoint.get("label_mode") != "binary_sensitive_intent"
        or checkpoint.get("selection_mode") != expected_selection_mode
        or checkpoint.get("sensitive_intent_labels") != list(sensitive_intent_labels)
        or checkpoint.get("decision_threshold") != decision_threshold
    ):
        raise ValueError(
            "Intent head checkpoint does not match the current binary-sensitive "
            "label set, best-head policy, or fixed decision threshold."
        )

    intent_head = SimpleSensitiveIntentHead(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_intents=checkpoint["num_intents"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    intent_head.load_state_dict(checkpoint["model_state_dict"])
    intent_head.decision_threshold = decision_threshold
    for param in intent_head.parameters():
        param.requires_grad = False
    intent_head.eval()

    print(f"[*] Loaded sensitive intent head from {checkpoint_path}")
    return intent_head


def infer_sensitive_intent_labels(transcripts, sensitive_keywords):
    labels = []

    for text in transcripts:
        text_lower = text.lower()
        is_sensitive = any(keyword in text_lower for keyword in sensitive_keywords)
        labels.append(1 if is_sensitive else 0)

    return torch.tensor(labels, dtype=torch.long)


def extract_sensitive_binary_targets(
    batch,
    sensitive_intent_labels: Sequence[str],
    sensitive_keywords,
    device,
) -> torch.Tensor:
    if batch.side_info is None:
        return infer_sensitive_intent_labels(
            batch.ground_truth_strings,
            sensitive_keywords,
        ).to(device)

    sensitive_labels = set(sensitive_intent_labels)
    labels = []
    for key in batch.keys:
        side_info = batch.side_info.get(key)
        if side_info is None or side_info.meta is None:
            return infer_sensitive_intent_labels(
                batch.ground_truth_strings,
                sensitive_keywords,
            ).to(device)
        intent_label = side_info.meta.get("intent_label")
        if intent_label is None:
            return infer_sensitive_intent_labels(
                batch.ground_truth_strings,
                sensitive_keywords,
            ).to(device)
        labels.append(1 if intent_label in sensitive_labels else 0)

    return torch.tensor(labels, dtype=torch.long, device=device)


def extract_raw_audio_batch(batch, device) -> torch.Tensor:
    raw_waveforms = []
    for key in batch.keys:
        side_info = batch.side_info.get(key)
        if side_info is None or side_info.original_tensor is None:
            raise ValueError(
                "FSC raw waveforms are required to apply the SR UAP before normalization."
            )
        raw_waveforms.append(side_info.original_tensor.squeeze().to(device))

    return pad_sequence(raw_waveforms, batch_first=True, padding_value=0.0)


def train_sensitive_intent_head(
    sr_model,
    intent_head,
    dataloader,
    eval_dataloaders,
    sensitive_intent_labels,
    sensitive_keywords,
    device,
    warmup_steps: int = 1000,
    lr: float = 1e-3,
    sensitive_class_weight: float = 1.0,
    eval_interval_steps: int = 500,
    eval_max_batches: int = None,
    decision_threshold: float = 0.35,
    num_intents: int = 2,
):
    if num_intents != 2:
        raise ValueError("Sensitive intent head must be binary: benign=0, sensitive=1.")

    print(f"[*] Warm-starting sensitive intent head for {warmup_steps} steps...")

    sr_model.eval()
    intent_head.train()
    optimizer = torch.optim.Adam(intent_head.parameters(), lr=lr)
    class_weight = torch.tensor(
        [1.0, sensitive_class_weight],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    num_updates = 0
    skipped_batches = 0
    total_seen_batches = 0
    total_negative_labels = 0
    total_sensitive_labels = 0
    best_metrics = None
    best_state_dict = None
    pbar = tqdm(total=warmup_steps, desc="Intent warm-up", leave=False)
    data_iter = iter(dataloader)

    while num_updates < warmup_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        total_seen_batches += 1

        batch = batch.to(device)
        target = extract_sensitive_binary_targets(
            batch=batch,
            sensitive_intent_labels=sensitive_intent_labels,
            sensitive_keywords=sensitive_keywords,
            device=device,
        )
        num_negative = int((target == 0).sum().item())
        num_sensitive = int((target == 1).sum().item())
        total_negative_labels += num_negative
        total_sensitive_labels += num_sensitive

        with torch.no_grad():
            sr_embedding, sr_embedding_lengths = sr_model.compute_speech_embedding(
                batch.network_input,
                batch.input_lengths,
            )

        logits = intent_head(sr_embedding.detach(), sr_embedding_lengths)
        loss = loss_fn(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_updates += 1
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "neg": num_negative,
            "sens": num_sensitive,
            "skipped": skipped_batches,
        })

        should_evaluate = (
            num_updates == warmup_steps
            or (eval_interval_steps > 0 and num_updates % eval_interval_steps == 0)
        )
        if should_evaluate:
            print(f"\n[*] Intent validation at update {num_updates}...")
            metrics = evaluate_sensitive_intent_head(
                sr_model=sr_model,
                intent_head=intent_head,
                dataloaders=eval_dataloaders,
                sensitive_intent_labels=sensitive_intent_labels,
                sensitive_keywords=sensitive_keywords,
                device=device,
                max_batches=eval_max_batches,
                num_intents=num_intents,
                decision_threshold=decision_threshold,
            )
            if best_metrics is None or metrics["sens_f1"] > best_metrics["sens_f1"]:
                best_metrics = metrics
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in intent_head.state_dict().items()
                }
                print(
                    "[+] New best intent head: "
                    f"update={num_updates}, "
                    f"sens_f1={metrics['sens_f1']:.4f}, "
                    f"threshold={metrics['decision_threshold']:.2f}"
                )
            intent_head.train()

    pbar.close()
    print(
        "[*] Intent label stats: "
        f"batches_seen={total_seen_batches}, "
        f"updates={num_updates}, "
        f"skipped={skipped_batches}, "
        f"neg_labels={total_negative_labels}, "
        f"sensitive_labels={total_sensitive_labels}"
    )
    if best_state_dict is not None:
        intent_head.load_state_dict(best_state_dict)
        intent_head.decision_threshold = best_metrics["decision_threshold"]
        print(
            "[+] Restored best sensitive intent head: "
            f"sens_f1={best_metrics['sens_f1']:.4f}, "
            f"threshold={best_metrics['decision_threshold']:.2f}"
        )
    intent_head.eval()
    print(f"[+] Sensitive intent head warm-start complete with {num_updates} updates.")
    return best_metrics


def _binary_sensitive_metrics(
    target: torch.Tensor,
    p_sensitive: torch.Tensor,
    threshold: float,
):
    prediction = (p_sensitive >= threshold).long()
    neg_mask = target == 0
    sens_mask = target == 1
    pred_sens_mask = prediction == 1

    total_negative = int(neg_mask.sum().item())
    total_sensitive = int(sens_mask.sum().item())
    true_negative = int(((prediction == 0) & neg_mask).sum().item())
    true_sensitive = int((pred_sens_mask & sens_mask).sum().item())
    false_sensitive = int((pred_sens_mask & neg_mask).sum().item())
    false_benign = int(((prediction == 0) & sens_mask).sum().item())
    total_samples = target.numel()

    accuracy = (true_negative + true_sensitive) / max(1, total_samples)
    negative_acc = true_negative / max(1, total_negative)
    sensitive_recall = true_sensitive / max(1, total_sensitive)
    sensitive_precision = true_sensitive / max(1, true_sensitive + false_sensitive)
    sensitive_f1 = (
        2 * sensitive_precision * sensitive_recall
        / max(1e-8, sensitive_precision + sensitive_recall)
    )
    balanced_acc = (negative_acc + sensitive_recall) / 2

    return {
        "samples": total_samples,
        "acc": accuracy,
        "balanced_acc": balanced_acc,
        "neg_acc": negative_acc,
        "sens_acc": sensitive_recall,
        "sens_precision": sensitive_precision,
        "sens_recall": sensitive_recall,
        "sens_f1": sensitive_f1,
        "neg": total_negative,
        "sens": total_sensitive,
        "tp": true_sensitive,
        "fp": false_sensitive,
        "fn": false_benign,
        "tn": true_negative,
        "decision_threshold": float(threshold),
    }


def evaluate_sensitive_intent_head(
    sr_model,
    intent_head,
    dataloaders,
    sensitive_intent_labels,
    sensitive_keywords,
    device,
    max_batches: int = None,
    num_intents: int = 2,
    decision_threshold: float = 0.35,
):
    if num_intents != 2:
        raise ValueError("Sensitive intent head must be binary: benign=0, sensitive=1.")

    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]

    sr_model.eval()
    intent_head.eval()

    batches_seen = 0
    targets = []
    probabilities = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:
                if max_batches is not None and batches_seen >= max_batches:
                    break

                batch = batch.to(device)
                target = extract_sensitive_binary_targets(
                    batch=batch,
                    sensitive_intent_labels=sensitive_intent_labels,
                    sensitive_keywords=sensitive_keywords,
                    device=device,
                )

                sr_embedding, sr_embedding_lengths = sr_model.compute_speech_embedding(
                    batch.network_input,
                    batch.input_lengths,
                )
                logits = intent_head(sr_embedding, sr_embedding_lengths)
                p_sensitive = torch.softmax(logits, dim=-1)[:, 1]
                targets.append(target.detach().cpu())
                probabilities.append(p_sensitive.detach().cpu())
                batches_seen += 1

            if max_batches is not None and batches_seen >= max_batches:
                break

    if not targets:
        raise ValueError("No samples available for sensitive intent evaluation.")

    target = torch.cat(targets)
    p_sensitive = torch.cat(probabilities)
    results = _binary_sensitive_metrics(target, p_sensitive, decision_threshold)
    results["batches"] = batches_seen
    intent_head.decision_threshold = results["decision_threshold"]

    print(
        "[*] Intent eval: "
        f"batches={batches_seen}, "
        f"samples={results['samples']}, "
        f"threshold={results['decision_threshold']:.2f}, "
        f"acc={results['acc']:.4f}, "
        f"balanced_acc={results['balanced_acc']:.4f}, "
        f"neg_acc={results['neg_acc']:.4f}, "
        f"sens_acc={results['sens_acc']:.4f}, "
        f"sens_precision={results['sens_precision']:.4f}, "
        f"sens_recall={results['sens_recall']:.4f}, "
        f"sens_f1={results['sens_f1']:.4f}, "
        f"neg={results['neg']}, "
        f"sens={results['sens']}"
    )
    return results


def initialize_intent_head(
    sr_model,
    sr_train_dataloader,
    sr_eval_dataloaders,
    sensitive_intent_labels,
    sensitive_keywords,
    warmup_steps,
    device,
    checkpoint_path=None,
    lr: float = 1e-3,
    sensitive_class_weight: float = 1.0,
    eval_interval_steps: int = 500,
    eval_max_batches: int = None,
    decision_threshold: float = 0.35,
    num_intents: int = 2,
):
    if num_intents != 2:
        raise ValueError("Sensitive intent head must be binary: benign=0, sensitive=1.")

    sample_sr_batch = next(iter(sr_train_dataloader)).to(device)
    with torch.no_grad():
        sample_sr_embedding, _ = sr_model.compute_speech_embedding(
            sample_sr_batch.network_input,
            sample_sr_batch.input_lengths,
        )

    intent_head = SimpleSensitiveIntentHead(
        input_dim=sample_sr_embedding.shape[-1],
        hidden_dim=256,
        num_intents=num_intents,
        dropout=0.1,
    ).to(device)
    checkpoint_payload = {
        "input_dim": sample_sr_embedding.shape[-1], 
        "hidden_dim": 256,
        "num_intents": num_intents,
        "dropout": 0.1,
        "label_mode": "binary_sensitive_intent",
        "selection_mode": "best_validation_sens_f1_fixed_threshold",
        "sensitive_intent_labels": list(sensitive_intent_labels),
        "sensitive_keywords": list(sensitive_keywords),
        "decision_threshold": decision_threshold,
    }

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"[*] Loading sensitive intent head from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "model_state_dict" in checkpoint:
                saved_input_dim = checkpoint.get("input_dim", checkpoint_payload["input_dim"])
                saved_hidden_dim = checkpoint.get("hidden_dim", checkpoint_payload["hidden_dim"])
                saved_num_intents = checkpoint.get("num_intents", checkpoint_payload["num_intents"])
                saved_dropout = checkpoint.get("dropout", checkpoint_payload["dropout"])
                saved_label_mode = checkpoint.get("label_mode")
                saved_selection_mode = checkpoint.get("selection_mode")
                saved_sensitive_labels = checkpoint.get("sensitive_intent_labels")
                saved_decision_threshold = checkpoint.get("decision_threshold")

                if (
                    saved_num_intents == num_intents
                    and saved_label_mode == checkpoint_payload["label_mode"]
                    and saved_selection_mode == checkpoint_payload["selection_mode"]
                    and saved_sensitive_labels == checkpoint_payload["sensitive_intent_labels"]
                    and saved_decision_threshold == decision_threshold
                ):
                    intent_head = SimpleSensitiveIntentHead(
                        input_dim=saved_input_dim,
                        hidden_dim=saved_hidden_dim,
                        num_intents=saved_num_intents,
                        dropout=saved_dropout,
                    ).to(device)
                    intent_head.load_state_dict(checkpoint["model_state_dict"])
                    intent_head.decision_threshold = decision_threshold
                else:
                    print(
                        "[!] Intent head checkpoint label definition mismatch: "
                        f"saved_classes={saved_num_intents}, requested_classes={num_intents}, "
                        f"saved_mode={saved_label_mode}, "
                        f"saved_selection={saved_selection_mode}, "
                        f"saved_threshold={saved_decision_threshold}. "
                        "Retraining intent head."
                    )
                    checkpoint = None
            else:
                print(
                    "[!] Legacy intent head checkpoint has no label definition. "
                    "Retraining binary sensitive intent head."
                )
                checkpoint = None

            if checkpoint is not None:
                for param in intent_head.parameters():
                    param.requires_grad = False

                intent_head.eval()
                evaluate_sensitive_intent_head(
                    sr_model=sr_model,
                    intent_head=intent_head,
                    dataloaders=sr_eval_dataloaders,
                    sensitive_intent_labels=sensitive_intent_labels,
                    sensitive_keywords=sensitive_keywords,
                    device=device,
                    max_batches=eval_max_batches,
                    num_intents=num_intents,
                    decision_threshold=decision_threshold,
                )
                return intent_head

    best_metrics = train_sensitive_intent_head(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloader=sr_train_dataloader,
        eval_dataloaders=sr_eval_dataloaders,
        sensitive_intent_labels=sensitive_intent_labels,
        sensitive_keywords=sensitive_keywords,
        device=device,
        warmup_steps=warmup_steps,
        lr=lr,
        sensitive_class_weight=sensitive_class_weight,
        eval_interval_steps=eval_interval_steps,
        eval_max_batches=eval_max_batches,
        decision_threshold=decision_threshold,
        num_intents=num_intents,
    )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if best_metrics is not None:
            checkpoint_payload["decision_threshold"] = best_metrics["decision_threshold"]
            checkpoint_payload["validation_sens_f1"] = best_metrics["sens_f1"]
        checkpoint_payload["model_state_dict"] = intent_head.state_dict()
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"[+] Best sensitive intent head saved to {checkpoint_path}")

    evaluate_sensitive_intent_head(
        sr_model=sr_model,
        intent_head=intent_head,
        dataloaders=sr_eval_dataloaders,
        sensitive_intent_labels=sensitive_intent_labels,
        sensitive_keywords=sensitive_keywords,
        device=device,
        max_batches=eval_max_batches,
        num_intents=num_intents,
        decision_threshold=decision_threshold,
    )

    for param in intent_head.parameters():
        param.requires_grad = False

    return intent_head 


def compute_sr_sensitive_intent_loss(
    sr_model,
    intent_head,
    x_adv,
    input_lengths,
    sensitive_class_ids,
    benign_mask=None,
):
    sr_embedding, sr_embedding_lengths = sr_model.compute_speech_embedding(
        x_adv,
        input_lengths,
    )
    logits = intent_head(sr_embedding, sr_embedding_lengths)
    probs = torch.softmax(logits, dim=-1)
    p_sens = probs[:, sensitive_class_ids].sum(dim=-1).clamp_min(1e-8)

    if benign_mask is not None:
        benign_mask = benign_mask.to(device=p_sens.device, dtype=torch.bool)
        p_sens = p_sens[benign_mask]

    if p_sens.numel() == 0:
        return logits.sum() * 0.0, float("nan")

    loss = -torch.log(p_sens).mean()

    return loss, p_sens.mean().detach().item()


def evaluate_sr_attack(
    sr_model,
    intent_head,
    dataloader,
    sensitive_intent_labels,
    sensitive_keywords,
    sensitive_class_ids,
    device,
    uap_delta,
    decision_threshold: float,
    max_batches: int = None,
):
    sr_model.eval()
    intent_head.eval()
    clean_probabilities = []
    adv_probabilities = []
    batches_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and batches_seen >= max_batches:
                break

            batch = batch.to(device)
            target = extract_sensitive_binary_targets(
                batch=batch,
                sensitive_intent_labels=sensitive_intent_labels,
                sensitive_keywords=sensitive_keywords,
                device=device,
            )
            benign_mask = target == 0
            batches_seen += 1
            if not benign_mask.any():
                continue

            clean_embedding, clean_lengths = sr_model.compute_speech_embedding(
                batch.network_input,
                batch.input_lengths,
            )
            clean_logits = intent_head(clean_embedding, clean_lengths)
            clean_p_sens = torch.softmax(clean_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            raw_audio = extract_raw_audio_batch(batch, device)
            adv_raw = apply_uap(raw_audio, uap_delta)
            adv_input = normalize_batch_like_pipeline(
                adv_raw,
                input_lengths=batch.input_lengths,
            )
            adv_embedding, adv_lengths = sr_model.compute_speech_embedding(
                adv_input,
                batch.input_lengths,
            )
            adv_logits = intent_head(adv_embedding, adv_lengths)
            adv_p_sens = torch.softmax(adv_logits, dim=-1)[
                :, sensitive_class_ids
            ].sum(dim=-1)

            clean_probabilities.append(clean_p_sens[benign_mask].cpu())
            adv_probabilities.append(adv_p_sens[benign_mask].cpu())

    if not clean_probabilities:
        raise ValueError("No benign samples available for SR attack evaluation.")

    p_clean = torch.cat(clean_probabilities)
    p_adv = torch.cat(adv_probabilities)
    clean_sensitive = p_clean >= decision_threshold
    adv_sensitive = p_adv >= decision_threshold
    eligible = ~clean_sensitive
    flipped = eligible & adv_sensitive

    results = {
        "batches": batches_seen,
        "benign_samples": int(p_clean.numel()),
        "clean_mean_p_sens": p_clean.mean().item(),
        "adv_mean_p_sens": p_adv.mean().item(),
        "delta_mean_p_sens": (p_adv - p_clean).mean().item(),
        "clean_sensitive_rate": clean_sensitive.float().mean().item(),
        "adv_sensitive_rate": adv_sensitive.float().mean().item(),
        "rate_increase": (
            adv_sensitive.float().mean() - clean_sensitive.float().mean()
        ).item(),
        "flip_rate": flipped.float().mean().item(),
        "conditional_flip_rate": (
            flipped.sum().item() / max(1, eligible.sum().item())
        ),
        "eligible_clean_benign": int(eligible.sum().item()),
        "uap_linf": uap_delta.detach().abs().max().item(),
        "uap_rms": uap_delta.detach().pow(2).mean().sqrt().item(),
    }

    print(
        "[*] SR attack eval on true benign: "
        f"batches={results['batches']}, "
        f"samples={results['benign_samples']}, "
        f"threshold={decision_threshold:.2f}, "
        f"p_clean={results['clean_mean_p_sens']:.4f}, "
        f"p_adv={results['adv_mean_p_sens']:.4f}, "
        f"delta_p={results['delta_mean_p_sens']:+.4f}, "
        f"clean_rate={results['clean_sensitive_rate']:.4f}, "
        f"adv_rate={results['adv_sensitive_rate']:.4f}, "
        f"rate_delta={results['rate_increase']:+.4f}, "
        f"flip={results['flip_rate']:.4f}, "
        f"cond_flip={results['conditional_flip_rate']:.4f}, "
        f"linf={results['uap_linf']:.5f}, "
        f"rms={results['uap_rms']:.5f}"
    )

    return results
