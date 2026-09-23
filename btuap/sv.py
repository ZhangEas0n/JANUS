import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.eval_metrics import calculate_eer
from src.evaluation.speaker.speaker_recognition_evaluator import EmbeddingSample

from .common import apply_uap, normalize_batch_like_pipeline


def evaluator_score_to_cosine_threshold(tau_eval: float) -> float:
    if not (0.0 <= tau_eval <= 1.0):
        raise ValueError(f"Expected evaluator-space threshold in [0,1], got {tau_eval}")
    return 2.0 * tau_eval - 1.0


def generate_pseudo_enrollment_set(victim_model, dataloader, num_samples=50, device=None):
    print(f"[*] Extracting {num_samples} unique-speaker pseudo-enrollment features as targets...")
    victim_model.eval()
    pseudo_embs = []
    collected = 0
    seen_speakers = set()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Pseudo-enrollment", leave=False)
        for batch in dataloader:
            batch = batch.to(device)
            x_clean = normalize_batch_like_pipeline(batch.network_input)
            embs = victim_model.compute_speaker_embedding(x_clean)

            speaker_ids = batch.ground_truth
            if len(speaker_ids.shape) == 0:
                speaker_ids = speaker_ids.unsqueeze(0)
            if len(embs.shape) == 1:
                embs = embs.unsqueeze(0)

            new_embeddings = []
            for speaker_id, emb in zip(speaker_ids.detach().cpu().tolist(), embs):
                if speaker_id in seen_speakers:
                    continue

                seen_speakers.add(speaker_id)
                new_embeddings.append(emb.unsqueeze(0))

                if len(seen_speakers) >= num_samples:
                    break

            if len(new_embeddings) > 0:
                batch_new_embs = torch.cat(new_embeddings, dim=0)
                pseudo_embs.append(batch_new_embs)
                new_collected = min(num_samples, len(seen_speakers))
                pbar.update(new_collected - collected)
                collected = new_collected

            if collected >= num_samples:
                break
        pbar.close()

    pseudo_embs = torch.cat(pseudo_embs, dim=0)[:num_samples]
    pseudo_embs = F.normalize(pseudo_embs, p=2, dim=-1)
    print("[+] Pseudo-enrollment set generated successfully!")
    return pseudo_embs


def compute_sv_risk_loss(adv_embs, pseudo_embs, tau_eval, beta):
    adv_embs_norm = F.normalize(adv_embs, p=2, dim=-1)
    s_matrix = torch.matmul(adv_embs_norm, pseudo_embs.T)
    tau_cosine = evaluator_score_to_cosine_threshold(tau_eval)
    crossing_prob = torch.sigmoid(beta * (s_matrix - tau_cosine))
    per_sample_prob_any = 1.0 - torch.prod(1.0 - crossing_prob, dim=1)
    loss_sv_risk = -torch.mean(per_sample_prob_any)

    return loss_sv_risk, torch.mean(per_sample_prob_any).item(), tau_cosine


def _collect_sv_embeddings(
    victim_model,
    dataloader,
    device,
    uap_delta=None,
    max_batches=None,
):
    samples = []
    cosine_sum = 0.0
    cosine_count = 0

    with torch.no_grad():
        pbar = tqdm(desc="SV eval embeddings", leave=False)
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = batch.to(device)
            x_raw = batch.network_input
            x_eval_raw = apply_uap(x_raw, uap_delta) if uap_delta is not None else x_raw
            x_eval = normalize_batch_like_pipeline(x_eval_raw)
            embedding = victim_model.compute_speaker_embedding(x_eval)
            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)

            if uap_delta is not None:
                x_clean = normalize_batch_like_pipeline(x_raw)
                clean_embedding = victim_model.compute_speaker_embedding(x_clean)
                if len(clean_embedding.shape) == 1:
                    clean_embedding = clean_embedding.unsqueeze(0)
                cosine = F.cosine_similarity(clean_embedding, embedding, dim=-1)
                cosine_sum += cosine.sum().item()
                cosine_count += cosine.numel()

            for idx, sample_id in enumerate(batch.keys):
                samples.append(
                    EmbeddingSample(
                        sample_id=sample_id,
                        embedding=embedding[idx].detach().cpu(),
                    )
                )

            pbar.update(1)
        pbar.close()

    mean_embedding_cosine = (
        cosine_sum / cosine_count if cosine_count > 0 else None
    )

    return samples, mean_embedding_cosine


def _compute_scores_for_pairs(evaluator, pairs, samples):
    sample_map = {sample.sample_id: sample for sample in samples}
    ground_truth = []
    prediction_pairs = []

    for pair in pairs:
        if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
            continue

        ground_truth.append(1 if pair.same_speaker else 0)
        prediction_pairs.append((sample_map[pair.sample1_id], sample_map[pair.sample2_id]))

    raw_scores = evaluator._compute_prediction_scores(prediction_pairs)
    scores = torch.tensor(raw_scores, dtype=torch.float32)
    scores = torch.clamp((scores + 1.0) / 2.0, 0.0, 1.0)

    return ground_truth, scores.tolist()


def evaluate_sv_attack(
    victim_model,
    evaluator,
    dataloader,
    pairs,
    device,
    uap_delta=None,
    max_batches=None,
):
    victim_model.eval()

    clean_samples, _ = _collect_sv_embeddings(
        victim_model=victim_model,
        dataloader=dataloader,
        device=device,
        uap_delta=None,
        max_batches=max_batches,
    )
    adv_samples, mean_embedding_cosine = _collect_sv_embeddings(
        victim_model=victim_model,
        dataloader=dataloader,
        device=device,
        uap_delta=uap_delta,
        max_batches=max_batches,
    )

    ground_truth, clean_scores = _compute_scores_for_pairs(evaluator, pairs, clean_samples)
    _, adv_scores = _compute_scores_for_pairs(evaluator, pairs, adv_samples)

    clean_eer, clean_threshold = calculate_eer(ground_truth, clean_scores, pos_label=1)
    adv_eer, adv_threshold = calculate_eer(ground_truth, adv_scores, pos_label=1)

    negative_scores = [score for gt, score in zip(ground_truth, adv_scores) if gt == 0]
    positive_scores = [score for gt, score in zip(ground_truth, adv_scores) if gt == 1]

    far = sum(score >= clean_threshold for score in negative_scores) / max(1, len(negative_scores))
    frr = sum(score < clean_threshold for score in positive_scores) / max(1, len(positive_scores))
    asr_sv = sum(
        ((gt == 0) and (score >= clean_threshold)) or ((gt == 1) and (score < clean_threshold))
        for gt, score in zip(ground_truth, adv_scores)
    ) / max(1, len(ground_truth))

    cosine_similarity_drop = (
        1.0 - mean_embedding_cosine if mean_embedding_cosine is not None else None
    )

    results = {
        "clean_eer": clean_eer,
        "clean_threshold": clean_threshold,
        "adv_eer": adv_eer,
        "adv_threshold": adv_threshold,
        "eer_delta": adv_eer - clean_eer,
        "far_at_clean_threshold": far,
        "frr_at_clean_threshold": frr,
        "asr_sv_at_clean_threshold": asr_sv,
        "mean_embedding_cosine": mean_embedding_cosine,
        "cosine_similarity_drop": cosine_similarity_drop,
        "num_pairs": len(ground_truth),
    }

    print(
        "[*] SV eval: "
        f"pairs={results['num_pairs']}, "
        f"clean_eer={results['clean_eer']:.4f}, "
        f"adv_eer={results['adv_eer']:.4f}, "
        f"eer_delta={results['eer_delta']:+.4f}, "
        f"far={results['far_at_clean_threshold']:.4f}, "
        f"frr={results['frr_at_clean_threshold']:.4f}, "
        f"asr_sv={results['asr_sv_at_clean_threshold']:.4f}, "
        f"mean_cos={results['mean_embedding_cosine']:.4f}, "
        f"cos_drop={results['cosine_similarity_drop']:.4f}"
    )

    return results
