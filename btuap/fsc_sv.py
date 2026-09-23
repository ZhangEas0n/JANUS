"""FSC-specific speaker-verification helpers for joint UAP training.

The functions in this module keep training in the same scaled-cosine score
space used by ``scripts/eval/evaluate_fsc_joint.py``.  In contrast to the legacy
any-enrollment objective, every selected different-speaker pair contributes
directly to the loss, so the optimization target matches FAR evaluation.
"""

from __future__ import print_function

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FSCEnrollmentGallery:
    """Clean enrollment prototypes and their FSC speaker identities."""

    embeddings: torch.Tensor
    speaker_ids: Tuple[str, ...]
    sample_ids: FrozenSet[str]
    enrollment_k: int

    def __post_init__(self):
        if self.embeddings.dim() != 2:
            raise ValueError(
                "Gallery embeddings must have shape [speakers, dimension], got {}".format(
                    tuple(self.embeddings.shape)
                )
            )
        if self.embeddings.shape[0] != len(self.speaker_ids):
            raise ValueError("Gallery embedding and speaker counts do not match.")
        if len(set(self.speaker_ids)) != len(self.speaker_ids):
            raise ValueError("Gallery speaker IDs must be unique.")
        if self.enrollment_k <= 0:
            raise ValueError("enrollment_k must be positive.")

    def to(self, device) -> "FSCEnrollmentGallery":
        return FSCEnrollmentGallery(
            embeddings=self.embeddings.to(device),
            speaker_ids=self.speaker_ids,
            sample_ids=self.sample_ids,
            enrollment_k=self.enrollment_k,
        )


@dataclass
class FSCSVLossResult:
    """Differentiable loss plus detached diagnostics for one training batch."""

    loss: torch.Tensor
    selected_scores: torch.Tensor
    target_indices: torch.Tensor
    hard_target_mask: torch.Tensor
    metrics: Dict[str, float]


def extract_fsc_speaker_ids(batch) -> List[str]:
    """Read speaker IDs from an FSC speech-recognition batch."""

    if batch.side_info is None:
        raise ValueError("FSC batches require side_info to recover speaker IDs.")

    speaker_ids = []
    for key in batch.keys:
        side_info = batch.side_info.get(key)
        if side_info is None or side_info.meta is None:
            raise ValueError("Missing FSC side_info for sample {}.".format(key))
        speaker_id = side_info.meta.get("speaker_id")
        if speaker_id is None or not str(speaker_id).strip():
            raise ValueError("Missing FSC speaker_id for sample {}.".format(key))
        speaker_ids.append(str(speaker_id).strip())
    return speaker_ids


def scaled_cosine_scores(
    query_embeddings: torch.Tensor,
    enrollment_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return cosine scores scaled from [-1, 1] to evaluator space [0, 1]."""

    if query_embeddings.dim() == 1:
        query_embeddings = query_embeddings.unsqueeze(0)
    if enrollment_embeddings.dim() == 1:
        enrollment_embeddings = enrollment_embeddings.unsqueeze(0)
    if query_embeddings.dim() != 2 or enrollment_embeddings.dim() != 2:
        raise ValueError("SV embeddings must be rank-two tensors.")
    if query_embeddings.shape[1] != enrollment_embeddings.shape[1]:
        raise ValueError("Query and enrollment embedding dimensions do not match.")

    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    enrollment_embeddings = F.normalize(enrollment_embeddings, p=2, dim=-1)
    cosine = torch.matmul(query_embeddings, enrollment_embeddings.transpose(0, 1))
    return torch.clamp((cosine + 1.0) / 2.0, 0.0, 1.0)


def build_fsc_enrollment_gallery(
    victim_model,
    dataloader,
    enrollment_k: int = 3,
    num_speakers: Optional[int] = None,
    device=None,
) -> FSCEnrollmentGallery:
    """Build one normalized clean prototype per complete FSC speaker.

    When ``num_speakers`` is omitted, all speakers with at least
    ``enrollment_k`` usable samples are retained.  This avoids failing when a
    malformed or filtered FSC row makes the dataset speaker count differ from
    the number of complete enrollment identities.
    """

    if enrollment_k <= 0:
        raise ValueError("enrollment_k must be positive.")
    if num_speakers is not None and num_speakers <= 0:
        raise ValueError("num_speakers must be positive when provided.")
    if device is None:
        try:
            device = next(victim_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    from tqdm import tqdm

    from btuap.common import normalize_batch_like_pipeline
    from btuap.sr import extract_raw_audio_batch

    embeddings_by_speaker = {}
    sample_ids_by_speaker = {}
    was_training = victim_model.training
    victim_model.eval()

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="FSC enrollment gallery", leave=False):
                batch = batch.to(device)
                raw_audio = extract_raw_audio_batch(batch, device)
                clean_input = normalize_batch_like_pipeline(
                    raw_audio, input_lengths=batch.input_lengths
                )
                embeddings = victim_model.compute_speaker_embedding(clean_input)
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)

                speaker_ids = extract_fsc_speaker_ids(batch)
                for sample_id, speaker_id, embedding in zip(
                    batch.keys, speaker_ids, embeddings
                ):
                    values = embeddings_by_speaker.setdefault(speaker_id, [])
                    if len(values) >= enrollment_k:
                        continue
                    values.append(embedding.detach())
                    sample_ids_by_speaker.setdefault(speaker_id, []).append(
                        str(sample_id)
                    )
    finally:
        if was_training:
            victim_model.train()

    complete_speakers = sorted(
        speaker_id
        for speaker_id, values in embeddings_by_speaker.items()
        if len(values) >= enrollment_k
    )
    if not complete_speakers:
        raise ValueError(
            "No FSC speaker has at least {} enrollment samples.".format(enrollment_k)
        )
    if num_speakers is not None:
        if len(complete_speakers) < num_speakers:
            raise ValueError(
                "Requested {} FSC speakers with K={}, but found {} complete speakers.".format(
                    num_speakers, enrollment_k, len(complete_speakers)
                )
            )
        complete_speakers = complete_speakers[:num_speakers]

    prototypes = []
    enrollment_sample_ids = set()
    for speaker_id in complete_speakers:
        stacked = F.normalize(
            torch.stack(embeddings_by_speaker[speaker_id][:enrollment_k]),
            p=2,
            dim=-1,
        )
        prototypes.append(F.normalize(stacked.mean(dim=0), p=2, dim=-1))
        enrollment_sample_ids.update(
            sample_ids_by_speaker[speaker_id][:enrollment_k]
        )

    gallery = FSCEnrollmentGallery(
        embeddings=torch.stack(prototypes).to(device),
        speaker_ids=tuple(complete_speakers),
        sample_ids=frozenset(enrollment_sample_ids),
        enrollment_k=enrollment_k,
    )
    print(
        "[+] FSC enrollment gallery: speakers={}, K={}, embeddings={}".format(
            len(gallery.speaker_ids),
            gallery.enrollment_k,
            tuple(gallery.embeddings.shape),
        )
    )
    return gallery


def sample_impostor_target_indices(
    scores: torch.Tensor,
    query_speaker_ids: Sequence[str],
    gallery_speaker_ids: Sequence[str],
    num_random_targets: int = 4,
    num_hard_targets: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select unique different-speaker targets for every query.

    Hard targets are the highest-scoring impostors in the current batch.
    Random targets are sampled uniformly from the remaining impostors, keeping
    the objective representative of random-pair FAR rather than only top-1 FAR.
    """

    if scores.dim() != 2:
        raise ValueError("scores must have shape [queries, gallery speakers].")
    if scores.shape[0] != len(query_speaker_ids):
        raise ValueError("Query score and speaker-ID counts do not match.")
    if scores.shape[1] != len(gallery_speaker_ids):
        raise ValueError("Gallery score and speaker-ID counts do not match.")
    if num_random_targets < 0 or num_hard_targets < 0:
        raise ValueError("Target counts cannot be negative.")
    targets_per_query = num_random_targets + num_hard_targets
    if targets_per_query <= 0:
        raise ValueError("At least one random or hard impostor target is required.")

    selected_rows = []
    hard_mask_rows = []
    for row_index, query_speaker_id in enumerate(query_speaker_ids):
        allowed = [
            index
            for index, gallery_speaker_id in enumerate(gallery_speaker_ids)
            if str(gallery_speaker_id) != str(query_speaker_id)
        ]
        if len(allowed) < targets_per_query:
            raise ValueError(
                "Query {} has {} impostors, but {} targets were requested.".format(
                    row_index, len(allowed), targets_per_query
                )
            )

        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=scores.device)
        hard_indices = allowed_tensor.new_empty((0,), dtype=torch.long)
        if num_hard_targets:
            allowed_scores = scores[row_index, allowed_tensor].detach()
            hard_positions = torch.topk(
                allowed_scores, k=num_hard_targets, largest=True, sorted=True
            ).indices
            hard_indices = allowed_tensor[hard_positions]

        hard_set = set(hard_indices.detach().cpu().tolist())
        random_candidates = [index for index in allowed if index not in hard_set]
        random_indices = allowed_tensor.new_empty((0,), dtype=torch.long)
        if num_random_targets:
            permutation = torch.randperm(
                len(random_candidates), generator=generator, device="cpu"
            )[:num_random_targets]
            random_indices = torch.tensor(
                [random_candidates[index] for index in permutation.tolist()],
                dtype=torch.long,
                device=scores.device,
            )

        selected_rows.append(torch.cat([random_indices, hard_indices], dim=0))
        hard_mask_rows.append(
            torch.tensor(
                [False] * num_random_targets + [True] * num_hard_targets,
                dtype=torch.bool,
                device=scores.device,
            )
        )

    return torch.stack(selected_rows), torch.stack(hard_mask_rows)


def compute_fsc_impostor_loss(
    adv_embeddings: torch.Tensor,
    query_speaker_ids: Sequence[str],
    gallery: FSCEnrollmentGallery,
    sv_threshold: float,
    beta: float = 10.0,
    num_random_targets: int = 4,
    num_hard_targets: int = 4,
    generator: Optional[torch.Generator] = None,
    target_indices: Optional[torch.Tensor] = None,
) -> FSCSVLossResult:
    """Compute a pair-aligned FSC false-acceptance training loss.

    Each selected impostor pair is assigned the attack target ``accepted``.
    Minimizing softplus(beta * (threshold - score)) therefore raises its score
    above the frozen validation threshold.  ``target_indices`` may be supplied
    to train against a fixed protocol instead of sampling targets dynamically.
    """

    if not 0.0 <= sv_threshold <= 1.0:
        raise ValueError("sv_threshold must be in scaled-cosine space [0, 1].")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    scores = scaled_cosine_scores(adv_embeddings, gallery.embeddings)
    if target_indices is None:
        target_indices, hard_target_mask = sample_impostor_target_indices(
            scores=scores,
            query_speaker_ids=query_speaker_ids,
            gallery_speaker_ids=gallery.speaker_ids,
            num_random_targets=num_random_targets,
            num_hard_targets=num_hard_targets,
            generator=generator,
        )
    else:
        target_indices = target_indices.to(device=scores.device, dtype=torch.long)
        if target_indices.dim() != 2 or target_indices.shape[0] != scores.shape[0]:
            raise ValueError("target_indices must have shape [queries, targets].")
        hard_target_mask = torch.zeros_like(target_indices, dtype=torch.bool)
        for row_index, query_speaker_id in enumerate(query_speaker_ids):
            for target_index in target_indices[row_index].detach().cpu().tolist():
                if str(gallery.speaker_ids[target_index]) == str(query_speaker_id):
                    raise ValueError("Fixed target indices contain a genuine pair.")

    selected_scores = torch.gather(scores, dim=1, index=target_indices)
    logits = beta * (selected_scores - float(sv_threshold))
    loss = F.softplus(-logits).mean()

    with torch.no_grad():
        accepted = selected_scores >= float(sv_threshold)
        margins = selected_scores - float(sv_threshold)
        metrics = {
            "sampled_far": float(accepted.float().mean()),
            "query_any_far": float(accepted.any(dim=1).float().mean()),
            "query_all_far": float(accepted.all(dim=1).float().mean()),
            "mean_impostor_score": float(selected_scores.mean()),
            "mean_threshold_margin": float(margins.mean()),
            "mean_best_impostor_score": float(selected_scores.max(dim=1).values.mean()),
            "num_pairs": float(selected_scores.numel()),
        }
        random_mask = ~hard_target_mask
        if random_mask.any():
            metrics["random_pair_far"] = float(
                accepted[random_mask].float().mean()
            )
        if hard_target_mask.any():
            metrics["hard_pair_far"] = float(
                accepted[hard_target_mask].float().mean()
            )

    return FSCSVLossResult(
        loss=loss,
        selected_scores=selected_scores,
        target_indices=target_indices,
        hard_target_mask=hard_target_mask,
        metrics=metrics,
    )


def compute_fsc_any_enrollment_loss(
    adv_embeddings: torch.Tensor,
    query_speaker_ids: Sequence[str],
    gallery: FSCEnrollmentGallery,
    sv_threshold: float,
    beta: float = 10.0,
    top_k: int = 1,
) -> FSCSVLossResult:
    """Optimize acceptance by any identity in a fixed FSC gallery.

    This objective models claimless/untargeted access: success means that the
    highest-scoring eligible enrollment identity crosses the frozen threshold.
    It intentionally uses the real top-k scores instead of a noisy-OR product,
    whose value can saturate without any individual score crossing threshold.
    """

    if not 0.0 <= sv_threshold <= 1.0:
        raise ValueError("sv_threshold must be in scaled-cosine space [0, 1].")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scores = scaled_cosine_scores(adv_embeddings, gallery.embeddings)
    if scores.shape[0] != len(query_speaker_ids):
        raise ValueError("Query embedding and speaker-ID counts do not match.")

    allowed_rows = []
    for query_speaker_id in query_speaker_ids:
        allowed_rows.append(
            [
                str(gallery_speaker_id) != str(query_speaker_id)
                for gallery_speaker_id in gallery.speaker_ids
            ]
        )
    allowed_mask = torch.tensor(
        allowed_rows, dtype=torch.bool, device=scores.device
    )
    available = allowed_mask.sum(dim=1)
    if (available < top_k).any():
        raise ValueError(
            "Each query needs at least {} different-speaker gallery targets.".format(
                top_k
            )
        )

    impostor_scores = scores.masked_fill(~allowed_mask, -1.0)
    selected_scores, target_indices = torch.topk(
        impostor_scores, k=top_k, dim=1, largest=True, sorted=True
    )
    best_scores = selected_scores[:, 0]
    loss = F.softplus(beta * (float(sv_threshold) - best_scores)).mean()

    with torch.no_grad():
        accepted = best_scores >= float(sv_threshold)
        metrics = {
            "any_enrollment_asr": float(accepted.float().mean()),
            "sampled_far": float(accepted.float().mean()),
            "query_any_far": float(accepted.float().mean()),
            "query_all_far": float(accepted.float().mean()),
            "mean_impostor_score": float(best_scores.mean()),
            "mean_threshold_margin": float(
                (best_scores - float(sv_threshold)).mean()
            ),
            "mean_best_impostor_score": float(best_scores.mean()),
            "num_pairs": float(best_scores.numel()),
        }

    return FSCSVLossResult(
        loss=loss,
        selected_scores=selected_scores,
        target_indices=target_indices,
        hard_target_mask=torch.ones_like(target_indices, dtype=torch.bool),
        metrics=metrics,
    )
