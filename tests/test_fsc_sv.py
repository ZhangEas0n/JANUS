import torch

from btuap.fsc_sv import (
    FSCEnrollmentGallery,
    compute_fsc_impostor_loss,
    sample_impostor_target_indices,
    scaled_cosine_scores,
)


def make_gallery():
    return FSCEnrollmentGallery(
        embeddings=torch.eye(4),
        speaker_ids=("a", "b", "c", "d"),
        sample_ids=frozenset({"a-1", "b-1", "c-1", "d-1"}),
        enrollment_k=1,
    )


def test_scaled_cosine_scores_use_evaluator_space():
    query = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    enrollment = torch.tensor([[1.0, 0.0]])

    scores = scaled_cosine_scores(query, enrollment)

    assert torch.allclose(scores[:, 0], torch.tensor([1.0, 0.0, 0.5]))


def test_target_sampling_excludes_genuine_speaker_and_duplicates():
    scores = torch.tensor(
        [
            [1.0, 0.8, 0.7, 0.6],
            [0.4, 1.0, 0.9, 0.8],
        ]
    )
    generator = torch.Generator().manual_seed(7)

    indices, hard_mask = sample_impostor_target_indices(
        scores,
        query_speaker_ids=["a", "b"],
        gallery_speaker_ids=["a", "b", "c", "d"],
        num_random_targets=1,
        num_hard_targets=1,
        generator=generator,
    )

    assert indices.shape == (2, 2)
    assert hard_mask.tolist() == [[False, True], [False, True]]
    assert 0 not in indices[0].tolist()
    assert 1 not in indices[1].tolist()
    assert len(set(indices[0].tolist())) == 2
    assert len(set(indices[1].tolist())) == 2
    assert indices[0, 1].item() == 1
    assert indices[1, 1].item() == 2


def test_pair_aligned_loss_rewards_higher_impostor_scores():
    gallery = make_gallery()
    targets = torch.tensor([[1], [0]])
    low_similarity = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    high_similarity = torch.tensor(
        [[0.2, 1.0, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0]],
        requires_grad=True,
    )

    low_result = compute_fsc_impostor_loss(
        low_similarity,
        ["a", "b"],
        gallery,
        sv_threshold=0.7,
        target_indices=targets,
    )
    high_result = compute_fsc_impostor_loss(
        high_similarity,
        ["a", "b"],
        gallery,
        sv_threshold=0.7,
        target_indices=targets,
    )

    assert high_result.loss < low_result.loss
    assert low_result.metrics["sampled_far"] == 0.0
    assert high_result.metrics["sampled_far"] == 1.0
    high_result.loss.backward()
    assert high_similarity.grad is not None


def test_fixed_targets_reject_genuine_pairs():
    gallery = make_gallery()
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    try:
        compute_fsc_impostor_loss(
            query,
            ["a"],
            gallery,
            sv_threshold=0.7,
            target_indices=torch.tensor([[0]]),
        )
    except ValueError as error:
        assert "genuine pair" in str(error)
    else:
        raise AssertionError("Expected a genuine-pair validation error.")
