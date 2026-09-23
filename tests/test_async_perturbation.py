import pytest
import torch

from btuap.common import apply_async_perturbation, repeat_or_crop_perturbation


def test_repeat_or_crop_perturbation():
    delta = torch.tensor([[1.0, 2.0, 3.0]])
    actual = repeat_or_crop_perturbation(delta, 8)
    expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]])
    assert torch.equal(actual, expected)


def test_async_offset_uses_circular_shift():
    clean = torch.zeros(6)
    delta = torch.tensor([0.1, 0.2, 0.3])
    adv, used = apply_async_perturbation(
        clean, delta, sample_rate=2, offset_seconds=0.5
    )
    expected = torch.tensor([0.3, 0.1, 0.2, 0.3, 0.1, 0.2])
    assert torch.allclose(adv, expected)
    assert torch.allclose(used, expected)


def test_async_epsilon_and_shape_are_preserved():
    clean = torch.zeros(1, 5)
    delta = torch.tensor([[0.4, -0.3]])
    adv, used = apply_async_perturbation(
        clean, delta, sample_rate=4, offset_seconds=0.25, epsilon=0.1
    )
    assert adv.shape == clean.shape
    assert used.abs().max().item() == pytest.approx(0.1)


def test_async_target_snr():
    clean = torch.ones(1, 100) * 0.1
    delta = torch.ones(1, 10) * 0.01
    adv, used = apply_async_perturbation(
        clean, delta, sample_rate=10, offset_seconds=0.0, target_snr_db=20.0
    )
    snr = 10.0 * torch.log10(clean.pow(2).mean() / used.pow(2).mean())
    assert snr.item() == pytest.approx(20.0, abs=1e-4)


def test_async_rejects_two_constraints():
    with pytest.raises(ValueError):
        apply_async_perturbation(
            torch.zeros(8),
            torch.ones(4),
            sample_rate=4,
            offset_seconds=0.0,
            epsilon=0.1,
            target_snr_db=20.0,
        )
