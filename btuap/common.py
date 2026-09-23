from pathlib import Path

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def normalize_batch_like_pipeline(
    x: torch.Tensor,
    input_lengths=None,
) -> torch.Tensor:
    if input_lengths is not None:
        if x.dim() != 2:
            raise ValueError(
                "Length-aware normalization expects padded waveform batches "
                f"with shape [batch, time], got {x.shape}"
            )

        normalized = torch.zeros_like(x)
        for index, length in enumerate(input_lengths):
            valid_length = int(length)
            valid_waveform = x[index : index + 1, :valid_length]
            mean = valid_waveform.mean(dim=1, keepdim=True)
            std = valid_waveform.std(dim=1, keepdim=True)
            normalized[index, :valid_length] = (
                (valid_waveform - mean) / (std + 1e-5)
            ).squeeze(0)

        return normalized

    if x.dim() == 3:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
    elif x.dim() == 2:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")

    return (x - mean) / (std + 1e-5)


def load_config_fragment(relative_path: str) -> DictConfig:
    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / relative_path
    return OmegaConf.load(config_path)


def apply_uap(x_clean_raw: torch.Tensor, uap_delta: torch.Tensor) -> torch.Tensor:
    current_len = x_clean_raw.shape[-1]
    if uap_delta.shape[-1] >= current_len:
        current_uap = uap_delta[..., :current_len]
    else:
        repeat_times = (current_len + uap_delta.shape[-1] - 1) // uap_delta.shape[-1]
        current_uap = uap_delta.repeat(1, repeat_times)[..., :current_len]

    if x_clean_raw.dim() == 3:
        current_uap = current_uap.unsqueeze(1)

    return torch.clamp(x_clean_raw + current_uap, min=-1.0, max=1.0)


def repeat_or_crop_perturbation(
    delta_seed: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Repeat or crop one perturbation seed along its final dimension."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if delta_seed.numel() == 0 or delta_seed.shape[-1] == 0:
        raise ValueError("delta_seed must contain at least one sample")

    if delta_seed.shape[-1] >= target_length:
        return delta_seed[..., :target_length]
    repeat_times = (target_length + delta_seed.shape[-1] - 1) // delta_seed.shape[-1]
    repeats = [1] * delta_seed.dim()
    repeats[-1] = repeat_times
    return delta_seed.repeat(*repeats)[..., :target_length]


def apply_async_perturbation(
    clean_audio: torch.Tensor,
    delta_seed: torch.Tensor,
    sample_rate: int,
    offset_seconds: float,
    epsilon=None,
    target_snr_db=None,
):
    """Apply a circularly shifted universal perturbation to a waveform.

    The function accepts ``[T]``, ``[1, T]``, or batched ``[B, T]`` audio and
    preserves that shape. The returned perturbation is the signal actually
    present after output clipping.
    """
    if clean_audio.dim() not in (1, 2, 3):
        raise ValueError(
            "clean_audio must have shape [T], [B, T], or [B, 1, T], got {}".format(
                tuple(clean_audio.shape)
            )
        )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if offset_seconds < 0:
        raise ValueError("offset_seconds must be non-negative")
    if epsilon is not None and target_snr_db is not None:
        raise ValueError("epsilon and target_snr_db are mutually exclusive")

    delta = delta_seed.to(device=clean_audio.device, dtype=clean_audio.dtype)
    if delta.dim() == 1:
        delta = delta.unsqueeze(0)
    if delta.dim() == 3 and delta.shape[1] == 1:
        delta = delta[:, 0, :]
    if delta.dim() != 2 or delta.shape[0] != 1:
        raise ValueError(
            "delta_seed must have shape [T] or [1, T], got {}".format(
                tuple(delta_seed.shape)
            )
        )

    shift_samples = int(round(float(offset_seconds) * sample_rate))
    shift_samples %= delta.shape[-1]
    shifted = torch.roll(delta, shifts=shift_samples, dims=-1)
    perturbation = repeat_or_crop_perturbation(shifted, clean_audio.shape[-1])
    if clean_audio.dim() == 3:
        perturbation = perturbation.unsqueeze(1)

    if epsilon is not None:
        epsilon = float(epsilon)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        perturbation = perturbation.clamp(-epsilon, epsilon)

    if target_snr_db is not None:
        clean_batch = clean_audio.unsqueeze(0) if clean_audio.dim() == 1 else clean_audio
        perturbation_batch = (
            perturbation.unsqueeze(0) if clean_audio.dim() == 1 else perturbation
        )
        reduce_dims = tuple(range(1, clean_batch.dim()))
        signal_power = clean_batch.pow(2).mean(dim=reduce_dims, keepdim=True)
        noise_power = perturbation_batch.pow(2).mean(
            dim=reduce_dims, keepdim=True
        ).clamp_min(1e-12)
        target_noise_power = signal_power / (10.0 ** (float(target_snr_db) / 10.0))
        perturbation_batch = perturbation_batch * torch.sqrt(
            target_noise_power / noise_power
        )
        perturbation = (
            perturbation_batch.squeeze(0)
            if clean_audio.dim() == 1
            else perturbation_batch
        )

    adv_audio = torch.clamp(clean_audio + perturbation, min=-1.0, max=1.0)
    return adv_audio, adv_audio - clean_audio
