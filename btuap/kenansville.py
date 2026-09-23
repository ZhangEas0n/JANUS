"""Kenansville-style spectral filtering utilities."""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def valid_lengths(waveforms: torch.Tensor, input_lengths=None) -> Iterable[int]:
    if input_lengths is None:
        return [waveforms.shape[-1]] * waveforms.shape[0]
    if torch.is_tensor(input_lengths):
        input_lengths = input_lengths.detach().cpu().tolist()
    return [int(length) for length in input_lengths]


def apply_kenansville_fft(
    waveforms: torch.Tensor,
    input_lengths=None,
    factor: float = 0.1,
) -> torch.Tensor:
    """Remove low-magnitude FFT bins independently for each utterance.

    ``factor <= 1`` is relative to an utterance's maximum FFT magnitude.
    Larger values are interpreted as absolute magnitude thresholds.
    """
    if factor < 0:
        raise ValueError("Kenansville factor must be non-negative.")

    outputs = waveforms.clone()
    lengths = valid_lengths(waveforms, input_lengths)
    for index, length in enumerate(lengths):
        length = max(1, min(int(length), waveforms.shape[-1]))
        signal = waveforms[index, :length]
        spectrum = torch.fft.fft(signal)
        threshold = torch.as_tensor(factor, device=signal.device, dtype=signal.dtype)
        if factor <= 1.0:
            threshold = threshold * spectrum.abs().amax()
        filtered = torch.where(
            spectrum.abs() < threshold,
            torch.zeros_like(spectrum),
            spectrum,
        )
        outputs[index, :length] = torch.fft.ifft(filtered).real.clamp(-1.0, 1.0)
    return outputs


def rescale_to_snr(
    clean: torch.Tensor,
    attacked: torch.Tensor,
    target_snr_db: float,
    input_lengths=None,
) -> torch.Tensor:
    """Rescale each transformed utterance to a fixed SNR."""
    outputs = clean.clone()
    lengths = valid_lengths(clean, input_lengths)
    ratio = 10.0 ** (float(target_snr_db) / 10.0)
    for index, length in enumerate(lengths):
        length = max(1, min(int(length), clean.shape[-1]))
        signal = clean[index, :length]
        delta = attacked[index, :length] - signal
        signal_power = signal.pow(2).mean().clamp_min(1e-12)
        delta_power = delta.pow(2).mean()
        if float(delta_power) <= 1e-20:
            outputs[index, :length] = signal
            continue
        target_noise_power = signal_power / ratio
        scale = torch.sqrt(target_noise_power / delta_power)
        outputs[index, :length] = (signal + scale * delta).clamp(-1.0, 1.0)
    return outputs


def perturbation_statistics(clean, attacked, input_lengths=None):
    """Return per-utterance Linf, RMS and SNR values."""
    rows = []
    lengths = valid_lengths(clean, input_lengths)
    for index, length in enumerate(lengths):
        length = max(1, min(int(length), clean.shape[-1]))
        signal = clean[index, :length]
        delta = attacked[index, :length] - signal
        signal_power = signal.pow(2).mean().clamp_min(1e-12)
        noise_power = delta.pow(2).mean().clamp_min(1e-12)
        rows.append(
            {
                "linf": float(delta.abs().amax().detach().cpu()),
                "rms": float(noise_power.sqrt().detach().cpu()),
                "snr_db": float(
                    (10.0 * torch.log10(signal_power / noise_power)).detach().cpu()
                ),
            }
        )
    return rows
