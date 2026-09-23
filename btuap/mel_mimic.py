"""Log-Mel environmental-sound mimic loss for JANUS perturbation training."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def rms(audio, eps=1e-8):
    return torch.sqrt(torch.mean(audio.pow(2)) + eps)


def normalize_to_rms(audio, target_rms, eps=1e-8):
    return audio / (rms(audio, eps=eps) + eps) * target_rms


def match_length(audio, target_len):
    audio = audio.reshape(-1)
    if audio.numel() < target_len:
        repeat = target_len // audio.numel() + 1
        audio = audio.repeat(repeat)
    return audio[:target_len]


def load_environment_sound(path, target_len, sample_rate, device, dtype):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Missing environment sound: {}".format(path))

    audio, source_sample_rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        audio = torchaudio.functional.resample(
            audio, source_sample_rate, sample_rate
        )
    audio = match_length(audio.squeeze(0), target_len)
    audio = audio - audio.mean()
    peak = audio.abs().max()
    if not torch.isfinite(peak) or float(peak) <= 0:
        raise ValueError("Environment sound is silent or non-finite: {}".format(path))
    audio = audio / peak
    return audio.unsqueeze(0).to(device=device, dtype=dtype)


class MelMimicLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, delta, env_template):
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        if env_template.dim() == 1:
            env_template = env_template.unsqueeze(0)
        delta_logmel = torch.log(self.mel(delta) + self.eps)
        env_logmel = torch.log(self.mel(env_template) + self.eps)
        return torch.mean(torch.abs(delta_logmel - env_logmel))


class MultiResolutionSTFTLoss(nn.Module):
    """Log-magnitude STFT loss at several time/frequency resolutions."""

    def __init__(self, fft_sizes=(256, 512, 1024), eps=1e-6):
        super().__init__()
        self.fft_sizes = tuple(int(value) for value in fft_sizes)
        self.eps = float(eps)

    def _log_magnitude(self, audio, n_fft):
        window = torch.hann_window(
            n_fft, device=audio.device, dtype=audio.dtype
        )
        spectrum = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=False,
        )
        magnitude = torch.sqrt(spectrum.pow(2).sum(dim=-1) + self.eps)
        return torch.log(magnitude + self.eps)

    def forward(self, delta, target):
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        losses = []
        for n_fft in self.fft_sizes:
            delta_logmag = self._log_magnitude(delta, n_fft)
            target_logmag = self._log_magnitude(target, n_fft)
            losses.append(torch.mean(torch.abs(delta_logmag - target_logmag)))
        return torch.stack(losses).mean()


def waveform_similarity_loss(delta, target, eps=1e-8):
    """Cosine distance that preserves the carrier waveform and its phase."""

    if delta.dim() == 1:
        delta = delta.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    similarity = F.cosine_similarity(
        delta.reshape(delta.shape[0], -1),
        target.reshape(target.shape[0], -1),
        dim=1,
        eps=eps,
    )
    return (1.0 - similarity).mean()
