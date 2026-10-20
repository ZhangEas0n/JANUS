from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import wave
from array import array
import sys

import torch

from btuap.common import apply_uap


def load_pcm16_mono_audio(audio_path, target_sr=16000):
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError("This generator expects 16-bit PCM WAV audio.")
    if sample_rate != target_sr:
        raise ValueError(
            f"Expected {target_sr} Hz audio, got {sample_rate} Hz. "
            "Resample the input audio before generation."
        )

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()

    if channels == 1:
        mono = samples
    else:
        mono = array(
            "h",
            [
                int(sum(samples[index : index + channels]) / channels)
                for index in range(0, len(samples), channels)
            ],
        )

    waveform = torch.tensor(mono, dtype=torch.float32).unsqueeze(0) / 32768.0
    return waveform, sample_rate


def save_pcm16_mono_audio(output_path, waveform, sample_rate):
    pcm = (
        waveform.squeeze(0)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .tolist()
    )
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def generate_adv_audio(
    audio_path,
    uap_path,
    output_path,
    target_sr=16000,
    scale_factor=1.0,
):
    audio_path = Path(audio_path)
    uap_path = Path(uap_path)
    output_path = Path(output_path)

    waveform, sample_rate = load_pcm16_mono_audio(audio_path, target_sr=target_sr)
    delta = torch.load(str(uap_path), map_location="cpu").reshape(1, -1)
    delta = delta * scale_factor
    waveform = waveform.cpu()

    adv_waveform = apply_uap(waveform, delta)
    applied_delta = adv_waveform - waveform

    signal_rms = waveform.pow(2).mean().sqrt()
    noise_rms = applied_delta.pow(2).mean().sqrt()
    snr_db = 20 * torch.log10(signal_rms / noise_rms.clamp_min(1e-12))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pcm16_mono_audio(output_path, adv_waveform, sample_rate)

    print(f"Input: {audio_path}")
    print(f"UAP: {uap_path}, shape={tuple(delta.shape)}, scale={scale_factor:.3f}")
    print(f"Output: {output_path}")
    print(
        "Applied perturbation: "
        f"linf={applied_delta.abs().max().item():.6f}, "
        f"rms={noise_rms.item():.6f}, "
        f"snr_db={snr_db.item():.2f}"
    )


if __name__ == "__main__":
    generate_adv_audio(
        audio_path="test.wav",
        uap_path="btuap.pt",
        output_path="test_adv.wav",
    )
