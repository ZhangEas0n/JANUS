"""DistilHuBERT feature extraction and lightweight FSC downstream heads."""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel


SAMPLE_RATE = 16000


class DistilHuBERTFSCHeads(nn.Module):
    """Frozen-backbone heads for speaker verification and sensitive intent."""

    def __init__(self, input_dim=768, sv_dim=256, sr_hidden_dim=256,
                 num_train_speakers=77, dropout=0.1):
        super().__init__()
        self.sv_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, sv_dim), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sv_classifier = nn.Linear(sv_dim, num_train_speakers)
        self.sr_classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, sr_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(sr_hidden_dim, 2),
        )

    def speaker_embedding(self, pooled):
        return F.normalize(self.sv_projection(pooled), p=2, dim=-1)

    def forward(self, pooled):
        sv_embedding = self.speaker_embedding(pooled)
        return sv_embedding, self.sv_classifier(sv_embedding), self.sr_classifier(pooled)


def load_backbone(backbone_path, device):
    model = AutoModel.from_pretrained(
        str(Path(backbone_path).resolve()), local_files_only=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_waveform(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("Empty audio: {}".format(path))
    mono = waveform.float().mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, SAMPLE_RATE)
    return mono


def collate_waveforms(paths: Sequence[Path]):
    waves = [load_waveform(Path(path)) for path in paths]
    lengths = torch.tensor([wave.numel() for wave in waves], dtype=torch.long)
    return pad_sequence(waves, batch_first=True, padding_value=0.0), lengths


@torch.inference_mode()
def encode_waveforms(backbone, waveforms, lengths, device):
    waveforms = waveforms.to(device)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        hidden = backbone(waveforms).last_hidden_state
    frame_lengths = backbone._get_feat_extract_output_lengths(lengths.to(device))
    index = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
    mask = (index < frame_lengths.unsqueeze(1)).unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.float()


@torch.inference_mode()
def extract_path_embeddings(backbone, paths: Sequence[Path], device, batch_size=32):
    output = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        waves, lengths = collate_waveforms(batch_paths)
        output.append(encode_waveforms(backbone, waves, lengths, device).cpu())
    return torch.cat(output, dim=0)


def scaled_cosine(query, gallery):
    query = F.normalize(query, p=2, dim=-1)
    gallery = F.normalize(gallery, p=2, dim=-1)
    return (query @ gallery.T + 1.0) / 2.0


def load_heads_checkpoint(checkpoint_path, device):
    payload = torch.load(str(checkpoint_path), map_location=device)
    cfg = payload["head_config"]
    heads = DistilHuBERTFSCHeads(**cfg).to(device)
    heads.load_state_dict(payload["model_state_dict"])
    heads.eval()
    for parameter in heads.parameters():
        parameter.requires_grad = False
    return heads, payload

