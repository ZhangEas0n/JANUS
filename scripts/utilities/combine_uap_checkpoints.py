"""Combine independently trained UAP tensors for an ablation baseline."""

from __future__ import print_function

import argparse
from pathlib import Path

import torch


def load_uap(path):
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("uap_delta", "delta", "perturbation", "uap"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        raise TypeError("Unsupported UAP checkpoint content in {}".format(path))
    tensor = payload.detach().float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError("Expected [1, samples] UAP in {}, got {}".format(
            path, tuple(tensor.shape)
        ))
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("sum", "mean"), default="sum",
        help="Use sum for the independently generated SV+SR baseline.",
    )
    args = parser.parse_args()

    tensors = [load_uap(path) for path in args.input]
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("All UAP tensors must have the same shape: {}".format(shapes))
    combined = torch.stack(tensors, dim=0).sum(dim=0)
    if args.mode == "mean":
        combined = combined / float(len(tensors))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, str(args.output))
    print(
        "[+] Saved {} UAPs using {} to {}: shape={}, linf={:.6f}, rms={:.6f}".format(
            len(tensors), args.mode, args.output, tuple(combined.shape),
            float(combined.abs().max()), float(combined.pow(2).mean().sqrt())
        )
    )


if __name__ == "__main__":
    main()
