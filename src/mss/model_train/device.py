"""Torch device resolution."""

from __future__ import annotations


def get_device(config_device: str):
    import torch

    d = config_device.lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")
