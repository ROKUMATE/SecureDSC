"""Reproducibility, device and logging helpers."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec (``"auto" | "cpu" | "cuda"``) to a device."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def get_logger(name: str = "securedsc", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if needed; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def snr_db_to_noise_std(snr_db: float, signal_power: float = 1.0) -> float:
    """Convert an SNR (dB) to a noise standard deviation for unit-ish signals.

    Assumes the signal has the given average power. ``noise_var = P / 10^(snr/10)``
    and the returned std applies per real dimension.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = signal_power / snr_linear
    return float(np.sqrt(noise_var))


def count_parameters(module: torch.nn.Module) -> int:
    """Total number of trainable parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
