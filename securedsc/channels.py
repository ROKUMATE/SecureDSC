"""Wireless channel models and the reciprocal-CSI helper.

Channels operate on real-valued symbol tensors of shape ``(B, L, C)`` that have
already been (approximately) power-normalised by the channel encoder. The
average symbol power is taken to be ~1, so the noise std follows directly from
the SNR.

``ReciprocalChannel`` is a *helper* (not an ``nn.Module`` in the forward path):
it synthesises CSI estimates for Alice/Bob/Eve such that Alice's and Bob's CSI
are correlated (reciprocal, coefficient ``rho``) while Eve's is independent.
This drives Improvement A (CSI-based key generation).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .utils import snr_db_to_noise_std


class Channel(ABC, torch.nn.Module):
    """Base class for differentiable channel layers.

    Subclasses implement :meth:`transmit`, mapping transmitted symbols ``x`` to
    received symbols ``y`` at a given SNR. The channel is differentiable so that
    gradients flow end-to-end through it.
    """

    @abstractmethod
    def transmit(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Return received symbols ``y`` for transmitted ``x`` at ``snr_db``."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:  # noqa: D401
        return self.transmit(x, snr_db)


def normalize_power(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise so the mean per-element power over (L, C) is 1 for each sample."""
    power = x.pow(2).mean(dim=(-2, -1), keepdim=True)
    return x / torch.sqrt(power + eps)


class AWGNChannel(Channel):
    """Additive white Gaussian noise channel: ``y = x + n``."""

    def transmit(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = snr_db_to_noise_std(snr_db, signal_power=1.0)
        noise = torch.randn_like(x) * std
        return x + noise


class RayleighChannel(Channel):
    """Flat Rayleigh fading channel: ``y = h * x + n``.

    A single complex Rayleigh gain per sample is applied (slow fading over the
    symbol block). We model the real symbol vector as interleaved I/Q pairs and
    apply the complex gain to each pair; with perfect CSI the receiver could
    equalise, but here the channel decoder learns to cope (as in DeepSC).
    """

    def transmit(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = snr_db_to_noise_std(snr_db, signal_power=1.0)
        b = x.shape[0]
        # Complex Rayleigh gain h = (a + jb)/sqrt(2), |h| Rayleigh, E[|h|^2]=1.
        hr = torch.randn(b, 1, 1, device=x.device) / math.sqrt(2.0)
        hi = torch.randn(b, 1, 1, device=x.device) / math.sqrt(2.0)

        c = x.shape[-1]
        if c % 2 == 0:
            xr = x[..., 0::2]
            xi = x[..., 1::2]
            yr = hr * xr - hi * xi
            yi = hr * xi + hi * xr
            y = torch.empty_like(x)
            y[..., 0::2] = yr
            y[..., 1::2] = yi
        else:  # odd channel dim: apply real gain magnitude
            y = torch.sqrt(hr * hr + hi * hi) * x
        return y + torch.randn_like(x) * std


def make_channel(kind: str) -> Channel:
    """Factory: ``"awgn"`` or ``"rayleigh"``."""
    kind = kind.lower()
    if kind == "awgn":
        return AWGNChannel()
    if kind == "rayleigh":
        return RayleighChannel()
    raise ValueError(f"Unknown channel kind: {kind}")


@dataclass
class CSIObservation:
    """CSI estimates observed by each party for one batch.

    Each tensor has shape ``(B, csi_dim)``. ``alice``/``bob`` are correlated
    (reciprocal); ``eve_a``/``eve_b`` (Eve's taps of the A->E and B->E links) are
    independent of the Alice<->Bob link.
    """

    alice: torch.Tensor
    bob: torch.Tensor
    eve_a: torch.Tensor
    eve_b: torch.Tensor


class ReciprocalChannel:
    """Synthesises reciprocal CSI for Alice/Bob and independent CSI for Eve.

    The ground-truth Alice<->Bob channel realisation ``g`` is drawn once; Alice
    and Bob each observe it through independent estimation noise, and additionally
    a correlation coefficient ``rho`` controls how reciprocal the two underlying
    observations are::

        h_AB = rho * g + sqrt(1 - rho^2) * e_A
        h_BA = rho * g + sqrt(1 - rho^2) * e_B

    so ``corr(h_AB, h_BA) ~= rho^2 / (rho^2 + (1-rho^2)) = rho^2`` before adding
    the (small) estimation noise. Eve's observations use a *fresh independent*
    ground truth, so her CSI is uncorrelated with the legitimate link.
    """

    def __init__(self, csi_dim: int, rho: float, snr_db: float = 15.0):
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        self.csi_dim = csi_dim
        self.rho = rho
        self.est_std = snr_db_to_noise_std(snr_db, signal_power=1.0)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CSIObservation:
        """Sample one batch of CSI observations."""
        dev = device or torch.device("cpu")

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, device=dev, generator=generator)

        d = self.csi_dim
        rho = self.rho
        comp = math.sqrt(max(0.0, 1.0 - rho * rho))

        # Legitimate Alice<->Bob ground truth and the two reciprocal views.
        g = randn(batch_size, d)
        h_ab = rho * g + comp * randn(batch_size, d)
        h_ba = rho * g + comp * randn(batch_size, d)
        alice = h_ab + self.est_std * randn(batch_size, d)
        bob = h_ba + self.est_std * randn(batch_size, d)

        # Eve's independent taps.
        eve_a = randn(batch_size, d) + self.est_std * randn(batch_size, d)
        eve_b = randn(batch_size, d) + self.est_std * randn(batch_size, d)
        return CSIObservation(alice=alice, bob=bob, eve_a=eve_a, eve_b=eve_b)
