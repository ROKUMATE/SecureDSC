"""Key generators: random baseline and CSI-based neural key generation.

A key generator produces, per batch, a :class:`KeyBundle` of *bit vectors* in
``{-1, +1}`` of width ``key_dim`` for Alice, Bob and Eve, plus agreement metrics.
These bit vectors are consumed by the model's key-processing network and turned
into key tokens for the encryptor/decryptor.

- :class:`RandomKeyGenerator` (baseline): Alice and Bob share an identical random
  key; Eve draws an independent random key.
- :class:`CSIKeyGenerator` (Improvement A): a small MLP maps each party's CSI
  estimate to soft bits; sign-quantisation + repetition-majority reconciliation
  makes Alice and Bob converge, while Eve (independent CSI) cannot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .channels import CSIObservation, ReciprocalChannel


@dataclass
class KeyBundle:
    """Per-batch keys (``{-1,+1}`` bit vectors, shape ``(B, key_dim)``) + metrics.

    ``alice`` and ``bob`` are the *reconciled, identical* keys used by the cipher
    path. ``eve`` is Eve's own derived key. Metrics describe the raw key quality
    *before* forcing reconciliation (the meaningful security signal).
    """

    alice: torch.Tensor
    bob: torch.Tensor
    eve: torch.Tensor
    ab_agreement: float  # raw Alice/Bob bit-agreement rate in [0, 1]
    eve_mismatch: float  # Eve-vs-Alice bit-mismatch rate in [0, 1]
    # Soft (differentiable) bits, used to train the CSI generator (None for random)
    alice_soft: Optional[torch.Tensor] = None
    bob_soft: Optional[torch.Tensor] = None
    eve_soft: Optional[torch.Tensor] = None


def bit_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of matching signs between two ``{-1,+1}`` bit tensors."""
    return (torch.sign(a) == torch.sign(b)).float().mean().item()


def majority_reconcile(hard_bits: torch.Tensor, repeat: int) -> torch.Tensor:
    """Repetition-code reconciliation.

    ``hard_bits`` has shape ``(B, key_dim * repeat)``; group every ``repeat``
    consecutive bits and take a majority vote, yielding ``(B, key_dim)``. More
    repeats reduce the Alice/Bob mismatch at the cost of more CSI bandwidth.
    """
    b, n = hard_bits.shape
    assert n % repeat == 0, "bit width must be divisible by repeat"
    blocks = hard_bits.view(b, n // repeat, repeat)
    voted = torch.sign(blocks.sum(dim=-1))
    # break ties (sum == 0, only possible for even repeat) toward +1
    voted = torch.where(voted == 0, torch.ones_like(voted), voted)
    return voted


class KeyGenerator(ABC, nn.Module):
    """Base class for key generators."""

    key_dim: int

    @abstractmethod
    def generate(
        self,
        batch_size: int,
        device: torch.device,
        csi: Optional[CSIObservation] = None,
    ) -> KeyBundle:
        """Produce a :class:`KeyBundle` for one batch."""


class RandomKeyGenerator(KeyGenerator):
    """Baseline: shared random key for Alice/Bob, independent key for Eve."""

    def __init__(self, key_dim: int):
        super().__init__()
        self.key_dim = key_dim

    def generate(
        self,
        batch_size: int,
        device: torch.device,
        csi: Optional[CSIObservation] = None,
    ) -> KeyBundle:
        shared = torch.randint(
            0, 2, (batch_size, self.key_dim), device=device, dtype=torch.float
        ) * 2.0 - 1.0
        eve = torch.randint(
            0, 2, (batch_size, self.key_dim), device=device, dtype=torch.float
        ) * 2.0 - 1.0
        return KeyBundle(
            alice=shared,
            bob=shared.clone(),
            eve=eve,
            ab_agreement=1.0,
            eve_mismatch=bit_agreement(eve, -shared),  # mismatch = 1 - agreement
        )


class CSIKeyGenerator(KeyGenerator):
    """Neural key generator mapping CSI estimates to reconciled key bits."""

    def __init__(self, csi_dim: int, key_dim: int, hidden: int = 128, repeat: int = 3):
        super().__init__()
        self.key_dim = key_dim
        self.repeat = repeat
        self.net = nn.Sequential(
            nn.Linear(csi_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, key_dim * repeat),
            nn.Tanh(),  # soft bits in (-1, 1)
        )

    def _derive(self, csi: torch.Tensor):
        soft = self.net(csi)  # (B, key_dim*repeat), differentiable
        hard = torch.sign(soft.detach())
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)
        recon = majority_reconcile(hard, self.repeat)  # (B, key_dim)
        # soft bits reconciled by averaging within each repeat block (for training)
        b = soft.shape[0]
        soft_blocks = soft.view(b, self.key_dim, self.repeat).mean(dim=-1)
        return recon, soft_blocks

    def generate(
        self,
        batch_size: int,
        device: torch.device,
        csi: Optional[CSIObservation] = None,
    ) -> KeyBundle:
        if csi is None:
            raise ValueError("CSIKeyGenerator requires a CSIObservation")
        a_bits, a_soft = self._derive(csi.alice)
        b_bits, b_soft = self._derive(csi.bob)
        # Eve derives from her own (independent) tap of the A->E link.
        e_bits, e_soft = self._derive(csi.eve_a)

        ab = bit_agreement(a_bits, b_bits)
        eve_mismatch = 1.0 - bit_agreement(e_bits, a_bits)

        # Cipher path uses a reconciled, identical key: after the public
        # reconciliation exchange Bob adopts Alice's reconciled bits. The
        # reported ``ab_agreement`` reflects the raw quality before this step.
        return KeyBundle(
            alice=a_bits,
            bob=a_bits.clone(),
            eve=e_bits,
            ab_agreement=ab,
            eve_mismatch=eve_mismatch,
            alice_soft=a_soft,
            bob_soft=b_soft,
            eve_soft=e_soft,
        )

    def agreement_loss(self, bundle: KeyBundle, margin: float = 0.5) -> torch.Tensor:
        """Training loss: pull Alice/Bob soft bits together, push Eve away.

        ``MSE(soft_A, soft_B)`` (agreement) plus a hinge that keeps Eve's soft
        bits at least ``margin`` away from Alice's (decorrelation).
        """
        assert bundle.alice_soft is not None and bundle.bob_soft is not None
        agree = torch.mean((bundle.alice_soft - bundle.bob_soft) ** 2)
        if bundle.eve_soft is not None:
            eve_dist = torch.mean((bundle.eve_soft - bundle.alice_soft.detach()) ** 2)
            push = torch.clamp(margin - eve_dist, min=0.0)
        else:
            push = torch.zeros((), device=agree.device)
        return agree + push


def make_key_generator(cfg, key_dim: int, csi_dim: int) -> KeyGenerator:
    """Factory from a :class:`KeyGenConfig`, the key width and the CSI width."""
    kind = cfg.kind.lower()
    if kind == "random":
        return RandomKeyGenerator(key_dim)
    if kind == "csi":
        return CSIKeyGenerator(
            csi_dim=csi_dim,
            key_dim=key_dim,
            hidden=cfg.csi_hidden,
            repeat=cfg.recon_repeat,
        )
    raise ValueError(f"Unknown keygen kind: {kind}")
