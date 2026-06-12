"""Lambda schedulers for the joint loss ``L_joint = L_B + |L_E - lambda|``.

- :class:`FixedLambda` (baseline) returns a constant.
- :class:`AdaptiveLambda` (Improvement B) tracks an EMA of Eve's loss and moves
  ``lambda`` toward a *target Eve loss* via a simple feedback rule, removing the
  need to hand-tune the absolute value and stabilising training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LambdaScheduler(ABC):
    """Base class. ``value`` is the current lambda; ``step`` updates it."""

    def __init__(self, initial: float):
        self.value = float(initial)

    @abstractmethod
    def step(self, l_b: float, l_e: float) -> float:
        """Update and return ``lambda`` given the latest Bob/Eve losses."""

    def get(self) -> float:
        return self.value


class FixedLambda(LambdaScheduler):
    """Constant lambda (the paper's baseline hyperparameter)."""

    def step(self, l_b: float, l_e: float) -> float:
        return self.value


class AdaptiveLambda(LambdaScheduler):
    """Feedback controller driving Eve's loss toward a target.

    Tracks ``ema_L_E`` and updates::

        lambda <- clip(lambda + eta * (target - ema_L_E), min_value, max_value)

    Intuition: ``|L_E - lambda|`` in the joint loss pulls Eve's achievable loss
    toward ``lambda``. If Eve is currently doing *better* than the target
    (``ema_L_E < target``), lambda is increased to push her harder; if she is
    already worse than the target, lambda relaxes. ``target`` is typically a high
    fraction of ``ln(vocab)`` (near-random guessing for Eve).
    """

    def __init__(
        self,
        initial: float,
        target: float,
        eta: float = 0.05,
        ema: float = 0.9,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ):
        super().__init__(initial)
        self.target = float(target)
        self.eta = float(eta)
        self.ema = float(ema)
        self.min_value = float(min_value)
        self.max_value = max_value
        self._ema_le: Optional[float] = None

    def step(self, l_b: float, l_e: float) -> float:
        if self._ema_le is None:
            self._ema_le = float(l_e)
        else:
            self._ema_le = self.ema * self._ema_le + (1.0 - self.ema) * float(l_e)
        new_val = self.value + self.eta * (self.target - self._ema_le)
        if self.max_value is not None:
            new_val = min(new_val, self.max_value)
        new_val = max(new_val, self.min_value)
        self.value = new_val
        return self.value

    @property
    def ema_le(self) -> Optional[float]:
        return self._ema_le


def make_lambda_scheduler(cfg, vocab_size: int) -> LambdaScheduler:
    """Factory from a :class:`LambdaConfig`; ``vocab_size`` sets the loss ceiling."""
    import math

    ln_vocab = math.log(vocab_size)
    kind = cfg.kind.lower()
    if kind == "fixed":
        if cfg.value < 0.5 * ln_vocab:
            import warnings

            warnings.warn(
                f"fixed lambda={cfg.value:.2f} is low vs ln(vocab)={ln_vocab:.2f}. "
                f"L_joint pulls Eve's loss toward lambda, so a low lambda trains "
                f"Alice to HELP Eve. For secrecy set lambda ~= {0.9 * ln_vocab:.2f} "
                f"(0.9*ln(vocab)) or use the adaptive scheduler.",
                stacklevel=2,
            )
        return FixedLambda(cfg.value)
    if kind == "adaptive":
        max_value = cfg.max_value if cfg.max_value is not None else ln_vocab
        return AdaptiveLambda(
            initial=cfg.value,
            target=cfg.target_frac * ln_vocab,
            eta=cfg.eta,
            ema=cfg.ema,
            min_value=cfg.min_value,
            max_value=max_value,
        )
    raise ValueError(f"Unknown lambda scheduler kind: {kind}")
