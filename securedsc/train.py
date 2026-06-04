"""Algorithm-1 trainer for SecureDSC.

Per batch the trainer cycles through four (optionally five) sub-steps:

  (i)   update semantic enc/dec        with ``L_sem``
  (ii)  update encryptor/decryptor     with ``L_cip``
  (iii) update the whole Alice/Bob net with ``L_joint = L_B + |L_E - lambda|``
  (iv)  update Eve's network           with ``L_E``
  (v)   [CSI key only] update the key generator with the agreement loss

``lambda`` is provided by a :class:`LambdaScheduler` (fixed or adaptive) and is
updated once per step from the latest Bob/Eve losses.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import Channel, ReciprocalChannel, make_channel
from .config import Config
from .data import Vocabulary, build_dataloaders
from .keygen import CSIKeyGenerator, KeyBundle, KeyGenerator, make_key_generator
from .lambda_sched import LambdaScheduler, make_lambda_scheduler
from .model import SecureDSC
from .utils import ensure_dir, get_logger, resolve_device, set_seed


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Token-level cross-entropy ignoring padding."""
    v = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, v), targets.reshape(-1), ignore_index=pad_id
    )


@dataclass
class EpochStats:
    """Average losses / metrics for one epoch."""

    epoch: int
    l_sem: float = 0.0
    l_cip: float = 0.0
    l_b: float = 0.0
    l_e: float = 0.0
    lam: float = 0.0
    l_key: float = 0.0
    ab_agreement: float = 1.0
    eve_mismatch: float = 0.0
    val_bleu_bob: float = 0.0
    val_bleu_eve: float = 0.0


class SecureDSCTrainer:
    """Owns the model, optimizers, channels and the Algorithm-1 loop."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger()
        set_seed(cfg.train.seed)
        self.device = resolve_device(cfg.train.device)

        self.train_loader, self.val_loader, self.vocab = build_dataloaders(
            cfg.data, cfg.train.batch_size, seed=cfg.train.seed
        )
        self.pad_id = self.vocab.pad_id

        self.model = SecureDSC(cfg.model, len(self.vocab), cfg.data.max_len).to(
            self.device
        )
        self.channel: Channel = make_channel(cfg.channel.kind).to(self.device)

        # Key generator (+ reciprocal CSI source if CSI-based).
        self.keygen: KeyGenerator = make_key_generator(
            cfg.keygen, cfg.model.key_len, cfg.channel.csi_dim
        ).to(self.device)
        self.is_csi = isinstance(self.keygen, CSIKeyGenerator)
        self.recip: Optional[ReciprocalChannel] = (
            ReciprocalChannel(cfg.channel.csi_dim, cfg.channel.csi_rho, cfg.channel.csi_snr_db)
            if self.is_csi
            else None
        )

        self.lam_sched: LambdaScheduler = make_lambda_scheduler(cfg.lam, len(self.vocab))

        # Optimizers (one per Algorithm-1 sub-step).
        t = cfg.train
        self.opt_sem = torch.optim.Adam(self.model.semantic_params(), lr=t.lr_sub)
        self.opt_cip = torch.optim.Adam(self.model.cipher_params(), lr=t.lr_sub)
        self.opt_joint = torch.optim.Adam(self.model.alice_bob_params(), lr=t.lr_joint)
        self.opt_eve = torch.optim.Adam(self.model.eve_params(), lr=t.lr_eve)
        self.opt_key = (
            torch.optim.Adam(self.keygen.parameters(), lr=t.lr_sub)
            if self.is_csi
            else None
        )

        self.history: List[EpochStats] = []

    # ------------------------------------------------------------------ #
    def _make_bundle(self, batch_size: int) -> KeyBundle:
        """Draw a key bundle for one batch (sampling CSI if CSI-based)."""
        csi = None
        if self.is_csi:
            assert self.recip is not None
            csi = self.recip.sample(batch_size, device=self.device)
        return self.keygen.generate(batch_size, self.device, csi=csi)

    def _train_batch(self, tokens: torch.Tensor, snr_db: float) -> Dict[str, float]:
        m = self.model
        bundle = self._make_bundle(tokens.shape[0])
        stats: Dict[str, float] = {}

        # (i) semantic enc/dec
        m.zero_grad(set_to_none=True)
        l_sem = ce_loss(m.forward_semantic(tokens), tokens, self.pad_id)
        l_sem.backward()
        self._clip(m.semantic_params())
        self.opt_sem.step()
        stats["l_sem"] = l_sem.item()

        # (ii) encryptor/decryptor
        m.zero_grad(set_to_none=True)
        recovered, target = m.forward_cipher(tokens, bundle, self.channel, snr_db)
        l_cip = F.mse_loss(recovered, target)
        l_cip.backward()
        self._clip(m.cipher_params())
        self.opt_cip.step()
        stats["l_cip"] = l_cip.item()

        # (iii) whole network with L_joint
        m.zero_grad(set_to_none=True)
        out = m.forward_full(tokens, bundle, self.channel, snr_db)
        l_b = ce_loss(out.bob_logits, tokens, self.pad_id)
        l_e = ce_loss(out.eve_logits, tokens, self.pad_id)
        lam = self.lam_sched.get()
        l_joint = l_b + torch.abs(l_e - lam)
        l_joint.backward()
        self._clip(m.alice_bob_params())
        self.opt_joint.step()
        stats["l_b"], stats["l_e"] = l_b.item(), l_e.item()

        # (iv) Eve does her best
        m.zero_grad(set_to_none=True)
        out_e = m.forward_full(tokens, bundle, self.channel, snr_db)
        l_e2 = ce_loss(out_e.eve_logits, tokens, self.pad_id)
        l_e2.backward()
        self._clip(m.eve_params())
        self.opt_eve.step()

        # (v) CSI key generator agreement
        if self.is_csi and self.opt_key is not None:
            assert isinstance(self.keygen, CSIKeyGenerator)
            key_bundle = self._make_bundle(tokens.shape[0])
            l_key = self.keygen.agreement_loss(key_bundle)
            self.opt_key.zero_grad(set_to_none=True)
            l_key.backward()
            self.opt_key.step()
            stats["l_key"] = l_key.item()
            stats["ab_agreement"] = key_bundle.ab_agreement
            stats["eve_mismatch"] = key_bundle.eve_mismatch
        else:
            stats["ab_agreement"] = bundle.ab_agreement
            stats["eve_mismatch"] = bundle.eve_mismatch

        # update lambda from the gap
        self.lam_sched.step(stats["l_b"], stats["l_e"])
        stats["lam"] = self.lam_sched.get()
        return stats

    def _clip(self, params) -> None:
        if self.cfg.train.grad_clip and self.cfg.train.grad_clip > 0:
            nn.utils.clip_grad_norm_(params, self.cfg.train.grad_clip)

    # ------------------------------------------------------------------ #
    def train(self) -> List[EpochStats]:
        """Run the full training schedule, returning per-epoch stats."""
        cfg = self.cfg
        snr = cfg.channel.train_snr_db
        self.logger.info(
            "Training '%s' | device=%s vocab=%d keygen=%s lambda=%s",
            cfg.train.run_name, self.device, len(self.vocab),
            cfg.keygen.kind, cfg.lam.kind,
        )
        for epoch in range(1, cfg.train.epochs + 1):
            self.model.train()
            agg: Dict[str, float] = {}
            n = 0
            t0 = time.time()
            for step, tokens in enumerate(self.train_loader):
                tokens = tokens.to(self.device)
                s = self._train_batch(tokens, snr)
                for k, v in s.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
                if (step + 1) % cfg.train.log_every == 0:
                    self.logger.info(
                        "  e%d s%d | L_sem=%.3f L_cip=%.3f L_B=%.3f L_E=%.3f lam=%.3f",
                        epoch, step + 1, s["l_sem"], s["l_cip"], s["l_b"],
                        s["l_e"], s["lam"],
                    )
            stats = EpochStats(epoch=epoch)
            for k in ("l_sem", "l_cip", "l_b", "l_e", "lam", "l_key",
                      "ab_agreement", "eve_mismatch"):
                if k in agg:
                    setattr(stats, k, agg[k] / max(1, n))
            bob_bleu, eve_bleu = self.evaluate_bleu(snr, num_batches=4)
            stats.val_bleu_bob, stats.val_bleu_eve = bob_bleu, eve_bleu
            self.history.append(stats)
            self.logger.info(
                "[epoch %d/%d] L_B=%.3f L_E=%.3f lam=%.3f BLEU(Bob)=%.3f "
                "BLEU(Eve)=%.3f agree=%.3f (%.1fs)",
                epoch, cfg.train.epochs, stats.l_b, stats.l_e, stats.lam,
                bob_bleu, eve_bleu, stats.ab_agreement, time.time() - t0,
            )
        return self.history

    @torch.no_grad()
    def evaluate_bleu(self, snr_db: float, num_batches: int = 8):
        """Quick BLEU(Bob)/BLEU(Eve) estimate at a single SNR (validation set)."""
        from .eval import bleu_for_loader

        return bleu_for_loader(
            self.model, self.val_loader, self.channel, snr_db, self.vocab,
            self.device, self._make_bundle, num_batches=num_batches,
        )

    # ------------------------------------------------------------------ #
    def save(self, path: Optional[str] = None) -> str:
        """Save model + vocab + config to ``out_dir/run_name/ckpt.pt``."""
        run_dir = ensure_dir(os.path.join(self.cfg.train.out_dir, self.cfg.train.run_name))
        path = path or os.path.join(run_dir, "ckpt.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "keygen": self.keygen.state_dict(),
                "vocab_itos": self.vocab.itos,
                "config": asdict(self.cfg),
                "history": [asdict(h) for h in self.history],
            },
            path,
        )
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump([asdict(h) for h in self.history], fh, indent=2)
        self.logger.info("Saved checkpoint -> %s", path)
        return path
