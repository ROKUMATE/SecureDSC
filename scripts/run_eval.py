#!/usr/bin/env python3
"""Evaluate a trained SecureDSC checkpoint: BLEU-vs-SNR (AWGN + Rayleigh) and,
for CSI keys, the key-agreement-vs-correlation sweep. Writes JSON + plots to
``results/<run_name>/``.

Usage:
    python -m scripts.run_eval --ckpt results/base_awgn/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from securedsc.channels import make_channel
from securedsc.config import config_from_dict
from securedsc.data import Vocabulary, build_dataloaders
from securedsc.eval import (
    bleu_vs_snr,
    key_metrics_vs_correlation,
    plot_bleu_vs_snr,
    plot_key_vs_correlation,
)
from securedsc.keygen import CSIKeyGenerator, make_key_generator
from securedsc.channels import ReciprocalChannel
from securedsc.model import SecureDSC
from securedsc.utils import ensure_dir, get_logger, resolve_device


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    vocab = Vocabulary(ckpt["vocab_itos"])
    # special tokens already embedded in itos; rebuild stoi-consistent object
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = SecureDSC(cfg.model, len(vocab), cfg.data.max_len)
    model.load_state_dict(ckpt["model"])
    keygen = make_key_generator(cfg.keygen, cfg.model.key_len, cfg.channel.csi_dim)
    keygen.load_state_dict(ckpt["keygen"])
    return cfg, vocab, model, keygen


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SecureDSC")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-batches", type=int, default=8)
    args = p.parse_args()
    logger = get_logger()

    cfg, vocab, model, keygen = load_checkpoint(args.ckpt)
    device = resolve_device(cfg.train.device)
    model = model.to(device).eval()
    keygen = keygen.to(device).eval()

    _, val_loader, _ = build_dataloaders(cfg.data, cfg.train.batch_size, cfg.train.seed)

    is_csi = isinstance(keygen, CSIKeyGenerator)
    recip = (
        ReciprocalChannel(cfg.channel.csi_dim, cfg.channel.csi_rho, cfg.channel.csi_snr_db)
        if is_csi
        else None
    )

    def make_bundle(bs: int):
        csi = recip.sample(bs, device=device) if is_csi else None
        return keygen.generate(bs, device, csi=csi)

    out_dir = ensure_dir(os.path.join(cfg.train.out_dir, cfg.train.run_name))
    results = {}
    sweeps = {}
    for kind in ("awgn", "rayleigh"):
        channel = make_channel(kind).to(device)
        sweep = bleu_vs_snr(
            model, val_loader, channel, cfg.eval.snr_db_list, vocab, device,
            make_bundle, num_batches=args.num_batches, max_n=cfg.eval.bleu_ngrams,
        )
        results[f"bleu_vs_snr_{kind}"] = sweep
        sweeps[kind.upper()] = sweep
        logger.info("%s BLEU(Bob)=%s", kind, [round(x, 3) for x in sweep["bob"]])
        logger.info("%s BLEU(Eve)=%s", kind, [round(x, 3) for x in sweep["eve"]])

    plot_bleu_vs_snr(
        sweeps, os.path.join(out_dir, "bleu_vs_snr.png"),
        title=f"BLEU vs SNR ({cfg.train.run_name})",
    )

    if is_csi:
        ks = key_metrics_vs_correlation(
            keygen, cfg.eval.rho_list, cfg.channel.csi_dim, cfg.channel.csi_snr_db,
            cfg.train.batch_size, device, num_batches=args.num_batches,
        )
        results["key_vs_correlation"] = ks
        plot_key_vs_correlation(ks, os.path.join(out_dir, "key_vs_correlation.png"))
        logger.info("key agreement vs rho: %s", ks)

    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote results -> %s", out_dir)


if __name__ == "__main__":
    main()
