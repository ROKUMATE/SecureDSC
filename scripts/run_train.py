#!/usr/bin/env python3
"""Train one SecureDSC configuration.

Usage:
    python -m scripts.run_train --config configs/base_awgn.yaml
    python -m scripts.run_train --config configs/smoke.yaml --epochs 2
"""

from __future__ import annotations

import argparse

from securedsc.config import load_config
from securedsc.train import SecureDSCTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Train SecureDSC")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--seed", type=int, default=None, help="override train.seed")
    p.add_argument("--run-name", type=str, default=None, help="override train.run_name")
    p.add_argument("--keygen", choices=["random", "csi"], default=None)
    p.add_argument("--lam", choices=["fixed", "adaptive"], default=None)
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.run_name is not None:
        cfg.train.run_name = args.run_name
    if args.keygen is not None:
        cfg.keygen.kind = args.keygen
    if args.lam is not None:
        cfg.lam.kind = args.lam
    if args.channel is not None:
        cfg.channel.kind = args.channel

    trainer = SecureDSCTrainer(cfg)
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
