#!/usr/bin/env python3
"""Full ablation grid: {random key, CSI key} x {fixed lambda, adaptive lambda}.

For each cell, train over multiple seeds and report mean/std of final BLEU(Bob),
BLEU(Eve), the Bob-Eve gap, Alice/Bob key agreement and Eve mismatch. Also emits
training-curve and BLEU-vs-SNR comparison plots. Writes everything under
``results/ablation/``.

Usage:
    python -m scripts.run_ablation --config configs/base_awgn.yaml --seeds 0 1 2
    python -m scripts.run_ablation --config configs/smoke.yaml --seeds 0   # quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import asdict
from typing import Dict, List

from securedsc.channels import make_channel
from securedsc.config import load_config
from securedsc.eval import bleu_vs_snr, plot_bleu_vs_snr, plot_training_curves
from securedsc.train import SecureDSCTrainer
from securedsc.utils import ensure_dir, get_logger

CELLS = [
    ("random", "fixed"),
    ("random", "adaptive"),
    ("csi", "fixed"),
    ("csi", "adaptive"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="SecureDSC ablation grid")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    logger = get_logger()

    out_dir = ensure_dir("results/ablation")
    rows: List[Dict] = []
    curves: Dict[str, List[dict]] = {}
    snr_sweeps: Dict[str, dict] = {}

    for keygen_kind, lam_kind in CELLS:
        per_seed: List[Dict] = []
        rep_history = None
        rep_trainer = None
        for seed in args.seeds:
            cfg = load_config(args.config)
            cfg.keygen.kind = keygen_kind
            cfg.lam.kind = lam_kind
            cfg.train.seed = seed
            cfg.train.run_name = f"abl_{keygen_kind}_{lam_kind}_s{seed}"
            if args.epochs is not None:
                cfg.train.epochs = args.epochs

            trainer = SecureDSCTrainer(cfg)
            history = trainer.train()
            final = history[-1]
            per_seed.append(
                {
                    "bleu_bob": final.val_bleu_bob,
                    "bleu_eve": final.val_bleu_eve,
                    "gap": final.val_bleu_bob - final.val_bleu_eve,
                    "ab_agreement": final.ab_agreement,
                    "eve_mismatch": final.eve_mismatch,
                    "l_b": final.l_b,
                    "l_e": final.l_e,
                }
            )
            if rep_history is None:
                rep_history = [asdict(h) for h in history]
                rep_trainer = trainer

        # aggregate across seeds
        def agg(key: str):
            vals = [d[key] for d in per_seed]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return mean, std

        row = {"keygen": keygen_kind, "lambda": lam_kind, "seeds": len(args.seeds)}
        for key in ("bleu_bob", "bleu_eve", "gap", "ab_agreement", "eve_mismatch"):
            m, s = agg(key)
            row[f"{key}_mean"] = round(m, 4)
            row[f"{key}_std"] = round(s, 4)
        rows.append(row)
        logger.info(
            "[%s/%s] BLEU Bob=%.3f+-%.3f Eve=%.3f+-%.3f gap=%.3f agree=%.3f",
            keygen_kind, lam_kind, row["bleu_bob_mean"], row["bleu_bob_std"],
            row["bleu_eve_mean"], row["bleu_eve_std"], row["gap_mean"],
            row["ab_agreement_mean"],
        )

        label = f"{keygen_kind}+{lam_kind}"
        curves[label] = rep_history
        # BLEU-vs-SNR for the representative (first-seed) model, AWGN.
        if rep_trainer is not None:
            channel = make_channel("awgn").to(rep_trainer.device)
            snr_sweeps[label] = bleu_vs_snr(
                rep_trainer.model, rep_trainer.val_loader, channel,
                rep_trainer.cfg.eval.snr_db_list, rep_trainer.vocab,
                rep_trainer.device, rep_trainer._make_bundle, num_batches=4,
            )

    # write CSV table
    csv_path = os.path.join(out_dir, "ablation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "ablation_table.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    # comparison plots
    plot_training_curves(curves, os.path.join(out_dir, "training_curves.png"),
                         title="Ablation training curves")
    plot_bleu_vs_snr(snr_sweeps, os.path.join(out_dir, "ablation_bleu_vs_snr.png"),
                     title="Ablation: BLEU vs SNR (AWGN)")

    logger.info("Ablation complete -> %s", out_dir)
    _print_markdown_table(rows, logger)


def _print_markdown_table(rows: List[Dict], logger) -> None:
    header = "| key | lambda | BLEU(Bob) | BLEU(Eve) | gap | agree |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['keygen']} | {r['lambda']} "
            f"| {r['bleu_bob_mean']:.3f}±{r['bleu_bob_std']:.3f} "
            f"| {r['bleu_eve_mean']:.3f}±{r['bleu_eve_std']:.3f} "
            f"| {r['gap_mean']:.3f} | {r['ab_agreement_mean']:.3f} |"
        )
    logger.info("Ablation table:\n%s", "\n".join(lines))


if __name__ == "__main__":
    main()
