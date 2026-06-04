"""Evaluation: BLEU-vs-SNR sweeps, correlation sweeps and plotting helpers."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .channels import Channel, ReciprocalChannel
from .data import Vocabulary
from .keygen import KeyBundle
from .metrics import corpus_bleu
from .model import SecureDSC
from .utils import ensure_dir

BundleFn = Callable[[int], KeyBundle]


@torch.no_grad()
def _decode_batch(
    logits: torch.Tensor, vocab: Vocabulary
) -> List[List[str]]:
    ids = logits.argmax(dim=-1)  # (B, L)
    return [vocab.decode(row.tolist()) for row in ids]


@torch.no_grad()
def bleu_for_loader(
    model: SecureDSC,
    loader,
    channel: Channel,
    snr_db: float,
    vocab: Vocabulary,
    device: torch.device,
    make_bundle: BundleFn,
    num_batches: int = 8,
    max_n: int = 4,
) -> Tuple[float, float]:
    """Return ``(BLEU_Bob, BLEU_Eve)`` averaged over up to ``num_batches`` batches."""
    model.eval()
    refs: List[List[str]] = []
    bob_hyp: List[List[str]] = []
    eve_hyp: List[List[str]] = []
    for i, tokens in enumerate(loader):
        if i >= num_batches:
            break
        tokens = tokens.to(device)
        bundle = make_bundle(tokens.shape[0])
        out = model.forward_full(tokens, bundle, channel, snr_db)
        refs.extend(vocab.decode(row.tolist()) for row in tokens)
        bob_hyp.extend(_decode_batch(out.bob_logits, vocab))
        eve_hyp.extend(_decode_batch(out.eve_logits, vocab))
    if not refs:
        return 0.0, 0.0
    return (
        corpus_bleu(refs, bob_hyp, max_n=max_n),
        corpus_bleu(refs, eve_hyp, max_n=max_n),
    )


@torch.no_grad()
def bleu_vs_snr(
    model: SecureDSC,
    loader,
    channel: Channel,
    snr_list: List[float],
    vocab: Vocabulary,
    device: torch.device,
    make_bundle: BundleFn,
    num_batches: int = 8,
    max_n: int = 4,
) -> Dict[str, List[float]]:
    """Sweep SNR, returning ``{snr, bob, eve}`` lists for plotting."""
    out = {"snr": [], "bob": [], "eve": []}
    for snr in snr_list:
        bob, eve = bleu_for_loader(
            model, loader, channel, snr, vocab, device, make_bundle,
            num_batches=num_batches, max_n=max_n,
        )
        out["snr"].append(float(snr))
        out["bob"].append(bob)
        out["eve"].append(eve)
    return out


@torch.no_grad()
def key_metrics_vs_correlation(
    keygen,
    rho_list: List[float],
    csi_dim: int,
    csi_snr_db: float,
    batch_size: int,
    device: torch.device,
    num_batches: int = 8,
) -> Dict[str, List[float]]:
    """Sweep the Alice<->Bob CSI correlation, reporting key agreement / Eve mismatch.

    Only meaningful for the CSI key generator.
    """
    out = {"rho": [], "ab_agreement": [], "eve_mismatch": []}
    for rho in rho_list:
        recip = ReciprocalChannel(csi_dim, rho, csi_snr_db)
        agree, mism = [], []
        for _ in range(num_batches):
            csi = recip.sample(batch_size, device=device)
            bundle = keygen.generate(batch_size, device, csi=csi)
            agree.append(bundle.ab_agreement)
            mism.append(bundle.eve_mismatch)
        out["rho"].append(float(rho))
        out["ab_agreement"].append(sum(agree) / len(agree))
        out["eve_mismatch"].append(sum(mism) / len(mism))
    return out


# --------------------------------------------------------------------------- #
# Plotting (matplotlib; saved to results/)
# --------------------------------------------------------------------------- #
def plot_bleu_vs_snr(
    sweeps: Dict[str, Dict[str, List[float]]], out_path: str, title: str = "BLEU vs SNR"
) -> str:
    """Plot one or more BLEU-vs-SNR curves. ``sweeps`` maps a label -> sweep dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, sw in sweeps.items():
        ax.plot(sw["snr"], sw["bob"], marker="o", label=f"{label} Bob")
        ax.plot(sw["snr"], sw["eve"], marker="x", linestyle="--", label=f"{label} Eve")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLEU (1-4 gram)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_training_curves(
    histories: Dict[str, List[dict]], out_path: str, title: str = "Training curves"
) -> str:
    """Plot L_B / L_E / lambda across epochs for one or more runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path) or ".")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["l_b"] for h in hist], marker="o", label=label)
        axes[1].plot(ep, [h["l_e"] for h in hist], marker="o", label=label)
        axes[2].plot(ep, [h["lam"] for h in hist], marker="o", label=label)
    for ax, name in zip(axes, ("L_B (Bob)", "L_E (Eve)", "lambda")):
        ax.set_xlabel("epoch")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_key_vs_correlation(sweep: Dict[str, List[float]], out_path: str) -> str:
    """Plot Alice/Bob agreement and Eve mismatch vs CSI correlation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["rho"], sweep["ab_agreement"], marker="o", label="Alice-Bob agreement")
    ax.plot(sweep["rho"], sweep["eve_mismatch"], marker="x", label="Eve mismatch")
    ax.set_xlabel("CSI correlation rho")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("CSI key quality vs channel correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
