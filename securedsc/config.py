"""Configuration dataclasses and YAML loading for SecureDSC.

A single nested :class:`Config` drives the whole project. Every ambiguous
architectural / training choice is exposed here so that experiments are fully
config-driven and reproducible. See ``configs/*.yaml`` for examples.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset / vocabulary settings."""

    source: str = "toy"  # "toy" | "europarl"
    path: Optional[str] = None  # path to a raw text file (one sentence per line)
    max_len: int = 30  # sentence length (tokens) after padding
    min_len: int = 4  # drop sentences shorter than this (real corpora)
    max_vocab: int = 10000  # cap vocabulary size (most frequent kept)
    max_sentences: Optional[int] = None  # subsample corpus for speed
    val_fraction: float = 0.1


@dataclass
class ModelConfig:
    """SecureDSC network dimensions."""

    d_model: int = 128
    nhead: int = 8
    dim_feedforward: int = 512
    num_sem_layers: int = 3  # semantic encoder / decoder depth
    num_cipher_layers: int = 2  # encryptor / decryptor depth
    channel_dim: int = 16  # channel symbols per token
    dropout: float = 0.1
    key_len: int = 64  # raw key vector length
    n_key_tokens: int = 4  # key tokens the cipher cross-attends to


@dataclass
class ChannelConfig:
    """Wireless channel settings used during training."""

    kind: str = "awgn"  # "awgn" | "rayleigh"
    train_snr_db: float = 12.0  # SNR used while training
    # Reciprocal-CSI helper (Improvement A):
    csi_rho: float = 0.95  # Alice<->Bob CSI correlation coefficient
    csi_dim: int = 32  # dimensionality of a CSI estimate vector
    csi_snr_db: float = 15.0  # estimation SNR for CSI observations


@dataclass
class KeyGenConfig:
    """Key generator selection (Improvement A toggle)."""

    kind: str = "random"  # "random" | "csi"
    csi_hidden: int = 128  # CSIKeyGenerator MLP hidden size
    recon_repeat: int = 3  # bit repetition for reconciliation (odd number)
    bits: int = 64  # number of key bits produced before key-proc


@dataclass
class LambdaConfig:
    """Lambda scheduler selection (Improvement B toggle).

    NOTE on direction: ``L_joint = L_B + |L_E - lambda|`` pulls Eve's loss
    *toward* ``lambda``. For security you want Eve's loss HIGH, so ``lambda`` must
    be a high target (~``ln(vocab)`` = random-guess loss). A small ``lambda``
    trains Alice to *help* Eve and destroys secrecy. Because the right absolute
    value is corpus-dependent (it scales with ``ln(vocab)``), tuning it by hand is
    fiddly -- which is exactly what the adaptive scheduler removes.
    """

    kind: str = "fixed"  # "fixed" | "adaptive"
    value: float = 6.0  # fixed lambda target (set ~0.9*ln(vocab) for your corpus)
    target_frac: float = 0.9  # adaptive: target Eve loss = target_frac * ln(vocab)
    eta: float = 0.05  # adaptive: feedback step size
    ema: float = 0.9  # adaptive: EMA factor for tracked L_E
    min_value: float = 0.0
    max_value: Optional[float] = None  # default: ln(vocab) (set at runtime)


@dataclass
class TrainConfig:
    """Optimisation / schedule settings."""

    epochs: int = 20
    batch_size: int = 64
    lr_joint: float = 1e-4
    lr_sub: float = 1e-3  # lr for the L_sem / L_cip sub-steps
    lr_eve: float = 1e-3
    grad_clip: float = 1.0
    log_every: int = 50  # steps
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    seed: int = 0
    out_dir: str = "results"
    run_name: str = "run"


@dataclass
class EvalConfig:
    """Evaluation sweeps."""

    snr_db_list: list = field(
        default_factory=lambda: [0, 3, 6, 9, 12, 15, 18]
    )
    rho_list: list = field(
        default_factory=lambda: [0.5, 0.7, 0.9, 0.95, 0.99]
    )
    num_batches: int = 8  # eval batches per operating point
    bleu_ngrams: int = 4


@dataclass
class Config:
    """Top-level config aggregating every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    keygen: KeyGenConfig = field(default_factory=KeyGenConfig)
    lam: LambdaConfig = field(default_factory=LambdaConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# --------------------------------------------------------------------------- #
# (de)serialisation helpers
# --------------------------------------------------------------------------- #
_SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "channel": ChannelConfig,
    "keygen": KeyGenConfig,
    "lam": LambdaConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _build_section(cls: type, values: Optional[Dict[str, Any]]):
    if not values:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(d: Dict[str, Any]) -> Config:
    """Build a :class:`Config` from a (possibly partial) nested dict."""
    unknown = set(d) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(
        **{
            name: _build_section(cls, d.get(name))
            for name, cls in _SECTION_TYPES.items()
        }
    )


def load_config(path: str) -> Config:
    """Load a YAML file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Serialise a :class:`Config` back to a plain dict (for logging)."""
    return dataclasses.asdict(cfg)
