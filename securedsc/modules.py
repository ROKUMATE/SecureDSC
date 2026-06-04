"""Neural building blocks for SecureDSC.

All decoders are *non-autoregressive* (predict the whole sentence at once), as in
DeepSC. The encryptor and decryptor are Transformer **decoder** stacks whose
cross-attention attends to *key tokens* produced by the key-processing network —
this is what makes a correct key invert the cipher and a wrong key fail.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderStack(nn.Module):
    """A stack of Transformer encoder layers (batch-first)."""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class CrossAttnDecoderStack(nn.Module):
    """A stack of Transformer decoder layers that cross-attend to ``memory``.

    Used for both the encryptor and decryptor: ``tgt`` is the feature sequence,
    ``memory`` is the sequence of key tokens.
    """

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Non-autoregressive: no causal mask (full self-attention over the seq).
        return self.decoder(tgt, memory)


class KeyProcessingNetwork(nn.Module):
    """Maps a key bit-vector ``(B, key_dim)`` to key tokens ``(B, n_tokens, d)``."""

    def __init__(self, key_dim: int, d_model: int, n_tokens: int, hidden: int = 256):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(key_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_tokens * d_model),
        )

    def forward(self, key: torch.Tensor) -> torch.Tensor:
        out = self.net(key)
        return out.view(key.shape[0], self.n_tokens, self.d_model)


class ChannelEncoder(nn.Module):
    """Dense channel encoder: ``d_model -> channel_dim`` per token, power-normalised."""

    def __init__(self, d_model: int, channel_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, channel_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChannelDecoder(nn.Module):
    """Dense channel decoder: ``channel_dim -> d_model`` per token."""

    def __init__(self, d_model: int, channel_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
