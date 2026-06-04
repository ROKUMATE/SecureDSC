"""The SecureDSC model: Alice/Bob shared network + Eve's eavesdropper network.

Exposes the three forward paths needed by Algorithm 1 (semantic-only, cipher-only,
and full Bob+Eve) plus parameter groups so the trainer can update each sub-network
in turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
import torch.nn as nn

from .channels import Channel, normalize_power
from .config import ModelConfig
from .keygen import KeyBundle
from .modules import (
    ChannelDecoder,
    ChannelEncoder,
    CrossAttnDecoderStack,
    KeyProcessingNetwork,
    PositionalEncoding,
    TransformerEncoderStack,
)


@dataclass
class FullOutput:
    """Logits from a full forward pass."""

    bob_logits: torch.Tensor  # (B, L, V)
    eve_logits: torch.Tensor  # (B, L, V)


class SecureDSC(nn.Module):
    """End-to-end secure semantic communication model with an eavesdropper."""

    def __init__(self, cfg: ModelConfig, vocab_size: int, max_len: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.d_model

        # ---- Alice/Bob shared transmitter+receiver ----
        self.embedding = nn.Embedding(vocab_size, d)
        self.pos_enc = PositionalEncoding(d, max_len=max_len, dropout=cfg.dropout)
        self.semantic_encoder = TransformerEncoderStack(
            d, cfg.nhead, cfg.num_sem_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.key_proc = KeyProcessingNetwork(cfg.key_len, d, cfg.n_key_tokens)
        self.encryptor = CrossAttnDecoderStack(
            d, cfg.nhead, cfg.num_cipher_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.channel_encoder = ChannelEncoder(d, cfg.channel_dim)
        self.channel_decoder = ChannelDecoder(d, cfg.channel_dim)
        self.decryptor = CrossAttnDecoderStack(
            d, cfg.nhead, cfg.num_cipher_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.semantic_decoder = TransformerEncoderStack(
            d, cfg.nhead, cfg.num_sem_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.prediction = nn.Linear(d, vocab_size)

        # ---- Eve's eavesdropper (own params, same shape) ----
        self.eve_channel_decoder = ChannelDecoder(d, cfg.channel_dim)
        self.eve_key_proc = KeyProcessingNetwork(cfg.key_len, d, cfg.n_key_tokens)
        self.eve_decryptor = CrossAttnDecoderStack(
            d, cfg.nhead, cfg.num_cipher_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.eve_semantic_decoder = TransformerEncoderStack(
            d, cfg.nhead, cfg.num_sem_layers, cfg.dim_feedforward, cfg.dropout
        )
        self.eve_prediction = nn.Linear(d, vocab_size)

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.embedding(tokens))

    def _semantic_encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.semantic_encoder(self._embed(tokens))

    def _transmit(
        self,
        cipher: torch.Tensor,
        channel: Channel,
        snr_db: float,
    ) -> torch.Tensor:
        """Channel-encode -> power-normalise -> channel -> channel-decode."""
        x = normalize_power(self.channel_encoder(cipher))
        y = channel(x, snr_db)
        return y  # raw received symbols (Eve also taps these)

    # ------------------------------------------------------------------ #
    # Forward paths (one per Algorithm-1 sub-step)
    # ------------------------------------------------------------------ #
    def forward_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Semantic autoencoder path (for ``L_sem``): no cipher, no channel."""
        sem = self._semantic_encode(tokens)
        dec = self.semantic_decoder(sem)
        return self.prediction(dec)

    def forward_cipher(
        self,
        tokens: torch.Tensor,
        bundle: KeyBundle,
        channel: Channel,
        snr_db: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cipher reconstruction path (for ``L_cip``).

        Returns ``(recovered_features, target_features)``; the decryptor should
        invert the encryptor through the channel given the correct key.
        """
        sem = self._semantic_encode(tokens).detach()  # fix semantic codec here
        key_a = self.key_proc(bundle.alice)
        cipher = self.encryptor(sem, key_a)
        y = self._transmit(cipher, channel, snr_db)
        feat = self.channel_decoder(y)
        key_b = self.key_proc(bundle.bob)
        recovered = self.decryptor(feat, key_b)
        return recovered, sem

    def forward_full(
        self,
        tokens: torch.Tensor,
        bundle: KeyBundle,
        channel: Channel,
        snr_db: float,
    ) -> FullOutput:
        """Full Bob path + Eve's eavesdropping path on the same channel output."""
        sem = self._semantic_encode(tokens)
        key_a = self.key_proc(bundle.alice)
        cipher = self.encryptor(sem, key_a)
        y = self._transmit(cipher, channel, snr_db)

        # Bob
        feat = self.channel_decoder(y)
        key_b = self.key_proc(bundle.bob)
        dec = self.decryptor(feat, key_b)
        sem_dec = self.semantic_decoder(dec)
        bob_logits = self.prediction(sem_dec)

        # Eve (own params, own key)
        efeat = self.eve_channel_decoder(y)
        ekey = self.eve_key_proc(bundle.eve)
        edec = self.eve_decryptor(efeat, ekey)
        esem = self.eve_semantic_decoder(edec)
        eve_logits = self.eve_prediction(esem)

        return FullOutput(bob_logits=bob_logits, eve_logits=eve_logits)

    # ------------------------------------------------------------------ #
    # Parameter groups for Algorithm 1
    # ------------------------------------------------------------------ #
    def _params(self, modules: list) -> Iterator[nn.Parameter]:
        for m in modules:
            yield from m.parameters()

    def semantic_params(self) -> list:
        """Params updated by the ``L_sem`` sub-step."""
        return list(
            self._params(
                [
                    self.embedding,
                    self.semantic_encoder,
                    self.semantic_decoder,
                    self.prediction,
                ]
            )
        )

    def cipher_params(self) -> list:
        """Params updated by the ``L_cip`` sub-step."""
        return list(
            self._params([self.key_proc, self.encryptor, self.decryptor])
        )

    def alice_bob_params(self) -> list:
        """All Alice/Bob params updated by the ``L_joint`` sub-step."""
        eve_modules = {
            self.eve_channel_decoder,
            self.eve_key_proc,
            self.eve_decryptor,
            self.eve_semantic_decoder,
            self.eve_prediction,
        }
        return [
            p
            for name, m in self.named_children()
            if m not in eve_modules
            for p in m.parameters()
        ]

    def eve_params(self) -> list:
        """All Eve params updated by the ``L_E`` sub-step."""
        return list(
            self._params(
                [
                    self.eve_channel_decoder,
                    self.eve_key_proc,
                    self.eve_decryptor,
                    self.eve_semantic_decoder,
                    self.eve_prediction,
                ]
            )
        )
