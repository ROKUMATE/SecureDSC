"""Text dataset, vocabulary and data loading.

Supports a built-in toy corpus (for CPU smoke tests) and a generic line-based
corpus loader (default target: a subset of Europarl ``en``). Sentences are
whitespace-tokenised, capped to ``max_len`` with ``<start>``/``<end>``/``<pad>``
markers, and unknown tokens mapped to ``<unk>``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD, START, END, UNK]


# A small, self-contained English corpus for smoke tests. Repeated/varied simple
# sentences so the model can actually learn to reconstruct them quickly.
TOY_SENTENCES: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "the early bird catches the worm in the morning",
    "knowledge is power and power comes with responsibility",
    "the sun rises in the east and sets in the west",
    "she sells sea shells by the sea shore",
    "all that glitters is not gold in this world",
    "actions speak louder than words every single day",
    "a picture is worth a thousand spoken words",
    "practice makes a person perfect over a long time",
    "honesty is the best policy in every situation",
    "the pen is mightier than the sharp sword",
    "where there is a will there is a way forward",
    "birds of a feather always flock together closely",
    "time and tide wait for no man on earth",
    "an apple a day keeps the doctor away nicely",
    "do not count your chickens before they are hatched",
    "every cloud has a silver lining behind it",
    "fortune favours the brave and the bold hearts",
    "rome was not built in a single short day",
]


class Vocabulary:
    """Token <-> id mapping with special tokens at fixed indices."""

    def __init__(self, tokens: Sequence[str]):
        # Special tokens always first so PAD == 0.
        self.itos: List[str] = list(SPECIAL_TOKENS) + [
            t for t in tokens if t not in SPECIAL_TOKENS
        ]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def start_id(self) -> int:
        return self.stoi[START]

    @property
    def end_id(self) -> int:
        return self.stoi[END]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, sentence: str, max_len: int) -> List[int]:
        """Encode a sentence to a fixed-length id list with start/end/pad."""
        words = sentence.strip().split()
        ids = [self.start_id]
        for w in words[: max_len - 2]:
            ids.append(self.stoi.get(w, self.unk_id))
        ids.append(self.end_id)
        ids += [self.pad_id] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
        """Decode ids to tokens, optionally dropping special markers."""
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if strip_special and tok in (PAD, START, END):
                if tok == END:
                    break
                continue
            out.append(tok)
        return out

    @classmethod
    def build(cls, sentences: Sequence[str], max_vocab: int) -> "Vocabulary":
        counter: Counter = Counter()
        for s in sentences:
            counter.update(s.strip().split())
        # most frequent first, leaving room for the special tokens
        most_common = [w for w, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS))]
        return cls(most_common)


class TextDataset(Dataset):
    """A dataset of fixed-length token-id tensors."""

    def __init__(self, sentences: Sequence[str], vocab: Vocabulary, max_len: int):
        self.vocab = vocab
        self.max_len = max_len
        self.data = torch.tensor(
            [vocab.encode(s, max_len) for s in sentences], dtype=torch.long
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_raw_sentences(cfg: DataConfig) -> List[str]:
    if cfg.source == "toy":
        # Repeat the toy corpus so there are enough samples for batching.
        return TOY_SENTENCES * 20
    if cfg.source == "europarl":
        if not cfg.path:
            raise ValueError(
                "data.path must point to a line-based text file for source='europarl'. "
                "Run `python -m scripts.get_europarl` to download/prepare one."
            )
        sentences = []
        with open(cfg.path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                words = _clean_line(line).split()
                if cfg.min_len <= len(words) <= cfg.max_len - 2:
                    sentences.append(" ".join(words))
        if not sentences:
            raise ValueError(f"No sentences in {cfg.path} within length bounds")
        return sentences
    raise ValueError(f"Unknown data.source: {cfg.source}")


_CLEAN_RE = re.compile(r"[^a-z0-9\s]")


def _clean_line(line: str) -> str:
    """Lower-case and strip punctuation (DeepSC-style preprocessing)."""
    line = line.lower()
    line = _CLEAN_RE.sub(" ", line)
    return re.sub(r"\s+", " ", line).strip()


def build_dataloaders(
    cfg: DataConfig, batch_size: int, seed: int = 0
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """Build train/val loaders and the shared vocabulary."""
    sentences = _load_raw_sentences(cfg)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(sentences), generator=g).tolist()
    sentences = [sentences[i] for i in perm]
    if cfg.max_sentences is not None:
        sentences = sentences[: cfg.max_sentences]

    n_val = max(1, int(len(sentences) * cfg.val_fraction))
    val_sents = sentences[:n_val]
    train_sents = sentences[n_val:] or sentences  # guard tiny corpora

    vocab = Vocabulary.build(train_sents, cfg.max_vocab)
    train_ds = TextDataset(train_sents, vocab, cfg.max_len)
    val_ds = TextDataset(val_sents, vocab, cfg.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab
