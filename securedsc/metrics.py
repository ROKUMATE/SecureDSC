"""Evaluation metrics: self-contained BLEU and key agreement.

BLEU is implemented directly (rather than via NLTK) so it can be unit-tested and
has no hidden dependencies. It uses the standard modified n-gram precision with a
brevity penalty and add-one smoothing on higher-order n-grams (so short sentences
do not collapse to zero).
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import torch


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(
    reference: Sequence[str],
    candidate: Sequence[str],
    max_n: int = 4,
    smooth: bool = True,
    epsilon: float = 0.1,
) -> float:
    """Compute sentence-level BLEU between a reference and a candidate.

    Returns a score in ``[0, 1]``. The effective n-gram order is capped at the
    candidate length (you cannot have 4-grams in a 3-word sentence), with uniform
    weights over the active orders. ``smooth`` uses NLTK-style "method 1": zero
    modified precisions are replaced by ``epsilon/total`` so a single missing
    higher-order n-gram does not collapse the score to zero, while genuinely
    disjoint sentences still score very low.
    """
    cand_len = len(candidate)
    if cand_len == 0:
        return 0.0

    eff_n = max(1, min(max_n, cand_len))
    weight = 1.0 / eff_n
    log_prec_sum = 0.0
    for n in range(1, eff_n + 1):
        cand_ng = _ngram_counts(candidate, n)
        ref_ng = _ngram_counts(reference, n)
        overlap = sum(min(c, ref_ng[g]) for g, c in cand_ng.items())
        total = max(1, cand_len - n + 1)
        if overlap == 0:
            if not smooth:
                return 0.0
            precision = epsilon / total
        else:
            precision = overlap / total
        log_prec_sum += weight * _safe_log(precision)

    # Brevity penalty.
    ref_len = len(reference)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = _safe_exp(1.0 - ref_len / max(1, cand_len))
    return bp * _safe_exp(log_prec_sum)


def corpus_bleu(
    references: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """Mean sentence-BLEU over a corpus (simple, robust aggregate)."""
    if not candidates:
        return 0.0
    scores = [
        sentence_bleu(r, c, max_n=max_n)
        for r, c in zip(references, candidates)
    ]
    return float(sum(scores) / len(scores))


def _safe_log(x: float) -> float:
    import math

    return math.log(x) if x > 0 else float("-inf")


def _safe_exp(x: float) -> float:
    import math

    return math.exp(x) if x > float("-inf") else 0.0


def key_agreement_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of matching signs between two ``{-1,+1}`` bit tensors."""
    return (torch.sign(a) == torch.sign(b)).float().mean().item()
