"""BLEU metric: identical, disjoint, partial overlap, brevity penalty."""

from securedsc.metrics import corpus_bleu, sentence_bleu


def test_identical_is_one():
    s = "the quick brown fox jumps".split()
    assert abs(sentence_bleu(s, s) - 1.0) < 1e-6


def test_disjoint_is_low():
    ref = "the quick brown fox".split()
    cand = "zebra plays violin underwater".split()
    assert sentence_bleu(ref, cand, smooth=True) < 0.2


def test_partial_overlap_between():
    ref = "the cat sat on the mat".split()
    cand = "the cat sat on the rug".split()
    score = sentence_bleu(ref, cand)
    assert 0.3 < score < 1.0


def test_empty_candidate_is_zero():
    assert sentence_bleu("a b c".split(), []) == 0.0


def test_brevity_penalty_shortens_score():
    ref = "the quick brown fox jumps over".split()
    full = sentence_bleu(ref, ref)
    short = sentence_bleu(ref, "the quick".split())
    assert short < full


def test_corpus_bleu_average():
    refs = [["a", "b", "c"], ["d", "e", "f"]]
    cands = [["a", "b", "c"], ["d", "e", "f"]]
    assert abs(corpus_bleu(refs, cands) - 1.0) < 1e-6
