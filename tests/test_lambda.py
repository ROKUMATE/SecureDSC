"""Lambda schedulers: fixed constant, adaptive feedback toward a target."""

import math

from securedsc.config import LambdaConfig
from securedsc.lambda_sched import (
    AdaptiveLambda,
    FixedLambda,
    make_lambda_scheduler,
)


def test_fixed_lambda_constant():
    s = FixedLambda(0.7)
    for _ in range(5):
        assert s.step(l_b=0.1, l_e=2.0) == 0.7
    assert s.get() == 0.7


def test_adaptive_increases_when_eve_too_good():
    # target high; Eve loss persistently low -> lambda should climb.
    s = AdaptiveLambda(initial=0.5, target=3.0, eta=0.1, ema=0.5, max_value=10.0)
    start = s.get()
    for _ in range(50):
        s.step(l_b=0.2, l_e=0.5)  # Eve doing well (low loss)
    assert s.get() > start


def test_adaptive_decreases_when_eve_already_bad():
    s = AdaptiveLambda(initial=5.0, target=3.0, eta=0.1, ema=0.5, min_value=0.0)
    start = s.get()
    for _ in range(50):
        s.step(l_b=0.2, l_e=6.0)  # Eve loss above target
    assert s.get() < start


def test_adaptive_clipped_to_bounds():
    s = AdaptiveLambda(initial=0.0, target=100.0, eta=1.0, ema=0.0,
                       min_value=0.0, max_value=2.0)
    for _ in range(20):
        s.step(l_b=0.1, l_e=0.0)
    assert 0.0 <= s.get() <= 2.0


def test_factory_adaptive_target_from_vocab():
    cfg = LambdaConfig(kind="adaptive", value=0.5, target_frac=0.9, eta=0.05)
    s = make_lambda_scheduler(cfg, vocab_size=1000)
    assert isinstance(s, AdaptiveLambda)
    assert math.isclose(s.target, 0.9 * math.log(1000), rel_tol=1e-6)
