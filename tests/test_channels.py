"""Channel math: noise level vs SNR, power normalisation, fading shape."""

import math

import torch

from securedsc.channels import (
    AWGNChannel,
    RayleighChannel,
    normalize_power,
    make_channel,
)
from securedsc.utils import snr_db_to_noise_std


def test_snr_to_noise_std():
    # 0 dB -> noise std 1 for unit power; 20 dB -> std 0.1.
    assert math.isclose(snr_db_to_noise_std(0.0), 1.0, rel_tol=1e-6)
    assert math.isclose(snr_db_to_noise_std(20.0), 0.1, rel_tol=1e-6)


def test_awgn_empirical_noise_power():
    torch.manual_seed(0)
    ch = AWGNChannel()
    x = torch.zeros(256, 30, 16)  # zero signal: y == noise
    snr_db = 10.0
    y = ch.transmit(x, snr_db)
    expected_var = 1.0 / (10 ** (snr_db / 10))
    assert abs(y.var().item() - expected_var) < 0.02


def test_normalize_power_unit():
    torch.manual_seed(0)
    x = torch.randn(8, 30, 16) * 5.0 + 2.0
    xn = normalize_power(x)
    p = xn.pow(2).mean(dim=(-2, -1))
    assert torch.allclose(p, torch.ones_like(p), atol=1e-4)


def test_rayleigh_shape_and_factory():
    ch = make_channel("rayleigh")
    assert isinstance(ch, RayleighChannel)
    x = normalize_power(torch.randn(4, 10, 16))
    y = ch.transmit(x, 12.0)
    assert y.shape == x.shape
    # received differs from transmitted (fading + noise)
    assert not torch.allclose(y, x)
