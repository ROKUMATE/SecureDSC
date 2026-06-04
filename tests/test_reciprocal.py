"""Reciprocal CSI: Alice/Bob correlated by ~rho, Eve independent."""

import torch

from securedsc.channels import ReciprocalChannel


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return (a * b).mean().item()


def test_alice_bob_correlated_eve_independent():
    torch.manual_seed(0)
    rho = 0.95
    recip = ReciprocalChannel(csi_dim=64, rho=rho, snr_db=25.0)
    csi = recip.sample(2048)

    ab = _corr(csi.alice, csi.bob)
    ae = _corr(csi.alice, csi.eve_a)

    # Alice/Bob strongly correlated (close to rho^2), Eve near zero.
    assert ab > 0.7, f"alice-bob corr too low: {ab}"
    assert abs(ae) < 0.15, f"alice-eve corr too high: {ae}"
    assert ab > abs(ae) + 0.5


def test_correlation_increases_with_rho():
    torch.manual_seed(0)
    corrs = []
    for rho in (0.3, 0.6, 0.9, 0.99):
        recip = ReciprocalChannel(csi_dim=64, rho=rho, snr_db=30.0)
        csi = recip.sample(4096)
        corrs.append(_corr(csi.alice, csi.bob))
    # monotone non-decreasing in rho
    for lo, hi in zip(corrs, corrs[1:]):
        assert hi >= lo - 0.05
