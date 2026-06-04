"""Key generation: random sharing, CSI agreement vs Eve, reconciliation."""

import torch

from securedsc.channels import ReciprocalChannel
from securedsc.keygen import (
    CSIKeyGenerator,
    RandomKeyGenerator,
    majority_reconcile,
)


def test_random_key_shared_and_eve_differs():
    torch.manual_seed(0)
    kg = RandomKeyGenerator(key_dim=64)
    b = kg.generate(128, torch.device("cpu"))
    # Alice and Bob identical, Eve roughly half mismatched.
    assert torch.equal(b.alice, b.bob)
    assert b.ab_agreement == 1.0
    assert 0.3 < b.eve_mismatch < 0.7


def test_majority_reconcile_reduces_errors():
    torch.manual_seed(0)
    key_dim, repeat = 32, 5
    truth = torch.sign(torch.randn(64, key_dim))
    # expand with repetition then flip 20% of raw bits
    raw = truth.repeat_interleave(repeat, dim=1).clone()
    flip = torch.rand_like(raw) < 0.2
    raw[flip] *= -1
    voted = majority_reconcile(raw, repeat)
    raw_err = (torch.sign(raw.view(64, key_dim, repeat)[..., 0]) != truth).float().mean()
    voted_err = (voted != truth).float().mean()
    assert voted_err < raw_err  # repetition coding helps


def test_csi_key_alice_bob_agree_more_than_eve():
    torch.manual_seed(0)
    kg = CSIKeyGenerator(csi_dim=32, key_dim=64, hidden=64, repeat=3)
    recip = ReciprocalChannel(csi_dim=32, rho=0.98, snr_db=25.0)
    csi = recip.sample(256)
    b = kg.generate(256, torch.device("cpu"), csi=csi)
    # Even before training, correlated CSI -> higher A/B agreement than Eve's.
    assert b.ab_agreement >= 1.0 - b.eve_mismatch  # agreement >= eve agreement
    # cipher path keys are identical after reconciliation
    assert torch.equal(b.alice, b.bob)


def test_csi_agreement_loss_is_scalar():
    kg = CSIKeyGenerator(csi_dim=32, key_dim=64, hidden=64, repeat=3)
    recip = ReciprocalChannel(csi_dim=32, rho=0.9, snr_db=20.0)
    csi = recip.sample(64)
    b = kg.generate(64, torch.device("cpu"), csi=csi)
    loss = kg.agreement_loss(b)
    assert loss.dim() == 0 and loss.item() >= 0.0
    loss.backward()  # differentiable wrt generator params
