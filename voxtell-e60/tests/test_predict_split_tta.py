"""GATE — the ported translation TTA must un-jitter onto the voxel it jittered away from.

A wrong sign or a wrong axis in the inverse does not crash and does not look wrong in a log:
it averages each pass against a NEIGHBOURING voxel, which blurs the prediction by `shift`
voxels in three directions and simply scores a bit worse. So the check is algebraic rather
than statistical -- feed a translation-EQUIVARIANT network, where a correct inverse must
reproduce the single-pass answer exactly, and show that dropping the inverse breaks it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import predict_split as PS  # noqa: E402

SHIFT = 6


class EquivariantPredictor:
    """Output at a voxel depends only on that voxel, so translating in translates out.

    Not a stand-in for VoxTell -- a stand-in for the ONE property the inverse jitter must
    respect. A real sliding window is only approximately equivariant, which is precisely why
    the averaging is worth anything and also why it cannot be the thing under test here.
    """

    def __init__(self, findings: int = 2):
        self.findings = findings

    def predict_sliding_window_return_logits(self, x, emb):
        return (x * 4.0 - 2.0).repeat(self.findings, 1, 1, 1)


def _interior(t):
    """Drop the `SHIFT` boundary slices on every spatial axis -- the replicated band where the
    jitter is deliberately not invertible."""
    return t[:, SHIFT:-SHIFT, SHIFT:-SHIFT, SHIFT:-SHIFT]


@pytest.fixture
def volume():
    torch.manual_seed(0)
    return torch.rand(1, 24, 22, 20)


def test_jitter_moves_content_and_composes_to_identity(volume):
    moved = PS.jitter(volume, 3, 1)
    assert torch.equal(moved[:, 5], volume[:, 2]), "jitter(+s) must move content toward +index"
    back = PS.jitter(moved, -3, 1)
    assert torch.equal(back[:, 3:-3], volume[:, 3:-3])
    assert not torch.equal(moved, volume), "a jitter that changes nothing tests nothing"


def test_tta_average_of_an_equivariant_net_equals_the_single_pass(volume):
    predictor = EquivariantPredictor()
    single = PS.predict_prob(predictor, volume, None, 0)
    averaged = PS.predict_prob(predictor, volume, None, SHIFT)

    assert single.shape == averaged.shape == (2, 24, 22, 20)
    assert torch.allclose(_interior(averaged), _interior(single), atol=1e-6), (
        "the inverse jitter is not landing on the voxel the forward jitter came from")


def test_the_gate_fires_when_the_inverse_jitter_is_dropped(volume):
    """--prove-fires. Without this the test above passes for any implementation that happens
    to return the same tensor seven times, including one that never jittered at all."""
    predictor = EquivariantPredictor()
    single = PS.predict_prob(predictor, volume, None, 0)

    def forward(x):
        return torch.sigmoid(
            predictor.predict_sliding_window_return_logits(x, None).to("cpu").float())

    no_inverse = forward(volume)
    for dim in (1, 2, 3):
        for s in (SHIFT, -SHIFT):
            no_inverse = no_inverse + forward(PS.jitter(volume, s, dim))
    no_inverse /= 7

    assert not torch.allclose(_interior(no_inverse), _interior(single), atol=1e-6)


def test_tta_off_is_exactly_one_forward(volume):
    calls = []

    class Counting(EquivariantPredictor):
        def predict_sliding_window_return_logits(self, x, emb):
            calls.append(x.shape)
            return super().predict_sliding_window_return_logits(x, emb)

    PS.predict_prob(Counting(), volume, None, 0)
    assert len(calls) == 1
    PS.predict_prob(Counting(), volume, None, SHIFT)
    assert len(calls) == 8  # 1 + 7
