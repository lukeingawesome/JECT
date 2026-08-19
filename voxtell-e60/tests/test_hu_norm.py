"""GATE — intensity normalisation is one decision, and inference cannot disagree with training.

Two failures this exists for, both of which leave every other gate green:

  * a checkpoint trained on the fixed affine is predicted under per-volume z-score (or the
    reverse). The masks come out, the scorer scores them, and the number is wrong by however
    much the input distribution moved. `gate_hu_norm` must REFUSE.
  * `--hu-norm ct` is accepted but never reaches the arithmetic, so the run reports a scheme
    it did not use. The normaliser must actually change the values, and by the amount the
    measurement predicted -- not merely "differ".

The threshold in the last test is not guessed. It is the spread the scheme was adopted to
remove, measured on the real cache: under per-volume z-score the same tissue lands anywhere
in a ~0.3 window across volumes; under the fixed affine it lands on one value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

torch = pytest.importorskip("torch")
import train_voxtell as T  # noqa: E402


def test_zscore_is_the_default_and_ct_is_the_measured_constants():
    assert T.hu_norm_spec(None) is None
    assert T.hu_norm_spec("zscore") is None
    spec = T.hu_norm_spec("ct")
    assert spec["clip"] == (-1000.0, 1000.0)
    # Pinned to the full train-split pass (2,992 volumes / 292.7G voxels). If someone
    # re-measures on a different split or a sample, this fails and the change is deliberate.
    assert spec["mean"] == pytest.approx(-540.9803, abs=1e-4)
    assert spec["std"] == pytest.approx(482.0892, abs=1e-4)


def test_an_unknown_scheme_is_refused_rather_than_defaulted():
    with pytest.raises(SystemExit, match="not a scheme"):
        T.hu_norm_spec("ctnorm")


def test_fixed_affine_matches_nnunet_ct_arithmetic_and_clips():
    spec = T.hu_norm_spec("ct")
    hu = np.array([-3000.0, -1000.0, -540.9803, 0.0, 1000.0, 32767.0], np.float32)
    got = T.normalise_fixed(hu, spec)
    want = (np.clip(hu, -1000.0, 1000.0) - spec["mean"]) / spec["std"]
    assert np.allclose(got, want, atol=1e-6)
    assert got[0] == got[1], "below the floor must clip, not extrapolate"
    assert got[-1] == got[-2], "the 32767 int16 saturation must clip, not blow up the scale"
    assert got[2] == pytest.approx(0.0, abs=1e-6)


def test_predictor_normalisation_swap_uses_the_same_expression():
    """`FixedAffineNormalization` is what the sweep and predict_split hand to VoxTell, so it
    must be the training arithmetic itself -- not a second implementation that agrees today."""
    spec = T.hu_norm_spec("ct")
    rng = np.random.default_rng(0)
    image = rng.uniform(-1200, 1200, size=(1, 8, 8, 8)).astype(np.float32)
    assert np.array_equal(T.FixedAffineNormalization(spec).run(image, None),
                          T.normalise_fixed(image, spec))


def test_gate_refuses_the_mismatch_it_was_written_for():
    trained_ct = {"args": {"hu_norm": "ct"}}
    trained_z = {"args": {"hu_norm": "zscore"}}
    legacy = {"network_weights": {}}          # B60/E60: written before the flag existed

    assert T.gate_hu_norm(trained_ct, "ct", "t") == T.hu_norm_spec("ct")
    assert T.gate_hu_norm(trained_z, "zscore", "t") is None
    assert T.gate_hu_norm(legacy, "zscore", "t") is None, "no record means z-score, its truth"

    for blob, asked in ((trained_ct, "zscore"), (trained_z, "ct"), (legacy, "ct")):
        with pytest.raises(SystemExit, match="different input distribution"):
            T.gate_hu_norm(blob, asked, "t")


def test_the_scheme_removes_the_per_volume_spread_it_was_adopted_for():
    """Synthetic volumes that differ only in artefact contamination and field of view --
    the two things that move a per-volume z-score without changing the tissue."""
    spec = T.hu_norm_spec("ct")
    rng = np.random.default_rng(7)
    tissue = 30.0
    fixed, per_volume = [], []
    for air_fraction, n_metal in ((0.55, 0), (0.55, 40), (0.75, 0), (0.35, 5)):
        n = 200_000
        vol = np.where(rng.random(n) < air_fraction, -1000.0, 20.0).astype(np.float32)
        vol[:n_metal] = 3071.0                       # a few voxels of metal, as measured
        per_volume.append((tissue - vol.mean()) / max(vol.std(), 1e-8))
        fixed.append(float(T.normalise_fixed(np.array([tissue], np.float32), spec)[0]))

    assert max(fixed) - min(fixed) == 0.0, "a fixed affine cannot depend on the volume"
    assert max(per_volume) - min(per_volume) > 0.3, (
        "the per-volume spread this scheme removes must actually be present in the fixture, "
        "or the test above proves nothing")
