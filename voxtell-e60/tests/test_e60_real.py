"""The gates that need the real assets, not a synthetic cache: the model VoxTell v1.0 builds
and the scorer that produces the number.

Skipped automatically when the assets are absent (`scripts/paths.py` says which and where),
so a checkout without them still runs `tests/test_gates.py` clean. They are NOT optional
where it matters -- these are the two things a synthetic fixture cannot cover:

  * the arm-E architecture is pinned to the parameter count and key set the published E60 /
    E60_Hu checkpoints were trained under. A change here does not crash: it produces a model
    a checkpoint fails to load into, or -- worse, if the shapes still line up -- one that
    loads and means something else.
  * the scorer of record is pinned by content hash. An upstream edit would keep every number
    flowing and change what they mean.

    python -m pytest tests/test_e60_real.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import paths                                                                  # noqa: E402

# Measured on the network this folder builds, and equal to `n_params` in the E60_Hu run.json.
ARM_E_PARAMS = 442_243_301
ARM_E_KEYS = 1111
STOCK_PARAMS = 440_029_541            # arm E adds 2,213,760 on top: the PWAM module


def _skip_without(var: str):
    if not paths.get(var).exists():
        pytest.skip(f"{var} -> {paths.get(var)} is absent; see scripts/paths.py")


@pytest.fixture(scope="module")
def T():
    _skip_without("REXG_VOXTELL_MODEL_DIR")
    pytest.importorskip("voxtell", reason="the VoxTell package is not installed")
    import train_voxtell
    return train_voxtell


def test_arm_e_architecture_is_the_one_e60_was_trained_under(T):
    """`--fuse trunk` must build exactly the network the published checkpoints fit.

    random_init=True skips the 5 GB v1.0 load -- it changes the WEIGHTS, never the shape --
    so this is the architecture assertion, on CPU, in about a minute."""
    net, n = T.build_network(random_init=True, deep_supervision=True, fuse_into_trunk=True)
    sd = net.state_dict()
    assert n == ARM_E_PARAMS, (
        f"arm E has {n:,} parameters, E60 was trained with {ARM_E_PARAMS:,}. A checkpoint "
        f"from this repo is no longer comparable to the published numbers.")
    assert len(sd) == ARM_E_KEYS, f"{len(sd)} state_dict entries, expected {ARM_E_KEYS}"


def test_fuse_off_is_the_stock_model_and_arm_e_only_adds_the_fusion_module(T):
    """The arm-E delta is the PWAM fusion module and nothing else -- that is what makes
    E-vs-B a one-flag comparison rather than two different models."""
    from voxtell_fusion import fusion_param_names
    stock, n_stock = T.build_network(random_init=True, deep_supervision=True,
                                     fuse_into_trunk=None)
    fused, n_fused = T.build_network(random_init=True, deep_supervision=True,
                                     fuse_into_trunk=True)
    assert n_fused - n_stock == 2_213_760, \
        f"the fusion module is {n_fused - n_stock:,} parameters, it was 2,213,760"
    assert n_stock == STOCK_PARAMS, f"stock model has {n_stock:,}, expected {STOCK_PARAMS:,}"
    added = set(fused.state_dict()) - set(stock.state_dict())
    assert added == fusion_param_names(fused), (
        "arm E adds keys that are not the fusion module's: "
        f"{sorted(added - fusion_param_names(fused))[:5]}")
    assert not set(stock.state_dict()) - set(fused.state_dict()), \
        "arm E DROPS a stock key; it is supposed to be a strict superset"


def test_the_scorer_of_record_is_present_and_unchanged():
    _skip_without("REXG_CHALLENGE_DIR")
    import scorer
    got = scorer.gate_scorer()
    assert set(got) == set(scorer.EVALUATOR_SHA)
    ev = scorer._load_evaluator(scorer.EVALUATOR)
    # The post-processing constants ARE part of the metric; postprocess() reads them off the
    # evaluator rather than restating them, and this is what proves they are still there.
    assert (ev.MIN_SIZE_DEFAULT, ev.CC_CONNECTIVITY_DEFAULT) == (10, 2)
    assert ev.MAX_COMPONENTS_PER_FINDING == 50


def test_the_native_config_points_at_data_that_exists():
    """Not a code gate -- a setup gate. It fails with the key that is wrong rather than
    letting `--build-text-bank-only` die on the first read."""
    from rexg_data.config import load_data_config
    cfg = load_data_config(REPO / "configs" / "rexg_native.yaml")
    missing = {k: getattr(cfg, k) for k in ("paired_root", "seg_root", "npy_root")
               if not Path(getattr(cfg, k)).exists()}
    if missing:
        pytest.skip(f"configs/rexg_native.yaml points at absent data: {missing} -- edit it "
                    f"for this machine (SETUP.md step 2)")
    for name in ("findings.csv", "volume_labels.csv"):
        assert (Path(cfg.paired_root) / name).is_file(), \
            f"{name} missing from paired_root; build_npy_cache.py cannot run without it"
    for name in ("index.csv", "stats.csv"):
        assert (Path(cfg.npy_root) / name).is_file(), \
            f"{name} missing from npy_root; run scripts/build_native_stats.py (SETUP.md step 3)"
