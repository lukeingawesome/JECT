"""The gates of the E60 path, on synthetic data.

No GPU, no real dataset, no 440M model: everything here runs on a synthetic 40^3 cache in
tmp_path in about a second. What it CANNOT cover is stated in scripts/train_voxtell.py's own
docstring -- the forward/backward, the memory, the sliding-window sweep. `test_e60_real.py`
covers the parts that need the real model and the real challenge data.

Ported from the upstream suite, minus every test that belonged to an arm this folder does not
ship (the auxiliary category head, the retrieval adapter, the dual-path E1 arm).

Run:  python -m pytest tests/test_gates.py -q
"""
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_voxtell.py"


def _mod():
    spec = spec_from_file_location("train_voxtell", SCRIPT)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _mod()

# --------------------------------------------------------------------- fake dataset
FINDING_COLUMNS = ["VolumeName", "finding_idx", "seg_channel", "category", "text", "pixels",
                   "entity_count", "volume_mm3", "rexg_split", "miccai_split", "fold",
                   "has_seg"]
STATS_COLUMNS = ["VolumeName", "seg_channel", "finding_idx", "nx", "ny", "nz", "hu_mean",
                 "hu_std", "fg_voxels", "bb_lo0", "bb_hi0", "bb_lo1", "bb_hi1", "bb_lo2",
                 "bb_hi2", "cen0", "cen1", "cen2"]

# < 192 on every axis (forces the pad path) and ANISOTROPIC on purpose: a cubic fixture
# cannot see a (2, 1, 0) axis swap, and that is exactly how one shipped past 40 green tests.
# Real volumes are 512 x 512 x nz with nz median 333; only 12 of 3,192 have nx == nz.
SHAPE = (40, 48, 56)


def _interior(cube, shape):
    """The un-padded region of a symmetric zero-pad up to 192^3, in `shape`'s axis order."""
    return cube[tuple(slice((192 - s) // 2, (192 - s) // 2 + s) for s in shape)]


def _finding(name, idx, channel, text, split="train", has_seg=1):
    return dict(zip(FINDING_COLUMNS, [name, idx, channel, "2b", text, 100, 1, 100.0,
                                      split, split, 0, has_seg]))


def _stats(name, channel, idx, centre=20.0):
    return dict(zip(STATS_COLUMNS, [name, channel, idx, *SHAPE, -500.0, 400.0, 100,
                                    0, 39, 0, 39, 0, 39, centre, centre, centre]))


@pytest.fixture
def cache(tmp_path):
    """A paired_root + npy cache with 2 train volumes, 1 val volume, and 1 maskless test CT.

    Mask values are 2 and 3 on purpose: the real cache's masks are entity-INSTANCE maps
    (measured unique values [0 1 2 3]) and that is the failure G-BINARY exists for.
    """
    paired, npy = tmp_path / "paired", tmp_path / "paired" / "npy"
    (npy / "ct").mkdir(parents=True)
    (npy / "seg").mkdir(parents=True)

    rows, stats = [], []
    for name, n_find, split in [("train_A.nii.gz", 2, "train"), ("train_B.nii.gz", 1, "train"),
                                ("valid_C.nii.gz", 1, "val")]:
        rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
        np.save(npy / "ct" / f"{name}.npy",
                rng.integers(-1000, 1000, SHAPE).astype(np.int16))
        mask = np.zeros((n_find, *SHAPE), np.uint8)
        for c in range(n_find):
            mask[c, 18:23, 18:23, 18:23] = c + 2          # instance labels 2, 3
        mask[0, 30:34, 5:9, 5:9] = 2       # asymmetric blob: an axis swap moves it
        np.savez_compressed(npy / "seg" / f"{name}.npz", mask=mask)
        for c in range(n_find):
            rows.append(_finding(name, c, c, f"finding {c} in the {name} lung", split))
            stats.append(_stats(name, c, c))
    # the maskless test CT: present in the cache, absent from any has_seg==1 row
    np.save(npy / "ct" / "test_D.nii.gz.npy", np.zeros(SHAPE, np.int16))
    rows.append(_finding("test_D.nii.gz", 0, 0, "unlabelled", "test", has_seg=0))

    pd.DataFrame(rows, columns=FINDING_COLUMNS).to_csv(paired / "findings.csv", index=False)
    pd.DataFrame(stats, columns=STATS_COLUMNS).to_csv(npy / "stats.csv", index=False)
    return paired, npy

# --------------------------------------------------------------------------- G-INDEX
def test_gindex_comes_from_findings_not_the_glob(cache):
    paired, npy = cache
    volumes = M.build_index(paired, npy, "train", drop_leak=False)
    assert [v["name"] for v in volumes] == ["train_A.nii.gz", "train_B.nii.gz"]
    assert sum(len(v["findings"]) for v in volumes) == 3
    # the cache holds 4 CTs; the index holds 2. That difference IS the gate.
    assert len(list((npy / "ct").iterdir())) == 4
    assert all(v["hu_mean"] == -500.0 and v["hu_std"] == 400.0 for v in volumes)
    assert volumes[0]["shape"] == SHAPE


def test_gindex_refuses_a_split_with_no_findings(cache):
    paired, npy = cache
    with pytest.raises(SystemExit, match="G-INDEX"):
        M.build_index(paired, npy, "nonexistent", drop_leak=False)


def test_gindex_refuses_a_missing_stats_row(cache):
    """A finding with no stats.csv row must raise, not quietly shrink the index."""
    paired, npy = cache
    stats = pd.read_csv(npy / "stats.csv")
    stats.iloc[:1].to_csv(npy / "stats.csv", index=False)
    with pytest.raises(SystemExit, match="stats.csv row"):
        M.build_index(paired, npy, "train", drop_leak=False)


def test_gindex_refuses_a_missing_stats_file(cache):
    paired, npy = cache
    (npy / "stats.csv").unlink()
    with pytest.raises(SystemExit, match="build_native_stats"):
        M.build_index(paired, npy, "train", drop_leak=False)

# -------------------------------------------------------------------------- G-BINARY
def test_gbinary_binarise_flattens_instance_labels():
    instances = np.array([[0, 2, 3], [5, 0, 4]], np.uint8)
    out = M.binarise(instances)
    assert set(np.unique(out).tolist()) == {0.0, 1.0}
    assert out.sum() == 4


def test_gbinary_fires_on_an_instance_valued_target():
    with pytest.raises(AssertionError, match="G-BINARY"):
        M.assert_binary(torch.tensor([[0.0, 1.0, 3.0]]), "unit")


def test_gbinary_passes_a_binary_target():
    M.assert_binary(torch.tensor([[0.0, 1.0, 1.0, 0.0]]), "unit")


def test_gbinary_holds_on_every_item_of_the_real_loader_path(cache):
    """End to end through __getitem__: the masks on disk carry 2 and 3."""
    paired, npy = cache
    volumes = M.build_index(paired, npy, "train", drop_leak=False)
    bank = {f["text"].lower(): np.ones(M.EMB_DIM, np.float32) * 3.0
            for v in volumes for f in v["findings"]}
    ds = M.VoxTellPatches(volumes, npy, bank, length=12, seed=0, fg_prob=0.5)
    for i in range(len(ds)):
        item = ds[i]
        M.assert_binary(item["gt"], f"item {i}")
        assert item["gt"].sum() > 0 or True          # emptiness is legal, non-binary is not


def test_gbinary_refuses_a_soft_mask():
    """`mask[c] > 0` would silently turn every p > 0 into foreground."""
    with pytest.raises(AssertionError, match="G-BINARY"):
        M.binarise(np.array([0.0, 0.3, 0.9], np.float32))


def test_gbinary_refuses_an_ignore_label():
    """nnU-Net's -1 ignore label: `> 0` folds it into background, silently."""
    with pytest.raises(AssertionError, match="G-BINARY"):
        M.binarise(np.array([-1, 0, 3], np.int16))


def test_the_raw_masks_really_are_non_binary(cache):
    """If this ever fails the fixture stopped exercising the failure it exists for."""
    _, npy = cache
    with np.load(npy / "seg" / "train_A.nii.gz.npz") as z:
        assert set(np.unique(z["mask"]).tolist()) - {0, 1}

# --------------------------------------------------------------------------- G-SHAPE
def test_gshape_pads_a_small_volume_to_192():
    assert M.pad_to_patch(np.ones((10, 104, 192), np.float32)).shape == M.PATCH
    assert M.pad_to_patch(np.ones((3, 40, 40, 40), np.float32)).shape == (3, *M.PATCH)


def test_gshape_pad_is_symmetric_and_zero():
    out = M.pad_to_patch(np.ones((190, 192, 192), np.float32))
    assert out[0, 0, 0] == 0 and out[-1, 0, 0] == 0
    assert out[1:-1].min() == 1.0


def test_gshape_refuses_an_oversized_patch():
    with pytest.raises(ValueError, match="larger than the patch"):
        M.pad_to_patch(np.ones((193, 192, 192), np.float32))


def test_gshape_every_item_is_exactly_192_cubed(cache):
    paired, npy = cache
    volumes = M.build_index(paired, npy, "train", drop_leak=False)
    bank = {f["text"].lower(): np.zeros(M.EMB_DIM, np.float32)
            for v in volumes for f in v["findings"]}
    ds = M.VoxTellPatches(volumes, npy, bank, length=8, seed=1, fg_prob=1.0)
    for i in range(len(ds)):
        item = ds[i]
        assert tuple(item["image"].shape) == (1, *M.PATCH)
        assert tuple(item["gt"].shape) == (M.QUERIES, *M.PATCH)
        assert tuple(item["text"].shape) == (M.QUERIES, M.EMB_DIM)

# ---------------------------------------------------------------------------- G-AXIS
def test_the_item_is_in_nnunet_axis_order_not_ras(cache):
    """The training patch must arrive in the SAME axis order val_sweep feeds the predictor.

    nnU-Net's NibabelIOWithReorient does `reoriented.get_fdata().transpose((2, 1, 0))`, so
    VoxTell was pretrained on (nz, ny, nx); the npy cache is RAS (nx, ny, nz) and
    `val_sweep` transposes. A cube patch hides the swap from every shape assert, so this
    test is the only thing standing between arm B and a checkpoint that is wrong for its
    own evaluator.
    """
    paired, npy = cache
    volumes = [v for v in M.build_index(paired, npy, "train", drop_leak=False)
               if v["name"] == "train_A.nii.gz"]
    bank = {f["text"].lower(): np.zeros(M.EMB_DIM, np.float32) for f in volumes[0]["findings"]}
    item = M.VoxTellPatches(volumes, npy, bank, length=1, seed=0, fg_prob=1.0)[0]

    ct = np.load(npy / "ct" / "train_A.nii.gz.npy")
    ras = M.normalise(ct, -500.0, 400.0)
    inner = _interior(item["image"][0].numpy(), SHAPE[::-1])
    assert inner.shape == SHAPE[::-1] != SHAPE, "the fixture stopped being anisotropic"
    assert np.allclose(inner, ras.transpose(2, 1, 0), atol=1e-5), \
        "the image is not in nnU-Net (nz, ny, nx) order"

    with np.load(npy / "seg" / "train_A.nii.gz.npz") as z:
        mask = z["mask"]
    for q, c in enumerate(item["channels"].tolist()):
        gt = _interior(item["gt"][q].numpy(), SHAPE[::-1])
        assert np.array_equal(gt, (mask[c] > 0).transpose(2, 1, 0).astype(np.float32)), \
            f"query {q}: the target is not in the image's axis order"


def test_crop_start_never_leaves_the_array():
    for n in (40, 191, 192, 300, 512):
        for centre in (-50.0, 0.0, 17.5, n / 2, n * 2.0):
            s = M.crop_start(n, centre, 192)
            assert 0 <= s and s + 192 <= max(n, 192)
            assert s == 0 if n <= 192 else s + 192 <= n


def test_a_single_finding_volume_still_yields_two_queries(cache):
    """35.3% of train volumes hold one finding. The declared behaviour is to repeat it."""
    paired, npy = cache
    volumes = [v for v in M.build_index(paired, npy, "train", drop_leak=False)
               if v["name"] == "train_B.nii.gz"]
    assert len(volumes[0]["findings"]) == 1
    bank = {volumes[0]["findings"][0]["text"].lower(): np.ones(M.EMB_DIM, np.float32)}
    item = M.VoxTellPatches(volumes, npy, bank, length=1, seed=3, fg_prob=1.0)[0]
    assert tuple(item["gt"].shape) == (M.QUERIES, *M.PATCH)
    assert item["channels"].tolist() == [0, 0]


def test_the_sample_order_is_identical_across_arms(cache):
    """Arm C must differ from arm B by the weight load ONLY, sample order included."""
    paired, npy = cache
    volumes = M.build_index(paired, npy, "train", drop_leak=False)
    bank = {f["text"].lower(): np.zeros(M.EMB_DIM, np.float32)
            for v in volumes for f in v["findings"]}
    a = M.VoxTellPatches(volumes, npy, bank, length=6, seed=42, fg_prob=0.66)
    b = M.VoxTellPatches(volumes, npy, bank, length=6, seed=42, fg_prob=0.66)
    for i in range(6):
        x, y = a[i], b[i]
        assert x["name"] == y["name"] and x["channels"].tolist() == y["channels"].tolist()
        assert torch.equal(x["image"], y["image"]) and torch.equal(x["gt"], y["gt"])

# ---------------------------------------------------------------- normalisation
def test_normalise_is_nnunet_zscore_on_raw_hu():
    hu = np.array([-1000, -500, 0, 500, 1000], np.int16)
    assert np.allclose(M.normalise(hu, -500.0, 400.0), (hu.astype(np.float32) + 500.0) / 400.0)


def test_normalise_uses_volume_statistics_not_patch_statistics():
    """The measured failure this prevents: patch z-score skews the mean by ~0.5 sigma."""
    rng = np.random.default_rng(0)
    volume = rng.normal(-500, 400, 4096).astype(np.int16)
    patch = volume[:64]
    by_volume = M.normalise(patch, float(volume.mean()), float(volume.std()))
    by_patch = (patch - patch.mean()) / patch.std()
    assert abs(float(by_volume.mean()) - float(by_patch.mean())) > 1e-3
    assert np.allclose(by_volume.mean(), (patch.mean() - volume.mean()) / volume.std(), atol=1e-4)


def test_normalise_survives_a_constant_volume():
    out = M.normalise(np.zeros(8, np.int16), 0.0, 0.0)
    assert np.isfinite(out).all()


def test_the_loader_normalises_with_the_stats_csv_numbers(cache):
    paired, npy = cache
    volumes = [v for v in M.build_index(paired, npy, "train", drop_leak=False)
               if v["name"] == "train_A.nii.gz"]
    bank = {f["text"].lower(): np.zeros(M.EMB_DIM, np.float32) for f in volumes[0]["findings"]}
    item = M.VoxTellPatches(volumes, npy, bank, length=1, seed=0, fg_prob=1.0)[0]
    ct = np.load(npy / "ct" / "train_A.nii.gz.npy")
    want = M.normalise(ct, -500.0, 400.0).transpose(2, 1, 0)   # stats.csv hu_mean / hu_std
    assert np.allclose(_interior(item["image"][0].numpy(), SHAPE[::-1]), want, atol=1e-5)

# ---------------------------------------------------------------------------- G-TEXT
def _bank(norm: float, n: int = 8, dim: int = None):
    dim = dim or M.EMB_DIM
    rng = np.random.default_rng(0)
    out = {}
    for i in range(n):
        v = rng.normal(size=dim).astype(np.float32)
        out[f"sentence {i}"] = v / np.linalg.norm(v) * norm
    return out


def _wrapper():
    try:
        from voxtell.utils.text_embedding import wrap_with_instruction
    except ImportError:                       # pragma: no cover - env without the package
        pytest.skip("the voxtell package is not importable in this interpreter")
    return wrap_with_instruction


def test_gtext_accepts_a_raw_qwen3_bank():
    wrap = _wrapper()
    bank = _bank(137.5)
    key = next(iter(bank))
    out = M.gate_text(bank, wrap([key])[0], key, list(bank))
    assert 130 < out["norm_median"] < 145


def test_gtext_rejects_an_l2_normalised_bank():
    """This is the BioLORD bank's shape: a 138x scale error into Linear(2560, 2048)."""
    wrap = _wrapper()
    bank = _bank(1.0)
    key = next(iter(bank))
    with pytest.raises(AssertionError, match="median"):
        M.gate_text(bank, wrap([key])[0], key, list(bank))


def test_gtext_rejects_a_bank_built_without_the_instruction_wrapper():
    wrap = _wrapper()
    bank = _bank(137.5)
    key = next(iter(bank))
    with pytest.raises(AssertionError, match="Instruct: "):
        M.gate_text(bank, key, key, list(bank))          # the bare sentence, unwrapped
    with pytest.raises(AssertionError, match="wrap_with_instruction"):
        M.gate_text(bank, "Instruct: something else\nQuery: x", key, list(bank))


def test_gtext_rejects_a_missing_sentence():
    wrap = _wrapper()
    bank = _bank(137.5)
    key = next(iter(bank))
    with pytest.raises(AssertionError, match="not in the bank"):
        M.gate_text(bank, wrap([key])[0], key, [*bank, "a sentence nobody encoded"])




def _tok_pair(enc, n=4):
    """A per-token bank and the pooled bank that MUST pair with it, consistent by
    construction for this encoder's pooling. Every G-BANK-CONSISTENT test below starts from
    a consistent pair and breaks exactly one thing, so a failure names its own cause."""
    rng = np.random.default_rng(0)
    labels = [f"sentence {i}" for i in range(n)]
    lengths = np.array([3, 5, 2, 7][:n], dtype=np.int32)
    arr = np.zeros((n, enc.bank_len, enc.emb_dim), np.float32)
    for i, L in enumerate(lengths):
        arr[i, :L] = rng.normal(size=(L, enc.emb_dim)).astype(np.float32)
    if enc.pooling == "last_token":
        pooled = {labels[i]: arr[i, lengths[i] - 1].copy() for i in range(n)}
    else:
        pooled = {labels[i]: arr[i, :lengths[i]].mean(0) for i in range(n)}
    meta = {"labels": np.array(labels, dtype=np.str_), "lengths": lengths,
            "prefix_tokens": np.array([enc.bank_prefix]),
            "wrapped_example": np.array(M.encoder_prompts(enc, [labels[0]]), dtype=np.str_)}
    return arr, {s: i for i, s in enumerate(labels)}, lengths, meta, pooled, labels
GATOR = None   # resolved lazily so a missing ENCODERS entry fails as a test, not at import


def _gator():
    global GATOR
    if GATOR is None:
        GATOR = M.ENCODERS["gatortron"]
    return GATOR


def test_gbank_consistent_accepts_a_mean_pooled_pair():
    enc = _gator()
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    out = M.gate_token_bank(arr, index, lengths, meta, pooled, labels, enc=enc)
    assert out["encoder"] == "gatortron" and out["prefix_tokens"] == 0
    assert out["consistency_max_delta"] < M.MEAN_TOL


def test_gbank_consistent_fires_when_the_mean_pooled_banks_disagree():
    """The defect: two banks encoded with different wrapper/padding/tokeniser. Simulated by
    moving ONE sentence's pooled vector, which is what such a drift looks like per row."""
    enc = _gator()
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    pooled[labels[2]] = pooled[labels[2]] + 0.5      # ~100x MEAN_TOL, ~ real drift's scale
    with pytest.raises(AssertionError, match="G-BANK-CONSISTENT"):
        M.gate_token_bank(arr, index, lengths, meta, pooled, labels, enc=enc)


def test_gbank_consistent_mean_would_break_if_cls_were_dropped():
    """Why gatortron's bank_prefix is 0: the mean must run over EXACTLY the stored positions.
    Dropping the first stored position from the token bank -- the shape [CLS] omission would
    take -- makes the pooled vector underivable and the gate must say so."""
    enc = _gator()
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    shifted = np.zeros_like(arr)
    for i, L in enumerate(lengths):                  # drop position 0, shift the rest left
        shifted[i, :L - 1] = arr[i, 1:L]
    with pytest.raises(AssertionError, match="G-BANK-CONSISTENT"):
        M.gate_token_bank(shifted, index, lengths - 1, meta, pooled, labels, enc=enc)


def test_gbank_consistent_fires_on_a_prefix_that_moved():
    enc = _gator()
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    meta = {**meta, "prefix_tokens": np.array([1])}
    with pytest.raises(AssertionError, match="prefix"):
        M.gate_token_bank(arr, index, lengths, meta, pooled, labels, enc=enc)


def test_gbank_consistent_fires_on_a_bank_of_the_wrong_width():
    enc = _gator()
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    with pytest.raises(AssertionError, match="token bank is"):
        M.gate_token_bank(arr[:, :, :512], index, lengths, meta, pooled, labels, enc=enc)


def test_gbank_consistent_accepts_the_qwen3_last_token_pair():
    """The legacy relation still holds exactly: pooled IS the stored last position."""
    _wrapper()                                        # skips if voxtell is not importable
    enc = M.ENCODERS["qwen3"]
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    out = M.gate_token_bank(arr, index, lengths, meta, pooled, labels, enc=enc)
    assert out["consistency_max_delta"] == 0.0        # bit-identical, not merely close


def test_gbank_consistent_qwen3_fires_on_one_changed_row():
    _wrapper()
    enc = M.ENCODERS["qwen3"]
    arr, index, lengths, meta, pooled, labels = _tok_pair(enc)
    pooled[labels[1]] = pooled[labels[1]] + 1e-3      # equality, so even a small delta fires
    with pytest.raises(AssertionError, match="G-BANK-CONSISTENT"):
        M.gate_token_bank(arr, index, lengths, meta, pooled, labels, enc=enc)


def test_encoder_pool_mean_matches_a_hand_computed_mean():
    """encoder_pool is what build_text_bank stores and what gate_token_bank's relation is
    stated against; if it silently included padding the gate would still pass."""
    enc = _gator()
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    got = M.encoder_pool(enc, hidden, mask)
    assert torch.allclose(got[0], hidden[0, :2].mean(0))
    assert torch.allclose(got[1], hidden[1, :3].mean(0))


def test_encoder_prompts_leaves_a_non_wrapping_encoder_alone():
    assert M.encoder_prompts(_gator(), ["mild atelectasis"]) == ["mild atelectasis"]


def _stub_bank(tmp_path, name, **extra):
    p = tmp_path / name
    np.savez(p, labels=np.array(["mild atelectasis"], dtype=np.str_),
             embeddings=np.zeros((1, 4), np.float32), **extra)
    return p


def test_encoder_for_reads_the_key_the_bank_stamped_on_itself(tmp_path):
    p = _stub_bank(tmp_path, "text_emb_gatortron.npz",
                   encoder_key=np.array(["gatortron"], dtype=np.str_))
    enc = M.encoder_for(p)
    assert (enc.key, enc.pooling, enc.wrap, enc.padding_side) == \
        ("gatortron", "mean", False, "right")


def test_encoder_for_treats_a_bank_with_no_key_as_qwen3(tmp_path):
    """Every bank built before the field existed is a qwen3 bank; there is no other option
    it could silently be mistaken for."""
    assert M.encoder_for(_stub_bank(tmp_path, "text_emb_qwen3.npz")).key == "qwen3"


def test_encoder_for_refuses_an_encoder_it_does_not_know(tmp_path):
    """A bank from a future arm must not fall back to qwen3's wrapper and pooling."""
    p = _stub_bank(tmp_path, "text_emb_something.npz",
                   encoder_key=np.array(["biolord"], dtype=np.str_))
    with pytest.raises(SystemExit, match="does not know about"):
        M.encoder_for(p)


def test_encoder_for_refuses_a_bank_that_is_not_there(tmp_path):
    with pytest.raises(SystemExit, match="no text bank"):
        M.encoder_for(tmp_path / "nope.npz")


def test_every_encoder_matches_the_pretrained_projection_width():
    """project_text_embed.0.weight is (2048, 2560). An entry with a different emb_dim would
    silently break build_network's strict=True load hours into a run."""
    assert {e.emb_dim for e in M.ENCODERS.values()} == {M.EMB_DIM}, \
        {k: e.emb_dim for k, e in M.ENCODERS.items()}


def test_last_token_pooling_implies_left_padding():
    """last_token_pool reads position -1, which is the last REAL token only under left
    padding. The two fields cannot be edited independently."""
    for k, e in M.ENCODERS.items():
        if e.pooling == "last_token":
            assert e.padding_side == "left", k
        if e.padding_side == "right":
            assert e.pooling != "last_token", k


def test_gbank_covers_accepts_a_bank_that_holds_every_prompt():
    """Case differs on purpose: the predictor lowercases before the lookup, the bank is
    keyed on s.lower(), so a capitalised findings.csv string must still count as covered."""
    bank = {"a small nodule in the right lower lobe", "mild atelectasis"}
    assert M.gate_bank_covers(bank, ["A small nodule in the RIGHT lower lobe",
                                     "mild atelectasis"]) == 2


def test_gbank_covers_rejects_one_missing_prompt():
    """The defect it exists for: ONE uncovered sentence is enough to route the sweep into
    the predictor's silent Qwen3 re-encode, which no metric would show."""
    bank = {"mild atelectasis"}
    with pytest.raises(SystemExit, match="not in the bank"):
        M.gate_bank_covers(bank, ["mild atelectasis", "a sentence nobody encoded"])


def test_gbank_covers_rejects_an_empty_prompt_set():
    """A sweep with nothing to score must not read as a pass."""
    with pytest.raises(SystemExit, match="no prompts"):
        M.gate_bank_covers({"mild atelectasis"}, [])


def test_gtext_rejects_a_single_outlier_vector():
    """Measured spread over real sentences is 135.7-137.9; one vector at 300 is degenerate."""
    wrap = _wrapper()
    bank = _bank(137.5)
    key = next(iter(bank))
    bank["a degenerate sentence"] = bank[key] / 137.5 * 300.0
    with pytest.raises(AssertionError, match="outside"):
        M.gate_text(bank, wrap([key])[0], key, list(bank))


def test_gtext_rejects_the_wrong_width():
    wrap = _wrapper()
    bank = _bank(137.5, dim=768)                    # BioLORD's width
    key = next(iter(bank))
    with pytest.raises(AssertionError, match="bank width"):
        M.gate_text(bank, wrap([key])[0], key, list(bank))


def test_gtext_lookup_is_lowercased_like_the_predictor():
    """VoxTellPredictor.embed_text_prompts does `p.lower()`; the bank keys must match."""
    wrap = _wrapper()
    bank = _bank(137.5)
    key = next(iter(bank))
    M.gate_text(bank, wrap([key])[0], key, [k.upper() for k in bank])


def test_the_bank_file_round_trips_through_voxtells_own_loader(tmp_path):
    """The bank we write must still be readable as a VoxTellPredictor embedding_bank."""
    wrap = _wrapper()
    keys = ["a finding", "another finding"]
    emb = np.stack([np.full(M.EMB_DIM, 137.5 / np.sqrt(M.EMB_DIM), np.float32)] * 2)
    path = tmp_path / "bank.npz"
    np.savez(path, labels=np.array(keys, dtype=np.str_), embeddings=emb,
             wrapped_example=np.array([wrap([keys[0]])[0]], dtype=np.str_))
    bank, wrapped, key = M.load_text_bank(path)
    assert set(bank) == set(keys) and wrapped.startswith("Instruct: ") and key == keys[0]
    with np.load(path, allow_pickle=False) as z:
        assert {"labels", "embeddings"} <= set(z.files)   # what load_embedding_bank reads


def test_load_text_bank_refuses_the_biolord_container(tmp_path):
    """A BioLORD bank writes a {keys, emb} container of 768-d L2-normalised vectors."""
    path = tmp_path / "text_emb.npz"
    np.savez(path, keys=np.array(["deadbeef"], dtype=np.str_),
             emb=np.zeros((1, 768), np.float32))
    with pytest.raises(SystemExit, match="BioLORD"):
        M.load_text_bank(path)


def test_load_text_bank_refuses_a_bank_with_no_wrapper_evidence(tmp_path):
    path = tmp_path / "bare.npz"
    np.savez(path, labels=np.array(["x"], dtype=np.str_),
             embeddings=np.zeros((1, M.EMB_DIM), np.float32))
    with pytest.raises(SystemExit, match="wrapped_example"):
        M.load_text_bank(path)

# ------------------------------------------------------------------- G-LOSS-POSITIVE
def test_loss_is_strictly_positive_on_a_binary_target():
    logits = torch.randn(2, 2, 8, 8, 8)
    for target in (torch.zeros(2, 2, 8, 8, 8), torch.ones(2, 2, 8, 8, 8),
                   (torch.rand(2, 2, 8, 8, 8) > 0.99).float()):
        assert float(M.dice_bce(logits, target)) > 0


def test_loss_goes_negative_on_an_instance_valued_target():
    """The exact failure G-BINARY prevents: BCE against target=3 is unbounded below."""
    logits = torch.full((1, 1, 8, 8, 8), 6.0)
    assert float(M.dice_bce(logits, torch.full((1, 1, 8, 8, 8), 3.0))) < 0


def test_loss_is_reduced_per_sample_and_query_not_over_the_batch():
    """The lesson the 0.1538 bar came from: a 200-voxel finding must weigh the same as a
    200,000-voxel one."""
    logits = torch.randn(3, 2, 8, 8, 8)
    target = (torch.rand(3, 2, 8, 8, 8) > 0.9).float()
    per = torch.stack([M.dice_bce(logits[i:i + 1, j:j + 1], target[i:i + 1, j:j + 1])
                       for i in range(3) for j in range(2)])
    assert torch.allclose(M.dice_bce(logits, target), per.mean(), atol=1e-6)


def test_a_perfect_prediction_scores_near_zero_dice():
    target = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    logits = (target * 2 - 1) * 30.0
    assert float(M.dice_bce(logits, target)) < 1e-4

# ---------------------------------------------------------------- deep supervision
def test_ds_targets_keep_a_one_voxel_finding_at_every_scale():
    gt = torch.zeros(1, 2, 16, 16, 16)
    gt[0, 1, 5, 9, 3] = 1.0
    heads = M.ds_targets(gt, [(16,) * 3, (8,) * 3, (4,) * 3, (2,) * 3, (1,) * 3])
    assert len(heads) == 5
    assert [float(h.sum()) for h in heads] == [1.0] * 5


def test_ds_targets_refuse_a_non_integer_division():
    with pytest.raises(ValueError, match="deep-supervision head"):
        M.ds_targets(torch.zeros(1, 1, 16, 16, 16), [(16,) * 3, (7, 8, 8)])


def test_ds_weights_sum_to_one():
    w = M.ds_weights(5)
    assert abs(sum(w) - 1) < 1e-9 and w[0] > w[-1]

# ---------------------------------------------------------------------- schedule
def test_poly_lr_starts_at_base_and_ends_at_zero():
    assert M.poly_lr(1e-4, 0, 10_000) == 1e-4
    assert M.poly_lr(1e-4, 10_000, 10_000) == 0.0
    assert M.poly_lr(1e-4, 5_000, 10_000) == pytest.approx(1e-4 * 0.5 ** 0.9)


def test_the_bar_is_the_official_number_not_the_broken_one():
    assert M.BAR == 0.1538, "0.0987 is the in-training 2 mm Dice from the broken-loss run"


def test_self_test_runs():
    assert M.self_test() == 0

# ------------------------------------------------------- G-RESUMABLE: checkpoints resume
def _tiny():
    """A 3-parameter stand-in. The gate is about the CONTRACT of the saved blob, which does
    not depend on the model being 440M, and a real one would make this test cost minutes."""
    net = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.99, nesterov=True)
    return net, opt


def _step(net, opt):
    opt.zero_grad()
    net(torch.randn(2, 4)).pow(2).mean().backward()
    opt.step()


class _Args:
    def __init__(self): self.steps = 10


def test_a_saved_checkpoint_carries_the_optimizer(tmp_path):
    net, opt = _tiny()
    _step(net, opt)
    p, state = M.save_checkpoint(tmp_path / "c.pth", net, opt, 1, "test", _Args())
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert set(M.CKPT_REQUIRED) <= set(blob)
    assert blob["optimizer_state"]["state"], "momentum buffers were not saved"
    assert set(state) == set(blob["network_weights"])   # returned weights match the file


def test_s_per_step_measures_steps_done_not_the_absolute_counter():
    """A resume restarts the clock but not the step counter, so `elapsed / step` divides a
    partial elapsed time by a full counter and under-reports the rate.

    Observed on the 2026-08-13 resume-from-10000: the log printed 0.12 s/step while the wall
    clock said 0.600, and the ETA it drove read 34 min against a real 173. Display only, but
    it is the number a person plans around. The expression under test is the one in the
    training loop's log record."""
    rate = lambda elapsed, step, start: elapsed / max(1, step - start)

    # the real case: 2,550 steps done in 1,530 s after resuming at 10,000
    assert abs(rate(1530.0, 12550, 10000) - 0.600) < 1e-9
    # the bug it replaces would have said 0.122 -- a 4.92x under-report
    assert abs(1530.0 / 12550 - 0.122) < 1e-3
    # a fresh run is unaffected, so older runs' logs still read the same
    assert rate(1530.0, 12550, 0) == 1530.0 / 12550
    # and the first logged step after a resume must not divide by zero
    assert rate(5.0, 10000, 10000) == 5.0


def test_resume_restores_momentum_exactly(tmp_path):
    torch.manual_seed(0)
    net, opt = _tiny()
    _step(net, opt); _step(net, opt)
    want = [v["momentum_buffer"].clone() for v in opt.state.values()]
    p, _ = M.save_checkpoint(tmp_path / "c.pth", net, opt, 2, "test", _Args())

    net2, opt2 = _tiny()
    assert not opt2.state, "a fresh optimizer must start with no momentum"
    step = M.load_resume(p, net2, opt2, torch.device("cpu"))
    got = [v["momentum_buffer"] for v in opt2.state.values()]
    assert step == 2
    assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want)), \
        "momentum did not survive the round trip -- a 'resume' would restart at zero velocity"


def test_the_gate_fires_on_a_weights_only_checkpoint(tmp_path):
    """The exact shape every checkpoint written before this gate has. A gate never shown the
    case it was written for is an untested branch that prints ok."""
    net, opt = _tiny()
    _step(net, opt)
    old = {"network_weights": net.state_dict(), "step": 5, "arm": "test", "args": {}}
    torch.save(old, tmp_path / "old.pth")

    with pytest.raises(AssertionError, match="missing"):
        M.assert_resumable(old, "old-format blob")
    with pytest.raises(AssertionError, match="weights-only"):
        M.load_resume(tmp_path / "old.pth", *_tiny(), torch.device("cpu"))


def test_the_gate_fires_on_an_optimizer_that_never_stepped(tmp_path):
    """`opt.state_dict()` before the first step has param_groups and an EMPTY state, so a
    presence-only check would pass it. Past step 0 that means the optimizer never trained
    this network."""
    net, opt = _tiny()                                   # deliberately not stepped
    with pytest.raises(AssertionError, match="optimizer state is empty"):
        M.save_checkpoint(tmp_path / "c.pth", net, opt, 7, "test", _Args())
    M.save_checkpoint(tmp_path / "zero.pth", net, opt, 0, "test", _Args())   # step 0 is legal


def test_resume_continues_the_same_sample_order(monkeypatch):
    """The sampler is indexed, not stateful, so continuation index 0 must BE original index
    `start_index` -- otherwise a resume silently re-shuffles the data it already saw."""
    full = M.VoxTellPatches([], Path("."), {}, length=100, seed=42, fg_prob=0.66)
    cont = M.VoxTellPatches([], Path("."), {}, length=100, seed=42, fg_prob=0.66,
                            start_index=40)
    assert len(full) == 100 and len(cont) == 60, "resume must not re-run consumed steps"

    seen = {}

    def spy(s):
        seen["seed"] = s
        raise RuntimeError("stop before touching disk")

    monkeypatch.setattr(M.np.random, "default_rng", spy)
    for ds, idx in ((full, 40), (cont, 0)):
        with pytest.raises(RuntimeError):
            ds[idx]
        assert seen["seed"] == (42 * 1_000_003 + 40) % (2 ** 63)


def test_only_the_newest_checkpoint_keeps_the_optimizer(tmp_path):
    """The disk contract: one resumable checkpoint, the rest inference-only."""
    net, opt = _tiny()
    _step(net, opt)
    p1, _ = M.save_checkpoint(tmp_path / "checkpoint_step1.pth", net, opt, 1, "t", _Args())
    assert torch.load(p1, map_location="cpu", weights_only=False)["optimizer_state"]["state"]

    before = {k: v.clone() for k, v in torch.load(
        p1, map_location="cpu", weights_only=False)["network_weights"].items()}
    _step(net, opt)
    p2, _ = M.save_checkpoint(tmp_path / "checkpoint_step2.pth", net, opt, 2, "t", _Args())

    old = torch.load(p1, map_location="cpu", weights_only=False)
    new = torch.load(p2, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in old and "rng" not in old, "the superseded one kept its state"
    assert old["optimizer_stripped"] == "checkpoint_step2.pth"
    assert new["optimizer_state"]["state"], "the newest one must stay resumable"
    # inference must still work off the stripped file, and its weights must be UNCHANGED
    assert set(old["network_weights"]) == set(before)
    assert all(torch.equal(old["network_weights"][k], v) for k, v in before.items()), \
        "stripping altered the weights -- the file no longer serves inference faithfully"
    assert old["step"] == 1 and old["arm"] == "t"


def test_resume_on_a_stripped_checkpoint_says_why_and_points_at_the_newest(tmp_path):
    net, opt = _tiny()
    _step(net, opt)
    M.save_checkpoint(tmp_path / "checkpoint_step1.pth", net, opt, 1, "t", _Args())
    _step(net, opt)
    M.save_checkpoint(tmp_path / "checkpoint_step2.pth", net, opt, 2, "t", _Args())
    with pytest.raises(SystemExit, match="checkpoint_step2.pth"):
        M.load_resume(tmp_path / "checkpoint_step1.pth", *_tiny(), torch.device("cpu"))


def test_a_new_checkpoint_never_leaves_zero_resumable_ones(tmp_path):
    """Stripping happens AFTER the new file lands. If it ran first, a crash in between would
    leave a run with no resumable checkpoint at all."""
    net, opt = _tiny()
    _step(net, opt)
    M.save_checkpoint(tmp_path / "checkpoint_step1.pth", net, opt, 1, "t", _Args())
    _step(net, opt)
    real = M._atomic_save
    seen = {}

    def spy(blob, path):
        # at the moment the new checkpoint is written, the old one must still be resumable
        if path.name.startswith("checkpoint_step2"):
            old = torch.load(tmp_path / "checkpoint_step1.pth", map_location="cpu",
                             weights_only=False)
            seen["old_still_resumable"] = "optimizer_state" in old
        real(blob, path)

    M._atomic_save = spy
    try:
        M.save_checkpoint(tmp_path / "checkpoint_step2.pth", net, opt, 2, "t", _Args())
    finally:
        M._atomic_save = real
    assert seen["old_still_resumable"] is True
    assert not any(tmp_path.glob("*.tmp")), "an atomic-save temp file was left behind"

# ------------------------------------------------- probe_class_readout: CLASSES has one home



# --------------------------------------------------- clsaux: class_target must reach the loader



# ----------------------------------------------- --init-from: weights-only checkpoint init
def test_init_from_and_resume_together_exits(monkeypatch):
    """--init-from (weights only, fresh optimizer/schedule) and --resume (full continuation)
    mean opposite things. Combining them must not silently pick one -- it must refuse, because
    silently restarting the schedule while claiming continuity is exactly the failure mode
    --resume's own weights-only gate exists to prevent one level up."""
    monkeypatch.setattr("sys.argv",
                        ["train_voxtell.py", "--init-from", "a.pth", "--resume", "b.pth"])
    with pytest.raises(SystemExit, match="mean different things") as exc:
        M.main()
    assert "silently restart the schedule while claiming continuity" in str(exc.value)





# (filename, enclosing function) pairs where a bare-identifier strict=True load of a
# network_weights-derived variable is SAFE BY CONSTRUCTION, not an unfixed instance of the bug --
# each entry says why, so this list documents an exception rather than hiding one:
NETWORK_WEIGHTS_EXEMPT = frozenset({
    # --init-from only ever receives a stock or arm-E checkpoint, and this folder ships no
    # wrapped arm at all -- gate_unwrapped is what makes a wrapped one impossible to load
    # anywhere else, so routing this call site through it too would be a tautology.
    ("train_voxtell.py", "load_init_from"),
})

_STRICT_LOAD_RE = re.compile(
    r"\.load_state_dict\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*strict\s*=\s*True\s*\)")


def _enclosing_def_name(lines: list[str], idx: int) -> str | None:
    for j in range(idx, -1, -1):
        s = lines[j].strip()
        if s.startswith("def "):
            return s[len("def "):].split("(")[0]
    return None


def network_weights_offenders(scan_dir: Path, exempt=NETWORK_WEIGHTS_EXEMPT) -> list[str]:
    """Every `X.load_state_dict(var, strict=True)` in `scan_dir`/*.py whose `var` traces (via
    the NEAREST preceding assignment to `var` in the same file) straight back to a
    `["network_weights"]` subscript, with no `gate_unwrapped(` call in between. One string per
    hit, file:line plus the offending source line, so a failing assert is self-explanatory.

    This is a SCAN, not a runtime check, because the defect is a call site that does not exist
    yet: a fourth loader added later would load a wrapped checkpoint into a stock network,
    predict plausible masks, and leave every runtime gate green. It was fixed three times
    inline in three separate call sites upstream before it got one home."""
    offenders = []
    for src in sorted(scan_dir.glob("*.py")):
        lines = src.read_text().splitlines()
        for i, line in enumerate(lines):
            m = _STRICT_LOAD_RE.search(line)
            if not m:
                continue
            var = m.group("var")
            routed = from_network_weights = False
            for j in range(i - 1, -1, -1):
                am = re.match(rf'^\s*{re.escape(var)}\s*=\s*(.*)$', lines[j])
                if not am:
                    continue
                rhs = am.group(1)
                if "gate_unwrapped(" in rhs:
                    routed = True
                elif '["network_weights"]' in rhs:
                    from_network_weights = True
                break                                    # nearest prior assignment only
            if from_network_weights and not routed:
                if (src.name, _enclosing_def_name(lines, i)) in exempt:
                    continue
                offenders.append(f"{src.name}:{i + 1}: `{line.strip()}` (var {var!r} traces to "
                                 "network_weights without going through gate_unwrapped)")
    return offenders


def test_no_loader_bypasses_gate_unwrapped():
    """The live scan over this folder's own scripts/. The exemption list is the only way a
    call site is allowed out, and every entry in it carries its reason."""
    offenders = network_weights_offenders(REPO / "scripts")
    assert not offenders, "\n".join(offenders)


def test_network_weights_gate_fires_on_a_planted_offender(tmp_path):
    """Proves the scan actually catches the real bug's shape, not just its own fixed code:
    plants a file OUTSIDE scripts/ that reproduces predict_split.py's original defect
    byte-for-byte in spirit (extract network_weights, load strict=True, no guard) and checks the
    scan reports it."""
    offender = tmp_path / "new_predict_variant.py"
    offender.write_text(
        'sd = torch.load(ckpt, map_location="cpu", weights_only=False)["network_weights"]\n'
        'def main():\n'
        '    predictor.network.load_state_dict(sd, strict=True)\n')
    found = network_weights_offenders(tmp_path, exempt=frozenset())
    assert found, "the planted offender was not caught -- the scan is not firing at all"
    assert any("new_predict_variant.py:3" in f and "sd" in f for f in found), found
    print(f"  planted-offender assertion (verbatim): {found}")



# ---- G-DEAD-CONFIG -------------------------------------------------------------------
# The failure this guards is not hypothetical: B60 and E60 each ran 60,000 steps naming a
# config whose preprocess block says 2 mm isotropic into a 192^3
# cube. train_voxtell.py reads paired_root and npy_root from it and nothing else. So the
# case below is the REAL config from the real runs, not a synthetic one.
NATIVE = REPO / "configs" / "rexg_native.yaml"
sys.path.insert(0, str(REPO / "src"))
from rexg_data.config import load_data_config          # noqa: E402


def test_dead_config_gate_fires_on_the_config_b60_and_e60_actually_ran(tmp_path):
    """The gate's own failure case, written out as a real YAML file: the 2 mm config B60 and
    E60 were actually launched under. `train_voxtell.py` reads NONE of those five keys -- it
    cuts 192^3 in VOXELS off the native grid -- so 60,000-step runs recorded an isotropic
    resample that never happened. A gate never shown its own failure case is an
    untested branch that prints ok."""
    bad = tmp_path / "rexg_res_2mm.yaml"
    bad.write_text(NATIVE.read_text()
                   .replace("target_spacing: null", "target_spacing: [2.0, 2.0, 2.0]")
                   .replace("cube_size: null", "cube_size: [192, 192, 192]")
                   .replace("cache2mm_root: null", "cache2mm_root: /some/cache_2mm"))
    with pytest.raises(SystemExit) as e:
        M.gate_dead_config(load_data_config(bad), str(bad))
    msg = str(e.value)
    # names the keys, so the reader does not have to diff two files to see which ones lied
    for key in ("target_spacing", "cube_size", "cache2mm_root"):
        assert key in msg, f"gate message does not name {key}: {msg}"
    assert "rexg_native.yaml" in msg, "gate must say what to use instead"


def test_dead_config_gate_fires_on_the_resize_path_too(tmp_path):
    """`spatial_size` is the old 96^3 Resized path. Also never read here."""
    bad = tmp_path / "resize.yaml"
    bad.write_text(NATIVE.read_text().replace("spatial_size: null", "spatial_size: [96, 96, 96]"))
    with pytest.raises(SystemExit):
        M.gate_dead_config(load_data_config(bad), str(bad))


def test_dead_config_keys_are_the_ones_train_voxtell_never_references():
    """The list is hand-written, so this asserts it against the source it describes.

    A key that the script DOES read must not be in DEAD_CONFIG_KEYS -- that would kill a
    config for declaring something real. Checked by grepping cfg.<key> in the script.
    """
    src = SCRIPT.read_text()
    for key in M.DEAD_CONFIG_KEYS:
        assert f"cfg.{key}" not in src, (
            f"{key} is in DEAD_CONFIG_KEYS but scripts/train_voxtell.py references cfg.{key}; "
            f"the gate would reject a config for declaring something the script honours")


def test_every_shipped_config_passes_the_gate():
    """If someone adds a second config here, this says so rather than letting two homes for
    the same decision drift (which is how the token-bank name broke upstream, 543875f)."""
    shipped = sorted((REPO / "configs").glob("*.yaml"))
    assert [p.name for p in shipped] == ["rexg_native.yaml"], (
        f"this folder ships one config on purpose; found {[p.name for p in shipped]}. "
        f"A second one is a second home for the preprocessing decision.")
    for path in shipped:
        M.gate_dead_config(load_data_config(path), str(path))
