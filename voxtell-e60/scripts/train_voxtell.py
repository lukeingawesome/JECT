#!/usr/bin/env python
"""VoxTell v1.0 full fine-tune for ReXGroundingCT referring segmentation -- arm E ("E60").

    # NOT this repo's venv: VoxTell needs nnunetv2 / acvl_utils / batchgenerators
    VT=/path/to/envs/voxtell/bin/python

    $VT scripts/train_voxtell.py --self-test                      # gates only, no GPU, no data
    $VT scripts/train_voxtell.py --build-text-bank-only           # once, ~8 min on a free GPU
    $VT scripts/train_voxtell.py --build-token-bank-only          # arm E needs the token bank too
    $VT scripts/train_voxtell.py --fuse trunk --steps 60000 \
        --val-at 30000,60000 --out-dir runs/E60                   # E60
    $VT scripts/train_voxtell.py --fuse trunk --hu-norm ct ...    # E60_Hu

WHAT E60 IS. VoxTell v1.0 (440M, text-prompted, pretrained on anatomy) fine-tuned for 60,000
steps with the fused text feature routed BACK INTO the encoder trunk (`--fuse trunk`, arm E)
rather than only into the decoder skips (`--fuse skip`, arm S). `--fuse off` is the stock
model, arms B (pretrained) and C (`--random-init` control). The three differ by that flag and
nothing else -- same seed, same sample order, same schedule, same gates.

Measured, at tile_step 0.25 / threshold 0.5 over the 200-case / 381-finding val split:

    E60      mean_global_dice_per_finding  0.34334
    E60_Hu   (`--hu-norm ct`)              0.34076      -- indistinguishable, see below
    bar      the FiLM-UNet this replaces   0.1538

E60_Hu is a SEPARATE 60,000-step run from the same v1.0 checkpoint, not a continuation, and
differs by `--hu-norm ct` alone: per-volume z-score becomes clip [-1000, 1000] plus a global
fixed affine (mean -540.9803, std 482.0892) measured over all 2,992 MICCAI train volumes.
The -0.00258 gap is inside the pre-registered +-0.008 detection floor, and the sign flips
between operating points (thr 0.6: -0.00231; the in-training sweep at tile 0.5: +0.00196), so
no direction can be claimed.

Nor is there a confirmed side effect on the gradient, which is what motivated the change: in
the two 60-step SMOKE runs the step-25 grad_norm was 54.457 (z-score) against 16.241 (ct)
with the clip at 12.0, but in the 60,000-step runs that produced these checkpoints it is
20.603 against 20.387 -- the same number. The smoke pair is 60 steps of one seed and does not
generalise to the run; do not quote it as if it were the run.

ARM C IS THE POINT for the pretrained claim. `--random-init` changes exactly one line -- the
`load_state_dict` is skipped -- and nothing else. Reporting arm B/E without arm C is
reporting an uninterpretable number.

RECIPE (verified by integration review; re-verification is not this script's job):
  * `strict=True` load of `fold_0/checkpoint_final.pth` gives missing 0 / unexpected 0 at
    both deep_supervision=False and True; 440.03M unique parameters (442.24M under --fuse
    trunk, which adds the PWAM fusion module).
  * `pos_embed` is a persistent (1728, 1, 2048) buffer -- 12^3 tokens at stride 16. The
    patch size is HARD-LOCKED at 192^3. It is not a tunable here, hence G-SHAPE.
  * deep supervision returns 5 outputs.
  * `ct.transpose(2, 1, 0)` is nnU-Net's `NibabelIOWithReorient` axis order. The npy cache
    is RAS (nx, ny, nz); VoxTell wants (nz, ny, nx). This applies to TRAINING PATCHES TOO
    (G-AXIS in `VoxTellPatches.__getitem__`), not only to the sweep -- the patch is a cube,
    so nothing about its shape reveals a training/inference axis swap.
  * nnU-Net `ZScoreNormalization` on RAW HU with VOLUME statistics -- read here from
    `<npy_root>/stats.csv`, never recomputed per patch. Patch-level z-score is a different
    number: measured median skew 0.49 sigma/voxel. `--hu-norm ct` replaces this scheme
    entirely; the checkpoint records which one it used and inference refuses to disagree.
  * 2 queries per patch, batch 2. Measured on 1xH200: 2x1 prompt = 22.9 GiB / 0.33 s/step,
    1x4 prompts = 21.4 GiB / 0.38 s/step.
  * SGD momentum 0.99 nesterov, lr 1e-4, poly (1 - t)^0.9, grad-norm clip 12.0, fg_prob 0.66.
  * the val sweep runs only at `--val-at` -- each sweep is ~17k sliding-window tiles over 200
    volumes, ~60-85 min of the training GPU.

THE GATES, all literal asserts. `python -O` deletes asserts, so `main()` refuses to start
under it -- that was the one flag that could skip all of them.

  G-BINARY  `binarise` checks the RAW mask channel is a non-negative INTEGER instance map
            before `> 0`; the loop re-checks the batched target. 49 of the 70 native seg
            channels are entity-INSTANCE maps carrying values 2..5 (measured: this cache's
            masks are uint8 with unique values [0 1 2 3]). Raw BCE against target=3 is
            UNBOUNDED BELOW -- measured loss -1.045 and still falling -- and the run
            completes normally with a garbage checkpoint.
            MEASURED HONESTY: the loop-level `assert_binary` CANNOT fire while `binarise`
            is in the path -- a {0,2,3} mask driven through `__getitem__` reaches it as
            {0,1}, by construction. It is defence against a future edit, not a live check.
            The check that can fire today is the one on `binarise`'s INPUT.
  G-AXIS    every item: the patch is transposed (2, 1, 0) into nnU-Net's (nz, ny, nx) order,
            the same order `val_sweep` feeds `VoxTellPredictor`. A cube patch hides an axis
            swap from every shape assert, so this one is enforced by a test, not a runtime
            check: tests/test_voxtell_gates.py builds an ANISOTROPIC volume and compares the
            item's interior against both orientations.
  G-SHAPE   every step: the patch is exactly 192^3 after padding. 104 of the 3,192 indexed
            volumes have nz < 192 (min 104), and pos_embed cannot absorb a different size.
  G-TEXT    preflight: 130 < median ||e||_2 < 145 over the bank, no vector outside
            (50, 500), and the bank's stored wrapped prompt starts with 'Instruct: ' and
            equals `voxtell.utils.text_embedding.wrap_with_instruction` of its key.
            VoxTell consumes RAW last-token-pooled Qwen3 vectors (measured median 137.5).
            An L2-NORMALISED bank fed to a pretrained `Linear(2560, 2048)` is a 138x scale
            error; cos(with wrapper, without) = 0.7366 median. Do not reuse one here.
  G-INDEX   preflight: the sample index comes from `findings.csv[miccai_split, has_seg]`,
            never from globbing `npy/ct/*.npy` -- 100 cached test CTs have no mask at all.
  G-LOSS-POSITIVE  every step: loss > 0. Soft Dice is in [0, 1] and BCE on a binarised
            target is > 0, so a non-positive loss means G-BINARY was bypassed.
  G-LOAD-KEYS  arm E adds parameters, so the v1.0 load relaxes to strict=False. This asserts
            the ONLY unsupplied keys are exactly the fusion module's.
  G-HU-NORM `gate_hu_norm` SystemExits when inference requests a scheme the checkpoint was
            not trained under. Masks come out either way and the scorer scores them either
            way, so without this gate no metric would ever show the mismatch.

G-TEXT's (130, 145) band is asserted BOTH on the median and per vector. The per-vector form
was checked before committing to it: 8 real val sentences encoded through this file's own
`build_text_bank` on CPU give ||e||_2 min 135.7 / median 136.9 / max 137.9, a 2.2-wide
spread inside a 15-wide band. The spread over the full 8,650-sentence corpus is NOT
measured, so if the per-vector assert ever fires, look at the named sentence first (empty,
truncated, or non-English) before widening the band -- it costs nothing, it is preflight.

RESUME. `--resume` continues optimizer, step and sampler position; `--init-from` loads
weights only and restarts the schedule at step 0. They are mutually exclusive, because
combining them would silently restart the schedule while claiming continuity.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import paths                                                                  # noqa: E402

# Two patients appear on BOTH sides of the MICCAI train/val boundary under different volume
# names. Training on them makes the val number partly a memory test.
LEAK_PATIENTS = ("train_1841", "train_2936")


def patient_id(volume_name):
    """`train_1841_b_1.nii.gz` -> `train_1841`. Works for the `valid_*` prefix too."""
    if isinstance(volume_name, str):
        return "_".join(volume_name.split("_")[:2])
    return volume_name.str.split("_").str[:2].str.join("_")
sys.path.insert(0, str(REPO / "scripts"))

MODEL_DIR = str(paths.get("REXG_VOXTELL_MODEL_DIR"))
QWEN = str(paths.get("REXG_QWEN_DIR"))
PATCH = (192, 192, 192)          # hard-locked by the pos_embed buffer; see G-SHAPE
EMB_DIM = 2560                   # Qwen3-Embedding-4B width == VoxTell's text_embedding_dim
QUERIES = 2
BAR = 0.1538                     # official mean_global_dice_per_finding to beat: the
                                 # FiLM-UNet baseline this replaced, [0.1340, 0.1732]
# One authoritative name for the official metric: val_sweep's dice extraction (below) and
# the run.json sweep records both read this constant, so the recorded score's
# score_key can never drift from the key val_sweep actually pulled the number out of.
OFFICIAL_SCORE_KEY = "mean_global_dice_per_finding"
TEXT_BANK_NAME = "text_emb_qwen3.npz"    # the --text-bank default; ENCODERS reuses it below
# v13's per-token bank deliberately has NO name constant: derive it from whichever pooled bank
# is in play with token_bank_path_for(). A constant is what let the two banks drift apart.
#
# TOKEN_BANK_PREFIX / TOKEN_BANK_LEN used to live here as module constants. They are now
# per-encoder fields (ENCODERS[...].bank_prefix / .bank_len) and the pad length is read from
# the loaded array's own shape, so a constant can no longer disagree with the bank it
# describes. Do not reintroduce them: a second home for a per-encoder value is exactly how
# the pooled and per-token banks drifted apart in 543875f.

# G-TEXT, two bands for two different questions.
#
# NORM_LO/HI gates the MEDIAN, i.e. the scale of the bank as a whole. This is the real gate:
# VoxTell consumes raw last-token-pooled Qwen3 vectors, and feeding it an L2-normalised bank
# (median 1.0) is a 138x scale error into a pretrained Linear(2560, 2048) that produces
# plausible garbage rather than a crash. Measured on the full 6,467-sentence corpus: 137.55.
NORM_LO, NORM_HI = 130.0, 145.0

# Each INDIVIDUAL vector is instead required to sit within this factor of the bank's own
# median. The tight NORM_LO/HI band was originally applied per-vector too, calibrated on a
# 380-sentence sample (spread 135.7-137.9). The full 6,467-sentence corpus spreads
# 128.11-139.93, so it rejected 6 sentences (0.09%) that are ordinary complete clinical English
# ("paraseptal emphysema at the apical levels of both lungs", 128.83).
#
# Relative rather than absolute because the quantity that matters is distance from the corpus
# scale, not from a number someone measured once: a few percent of variation says nothing,
# while a degenerate vector is off by an order of magnitude. At the measured median this admits
# 68.8-275.1, so it still rejects an L2-normalised bank (1.0), a zero vector, and a 300.0
# outlier -- and it re-derives itself if the encoder or corpus ever changes.
VEC_NORM_FACTOR = 2.0


# ------------------------------------------------------------------- text encoder config

class TextEncoder(NamedTuple):
    """Everything that has to move together when the text encoder changes.

    NamedTuple and not @dataclass on purpose: this module is loaded via
    spec_from_file_location by tests/test_voxtell_gates.py and by scripts/probes/*, and a
    dataclass raises AttributeError under that loader because it looks its own module up in
    sys.modules. NamedTuple is immutable, stdlib, and has no such dependency.

    ONE home, because the alternative already failed here: TOKEN_BANK_NAME was a separate
    constant, drifted from the pooled bank it was supposed to pair with, and 2675 of 6467
    sentences silently mismatched (543875f). Six independently-edited constants is that
    defect six times over -- an arm that changes the model path but not the pooling produces
    a bank that is wrong in a way no shape check can see.

    `emb_dim` is NOT a free choice: VoxTell's pretrained project_text_embed.0.weight is
    (2048, 2560), so an encoder of a different width forces that tensor to be dropped and
    re-initialised and breaks build_network's strict=True. Both encoders here are 2560, so
    the network is untouched and only the banks' CONTENTS differ.
    """
    key: str
    hf_path: str
    emb_dim: int
    pooling: str          # "last_token" (causal, needs left padding) | "mean"
    padding_side: str
    wrap: bool            # the Instruct:/Query: wrapper -- Qwen3-Embedding's own protocol
    bank_prefix: int      # leading positions dropped from the per-token bank
    bank_len: int
    norm_lo: float
    norm_hi: float
    bank_name: str


ENCODERS = {
    "qwen3": TextEncoder(
        key="qwen3", hf_path=QWEN, emb_dim=2560,
        pooling="last_token", padding_side="left", wrap=True,
        # 23, not 24: the space after "Query:" merges into the following word for some
        # sentences, so only the first 23 ids are shared by all of them.
        bank_prefix=23,
        bank_len=64,                    # measured max query length 61
        norm_lo=NORM_LO, norm_hi=NORM_HI,   # measured median 137.55 over 6467 sentences
        bank_name=TEXT_BANK_NAME),
    "gatortron": TextEncoder(
        key="gatortron", hf_path="UFNLP/gatortron-medium", emb_dim=2560,
        # mean, NOT cls: CLS pooling scores 0.9306 on GATE 1 and fails the 0.95 floor,
        # while mean scores 0.9942. mean also has the tighter norm spread (8.87-12.89 vs
        # 9.09-15.73) and the better pooled swap margin (0.974 vs 0.980). See
        # (measured in a text-encoder swap probe, section 4).
        pooling="mean", padding_side="right", wrap=False,
        # 0: the whole attention-masked span is stored, [CLS] included, because the
        # pooled-vs-token equality below is mean over EXACTLY the stored positions. Dropping
        # [CLS] here would make that equality unstatable, and G-BANK-CONSISTENT would be
        # gone -- the gate that caught the 2675-sentence mismatch.
        bank_prefix=0,
        bank_len=64,                    # measured max 52 slots; same as qwen3, 0 truncation
        # A band this wide is a WEAKER gate than qwen3's +-5.5%: GatorTron's norms spread
        # 8.87-12.89, so the same relative band would reject 1478 of 10656 real sentences.
        # It still catches an L2-normalised bank (1.0) and gross scale errors; it will NOT
        # catch subtle drift. Recorded as a known weakness of this arm, not hidden.
        norm_lo=7.0, norm_hi=16.0,
        bank_name="text_emb_gatortron.npz"),
}


def encoder_for(text_bank_path) -> TextEncoder:
    """Which encoder built this bank -- read from the bank ITSELF, never passed in beside it.

    A separate --encoder flag could disagree with --text-bank, and that disagreement is
    exactly the class of bug this repo already paid for. So build_text_bank stamps
    `encoder_key` into the npz and this reads it back: the bank and its conventions cannot
    be separated because they are the same file.

    A bank with no `encoder_key` array predates this field, and qwen3 was the only encoder
    that existed then -- so absence means qwen3, unambiguously. gate_text's independent
    `Instruct: ` wrapper check confirms it rather than taking that on faith.
    """
    path = Path(text_bank_path)
    if not path.is_file():
        raise SystemExit(f"no text bank at {path}")
    with np.load(path, allow_pickle=True) as z:
        key = str(z["encoder_key"][0]) if "encoder_key" in z.files else "qwen3"
    if key not in ENCODERS:
        raise SystemExit(f"{path.name} was built by encoder {key!r}, which this code does "
                         f"not know about (have: {sorted(ENCODERS)})")
    return ENCODERS[key]


def encoder_for_new_bank(out_path) -> TextEncoder:
    """Which encoder to BUILD a bank with -- the one place where the file cannot be asked,
    because it does not exist yet.

    Resolved from the output filename against each encoder's own `bank_name`, so the name on
    disk and the conventions inside it are chosen by the same string. `--bank-splits`
    qualifiers ride along: text_emb_gatortron_all.npz still resolves to gatortron.

    No default. A name that matches nothing raises rather than falling back to qwen3 -- a
    silent fallback here writes a bank whose contents contradict its own filename, and
    `encoder_for` would then read the stamped key and happily believe it.
    """
    stem = Path(out_path).name
    hits = [e for e in ENCODERS.values() if stem.startswith(Path(e.bank_name).stem)]
    if len(hits) != 1:
        raise SystemExit(
            f"cannot tell which encoder should build {stem!r}: matched {[e.key for e in hits]}. "
            "Name the output after one encoder's bank_name "
            f"({', '.join(sorted(e.bank_name for e in ENCODERS.values()))}), optionally with a "
            "suffix before .npz.")
    return hits[0]


# ----------------------------------------------------------------------------- G-INDEX

def build_index(paired_root: Path, npy_root: Path, split: str,
                drop_leak: bool) -> list[dict]:
    """Volumes for `split`, from findings.csv ONLY (G-INDEX), joined to stats.csv.

    Returns one dict per VOLUME (the sampling unit is the volume because a patch carries
    QUERIES findings of the same CT), each with the volume's HU statistics and one entry
    per finding carrying that finding's centroid.

    Globbing `npy/ct/*.npy` gives 3,292 CTs; 100 of them are `miccai_split == "test"` and
    have no mask file at all. A loader driven by the glob trains on whatever `np.load` of a
    missing .npz raises -- or worse, on a silently skipped sample.
    """
    findings = pd.read_csv(paired_root / "findings.csv", dtype={"VolumeName": str})
    rows = findings[(findings["miccai_split"] == split) & (findings["has_seg"] == 1)]
    if rows.empty:
        raise SystemExit(f"G-INDEX: no findings with miccai_split == {split!r} and has_seg == 1")

    if drop_leak:
        before = len(rows)
        rows = rows[~rows["VolumeName"].map(patient_id).isin(LEAK_PATIENTS)]
        print(f"G-INDEX: dropped {before - len(rows)} findings on leak patients "
              f"{LEAK_PATIENTS} (same patient on both sides of the MICCAI boundary)")

    stats_path = npy_root / "stats.csv"
    if not stats_path.is_file():
        raise SystemExit(f"G-INDEX: {stats_path} missing -- run scripts/build_native_stats.py. "
                         "Recomputing volume HU statistics per patch is 3.2x data-bound and "
                         "patch-level statistics are a DIFFERENT normalisation.")
    # findings.csv is the authority for finding_idx/text; stats.csv contributes only the
    # measured geometry. Merging the whole frame would give finding_idx_x / finding_idx_y.
    stats = pd.read_csv(stats_path, dtype={"VolumeName": str})[
        ["VolumeName", "seg_channel", "nx", "ny", "nz", "hu_mean", "hu_std",
         "cen0", "cen1", "cen2"]]
    merged = rows.merge(stats, on=["VolumeName", "seg_channel"], how="left",
                        validate="one_to_one")
    missing = merged[merged["hu_mean"].isna()]
    if len(missing):
        raise SystemExit(
            f"G-INDEX: {len(missing)} of {len(merged)} {split} findings have no stats.csv row "
            f"(e.g. {missing['VolumeName'].head(3).tolist()}). Rebuild stats.csv over this "
            "split rather than training on a silently smaller index.")

    index: dict[str, dict] = {}
    for r in merged.to_dict("records"):
        vol = index.setdefault(r["VolumeName"], {
            "name": r["VolumeName"],
            "shape": (int(r["nx"]), int(r["ny"]), int(r["nz"])),
            "hu_mean": float(r["hu_mean"]), "hu_std": float(r["hu_std"]),
            "findings": []})
        vol["findings"].append({
            "seg_channel": int(r["seg_channel"]), "finding_idx": int(r["finding_idx"]),
            "text": str(r["text"]),
            "centroid": (float(r["cen0"]), float(r["cen1"]), float(r["cen2"]))})
    out = [index[k] for k in sorted(index)]
    short = sum(1 for v in out if min(v["shape"]) < min(PATCH))
    print(f"G-INDEX: {len(out)} volumes / {len(merged)} findings from "
          f"findings.csv[miccai_split == {split!r}, has_seg == 1]; "
          f"{short} of them are smaller than {PATCH[0]} on some axis and will be padded")
    return out


# ------------------------------------------------------------------- patch geometry

def crop_start(n: int, centre: float, size: int) -> int:
    """Start index of a `size`-long window centred on `centre`, clipped into [0, n - size].

    For n < size the only legal start is 0 and the caller pads. Returning a negative start
    here instead would slice from the far end of the array in numpy and read garbage.
    """
    return int(np.clip(round(centre - size / 2), 0, max(0, n - size)))


def pad_to_patch(arr: np.ndarray, patch=PATCH) -> np.ndarray:
    """Symmetric zero-pad the trailing 3 axes up to `patch`. G-SHAPE asserts the result.

    Zero is the right pad value only because this runs AFTER normalisation, where 0 is the
    volume mean -- which is exactly what `VoxTellPredictor.predict_sliding_window_return_logits`
    does (`pad_nd_image(..., 'constant', {'value': 0})` on the already-normalised image).
    """
    spatial = arr.shape[-3:]
    if any(s > p for s, p in zip(spatial, patch)):
        raise ValueError(f"pad_to_patch got {spatial}, larger than the patch {patch}")
    pads = [(0, 0)] * (arr.ndim - 3)
    for s, p in zip(spatial, patch):
        lo = (p - s) // 2
        pads.append((lo, p - s - lo))
    out = np.pad(arr, pads, mode="constant", constant_values=0)
    assert out.shape[-3:] == tuple(patch), \
        f"G-SHAPE: padded patch is {out.shape[-3:]}, must be {tuple(patch)} (pos_embed is fixed)"
    return out


# ------------------------------------------------------------------- intensity normalisation
#
# ONE authoritative home for "how a CT becomes model input". Four places consume it -- the
# training patch, the in-training val sweep, `predict_split.py`, and any re-scoring -- and they
# must agree or the number means nothing. A checkpoint records which one it was trained under
# and inference refuses to disagree (`gate_hu_norm`).
#
# `ct` is nnU-Net's own CT scheme (CTNormalization): clip, then a FIXED affine from
# dataset-level statistics. The constants below were measured over the TRAIN split only --
# 2,992 volumes / 292.7G voxels, clipped to [-1000, 1000] first -- so no test intensity enters
# the normalisation. Two disjoint 128-volume samples agreed to 2.97% on the mean before the
# full pass; the full pass is what is pinned here.
#
# WHY it exists: per-volume z-score lets a handful of voxels rescale the whole volume. 504 of
# 3,192 volumes (15.8%) have hu_std > 600 against a median of 483, and one measured case went
# 492 -> 1601 from 0.1% of voxels above 1000 HU. The same tissue then lands at a different
# input value depending on the patient. Measured on 10 volumes: HU +30 spans 0.52..1.23 under
# raw per-volume z-score, 0.94..1.24 after clipping, and a single value under this scheme --
# which sits within 0.03 of the per-volume median, so a normal volume barely moves.
HU_NORM_CT = {"clip": (-1000.0, 1000.0), "mean": -540.9803, "std": 482.0892}


def hu_norm_spec(name: str | None) -> dict | None:
    """`None`/"zscore" -> per-volume z-score (what B60 and E60 trained under). "ct" -> fixed.

    Returns a plain dict so it can go straight into a run record and a checkpoint and be
    compared field-by-field later. An unknown name is a refusal, not a fallback: silently
    training under z-score while the run record says otherwise is the failure this guards.
    """
    if name in (None, "", "zscore"):
        return None
    if name == "ct":
        return {"clip": tuple(HU_NORM_CT["clip"]), "mean": HU_NORM_CT["mean"],
                "std": HU_NORM_CT["std"]}
    raise SystemExit(f"--hu-norm {name!r} is not a scheme; use 'zscore' or 'ct'")


def normalise_fixed(hu: np.ndarray, spec: dict) -> np.ndarray:
    """nnU-Net `CTNormalization.run` arithmetic: clip, subtract mean, divide std, float32.

    Deliberately the same operation order and the same `max(std, eps)` guard as the upstream
    class, because `FixedAffineNormalization` hands this to `VoxTellPredictor` in place of its
    own normaliser -- train and inference must be the same expression, not two that agree.
    """
    lo, hi = spec["clip"]
    image = np.clip(hu.astype(np.float32), np.float32(lo), np.float32(hi))
    image -= np.float32(spec["mean"])
    image /= np.float32(max(spec["std"], 1e-8))
    return image


class FixedAffineNormalization:
    """Drop-in for nnU-Net's `ZScoreNormalization` on a `VoxTellPredictor`.

    The predictor calls `self.normalization.run(image, seg)` inside `preprocess`, so replacing
    the attribute is what makes the SWEEP and the submission path use the same normalisation
    the patches were trained under. Pre-clipping the array instead would only reproduce
    clip-then-z-score, which is a different scheme -- and one whose per-volume statistics still
    depend on how much air the field of view happens to contain.
    """

    def __init__(self, spec: dict) -> None:
        self.spec = spec

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return normalise_fixed(image, self.spec)


def gate_hu_norm(checkpoint_blob: dict, requested: str, where: str) -> dict | None:
    """Refuse to run inference under a normalisation the checkpoint was not trained under.

    Nothing downstream would notice: the masks come out, the scorer scores them, every gate
    stays green, and the number is simply wrong by however much the input distribution moved.
    Checkpoints written before this flag existed carry no `hu_norm`, and z-score is the right
    reading of that -- it is what they trained under.
    """
    trained = str((checkpoint_blob.get("args") or {}).get("hu_norm", "zscore") or "zscore")
    if trained != requested:
        raise SystemExit(
            f"{where}: checkpoint was trained with --hu-norm {trained!r} but this run asks for "
            f"{requested!r}. The masks would be produced from a different input distribution "
            f"than the weights were fitted to, and no metric would show it.")
    return hu_norm_spec(requested)


def normalise(hu: np.ndarray, mean: float, std: float) -> np.ndarray:
    """nnU-Net ZScoreNormalization, VOLUME statistics, RAW HU.

    Verbatim arithmetic from `ZScoreNormalization.run` with `use_mask_for_norm` unset:
    `image -= mean; image /= max(std, 1e-8)` in float32. The statistics come from
    stats.csv, i.e. over EVERY voxel of the whole volume -- not this patch, and not the
    `crop_to_nonzero` box (measured a no-op on 5/5 CTs: HU air is -1000, so "nonzero"
    selects the whole array).
    """
    return (hu.astype(np.float32) - np.float32(mean)) / np.float32(max(std, 1e-8))


def binarise(mask_channel: np.ndarray) -> np.ndarray:
    """G-BINARY at the source: check the RAW channel is an instance map, then `> 0`.

    The check is on the INPUT, not the output. `(x > 0).astype(float32)` can only ever
    produce {0, 1} -- asserting that afterwards is a tautology and was measured to be
    unreachable for uint8 {0,2,3}, floats, NaN and -1 alike. What `> 0` DOES silently
    mangle is an input that is not a non-negative integer instance map:
      * a float probability/soft mask -> everything above 0.0 becomes foreground;
      * nnU-Net's -1 "ignore" label   -> becomes background, silently.
    Both are cache-rebuild accidents, both survive to a garbage checkpoint. This fires.
    """
    if not np.issubdtype(mask_channel.dtype, np.integer) or mask_channel.min() < 0:
        raise AssertionError(
            f"G-BINARY: mask channel is dtype {mask_channel.dtype} with min "
            f"{mask_channel.min()}, not a non-negative integer instance map. "
            "'mask[c] > 0' would silently threshold a soft mask, or fold a -1 ignore "
            "label into background. Fix the cache, do not widen this check.")
    return (mask_channel > 0).astype(np.float32)


def assert_binary(t: torch.Tensor, where: str) -> None:
    """G-BINARY on a batched tensor. Cheap: two comparisons, no unique() sort.

    HONEST STATUS: unreachable while `binarise` sits in the gt path -- measured, it does not
    fire on a real {0,2,3} mask driven through `__getitem__`. It is kept because it costs two
    comparisons and it IS the check that fires the day someone edits the gt path.
    """
    lo, hi = float(t.min()), float(t.max())
    assert (t == 0).logical_or(t == 1).all(), (
        f"G-BINARY [{where}]: target range [{lo}, {hi}] holds values outside {{0, 1}}. "
        "49 of 70 native seg channels are entity-INSTANCE maps (values 2..5); BCE against "
        "target=3 is unbounded below and the run would finish with a garbage checkpoint.")


# ------------------------------------------------------------------------- dataset

class VoxTellPatches(torch.utils.data.Dataset):
    """One item = one 192^3 patch of one CT + QUERIES sentences of THAT CT + their masks.

    The item is emitted in nnU-Net's (nz, ny, nx) axis order -- see G-AXIS below -- which is
    the order `val_sweep` feeds `VoxTellPredictor` and the order VoxTell was pretrained in.
    The cache, the stats.csv centroids and all the crop arithmetic stay in RAS (nx, ny, nz);
    only the finished cube is permuted.

    Length is `steps * batch_size`, so an "epoch" is the run. Index i seeds its own RNG, so
    the sample order is identical across arms B and C, reproducible, and independent of the
    number of dataloader workers.

    Sampling, stated plainly:
      * a volume is drawn uniformly at random (not a finding), because the patch is shared;
      * QUERIES findings are drawn from that volume WITHOUT replacement when it has enough,
        and WITH replacement when it does not. 35.3% of train volumes hold exactly one
        finding (13.7% of train findings), and those patches therefore carry the same
        sentence twice. The alternative -- padding with a sentence from another volume as a
        negative -- was rejected: "nodule in the right lung" is frequently ALSO true of the
        other volume, so the all-zero target would be wrong label noise, not a negative.
      * with probability `fg_prob` the patch is centred on a query's stats.csv centroid plus
        uniform jitter of +-patch/4, which keeps that finding inside the patch; otherwise the
        start is uniform over the legal range.
    """

    def __init__(self, volumes: list[dict], npy_root: Path, bank: dict[str, np.ndarray],
                 length: int, seed: int, fg_prob: float, queries: int = QUERIES,
                 tok_bank=None, start_index: int = 0, hu_norm: dict | None = None):
        # None keeps the per-volume z-score arms B/C/E trained under; a spec switches every
        # patch to the fixed affine. One flag reaches here and `val_sweep` from the same
        # `hu_norm_spec` call, so the patches and the sweep cannot disagree.
        self.hu_norm = hu_norm
        self.volumes, self.bank = volumes, bank
        self.ct_dir, self.seg_dir = npy_root / "ct", npy_root / "seg"
        self.length, self.seed, self.fg_prob, self.queries = length, seed, fg_prob, queries
        # v13 only. None keeps the returned dict exactly what arms B and C saw.
        self.tok_bank = tok_bank
        # --resume: skip the indices the interrupted run already consumed. Every index seeds
        # its own rng from (seed, i), so index i is the SAME sample either way -- this is the
        # same data in the same order continued, not a re-shuffle.
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return self.length - self.start_index

    def __getitem__(self, i: int) -> dict:
        i += self.start_index          # --resume: continuation index 0 IS original index start_index
        rng = np.random.default_rng((self.seed * 1_000_003 + i) % (2 ** 63))
        vol = self.volumes[rng.integers(len(self.volumes))]
        n_f = len(vol["findings"])
        pick = rng.choice(n_f, self.queries, replace=n_f < self.queries)
        chosen = [vol["findings"][int(k)] for k in pick]

        shape = vol["shape"]
        if rng.random() < self.fg_prob:
            centre = chosen[int(rng.integers(self.queries))]["centroid"]
            jitter = rng.integers(-(PATCH[0] // 4), PATCH[0] // 4 + 1, size=3)
            start = [crop_start(n, c + j, p)
                     for n, c, j, p in zip(shape, centre, jitter, PATCH)]
        else:
            start = [int(rng.integers(0, max(1, n - p + 1))) for n, p in zip(shape, PATCH)]
        sl = tuple(slice(s, s + p) for s, p in zip(start, PATCH))

        ct = np.load(self.ct_dir / f"{vol['name']}.npy", mmap_mode="r")
        patch_hu = np.asarray(ct[sl])
        image = pad_to_patch(normalise_fixed(patch_hu, self.hu_norm) if self.hu_norm
                             else normalise(patch_hu, vol["hu_mean"], vol["hu_std"]))

        with np.load(self.seg_dir / f"{vol['name']}.npz") as z:
            masks = z["mask"]
            channels = [f["seg_channel"] for f in chosen]
            if max(channels) >= masks.shape[0]:
                raise IndexError(f"{vol['name']}: seg_channel {max(channels)} but the cached "
                                 f"mask has {masks.shape[0]} channels")
            gt = np.stack([binarise(masks[c][sl]) for c in channels])
        gt = pad_to_patch(gt)

        # G-AXIS. nnU-Net's NibabelIOWithReorient reorients to RAS and THEN transposes
        # (2, 1, 0) -- `reoriented_image.get_fdata().transpose((2, 1, 0))`, verbatim -- so
        # every array VoxTell was pretrained on is (nz, ny, nx). The npy cache is RAS
        # (nx, ny, nz), and `val_sweep` already feeds the predictor `ct.transpose(2, 1, 0)`.
        # Without this the trainer optimises in one axis order and the sweep scores in
        # another; the patch is a cube, so G-SHAPE cannot see it and the run finishes
        # normally with a checkpoint that is wrong for its own evaluator. Only 12 of the
        # 3,192 cached volumes have nx == nz, so this is a real permutation, not a no-op.
        # Applied AFTER padding: the pads are per-axis and symmetric, so pad-then-transpose
        # and transpose-then-pad are the same array.
        image = image.transpose(2, 1, 0)
        gt = gt.transpose(0, 3, 2, 1)

        text = np.stack([self.bank[f["text"].lower()] for f in chosen]).astype(np.float32)
        out = {"image": torch.from_numpy(image[None].copy()),
               "text": torch.from_numpy(text),
               "gt": torch.from_numpy(gt.copy()),
               "name": vol["name"],
               "channels": torch.tensor(channels, dtype=torch.long)}
        if self.tok_bank is not None:
            arr, index, lengths = self.tok_bank
            rows = [index[f["text"].lower()] for f in chosen]
            out["tokens"] = torch.from_numpy(np.stack([np.asarray(arr[r]) for r in rows]))
            # Pad length comes from the bank's own second dimension, never from a constant:
            # a constant can disagree with the array it describes, and a too-short mask would
            # silently hide real tokens from attention with no shape error.
            out["tok_mask"] = torch.stack([
                torch.arange(arr.shape[1]) < int(lengths[r]) for r in rows])
        return out


# ---------------------------------------------------------------------------- loss

def dice_bce(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Soft Dice + BCE, reduced PER (sample, query) and then averaged.

    The reduction is the whole point, and it is the lesson the 0.1538 bar came from: the official
    metric is `mean_global_dice_per_finding`, so a 200-voxel finding and a 200,000-voxel one
    must carry the same weight. Summing the Dice numerator over the batch first scores the
    batch as ONE region, and at this foreground fraction that region is whichever query
    happens to be largest.

    Both terms are strictly bounded below by 0 on a BINARISED target, which is what makes
    G-LOSS-POSITIVE able to catch a G-BINARY bypass.

    `mask` is an optional (sample, query) bool selecting which pairs enter the mean. It exists
    for the multi-class trainer, where a volume holds 2-3 of 14 classes and averaging over all
    14 makes the objective mostly "predict nothing": on an EMPTY target the Dice term is a
    cliff -- 0 for an exactly-zero prediction and ~1 for any nonzero one -- so eleven empty
    channels outvote the two real ones and the model collapses. `None` keeps the historical
    behaviour for every existing caller; the change is additive.
    """
    logits = logits.float()
    dims = tuple(range(2, logits.ndim))
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dims)
    dice = 1.0 - (2.0 * inter + smooth) / (prob.sum(dims) + target.sum(dims) + smooth)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dims)
    per = dice + bce
    if mask is None:
        return per.mean()
    if mask.shape != per.shape:
        raise ValueError(f"dice_bce mask is {tuple(mask.shape)}, expected {tuple(per.shape)}")
    if not bool(mask.any()):
        raise ValueError("dice_bce mask selects no (sample, query) pair; the step would have no "
                         "gradient and returning 0 would hide that")
    return per[mask].mean()


def ds_targets(gt: torch.Tensor, shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    """Deep-supervision targets by MAX pooling, so a 1-voxel finding survives downsampling.

    `F.interpolate(mode='nearest')` drops it. Each requested shape must be an exact power-of-
    two division of the previous one; anything else raises rather than silently resampling.
    """
    out, cur = [], gt
    for k, want in enumerate(shapes):
        if k:
            factor = tuple(p // w for p, w in zip(out[-1].shape[2:], want))
            if any(f < 1 for f in factor) or \
                    tuple(p // f for p, f in zip(out[-1].shape[2:], factor)) != tuple(want):
                raise ValueError(f"deep-supervision head {k} wants {want}, which is not an "
                                 f"integer division of {tuple(out[-1].shape[2:])}")
            cur = F.max_pool3d(out[-1], kernel_size=factor, stride=factor)
        if tuple(cur.shape[2:]) != tuple(want):
            raise ValueError(f"deep-supervision head {k}: target {tuple(cur.shape[2:])} != "
                             f"output {tuple(want)}")
        out.append(cur)
    return out


def ds_weights(n: int) -> list[float]:
    """nnU-Net's 1 / 2^i, normalised to sum 1."""
    w = [0.5 ** i for i in range(n)]
    total = sum(w)
    return [x / total for x in w]


# ------------------------------------------------------------------------- G-TEXT

def gate_text(bank: dict[str, np.ndarray], wrapped_example: str, example_key: str,
              needed: list[str], enc: TextEncoder = None) -> dict:
    """G-TEXT. Raises before a single GPU-second is spent.

    Three separate failure modes, all of them silent at runtime:
      1. an L2-normalised bank (the BioLORD one) -- a 138x scale error into a pretrained
         Linear(2560, 2048), which produces plausible-looking garbage, not a crash;
      2. a bank built WITHOUT the instruction wrapper -- cos(with, without) = 0.7366 median;
      3. a sentence that is not in the bank at all, found on volume 137 of 200.

    `enc` says which conventions to check against; every production call site resolves it
    with encoder_for(bank_path) so it comes from the bank itself. It defaults to qwen3 only
    because that is what every bank predating the field is, and what the tests exercise.
    """
    enc = enc or ENCODERS["qwen3"]
    if not bank:
        raise SystemExit("G-TEXT: the text bank is empty")
    keys = list(bank)
    norms = np.linalg.norm(np.stack([np.asarray(bank[k], dtype=np.float32)
                                     for k in keys]), axis=1)
    med = float(np.median(norms))
    assert enc.norm_lo < med < enc.norm_hi, (
        f"G-TEXT[{enc.key}]: median ||e||_2 = {med:.3f}, outside ({enc.norm_lo}, {enc.norm_hi}). "
        "VoxTell consumes RAW last-token-pooled Qwen3 vectors (measured 137.5); an "
        "L2-normalised bank reads 1.0 and is a 138x scale error into project_text_embed. "
        "a BioLORD bank is 768-d AND L2-normalised -- it must "
        "never be used here.")
    # The MEDIAN band above is the scale gate and stays tight -- it is what catches an
    # L2-normalised bank. The per-vector check below deliberately does NOT reuse it.
    #
    # It used to, with (130, 145), and that band came from a 380-sentence sample whose spread
    # was 135.7-137.9. On the real 6,467-sentence corpus the range is 128.11-139.93, so the
    # tight band failed 6 legitimate sentences (0.09%) -- all six read as ordinary complete
    # clinical English ("paraseptal emphysema at the apical levels of both lungs", 128.83).
    # A per-vector band tuned on a sample is the wrong instrument for a scale error: it cannot
    # tell 128.11 from 137.5 in any way that matters, while the thing it exists to catch is
    # off by a factor of 138. So the per-vector check now asks only whether a vector is
    # DEGENERATE -- non-finite, or an order of magnitude from the corpus scale. An
    # L2-normalised bank (1.0) and a zero vector still fail it by ~100x.
    assert np.isfinite(norms).all(), (
        f"G-TEXT: {int((~np.isfinite(norms)).sum())} of {len(norms)} vectors are non-finite. "
        "A NaN row survives the median check and silently poisons every batch it lands in.")
    lo, hi = med / VEC_NORM_FACTOR, med * VEC_NORM_FACTOR
    bad = [i for i, n in enumerate(norms) if not lo < n < hi]
    assert not bad, (
        f"G-TEXT: {len(bad)} of {len(norms)} vectors have ||e||_2 outside "
        f"({lo:.1f}, {hi:.1f}) -- more than {VEC_NORM_FACTOR}x from the bank's own median "
        f"{med:.1f}. e.g. {[(keys[i][:60], round(float(norms[i]), 2)) for i in bad[:3]]}. "
        "A vector this far from the corpus scale is degenerate (empty or non-English "
        "sentence, truncated encode, or a row built at the wrong scale).")

    # The wrapper is Qwen3-Embedding's own retrieval protocol, not a general convention. For
    # an encoder that does not use it, the check inverts: the stored prompt must be the BARE
    # key, which is what proves no wrapper leaked in from a copy-pasted build path.
    if enc.wrap:
        from voxtell.utils.text_embedding import wrap_with_instruction
        assert wrapped_example.startswith("Instruct: "), (
            f"G-TEXT[{enc.key}]: the bank's stored prompt starts {wrapped_example[:24]!r}, not "
            "'Instruct: '. It was built without voxtell.utils.text_embedding."
            "wrap_with_instruction; cos(with wrapper, without) = 0.7366 median.")
        want = wrap_with_instruction([example_key])[0]
        assert wrapped_example == want, (
            "G-TEXT: the bank's stored prompt is not what wrap_with_instruction produces today "
            f"for its own key.\n  bank: {wrapped_example[:120]!r}\n  now : {want[:120]!r}")
    else:
        assert wrapped_example == example_key, (
            f"G-TEXT[{enc.key}]: this encoder takes NO instruction wrapper, but the bank's "
            f"stored prompt {wrapped_example[:60]!r} is not its bare key {example_key[:60]!r}. "
            "The bank was built through the qwen3 path.")

    missing = sorted({s.lower() for s in needed} - set(bank))
    assert not missing, (f"G-TEXT: {len(missing)} sentences are not in the bank, e.g. "
                         f"{missing[:3]}. Rebuild it with --build-text-bank-only.")
    dim = len(next(iter(bank.values())))
    assert dim == enc.emb_dim, \
        f"G-TEXT[{enc.key}]: bank width {dim} != VoxTell's text_embedding_dim {enc.emb_dim}"
    print(f"G-TEXT ok: {len(bank)} sentences, ||e||_2 min {norms.min():.1f} / median "
          f"{med:.1f} / max {norms.max():.1f}, dim {dim}, wrapper {wrapped_example[:32]!r}")
    return {"n": len(bank), "norm_median": med, "norm_min": float(norms.min()),
            "norm_max": float(norms.max())}


def load_text_bank(path: Path) -> tuple[dict[str, np.ndarray], str, str]:
    """Read the bank written by `build_text_bank`. Also loadable by VoxTellPredictor."""
    with np.load(path, allow_pickle=False) as z:
        if not {"labels", "embeddings"} <= set(z.files):
            raise SystemExit(
                f"{path} holds arrays {sorted(z.files)}, not VoxTell's "
                "{'labels', 'embeddings'} container. A BioLORD bank instead writes "
                "{'keys', 'emb'} -- that is the 768-d L2-NORMALISED BioLORD bank and it "
                "must never be fed to VoxTell. Build this one with --build-text-bank-only.")
        labels = [str(x) for x in z["labels"]]
        emb = z["embeddings"].astype(np.float32)
        wrapped = str(z["wrapped_example"][0]) if "wrapped_example" in z.files else ""
    if not wrapped:
        raise SystemExit(f"{path} has no 'wrapped_example' array, so G-TEXT cannot prove it "
                         "went through wrap_with_instruction. Rebuild it here.")
    return {lab.lower(): emb[i] for i, lab in enumerate(labels)}, wrapped, labels[0].lower()


def encoder_prompts(enc: TextEncoder, keys) -> list[str]:
    """The exact strings handed to the tokenizer. ONE definition, used by BOTH bank builders
    and by the gates, so the pooled bank and the per-token bank cannot wrap differently --
    which is the drift the two-bank equality check exists to catch."""
    if enc.wrap:
        from voxtell.utils.text_embedding import wrap_with_instruction
        return wrap_with_instruction(list(keys))
    return [str(k) for k in keys]


def encoder_load(enc: TextEncoder, device: str):
    """Tokenizer + model with THIS encoder's padding side. Padding side is not cosmetic:
    last_token_pool reads position -1, which is the last real token only under left padding."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc.hf_path, padding_side=enc.padding_side)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(enc.hf_path, dtype=dtype).to(device).eval()
    return tok, model


def encoder_pool(enc: TextEncoder, hidden, attention_mask):
    """The pooled vector, in float32. `gate_token_bank` re-derives this from the per-token
    bank and compares, so any change here must keep that relation statable."""
    hidden = hidden.float()
    if enc.pooling == "last_token":
        from voxtell.utils.text_embedding import last_token_pool
        return last_token_pool(hidden, attention_mask)
    if enc.pooling == "mean":
        m = attention_mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1)
    raise SystemExit(f"unknown pooling {enc.pooling!r} for encoder {enc.key!r}")


def build_text_bank(sentences: list[str], out: Path, device: str, batch: int = 16,
                    enc: TextEncoder = None) -> Path:
    """Encode every unique sentence with `enc` and write the pooled bank.

    RAW pooled vectors -- no standardising, no L2 normalisation. For qwen3 the pooling and
    the instruction wrapper are IMPORTED from `voxtell.utils.text_embedding`, the same two
    functions `VoxTellPredictor._compute_text_embeddings` calls, so the training bank and the
    inference path cannot drift.

    Written in VoxTell's own {labels, embeddings} container plus two extra arrays
    (`load_embedding_bank` reads only the first two, so the file stays usable as a predictor
    bank): `wrapped_example`, which G-TEXT checks the wrapper against, and `encoder_key`,
    which is how `encoder_for` recovers the conventions from the bank itself rather than from
    a flag someone could set inconsistently.
    """
    enc = enc or ENCODERS["qwen3"]
    keys = list(dict.fromkeys(s.lower() for s in sentences))
    tok, model = encoder_load(enc, device)
    chunks, t0 = [], time.time()
    with torch.inference_mode():
        for start in range(0, len(keys), batch):
            batch_in = tok(encoder_prompts(enc, keys[start:start + batch]), padding=True,
                           truncation=True, max_length=8192, return_tensors="pt")
            batch_in = {k: v.to(device) for k, v in batch_in.items()}
            hidden = model(**batch_in).last_hidden_state
            chunks.append(encoder_pool(enc, hidden, batch_in["attention_mask"]).cpu().numpy())
            if (start // batch) % 20 == 0:
                print(f"  {start + len(chunks[-1])}/{len(keys)}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    emb = np.concatenate(chunks).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, labels=np.array(keys, dtype=np.str_), embeddings=emb,
             wrapped_example=np.array(encoder_prompts(enc, [keys[0]]), dtype=np.str_),
             encoder_key=np.array([enc.key], dtype=np.str_))
    norms = np.linalg.norm(emb, axis=1)
    print(f"wrote {out}  [{enc.key}] {len(keys)} sentences x {emb.shape[1]}  "
          f"||e||_2 min {norms.min():.1f} / median {np.median(norms):.1f} / "
          f"max {norms.max():.1f}  {(time.time() - t0) / 60:.1f} min")
    return out


def token_bank_path_for(text_bank_path: Path) -> Path:
    """The per-token bank that must pair with a given pooled bank.

    Whatever qualifier the pooled bank carries (e.g. the "_all" test-split extension in
    text_emb_qwen3_all.npz) has to carry over, or the two banks silently stop describing the
    same sentence set. Centralised here so the trainer, predict_split.py, and the probes never
    each hardcode TOKEN_BANK_NAME and drift independently -- which is exactly how this broke:
    2026-08-11, `--fuse trunk --text-bank text_emb_qwen3_all.npz` against the test split loaded
    the extended pooled bank but the UNEXTENDED text_tok_qwen3.npy token bank, and
    gate_token_bank correctly refused (2675/6467 sentences mismatched) rather than scoring the
    submission against silently-wrong text.
    """
    return text_bank_path.with_name(
        text_bank_path.name.replace("text_emb_", "text_tok_", 1)).with_suffix(".npy")


def load_token_bank(path: Path):
    """mmap the token array so training and eval processes share one copy of the 4 GB."""
    meta_path = path.with_name(path.stem + "_meta.npz")
    if not path.is_file() or not meta_path.is_file():
        raise SystemExit(f"no token bank at {path} (+ {meta_path.name}). "
                         "Run --build-token-bank-only first.")
    meta = np.load(meta_path, allow_pickle=True)
    arr = np.load(path, mmap_mode="r")
    labels = [str(s) for s in meta["labels"]]
    return arr, {s: i for i, s in enumerate(labels)}, meta["lengths"], meta


# G-BANK-CONSISTENT's bit-identical form only holds under `last_token` pooling with left
# padding, where the pooled vector IS a stored position and no arithmetic happens. For a
# mean-pooled encoder the two banks sum the same bf16 hidden states in different orders, so
# this bounds the disagreement instead.
#
# MEASURED, not guessed: over the real 6,467-sentence gatortron pair the max |delta| is
# 5.96e-08 -- float32 rounding on a ~14-term sum of ~0.2-magnitude elements, i.e. the two
# banks agree as exactly as float32 permits. 1e-5 leaves ~170x headroom for longer sentences
# (max length 53 vs median 14) and different batch composition, while staying ~10,000x
# tighter than the 1e-1 that a real wrapper/padding/tokeniser drift produces. A tolerance
# set loose "to be safe" is a gate that cannot fail; this one is set from the measurement.
MEAN_TOL = 1e-5


def gate_token_bank(arr, index, lengths, meta, pooled: dict[str, np.ndarray],
                    sentences: list[str], enc: TextEncoder = None) -> dict:
    """G-TEXT-TOK + G-BANK-CONSISTENT. Raises before a single GPU-second is spent.

    G-TEXT's norm band is NOT reused: it was measured on pooled vectors and says nothing
    about the other 60-odd positions, so a band copied across would be a check that passes
    by construction. What is checked instead is the one relation between the two banks that
    must hold, and its form depends on the pooling:

      last_token (qwen3)  the pooled bank stores `last_token_pool`, which under left padding
                          IS the last real token, so tok[i, len-1] must EQUAL pooled[i].
                          Measured over all 6,467 sentences it is bit-identical, so the
                          assert is equality, not a threshold.
      mean (gatortron)    the pooled vector is not any single stored position, so the
                          relation is pooled[i] == mean of tok[i, :len]. That is arithmetic
                          over bf16 hidden states summed in a different order by the two
                          builders, so it is NOT bit-identical and cannot be asserted as
                          equality; MEAN_TOL bounds it instead. The check still catches what
                          it exists to catch -- a wrapper, tokeniser, padding-side or model
                          difference between the two builds moves vectors by ~1e-1, not 1e-3.

    Either way the relation only holds if the positions the pooling covers are exactly the
    positions the token bank stores, which is why gatortron's bank_prefix is 0.
    """
    enc = enc or ENCODERS["qwen3"]
    labels = [str(s) for s in meta["labels"]]
    n = len(labels)
    assert arr.shape == (n, enc.bank_len, enc.emb_dim), \
        f"G-TEXT-TOK[{enc.key}]: token bank is {arr.shape}, expected {(n, enc.bank_len, enc.emb_dim)}"
    assert int(meta["prefix_tokens"][0]) == enc.bank_prefix, (
        f"G-TEXT-TOK[{enc.key}]: bank was built with prefix {int(meta['prefix_tokens'][0])} but "
        f"the code now says {enc.bank_prefix}; every sentence's first token would shift")
    assert lengths.min() > 0 and lengths.max() <= enc.bank_len, \
        f"G-TEXT-TOK: lengths span [{lengths.min()}, {lengths.max()}], outside (0, {enc.bank_len}]"

    wrapped = str(meta["wrapped_example"][0])
    want = encoder_prompts(enc, [labels[0]])[0]
    assert wrapped == want, ("G-TEXT-TOK: the bank's stored prompt is not what this encoder "
                             f"produces today.\n  bank: {wrapped[:110]!r}\n  now : {want[:110]!r}")

    missing = sorted({s.lower() for s in sentences} - set(index))
    assert not missing, (f"G-TEXT-TOK: {len(missing)} sentences are not in the token bank, "
                         f"e.g. {missing[:2]}")

    # ---- G-BANK-CONSISTENT
    pooled_stack = np.stack([pooled[s] for s in labels])
    if enc.pooling == "last_token":
        got = np.asarray(arr[np.arange(n), lengths - 1])
        bad = int((got != pooled_stack).any(axis=1).sum())
        detail = "have a last token that differs from"
        dev = float(np.abs(got - pooled_stack).max())
    elif enc.pooling == "mean":
        got = np.stack([np.asarray(arr[i, :int(lengths[i])]).mean(0) for i in range(n)])
        dev = float(np.abs(got - pooled_stack).max())
        bad = int((np.abs(got - pooled_stack) > MEAN_TOL).any(axis=1).sum())
        detail = f"have a mean over stored positions more than {MEAN_TOL} from"
    else:
        raise SystemExit(f"G-BANK-CONSISTENT: no relation defined for pooling {enc.pooling!r}")
    assert bad == 0, (
        f"G-BANK-CONSISTENT[{enc.key}]: {bad} of {n} sentences {detail} the pooled bank "
        f"(max |delta| = {dev:.2e}). The two banks did not come from the same encoding -- "
        "check the wrapper, the padding side, the tokeniser and the model revision before "
        "trusting any number from this arm.")

    pad = float(np.abs(arr[::97, -1]).max()) if int(lengths.max()) < enc.bank_len else 0.0
    assert pad == 0.0, f"G-TEXT-TOK: padded positions are not zero (max |v| = {pad})"

    # Reported for eyeballing the bank's scale; the gate above is what actually decides.
    norms = np.linalg.norm(np.asarray(arr[np.arange(n), lengths - 1]), axis=1)
    rel = ("every last token is bit-identical to the pooled bank" if enc.pooling == "last_token"
           else f"every mean over stored positions is within {dev:.2e} of the pooled bank "
                f"(tol {MEAN_TOL})")
    print(f"G-TEXT-TOK[{enc.key}] ok: {n} sentences x {enc.bank_len} x {enc.emb_dim}, prefix "
          f"{enc.bank_prefix} dropped, lengths {lengths.min()}-{lengths.max()} "
          f"(median {int(np.median(lengths))}); G-BANK-CONSISTENT ok: {rel}, "
          f"||e||_2 median {np.median(norms):.1f}")
    return {"n": n, "len_median": int(np.median(lengths)), "len_max": int(lengths.max()),
            "prefix_tokens": enc.bank_prefix, "pad_len": enc.bank_len,
            "encoder": enc.key, "consistency_max_delta": dev}


def build_token_bank(sentences: list[str], out_npy: Path, device: str, batch: int = 16,
                     enc: TextEncoder = None) -> Path:
    """Per-token vectors for the v13 encoder-fusion arms. The pooled bank is untouched.

    Same sentences, same wrapper, same tokenizer, same batch size and same order as
    `build_text_bank` -- the two banks are the SAME encoding read at two granularities, and
    `gate_token_bank` is what proves they stayed that way.

    Which positions get stored is the encoder's `bank_prefix`, and it is not a free choice:

      qwen3      prefix 23. Left padding puts the real tokens last, so sample i's query
                 tokens are the final `n_i - 23` positions. The instruction prefix is
                 dropped because it is identical in every sentence and would otherwise
                 occupy a third of the attention keys. 23 and not 24: tokenising the prefix
                 alone gives 25 ids (24 + its own EOS), but the space after "Query:" merges
                 into the following word for some sentences, so only the first 23 are shared
                 by all 6,467. Cutting at 24 silently eats the first query token of every
                 sentence that merged.
      gatortron  prefix 0. Right padding puts the real tokens FIRST, and the pooling is a
                 mean over the attention-masked span -- so the stored span must be exactly
                 that span, [CLS] included. Dropping [CLS] here would make the pooled vector
                 underivable from the token bank and delete G-BANK-CONSISTENT entirely.

    Writes .npy + a meta .npz rather than one container so the 4.5 GB array can be mmapped by
    several training and eval processes at once.
    """
    enc = enc or ENCODERS["qwen3"]
    keys = list(dict.fromkeys(s.lower() for s in sentences))
    tok, model = encoder_load(enc, device)

    toks = np.zeros((len(keys), enc.bank_len, enc.emb_dim), dtype=np.float32)
    lengths = np.zeros(len(keys), dtype=np.int32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(keys), batch):
            chunk = keys[start:start + batch]
            batch_in = tok(encoder_prompts(enc, chunk), padding=True, truncation=True,
                           max_length=8192, return_tensors="pt")
            batch_in = {k: v.to(device) for k, v in batch_in.items()}
            hidden = model(**batch_in).last_hidden_state.float()
            n = batch_in["attention_mask"].sum(dim=1)
            for i in range(len(chunk)):
                q = int(n[i]) - enc.bank_prefix
                assert 0 < q <= enc.bank_len, (
                    f"{keys[start + i]!r} has {q} stored tokens, outside (0, {enc.bank_len}]. "
                    f"Raise {enc.key}'s bank_len; do NOT truncate.")
                # Left padding -> the real span is the LAST q positions; right padding -> the
                # FIRST q. Taking the wrong end yields pure padding and no shape check sees it.
                span = hidden[i, -q:] if enc.padding_side == "left" else hidden[i, :q]
                toks[start + i, :q] = span.cpu().numpy()
                lengths[start + i] = q
            if (start // batch) % 40 == 0:
                print(f"  {start + len(chunk)}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)

    assert np.isfinite(toks).all(), "non-finite token vectors"
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, toks)
    np.savez(out_npy.with_name(out_npy.stem + "_meta.npz"),
             labels=np.array(keys, dtype=np.str_), lengths=lengths,
             prefix_tokens=np.array([enc.bank_prefix]),
             encoder_key=np.array([enc.key], dtype=np.str_),
             wrapped_example=np.array(encoder_prompts(enc, [keys[0]]), dtype=np.str_))

    last = toks[np.arange(len(keys)), lengths - 1]
    norms = np.linalg.norm(last, axis=1)
    print(f"wrote {out_npy}  [{enc.key}] {len(keys)} x {enc.bank_len} x {enc.emb_dim}  "
          f"({toks.nbytes / 2**30:.2f} GiB)  lengths min {lengths.min()} "
          f"median {int(np.median(lengths))} max {lengths.max()}  "
          f"last-token ||e||_2 median {np.median(norms):.1f}  {(time.time() - t0) / 60:.1f} min")
    return out_npy


# ---------------------------------------------------------------------------- model

def parse_val_at(spec: str) -> list[int]:
    """"5000,10000" -> [5000, 10000]; "none"/"off"/"" -> []. One parser, two callers.

    `none` exists because an argv cannot carry an empty string through every launcher, and
    because "--val-at ," would work by accident and read as a typo to the next person.
    """
    if spec.strip().lower() in ("", "none", "off"):
        return []
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def val_at_early_check(args) -> bool:
    """Does this run have any in-training sweep? Parsed exactly as main() parses it later.

    Separate function only so the refusal above and the schedule below cannot drift into
    disagreeing about what "no sweeps" means.
    """
    return bool(parse_val_at(args.val_at))


def build_network(random_init: bool, deep_supervision: bool, fuse_into_trunk: bool = None):
    """VoxTell v1.0, either the pretrained weights or arm C's control.

    Arm C differs from arm B by the `load_state_dict` call and NOTHING else -- same
    construction, same seed, same everything downstream. `strict=True` is not decoration:
    a silently-partial load would make arm B a slightly-warmer arm C.

    `fuse_into_trunk` defaults to None, which builds the stock model and takes
    the strict=True path unchanged -- arms B, C and every existing eval keep running the
    code they always ran. True/False select the v13 ablation pair (see voxtell_fusion), and
    only there does the load relax to strict=False, guarded by G-LOAD-KEYS below.

    """
    import pydoc
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from voxtell.model.voxtell_model import VoxTellModel

    paths.require("REXG_VOXTELL_MODEL_DIR",
                  needs="VoxTell v1.0: plans.json + fold_0/checkpoint_final.pth")
    plans = load_json(join(MODEL_DIR, "plans.json"))
    conf = plans["configurations"]["3d_fullres"]
    if tuple(conf["patch_size"]) != PATCH:
        raise SystemExit(f"plans.json patch_size {conf['patch_size']} != {PATCH}; the "
                         "pos_embed buffer fixes this and every geometry constant here")
    kwargs = dict(**conf["architecture"]["arch_kwargs"])
    for key in conf["architecture"]["_kw_requires_import"]:
        if kwargs[key] is not None:
            kwargs[key] = pydoc.locate(kwargs[key])
    cls, extra = VoxTellModel, {}
    expected_new = None
    if fuse_into_trunk is not None:
        from voxtell_fusion import FusedVoxTellModel, fusion_param_names
        cls, extra = FusedVoxTellModel, dict(fuse_into_trunk=fuse_into_trunk,
                                             text_token_dim=EMB_DIM)
        expected_new = fusion_param_names
    net = cls(input_channels=1, **kwargs, decoder_layer=4,
              text_embedding_dim=EMB_DIM, num_maskformer_stages=5, num_heads=32,
              query_dim=2048, project_to_decoder_hidden_dim=2048,
              deep_supervision=deep_supervision, **extra)
    n_params = sum(p.numel() for p in {id(p): p for p in net.parameters()}.values())
    if random_init:
        # Two callers, one flag: `main()` passes random_init for ARM C, the control. The
        # sweep and predict_split pass it to skip a 5 GB load whose weights the run's own
        # checkpoint is about to replace anyway. Saying "ARM C" in an arm-E sweep log is how
        # a reader concludes the wrong model was scored, so the message says only what is true.
        print(f"*** --random-init: the pretrained checkpoint is NOT loaded "
              f"({n_params / 1e6:.2f}M params). Training this way is arm C, the control; in a "
              f"sweep it just skips a load the checkpoint replaces. ***")
        return net, n_params
    ckpt = torch.load(join(MODEL_DIR, "fold_0", "checkpoint_final.pth"),
                      map_location="cpu", weights_only=False)
    if expected_new is None:
        result = net.load_state_dict(ckpt["network_weights"], strict=True)
        print(f"loaded {MODEL_DIR}/fold_0/checkpoint_final.pth strict=True: {result}, "
              f"{n_params / 1e6:.2f}M params")
        return net, n_params

    # ---- G-LOAD-KEYS -----------------------------------------------------------------
    # Arm E/S add parameters, so strict=True cannot hold. What must still hold is that the
    # ONLY thing the checkpoint fails to supply is the new fusion module: any other missing
    # key would be a silently-partial load, which is exactly the "arm B is a slightly-warmer
    # arm C" failure strict=True was there to prevent.
    result = net.load_state_dict(ckpt["network_weights"], strict=False)
    expected = expected_new(net)
    missing, unexpected = set(result.missing_keys), set(result.unexpected_keys)
    assert not unexpected, f"G-LOAD-KEYS: checkpoint has {len(unexpected)} keys the model " \
                           f"does not, e.g. {sorted(unexpected)[:4]}"
    assert missing == expected, (
        f"G-LOAD-KEYS: missing keys are not exactly the fusion parameters. "
        f"unsupplied non-fusion: {sorted(missing - expected)[:6]} ; "
        f"fusion keys the checkpoint somehow had: {sorted(expected - missing)[:6]}")
    arm = "E fuse-into-trunk" if fuse_into_trunk else "S skip-only"
    print(f"loaded {MODEL_DIR}/fold_0/checkpoint_final.pth strict=False, G-LOAD-KEYS ok: "
          f"{len(missing)} missing keys, all of them new; {n_params / 1e6:.2f}M params "
          f"(arm {arm})")
    return net, n_params


def poly_lr(base: float, step: int, total: int, power: float = 0.9) -> float:
    return base * (1.0 - step / total) ** power


# ------------------------------------------------------------------- checkpoint / resume

CKPT_REQUIRED = ("network_weights", "optimizer_state", "step", "arm", "args")


def save_checkpoint(path: Path, net, opt, step: int, arm: str, args) -> tuple[Path, dict]:
    """THE only place a checkpoint is written, and it always carries the optimizer.

    Weights alone are not a resumable state. SGD here runs momentum 0.99 with Nesterov, so a
    "resume" that drops the momentum buffers restarts with zero velocity on a loss surface the
    old velocity was already descending -- it is a different run wearing the old run's weights.
    v13's arm E ended at 30,000 steps with the schedule annealed to zero and no way to extend
    it, so the only way to ask "would more steps help?" was a fresh 60k run costing twice the
    GPU. That is the cost this function exists to stop repeating.

    Every caller goes through here; `assert_resumable` is what makes that checkable rather
    than a convention someone remembers.
    """
    blob = {
        "network_weights": {k: v.detach().to("cpu") for k, v in net.state_dict().items()},
        "optimizer_state": opt.state_dict(),
        "step": step,
        "arm": arm,
        "args": vars(args),
        # The prompt decoder runs dropout=0.1 in train mode, drawing from the GLOBAL RNG. A
        # resume that only restores weights and momentum re-seeds that stream from scratch,
        # so the continuation trains against a different mask sequence -- measured on a 6-step
        # smoke as a grad-norm difference of up to 31% on the first resumed step. Carrying the
        # RNG makes the continuation the same run rather than a statistically similar one.
        "rng": {"cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    assert_resumable(blob, f"about to write {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(blob, path)
    # AFTER the new one is safely on disk, never before: stripping first would leave a window
    # with no resumable checkpoint at all, which is the one state this whole mechanism exists
    # to prevent.
    strip_superseded(path.parent, keep=path)
    # The weights are handed back rather than re-read: val_sweep wants them immediately and
    # this file is ~4.3 GB, so a round trip through disk costs real minutes per sweep.
    return path, blob["network_weights"]


def _atomic_save(blob: dict, path: Path) -> None:
    """Write beside the target and rename. A 4.3 GB torch.save interrupted in place leaves a
    truncated file that torch.load raises on -- and it would be the file a resume needs."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


def strip_superseded(run_dir: Path, keep: Path) -> list[Path]:
    """Only the newest checkpoint stays resumable; the rest keep their weights and lose the
    optimizer and RNG.

    Optimizer state is ~1.4 GB per checkpoint here and is worth exactly nothing once a newer
    one exists -- you would never resume from the older point. Weights are left untouched, so
    every stripped file still loads for inference and for `val_sweep`, which reads
    `network_weights` and nothing else.

    The marker matters as much as the saving: without it a later `--resume` on an old
    checkpoint can only say "this has no optimizer state", which reads like a bug. With it,
    it can say the state was deliberately freed and point at the newest file.
    """
    freed = []
    for p in sorted(run_dir.glob("checkpoint_step*.pth")):
        if p.resolve() == keep.resolve():
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if not any(k in blob for k in ("optimizer_state", "rng")):
            continue
        blob.pop("optimizer_state", None)
        blob.pop("rng", None)
        blob["optimizer_stripped"] = str(keep.name)
        _atomic_save(blob, p)
        freed.append(p)
    if freed:
        print(f"  freed optimizer state from {len(freed)} superseded checkpoint(s): "
              f"{', '.join(f.name for f in freed)} (still loadable for inference)", flush=True)
    return freed


def assert_resumable(blob: dict, where: str) -> None:
    """G-RESUMABLE. Raises rather than letting a weights-only checkpoint out of the door.

    Checks the SHAPE, not just the presence of a key: `opt.state_dict()` on a never-stepped
    optimizer has `param_groups` but an EMPTY `state`, so a checkpoint written before the
    first optimizer step is legitimately momentum-free -- and one written after step 1 must
    not be. Checking only `"optimizer_state" in blob` would pass an empty dict and the gate
    would be decorative.
    """
    missing = [k for k in CKPT_REQUIRED if k not in blob]
    assert not missing, f"G-RESUMABLE: {where} is missing {missing}; this checkpoint cannot resume"
    o = blob["optimizer_state"]
    assert isinstance(o, dict) and "param_groups" in o, \
        f"G-RESUMABLE: {where} has no optimizer param_groups"
    assert blob["step"] == 0 or o.get("state"), (
        f"G-RESUMABLE: {where} is at step {blob['step']} but the optimizer state is empty. "
        "Momentum buffers exist from the first optimizer step onward, so an empty one here "
        "means the optimizer was not the one that trained this network.")


def load_resume(path: Path, net, opt, device) -> int:
    """Restore weights AND momentum, and return the step to continue from.

    Refuses a weights-only checkpoint loudly instead of silently resuming with zero velocity,
    which would look like a resume and behave like a restart.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer_stripped" in blob:
        raise SystemExit(
            f"--resume {path} was superseded by {blob['optimizer_stripped']} and its optimizer "
            "state was deliberately freed to save disk; only the NEWEST checkpoint of a run is "
            f"resumable. Resume from {path.parent / blob['optimizer_stripped']} instead. This "
            "file still loads fine for inference and evaluation.")
    assert "optimizer_state" in blob, (
        f"--resume {path} is a weights-only checkpoint (keys: {sorted(blob)}). Resuming from "
        "it would silently reset SGD momentum, so it is refused. Train from scratch, or "
        "resume from a checkpoint written after this gate landed.")
    assert_resumable(blob, str(path))
    net.load_state_dict(blob["network_weights"], strict=True)
    opt.load_state_dict(blob["optimizer_state"])
    for st in opt.state.values():                       # momentum buffers follow the model
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)
    rng = blob.get("rng")
    if rng:
        torch.set_rng_state(rng["cpu"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    print(f"resumed {path} at step {blob['step']} (arm {blob.get('arm')}), "
          f"{len(blob['optimizer_state'].get('state', {}))} momentum buffers restored, "
          f"rng {'restored' if rng else 'ABSENT -- dropout will draw a different stream'}")
    return int(blob["step"])


def gate_unwrapped(state: dict) -> dict:
    """THE single place that decides a checkpoint's `network_weights` is loadable strict=True.

    Every caller that loads a checkpoint into a stock or fused net routes through here --
    `val_sweep` below and `scripts/predict_split.py`. The upstream repo this was extracted
    from also trained WRAPPED arms (an auxiliary category head, a retrieval adapter), whose
    checkpoints prefix every trunk key with `net.` and add the wrapper's own parameters.
    Those arms are not in this folder, so such a checkpoint cannot be scored here correctly
    -- and the tempting repair (strip the prefix and load what is left) would silently score
    the STOCK net while the wrapper, which is the entire point of those arms, is dropped.
    Every gate above it would stay green and the number would look ordinary.

    So: refuse, loudly, naming the file. This was fixed three times inline in three separate
    call sites before it got one home.
    """
    wrapped = sorted(k for k in state if k.startswith("net."))
    if wrapped:
        raise SystemExit(
            f"G-WRAPPED-CKPT: {len(wrapped)} of {len(state)} weight keys are `net.`-prefixed, "
            f"e.g. {wrapped[:3]}. This checkpoint comes from a WRAPPED arm (auxiliary head or "
            "retrieval adapter), which this folder does not implement. Stripping the prefix "
            "would score the stock network and silently discard the wrapper -- refusing "
            "instead. Score it with the arm's own trainer.")
    return state


def load_init_from(path: Path, net) -> None:
    """Load ONLY the weights from a checkpoint: no optimizer, no step, no sampler position.

    The counterpart to `load_resume` (which refuses a weights-only checkpoint because IT
    claims continuation): this refuses nothing about `step`/`optimizer_state` being absent
    because it never looks at them. Training starts at step 0 with a fresh SGD state and a
    fresh poly-LR schedule -- only the trunk's weights carry over.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "network_weights" not in blob:
        raise SystemExit(
            f"--init-from {path} has no 'network_weights' key (keys present: {sorted(blob)}). "
            "Nothing to load.")
    state = blob["network_weights"]
    net.load_state_dict(state, strict=True)
    n_loaded = len(state)
    print(f"init-from {path}: checkpoint's own step={blob.get('step')} arm={blob.get('arm')!r} "
          f"-- {n_loaded} weight keys loaded, optimizer and schedule NOT restored "
          "(fresh SGD state, fresh poly-LR, training starts at step 0)")


# ------------------------------------------------------------------------ val sweep

def gate_bank_covers(bank_keys, prompts) -> int:
    """G-BANK-COVERS. Every prompt the sweep will ask for must already be in the bank.

    `VoxTellPredictor.embed_text_prompts` looks a prompt up in the bank and, ON A MISS,
    SILENTLY falls through to `_compute_text_embeddings` -- which hardcodes
    `wrap_with_instruction` + `last_token_pool` + `padding_side='left'`, i.e. the Qwen3
    convention, whatever `text_encoding_model` happens to be. No exception, no log line.

    Today that is nearly harmless -- the fallback's encoder IS the bank's encoder, so a miss
    costs GPU time and a fp16 round-trip but yields the right vector. It stops being harmless
    the moment the bank comes from a different encoder: a Qwen-shaped vector lands in a
    GatorTron bank (if the model string is still Qwen), or a wrongly-pooled GatorTron vector
    does (if it is not). Either way the sweep produces a plausible dice number computed
    against text the model was never conditioned on. So this gate is a prerequisite for any
    encoder-swap arm, and meanwhile it turns a silent slow path into a loud one.

    Shown to fire on the real case, not only synthetic: the test split's 542 prompts against
    the unextended text_emb_qwen3.npz miss 459 -- the 2026-08-11 pairing incident's shape --
    while train (6148) and val (365) pass, so it is not merely rejecting everything.

    Same shape as the known `eval_worker.py:83` hole -- a consumer that loads a bank without
    gating it. Prompts are lowercased here because `embed_text_prompts` lowercases before
    the lookup, and the bank is keyed on `s.lower()` by `build_text_bank`.

    Returns the number of prompts checked, so a caller cannot mistake a no-op for a pass.
    """
    have = set(bank_keys)
    wanted = {str(p).lower() for p in prompts}
    if not wanted:
        raise SystemExit("G-BANK-COVERS: no prompts to check -- the sweep would score nothing")
    missing = sorted(wanted - have)
    if missing:
        raise SystemExit(
            f"G-BANK-COVERS: {len(missing)} of {len(wanted)} sweep prompts are not in the "
            f"bank ({len(have)} keys). The predictor would silently re-encode them with the "
            f"Qwen3 wrapper/pooling and score against the wrong vectors. First 3: "
            + " | ".join(repr(m) for m in missing[:3]))
    return len(wanted)


def val_sweep(state_dict, bank_path: Path, cfg, out_dir: Path, tag: str, device: str,
              limit: int, workers: int, heartbeat=None, fuse_into_trunk: bool = None,
              tok_bank=None, hu_norm=None) -> dict:
    """The OFFICIAL number: native-grid predictions -> rexrank_eval.py.

    Every step of this is the published inference path or the repo's existing evaluator
    wiring; nothing here re-implements a metric or a transform.
      * `VoxTellPredictor` owns preprocessing (crop_to_nonzero + ZScoreNormalization),
        the 0.5-overlap Gaussian sliding window and sigmoid > 0.5. Only the WEIGHTS change.
      * the text bank is ours, so the Qwen3 backbone never loads during a sweep;
      * `scorer.postprocess` is rexrank_eval's own cc_filter + top-K cap;
      * `scorer.ras_to_lps` is the flip back to the GT's index order;
      * `rexrank_eval.py` itself produces the number that gets compared to 0.1538.
    """
    import nibabel as nib
    import scorer as P
    from voxtell.inference.predictor import VoxTellPredictor

    paired = Path(cfg.paired_root)
    gt_dir = Path(cfg.seg_root)
    npy_root = Path(cfg.npy_root)
    findings = pd.read_csv(paired / "findings.csv", dtype={"VolumeName": str})
    val = findings[(findings["miccai_split"] == "val") & (findings["has_seg"] == 1)]
    names = sorted(val["VolumeName"].unique())
    partial = bool(limit) and limit < len(names)
    if partial:
        names = names[:limit]
        val = val[val["VolumeName"].isin(set(names))]
        print(f"*** PARTIAL SWEEP: {len(names)} of 200 volumes -- NOT the official number ***")
    index = pd.read_csv(npy_root / "index.csv", dtype={"VolumeName": str}).set_index("VolumeName")

    sweep_dir = out_dir / f"eval_{tag}"
    pred_dir = sweep_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ev = P._load_evaluator(P.EVALUATOR)

    predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=torch.device(device),
                                 text_encoding_model=QWEN, embedding_bank=str(bank_path),
                                 use_precomputed_embeddings=False)
    if hu_norm is not None:
        # WITHOUT this the sweep would normalise with the predictor's own per-volume z-score
        # while the weights were fitted on the fixed affine -- the run's own 30k kill criterion
        # would then be measured under a preprocessing the model never saw.
        predictor.normalization = FixedAffineNormalization(hu_norm)
        print(f"val_sweep: hu_norm fixed affine {hu_norm}", flush=True)
    # ONE call, before the branch: both branches load the same dict, and putting the check
    # inside each of them is how the fused branch went a whole refactor without one.
    state_dict = gate_unwrapped(state_dict)
    if fuse_into_trunk is None:
        predictor.network.load_state_dict(state_dict, strict=True)
    else:
        # An arm E/S checkpoint does not fit the predictor's stock network, so swap in the fused
        # one. random_init=True skips the v1.0 load that the run's own state_dict replaces
        # anyway; the load stays strict=True because such a checkpoint MUST supply every key
        # including the fusion module.
        assert tok_bank is not None, "fused sweep without a token bank"
        net, _ = build_network(random_init=True, deep_supervision=False,
                               fuse_into_trunk=fuse_into_trunk)
        net.load_state_dict(state_dict, strict=True)
        predictor.network = net.to(torch.device(device))
    predictor.network.eval()

    # Before a single volume runs: prove the bank covers every sentence this sweep will ask
    # for, so no prompt can reach the predictor's silent Qwen3 re-encode path.
    n_checked = gate_bank_covers(predictor.embedding_bank, val["text"])
    # Resolved from the bank the sweep is actually reading, so the pairing gate below
    # reconstructs the pooled vector the way this bank was built rather than by default.
    sweep_enc = encoder_for(bank_path)
    print(f"G-BANK-COVERS ok: {n_checked} distinct sweep prompts all present in "
          f"{bank_path.name} [{sweep_enc.key}, pooling {sweep_enc.pooling}] "
          f"({len(predictor.embedding_bank)} keys)", flush=True)

    t0 = 0.0
    started = time.time()
    for i, name in enumerate(names, 1):
        rows = val[val["VolumeName"] == name].sort_values("seg_channel")
        gt_shape = nib.load(str(gt_dir / name)).shape
        if sorted(rows["seg_channel"]) != list(range(gt_shape[0])):
            raise SystemExit(f"{name}: findings.csv channels {sorted(rows['seg_channel'])} "
                             f"vs GT {gt_shape[0]} -- the prediction would be misfiled")
        ct = np.load(npy_root / "ct" / f"{name}.npy")
        prompts = [str(t) for t in rows["text"]]
        if fuse_into_trunk is not None:
            arr, tok_index, lengths = tok_bank   # NOT `index`: that is index.csv, below
            r = [tok_index[p.lower()] for p in prompts]
            predictor.network.set_tokens(
                torch.from_numpy(np.stack([np.asarray(arr[k]) for k in r]))[None].to(device),
                # arr.shape[1], not a constant -- see the same note in the dataset's collate.
                torch.stack([torch.arange(arr.shape[1]) < int(lengths[k])
                             for k in r])[None].to(device),
                pooling=sweep_enc.pooling)
        try:
            seg = predictor.predict_single_image(np.ascontiguousarray(ct.transpose(2, 1, 0)),
                                                 prompts)
        finally:
            # Always, even on an exception: a stale token set would silently score the NEXT
            # volume against this volume's sentences, and no metric would show it.
            if fuse_into_trunk is not None:
                predictor.network.clear_tokens()
        seg = seg.transpose(0, 3, 2, 1)                       # back to RAS (nx, ny, nz)
        pos = str(index.loc[name, "patient_position"])
        arr = np.ascontiguousarray(np.stack(
            [P.ras_to_lps(P.postprocess(seg[k] > 0, ev), pos) for k in range(len(rows))]
        )).astype(np.uint8)
        if arr.shape != gt_shape:
            raise SystemExit(f"{name}: prediction {arr.shape} != GT {gt_shape}")
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(pred_dir / name))
        t0 = time.time() - started
        print(f"  [{tag} {i}/{len(names)}] {name} fg={[int(c.sum()) for c in arr]} "
              f"{t0 / i:.1f}s/vol eta {t0 / i * (len(names) - i) / 60:.1f}min", flush=True)
        if heartbeat is not None:
            heartbeat(i, len(names))

    # Same entry point reproduce/submit.sh uses, so a submission cannot be scored under a
    # different invocation than the sweep that chose its checkpoint.
    summary = P.score(pred_dir, val, gt_dir, sweep_dir, workers=workers,
                      expect=None if partial else (P.MICCAI_VAL_CASES, P.MICCAI_VAL_FINDINGS))
    dice = float(summary[OFFICIAL_SCORE_KEY])
    print(f"\n[{tag}] official {OFFICIAL_SCORE_KEY} = {dice:.4f}  "
          f"(bar {BAR}; {'CLEARS' if dice > BAR else 'does NOT clear'} it)"
          f"{'  *** PARTIAL, not the official set ***' if partial else ''}", flush=True)
    return {"tag": tag, "dice": dice, "partial": partial, "minutes": t0 / 60,
            "summary": summary}


# ------------------------------------------------------------------------- self-test

def self_test() -> int:
    """Every gate and every piece of geometry, on synthetic data. No GPU, no dataset."""
    assert crop_start(300, 150.0, 192) == 54 and crop_start(300, 0.0, 192) == 0
    assert crop_start(300, 999.0, 192) == 108 and crop_start(104, 50.0, 192) == 0

    small = np.ones((10, 104, 192), np.float32)
    assert pad_to_patch(small).shape == PATCH
    assert pad_to_patch(np.ones((2, 10, 104, 192), np.float32)).shape == (2, *PATCH)

    inst = np.array([[0, 2], [3, 0]], np.uint8)
    assert set(np.unique(binarise(inst)).tolist()) == {0.0, 1.0}
    for bad in (np.array([0.0, 0.3], np.float32), np.array([-1, 0, 3], np.int16)):
        try:
            binarise(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"G-BINARY did not fire on {bad.dtype} {bad.tolist()}")
    try:
        assert_binary(torch.tensor([[0.0, 3.0]]), "self-test")
    except AssertionError:
        pass
    else:
        raise AssertionError("G-BINARY did not fire on a target holding 3")

    hu = np.array([-1000, 0, 1000], np.int16)
    z = normalise(hu, 10.0, 4.0)
    assert np.allclose(z, (hu - 10.0) / 4.0), z
    assert np.isfinite(normalise(hu, 0.0, 0.0)).all()

    logits = torch.full((2, 2, 4, 4, 4), -8.0)
    target = torch.zeros_like(logits)
    empty = float(dice_bce(logits, target))
    assert empty > 0, empty
    target[0, 0] = 1.0
    assert float(dice_bce(logits, target)) > empty

    gt = torch.zeros(1, 2, 8, 8, 8)
    gt[0, 0, 3, 3, 3] = 1.0
    heads = ds_targets(gt, [(8, 8, 8), (4, 4, 4), (2, 2, 2)])
    assert [tuple(h.shape[2:]) for h in heads] == [(8, 8, 8), (4, 4, 4), (2, 2, 2)]
    assert all(float(h.sum()) == 1.0 for h in heads), "max-pool lost the 1-voxel finding"
    assert abs(sum(ds_weights(5)) - 1.0) < 1e-9

    # G-HU-NORM: inference must refuse a scheme the checkpoint was not trained under.
    assert hu_norm_spec("zscore") is None and hu_norm_spec("ct") == HU_NORM_CT
    for trained, asked in (("zscore", "ct"), ("ct", "zscore"), ("ct", None)):
        try:
            gate_hu_norm({"args": {"hu_norm": trained}}, asked, "self-test")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"G-HU-NORM did not fire on trained={trained} asked={asked}")
    assert gate_hu_norm({"args": {"hu_norm": "ct"}}, "ct", "self-test") == HU_NORM_CT

    # G-WRAPPED-CKPT: a `net.`-prefixed checkpoint is refused, not silently unwrapped.
    assert gate_unwrapped({"encoder.w": 1}) == {"encoder.w": 1}
    try:
        gate_unwrapped({"net.encoder.w": 1, "cls_head.weight": 2})
    except SystemExit:
        pass
    else:
        raise AssertionError("G-WRAPPED-CKPT did not fire on a `net.`-prefixed state dict")

    print("self-test ok")
    return 0


# ------------------------------------------------------------------------------ main

# ---- G-DEAD-CONFIG ---------------------------------------------------------------
# The six keys this script does not read. Listed, not derived: "derive what the code reads"
# is precisely the check that cannot see a key nobody reads.
DEAD_CONFIG_KEYS = ("spatial_size", "target_spacing", "cube_size", "iso_spacing", "hu_window",
                    "cache2mm_root")


def gate_dead_config(cfg, config_path) -> None:
    """Reject a config that describes preprocessing this script does not do.

    The failure is not hypothetical: B60 and E60 -- the two best arms -- each ran 60,000
    steps naming a config whose preprocess block says 2 mm isotropic resample into a 192^3
    (384 mm) cube. This script reads `paired_root` and `npy_root` from that file and NOTHING
    else: `VoxTellPatches` opens the native-grid npy and cuts 192^3 in VOXELS, and
    `VoxTellPredictor.preprocess` only crops to nonzero and z-scores. Both the YAML and the
    run.json therefore recorded a resample that never happened, and reading the config back
    is how two sessions concluded the input scale was already normalised. It is not: the
    192^3 patch spans 58-188 mm in plane and 96-576 mm in z depending on the volume.

    A written record cannot fix this, because the record is not what the next session reads --
    the config is. So the config has to stop being loadable while it lies.
    """
    lying = [k for k in DEAD_CONFIG_KEYS if getattr(cfg, k) is not None]
    if lying:
        raise SystemExit(
            f"{config_path}: sets {lying}, and this script reads NONE of them.\n"
            f"VoxTell trains and predicts on the NATIVE grid -- VoxTellPatches cuts "
            f"PATCH={PATCH} in voxels off npy_root/ct, and plans.json has spacing=None. A "
            f"config naming a spacing or a cube here records preprocessing that will not "
            f"happen -- B60 and E60 each ran 60k steps under a config claiming 2 mm "
            f"isotropic resampling that never occurred.\n"
            f"Use configs/rexg_native.yaml. If this arm is genuinely meant to resample, "
            f"the resample has to be implemented in VoxTellPatches first -- setting the key "
            f"does not do it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/rexg_native.yaml"))
    ap.add_argument("--out-dir", default=None,
                    help="run directory: checkpoints, run.json, train_log.jsonl, eval_step*/. "
                         "Required to train; the --self-test and --build-*-bank-only modes "
                         "write nothing there and do not need it.")
    ap.add_argument("--text-bank", default=None,
                    help=f"default <paired_root>/{TEXT_BANK_NAME}")
    ap.add_argument("--build-text-bank-only", action="store_true")
    ap.add_argument("--bank-splits", default=None,
                    help="which miccai_splits' sentences go in the bank, e.g. train,val,test. "
                         "Default (None) keeps the training behaviour: every finding with a "
                         "mask. The test split has no masks, so a submission needs this.")
    ap.add_argument("--build-token-bank-only", action="store_true",
                    help="v13: per-token bank next to the pooled one. Reads the same sentences.")
    ap.add_argument("--random-init", action="store_true",
                    help="ARM C: skip the pretrained load. The only difference from arm B.")
    ap.add_argument("--hu-norm", choices=("zscore", "ct"), default="zscore",
                    help="intensity normalisation. 'zscore' is per-volume statistics on RAW HU "
                         "-- what B60 and E60 trained under. 'ct' is nnU-Net's CT scheme: clip "
                         "to [-1000,1000] then a FIXED affine from train-split statistics, so "
                         "the same tissue lands at the same value in every patient. CHANGES "
                         "THE NUMBERS; the checkpoint records it and inference refuses to "
                         "disagree.")
    ap.add_argument("--fuse", choices=("off", "trunk", "skip"), default="off",
                    help="off = arms B/C, the stock model. trunk = arm E (E60), the fused "
                         "text feature re-enters the encoder. skip = arm S, it goes to the "
                         "decoder only. E and S differ by that and nothing else.")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--val-at", default="30000,60000",
                    help="steps at which the 32-minute official sweep runs")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="volumes per sweep; 0 = the official 200. Anything else is labelled "
                         "PARTIAL and is not a comparable number")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--queries", type=int, default=QUERIES)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--clip", type=float, default=12.0, help="nnU-Net's grad-norm clip")
    ap.add_argument("--fg-prob", type=float, default=0.66)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint written by this trainer: weights, SGD "
                         "momentum, LR schedule position and sampler position all carry over. "
                         "Refuses a weights-only checkpoint rather than silently restarting "
                         "with zero velocity.")
    ap.add_argument("--init-from", default=None,
                    help="load ONLY network_weights from a checkpoint -- no optimizer state, "
                         "no step, no sampler/schedule position. Training starts fresh at "
                         "step 0 with a new optimizer and a new poly-LR schedule. This is what "
                         "a strip_superseded'd checkpoint (weights + step only) needs, and it "
                         "is mutually exclusive with --resume, which means the opposite: full "
                         "continuation.")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also write a checkpoint every N steps, without an eval sweep. "
                         "0 = only at --val-at and the final step (the old behaviour).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.init_from and args.resume:
        raise SystemExit(
            "--init-from and --resume mean different things: --init-from loads WEIGHTS ONLY "
            "and starts a fresh optimizer and schedule at step 0; --resume continues "
            "everything (optimizer momentum, step, sampler position). Combining them would "
            "silently restart the schedule while claiming continuity. Pass exactly one.")
    # Every gate below is an `assert`, and `python -O` deletes asserts. "None skippable by a
    # flag" is only true if this flag is refused.
    if not __debug__:
        raise SystemExit("refusing to run under python -O: -O deletes every assert, which is "
                         "every gate in this script (G-BINARY, G-SHAPE, G-TEXT, "
                         "G-LOSS-POSITIVE). Drop -O.")
    if args.self_test:
        return self_test()

    from rexg_data.config import load_data_config

    cfg = load_data_config(args.config)
    gate_dead_config(cfg, args.config)
    paired_root, npy_root = Path(cfg.paired_root), Path(cfg.npy_root)
    bank_path = Path(args.text_bank) if args.text_bank else paired_root / TEXT_BANK_NAME
    # The bank builders return before any of this; only a real run needs somewhere to write.
    building = args.build_text_bank_only or args.build_token_bank_only
    if not building and not args.out_dir:
        raise SystemExit("--out-dir is required to train (it holds the checkpoints, run.json "
                         "and train_log.jsonl this run is the only copy of).")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- G-INDEX -----------------------------------------------------------------
    train_volumes = build_index(paired_root, npy_root, "train", drop_leak=True)
    all_findings = pd.read_csv(paired_root / "findings.csv", dtype={"VolumeName": str})
    needed = sorted({str(t) for t in all_findings.loc[
        all_findings["has_seg"] == 1, "text"]})
    if args.bank_splits:
        # A submission needs embeddings for prompts that have NO mask -- the test split is
        # exactly that, so has_seg == 1 excludes every sentence we are about to be scored on.
        keep = {x.strip() for x in args.bank_splits.split(",") if x.strip()}
        unknown = keep - set(all_findings["miccai_split"].unique())
        assert not unknown, f"--bank-splits has unknown splits {unknown}"
        needed = sorted({str(t) for t in all_findings.loc[
            all_findings["miccai_split"].isin(keep), "text"]})
        print(f"bank covers splits {sorted(keep)}: {len(needed)} unique sentences")

    if args.build_text_bank_only:
        enc = encoder_for_new_bank(bank_path)
        print(f"building a {enc.key} bank: {enc.hf_path} | pooling {enc.pooling} | "
              f"padding {enc.padding_side} | wrapper {enc.wrap} -> {bank_path}", flush=True)
        build_text_bank(needed, bank_path, args.device, enc=enc)
        return 0
    if args.build_token_bank_only:
        # Read the sentence set FROM the pooled bank on disk rather than recomputing it from
        # findings.csv + --bank-splits: recomputing has to reproduce the exact same selection
        # logic that built bank_path, and drifting from it even slightly is how this broke
        # (2026-08-11 -- the pooled bank was extended to 6926 sentences for the test split, the
        # token bank stayed at its original 6467, and gate_token_bank correctly refused rather
        # than pairing the wrong text against a submission). Reading bank_path's own labels
        # makes that drift structurally impossible: the token bank always covers exactly what
        # the pooled bank covers, whatever criteria produced it.
        assert bank_path.is_file(), (
            f"--build-token-bank-only reads its sentence set FROM {bank_path} -- build the "
            "pooled bank there first with --build-text-bank-only.")
        pooled, _, _ = load_text_bank(bank_path)
        # NOT sorted(pooled): load_text_bank's keys are lowercased, and the pooled bank was
        # itself built from sorted(ORIGINAL-case sentences) -- ASCII-sorting the lowercased
        # keys gives a DIFFERENT order (case-sensitive sort moves capitals before lowercase),
        # which changes batch composition and, with it, which sentences share a batch's
        # padding. list(pooled) instead: dict iteration preserves the npz's stored order,
        # which IS build_text_bank's batching order, because load_text_bank builds the dict
        # by enumerating `labels` in file order. Caught by the smoke: sorted() reproduced the
        # 41% mismatch down to 1% (72/6926) rather than 0 -- still wrong, just less wrong.
        needed = list(pooled)
        # The encoder comes from the POOLED bank, not from the filename: the two banks must
        # be the same encoding, and reading it off the file that already exists is the only
        # way that cannot be contradicted.
        enc = encoder_for(bank_path)
        print(f"token bank sourced from {bank_path.name}: {len(needed)} sentences, "
              f"in the bank's own order (guaranteed to match, not recomputed); "
              f"encoder {enc.key} read from the pooled bank")
        build_token_bank(needed, token_bank_path_for(bank_path), args.device, enc=enc)
        return 0
    if not bank_path.is_file():
        raise SystemExit(f"no text bank at {bank_path}. Run --build-text-bank-only first "
                         "(reproduce/text_banks.sh). Do NOT point --text-bank at a BioLORD "
                         "bank: those are 768-d and L2-normalised, and G-TEXT will refuse it.")

    # ---- G-TEXT ------------------------------------------------------------------
    # The conventions come from the bank itself, so they cannot contradict the file the run
    # is actually about to read.
    enc = encoder_for(bank_path)
    print(f"text encoder: {enc.key} ({enc.hf_path}) | pooling {enc.pooling} | "
          f"padding {enc.padding_side} | wrapper {enc.wrap} | norm band "
          f"({enc.norm_lo}, {enc.norm_hi})", flush=True)
    bank, wrapped_example, example_key = load_text_bank(bank_path)
    text_stats = gate_text(bank, wrapped_example, example_key,
                           [f["text"] for v in train_volumes for f in v["findings"]], enc=enc)

    # ---- G-TEXT-TOK, G-BANK-CONSISTENT (v13 only) ---------------------------------
    fuse_into_trunk = {"off": None, "trunk": True, "skip": False}[args.fuse]
    tok_bank, tok_stats = None, None
    if fuse_into_trunk is not None:
        arr, index, lengths, meta = load_token_bank(token_bank_path_for(bank_path))
        tok_stats = gate_token_bank(arr, index, lengths, meta, bank,
                                    [f["text"] for v in train_volumes for f in v["findings"]],
                                    enc=enc)
        tok_bank = (arr, index, lengths)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net, n_params = build_network(args.random_init, deep_supervision=True,
                                  fuse_into_trunk=fuse_into_trunk)
    # Resolved ONCE and handed to both the patches and the sweep. A second `hu_norm_spec` call
    # somewhere downstream is exactly how the two drift apart.
    hu_norm = hu_norm_spec(args.hu_norm)
    print(f"hu_norm: {args.hu_norm} -> {hu_norm}", flush=True)

    device = torch.device(args.device)
    net = net.to(device).train()
    # Only the parameters that are actually open. Handing SGD the frozen ones would build
    # momentum buffers for 440M tensors that never move -- and, worse, would make a later
    # `requires_grad_(True)` regression silently start training the encoder.
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum, nesterov=True)

    # ---- resume ---------------------------------------------------------------------
    # The sampler is indexed, not stateful: `__getitem__(i)` seeds its own rng from
    # (seed, i), so sample i is the same sample whether it is drawn in the first run or the
    # continuation. Skipping the consumed indices is therefore an exact continuation of the
    # data order, not an approximation -- and the poly LR is a pure function of `step`, so it
    # continues too. Those two facts are what make a resume here equal to an uninterrupted run.
    start_step = 0
    if args.init_from:
        # Weights only. Deliberately does not touch start_step, opt, or the sampler: this
        # arm needs a FRESH optimizer and an identical schedule from step 0, not a resume.
        load_init_from(Path(args.init_from), net)
    if args.resume:
        start_step = load_resume(Path(args.resume), net, opt, device)
        assert start_step < args.steps, (
            f"--resume is at step {start_step} but --steps is {args.steps}; there is nothing "
            "left to run. Raise --steps to extend the schedule.")
        print(f"continuing to {args.steps}: {args.steps - start_step} steps remain, "
              f"LR restarts from poly({start_step}) = {poly_lr(args.lr, start_step, args.steps):.3e}")

    dataset = VoxTellPatches(train_volumes, npy_root, bank,
                             length=args.steps * args.batch_size, seed=args.seed,
                             fg_prob=args.fg_prob, queries=args.queries, tok_bank=tok_bank,
                             start_index=start_step * args.batch_size, hu_norm=hu_norm)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)

    arm = {"trunk": "E (fusion into the encoder trunk)",
           "skip": "S (fusion into the skips only)"}.get(
        args.fuse, "C (random-init CONTROL)" if args.random_init else "B (v1.0 pretrained)")
    val_at = parse_val_at(args.val_at)
    run = {"arm": arm, "bar": BAR, "args": vars(args), "n_params": n_params,
           "n_train_volumes": len(train_volumes),
           "n_train_findings": sum(len(v["findings"]) for v in train_volumes),
           "text": text_stats, "text_bank": str(bank_path), "val_at": val_at,
           "fuse": args.fuse, "token_bank": tok_stats,
           "label": f"BOUNDED {args.steps}-step probe, arm {arm}"
                    f"{'; PARTIAL sweeps' if args.val_limit else ''}",
           "git_sha": _git_sha(), "sweeps": []}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    events = (out_dir / "train_log.jsonl").open("a", buffering=1)
    print(f"\narm {arm}: {args.steps} steps, batch {args.batch_size} x {args.queries} "
          f"queries, SGD lr {args.lr} momentum {args.momentum} nesterov, poly^0.9, "
          f"sweeps at {val_at}\n", flush=True)

    started, losses = time.time(), []
    for step, batch in enumerate(loader, start=start_step + 1):
        lr = poly_lr(args.lr, step - 1, args.steps)
        for group in opt.param_groups:
            group["lr"] = lr

        image = batch["image"].to(device, non_blocking=True)
        text = batch["text"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        # ---- G-SHAPE, G-BINARY: every step, on the tensors that reach the model
        assert tuple(image.shape[-3:]) == PATCH, (
            f"G-SHAPE: image patch {tuple(image.shape[-3:])} != {PATCH}; VoxTell's pos_embed "
            f"is a fixed (1728, 1, 2048) buffer and cannot absorb another size")
        assert tuple(gt.shape[-3:]) == PATCH, f"G-SHAPE: target {tuple(gt.shape[-3:])} != {PATCH}"
        assert_binary(gt, f"step {step} {batch['name']}")

        if fuse_into_trunk is not None:
            net.set_tokens(batch["tokens"].to(device, non_blocking=True),
                           batch["tok_mask"].to(device, non_blocking=True),
                           pooling=enc.pooling)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = net(image, text)
        if fuse_into_trunk is not None:
            net.clear_tokens()
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        targets = ds_targets(gt, [tuple(o.shape[2:]) for o in outs])
        weights = ds_weights(len(outs))
        loss = sum(w * dice_bce(o, t) for w, o, t in zip(weights, outs, targets))

        # ---- G-LOSS-POSITIVE: every step
        assert torch.isfinite(loss) and float(loss.detach()) > 0, (
            f"G-LOSS-POSITIVE: loss = {float(loss)} at step {step}. Soft Dice is in [0, 1] "
            "and BCE on a BINARISED target is > 0, so this means the target is not binary "
            "(instance labels 2..5 make BCE unbounded below: measured -1.045 and falling) "
            "or the forward produced non-finite logits.")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - started
            # `step - start_step`, not `step`: on a resume the clock starts at relaunch while
            # the counter starts at the checkpoint, so dividing by the absolute step
            # under-reports the rate by step/(step-start_step). Measured on the 2026-08-13
            # resume-from-10000: it printed 0.12 s/step against a wall-clock 0.600, exactly
            # the 4.92x the ratio predicts, and the ETA it drove was 34 min against a real
            # 173. Display only -- it never touched the training maths -- but an ETA that is
            # 5x optimistic is what someone plans a night around.
            record = {"step": step, "loss": float(np.mean(losses[-args.log_every:])),
                      "lr": lr, "grad_norm": float(grad_norm),
                      "s_per_step": elapsed / max(1, step - start_step),
                      "peak_gib": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                                   if device.type == "cuda" else 0.0)}
            events.write(json.dumps(record) + "\n")
            print(f"[{step}/{args.steps}] loss {record['loss']:.4f} lr {lr:.3e} "
                  f"gn {record['grad_norm']:.2f} {record['s_per_step']:.2f}s/step "
                  f"peak {record['peak_gib']:.1f}GiB "
                  f"eta {(args.steps - step) * record['s_per_step'] / 60:.0f}min", flush=True)

        # Saving is deliberately decoupled from sweeping. With sweeps far apart -- v12's are
        # 15,000 steps or ~4 h of GPU time apart -- a crash between them costs everything since
        # the last one, and this trainer has no resume. `--ckpt-every` buys a floor on that loss
        # for the price of a file; it runs no evaluation and reads state_dict only, so it cannot
        # change training.
        periodic = args.ckpt_every and step % args.ckpt_every == 0
        if step in val_at or step == args.steps or periodic:
            ckpt, state = save_checkpoint(out_dir / f"checkpoint_step{step}.pth",
                                          net, opt, step, arm, args)
            print(f"\nwrote {ckpt}", flush=True)
            if step in val_at:
                net.eval()
                result = val_sweep(state, bank_path, cfg, out_dir, f"step{step}",
                                   args.device, args.val_limit, args.eval_workers,
                                   fuse_into_trunk=fuse_into_trunk, tok_bank=tok_bank,
                                   hu_norm=hu_norm)
                net.train()
                run["sweeps"].append(result)
                events.write(json.dumps({"step": step, **{k: result[k] for k in
                                                          ("tag", "dice", "partial")}}) + "\n")
                (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str),
                                                  encoding="utf-8")
        if step >= args.steps:
            break

    run["wall_minutes"] = (time.time() - started) / 60
    run["final_train_loss"] = float(np.mean(losses[-100:]))
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    events.close()
    best = max((s["dice"] for s in run["sweeps"]), default=float("nan"))
    # a PARTIAL sweep is not comparable to 0.1538; the CLEARS/does-NOT-clear verdict on the
    # last line is the one a reader quotes, so it carries the label too.
    verdict = ("PARTIAL SWEEP, NOT COMPARABLE to" if any(s["partial"] for s in run["sweeps"])
               else ("CLEARS" if best > BAR else "does NOT clear"))
    print(f"\n{run['label']}\n  official dice: "
          f"{[(s['tag'], round(s['dice'], 4)) for s in run['sweeps']]}\n"
          f"  best {best:.4f} vs bar {BAR} -> {verdict} the bar\n"
          f"  {run['wall_minutes']:.0f} min wall, artifacts in {out_dir}")
    return 0


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
