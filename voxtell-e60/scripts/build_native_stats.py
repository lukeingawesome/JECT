#!/usr/bin/env python
"""Per-volume RAW-HU statistics and per-finding foreground geometry, on the NATIVE grid.

    python scripts/build_native_stats.py [--workers 12] [--limit N] [--out PATH]
    python scripts/build_native_stats.py --self-test        # no data needed, ~0.1 s

Writes `<npy_root>/stats.csv`, ONE ROW PER FINDING (join key `VolumeName` + `seg_channel`,
the same key `findings.csv` and the 2 mm cache use). The volume-level columns repeat across
a volume's findings; that duplication is the whole point -- a loader that samples findings
gets both halves in one lookup.

    hu_mean, hu_std      volume-level, over EVERY voxel of the RAW HU array (air included)
    bb_lo{0,1,2}         inclusive foreground bounding box of THIS finding's mask
    bb_hi{0,1,2}
    cen{0,1,2}           foreground centroid (mean index), float
    fg_voxels            foreground voxel count
    n{x,y,z}             volume shape

Axes are the npy cache's RAS index order (`build_npy_cache.py`), not the on-disk NIfTI
order.  Indices are voxels on the native grid, so a consumer working at another spacing
scales them itself.

WHY this file exists, measured (2026-08-06/07):

- Without the precompute the VoxTell loader runs at 1.339 s/step against a 0.423 s/step
  compute budget -- 3.2x data-bound, a 22-hour run found at hour 3.
- Volume statistics are not an optimisation, they are the correct number. Patch-level
  z-score is NOT volume-level z-score: measured median skew 0.49 sigma/voxel, sigma ratio
  0.20-1.09. VoxTell's published inference runs nnU-Net `ZScoreNormalization` on RAW HU with
  VOLUME statistics, so that is what this computes.

DELETED FROM THE RECIPE ON PURPOSE: `crop_to_nonzero`. Measured, it returned the FULL volume
on 5/5 CTs -- HU air is -1000, not 0, so "nonzero" selects everything -- while costing
0.929 s/sample of `binary_fill_holes`. There is no crop column here and there should not be.

SOURCE OF THE ARRAYS. This reads the npy cache (`build_npy_cache.py`: int16 HU, RAS,
uncompressed, 0.08 s/volume) rather than re-decoding 3,192 gzipped NIfTI at ~2.5 s each
(2.2 h). The cache was written through `HuRescaled`, which is `ct_volume.load_ct_hu`'s
no-header-rescale branch -- the CT-RATE slope/intercept join, without which a raw nibabel
read gives wrong HU. That equivalence is not asserted in prose: GATE HU below re-reads N
volumes from the original NIfTI through `load_ct_hu` and requires the mean/std to match.
Reorientation is a flip/permute, and mean/std are invariant to both, so the gate is exact.

GATES (all fail loud; none of them is skippable by a flag):

  G-INDEX   the volume list comes from `findings.csv` filtered on [`miccai_split`,
            `has_seg`], NEVER from globbing the cache. Globbing yields 3,292 CTs; 100 of
            them are `miccai_split == "test"` with NO mask, and a per-finding stats file
            built over them would carry 100 volumes that cannot have a finding row.
  G-PIXELS  every row's measured `fg_voxels` must equal `findings.csv`'s `pixels`. This is
            what proves the mask row index really is `seg_channel` and that the bbox belongs
            to the finding it is filed under.
  GATE HU   `--verify N` volumes re-read from NIfTI through `load_ct_hu` (default 3).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS/OMP thread per worker: the pass is I/O + a numpy reduction, and a shared box may
# be training. Set before numpy is imported or it has no effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# int16 clipping in the npy cache: 16 of the 3,192 selected volumes have 1..78 voxels pinned
# at 32767 (build_npy_cache's `hu_clipped`). Their hu_mean/hu_std are therefore off by
# <1e-4 HU. They are excluded from GATE HU's sample so the gate can demand exact agreement
# instead of a tolerance that would also hide a real bug.
CLIP_COLUMN = "hu_clipped"

COLUMNS = ["VolumeName", "seg_channel", "finding_idx", "nx", "ny", "nz",
           "hu_mean", "hu_std", "fg_voxels",
           "bb_lo0", "bb_hi0", "bb_lo1", "bb_hi1", "bb_lo2", "bb_hi2",
           "cen0", "cen1", "cen2"]


def volume_hu_stats(ct: np.ndarray, slab: int = 64) -> tuple[float, float]:
    """(mean, std) over every voxel, accumulated EXACTLY.

    The values are int16, so sum and sum-of-squares are integers: for the largest volume
    here (512*512*~400 = 1.0e8 voxels, |HU| <= 32768) the sums bound at 3.4e12 and 1.1e17,
    both inside int64. So `E[x^2] - E[x]^2` is computed from exact integers and the usual
    catastrophic-cancellation objection to the one-pass form does not apply. Slabbed only to
    keep the int64 temporary at ~130 MB instead of one 8x copy of the volume.
    """
    total = square = 0
    for i in range(0, ct.shape[0], slab):
        x = ct[i:i + slab].astype(np.int64)
        total += int(x.sum())
        square += int((x * x).sum())
    n = ct.size
    mean = total / n
    return mean, float(np.sqrt(max(square / n - mean * mean, 0.0)))


def mask_geometry(mask: np.ndarray) -> tuple[int, list[int], list[int], list[float]]:
    """(fg_voxels, bbox_lo inclusive, bbox_hi inclusive, centroid) of one finding's mask."""
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        return 0, [-1] * 3, [-1] * 3, [float("nan")] * 3
    return (int(len(idx[0])),
            [int(c.min()) for c in idx],
            [int(c.max()) for c in idx],
            [float(c.mean()) for c in idx])


def process_volume(job: dict) -> dict:
    """One volume: HU stats once, geometry per requested seg_channel."""
    name, ct_dir, seg_dir = job["name"], Path(job["ct_dir"]), Path(job["seg_dir"])
    ct = np.load(ct_dir / f"{name}.npy")
    mean, std = volume_hu_stats(ct)
    with np.load(seg_dir / f"{name}.npz") as z:
        masks = z["mask"]
    rows = []
    for channel, finding_idx in job["findings"]:
        if not 0 <= channel < masks.shape[0]:
            raise ValueError(f"{name}: seg_channel {channel} out of range for a cached mask "
                             f"with {masks.shape[0]} findings -- the cache and findings.csv "
                             "disagree about this volume")
        fg, lo, hi, cen = mask_geometry(masks[channel])
        rows.append(dict(zip(COLUMNS, [
            name, channel, finding_idx, *(int(v) for v in ct.shape), mean, std, fg,
            lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], *cen])))
    return {"name": name, "rows": rows}


def gate_hu(paired_root: Path, ct_dir: Path, names: list[str], n: int) -> None:
    """GATE HU: the cached array really is `load_ct_hu`'s output (mean/std are exact here)."""
    from rexg_data.ct_volume import load_ct_hu, rescale_map

    rescale = rescale_map(paired_root)
    for name in names[:n]:
        nifti = paired_root / "ct" / name
        if not nifti.is_file():
            raise FileNotFoundError(f"GATE HU: {nifti} missing, cannot verify the cache")
        slope, intercept = rescale[name]
        fresh = load_ct_hu(nifti, slope, intercept)
        want = (float(fresh.mean()), float(fresh.std()))
        got = volume_hu_stats(np.load(ct_dir / f"{name}.npy"))
        if not np.allclose(want, got, rtol=1e-6, atol=1e-4):
            raise SystemExit(
                f"GATE HU FAILED on {name}: npy cache (mean, std) = {got}, but "
                f"ct_volume.load_ct_hu on {nifti} gives {want}. The cache is not raw HU; "
                "do not build stats from it.")
        print(f"  GATE HU ok  {name}  mean/std {got[0]:.4f}/{got[1]:.4f} "
              f"== load_ct_hu {want[0]:.4f}/{want[1]:.4f}", flush=True)


def self_test() -> None:
    """Runnable check for the two pieces of real logic. No data, no GPU."""
    rng = np.random.default_rng(0)
    a = (rng.normal(-400, 500, size=(9, 8, 7))).astype(np.int16)
    mean, std = volume_hu_stats(a, slab=4)
    assert abs(mean - a.mean()) < 1e-9, (mean, a.mean())
    assert abs(std - a.std()) < 1e-6, (std, a.std())

    m = np.zeros((6, 7, 8), np.uint8)
    m[2:5, 1:3, 4] = 1                       # lo=(2,1,4) hi=(4,2,4) centroid=(3,1.5,4)
    fg, lo, hi, cen = mask_geometry(m)
    assert (fg, lo, hi, cen) == (6, [2, 1, 4], [4, 2, 4], [3.0, 1.5, 4.0]), (fg, lo, hi, cen)
    assert mask_geometry(np.zeros((3, 3, 3), np.uint8))[0] == 0
    print("self-test ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/rexg_native.yaml"))
    ap.add_argument("--splits", default="train,val",
                    help="miccai_split values to include (G-INDEX)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N volumes only -- writes a PARTIAL artifact, labelled as "
                         "such in the header line and in the filename suffix")
    ap.add_argument("--verify", type=int, default=3, help="GATE HU sample size")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

    from rexg_data.config import load_data_config

    cfg = load_data_config(args.config)
    paired_root = Path(cfg.paired_root)
    ct_dir, seg_dir = cfg.npy_ct_dir, cfg.npy_seg_dir
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # ---- G-INDEX -------------------------------------------------------------------
    findings = pd.read_csv(paired_root / "findings.csv", dtype={"VolumeName": str})
    rows = findings[findings["miccai_split"].isin(splits) & (findings["has_seg"] == 1)]
    if rows.empty:
        raise SystemExit(f"G-INDEX: no findings with miccai_split in {splits} and has_seg==1")
    names = sorted(rows["VolumeName"].unique())
    cached = {p.name[:-4] for p in ct_dir.iterdir() if p.suffix == ".npy"}
    missing = [n for n in names if n not in cached]
    if missing:
        raise SystemExit(f"G-INDEX: {len(missing)} selected volumes have no npy cache entry, "
                         f"e.g. {missing[:5]} -- run scripts/build_npy_cache.py first")
    print(f"G-INDEX: {len(names)} volumes / {len(rows)} findings from findings.csv"
          f"[{splits}, has_seg==1]; the cache holds {len(cached)} CTs, so globbing it would "
          f"have added {len(cached) - len(names)} maskless volumes", flush=True)

    if args.verify:
        clipped = set(pd.read_csv(Path(cfg.npy_root) / "index.csv",
                                  dtype={"VolumeName": str})
                      .dropna(subset=[CLIP_COLUMN])["VolumeName"])
        gate_hu(paired_root, ct_dir, [n for n in names if n not in clipped], args.verify)

    partial = bool(args.limit) and args.limit < len(names)
    if partial:
        names = names[:args.limit]
        rows = rows[rows["VolumeName"].isin(set(names))]
        print(f"PARTIAL RUN: first {len(names)} volumes / {len(rows)} findings only",
              flush=True)
    out = Path(args.out) if args.out else Path(cfg.npy_root) / (
        f"stats.partial{len(names)}.csv" if partial else "stats.csv")

    by_volume: dict[str, list] = {}
    for name, channel, finding_idx in zip(rows["VolumeName"],
                                          rows["seg_channel"].astype(int),
                                          rows["finding_idx"].astype(int)):
        by_volume.setdefault(name, []).append((channel, finding_idx))
    jobs = [{"name": n, "ct_dir": str(ct_dir), "seg_dir": str(seg_dir),
             "findings": sorted(by_volume[n])} for n in names]

    collected, failures, t0 = [], {}, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_volume, j): j["name"] for j in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                collected.extend(future.result()["rows"])
            except Exception as exc:                      # counted, never swallowed
                failures[futures[future]] = f"{type(exc).__name__}: {exc}"
            if done % 200 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(f"  {done}/{len(jobs)} volumes  {elapsed / 60:.1f} min  "
                      f"eta {elapsed / done * (len(jobs) - done) / 60:.1f} min", flush=True)

    stats = pd.DataFrame(collected, columns=COLUMNS).sort_values(
        ["VolumeName", "seg_channel"], ignore_index=True)
    stats.to_csv(out, index=False)
    wall = time.time() - t0
    print(f"\nwrote {out}  {len(stats)} rows  {wall / 60:.2f} min wall "
          f"({wall / max(len(jobs), 1):.2f} s/volume, {args.workers} workers)")
    if partial:
        print("*** PARTIAL ARTIFACT: this is a labelled subset, not the full pass ***")
    if failures:
        raise SystemExit(f"{len(failures)} volumes FAILED (stats.csv above holds only the "
                         f"survivors): {dict(list(failures.items())[:5])}")

    # ---- G-PIXELS ------------------------------------------------------------------
    merged = stats.merge(rows[["VolumeName", "seg_channel", "pixels"]],
                         on=["VolumeName", "seg_channel"], how="left", validate="one_to_one")
    bad = merged[merged["fg_voxels"] != merged["pixels"].astype(int)]
    if len(bad):
        raise SystemExit(
            f"G-PIXELS FAILED on {len(bad)}/{len(merged)} rows: measured fg_voxels != "
            f"findings.csv pixels, so a bbox is filed under the wrong finding. "
            f"{bad[['VolumeName', 'seg_channel', 'fg_voxels', 'pixels']].head().to_dict('records')}")
    print(f"G-PIXELS ok: all {len(merged)} rows match findings.csv pixels exactly")
    for column in ("hu_mean", "hu_std"):
        by_vol = stats.drop_duplicates("VolumeName")[column]
        print(f"{column:8s} min {by_vol.min():10.3f}  median {by_vol.median():10.3f}  "
              f"max {by_vol.max():10.3f}   (n={len(by_vol)} volumes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
