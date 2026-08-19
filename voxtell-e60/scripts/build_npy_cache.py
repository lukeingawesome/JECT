#!/usr/bin/env python
"""Precompute the CT/mask cache: HU-rescaled, RAS-oriented, original grid.

Reading a paired volume today costs ~2.5 s (gzip decode + HU rescale) and every epoch pays
it again. This bakes ONLY the two transforms that are settled -- the CT-RATE HU rescale and
the LPS->RAS reorientation -- and stops there. Voxel spacing, cropping, the intensity window
and the input size stay open decisions, so choosing them later does not mean rebuilding
367 GB.

    CT   -> <npy_root>/ct/<VolumeName>.npy    int16 HU, RAS index order, UNCOMPRESSED
    mask -> <npy_root>/seg/<VolumeName>.npz   uint8 (F,H,W,D), RAS index order, compressed
    index  <npy_root>/index.csv               shape, spacing, affine, position, HU range

Compression is split on measurement, not taste: CT compresses 1.7x but costs 6.7 s/volume to
write and 0.6 s to read, while masks are 0.03-0.8% foreground and compress ~900x (960 GB ->
1.1 GB). So the CT stays raw and mmap-able at 0.1 s, and the masks are compressed.

int16 is lossless here -- verified across the corpus, HU values are integral and inside
int16 range (slope is 1 and the stored values are integers).

Geometry comes from the same transforms the live pipeline uses (`RasAffined`, `Orientationd`,
`HuRescaled`), not a re-implementation, so a cached tensor cannot drift from a fresh one.

    python scripts/build_npy_cache.py [--workers 8] [--limit N] [--config PATH]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REQUIRED = ("sx", "sy", "sz", "rescale_slope", "rescale_intercept", "patient_position")


def _chain():
    """Load -> true affine -> RAS -> HU. No union, no window, no resize."""
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd
    from rexg_data.ct_transforms import HuRescaled, RasAffined
    return Compose([
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image"]),
        RasAffined(keys=["image", "label"], allow_missing_keys=True),
        Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True,
                     labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        HuRescaled(keys=["image"]),
    ])


_CHAIN = None


def _validate_mask(raw: np.ndarray, name: str) -> None:
    """Same contract UnionMaskd enforces: finite, integral, non-negative."""
    bad = {"negative": int(np.count_nonzero(raw < 0))}
    if raw.dtype.kind == "f":
        finite = np.isfinite(raw)
        bad["non_finite"] = raw.size - int(np.count_nonzero(finite))
        bad["fractional"] = int(np.count_nonzero(finite & (raw != np.rint(raw))))
    if any(bad.values()):
        raise ValueError(f"{name}: invalid mask {bad} of {raw.size} voxels")


def build_one(job: dict) -> dict:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = _chain()
    import torch

    name = job["name"]
    ct_out, seg_out = Path(job["ct_out"]), Path(job["seg_out"])
    item = {"image": job["image"], **{k: job[k] for k in REQUIRED}, "name": name}
    if job.get("label"):
        item["label"] = job["label"]

    out = _CHAIN(item)
    img = out["image"]
    arr = np.asarray(img.detach().cpu() if isinstance(img, torch.Tensor) else img)[0]
    if arr.size == 0:
        raise ValueError(f"{name}: empty volume after transforms")
    if np.any(arr != np.rint(arr)):
        raise ValueError(f"{name}: HU values are not integral — int16 would lose data")
    # A handful of volumes carry metal-artifact values far outside the physical CT range
    # (observed max 36,187 HU). Those are clipped to int16 rather than widening every file to
    # int32 -- bone tops out near 2,000 HU and any intensity window we apply clips them
    # anyway. Clipping is COUNTED per volume, not silent, and a volume where it touches more
    # than 0.1% of voxels fails instead: at that point the excursion is not a streak artifact
    # and int16 is the wrong container.
    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    clipped = int(np.count_nonzero((arr < lo) | (arr > hi)))
    if clipped / arr.size > 1e-3:
        raise ValueError(f"{name}: {clipped} voxels ({clipped/arr.size:.2%}) outside int16 "
                         f"range [{arr.min()}, {arr.max()}] — too many to be metal artifact")
    hu = np.clip(arr, lo, hi).astype(np.int16)

    # the tmp name must still END in .npy: np.save APPENDS the extension when it is absent,
    # so a ".npy.tmp" temp file is written as ".npy.tmp.npy" and the rename then misses.
    tmp = ct_out.with_name(ct_out.name + ".tmp.npy")
    np.save(tmp, hu)
    os.replace(tmp, ct_out)            # same fs -> atomic; a killed run leaves no half file

    n_findings = 0
    if job.get("label"):
        lab = out["label"]
        m = np.asarray(lab.detach().cpu() if isinstance(lab, torch.Tensor) else lab)
        _validate_mask(m, name)
        m = m.astype(np.uint8)
        if m.shape[1:] != hu.shape:
            raise ValueError(f"{name}: mask spatial {m.shape[1:]} != CT {hu.shape}")
        n_findings = int(m.shape[0])
        tmpz = seg_out.with_name(seg_out.name + ".tmp.npz")   # savez appends .npz too
        np.savez_compressed(tmpz, mask=m)
        os.replace(tmpz, seg_out)

    affine = np.asarray(img.affine.detach().cpu() if hasattr(img.affine, "detach") else img.affine)
    return {
        "VolumeName": name, "nx": hu.shape[0], "ny": hu.shape[1], "nz": hu.shape[2],
        "sx": job["sx"], "sy": job["sy"], "sz": job["sz"],
        "patient_position": job["patient_position"],
        "hu_min": int(hu.min()), "hu_max": int(hu.max()), "hu_clipped": clipped,
        "n_findings": n_findings,
        "affine": " ".join(f"{v:.6g}" for v in affine.reshape(-1)),
        "ct_bytes": ct_out.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import pandas as pd
    from rexg_data.config import load_data_config

    cfg = load_data_config(args.config)
    # One row per VolumeName, carrying the CT-RATE join this cache cannot be built without:
    # rescale_slope/intercept (HU) and sx/sy/sz/patient_position (geometry). Empty rexg__*
    # cells stay NaN -- unknown, never 0.
    vol = pd.read_csv(Path(cfg.paired_root) / "volume_labels.csv", dtype={"VolumeName": str})
    ct_dir, seg_dir = cfg.npy_ct_dir, cfg.npy_seg_dir
    ct_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    src_ct, src_seg = Path(cfg.paired_root) / "ct", Path(cfg.seg_root)

    have = vol[vol.has_ct == 1]
    missing = [c for c in REQUIRED if c not in have.columns]
    if missing:
        raise SystemExit(
            f"{Path(cfg.paired_root) / 'volume_labels.csv'} is missing {missing}. Those "
            f"columns come from the CT-RATE metadata join, not from the paired NIfTI headers "
            f"-- see src/rexg_data/ct_transforms.py for why a default is not an "
            f"option here.")

    jobs = []
    for _, r in have.iterrows():
        name = str(r.VolumeName)
        out = ct_dir / f"{name}.npy"
        if out.exists():                                   # resumable
            continue
        seg = src_seg / name
        jobs.append({"name": name, "image": str(src_ct / name),
                     "label": str(seg) if (r.has_seg == 1 and seg.exists()) else "",
                     "ct_out": str(out), "seg_out": str(seg_dir / f"{name}.npz"),
                     **{k: (float(r[k]) if k != "patient_position" else str(r[k])) for k in REQUIRED}})
    cached = len(have) - len(jobs)
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(have):,} volumes with CT; {cached:,} already cached; "
          f"{len(jobs):,} to build now -> {ct_dir}", flush=True)
    if not jobs:
        print("nothing to do")
        return 0

    rows, failed, started = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, j): j["name"] for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:                        # noqa: BLE001 - count, report, continue
                failed.append((name, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            if n % 50 == 0 or n == len(jobs):
                el = time.time() - started
                gb = sum(r["ct_bytes"] for r in rows) / 1e9
                print(f"[{n}/{len(jobs)}] ok={len(rows)} fail={len(failed)} "
                      f"{gb:.1f} GB  {el/60:.1f} min  eta {el/n*(len(jobs)-n)/60:.1f} min", flush=True)

    if not rows:
        print(f"\nNOTHING BUILT — {len(failed)} failures, index untouched")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        return 1

    index = Path(cfg.npy_root) / "index.csv"
    old = list(csv.DictReader(index.open())) if index.exists() else []
    known = {r["VolumeName"] for r in rows}
    allrows = [r for r in old if r["VolumeName"] not in known] + [
        {k: str(v) for k, v in r.items()} for r in rows]
    # Union the field names and backfill: a column added between runs (hu_clipped) is absent
    # from rows written by the earlier build, and DictWriter raises rather than filling it.
    fields = list(dict.fromkeys(k for r in allrows for k in r))
    with index.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(allrows)

    gb = sum(r["ct_bytes"] for r in rows) / 1e9
    print(f"\ndone: {len(rows)} built, {len(failed)} failed, {gb:.1f} GB, "
          f"{(time.time()-started)/60:.1f} min; index -> {index}")
    if failed:
        print("FAILURES (re-run to retry):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
