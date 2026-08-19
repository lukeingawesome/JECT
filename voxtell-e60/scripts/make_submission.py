#!/usr/bin/env python
"""Verify a test prediction directory against the official manifest, then zip it for ReXrank.

The challenge asks for "a single .zip file (not a folder), containing your 300 prediction
.nii.gz files", scored automatically. Everything that can go wrong here is silent: a missing
case, a case whose finding count does not match the manifest, a mask stored at the wrong shape,
or a stray file in the archive. None of it raises anywhere -- you find out from a leaderboard
score that is quietly wrong, hours later, with no way to tell which of those it was.

So this refuses to write the zip unless all of the following hold, checked against
MICCAI_challenge_dataset.json rather than against our own derived tables:

  * exactly the 300 official test case names, no more and no fewer;
  * each file's shape is (F, H, W, D) with H,W,D equal to the manifest's `shape` and F equal to
    the number of findings the manifest lists for that case -- F is the axis a submission gets
    wrong most easily, because it comes from the finding order, not from the image;
  * every mask is binary and dtype uint8, since a label-valued mask would be scored as
    instances the model never claimed;
  * the archive is flat, with no directory prefix.

It also writes submission_manifest.json next to the zip: which prediction directory, which
threshold, which checkpoint and text bank (read from the run_shard records predict_split
leaves), file count and total size. A leaderboard entry you cannot trace back to a model is
worth very little once there are several of them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                                                  # noqa: E402

MANIFEST = paths.CHALLENGE_DIR / "MICCAI_challenge_dataset.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred-dir", required=True, help="one .nii.gz per test case")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="recorded in the manifest")
    ap.add_argument("--prob-dir", default=None,
                    help="the prob dir these came from; its run_shard*.json give provenance")
    a = ap.parse_args()

    # Provenance first, because it is the one check that costs nothing: resolving it after the
    # 300-file verification means a misspelled --prob-dir is only reported four minutes later.
    # predict_split writes run_shard*.json BESIDE the prob/ directory, not inside it, so look in
    # both -- passing prob/ used to find nothing, leave prov empty, and write the manifest
    # anyway, producing a submission with no way back to the checkpoint that made it.
    recs = []
    if a.prob_dir:
        d = Path(a.prob_dir)
        recs = sorted(d.glob("run_shard*.json")) or sorted(d.parent.glob("run_shard*.json"))
        if not recs:
            print(f"REFUSING -- --prob-dir {d} has no run_shard*.json in it or in {d.parent}, "
                  f"so the manifest could not say which checkpoint, text bank or tile_step made "
                  f"these masks. Point it at the predict_split output directory, or drop the "
                  f"flag to declare the provenance unknown.")
            return 1

    official = json.loads(MANIFEST.read_text())["test"]
    want_shape = {e["name"]: tuple(e["shape"]) for e in official}
    want_f = {e["name"]: len(e["findings"]) for e in official}
    pred = Path(a.pred_dir)

    have = sorted(p.name for p in pred.glob("*.nii.gz"))
    missing = sorted(set(want_shape) - set(have))
    extra = sorted(set(have) - set(want_shape))
    problems = []
    if missing:
        problems.append(f"{len(missing)} official cases have no prediction, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} files are not official test cases, e.g. {extra[:3]}")

    checked = 0
    for name in have:
        if name not in want_shape:
            continue
        img = nib.load(str(pred / name))
        shape = tuple(img.shape)
        if len(shape) != 4:
            problems.append(f"{name}: shape {shape} is not (F, H, W, D)")
            continue
        if shape[1:] != want_shape[name]:
            problems.append(f"{name}: spatial {shape[1:]} != manifest {want_shape[name]}")
        if shape[0] != want_f[name]:
            problems.append(f"{name}: {shape[0]} findings, manifest says {want_f[name]}")
        arr = np.asanyarray(img.dataobj)
        if arr.dtype != np.uint8:
            problems.append(f"{name}: dtype {arr.dtype}, expected uint8")
        u = np.unique(arr)
        if u.size and not set(u.tolist()) <= {0, 1}:
            problems.append(f"{name}: not binary, values {u[:5].tolist()}")
        checked += 1

    print(f"checked {checked} files against the official manifest ({len(official)} cases)")
    if problems:
        print(f"\nREFUSING TO WRITE THE ZIP -- {len(problems)} problem(s):")
        for p in problems[:15]:
            print(f"  {p}")
        if len(problems) > 15:
            print(f"  ... and {len(problems) - 15} more")
        return 1

    out = Path(a.zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in have:
            z.write(pred / name, arcname=name)          # flat: no directory prefix
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert len(names) == len(official) and not any("/" in n for n in names), \
        "the archive is not a flat set of exactly the official cases"

    prov = {}
    if recs:
        r = json.loads(recs[0].read_text())
        prov = {k: r[k] for k in ("checkpoint", "checkpoint_sha", "checkpoint_step", "arm",
                                  "text_bank", "text_bank_sha", "tile_step", "code_sha")}
        shas = {json.loads(f.read_text())["checkpoint_sha"] for f in recs}
        prov["shards_agree_on_checkpoint"] = len(shas) == 1
    manifest = {"zip": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
                "files": len(names), "bytes": out.stat().st_size,
                "pred_dir": str(pred), "threshold": a.threshold, **prov}
    (out.with_name(out.stem + "_manifest.json")).write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}  {len(names)} files  {out.stat().st_size / 2**20:.0f} MiB")
    print(f"  manifest -> {out.with_name(out.stem + '_manifest.json').name}")
    for k, v in prov.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
