"""Invert model-space predictions to the ORIGINAL CT geometry, then score (val) or zip (test).

Two modes, and you should always run the first before the second:

  --mode score   invert val predictions and Dice them against the ORIGINAL-geometry ground truth.
                 This is the number that matters: a good score in model space means nothing if the
                 inversion loses the mask. Run this to confirm your val result reproduces end to end.

  --mode zip     invert test predictions, write one file per case named exactly like the CT with
                 shape (F, H, W, D), and zip them for submission.

Post-processing (`--closing`) is applied here, so the SAME value must be used for both modes --
otherwise the submitted masks are not the ones any Dice was measured on.
"""
from __future__ import annotations
import argparse, glob, os, zipfile

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from nibabel.processing import resample_from_to

from ject.submission.invert import npz_affine


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def invert_case(pred: np.ndarray, ct_path: str, ts_path: str, crop: int, spacing: float,
                closing: int) -> tuple[np.ndarray, np.ndarray]:
    """(F,S,S,S) model-space mask -> (F,*orig_shape) on the original grid, plus the CT affine."""
    S = pred.shape[1]
    assert pred.shape[1:] == (S, S, S), f"expected a cube, got {pred.shape[1:]}"
    if S > crop:
        raise SystemExit(f"prediction cube {S} exceeds the preprocess crop {crop}; "
                         f"--crop/--spacing do not match the data this model was trained on.")
    off = (crop - S) // 2                                    # undo the centre crop S -> crop
    ct = nib.load(ct_path)
    orig_shape, A0 = ct.shape[:3], ct.affine
    Anpz, _, _ = npz_affine(ct_path, ts_path, T=crop, SP=spacing)
    inv = np.zeros((pred.shape[0], *orig_shape), np.uint8)
    for f in range(pred.shape[0]):
        m = pred[f] > 0
        if m.sum() == 0:
            continue
        if closing > 0:
            m = ndi.binary_closing(m, iterations=closing)
        cube = np.zeros((crop, crop, crop), np.uint8)
        cube[off:off + S, off:off + S, off:off + S] = m.astype(np.uint8)
        img = nib.Nifti1Image(np.ascontiguousarray(cube), Anpz)
        res = resample_from_to(img, (orig_shape, A0), order=0)
        inv[f] = (np.asanyarray(res.dataobj) > 0).astype(np.uint8)
    return inv, A0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["score", "zip"])
    ap.add_argument("--pred_dir", required=True, help="dir of <id>_pred.nii.gz from predict.py")
    ap.add_argument("--orig_dir", required=True, help="original CT .nii.gz")
    ap.add_argument("--totalseg_dir", required=True, help="TotalSegmentator labels .nii.gz")
    ap.add_argument("--gt_dir", default=None, help="ORIGINAL-geometry GT masks (--mode score)")
    ap.add_argument("--out_dir", default=None, help="where inverted masks go (--mode zip)")
    ap.add_argument("--zip_path", default=None)
    ap.add_argument("--crop", type=int, required=True,
                    help="preprocess crop size T: 256 (1.5mm pipeline) or 384 (1.0mm pipeline)")
    ap.add_argument("--spacing", type=float, required=True, help="preprocess spacing mm: 1.5 or 1.0")
    ap.add_argument("--closing", type=int, default=0,
                    help="binary-closing iterations. MUST match between --mode score and --mode zip.")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected case count; a short zip is flagged loudly (test split = 300)")
    a = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(a.pred_dir, "*_pred.nii.gz")))
    if not preds:
        raise SystemExit(f"no *_pred.nii.gz in {a.pred_dir}")
    print(f"[pack] {len(preds)} predictions, mode={a.mode}, crop={a.crop} spacing={a.spacing} "
          f"closing={a.closing}", flush=True)
    if a.mode == "zip":
        if not a.out_dir:
            raise SystemExit("--mode zip needs --out_dir")
        os.makedirs(a.out_dir, exist_ok=True)
        # A pre-populated out_dir is refused: stale files from another split/checkpoint would be
        # zipped AND would satisfy the --expect completeness check, defeating the very guard that
        # exists to catch a short submission.
        stale = sorted(glob.glob(os.path.join(a.out_dir, "*.nii.gz")))
        if stale:
            raise SystemExit(
                f"{len(stale)} .nii.gz already in {a.out_dir} (e.g. {os.path.basename(stale[0])}). "
                f"Use an empty --out_dir: stale files would be zipped into the submission and could "
                f"hide missing cases from --expect.")
    elif not a.gt_dir:
        raise SystemExit("--mode score needs --gt_dir (original-geometry ground truth)")

    dices, missing, done = [], [], 0
    written = []          # only these are zipped
    for p in preds:
        cid = os.path.basename(p)[:-len("_pred.nii.gz")]
        ctp = os.path.join(a.orig_dir, f"{cid}.nii.gz")
        tsp = os.path.join(a.totalseg_dir, f"{cid}.nii.gz")
        if not (os.path.exists(ctp) and os.path.exists(tsp)):
            missing.append(cid)
            print(f"  [MISSING] {cid}: no original CT or totalseg -> absent from output")
            continue
        pred = np.asanyarray(nib.load(p).dataobj)
        inv, A0 = invert_case(pred, ctp, tsp, a.crop, a.spacing, a.closing)

        if a.mode == "zip":
            dst = os.path.join(a.out_dir, f"{cid}.nii.gz")
            nib.save(nib.Nifti1Image(inv, A0), dst)
            written.append(dst)
        else:
            gt = np.asanyarray(nib.load(os.path.join(a.gt_dir, f"{cid}.nii.gz")).dataobj)
            if gt.ndim == 3:
                gt = gt[None]
            for f in range(min(inv.shape[0], gt.shape[0])):
                if (gt[f] > 0).sum() == 0:
                    continue
                dices.append(_dice(inv[f], gt[f]))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(preds)}", flush=True)

    if a.mode == "score":
        if not dices:
            raise SystemExit("no findings scored -- check --gt_dir")
        print(f"\n[pack] ORIGINAL-GEOMETRY Dice over {len(dices)} findings: "
              f"mean {np.mean(dices):.4f}  median {np.median(dices):.4f}")
        print(f"[pack] hit@0.1: {np.mean([d >= 0.1 for d in dices]):.4f}")
        print("[pack] Compare against the model-space val Dice. A large drop means the inversion "
              "is losing mask, not that the model is worse.")
        return

    zpath = a.zip_path or (a.out_dir.rstrip("/") + ".zip")
    files = sorted(written)        # NOT a glob of out_dir -- only this run's output
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))
    print(f"\n[pack] {len(files)} files -> {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB)")
    if (a.expect and len(files) != a.expect) or missing:
        print(f"[pack] *** WARNING: {len(files)} files, expected {a.expect}; "
              f"{len(missing)} case(s) missing inputs: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}. Missing cases score ZERO. "
              f"DO NOT SUBMIT until resolved. ***")
    else:
        print("[pack] completeness OK.")


if __name__ == "__main__":
    main()
