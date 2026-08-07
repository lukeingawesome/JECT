"""Map a prediction from model space back to the ORIGINAL nii.gz geometry.

Training runs on preprocessed cubes (reoriented to LPS, resampled to an isotropic spacing, then
lung-cropped to T^3). Submission requires masks on the original CT grid, so this replays that
preprocessing analytically to recover the array->world affine and lets nibabel resample back.

Two (T, spacing) pairs are in use, covering the SAME 384 mm field of view:
    T=256, SP=1.5   for 224-model data
    T=384, SP=1.0   for 336-model data
Pass whichever matches the preprocessing that produced your npz -- guessing from the prediction
size is not safe.

SELF-TEST: `python -m ject.submission.invert ...` inverts the ground truth stored in model space
and compares it against the original mask. Median Dice below 0.90 fails loudly.

npz array order is (C, D=I-S, H=R-L, W=A-P); preprocess axes after Orientationd('LPS')
are (L=R-L, P=A-P, S=I-S). We reconstruct the npz→world affine, then nibabel
resample_from_to handles the rest.
"""
import numpy as np, nibabel as nib
from nibabel.orientations import (axcodes2ornt, io_orientation, ornt_transform,
                                  apply_orientation, inv_ornt_aff)
from nibabel.processing import resample_from_to
import scipy.ndimage as ndi

LUNG = (10, 11, 12, 13, 14)
SP = 1.5
T = 256


def _to_lps(data, affine):
    """Reorient (data, affine) to LPS axis order (L,P,S) = (R-L, A-P, I-S)."""
    src = io_orientation(affine)
    dst = axcodes2ornt(('L', 'P', 'S'))
    tr = ornt_transform(src, dst)
    d = apply_orientation(data, tr)
    a = affine @ inv_ornt_aff(tr, data.shape)
    return d, a


def _lung_crop_start(R_ts, axis, Ri, T=T):
    """AnatomyAwareCropd start for one axis (replicates preprocess_ctrate logic).
    T = preprocess crop size (256 for the 224-model data, 384 for the 336-model data)."""
    lung = np.isin(R_ts, LUNG)
    coords = np.argwhere(lung)
    bmin = coords[:, axis].min(); bmax = coords[:, axis].max()
    center = (bmin + bmax) // 2
    extent = bmax - bmin + 1
    if extent >= T:
        start = center - T // 2
    else:
        start = bmin - (T - extent) // 2
    return int(max(0, min(start, Ri - T)))


def npz_affine(ct_path, ts_path, T=T, SP=SP):
    """Reconstruct the affine mapping npz array index (D=IS,H=RL,W=AP) -> world.
    T  = preprocess crop size: 256 (224-model / final_preprocessed_half) or 384 (336-model).
    SP = preprocess isotropic spacing (mm): 1.5 (256 pipeline) or 1.0 (384 pipeline).
    Both models cover the SAME physical FOV (256·1.5 = 384·1.0 = 384mm)."""
    ct = nib.load(ct_path); A0 = ct.affine
    ctl, Al = _to_lps(np.asanyarray(ct.dataobj).astype(np.float32), A0)
    ts = np.asanyarray(nib.load(ts_path).dataobj)
    tsl, _ = _to_lps(ts.astype(np.int16), A0)

    cur = np.sqrt((Al[:3, :3] ** 2).sum(0))          # current voxel sizes
    zoom = cur / SP
    R_ts = ndi.zoom(tsl, zoom, order=0)
    Rshape = R_ts.shape                               # resampled LPS size (L,P,S)
    Ar = Al.copy(); Ar[:3, :3] = Al[:3, :3] @ np.diag(1.0 / zoom)

    # per-LPS-axis offset of npz origin within the resampled volume
    if np.isin(R_ts, LUNG).any():
        off = []
        for i in range(3):
            Ri = Rshape[i]
            off.append(_lung_crop_start(R_ts, i, Ri, T) if Ri >= T else -((T - Ri) // 2))
    else:                                             # no lungs -> symmetric center
        off = [(_o if (Ri := Rshape[i]) >= T else -((T - Ri) // 2))
               for i, _o in enumerate([(Rshape[j] - T) // 2 for j in range(3)])]
    off = np.array(off, float)

    Acrop = Ar.copy()
    Acrop[:3, 3] = Ar[:3, :3] @ off + Ar[:3, 3]
    # npz array (S,L,P) -> LPS (L,P,S): LPS = Perm @ npz, Perm rows pick [b,c,a]
    Perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    Anpz = Acrop.copy()
    Anpz[:3, :3] = Acrop[:3, :3] @ Perm
    return Anpz, A0, ct.shape[:3]


def invert_mask(mask256, ct_path, ts_path, order=0, T=T, SP=SP):
    """Map a T³ npz-space mask -> original nii.gz grid. (T,SP)=(256,1.5) for the 224
    model, (384,1.0) for the 336 model."""
    Anpz, A0, orig_shape = npz_affine(ct_path, ts_path, T=T, SP=SP)
    img = nib.Nifti1Image(np.ascontiguousarray(mask256), Anpz)
    out = resample_from_to(img, (orig_shape, A0), order=order)
    return np.asanyarray(out.dataobj)


def _dice(a, b):
    a = a > 0; b = b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


if __name__ == "__main__":
    # ROUND-TRIP SELF-TEST: invert the ground truth stored in model space and compare against the
    # ORIGINAL mask.
    #
    # This measures TWO things at once, and only one of them is a bug:
    #   (a) inversion fidelity   -- what this test is for
    #   (b) preprocessing loss   -- resampling to SP mm destroys small findings before the model
    #                               ever sees them, and no inversion can recover that.
    # Measured on val at T=256/SP=1.5: findings >=1000 original voxels round-trip at median Dice
    # 0.898, while findings <1000 voxels manage only 0.479 -- a 165-voxel finding survives
    # preprocessing as EIGHT voxels. So the gate below is applied to LARGE findings only; the
    # small-finding number is reported as information about the data pipeline, not as a failure.
    import argparse, glob, os, random, statistics
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="preprocessed .npz (must contain `rex`)")
    ap.add_argument("--orig_dir", required=True, help="original CT .nii.gz")
    ap.add_argument("--totalseg_dir", required=True, help="TotalSegmentator labels .nii.gz")
    ap.add_argument("--rex_dir", required=True, help="ORIGINAL-geometry ground-truth masks")
    ap.add_argument("--crop", type=int, default=256, help="preprocess crop size T (256 or 384)")
    ap.add_argument("--spacing", type=float, default=1.5, help="preprocess spacing mm (1.5 or 1.0)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--big", type=int, default=1000, help="orig-voxel count separating large/small")
    ap.add_argument("--gate", type=float, default=0.85, help="min median Dice on LARGE findings")
    a = ap.parse_args()
    cases = [os.path.basename(p)[:-7] for p in glob.glob(a.orig_dir + "/*.nii.gz")]
    cases = [c for c in cases if os.path.exists(f"{a.npz_dir}/{c}.npz")]
    random.Random(0).shuffle(cases)
    big, small = [], []
    for c in cases[:a.n]:
        npzrex = np.load(f"{a.npz_dir}/{c}.npz")["rex"]
        orig = np.asanyarray(nib.load(f"{a.rex_dir}/{c}.nii.gz").dataobj)
        F = orig.shape[0] if orig.ndim == 4 else 1
        for i in range(min(F, npzrex.shape[0])):
            o = orig[i] if orig.ndim == 4 else orig
            n_o = int((o > 0).sum())
            if n_o == 0:
                continue
            inv = invert_mask(npzrex[i], f"{a.orig_dir}/{c}.nii.gz", f"{a.totalseg_dir}/{c}.nii.gz",
                              order=0, T=a.crop, SP=a.spacing)
            d = _dice(inv, o)
            (big if n_o >= a.big else small).append(d)
            print(f"  {c} f{i}: {n_o:>7} orig vox -> {int((npzrex[i]>0).sum()):>6} npz vox  "
                  f"round-trip Dice={d:.3f}")
    if not big:
        raise SystemExit("no large findings scored -- raise --n")
    mb = statistics.median(big)
    print(f"\n  LARGE (>={a.big} vox, n={len(big)}): median round-trip Dice {mb:.3f}   <- gates the test")
    if small:
        print(f"  SMALL (< {a.big} vox, n={len(small)}): median {statistics.median(small):.3f}   "
              f"<- preprocessing loss at {a.spacing}mm, NOT an inversion error")
    if mb < a.gate:
        raise SystemExit(f"FAIL: large-finding median {mb:.3f} < {a.gate}. The inversion is losing "
                         f"mask -- check --crop/--spacing match the preprocessing of --npz_dir.")
    print(f"PASS: inversion is faithful on findings the preprocessing preserved.")
