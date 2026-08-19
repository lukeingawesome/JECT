"""The single CT-array boundary: NIfTI path -> float32 3D array.

Every consumer that turns a paired CT file into a `VLMTaskEvalItem` volume calls THIS
(the fold loader, the MedGemma/M3D task-eval runners, and the orchestrator's validate
stage). A per-caller copy is how axis order, dtype, or intensity handling silently drift
apart between the paths that select volumes differently but must read them identically.

Not in scope: `m3d_canonical_volume` (vlm_client) owns the M3D-specific ZYX transpose and
min-max normalisation, and the segmentation datamodule reads through MONAI transforms.
Both are separate, documented boundaries — this one says how bytes become an array, plus
the two things the paired NIfTI headers get wrong and only CT-RATE's metadata knows: the
intensity rescale (`load_ct_hu` / `rescale_map`) and the geometry (`spacing_map` /
`ras_affine`).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_ct_volume(path: str | Path) -> NDArray[np.float32]:
    """Read a NIfTI CT volume as a float32 array.

    float32 is requested from nibabel directly rather than casting after a float64 read:
    a 512x512x300 volume is ~300 MB at float64, and the cast would discard the extra
    precision anyway.
    """
    import nibabel as nib   # deferred: keeps this module importable without nibabel

    return np.asarray(
        nib.Nifti1Image.from_filename(str(path)).get_fdata(dtype=np.float32),
        dtype=np.float32,
    )


HU_EVIDENCE_MIN = -100.0
"""Below this, an array cannot be raw stored CT values -- it is already in HU.

Measured on 40 randomly sampled real paired volumes read in FULL (20 per
`rescale_intercept` group, -1024 and -8192, seed 0): every one reports header
`(1.0, 0.0)`, and no volume goes below 0.0. Per-volume minima run 0.0..7025.0 (both group
minima are exactly 0.0), maxima 2671..40348. So this bound sits 100 below the lowest
stored value in the measured corpus and ~900 above the about -1000 air floor an
already-rescaled array shows -- it cannot fire on these files.
"""


def load_ct_hu(path: str | Path, slope: float, intercept: float) -> NDArray[np.float32]:
    """Read a NIfTI CT volume as real Hounsfield units: `stored * slope + intercept`.

    ReXGroundingCT's paired CT files carry NO rescale in the NIfTI header (`scl_slope` /
    `scl_inter` are NaN on disk), so `load_ct_volume` returns raw stored values (0..N) and
    any HU-based windowing on them is wrong. The slope/intercept live in CT-RATE's metadata
    (see `rescale_map`) and DIFFER PER VOLUME (observed intercepts from -1024 to -8192), so
    no constant fixes it.

    But `get_fdata()` DOES apply the header scaling when the file carries one, so applying
    the sidecar unconditionally would double-apply it on any normal CT. Measured (nibabel
    5.4.2) on a file written with `set_slope_inter(2, -1024)` over raw 0,1,2:
    `get_fdata()` -> -1024, -1022, -1020. Note the header FIELDS are useless for detecting
    this: on a loaded image `header['scl_slope']` is NaN and `header.get_slope_inter()` is
    `(None, None)` **regardless** of what is on disk, because nibabel moves the scaling onto
    the array proxy (precisely to stop double application). `dataobj.slope`/`.inter` is the
    only reliable read; it is `(1.0, 0.0)` for both a NaN and an identity header (verified
    on real ReXGroundingCT files and on both synthetic cases).

    So: no header scaling -> apply the sidecar; header scaling that agrees with the sidecar
    -> already HU, return as-is; header scaling that disagrees -> raise, no winner picked.

    `(1.0, 0.0)` is the ambiguous one -- a NaN header and an explicit identity header read
    identically -- so there the VALUES decide, not the header: raw stored CT values are
    non-negative, HU are not (see `HU_EVIDENCE_MIN` for the measurement and its margin). An
    array that is already HU plus a sidecar that is not the identity means applying it would
    double-shift (HU air -1024 with a (1, -1024) sidecar -> -2048), so that RAISES too.
    """
    import nibabel as nib   # deferred: keeps this module importable without nibabel

    # from_filename is lazy (no pixel read), so the array still comes from load_ct_volume
    proxy = nib.Nifti1Image.from_filename(str(path)).dataobj
    header = (float(proxy.slope), float(proxy.inter))
    array = load_ct_volume(path)
    if header == (1.0, 0.0):
        if (slope, intercept) != (1.0, 0.0) and float(array.min()) < HU_EVIDENCE_MIN:
            raise ValueError(
                f"{path}: data range [{float(array.min())}, {float(array.max())}] is below "
                f"{HU_EVIDENCE_MIN}, so the array is already in HU, but the metadata rescale "
                f"is {(float(slope), float(intercept))}; applying it would rescale twice "
                f"(the NIfTI header is {header}, which cannot tell the two cases apart)")
        return (array * np.float32(slope) + np.float32(intercept)).astype(np.float32)
    if not np.allclose(header, (slope, intercept), rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"{path}: NIfTI header rescale {header} disagrees with the metadata rescale "
            f"{(float(slope), float(intercept))}; refusing to guess which one is right")
    return array


def _volume_column_map(root: str | Path, columns: tuple[str, ...],
                       cast=float) -> dict[str, tuple]:
    """VolumeName -> the named columns of `<root>/volume_labels.csv`, each through `cast`.

    Fails loud on a missing file or missing columns. Every value here has a wrong-but-
    plausible default (intercept 0 keeps the raw-value bug alive, spacing 1.0 is exactly
    the lie the NIfTI headers already tell), so there is no safe fallback.
    """
    path = Path(root) / "volume_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"CT volume metadata not found: {path} (needed for {', '.join(columns)})")
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        missing = [c for c in ("VolumeName", *columns) if c not in (rows.fieldnames or ())]
        if missing:
            raise KeyError(f"{path} is missing required column(s): {missing}")
        return {row["VolumeName"]: tuple(cast(row[c]) for c in columns) for row in rows}


def rescale_map(root: str | Path) -> dict[str, tuple[float, float]]:
    """VolumeName -> (rescale_slope, rescale_intercept) from `<root>/volume_labels.csv`."""
    return _volume_column_map(root, ("rescale_slope", "rescale_intercept"))  # type: ignore[return-value]


def spacing_map(root: str | Path) -> dict[str, tuple[float, float, float]]:
    """VolumeName -> (sx, sy, sz) in mm from `<root>/volume_labels.csv`.

    The paired NIfTI headers claim `pixdim == (1, 1, 1)` for every volume; the real DICOM
    spacing (sx == sy in 0.30..0.98, sz in 0.50..3.00) only exists in CT-RATE's metadata.
    """
    return _volume_column_map(root, ("sx", "sy", "sz"))  # type: ignore[return-value]


def origin_map(root: str | Path) -> dict[str, tuple[float, float, float]]:
    """VolumeName -> (origin_x, origin_y, origin_z) in mm from `<root>/volume_labels.csv`.

    CT-RATE's metadata publishes `ImagePositionPatient` (LPS mm) for the FIRST slice only;
    the builder carries it as these three columns. It is NOT usable as an affine translation
    on its own -- read `ras_affine` for what a caller has to establish first, and pair it
    with `patient_position_map`.
    """
    return _volume_column_map(root, ("origin_x", "origin_y", "origin_z"))  # type: ignore[return-value]


def patient_position_map(root: str | Path) -> dict[str, str]:
    """VolumeName -> DICOM PatientPosition ("HFS" / "FFS") from `<root>/volume_labels.csv`.

    Which end of the stored array the published origin sits at depends on it, and the FFS
    volumes additionally have an inverted slice direction (`ct_transforms.SLICE_AXIS_SIGN`).
    """
    return {name: value for name, (value,)
            in _volume_column_map(root, ("patient_position",), cast=str).items()}


def ras_affine(sx: float, sy: float, sz: float,
               origin: tuple[float, float, float] | None = None) -> NDArray[np.float64]:
    """The 4x4 voxel->RAS affine these volumes actually have: `diag(-sx, -sy, +sz, 1)`.

    NOT `diag(sx, sy, sz, 1)`. The arrays are stored in **LPS index order** (axis0 = +Left,
    axis1 = +Posterior, axis2 = +Superior) while the identity affine in the file claims RAS.
    Negating x and y is the standard LPS->RAS conversion.

    Evidence, so the next reader does not have to re-derive it:

    - All 3492 paired volumes have DICOM `ImageOrientationPatient == [1, 0, 0, 0, 1, 0]`
      (row -> +x = Left, column -> +y = Posterior in LPS), checked directly.
    - Centroids of real val masks whose report text names one side / lobe only:

      | report text                    | mean centroid (normalised) |  n |
      |--------------------------------|----------------------------|----|
      | "left ..."                     | axis0 = 0.751              | 10 |
      | "right ..."                    | axis0 = 0.306              | 14 |
      | "upper lobe / apical / apex"   | axis2 = 0.736              |  8 |
      | "lower lobe / basal / base"    | axis2 = 0.419              | 22 |

      i.e. axis0 grows toward the patient's LEFT and axis2 toward SUPERIOR.

    Translation defaults to zero, so the default affine fixes direction and scale only. NOT
    because the origin is unavailable -- CT-RATE's metadata DOES publish
    `ImagePositionPatient`, and the builder carries it as `origin_x/y/z` (see `origin_map`).
    It defaults off because that column is the FIRST DICOM SLICE's position, and which end
    of the stored array that slice is has never been verified here. Assuming it is voxel
    (0,0,0) would just swap one unverified origin for another.

    Before passing `origin` (LPS mm; x and y are negated to RAS, matching the direction
    terms above), a caller must establish, for the volumes it is loading:

    1. that stored index axis2 == 0 really is the slice whose ImagePositionPatient was
       published -- if the array is stored the other way round, the translation is wrong by
       (n_slices - 1) * sz along z;
    2. the volume's PatientPosition (`patient_position_map`): the FFS volumes have their
       slice axis negated (`ct_transforms.SLICE_AXIS_SIGN`), which moves the first
       slice to the other end of the array;
    3. that the paired mask NIfTI is stored on the same grid, since image and label must
       take the same affine.
    """
    affine = np.diag(np.array([-sx, -sy, sz, 1.0], dtype=np.float64))
    if origin is not None:
        ox, oy, oz = (float(v) for v in origin)
        affine[:3, 3] = (-ox, -oy, oz)      # LPS -> RAS, the same negation as the diagonal
    return affine
