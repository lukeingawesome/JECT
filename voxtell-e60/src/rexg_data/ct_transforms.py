"""The two transforms that turn a paired ReXGroundingCT NIfTI into the cached array.

Extracted verbatim from the datamodule they were written in. Both exist because the paired
files are missing metadata that only the CT-RATE join carries -- the HU rescale and the real
spacing/orientation -- and both RAISE on a missing key rather than defaulting, because the
default is in each case a specific silent-wrong-value bug the comments name.

`scripts/build_npy_cache.py` is the only consumer: it bakes these two settled transforms into
the cache and stops there, so spacing, cropping and window stay open decisions.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform

from .ct_volume import ras_affine

log = logging.getLogger(__name__)

# DICOM PatientPosition -> the sign of the slice axis (axis2) in ras_affine().
# ImageOrientationPatient is [1,0,0,0,1,0] on ALL 3492 paired volumes and it is
# patient-relative, so the IN-PLANE axes are LPS regardless of table direction -- but the
# SLICE axis is not, and it is measurably inverted on the feet-first ones:
#
#   HFS (head-first supine, 3488/3492): axis2 grows toward SUPERIOR   -> +sz
#   FFS (feet-first supine, 4/3492):    axis2 grows toward INFERIOR   -> -sz
#
# Measured on the 3 FFS volumes that ship masks (5 findings): their "upper lobe" findings
# sit at axis2 0.13-0.31 (normalised), where the 66 HFS upper-lobe findings sit at
# 0.687 +- 0.20. The 4th FFS volume, train_13586_a_1.nii.gz, has has_seg == 0 -- no pixels
# to measure, so it is UNVERIFIED and flipped by this same metadata rule rather than
# name-cased out: a per-name exception is exactly the hardcoded allowlist this replaced.
# Any OTHER position RAISES in RasAffined: neither measured nor assumable, and defaulting
# it to HFS is the silent top-to-bottom mirror this whole rule exists to prevent. HFP is
# not hypothetical -- CT-RATE's train_metadata.csv carries HFS 47090 / FFS 57 / HFP 2, and
# a prone volume also mirrors the in-plane axes, which ras_affine's LPS assumption denies.
SLICE_AXIS_SIGN = {"HFS": 1.0, "FFS": -1.0}


class HuRescaled(MapTransform):
    """Raw stored value -> Hounsfield units: `v * rescale_slope + rescale_intercept`.

    ReXGroundingCT's paired CT NIfTI carry NO rescale in the header, so the loaded array
    is raw (0..N) and the true HU needs CT-RATE's PER-VOLUME slope/intercept
    (`ct_volume.rescale_map`). Without this, ScaleIntensityRanged(-1000, 1000) windows the
    wrong numbers (measured: 27.8%-100% of voxels saturate to 1.0 depending on the volume).
    A missing key RAISES: defaulting to (1, 0) would silently mean "already HU" and
    re-introduce exactly that bug."""

    def __call__(self, data):
        d = dict(data)
        missing = [k for k in ("rescale_slope", "rescale_intercept") if d.get(k) is None]
        if missing:
            raise KeyError(f"{missing} missing for {d.get('name')!r}: raw CT cannot be "
                           f"assumed to already be in HU (see ct_volume.rescale_map)")
        slope, intercept = float(d["rescale_slope"]), float(d["rescale_intercept"])
        for key in self.keys:
            d[key] = d[key] * slope + intercept
        return d


class RasAffined(MapTransform):
    """Replace the NIfTI's placeholder affine with the TRUE one: `ras_affine(sx, sy, sz)`.

    The paired files carry an identity affine and pixdim (1,1,1), but the arrays are stored
    in LPS index order -- axis0 grows toward the patient's LEFT, axis1 toward POSTERIOR,
    axis2 toward SUPERIOR -- with real anisotropic spacing that exists only in the CT-RATE
    join (`ct_volume.spacing_map`). Measured on the real val split: 83 "left ..." findings
    have axis0 centroid 0.694 vs 112 "right ..." at 0.312, and all 28 volumes carrying both
    a left and a right finding agree (see tests/test_orientation.py). Corroborated by DICOM
    ImageOrientationPatient == [1,0,0,0,1,0] on all 3492 volumes.

    With the identity affine in place, `Orientationd(axcodes="RAS")` is a NO-OP -- MONAI
    already believes the data is RAS -- so the tensor is LPS content wearing a RAS label,
    with left/right and anterior/posterior mirrored. Setting the truthful affine makes that
    Orientationd do the flip; the flip is never hand-rolled here.

    Missing sx/sy/sz/patient_position RAISES, exactly like HuRescaled does for the rescale:
    a defaulted 1.0 spacing -- or a defaulted "HFS" -- is the same silent-wrong-value bug,
    in the spatial axis.
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.n_z_flipped = 0           # counted, never silent (see SLICE_AXIS_SIGN)

    def __call__(self, data):
        d = dict(data)
        name = d.get("name")
        missing = [k for k in ("sx", "sy", "sz", "patient_position") if d.get(k) is None]
        if missing:
            raise KeyError(f"{missing} missing for {name!r}: the paired NIfTI headers all "
                           f"report pixdim (1,1,1) and carry no PatientPosition, so real "
                           f"spacing/orientation exists only in the CT-RATE join (see "
                           f"ct_volume.spacing_map and volume_labels.csv)")
        position = str(d["patient_position"])
        if position not in SLICE_AXIS_SIGN:
            raise ValueError(
                f"{name}: PatientPosition {position!r} has no measured slice-axis "
                f"direction (known: {sorted(SLICE_AXIS_SIGN)}); refusing to assume HFS")
        affine = np.asarray(ras_affine(float(d["sx"]), float(d["sy"]), float(d["sz"])),
                            dtype=np.float64).copy()
        if SLICE_AXIS_SIGN[position] < 0:
            affine[2, 2] = -affine[2, 2]     # feet-first: axis2 grows toward INFERIOR
            self.n_z_flipped += 1
            log.warning("%s: PatientPosition == %s, slice axis negated", name, position)
        aff = torch.as_tensor(affine)
        for key in self.key_iterator(d):
            src = d[key]
            if not isinstance(src, MetaTensor):
                raise TypeError(f"{name}/{key}: expected a MetaTensor carrying an affine, "
                                f"got {type(src).__name__}")
            # HARD GATE: we may only overwrite a placeholder. A file that ships REAL
            # geometry would make the CT-RATE spacing join the wrong source, and this
            # transform would silently replace correct direction cosines with an assumption.
            # The WHOLE 4x4 is checked, translation included: ras_affine() has no origin,
            # so a file carrying a real one (e.g. ImagePositionPatient) would have it
            # silently zeroed by the overwrite below -- the same class of bug, in the
            # translation column. Every paired volume checked has translation (0,0,0).
            existing = np.asarray(src.affine, dtype=np.float64)
            if not np.allclose(existing, np.eye(4)):
                raise ValueError(
                    f"{name}/{key}: NIfTI affine is not the identity placeholder this "
                    f"pipeline is allowed to overwrite: {existing.tolist()}")
            src.affine = aff
        return d
