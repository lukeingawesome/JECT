"""Step 2: turn one staged case into a single npz file.

Pipeline, in order: load -> channel-first -> reorient -> resample to the target
spacing -> optional HU windowing -> pad -> crop around the lungs -> cast -> save.
The npz holds ``ct`` plus whichever masks the case has, all in ``(C, D, H, W)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)

from .settings import PreprocessConfig
from .geometry import to_channel_depth_first
from .layout import CT, GROUNDING, MODALITIES, TOTALSEG, Case
from .transforms import LungBBoxCropd, MatchReferenceGeometryd, NoLungLabelsError

ALL_KEYS: Tuple[str, ...] = tuple(m.key for m in MODALITIES)
MASK_KEYS: Tuple[str, ...] = (TOTALSEG.key, GROUNDING.key)


def build_transforms(config: PreprocessConfig) -> Compose:
    """Assemble the MONAI pipeline described by ``config``."""
    steps = [
        LoadImaged(keys=ALL_KEYS, reader="NibabelReader", allow_missing_keys=True),
        EnsureChannelFirstd(keys=(CT.key, TOTALSEG.key), allow_missing_keys=True),
        # Grounding masks are multi-channel and carry no reliable affine.
        MatchReferenceGeometryd(keys=(GROUNDING.key,), ref_key=CT.key, allow_missing_keys=True),
        Orientationd(
            keys=ALL_KEYS,
            axcodes=config.orientation,
            # Pinned so the axis semantics stay the same across MONAI versions.
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
            allow_missing_keys=True,
        ),
        Spacingd(
            keys=(CT.key,),
            pixdim=config.target_spacing,
            mode="trilinear",
            align_corners=True,
        ),
        Spacingd(
            keys=MASK_KEYS,
            pixdim=config.target_spacing,
            mode="nearest",
            align_corners=True,
            allow_missing_keys=True,
        ),
    ]

    if config.hu_window is not None:
        steps.append(
            ScaleIntensityRanged(
                keys=(CT.key,),
                a_min=config.hu_window[0],
                a_max=config.hu_window[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )

    steps += [
        # Pad first so the crop below always has at least the target size.
        SpatialPadd(
            keys=(CT.key,),
            spatial_size=config.target_shape,
            constant_values=config.resolved_image_pad_value,
        ),
        SpatialPadd(
            keys=MASK_KEYS,
            spatial_size=config.target_shape,
            constant_values=config.label_pad_value,
            allow_missing_keys=True,
        ),
        LungBBoxCropd(
            keys=ALL_KEYS,
            seg_key=TOTALSEG.key,
            roi_size=config.target_shape,
            lung_labels=config.lung_labels,
            allow_missing_keys=True,
        ),
        CastToTyped(keys=MASK_KEYS, dtype=config.label_np_dtype, allow_missing_keys=True),
    ]
    return Compose(steps)


def _causes(error: BaseException):
    """The exception and everything it was raised from.

    MONAI's ``Compose`` re-raises transform failures as ``RuntimeError``, so the
    interesting exception type is only reachable through the cause chain.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def _to_storage(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Round/clip to the representable range before casting to an integer dtype."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        array = np.clip(np.round(array), info.min, info.max)
    return to_channel_depth_first(np.asarray(array)).astype(dtype, copy=False)


def _blank_record(case: Case, out_file: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {"case_id": case.case_id, "status": None, "out_npz": str(out_file)}
    record.update({f"{m.key}_path": str(case.paths.get(m.key, "")) for m in MODALITIES})
    record.update({"source_shape": None, "source_spacing": None})
    record.update({f"{m.key}_shape": None for m in MODALITIES})
    return record


def process_case(
    case: Case,
    out_dir: Path,
    config: PreprocessConfig,
    transforms: Optional[Compose] = None,
) -> Dict[str, Any]:
    """Preprocess one case and write ``<out_dir>/<case_id>.npz``.

    Returns a metadata record with the resulting status; failures are reported
    in ``status`` instead of raising, so one bad case cannot stop a large run.
    """
    transforms = transforms or build_transforms(config)
    out_file = Path(out_dir) / f"{case.case_id}.npz"
    record = _blank_record(case, out_file)

    if out_file.exists() and not config.overwrite:
        record["status"] = "skipped_exists"
        return record

    missing = case.missing_required()
    if missing:
        record["status"] = f"skipped_missing:{','.join(missing)}"
        return record

    try:
        image = nib.load(str(case.paths[CT.key]))
        source_shape = tuple(int(s) for s in image.shape[:3])
        record["source_shape"] = source_shape
        record["source_spacing"] = tuple(float(z) for z in image.header.get_zooms()[:3])
    except Exception as error:  # noqa: BLE001 - reported per case
        record["status"] = f"error_load:{type(error).__name__}: {error}"
        return record

    if any(size < minimum for size, minimum in zip(source_shape, config.min_source_shape)):
        record["status"] = f"skipped_too_small:{source_shape}"
        return record

    data = {key: str(path) for key, path in case.paths.items()}
    try:
        transformed = transforms(data)

        arrays = {CT.key: _to_storage(transformed[CT.key], config.image_np_dtype)}
        for key in MASK_KEYS:
            if key in transformed:
                arrays[key] = _to_storage(transformed[key], config.label_np_dtype)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if config.compress else np.savez
        save(out_file, **arrays)

        for key, array in arrays.items():
            record[f"{key}_shape"] = tuple(array.shape)
        record["status"] = "processed"
    except Exception as error:  # noqa: BLE001 - reported per case
        no_lungs = next((c for c in _causes(error) if isinstance(c, NoLungLabelsError)), None)
        if no_lungs is not None:
            record["status"] = f"error_no_lungs:{no_lungs}"
        else:
            record["status"] = f"error_transform:{type(error).__name__}: {error}"

    return record
