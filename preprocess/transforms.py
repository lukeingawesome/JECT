"""Custom MONAI transforms used by the chest-CT preprocessing pipeline."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

from .geometry import crop_starts

GEOMETRY_META_KEYS: Tuple[str, ...] = ("affine", "original_affine", "pixdim")


class NoLungLabelsError(RuntimeError):
    """Raised when the TotalSegmentator mask contains none of the lung labels."""


def _meta_of(data: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Metadata of ``key``, from its sidecar dict or from the MetaTensor."""
    sidecar = data.get(f"{key}_meta_dict")
    if sidecar:
        return dict(sidecar)
    return dict(getattr(data.get(key), "meta", {}) or {})


def _copy(value: Any) -> Any:
    """Copy arrays/tensors so masks never share geometry buffers with the CT."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _spatial_shape(volume) -> Tuple[int, ...]:
    if volume.ndim == 4:
        return tuple(int(s) for s in volume.shape[1:])
    if volume.ndim == 3:
        return tuple(int(s) for s in volume.shape)
    raise ValueError(f"Expected a 3D or 4D volume, got shape={tuple(volume.shape)}.")


class MatchReferenceGeometryd(MapTransform):
    """Make multi-channel masks channel-first and give them the reference geometry.

    ReXGrounding masks stack one binary channel per phrase and are stored either
    as ``(F, H, W, D)`` or ``(H, W, D, F)``, often without a usable affine. This
    normalises the layout and copies the CT's affine/pixdim onto them, so the
    later ``Orientationd``/``Spacingd`` resample every key in the same physical
    space.
    """

    def __init__(self, keys: KeysCollection, ref_key: str, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.ref_key = ref_key

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        reference = d.get(self.ref_key)
        if reference is None:
            raise KeyError(f"Reference key '{self.ref_key}' is missing.")
        ref_spatial = _spatial_shape(reference)
        ref_meta = _meta_of(d, self.ref_key)
        geometry = {k: ref_meta[k] for k in GEOMETRY_META_KEYS if ref_meta.get(k) is not None}
        if "original_affine" not in geometry and "affine" in geometry:
            geometry["original_affine"] = geometry["affine"]

        for key in self.key_iterator(d):
            volume = d[key]
            if volume is None:
                continue
            if not isinstance(volume, MetaTensor):
                volume = MetaTensor(torch.as_tensor(np.asarray(volume)))

            if volume.ndim == 3:
                volume = volume.unsqueeze(0)
            elif volume.ndim == 4 and tuple(volume.shape[:3]) == ref_spatial:
                volume = torch.movedim(volume, -1, 0)  # (H, W, D, F) -> (F, H, W, D)
            elif volume.ndim != 4 or tuple(volume.shape[1:]) != ref_spatial:
                raise ValueError(
                    f"[{key}] shape {tuple(volume.shape)} matches the reference spatial "
                    f"shape {ref_spatial} neither as (F, H, W, D) nor as (H, W, D, F)."
                )

            for name, value in geometry.items():
                volume.meta[name] = _copy(value)
            volume.meta["spatial_shape"] = ref_spatial
            volume.meta["original_channel_dim"] = 0
            d[key] = volume

        return d


class LungBBoxCropd(MapTransform):
    """Crop every key to ``roi_size`` around the lung bounding box.

    The bounding box comes from the TotalSegmentator lung labels, which keeps
    the lungs inside the volume instead of trusting the scan to be centred (a
    plain centre crop drifts into the abdomen on whole-body series).
    """

    def __init__(
        self,
        keys: KeysCollection,
        seg_key: str,
        roi_size: Sequence[int],
        lung_labels: Sequence[int],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.seg_key = seg_key
        self.roi_size = tuple(int(s) for s in roi_size)
        self.lung_labels = tuple(int(v) for v in lung_labels)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        d = dict(data)

        segmentation = d.get(self.seg_key)
        if segmentation is None:
            raise NoLungLabelsError(
                f"No segmentation under key '{self.seg_key}'; cannot crop around the lungs."
            )

        seg = np.asarray(segmentation)
        seg = seg[0] if seg.ndim == 4 else seg
        lungs = np.isin(seg, self.lung_labels)
        if not lungs.any():
            raise NoLungLabelsError(
                f"None of the lung labels {list(self.lung_labels)} are present; "
                f"labels found: {np.unique(seg).tolist()}"
            )

        coords = np.argwhere(lungs)
        starts = crop_starts(seg.shape, self.roi_size, coords.min(axis=0), coords.max(axis=0))
        slices = (slice(None),) + tuple(slice(s, s + r) for s, r in zip(starts, self.roi_size))

        for key in self.key_iterator(d):
            if d[key] is not None:
                d[key] = d[key][slices]
        return d
