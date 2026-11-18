"""Cropping maths, free of MONAI/torch so it can be unit tested on its own."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def crop_starts(
    spatial_shape: Sequence[int],
    roi_size: Sequence[int],
    bbox_min: Sequence[int],
    bbox_max: Sequence[int],
) -> Tuple[int, ...]:
    """Start index per axis of an ``roi_size`` crop placed around a bounding box.

    The crop contains the whole bounding box when it fits, otherwise it is
    centred on the bounding box centre. Starts are clamped so the crop always
    stays inside the volume; axes smaller than the ROI start at 0 (the caller is
    expected to have padded them already).
    """
    if not len(spatial_shape) == len(roi_size) == len(bbox_min) == len(bbox_max):
        raise ValueError(
            f"Rank mismatch: shape={tuple(spatial_shape)} roi={tuple(roi_size)} "
            f"bbox={tuple(bbox_min)}..{tuple(bbox_max)}"
        )

    starts = []
    for axis, (size, roi) in enumerate(zip(spatial_shape, roi_size)):
        if size <= roi:
            starts.append(0)
            continue

        low, high = int(bbox_min[axis]), int(bbox_max[axis])
        extent = high - low + 1
        if extent >= roi:
            start = (low + high) // 2 - roi // 2
        else:
            start = low - (roi - extent) // 2
        starts.append(int(np.clip(start, 0, size - roi)))

    return tuple(starts)


def to_channel_depth_first(array: np.ndarray) -> np.ndarray:
    """Convert a channel-first ``(C, H, W, D)`` array to ``(C, D, H, W)``."""
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D (C, H, W, D) array, got shape={array.shape}.")
    return np.ascontiguousarray(np.transpose(array, (0, 3, 1, 2)))
