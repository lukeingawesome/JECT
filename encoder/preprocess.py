"""Input preprocessing for the JECT encoder.

Input contract used during pretraining (features degrade without warning otherwise):

  1. Orientation LPS, axes ordered (D, H, W) = (R-L, A-P, I-S), as produced by MONAI's
     ``Orientationd(axcodes="LPS")``.
  2. Isotropic resampling, then a lung-centred crop/pad to a fixed cube of raw HU
     (int16, padded with -1024 HU):
        * 1.5 mm spacing, 256^3 cube: input for the 224 checkpoint
        * 1.0 mm spacing, 384^3 cube: input for the 336 checkpoint
     Both give a ~336 mm physical field of view at the model input.
  3. Centre crop / pad (with -1000 HU) from that cube to the model's input size.
  4. HU window, then linear scale to [0, 1]. The default is the general window
     (id 2, centre 0, width 2000, i.e. [-1000, 1000]). Pass the same id to the encoder
     as ``window_id``; it selects an additive token embedding.

``preprocess_hu_volume`` performs steps 3-4 on an already-resampled cube and is the
usual entry point. ``resample_to_spacing`` covers the resampling in step 2 for raw
NIfTI input; it is a plain trilinear resize without the lung-bounding-box crop, so
features differ slightly from those of the original pipeline.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

try:   # importable as a package module or as a plain script in this directory
    from .model import CT_TYPE_MAP, DEFAULT_CT_TYPE, GENERAL_WINDOW_ID, WINDOW_PARAMS
except ImportError:
    from model import CT_TYPE_MAP, DEFAULT_CT_TYPE, GENERAL_WINDOW_ID, WINDOW_PARAMS

Triple = Tuple[int, int, int]

AIR_HU = -1000.0     # pad value for the centre crop/pad


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def apply_window(volume_hu: np.ndarray, window_id: int = GENERAL_WINDOW_ID) -> np.ndarray:
    """Clamp raw HU to a window and rescale to [0, 1]."""
    if window_id not in WINDOW_PARAMS:
        raise ValueError(f"window_id must be one of {sorted(WINDOW_PARAMS)}, got {window_id}")
    p = WINDOW_PARAMS[window_id]
    low = p["center"] - p["width"] / 2.0
    high = p["center"] + p["width"] / 2.0
    out = (volume_hu.astype(np.float32) - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def ct_type_to_id(ct_type) -> int:
    """Map a ct_type string ('non-contrast' | 'contrast' | 'ldct') to its embedding index."""
    if ct_type is None:
        return DEFAULT_CT_TYPE
    if isinstance(ct_type, (int, np.integer)):
        return int(ct_type)
    return CT_TYPE_MAP.get(str(ct_type).strip().lower(), DEFAULT_CT_TYPE)


# ---------------------------------------------------------------------------
# Crop / pad
# ---------------------------------------------------------------------------
def make_crop_pad_transform(src_shape: Triple, dst_shape: Triple, random_crop: bool = False):
    """Slices + pads that take ``src_shape`` to exactly ``dst_shape`` (centre crop by default)."""
    slices, pads = [], []
    for src, dst in zip(src_shape, dst_shape):
        if src >= dst:
            start = (src - dst) // 2
            if random_crop and src > dst:
                start = int(np.random.randint(0, src - dst + 1))
            slices.append(slice(start, start + dst))
            pads.append((0, 0))
        else:
            slices.append(slice(0, src))
            before = (dst - src) // 2
            pads.append((before, dst - src - before))
    return tuple(slices), tuple(pads)


def apply_crop_pad_3d(arr: np.ndarray, slices, pads, constant: float = AIR_HU) -> np.ndarray:
    """Apply crop/pad to the last three dims of ``arr``."""
    cropped = arr[(...,) + tuple(slices)]
    if any(a > 0 or b > 0 for a, b in pads):
        pad_width = [(0, 0)] * (cropped.ndim - 3) + list(pads)
        cropped = np.pad(cropped, pad_width=pad_width, mode="constant", constant_values=constant)
    return cropped


def center_crop_pad(volume_hu: np.ndarray, target: Triple) -> np.ndarray:
    """Centre crop/pad a raw-HU volume (last three dims) to ``target``, padding with air."""
    slices, pads = make_crop_pad_transform(tuple(volume_hu.shape[-3:]), tuple(target))
    return apply_crop_pad_3d(volume_hu, slices, pads, constant=AIR_HU)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def preprocess_hu_volume(volume_hu: np.ndarray, input_size: Triple,
                         window_id: int = GENERAL_WINDOW_ID) -> torch.Tensor:
    """Convert a raw-HU cube to a model input tensor ``[1, D, H, W]`` in [0, 1].

    Crop/pad happens in HU space (padding with air) before windowing, so padded
    borders map to 0.0 in the general window.
    """
    vol = np.asarray(volume_hu)
    if vol.ndim == 4:
        if vol.shape[0] != 1:
            raise ValueError(f"expected a single-channel volume, got shape {vol.shape}")
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D HU volume, got shape {vol.shape}")
    vol = center_crop_pad(vol, tuple(input_size))
    return torch.from_numpy(apply_window(vol, window_id)).unsqueeze(0)


def load_npz_volume(path: str, input_size: Triple, key: str = "ct",
                    window_id: int = GENERAL_WINDOW_ID) -> torch.Tensor:
    """Load a preprocessed ``.npz`` (raw int16 HU under ``key``) into a model input tensor."""
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path = path + ".npz"
    with np.load(path) as d:
        if key not in d:
            raise KeyError(f"{path} has no '{key}' array (found {list(d.files)})")
        ct = d[key]
    return preprocess_hu_volume(ct, input_size, window_id=window_id)


def resample_to_spacing(volume_hu: np.ndarray, src_spacing: Sequence[float],
                        dst_spacing: float = 1.5) -> np.ndarray:
    """Trilinear-resample a raw-HU volume to isotropic ``dst_spacing`` mm.

    Convenience for raw NIfTI input. It does not apply the lung-bounding-box crop
    used in pretraining; see the module docstring.
    """
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D HU volume, got shape {vol.shape}")
    if len(src_spacing) != 3:
        raise ValueError(f"src_spacing must have 3 entries, got {src_spacing!r}")
    scale = [float(s) / float(dst_spacing) for s in src_spacing]
    out_shape = tuple(max(1, int(round(n * s))) for n, s in zip(vol.shape, scale))
    t = torch.from_numpy(vol)[None, None]
    t = torch.nn.functional.interpolate(t, size=out_shape, mode="trilinear", align_corners=True)
    return t[0, 0].numpy()


def stack_batch(volumes: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack [1, D, H, W] tensors into a [B, 1, D, H, W] batch."""
    if not volumes:
        raise ValueError("stack_batch got an empty list")
    return torch.stack([v if v.ndim == 4 else v.unsqueeze(0) for v in volumes], dim=0)
