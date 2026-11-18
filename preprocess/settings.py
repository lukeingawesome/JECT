"""Preprocessing configuration: settable from YAML or the CLI, saved with the run."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import yaml

AIR_HU = -1024.0
# TotalSegmentator v1/v2 lung lobes: upper/lower left, upper/middle/lower right.
LUNG_LOBE_LABELS: Tuple[int, ...] = (10, 11, 12, 13, 14)
ORIENTATION_CODES = frozenset("RLAPIS")


def _as_tuple(value: Any, length: int, cast, name: str) -> Tuple:
    items = tuple(cast(v) for v in value)
    if len(items) != length:
        raise ValueError(f"{name} must have {length} entries, got {items}.")
    return items


@dataclass(frozen=True)
class PreprocessConfig:
    """Resampling, windowing and cropping settings for one preprocessing run."""

    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm
    target_shape: Tuple[int, int, int] = (384, 384, 384)  # voxels, oriented axis order
    orientation: str = "LPS"
    hu_window: Optional[Tuple[float, float]] = None  # scaled to [0, 1]; None keeps raw HU
    image_dtype: str = "int16"
    label_dtype: str = "uint16"
    image_pad_value: Optional[float] = None  # None -> see resolved_image_pad_value
    label_pad_value: float = 0.0
    lung_labels: Tuple[int, ...] = LUNG_LOBE_LABELS  # their bounding box centres the crop
    # Localisers and partial series would come out as mostly padding, so skip them.
    min_source_shape: Tuple[int, int, int] = (200, 200, 100)
    compress: bool = True
    overwrite: bool = False
    num_workers: int = 8

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "target_spacing", _as_tuple(self.target_spacing, 3, float, "target_spacing"))
        set_(self, "target_shape", _as_tuple(self.target_shape, 3, int, "target_shape"))
        set_(self, "lung_labels", tuple(int(v) for v in self.lung_labels))
        set_(self, "min_source_shape", _as_tuple(self.min_source_shape, 3, int, "min_source_shape"))
        if self.hu_window is not None:
            set_(self, "hu_window", _as_tuple(self.hu_window, 2, float, "hu_window"))

        if any(s <= 0 for s in self.target_spacing):
            raise ValueError(f"target_spacing must be positive, got {self.target_spacing}.")
        if any(s <= 0 for s in self.target_shape):
            raise ValueError(f"target_shape must be positive, got {self.target_shape}.")
        if self.hu_window is not None and self.hu_window[0] >= self.hu_window[1]:
            raise ValueError(f"hu_window must be increasing, got {self.hu_window}.")
        if not self.lung_labels:
            raise ValueError("lung_labels must not be empty.")
        if len(self.orientation) != 3 or set(self.orientation.upper()) - ORIENTATION_CODES:
            raise ValueError(f"orientation must be 3 axis codes from RLAPIS, got '{self.orientation}'.")
        set_(self, "orientation", self.orientation.upper())
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}.")

        # Fail early on unusable dtype names rather than inside a worker.
        self.image_np_dtype
        self.label_np_dtype

    @property
    def image_np_dtype(self) -> np.dtype:
        return np.dtype(self.image_dtype)

    @property
    def label_np_dtype(self) -> np.dtype:
        return np.dtype(self.label_dtype)

    @property
    def resolved_image_pad_value(self) -> float:
        """Air HU for raw volumes; 0 for windowed ones, where -1024 would be out of range."""
        if self.image_pad_value is not None:
            return float(self.image_pad_value)
        return 0.0 if self.hu_window is not None else AIR_HU

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]] = None) -> "PreprocessConfig":
        values = dict(mapping or {})
        unknown = set(values) - set(cls.field_names())
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. Valid keys: {list(cls.field_names())}"
            )
        return cls(**values)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "PreprocessConfig":
        """YAML file plus overrides; None overrides fall back to YAML, then to defaults."""
        values: Dict[str, Any] = {}
        if path is not None:
            loaded = yaml.safe_load(Path(path).read_text()) or {}
            if not isinstance(loaded, Mapping):
                raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}.")
            values.update(loaded)
        values.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls.from_mapping(values)
