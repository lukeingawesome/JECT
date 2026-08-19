"""Typed loader for `configs/*.yaml`.

CLAUDE.md asks that a new experiment be runnable by changing YAML only. That only holds if
the YAML is the ONLY place the values live, so this loader fails loud on an unknown key
rather than ignoring it: a silently dropped `spatial_size` would run the whole experiment at
whatever the code happened to default to, which is the failure this config exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

_SECTIONS = {
    "paths": {"paired_root", "seg_root", "npy_root", "cache2mm_root"},
    "preprocess": {"spatial_size", "iso_spacing", "hu_window", "orientation",
                   "target_spacing", "cube_size"},
    "selection": {"split_column", "split", "fold", "label_source"},
}


class ConfigError(ValueError):
    """Raised for anything the config says that the code cannot honour."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    paired_root: str
    seg_root: str
    npy_root: str
    cache2mm_root: str | None
    spatial_size: tuple[int, int, int] | None
    target_spacing: tuple[float, float, float] | None
    cube_size: tuple[int, int, int] | None
    iso_spacing: float | None
    hu_window: tuple[float, float] | None
    orientation: str
    split_column: str
    split: str
    fold: int | None
    label_source: str

    @property
    def cache2mm_ct_dir(self) -> Path:
        if self.cache2mm_root is None:
            raise ConfigError("cache2mm_root is null in this config")
        return Path(self.cache2mm_root) / "ct"

    @property
    def cache2mm_seg_dir(self) -> Path:
        if self.cache2mm_root is None:
            raise ConfigError("cache2mm_root is null in this config")
        return Path(self.cache2mm_root) / "seg"

    @property
    def npy_ct_dir(self) -> Path:
        return Path(self.npy_root) / "ct"

    @property
    def npy_seg_dir(self) -> Path:
        return Path(self.npy_root) / "seg"


def _require(mapping: dict[str, Any], section: str) -> dict[str, Any]:
    got = mapping.get(section)
    if not isinstance(got, dict):
        raise ConfigError(f"config section {section!r} missing or not a mapping")
    unknown = set(got) - _SECTIONS[section]
    missing = _SECTIONS[section] - set(got)
    if unknown or missing:
        raise ConfigError(
            f"section {section!r}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return got


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Read a data config. `path` defaults to `configs/rexg_native.yaml`."""
    import yaml   # deferred: keeps this module importable without PyYAML

    path = Path(path) if path is not None else CONFIG_DIR / "rexg_native.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"data config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ConfigError(f"{path}: unknown top-level sections {sorted(unknown_sections)}")

    paths, pre, sel = (_require(raw, s) for s in ("paths", "preprocess", "selection"))

    def _triple(key, cast, section=pre):
        """None stays None; anything else must be exactly three positive values.
        A config that reaches training with a two-element cube_size is a crash at step 1,
        so it is rejected here where the message can name the key."""
        raw = section[key]
        if raw is None:
            return None
        got = tuple(cast(v) for v in raw)
        if len(got) != 3 or any(v <= 0 for v in got):
            raise ConfigError(f"{key} must be three positive values or null, got {raw!r}")
        return got

    size = _triple("spatial_size", int)
    target_spacing = _triple("target_spacing", float)
    cube_size = _triple("cube_size", int)
    # Three paths, not two. Both null is the NATIVE path: no resample, no fixed cube, patches
    # cut in voxels straight off the native-grid npy cache -- which is what VoxTell does, and
    # the only path this folder ships. Requiring one of the other two forced a native arm to
    # name a config whose preprocess block it does not read, which is how a 60k-step run came
    # to record a 2 mm isotropic resample that never happened. Any crop/resize consumer
    # rejects a null cube_size on its own, so this check was the duplicate, not the guard.
    if size is not None and cube_size is not None:
        raise ConfigError("spatial_size (resize path) and cube_size (crop path) are both set; "
                          "a config is one path or the other, or null/null for the native path")
    # Nullable for the native path: VoxTell normalises per volume (z-score over the patch's
    # own volume stats), so a window here would be a number nothing reads. Every consumer that
    # windows already dereferences it and would raise on None.
    if pre["hu_window"] is None:
        window = None
    else:
        window = tuple(float(v) for v in pre["hu_window"])
        if len(window) != 2 or window[0] >= window[1]:
            raise ConfigError(f"hu_window must be [lo, hi] with lo < hi, got {pre['hu_window']!r}")
    iso = pre["iso_spacing"]
    if iso is not None and float(iso) <= 0:
        raise ConfigError(f"iso_spacing must be positive or null, got {iso!r}")
    if sel["split_column"] not in ("rexg_split", "miccai_split"):
        raise ConfigError(f"split_column must be rexg_split or miccai_split, got {sel['split_column']!r}")
    if sel["label_source"] not in ("ctrate", "rexg"):
        raise ConfigError(f"label_source must be ctrate or rexg, got {sel['label_source']!r}")

    return DataConfig(
        paired_root=str(paths["paired_root"]), seg_root=str(paths["seg_root"]),
        npy_root=str(paths["npy_root"]),
        cache2mm_root=None if paths["cache2mm_root"] is None else str(paths["cache2mm_root"]),
        spatial_size=size, target_spacing=target_spacing, cube_size=cube_size,
        iso_spacing=None if iso is None else float(iso),
        hu_window=window, orientation=str(pre["orientation"]),
        split_column=str(sel["split_column"]), split=str(sel["split"]),
        fold=None if sel["fold"] is None else int(sel["fold"]),
        label_source=str(sel["label_source"]),
    )
