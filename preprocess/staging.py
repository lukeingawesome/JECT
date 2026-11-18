"""Step 1: collect scattered NIfTI files into the canonical dataset layout.

CT volumes, TotalSegmentator masks and ReXGrounding masks usually arrive from
different places (raw download, segmentation job, grounding export). Staging
puts them under one dataset root as ``original/``, ``totalseg/`` and
``rexgrounding/`` so every later step only needs a ``--data-root``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .layout import CT, MODALITIES, MODALITIES_BY_KEY, NIFTI_SUFFIXES, DatasetLayout, case_id_from_path

LINK_MODES = ("symlink", "hardlink", "copy")


def collect_sources(roots: Iterable[Path]) -> Dict[str, Path]:
    """Map ``case_id -> path`` for the NIfTI files under the given roots."""
    found: Dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Source path does not exist: {root}")

        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                continue
            case_id = case_id_from_path(path)
            previous = found.get(case_id)
            if previous is not None and previous != path:
                raise ValueError(
                    f"Case id '{case_id}' found twice in the sources: {previous} and {path}."
                )
            found[case_id] = path
    return found


def _place(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(os.path.realpath(source), destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown link mode '{mode}'; expected one of {LINK_MODES}.")


def stage_dataset(
    layout: DatasetLayout,
    sources: Mapping[str, Sequence[Path]],
    link_mode: str = "symlink",
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Place source volumes into ``layout``, one row per file in the report.

    Args:
        layout: destination dataset layout.
        sources: modality key -> source files/directories to pull in. Modalities
            absent from the mapping are left untouched.
        link_mode: how files are placed: ``symlink`` (default, no extra disk),
            ``hardlink`` or ``copy``.
        overwrite: replace destinations that already exist.
        dry_run: report the planned actions without touching the filesystem.

    Returns:
        DataFrame with columns ``modality, case_id, source, destination, action``.
    """
    if link_mode not in LINK_MODES:
        raise ValueError(f"Unknown link mode '{link_mode}'; expected one of {LINK_MODES}.")
    unknown = set(sources) - set(MODALITIES_BY_KEY)
    if unknown:
        raise ValueError(f"Unknown modality keys: {sorted(unknown)}")

    if not dry_run:
        layout.mkdirs()

    rows: List[Dict[str, str]] = []
    for modality in MODALITIES:
        roots = sources.get(modality.key)
        if not roots:
            continue

        for case_id, source in sorted(collect_sources(roots).items()):
            suffix = next(s for s in NIFTI_SUFFIXES if source.name.endswith(s))
            destination = layout.path_of(modality.key, case_id, suffix)

            exists = destination.exists() or destination.is_symlink()
            if exists and not overwrite:
                action = "exists"
            elif dry_run:
                action = "planned"
            else:
                if exists:
                    destination.unlink()
                _place(source, destination, link_mode)
                action = "staged"

            rows.append(
                {
                    "modality": modality.key,
                    "case_id": case_id,
                    "source": str(source),
                    "destination": str(destination),
                    "action": action,
                }
            )

    return pd.DataFrame(rows, columns=["modality", "case_id", "source", "destination", "action"])


def coverage(layout: DatasetLayout) -> pd.DataFrame:
    """Per-case availability of each modality in a staged dataset."""
    cases = layout.discover()
    rows = [
        {"case_id": case.case_id, **{m.key: case.has(m.key) for m in MODALITIES}}
        for case in cases
    ]
    return pd.DataFrame(rows, columns=["case_id", *(m.key for m in MODALITIES)])


def summarize_coverage(table: pd.DataFrame) -> str:
    """One-line-per-modality summary of :func:`coverage`."""
    total = len(table)
    lines = [f"{total} case(s) with a {CT.label} under the dataset root"]
    for modality in MODALITIES:
        if modality.key == CT.key or modality.key not in table:
            continue
        present = int(table[modality.key].sum())
        flag = "" if present == total else f"  <- {total - present} missing"
        lines.append(f"  {modality.label:<28} {present}/{total}{flag}")
    return "\n".join(lines)
