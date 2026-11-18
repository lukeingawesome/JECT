"""Canonical on-disk layout of a chest-CT dataset.

A dataset root holds one directory per modality, each containing NIfTI volumes
named after the case id::

    <root>/original/<case_id>.nii.gz       # chest CT
    <root>/totalseg/<case_id>.nii.gz       # TotalSegmentator label map
    <root>/rexgrounding/<case_id>.nii.gz   # ReXGrounding phrase masks (optional)

Every other module addresses the data through :class:`DatasetLayout`, so the
directory names live here and nowhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

NIFTI_SUFFIXES: Tuple[str, ...] = (".nii.gz", ".nii")


@dataclass(frozen=True)
class Modality:
    """One volume type of a case."""

    key: str  # transform-dict key and array name inside the npz
    dirname: str  # default sub-directory of the dataset root
    required: bool  # can a case without it still be preprocessed?
    label: str  # human readable, for CLI help and logs
    flag: str  # CLI flag stem, e.g. --grounding / --grounding-dirname


CT = Modality(key="ct", dirname="original", required=True, label="chest CT", flag="ct")
TOTALSEG = Modality(
    key="totalseg",
    dirname="totalseg",
    required=True,
    label="TotalSegmentator mask",
    flag="totalseg",
)
GROUNDING = Modality(
    key="rex",
    dirname="rexgrounding",
    required=False,
    label="ReXGrounding mask",
    flag="grounding",
)

MODALITIES: Tuple[Modality, ...] = (CT, TOTALSEG, GROUNDING)
MODALITIES_BY_KEY: Mapping[str, Modality] = {m.key: m for m in MODALITIES}


def case_id_from_path(path: Path) -> str:
    """Strip NIfTI suffixes from a filename to recover the case id."""
    name = path.name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def index_nifti_dir(directory: Path) -> Dict[str, Path]:
    """Map ``case_id -> path`` for every NIfTI file under ``directory``.

    Sub-directories are searched recursively, which keeps split-per-folder
    sources working; case ids must still be unique within the modality.
    """
    index: Dict[str, Path] = {}
    if not directory.is_dir():
        return index

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
            continue
        case_id = case_id_from_path(path)
        previous = index.get(case_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate case id '{case_id}' in {directory}: {previous} and {path}."
            )
        index[case_id] = path
    return index


@dataclass(frozen=True)
class Case:
    """A single case with the volumes that actually exist on disk."""

    case_id: str
    paths: Mapping[str, Path]

    def has(self, key: str) -> bool:
        return key in self.paths

    def missing_required(self) -> Tuple[str, ...]:
        return tuple(m.key for m in MODALITIES if m.required and m.key not in self.paths)


@dataclass
class DatasetLayout:
    """Resolves modality directories under a dataset root."""

    root: Path
    dirnames: Dict[str, str] = field(default_factory=lambda: {m.key: m.dirname for m in MODALITIES})

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        unknown = set(self.dirnames) - set(MODALITIES_BY_KEY)
        if unknown:
            raise ValueError(f"Unknown modality keys in dirnames: {sorted(unknown)}")
        for modality in MODALITIES:
            self.dirnames.setdefault(modality.key, modality.dirname)

    def dir_of(self, key: str) -> Path:
        return self.root / self.dirnames[key]

    def path_of(self, key: str, case_id: str, suffix: str = NIFTI_SUFFIXES[0]) -> Path:
        return self.dir_of(key) / f"{case_id}{suffix}"

    def mkdirs(self) -> None:
        for modality in MODALITIES:
            self.dir_of(modality.key).mkdir(parents=True, exist_ok=True)

    def discover(self, case_ids: Optional[Iterable[str]] = None) -> List[Case]:
        """Cases driven by the CT directory, with whatever masks exist alongside.

        Ids in ``case_ids`` that have no CT volume are reported by
        :meth:`missing_case_ids` rather than silently dropped.
        """
        indices = {m.key: index_nifti_dir(self.dir_of(m.key)) for m in MODALITIES}
        wanted = set(case_ids) if case_ids is not None else None

        cases: List[Case] = []
        for case_id in sorted(indices[CT.key]):
            if wanted is not None and case_id not in wanted:
                continue
            paths = {key: index[case_id] for key, index in indices.items() if case_id in index}
            cases.append(Case(case_id=case_id, paths=paths))
        return cases

    def missing_case_ids(self, case_ids: Iterable[str]) -> List[str]:
        """Requested ids that have no CT volume."""
        available = set(index_nifti_dir(self.dir_of(CT.key)))
        return sorted(set(case_ids) - available)


def load_case_ids(path: Path, splits: Optional[Sequence[str]] = None) -> List[str]:
    """Read a case-id whitelist from a JSON, CSV manifest or text file.

    Args:
        path: ``.json`` holding a list of ids or a mapping of split name to ids;
            ``.csv`` with a ``case_id`` or ``volume`` column (the manifest
            written by ``cli.py select``); any other extension is read as
            one id per line.
        splits: when the JSON is a mapping, restrict to these splits (default:
            union of all splits).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        import csv

        if splits:
            raise ValueError(f"--splits given but {path} is a manifest, which has no splits.")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        column = next((c for c in ("case_id", "volume") if c in rows[0]), None)
        if column is None:
            raise ValueError(f"{path} has no 'case_id' or 'volume' column.")
        return sorted({case_id_from_path(Path(row[column])) for row in rows if row[column]})

    if suffix != ".json":
        with path.open() as handle:
            return [line.strip() for line in handle if line.strip()]

    with path.open() as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        selected = list(splits) if splits else sorted(data)
        unknown = [split for split in selected if split not in data]
        if unknown:
            raise ValueError(f"Splits {unknown} not in {path}; available: {sorted(data)}")
        ids = {case_id for split in selected for case_id in data[split]}
        return sorted(ids)

    if splits:
        raise ValueError(f"--splits given but {path} holds a flat list of ids.")
    return sorted(set(data))
