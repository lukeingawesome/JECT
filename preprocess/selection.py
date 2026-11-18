"""Step 1b: keep one volume per study.

CT-RATE names volumes ``{split}_{patient}_{study}_{reconstruction}.nii.gz`` and
reconstructs each scan with several kernels -- "25,692 non-contrast chest CT
volumes, expanded to 50,188 through various reconstructions". Preprocessing all
of them would train on several renderings of one acquisition, so exactly one
volume per study is kept.

Applied per study, in order:

1. prefer the native in-plane matrix (see :data:`NATIVE_MATRIX`);
2. prefer an axial series over a reformat;
3. rank what is left with ``rule``.

Both preferences fall back rather than empty a study, so nothing is dropped
unless a better sibling replaces it. One exception overrides all of it: a volume
carrying a ReXGrounding mask is kept even when a sibling wins its study, since
the mask annotates that specific reconstruction. Such a study contributes two
rows, told apart by the manifest's ``kept_for`` column.

Ranking ``soft512`` (default): contrast-phase series last -- CT-RATE is
non-contrast, so a delayed/arterial series would not match the report -- then
soft-tissue kernel over lung over bone, then the standard matrix, the thinnest
slice, the series closest to a chest-limited z-coverage, and finally the lowest
reconstruction index as a deterministic tie-break. ``recon1`` instead keeps the
``_1`` volume of every study.

NIfTI headers are parsed straight from the first 348 bytes of the gzip stream,
so scanning 50k volumes never decompresses a single image.
"""

from __future__ import annotations

import gzip
import re
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .layout import CT, GROUNDING, Case, DatasetLayout

RULES: Tuple[str, ...] = ("soft512", "recon1")

# DICOM convolution kernels seen in CT-RATE, grouped by reconstruction intent.
# Siemens Br36/Br40/Br44 and Philips A/B/SA/SB are soft-tissue (mediastinal
# window); Bl56/Bl57/Br60/L/E/EA/YA/YB/LungB are sharp lung kernels; D (KEMIK)
# and UB are bone kernels.
SOFT_KERNELS = frozenset({"A", "B", "Br36d", "Br40f", "Br44f", "SA", "SB"})
BONE_KERNELS = frozenset({"D", "UB"})

# Contrast phases ("15 DK GEC FAZ" = 15 min delayed phase) exist in a handful of
# studies; the report describes the non-contrast series.
CONTRAST_PHASE_RE = re.compile(
    r"GEC\s*FAZ|GECIKMIS|ARTERIYEL|ARTERIAL|VENOZ|VENOUS|PORTAL|KONTRASTLI", re.I
)

# Typical thoracic z-coverage (mm); a late tie-break preferring a chest-limited
# series over one extended to the pelvis / whole body.
TYPICAL_Z_EXTENT_MM = 350.0

AXIAL_IOP: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
IOP_TOLERANCE = 1e-3

# Default in-plane matrix, kept when a study offers a choice. CT-RATE ships
# 512, 768 and 1024 reconstructions; the larger matrices are the sharp-kernel,
# high in-plane resolution series, and they are also the anisotropic ones (in
# the validation split every 768/1024 volume has 1.5 mm slices, against a median
# of 1.0 mm at 512). Preferring 512 therefore trades in-plane detail for
# through-plane resolution and a smoother, mediastinal-window rendering. Set
# `native_matrix` to another size, or to 0, when a task wants the opposite.
NATIVE_MATRIX = 512

# CT-RATE volume naming: split_patient_study_reconstruction.
CASE_ID_RE = re.compile(r"^(?P<split>.+)_(?P<patient>\d+)_(?P<study>[A-Za-z]+)_(?P<recon>\d+)$")

# `kept_for` values: the volume representing its study, and the extra rows kept
# only because they carry a grounding mask.
KEPT_FOR_STUDY = "study_pick"
KEPT_FOR_GROUNDING = "grounding"

MANIFEST_COLUMNS: Tuple[str, ...] = (
    "case_id", "volume", "kept_for", "study_id", "split", "patient", "study", "recon",
    "nx", "ny", "nz", "dx", "dy", "dz", "z_extent_mm", "axial", "native_matrix",
    "iop", "kernel", "series_description", "manufacturer", "patient_sex",
    "patient_age", "study_date", "bytes", "n_recons_in_study", "dropped_siblings",
)

NIFTI_HEADER_BYTES = 348


# --------------------------------------------------------------------------- #
# NIfTI-1 header
# --------------------------------------------------------------------------- #

def read_nifti_header(path: Path) -> Dict[str, Any]:
    """Read ``dim``/``pixdim``/``srow`` from a NIfTI-1 header without nibabel."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        raw = handle.read(NIFTI_HEADER_BYTES)
    if len(raw) < NIFTI_HEADER_BYTES:
        raise ValueError("truncated header")

    endian = "<"
    (sizeof_hdr,) = struct.unpack(endian + "i", raw[0:4])
    if sizeof_hdr != NIFTI_HEADER_BYTES:
        endian = ">"
        (sizeof_hdr,) = struct.unpack(endian + "i", raw[0:4])
        if sizeof_hdr != NIFTI_HEADER_BYTES:
            raise ValueError("not a NIfTI-1 file")

    return {
        "dim": struct.unpack(endian + "8h", raw[40:56]),
        "pixdim": struct.unpack(endian + "8f", raw[76:108]),
        "srow": (
            struct.unpack(endian + "4f", raw[280:296]),
            struct.unpack(endian + "4f", raw[296:312]),
            struct.unpack(endian + "4f", raw[312:328]),
        ),
    }


def probe(item: Tuple[str, str]) -> Dict[str, Any]:
    """Header facts of one volume; failures are returned, never raised."""
    case_id, path_str = item
    path = Path(path_str)
    try:
        header = read_nifti_header(path)
    except Exception as error:  # noqa: BLE001 - reported per volume
        return {"case_id": case_id, "volume": path.name, "error": f"{type(error).__name__}: {error}"}

    dim, pixdim = header["dim"], header["pixdim"]
    row_x, row_y, row_z = header["srow"]
    # Off-diagonal terms of the direction matrix: non-zero means the volume is
    # stored obliquely / reformatted rather than axis-aligned axial.
    offdiag = max(abs(v) for v in (row_x[1], row_x[2], row_y[0], row_y[2], row_z[0], row_z[1]))

    return {
        "case_id": case_id,
        "volume": path.name,
        "nx": int(dim[1]),
        "ny": int(dim[2]),
        "nz": int(dim[3]),
        "dx": round(float(pixdim[1]), 5),
        "dy": round(float(pixdim[2]), 5),
        "dz": round(float(pixdim[3]), 5),
        "offdiag": round(float(offdiag), 6),
        "bytes": path.stat().st_size,
        "error": "",
    }


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def load_metadata(paths: Optional[Iterable[Path]]) -> Dict[str, Mapping[str, Any]]:
    """Index the official CT-RATE metadata CSVs by ``VolumeName``."""
    metadata: Dict[str, Mapping[str, Any]] = {}
    for path in paths or ():
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "VolumeName" not in table.columns:
            raise ValueError(f"{path} has no 'VolumeName' column; is it CT-RATE metadata?")
        metadata.update({row["VolumeName"]: row for row in table.to_dict("records")})
    return metadata


def parse_case_id(case_id: str) -> Tuple[str, str, str, int]:
    """``train_10000_a_2`` -> ``('train', '10000', 'a', 2)``.

    Ids that do not follow the CT-RATE convention become their own study with
    reconstruction 0, so non-CT-RATE datasets pass through unchanged.
    """
    match = CASE_ID_RE.match(case_id)
    if not match:
        return "", "", "", 0
    return match["split"], match["patient"], match["study"], int(match["recon"])


def study_id_of(case_id: str) -> str:
    split, patient, study, _ = parse_case_id(case_id)
    return f"{split}_{patient}_{study}" if split else case_id


def clean_kernel(raw: Any) -> str:
    """CT-RATE stores multi-valued kernels as ``"['Br40f', '3']"``; take the name."""
    text = str(raw).strip().strip("[]")
    return text.split(",")[0].strip().strip("'\"")


def parse_iop(raw: Any) -> Optional[List[float]]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        values = [float(v) for v in str(raw).strip("[]").split(",")]
    except ValueError:
        return None
    return values if len(values) == 6 else None


def is_axial(iop: Optional[Sequence[float]]) -> bool:
    """True when the slice plane is the patient axial plane."""
    if iop is None:
        return False
    return all(abs(a - b) <= IOP_TOLERANCE for a, b in zip(iop, AXIAL_IOP))


def kernel_tier(kernel: str) -> int:
    """2 = soft tissue / mediastinal, 1 = lung, 0 = bone. Higher wins."""
    if kernel in SOFT_KERNELS:
        return 2
    if kernel in BONE_KERNELS:
        return 0
    return 1


def prefer(candidates: Sequence[Mapping[str, Any]], predicate) -> Tuple[List, List]:
    """Split a study's candidates into ``(kept, dropped)`` by ``predicate``.

    Nothing is ever lost: when no candidate satisfies the predicate the study
    keeps everything it has, so a preference can never empty a study.
    """
    kept = [row for row in candidates if predicate(row)]
    if not kept:
        return list(candidates), []
    return kept, [row for row in candidates if not predicate(row)]


def is_native_matrix(row: Mapping[str, Any], native_matrix: Optional[int] = NATIVE_MATRIX) -> bool:
    """True when the volume is on the preferred in-plane matrix."""
    return not native_matrix or row["nx"] == native_matrix


def selection_key(row: Mapping[str, Any], rule: str) -> Tuple:
    """Ranking key for one candidate volume; the largest key wins the study."""
    if rule == "recon1":
        return (-row["recon"],)
    if rule != "soft512":
        raise ValueError(f"Unknown rule '{rule}'; expected one of {RULES}.")
    return (
        0 if CONTRAST_PHASE_RE.search(row["series_description"]) else 1,
        kernel_tier(row["kernel"]),
        1 if row["nx"] == 512 else 0,
        -row["dz"],
        -abs(row["z_extent_mm"] - TYPICAL_Z_EXTENT_MM),
        -row["recon"],
    )


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

@dataclass
class SelectionResult:
    """Outcome of a selection run."""

    selected: pd.DataFrame  # manifest rows, in MANIFEST_COLUMNS order
    rejected: pd.DataFrame  # everything else, with a `reason` column

    def case_ids(self) -> List[str]:
        return list(self.selected["case_id"])

    def study_picks(self) -> pd.DataFrame:
        """Only the one-volume-per-study rows, without the grounding extras."""
        if "kept_for" not in self.selected:
            return self.selected
        return self.selected[self.selected["kept_for"] == KEPT_FOR_STUDY]

    def summary(self) -> str:
        reasons = self.rejected["reason"].value_counts() if not self.rejected.empty else {}
        picks = len(self.study_picks())
        extra = len(self.selected) - picks
        headline = f"selected {len(self.selected)} volume(s): {picks} study pick(s)"
        if extra:
            headline += f" + {extra} kept for their grounding mask"
        lines = [
            headline,
            *[f"  rejected ({reason}): {count}" for reason, count in dict(reasons).items()],
        ]
        if "native_matrix" in self.selected and not self.selected.empty:
            fallback = int((~self.selected["native_matrix"].astype(bool)).sum())
            if fallback:
                lines.append(
                    f"  kept a non-native-matrix volume for {fallback} study(ies) that "
                    "offered nothing else"
                )
        for column in ("nx", "dz", "kernel"):
            if column in self.selected and not self.selected.empty:
                counts = self.selected[column].value_counts().head(5)
                spread = ", ".join(f"{value}: {count}" for value, count in counts.items())
                lines.append(f"  {column:<8} {spread}")
        if "bytes" in self.selected and not self.selected.empty:
            lines.append(f"  total    {self.selected['bytes'].sum() / 1e9:.1f} GB")
        return "\n".join(lines)


def annotate(rows: List[Dict[str, Any]], metadata: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Merge DICOM metadata into probed header rows."""
    for row in rows:
        split, patient, study, recon = parse_case_id(row["case_id"])
        meta = metadata.get(row["volume"], {})
        iop = parse_iop(meta.get("ImageOrientationPatient"))
        row.update(
            split=split,
            patient=patient,
            study=study,
            recon=recon,
            study_id=study_id_of(row["case_id"]),
            kernel=clean_kernel(meta.get("ConvolutionKernel", "")),
            series_description=str(meta.get("SeriesDescription") or "").strip(),
            manufacturer=str(meta.get("Manufacturer") or "").strip(),
            patient_sex=str(meta.get("PatientSex") or "").strip(),
            patient_age=str(meta.get("PatientAge") or "").strip(),
            study_date=str(meta.get("StudyDate") or "").strip(),
            iop=meta.get("ImageOrientationPatient", ""),
            # DICOM decides when it is available; otherwise fall back to the
            # affine, which still catches reformatted and oblique volumes.
            axial=is_axial(iop) if iop is not None else row["offdiag"] == 0.0,
            z_extent_mm=round(row["nz"] * row["dz"], 1),
        )
    return rows


def select_volumes(
    cases: Sequence[Case],
    metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    rule: str = RULES[0],
    workers: int = 8,
    native_matrix: Optional[int] = NATIVE_MATRIX,
) -> SelectionResult:
    """Keep one natively reconstructed volume per study.

    Args:
        cases: candidates, normally ``layout.discover()``.
        metadata: CT-RATE metadata indexed by volume name; without it the axial
            test falls back to the affine.
        rule: ``soft512`` (quality ranking) or ``recon1`` (keep ``_1``).
        workers: processes used to read the NIfTI headers.
        native_matrix: preferred in-plane matrix; a volume on another matrix
            loses to a sibling on this one. ``None``/``0`` disables the check.

    Returns:
        A :class:`SelectionResult` with the manifest and the rejected volumes.
    """
    if rule not in RULES:
        raise ValueError(f"Unknown rule '{rule}'; expected one of {RULES}.")

    grounded = {case.case_id for case in cases if case.has(GROUNDING.key)}

    items = [(case.case_id, str(case.paths[CT.key])) for case in cases]
    if workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(probe, items, chunksize=8))
    else:
        rows = [probe(item) for item in items]

    rejected: List[Dict[str, Any]] = [
        {**row, "reason": "unreadable"} for row in rows if row.get("error")
    ]
    rows = annotate([row for row in rows if not row.get("error")], metadata or {})

    by_study: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_study.setdefault(row["study_id"], []).append(row)

    selected: List[Dict[str, Any]] = []
    for study_id in sorted(by_study):
        study = by_study[study_id]
        # Both preferences fall back rather than empty a study, so a volume is
        # only ever dropped when a better sibling exists to replace it.
        eligible, off_matrix = prefer(study, lambda r: is_native_matrix(r, native_matrix))
        eligible, reformatted = prefer(eligible, lambda r: r["axial"])
        candidates = sorted(eligible, key=lambda r: selection_key(r, rule), reverse=True)

        keep = candidates[0]
        keep["kept_for"] = KEPT_FOR_STUDY
        keep["native_matrix"] = is_native_matrix(keep, native_matrix)
        # Reported over the whole study, so the manifest still accounts for
        # every sibling that existed, off-matrix ones included.
        keep["n_recons_in_study"] = len(study)
        keep["dropped_siblings"] = ";".join(
            row["volume"] for row in study if row["volume"] != keep["volume"]
        )
        selected.append(keep)

        # A grounding mask annotates one specific reconstruction, so dropping
        # that volume would throw the annotation away. Keep it alongside the
        # study's pick instead; `kept_for` tells the two apart.
        dropped = off_matrix + reformatted + candidates[1:]
        for row in dropped:
            if row["case_id"] in grounded:
                row["kept_for"] = KEPT_FOR_GROUNDING
                row["native_matrix"] = is_native_matrix(row, native_matrix)
                row["n_recons_in_study"] = len(study)
                row["dropped_siblings"] = ""
                selected.append(row)

        rejected += [
            {**row, "reason": reason}
            for rows, reason in (
                (off_matrix, "non_native_matrix"),
                (reformatted, "not_axial"),
                (candidates[1:], "duplicate_recon"),
            )
            for row in rows
            if row["case_id"] not in grounded
        ]

    manifest = pd.DataFrame(selected, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values("case_id", ignore_index=True)
    rejected_table = pd.DataFrame(rejected).sort_values("case_id", ignore_index=True) if rejected else pd.DataFrame(columns=["case_id", "volume", "reason"])
    return SelectionResult(selected=manifest, rejected=rejected_table)


def select_dataset(
    layout: DatasetLayout,
    metadata_paths: Optional[Iterable[Path]] = None,
    rule: str = RULES[0],
    workers: int = 8,
    case_ids: Optional[Sequence[str]] = None,
    native_matrix: Optional[int] = NATIVE_MATRIX,
) -> SelectionResult:
    """Run :func:`select_volumes` over the CT volumes of a staged dataset."""
    return select_volumes(
        cases=layout.discover(case_ids),
        metadata=load_metadata(metadata_paths),
        rule=rule,
        workers=workers,
        native_matrix=native_matrix,
    )
