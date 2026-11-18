"""Chest-CT preprocessing: stage NIfTI volumes, then convert them to npz.

``layout``/``staging``/``settings``/``geometry`` are dependency-light. The MONAI
pipeline lives in :mod:`preprocess.pipeline` and its driver in
:mod:`preprocess.runner`; import those directly so tools that only read the
layout do not need MONAI installed.
"""

from .settings import LUNG_LOBE_LABELS, PreprocessConfig
from .layout import (
    CT,
    GROUNDING,
    MODALITIES,
    TOTALSEG,
    Case,
    DatasetLayout,
    Modality,
    case_id_from_path,
    load_case_ids,
)
from .selection import RULES, SelectionResult, select_dataset, select_volumes
from .staging import coverage, stage_dataset, summarize_coverage

__all__ = [
    "CT",
    "GROUNDING",
    "LUNG_LOBE_LABELS",
    "MODALITIES",
    "RULES",
    "TOTALSEG",
    "Case",
    "DatasetLayout",
    "Modality",
    "PreprocessConfig",
    "SelectionResult",
    "case_id_from_path",
    "coverage",
    "load_case_ids",
    "select_dataset",
    "select_volumes",
    "stage_dataset",
    "summarize_coverage",
]
