"""Multiprocessing driver for the preprocessing pipeline."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from .settings import PreprocessConfig
from .layout import Case, DatasetLayout
from .pipeline import build_transforms, process_case

METADATA_FILENAME = "preprocessing_metadata.csv"
CONFIG_FILENAME = "preprocess_config.yaml"

_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(config: PreprocessConfig, out_dir: Path) -> None:
    """Build the transform chain once per worker instead of once per case."""
    _WORKER_STATE["config"] = config
    _WORKER_STATE["out_dir"] = out_dir
    _WORKER_STATE["transforms"] = build_transforms(config)


def _process(case: Case) -> Dict[str, Any]:
    return process_case(
        case,
        out_dir=_WORKER_STATE["out_dir"],
        config=_WORKER_STATE["config"],
        transforms=_WORKER_STATE["transforms"],
    )


def run_preprocess(
    layout: DatasetLayout,
    out_dir: Path,
    config: PreprocessConfig,
    case_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Preprocess a whole dataset into ``out_dir`` and write the run metadata.

    Args:
        layout: staged dataset to read from.
        out_dir: destination for ``<case_id>.npz``, the metadata CSV and the
            resolved config.
        config: preprocessing settings.
        case_ids: optional whitelist of case ids.
        limit: process at most this many cases (useful for smoke tests).
        progress: show a tqdm bar.

    Returns:
        One row per case, as written to ``preprocessing_metadata.csv``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / CONFIG_FILENAME)

    cases = layout.discover(case_ids)
    if case_ids is not None:
        missing = layout.missing_case_ids(case_ids)
        if missing:
            print(f"[warn] {len(missing)} requested id(s) have no CT volume, e.g. {missing[:5]}")
    if limit is not None:
        cases = cases[:limit]

    print(f"Preprocessing {len(cases)} case(s) from {layout.root} -> {out_dir}")
    if not cases:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    workers = min(config.num_workers, len(cases))
    if workers == 1:
        _init_worker(config, out_dir)
        records = [_process(case) for case in _bar(cases, progress)]
    else:
        with mp.Pool(workers, initializer=_init_worker, initargs=(config, out_dir)) as pool:
            records = list(_bar(pool.imap_unordered(_process, cases), progress, total=len(cases)))

    table = pd.DataFrame(records).sort_values("case_id", ignore_index=True)
    table.to_csv(out_dir / METADATA_FILENAME, index=False)
    print(f"\nWrote {out_dir / METADATA_FILENAME}\n")
    print(summarize(table))
    return table


def _bar(iterable: Iterable, progress: bool, total: Optional[int] = None):
    if not progress:
        return iterable
    return tqdm(iterable, total=total, desc="Preprocessing", unit="case", ncols=100)


def summarize(table: pd.DataFrame) -> str:
    """Status counts plus the first few failures of each kind."""
    if table.empty:
        return "No cases processed."

    kinds = table["status"].str.split(":").str[0]
    lines = ["Status counts:"]
    lines += [f"  {kind:<24} {count}" for kind, count in kinds.value_counts().items()]

    failures = table[kinds.str.startswith("error")]
    if not failures.empty:
        lines.append(f"\nFirst {min(5, len(failures))} of {len(failures)} failure(s):")
        for _, row in failures.head(5).iterrows():
            lines.append(f"  {row['case_id']}: {row['status']}")
    return "\n".join(lines)
