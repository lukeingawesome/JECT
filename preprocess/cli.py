#!/usr/bin/env python3
"""Chest-CT preprocessing CLI.

    python preprocess/cli.py stage    --data-root ... --ct ... --totalseg ... --grounding ...
    python preprocess/cli.py select   --data-root ... --metadata ... --manifest manifest.csv
    python preprocess/cli.py coverage --data-root ...
    python preprocess/cli.py run      --data-root ... --out ... --id-list manifest.csv

``stage`` gathers NIfTI files into ``<data-root>/{original,totalseg,rexgrounding}``,
``select`` keeps one axial reconstruction per study, ``run`` converts each case
into one npz. Defaults live in ``preprocess/settings.py``; a YAML file and the
flags below override them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # allow `python preprocess/cli.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess import (
    MODALITIES,
    DatasetLayout,
    PreprocessConfig,
    coverage,
    load_case_ids,
    stage_dataset,
    summarize_coverage,
)
from preprocess.selection import NATIVE_MATRIX, RULES, select_dataset
from preprocess.staging import LINK_MODES

CONFIG_FLAGS = (
    "target_spacing",
    "target_shape",
    "orientation",
    "hu_window",
    "image_dtype",
    "label_dtype",
    "image_pad_value",
    "label_pad_value",
    "lung_labels",
    "min_source_shape",
    "compress",
    "overwrite",
    "num_workers",
)


def _add_layout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory.")
    for modality in MODALITIES:
        parser.add_argument(
            f"--{modality.flag}-dirname",
            default=modality.dirname,
            help=f"Sub-directory holding the {modality.label} (default: %(default)s).",
        )


def _layout(args: argparse.Namespace) -> DatasetLayout:
    return DatasetLayout(
        root=args.data_root,
        dirnames={m.key: getattr(args, f"{m.flag}_dirname") for m in MODALITIES},
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Config overrides. All default to None so YAML/dataclass defaults win."""
    group = parser.add_argument_group("preprocessing settings (override the config file)")
    group.add_argument("--config", type=Path, help="YAML file with preprocessing settings.")
    group.add_argument("--target-spacing", nargs=3, type=float, metavar=("X", "Y", "Z"))
    group.add_argument("--target-shape", nargs=3, type=int, metavar=("X", "Y", "Z"))
    group.add_argument("--orientation", help="Axis codes, e.g. LPS.")
    group.add_argument(
        "--hu-window",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Scale this HU window to [0, 1]; omit to store raw HU.",
    )
    group.add_argument("--image-dtype", help="Storage dtype of the CT, e.g. int16.")
    group.add_argument("--label-dtype", help="Storage dtype of the masks, e.g. uint16.")
    group.add_argument("--image-pad-value", type=float, help="CT padding value.")
    group.add_argument("--label-pad-value", type=float, help="Mask padding value.")
    group.add_argument(
        "--lung-labels",
        nargs="+",
        type=int,
        help="TotalSegmentator labels whose bounding box centres the crop.",
    )
    group.add_argument("--min-source-shape", nargs=3, type=int, metavar=("X", "Y", "Z"))
    group.add_argument("--compress", action=argparse.BooleanOptionalAction, default=None)
    group.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reprocess cases whose npz already exists.",
    )
    group.add_argument("--num-workers", type=int)


def _config(args: argparse.Namespace) -> PreprocessConfig:
    overrides: Dict[str, Any] = {name: getattr(args, name, None) for name in CONFIG_FLAGS}
    return PreprocessConfig.load(args.config, overrides)


def cmd_stage(args: argparse.Namespace) -> int:
    sources = {m.key: getattr(args, m.flag) for m in MODALITIES}
    if not any(sources.values()):
        flags = "/".join(f"--{m.flag}" for m in MODALITIES)
        raise SystemExit(f"Nothing to stage: pass at least one of {flags}.")

    layout = _layout(args)
    report = stage_dataset(
        layout,
        sources={key: paths for key, paths in sources.items() if paths},
        link_mode=args.link_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.report:
        report.to_csv(args.report, index=False)
        print(f"Wrote {args.report}")

    print(f"Staged into {layout.root} ({args.link_mode}{', dry run' if args.dry_run else ''})")
    if not report.empty:
        counts = report.groupby(["modality", "action"]).size()
        print(counts.to_string())
    if not args.dry_run:
        print(summarize_coverage(coverage(layout)))
    return 0


def _whitelist(args: argparse.Namespace) -> Optional[List[str]]:
    """Case ids from ``--id-list``, or ``None`` for the whole dataset."""
    if args.id_list:
        case_ids = load_case_ids(args.id_list, args.splits)
        print(f"Case-id filter: {args.id_list} -> {len(case_ids)} id(s)")
        return case_ids
    if args.splits:
        raise SystemExit("--splits requires --id-list.")
    return None


def cmd_select(args: argparse.Namespace) -> int:
    layout = _layout(args)
    result = select_dataset(
        layout,
        metadata_paths=args.metadata,
        rule=args.rule,
        workers=args.workers,
        case_ids=_whitelist(args),
        native_matrix=args.native_matrix,
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    result.selected.to_csv(args.manifest, index=False)
    print(result.summary())
    print(f"\nManifest -> {args.manifest}  (feed it to `run --id-list {args.manifest}`)")

    if args.rejected:
        result.rejected.to_csv(args.rejected, index=False)
        print(f"Rejected  -> {args.rejected}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    layout = _layout(args)
    table = coverage(layout)
    print(summarize_coverage(table))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from preprocess.runner import run_preprocess  # imports MONAI

    layout = _layout(args)
    config = _config(args)
    run_preprocess(layout, args.out, config, case_ids=_whitelist(args), limit=args.limit)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage",
        help="Collect NIfTI files into the canonical dataset layout.",
        description="Step 1: place CT volumes and masks under one dataset root.",
    )
    _add_layout_args(stage)
    for modality in MODALITIES:
        stage.add_argument(
            f"--{modality.flag}",
            nargs="+",
            type=Path,
            metavar="PATH",
            help=f"Source files or directories holding {modality.label} volumes.",
        )
    stage.add_argument("--link-mode", choices=LINK_MODES, default=LINK_MODES[0])
    stage.add_argument("--overwrite", action="store_true", help="Replace existing destinations.")
    stage.add_argument("--dry-run", action="store_true", help="Only report what would be staged.")
    stage.add_argument("--report", type=Path, help="Write the per-file staging report to this CSV.")
    stage.set_defaults(func=cmd_stage)

    select = subparsers.add_parser(
        "select",
        help="Keep one natively reconstructed volume per study.",
        description=(
            "Step 1b: keep one reconstruction per study, writing a manifest for "
            "`run --id-list`."
        ),
    )
    _add_layout_args(select)
    select.add_argument(
        "--metadata",
        nargs="+",
        type=Path,
        help="Official CT-RATE metadata CSV(s). Without them the axial test "
             "falls back to the NIfTI affine.",
    )
    select.add_argument("--manifest", type=Path, required=True, help="Manifest CSV to write.")
    select.add_argument("--rejected", type=Path, help="Also write the rejected volumes here.")
    select.add_argument(
        "--rule",
        choices=RULES,
        default=RULES[0],
        help="soft512: rank by kernel/matrix/thickness; recon1: keep _1 "
             "(default: %(default)s).",
    )
    select.add_argument(
        "--native-matrix",
        type=int,
        default=NATIVE_MATRIX,
        help="Preferred in-plane matrix; a volume on another matrix loses to a "
             "sibling on this one. CT-RATE's larger matrices are the sharp-kernel, "
             "thicker-slice series. 0 disables the check (default: %(default)s).",
    )
    select.add_argument("--id-list", type=Path, help="Restrict to these case ids before selecting.")
    select.add_argument("--splits", nargs="+", help="Splits to use when --id-list is a split mapping.")
    select.add_argument("--workers", type=int, default=8, help="Header-reading processes.")
    select.set_defaults(func=cmd_select)

    cov = subparsers.add_parser("coverage", help="Report which modalities each case has.")
    _add_layout_args(cov)
    cov.add_argument("--out", type=Path, help="Write the per-case table to this CSV.")
    cov.set_defaults(func=cmd_coverage)

    run = subparsers.add_parser(
        "run",
        help="Convert staged cases into one npz per case.",
        description="Step 2: resample, window, crop around the lungs and save npz files.",
    )
    _add_layout_args(run)
    run.add_argument("--out", type=Path, required=True, help="Output directory for the npz files.")
    run.add_argument("--id-list", type=Path, help="JSON/txt whitelist of case ids.")
    run.add_argument("--splits", nargs="+", help="Splits to use when --id-list is a split mapping.")
    run.add_argument("--limit", type=int, help="Process at most this many cases (smoke tests).")
    _add_config_args(run)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
