#!/usr/bin/env python3
"""Sanity-check figures for preprocessed npz cases.

Samples a few cases from a preprocessing output directory and renders, for each
one, axial / coronal / sagittal slices of the CT, the TotalSegmentator labels
and the ReXGrounding phrase masks::

    python preprocess/visualize.py --npz-dir /data/preprocessed --num-cases 2 --out figures

Arrays are read as ``(C, D, H, W)`` in the orientation produced by
``cli.py run`` (LPS): ``D`` runs inferior->superior, ``H`` right->left,
``W`` anterior->posterior. Slices are picked where the grounding mask is
largest, falling back to the middle of the lung bounding box.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

CT_KEY = "ct"
SEG_KEY = "totalseg"
GROUNDING_KEY = "rex"
LUNG_LABELS: Tuple[int, ...] = (10, 11, 12, 13, 14)
DEFAULT_HU_WINDOW: Tuple[float, float] = (-1000.0, 300.0)
VIEWS: Tuple[str, ...] = ("axial", "coronal", "sagittal")


def load_case(path: Path) -> Dict[str, np.ndarray]:
    """Read a preprocessed npz into ``{key: (C, D, H, W) array}``."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def take_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """Extract one 2D slice from a ``(D, H, W)`` volume in display orientation."""
    if view == "axial":
        return volume[index].T  # rows: anterior->posterior, cols: right->left
    if view == "coronal":
        return volume[:, :, index]  # rows: inferior->superior, cols: right->left
    if view == "sagittal":
        return volume[:, index, :]  # rows: inferior->superior, cols: anterior->posterior
    raise ValueError(f"Unknown view '{view}'; expected one of {VIEWS}.")


def slice_axis(view: str) -> int:
    """Axis of a ``(D, H, W)`` volume that ``view`` indexes into."""
    return {"axial": 0, "sagittal": 1, "coronal": 2}[view]


def pick_indices(shape: Sequence[int], *candidates: Optional[np.ndarray]) -> Dict[str, int]:
    """Slice index per view: where the first non-empty candidate mask is densest.

    Falls back to the centre of the volume when no candidate has any voxel.
    """
    mask = next((c for c in candidates if c is not None and c.any()), None)
    indices = {}
    for view in VIEWS:
        axis = slice_axis(view)
        if mask is None:
            indices[view] = int(shape[axis]) // 2
            continue
        other = tuple(a for a in range(3) if a != axis)
        indices[view] = int(mask.sum(axis=other).argmax())
    return indices


def label_colormap(num_colors: int, seed: int) -> ListedColormap:
    """Distinct random colours, index 0 transparent (background)."""
    rng = np.random.default_rng(seed)
    colors = rng.random((max(num_colors, 1), 4))
    colors[:, :3] = 0.25 + 0.7 * colors[:, :3]  # keep them bright on a grey CT
    colors[:, 3] = 1.0
    colors[0] = (0.0, 0.0, 0.0, 0.0)
    return ListedColormap(colors)


def show_ct(axis, image: np.ndarray, window: Optional[Tuple[float, float]], view: str) -> None:
    kwargs = {"cmap": "gray", "origin": "upper" if view == "axial" else "lower"}
    if window is not None:
        kwargs.update(vmin=window[0], vmax=window[1])
    axis.imshow(image, **kwargs)
    axis.set_xticks([])
    axis.set_yticks([])


def overlay(axis, mask: np.ndarray, cmap: ListedColormap, alpha: float, view: str) -> None:
    axis.imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap=cmap,
        vmin=0,
        vmax=max(cmap.N - 1, 1),
        alpha=alpha,
        interpolation="nearest",
        origin="upper" if view == "axial" else "lower",
    )


def grounding_label_map(grounding: np.ndarray) -> np.ndarray:
    """Flatten ``(F, D, H, W)`` phrase masks into a single label map.

    A channel may already hold several phrase ids, so each channel's values are
    offset by the ids used so far and stay distinguishable in the overlay.
    """
    label_map = np.zeros(grounding.shape[1:], dtype=np.int32)
    offset = 0
    for channel in grounding:
        values = channel.astype(np.int32)
        label_map = np.where(values > 0, values + offset, label_map)
        offset += int(values.max())
    return label_map


def render_case(
    case_id: str,
    arrays: Dict[str, np.ndarray],
    out_path: Path,
    window: Optional[Tuple[float, float]],
    alpha: float,
    seed: int,
    dpi: int,
) -> None:
    """Write one figure: rows are views, columns are CT / TotalSeg / grounding."""
    ct = arrays[CT_KEY][0].astype(np.float32)
    seg = arrays[SEG_KEY][0] if SEG_KEY in arrays else None
    grounding = arrays.get(GROUNDING_KEY)

    lungs = np.isin(seg, LUNG_LABELS) if seg is not None else None
    phrases = grounding_label_map(grounding) if grounding is not None else None
    indices = pick_indices(ct.shape, phrases, lungs)

    columns: List[Tuple[str, Optional[np.ndarray], Optional[ListedColormap]]] = [("CT", None, None)]
    if seg is not None:
        columns.append((f"TotalSegmentator ({len(np.unique(seg)) - 1} labels)", seg, label_colormap(int(seg.max()) + 1, seed)))
    if phrases is not None:
        num_phrases = len(np.unique(phrases)) - 1
        columns.append(
            (
                f"ReXGrounding ({num_phrases} phrase{'s' if num_phrases != 1 else ''})",
                phrases,
                label_colormap(int(phrases.max()) + 1, seed + 1),
            )
        )

    figure, axes = plt.subplots(
        len(VIEWS),
        len(columns),
        figsize=(4.0 * len(columns), 4.0 * len(VIEWS)),
        squeeze=False,
    )
    for row, view in enumerate(VIEWS):
        index = indices[view]
        ct_slice = take_slice(ct, view, index)
        for column, (title, mask, cmap) in enumerate(columns):
            axis = axes[row][column]
            show_ct(axis, ct_slice, window, view)
            if mask is not None:
                overlay(axis, take_slice(mask, view, index), cmap, alpha, view)
            if row == 0:
                axis.set_title(title, fontsize=11)
            if column == 0:
                axis.set_ylabel(f"{view} @ {index}", fontsize=10)

    figure.suptitle(f"{case_id}  —  CT {tuple(ct.shape)}", fontsize=13)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def select_cases(
    npz_dir: Path,
    num_cases: int,
    case_ids: Optional[Sequence[str]],
    prefer_grounding: bool,
    seed: int,
) -> List[Path]:
    """Sample npz files, preferring cases that carry grounding masks."""
    available = sorted(npz_dir.glob("*.npz"))
    if not available:
        raise SystemExit(f"No npz files in {npz_dir}.")

    if case_ids:
        by_id = {path.stem: path for path in available}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise SystemExit(f"Case id(s) not found in {npz_dir}: {missing}")
        return [by_id[case_id] for case_id in case_ids]

    rng = random.Random(seed)
    pool = list(available)
    rng.shuffle(pool)
    if not prefer_grounding:
        return pool[:num_cases]

    selected: List[Path] = []
    deferred: List[Path] = []
    for path in pool:
        if len(selected) >= num_cases:
            break
        with np.load(path) as data:
            has_grounding = GROUNDING_KEY in data.files
        (selected if has_grounding else deferred).append(path)
    return (selected + deferred)[:num_cases]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--npz-dir", type=Path, required=True, help="Directory of preprocessed npz files.")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Output directory (default: %(default)s).")
    parser.add_argument("--num-cases", type=int, default=2, help="How many cases to sample (default: %(default)s).")
    parser.add_argument("--case-id", nargs="+", help="Visualize these case ids instead of sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling and colour seed (default: %(default)s).")
    parser.add_argument(
        "--hu-window",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=list(DEFAULT_HU_WINDOW),
        help="Display window for raw-HU volumes (default: %(default)s).",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=130, help="Figure dpi (default: %(default)s).")
    parser.add_argument(
        "--any-case",
        action="store_true",
        help="Sample uniformly instead of preferring cases with grounding masks.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    paths = select_cases(
        npz_dir=args.npz_dir,
        num_cases=args.num_cases,
        case_ids=args.case_id,
        prefer_grounding=not args.any_case,
        seed=args.seed,
    )

    for path in paths:
        arrays = load_case(path)
        # Windowed volumes are already scaled to [0, 1]; only raw HU needs a window.
        window = tuple(args.hu_window) if np.issubdtype(arrays[CT_KEY].dtype, np.integer) else None
        out_path = args.out / f"{path.stem}.png"
        render_case(
            case_id=path.stem,
            arrays=arrays,
            out_path=out_path,
            window=window,
            alpha=args.alpha,
            seed=args.seed,
            dpi=args.dpi,
        )
        present = ", ".join(sorted(arrays))
        print(f"{path.stem}: keys=[{present}] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
