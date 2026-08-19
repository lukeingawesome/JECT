#!/usr/bin/env python
"""The challenge's own scorer, bound to this folder, plus the two array transforms that have
to agree with it.

NOTHING here re-implements a metric. `rexrank_eval_fast.py` is imported as a module and its
`cc_filter` IS the post-processing -- the same probability map scores orders of magnitude
differently either side of it, so a local re-implementation would be a second definition of
the metric that drifts silently. `ras_to_lps` exists for the same reason on the geometry
side: the cache is RAS index order and the GT on disk is LPS, and a mask that is scored in
the wrong index order still scores, just wrongly.

    python scripts/scorer.py        # check the scorer is present and unchanged
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths                                                                  # noqa: E402

_EVAL_DIR = paths.CHALLENGE_DIR

# The scorer of record. `rexrank_eval_fast.py` is the challenge's own file, shipped in the same
# HuggingFace repo as `rexrank_eval.py`; its only difference is an 11-line union-bounding-box
# crop before connected components -- every voxel outside the union is background in BOTH gt
# and pred, so the components, the dice and the matching are identical, just computed on a
# smaller array. Measured on arm E's real 200-case prediction set: 0.3241677390055121 against
# the official 0.32416773900551205, a 5.55e-17 difference that is float64 summation order,
# in 417 s instead of 61-155 min.
#
# EVALUATOR_REFERENCE is kept so the equivalence stays checkable rather than remembered, and
# the hashes below are what makes it expire: if either file changes upstream the gate goes red
# and the equivalence has to be re-established before any number is trusted again.
EVALUATOR = _EVAL_DIR / "rexrank_eval_fast.py"
EVALUATOR_REFERENCE = _EVAL_DIR / "rexrank_eval.py"
EVALUATOR_SHA = {
    "rexrank_eval.py": "92d7badc482356de",          # == HuggingFace main, verified byte-for-byte
    "rexrank_eval_fast.py": "4ce61b00f25f8ed4",
}

# The MICCAI val split, counted from MICCAI_challenge_dataset.json and findings.csv (they
# agree exactly: 200 cases, 381 findings). GATE 4 asserts the evaluator saw all of them.
MICCAI_VAL_CASES, MICCAI_VAL_FINDINGS = 200, 381


def ras_to_lps(arr: np.ndarray, patient_position: str) -> np.ndarray:
    """Step 5. The cache is RAS index order (build_npy_cache: RasAffined + Orientationd);
    the files on disk -- and therefore the GT the evaluator loads -- are LPS index order.

    axes 0 and 1 always (R<->L, A<->P). Axis 2 ONLY for FFS: `RasAffined` negates the slice
    axis for feet-first volumes, so `Orientationd` flipped it on the way in and skipping the
    flip here would leave those 4/3492 volumes upside down (see SLICE_AXIS_SIGN).

    Verified against the source NIfTI on 7 cached volumes (3 FFS, 4 HFS): this reproduces
    the on-disk mask exactly, instance labels included, and the other flip does not.
    """
    if patient_position not in ("HFS", "FFS"):
        raise ValueError(f"PatientPosition {patient_position!r}: no measured slice-axis "
                         f"direction, refusing to assume HFS")
    return np.flip(arr, axis=[0, 1] if patient_position == "HFS" else [0, 1, 2])


def _load_evaluator(path: Path):
    """Import rexrank_eval.py itself. Its cc_filter IS the post-processing (step 4): the
    same probability map scores orders of magnitude differently either side of it, so a
    re-implementation here would be a second, drifting definition of the metric."""
    spec = importlib.util.spec_from_file_location("rexrank_eval", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"official evaluator not importable: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def postprocess(binary: np.ndarray, ev) -> np.ndarray:
    """Step 4: the evaluator's own cc_filter + its top-K cap, applied BEFORE writing so the
    file on disk is exactly what gets scored. Idempotent w.r.t. the evaluator's own pass.
    Returns BINARY uint8 -- not the component labels: `load_4d` casts to uint8, so a 300th
    component would wrap to label 44 and a 256th would vanish into background."""
    cc = ev.cc_filter(binary, ev.MIN_SIZE_DEFAULT, ev.CC_CONNECTIVITY_DEFAULT)
    if ev.MAX_COMPONENTS_PER_FINDING:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        ids = np.flatnonzero(sizes)
        if ids.size > ev.MAX_COMPONENTS_PER_FINDING:
            keep = ids[np.argsort(sizes[ids])[::-1][:ev.MAX_COMPONENTS_PER_FINDING]]
            cc[~np.isin(cc, keep)] = 0
    return (cc > 0).astype(np.uint8)


def score(pred_dir: Path, findings, gt_dir: Path, out_dir: Path, *,
          workers: int = 16, expect: tuple[int, int] | None = None) -> dict:
    """Run the official scorer over `pred_dir` and return its summary.

    ONE home for this, called by the in-training sweep and by reproduce/submit.sh alike, so a
    submission cannot be scored under a slightly different invocation than the sweep that
    chose its checkpoint. `findings` is the DataFrame slice for the volumes being scored --
    the dataset_json the evaluator builds its work list from comes from that, not from a glob
    of pred_dir, so a missing prediction is a MISS in the score rather than a shorter list.

    `expect` is the (cases, findings) the caller believes it scored. It is checked, because a
    scorer that silently saw fewer cases returns a perfectly plausible number.
    """
    import json
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(findings["VolumeName"].unique())
    dataset_json = out_dir / "dataset_eval.json"
    dataset_json.write_text(json.dumps({"test": [
        {"name": n, "seg_path": str(gt_dir / n),
         "findings": {str(r["finding_idx"]): r["text"]
                      for r in findings[findings["VolumeName"] == n].to_dict("records")}}
        for n in names]}, indent=2), encoding="utf-8")
    eval_json = out_dir / "eval.json"
    eval_json.unlink(missing_ok=True)      # the evaluator RESUMES from an existing file
    cmd = [sys.executable, str(EVALUATOR), "--gt_dir", str(gt_dir), "--pred_dir",
           str(pred_dir), "--output_json", str(eval_json), "--dataset_json",
           str(dataset_json), "--num_workers", str(workers)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    summary = json.loads(eval_json.read_text())["summary"]
    if expect is not None and (summary["total_cases"], summary["total_findings"]) != expect:
        raise SystemExit(f"the evaluator scored {summary['total_cases']} cases / "
                         f"{summary['total_findings']} findings, expected {expect[0]}/"
                         f"{expect[1]}. A short work list scores a plausible wrong number.")
    return summary


def gate_scorer() -> dict[str, str]:
    """The scorer of record must be PRESENT and BYTE-IDENTICAL to what the equivalence was
    established against. A silently-updated upstream file would keep every number flowing and
    change what they mean, so this turns red instead."""
    paths.require("REXG_CHALLENGE_DIR",
                  needs="the ReXGroundingCT release: rexrank_eval_fast.py, rexrank_eval.py, "
                        "segmentations/, MICCAI_challenge_dataset.json")
    got = {}
    for f, want in EVALUATOR_SHA.items():
        p = _EVAL_DIR / f
        if not p.is_file():
            raise SystemExit(f"scorer of record missing: {p}. It ships with the challenge "
                             f"data, not with this repository.")
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        got[f] = h
        if h != want:
            raise SystemExit(
                f"G-SCORER: {p} hashes {h}, pinned {want}. Upstream changed the scorer -- the "
                f"fast/reference equivalence (measured 5.55e-17 over 200 real cases) is now an "
                f"ASSUMPTION, not a measurement. Re-establish it before trusting a number, "
                f"then update EVALUATOR_SHA.")
    return got


if __name__ == "__main__":
    print(f"scorer dir {_EVAL_DIR}")
    for f, h in gate_scorer().items():
        print(f"  ok  {f}  sha256[:16]={h}")
    ev = _load_evaluator(EVALUATOR)
    print(f"  loaded: MIN_SIZE={ev.MIN_SIZE_DEFAULT} CC={ev.CC_CONNECTIVITY_DEFAULT} "
          f"MAX_COMPONENTS={ev.MAX_COMPONENTS_PER_FINDING}")
    print(f"  official val split: {MICCAI_VAL_CASES} cases / {MICCAI_VAL_FINDINGS} findings")
