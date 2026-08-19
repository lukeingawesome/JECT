#!/usr/bin/env python
"""The four host-specific paths, in ONE place, each overridable by an environment variable.

Everything else in this folder is host-independent: the data roots come from the YAML config
(`configs/rexg_res_2mm.yaml`), and the run directory comes from `--out-dir`. What is left is
three external assets that are not part of this repository and cannot be inferred:

    REXG_VOXTELL_MODEL_DIR   VoxTell v1.0's nnU-Net model dir -- plans.json plus
                             fold_0/checkpoint_final.pth. The architecture is rebuilt from
                             that plans.json, so it is required even with --random-init.
    REXG_QWEN_DIR            Qwen3-Embedding-4B. Loaded ONLY while building a text bank;
                             training and inference read the bank, never the encoder.
    REXG_CHALLENGE_DIR       The ReXGroundingCT release: rexrank_eval_fast.py (the scorer of
                             record), rexrank_eval.py, segmentations/ and
                             MICCAI_challenge_dataset.json.

The defaults are the paths these runs actually used, kept so the provenance in
docs is readable, not because they will exist on your machine. `require()` fails with the
variable's name rather than letting a missing path surface as a FileNotFoundError six frames
down inside a model loader -- or, worse, as a silently empty file list.

    python scripts/paths.py        # print what is resolved and whether it exists
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "REXG_VOXTELL_MODEL_DIR": "/home1/irteam/data-vol1/models/voxtell/voxtell_v1.0",
    "REXG_QWEN_DIR": "/home1/irteam/data-vol1/models/Qwen3-Embedding-4B",
    "REXG_CHALLENGE_DIR": "/home1/irteam/data-vol1/data/ReXGroundingCT",
}


def get(var: str) -> Path:
    """The configured path. Not checked -- `require` is what checks."""
    if var not in _DEFAULTS:
        raise KeyError(f"{var} is not one of {sorted(_DEFAULTS)}")
    return Path(os.environ.get(var) or _DEFAULTS[var])


def require(var: str, *, needs: str = "") -> Path:
    """The configured path, or SystemExit naming the variable and what belongs there."""
    p = get(var)
    if not p.exists():
        raise SystemExit(
            f"{var} points at {p}, which does not exist.\n"
            f"  what belongs there: {needs or _DEFAULTS[var]}\n"
            f"  set it with:        export {var}=/your/path\n"
            f"  (the default is the path this work ran on, not a guess that it exists here)")
    return p


MODEL_DIR = get("REXG_VOXTELL_MODEL_DIR")
QWEN = get("REXG_QWEN_DIR")
CHALLENGE_DIR = get("REXG_CHALLENGE_DIR")


if __name__ == "__main__":
    for var in _DEFAULTS:
        p = get(var)
        print(f"{'ok ' if p.exists() else 'MISSING'}  {var:<24} {p}")
