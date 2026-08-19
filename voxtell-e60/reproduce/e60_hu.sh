#!/usr/bin/env bash
# E60_Hu -- E60 with `--hu-norm ct` and NOTHING else changed. A separate 60,000-step run from
# the same v1.0 checkpoint, not a continuation, so the two 60k endpoints are comparable.
#
# Result: 0.34076 against E60's 0.34334, a -0.00258 gap that is inside the pre-registered
# +-0.008 detection floor and flips sign between operating points. Indistinguishable -- on
# the gradient too: step-25 grad_norm is 20.387 here against 20.603 under z-score.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-runs/E60_Hu}

exec "$PY" scripts/train_voxtell.py \
  --config configs/rexg_native.yaml \
  --out-dir "$OUT" \
  --fuse trunk --hu-norm ct \
  --steps 60000 --val-at 30000,60000 --ckpt-every 5000 \
  --batch-size 2 --queries 2 \
  --lr 1e-4 --momentum 0.99 --clip 12.0 --fg-prob 0.66 \
  --seed 42 --workers 6 --eval-workers 16 \
  --device "${DEVICE:-cuda:0}"
