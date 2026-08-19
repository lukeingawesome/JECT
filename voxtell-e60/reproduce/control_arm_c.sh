#!/usr/bin/env bash
# Arm C -- the control that makes E60's number mean anything. `--random-init` skips the
# pretrained load and changes NOTHING else: same seed, same sample order, same schedule, same
# gates. Without it, "VoxTell pretraining helps" is an unmeasured claim.
#
# Shorter on purpose: the control answers "does the pretrained init do work", which separates
# long before 60k. Compare it to E60's own step-20000 sweep, never to E60's 60k endpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-runs/C20_control}

exec "$PY" scripts/train_voxtell.py \
  --config configs/rexg_native.yaml \
  --out-dir "$OUT" \
  --fuse trunk --random-init \
  --steps 20000 --val-at 20000 --ckpt-every 5000 \
  --batch-size 2 --queries 2 \
  --lr 1e-4 --momentum 0.99 --clip 12.0 --fg-prob 0.66 \
  --seed 42 --workers 6 --eval-workers 16 \
  --device "${DEVICE:-cuda:0}"
