#!/usr/bin/env bash
# E60 -- arm E (fused text feature re-enters the encoder trunk), 60,000 steps, per-volume
# z-score. This is the run that scored mean_global_dice_per_finding 0.34334 on the 200-case
# val split at tile_step 0.25 / threshold 0.5.
#
# Every value below is the one the published run used (its run.json "args"); nothing here is
# a rounded-off approximation of it. ~15.5 h on one H200, plus the two in-training sweeps.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-runs/E60}

exec "$PY" scripts/train_voxtell.py \
  --config configs/rexg_native.yaml \
  --out-dir "$OUT" \
  --fuse trunk \
  --steps 60000 --val-at 30000,60000 --ckpt-every 5000 \
  --batch-size 2 --queries 2 \
  --lr 1e-4 --momentum 0.99 --clip 12.0 --fg-prob 0.66 \
  --seed 42 --workers 6 --eval-workers 16 \
  --device "${DEVICE:-cuda:0}"
