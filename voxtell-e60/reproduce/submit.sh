#!/usr/bin/env bash
# checkpoint -> val score -> test masks -> submission zip.
#
# Scores val FIRST and prints the number, always. The operating point below (tile_step 0.25,
# threshold 0.5) is the one E60 and E60_Hu were submitted at, so this run is not choosing it
# -- but submitting without a val number is submitting blind, and the val number is what a
# leaderboard surprise gets read against.
#
# --save-prob stores the sigmoid sparsely above a floor, so re-thresholding later costs
# minutes instead of a second 40-minute pass over 300 volumes.
#
#   reproduce/submit.sh runs/E60_Hu/checkpoint_step60000.pth ct       # E60_Hu
#   reproduce/submit.sh runs/E60/checkpoint_step60000.pth     zscore  # E60
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: submit.sh <checkpoint.pth> <zscore|ct> [tta_shift]}
HU=${2:?usage: submit.sh <checkpoint.pth> <zscore|ct> [tta_shift]}
TTA=${3:-0}                        # 6 is JECT's translation-jitter value; 0 = off
PY=${PY:-python}
THR=0.5
STEP=0.25
TAG="$(basename "$(dirname "$CKPT")")_thr${THR}_tile${STEP}$([ "$TTA" != 0 ] && echo "_tta$TTA")"
NGPU=${NGPU:-2}
SHARDS=${SHARDS:-$((NGPU * 3))}

PAIRED=$("$PY" -c "
import sys; sys.path.insert(0, 'src')
from rexg_data.config import load_data_config
print(load_data_config('configs/rexg_native.yaml').paired_root)")
BANK="$PAIRED/text_emb_qwen3_all.npz"          # test prompts are only in the _all bank

predict () {                                   # $1 = split, $2 = out dir
  local i dev
  for i in $(seq 0 $((SHARDS - 1))); do
    dev=$((i % NGPU))
    "$PY" scripts/predict_split.py \
      --checkpoint "$CKPT" --fuse trunk --split "$1" --out "$2" \
      --device "cuda:$dev" --shard "$i/$SHARDS" --thresholds "$THR" \
      --tile-step "$STEP" --save-prob 0.05 --hu-norm "$HU" --tta "$TTA" \
      --text-bank "$BANK" &
  done
  wait
}

echo "== val =="
predict val "runs/pred_${TAG}_val"
n=$(ls "runs/pred_${TAG}_val/thr${THR}"/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -eq 200 ] || { echo "val incomplete: $n of 200 -- refusing to build a submission"; exit 1; }

"$PY" - "runs/pred_${TAG}_val/thr${THR}" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import pandas as pd
import scorer
from rexg_data.config import load_data_config

cfg = load_data_config("configs/rexg_native.yaml")
pred = Path(sys.argv[1])
findings = pd.read_csv(Path(cfg.paired_root) / "findings.csv", dtype={"VolumeName": str})
val = findings[(findings["miccai_split"] == "val") & (findings["has_seg"] == 1)]
# The SAME function the in-training sweep calls, expecting the same official 200/381.
s = scorer.score(pred, val, Path(cfg.seg_root), pred.parent,
                 expect=(scorer.MICCAI_VAL_CASES, scorer.MICCAI_VAL_FINDINGS))
print(f"\nVAL mean_global_dice_per_finding = {s['mean_global_dice_per_finding']:.5f}  "
      f"({s['total_cases']} cases / {s['total_findings']} findings)\n")
PYEOF

echo "== test =="
predict test "runs/pred_${TAG}_test"
n=$(ls "runs/pred_${TAG}_test/thr${THR}"/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -eq 300 ] || { echo "test incomplete: $n of 300 -- refusing to zip"; exit 1; }

"$PY" scripts/make_submission.py \
  --pred-dir "runs/pred_${TAG}_test/thr${THR}" \
  --prob-dir "runs/pred_${TAG}_test" \
  --threshold "$THR" --zip "runs/submission_${TAG}.zip"
echo "wrote runs/submission_${TAG}.zip and its manifest"
