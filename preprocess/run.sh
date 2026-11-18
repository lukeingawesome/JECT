#!/usr/bin/env bash
# End-to-end preprocessing: stage the NIfTI files, keep one axial volume per
# study, convert to npz, sanity-check.
# Run from the repository root:  bash preprocess/run.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data_1/ct}"
OUT_DIR="${OUT_DIR:-/data_1/preprocessed_full}"
CONFIG="${CONFIG:-preprocess/configs/ctrate_full.yaml}"
FIGURE_DIR="${FIGURE_DIR:-figures}"

# Selection: one axial reconstruction per study. METADATA is the official
# CT-RATE metadata CSV (space-separated if you have several); without it the
# axial test falls back to the NIfTI affine.
MANIFEST="${MANIFEST:-$OUT_DIR/selected_volumes.csv}"
METADATA="${METADATA:-}"
RULE="${RULE:-soft512}"
SELECT_WORKERS="${SELECT_WORKERS:-8}"

# Sources to stage into $DATA_ROOT. Point these at wherever the raw files live;
# leave a variable empty to skip that modality (already staged, or unavailable).
CT_SRC="${CT_SRC:-}"
TOTALSEG_SRC="${TOTALSEG_SRC:-}"
GROUNDING_SRC="${GROUNDING_SRC:-}"

# Optional: restrict selection to a case-id whitelist (JSON list or split -> ids).
ID_LIST="${ID_LIST:-}"
SPLITS="${SPLITS:-}"

# ---------------------------------------------------------------- 1. stage ---
stage_args=()
[[ -n "$CT_SRC" ]]        && stage_args+=(--ct "$CT_SRC")
[[ -n "$TOTALSEG_SRC" ]]  && stage_args+=(--totalseg "$TOTALSEG_SRC")
[[ -n "$GROUNDING_SRC" ]] && stage_args+=(--grounding "$GROUNDING_SRC")

if (( ${#stage_args[@]} )); then
  python preprocess/cli.py stage \
    --data-root "$DATA_ROOT" \
    --link-mode symlink \
    "${stage_args[@]}"
else
  echo "No *_SRC set; assuming $DATA_ROOT is already staged."
  python preprocess/cli.py coverage --data-root "$DATA_ROOT"
fi

# --------------------------------------------------------------- 2. select ---
# One volume per study: CT-RATE reconstructs each scan with several kernels, so
# this keeps a single rendering instead of preprocessing the same scan repeatedly.
select_args=()
[[ -n "$METADATA" ]] && select_args+=(--metadata $METADATA)
[[ -n "$ID_LIST" ]]  && select_args+=(--id-list "$ID_LIST")
[[ -n "$SPLITS" ]]   && select_args+=(--splits $SPLITS)

mkdir -p "$(dirname "$MANIFEST")"
python preprocess/cli.py select \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --rejected "${MANIFEST%.csv}_rejected.csv" \
  --rule "$RULE" \
  --workers "$SELECT_WORKERS" \
  ${select_args[@]+"${select_args[@]}"}

# ----------------------------------------------------------- 3. preprocess ---
python preprocess/cli.py run \
  --data-root "$DATA_ROOT" \
  --out "$OUT_DIR" \
  --config "$CONFIG" \
  --id-list "$MANIFEST"

# ------------------------------------------------------------ 4. visualize ---
python preprocess/visualize.py \
  --npz-dir "$OUT_DIR" \
  --num-cases 2 \
  --out "$FIGURE_DIR"

echo "Done. npz -> $OUT_DIR, figures -> $FIGURE_DIR"
