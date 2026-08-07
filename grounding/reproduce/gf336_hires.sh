#!/bin/bash
# =============================================================================
# gf336 + HIRES HEAD -- native 336^3 output without convolving at full resolution.
#
# WHY. A res/2 output caps Dice at 0.826 median even for a PERFECT model (measured on val GT with
# our max-pooled target + trilinear lift + swept threshold: 0.749 for findings <500 voxels, 0.921
# for >5k). Our runs sit at quality-among-hits ~0.38, i.e. 46% of that ceiling, so the output grid
# is a real constraint on segmentation precision.
#
# Native output normally means running `up4` at 336^3: two CxC 3x3x3 convs = 12.85 TFLOP, ~50 h/epoch,
# and its input concat alone needs 15.8 GiB. --hires_head instead lifts the res/2 feature map
# trilinearly and applies only the 1x1x1 output conv and the --txt_head dot product there, both O(C)
# per voxel -- about 1600x fewer FLOPs for the same 336^3 logit.
#
# The substantive change is not how the logit is formed but that the LOSS is now computed at full
# resolution: the model is trained against 336^3 targets instead of a max-pooled 168^3 stand-in.
#
# Same family as HRMedSeg's ECM-Decoder (inner product between a query embedding and per-pixel
# features, interpolation deferred to the end; 92% memory reduction vs UNet at 1024^2) and
# EffiDec3D (CVPR 2025), which removes high-resolution decoder layers where they do not earn their
# cost. NOTE those lift the 1-CHANNEL logit; we lift the C-channel features. Measured, the two are
# NOT interchangeable -- correlation 0.995 but thresholded-mask Dice 0.0011 at thr 0.3 -- because
# the L2 normalisation is nonlinear. Which is better on this data is exactly what this run tests.
#
# COST at batch 1: 1.94 h/epoch, 49.9 GB. Batch 2 fits ~100 GB. 12 epochs to stay inside a day.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${JECT_DATA:?set JECT_DATA (384^3 npz dir; cropped to 336 at load)}"
: "${JECT_REX:?set JECT_REX}"
: "${JECT_EMB:?set JECT_EMB}"
: "${JECT_INIT:?set JECT_INIT to the 336-cooled checkpoint}"

export GF_CUDNN_BENCH="${GF_CUDNN_BENCH:-0}"

exec python train.py \
  --init "$JECT_INIT" \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 336 --out_res 336 --hires_head --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s1 s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-12}" --lr 1.2e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/gf336_hires}"
