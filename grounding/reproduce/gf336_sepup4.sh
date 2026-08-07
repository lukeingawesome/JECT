#!/bin/bash
# =============================================================================
# gf336 + SEPARABLE full-resolution up-block.
#
# The only decoder here that computes REAL convolutions at 336^3 rather than interpolating res/2
# features. Each dense 3x3x3 (C*C*27 MACs/voxel) becomes depthwise 3x3x3 + pointwise 1x1x1, and the
# skip is ADDED rather than concatenated (a 2C-channel full-res tensor is 15.8 GiB and OOMs before
# any conv runs). Net: 18x fewer FLOPs and half the peak memory of the dense block.
# Same "decomposed 3D convolutions" idea as Text3DSAM.
#
# WHY vs --hires_head: hires_head lifts res/2 features trilinearly, so the logit carries only res/2
# information -- it moves the output grid without adding detail. sep_up4 can express structure the
# res/2 features do not contain, which is what the measured ceiling (0.826 median, 0.749 for
# findings <500 voxels) says is missing.
#
# COST measured: 3.29 h/epoch at batch 1, 76.8 GB. Batch 2 does not fit (~150 GB).
# 2.2x slower than --hires_head (1.52 h/ep at batch 2) and 2.5x slower than out_res 168.
# Uses the 20-epoch LR schedule so trajectories are comparable to the out_res-168 arm at matched
# epochs; stop it at the time budget rather than shortening the schedule.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
: "${JECT_DATA:?}"; : "${JECT_REX:?}"; : "${JECT_EMB:?}"; : "${JECT_INIT:?}"
export GF_CUDNN_BENCH="${GF_CUDNN_BENCH:-0}"
python train.py \
  --init "$JECT_INIT" \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 336 --out_res 336 --sep_up4 --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s1 s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-1}" --grad_accum "${ACCUM:-3}" --epochs "${EPOCHS:-20}" --lr 1.2e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/gf336_sepup4}"
