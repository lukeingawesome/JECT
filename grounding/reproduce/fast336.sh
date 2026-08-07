#!/bin/bash
# =============================================================================
# fast336 -- CEPA's decoder GEOMETRY at 336: never convolve at full resolution.
#
# WHY. Our stem runs a real 3x3x3 conv over the whole input cube: at 336^3 x hi_ch 56 that is one
# tensor of 2.12e9 elements -- 0.99x the 2^31 cuDNN index limit, 6.54 TFLOP -- and its only consumer
# is up4's skip. --stem_stride2 downsamples in the stem's FIRST conv, so nothing wider than the
# 1-channel input is ever held at res^3. The finest wide tensor drops to 168^3 x 56 = 2.65e8, ~8x
# smaller and far off the cliff.
#
# This is the geometry CEPA uses (grounding_unet.py: 224 -> conv stride2 -> 112, and its MAIN logit
# is emitted at 112^3 and upsampled only at eval). It trains 8 epochs in ~4.7 h at the same input
# resolution where we take 1.87 h PER EPOCH. Speed is the point: at ~30 h/run we get 1.6 runs/day
# across two GPUs, so every launch is a bet; at ~5 h/run the same backlog becomes a sweep.
#
# THE GATE. This must reach gf336_hires's 0.2646. If it does, the iteration loop gets ~6x cheaper
# and hi_ch 128 becomes affordable (CEPA runs 128 -- more channels than us, but at low resolution).
# If it does not, output resolution genuinely costs us and we pay 30 h runs knowingly. Either answer
# is worth 5 h. Our one prior out_res 168 run (gf336_cooled, 0.2203) underperformed hires, but it is
# confounded by epochs (20 vs 12), batch (3 vs 2) and the hires_head output path -- this isolates it.
#
# EPOCHS. OneCycleLR derives its whole schedule from --epochs and CANNOT be extended later; that is
# what capped gf336_hires at 12 while it was still improving. Measure epoch time from the first
# epoch of this run and set EPOCHS so the run lands inside your budget.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${JECT_DATA:?set JECT_DATA (384^3 npz dir; cropped to 336 at load)}"
: "${JECT_REX:?set JECT_REX}"
: "${JECT_EMB:?set JECT_EMB}"
: "${JECT_INIT:?set JECT_INIT to the 336-cooled checkpoint}"

export GF_CUDNN_BENCH="${GF_CUDNN_BENCH:-0}"

python train.py \
  --init "$JECT_INIT" \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 336 --out_res 168 --stem_stride2 --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-20}" --lr 1e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/fast336}"
