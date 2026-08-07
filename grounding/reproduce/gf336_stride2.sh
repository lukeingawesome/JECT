#!/bin/bash
# =============================================================================
# gf336_stride2 -- gf336_revert (0.3103, the best single model) with the full-resolution stem
# REMOVED. One flag different from gf336_revert.sh; everything else is identical.
#
# WHY THIS IS ALMOST FREE, IN THEORY. Under --hires_head the decoder never convolves above res/2:
#
#     st0 = self.stem0(vol)              # res^3 x 56 = 2.12e9 elements, 6.54 TFLOP
#     st1 = self.stem1(st0)              # stride-2 -> res/2
#     x   = self.up3(x, st1)
#     elif self.hires_head:
#         x = F.interpolate(...)         # st0 never reaches the output path
#
# `st0`'s only other consumer is `up4`, and hires_head does not take that branch (model.py:404-418).
# So the entire full-resolution feature map exists to be the input of one stride-2 convolution.
# --stem_stride2 strides straight from the raw 1-channel volume instead, and the constructor guard
# explicitly blesses this pairing: `--out_res <res> --hires_head` is one of the two legal
# combinations (model.py:85-89).
#
# WHAT IS ACTUALLY BEING TESTED. Whether that intermediate res^3 x 56 convolution earns 6.54 TFLOP
# when nothing downstream reads it at full resolution. Hold Dice near 0.3103 and every future run at
# the top of the table gets cheaper, which compounds across the whole backlog. Lose Dice and we have
# learned the full-res stem matters for a reason other than the output path -- also worth knowing.
#
# DO NOT EXPECT 2x. fast336 was ~2x faster than gf336_revert, but that bundled --stem_stride2 WITH
# out_res 168. This isolates the stem; 20-40% is the honest guess. Measure epoch 1 and believe it.
#
# NOT WARM-STARTABLE from gf336_revert: --stem_stride2 deletes stem0 and changes stem1's first conv
# from cs->cs to 1->cs, so the head state_dict differs and --init_grounding aborts by design
# (train.py:238). This is a from-scratch skeleton run. That is what the GPU running NEW SKELETONS is
# for; loss and optimisation changes belong in the warm-start slot -- see gf336_revert_ft.sh.
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
  --res 336 --out_res 336 --hires_head --stem_stride2 --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-20}" --lr "${LR:-1e-3}" --weight_decay 1e-4 --clip 5.0 --seed "${SEED:-0}" \
  --out "${OUT:-runs/gf336_stride2}"
