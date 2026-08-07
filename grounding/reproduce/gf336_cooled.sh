#!/bin/bash
# =============================================================================
# gf336 @ 336  --  grounding on a HIGH-RES COOLED backbone.
#
# Backbone: the 224 pretrained model cooled at 336^3 for 6 epochs (V-JEPA-2-style resolution
# phase: low-res bulk pretraining, then a short high-res adaptation). The cooldown interpolates
# the absolute position embeddings (s3 14^3->21^3, s4 7^3->11^3); RoPE needs nothing, its
# coordinates are normalised to [-1,1] and so are resolution-agnostic.
#
# Differences from muplogit_224.sh, and why:
#   --res 336            3.375x the voxels. Encoder cost is negligible (0.3 s/forward).
#   --enc_skips s1 s2 s3 take EVERY decoder skip from the encoder. `enc_skips` selects the SOURCE
#                        of a skip (encoder eX vs the pretrained dense decoder), it does not add
#                        one. The dense decoder attenuates phrase signal -- measured at s2,
#                        encoder +0.042 (8.8se) vs dense +0.001 (n.s.) -- so at 336 we take all
#                        four scales from the encoder. Costs nothing.
#   --hi_ch 56           NOT 64. The decoder stem holds a res^3 x hi_ch tensor; at 336 that is
#                        2.43B elements with hi_ch=64, which exceeds the 2^31 limit that fast
#                        cuDNN kernels index with, and falls back ~29x slower.
#                        Measured: hi_ch 64 -> 9.70 s/forward, hi_ch 56 -> 0.33 s.
#                        Cap is 2^31 / 336^3 = 56.
#   --out_res 168        The decoder emits res/2 (up3) or res (up4) and nothing between.
#                        336 native costs ~50 h/epoch; 168 costs ~0.3 s/forward. Output is
#                        upsampled to 336^3 for scoring.
#   --batch 3            Max that fits in ~124 GB. batch 4 OOMs at this width.
#   --lr 1.2e-3          Scaled for batch 3 (muplogit used 1e-3 at batch 2).
#
# Early result vs the 224 arms at matched epoch:
#   ep1  Dice 0.1378 / hit 0.3990    (224 arms: 0.104-0.115 Dice, 0.297-0.307 hit)
# i.e. +20% Dice and +0.09 hit at ep1. Hit rate is the term the miss analysis identified as the
# only lever that moves Dice, so this is the expected signature if retrieval was resolution-bound.
#
# Set EXTRA="--txt_head" for the arm that also makes the full-resolution logit text-dependent.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${JECT_DATA:?set JECT_DATA to the 384^3 .npz volume dir (cropped to 336 at load)}"
: "${JECT_REX:?set JECT_REX to the ReXGrounding CSV}"
: "${JECT_EMB:?set JECT_EMB to the phrase-embedding .pt}"
: "${JECT_INIT:?set JECT_INIT to the 336-COOLED checkpoint}"

# cuDNN autotune searches real convolutions per shape; at 336 that stalls the first step for
# many minutes with no output. Off by default here, on for a fixed 224 input where it pays off.
export GF_CUDNN_BENCH="${GF_CUDNN_BENCH:-0}"

exec python train.py \
  --init "$JECT_INIT" \
  ${WARM:+--init_grounding "$WARM"} \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 336 --out_res 168 --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s1 s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch 3 --epochs 20 --lr 1.2e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/gf336_cooled}"
