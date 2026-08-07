#!/bin/bash
# =============================================================================
# muplogit @ 224  --  best SINGLE-MODEL result: val Dice 0.3070 (epoch 15 of 28)
#
# Backbone: 224^3 pretrained checkpoint (ART-JEPA v2, ep40). Frozen; 34.7M trainable.
#
# The three flags that carry the method, in order of measured contribution:
#   --predictor_prior   frozen CFG residual pred(phrase) - pred(null) from the pretrained
#                       predictor, injected into the decoder.  xattn 0.1976 -> 0.2905
#   --mup_logit         the frozen MUP-inversion field added to the OUTPUT LOGIT rather than
#                       gating the bottleneck. As a bottleneck gate the decoder washed it out
#                       (an identity-init concat reproduced the baseline exactly at ep1);
#                       on the logit the decoder can only learn a residual OVER it.
#                       maxdice 0.2978 -> 0.3070
#   --mup_prior s4      the field itself: invert the masked-utterance-prediction head at s4 and
#                       score each site by cos(predicted text, phrase). Frozen, no finetuning.
#                       Standalone at 7^3: Dice 0.403, swap-gap +0.163 (5.3se), hit@0.1 0.969.
#
# Trajectory (val best-threshold Dice / hit@0.1):
#   ep1  0.1077/0.297   ep5  0.1880/0.483   ep9  0.2769/0.698
#   ep13 0.2970/0.719   ep15 0.3070/0.743  <- best   ep20 0.3050/0.709
# Nearly all the gain is HIT RATE (0.297 -> 0.743); Dice-among-hits stayed ~0.40 throughout.
#
# NOTE the reported number is best-of-swept-threshold on val, i.e. val-tuned. For a held-out
# figure, fix the threshold on val and apply it unchanged to test.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${JECT_DATA:?set JECT_DATA to the .npz volume dir}"
: "${JECT_REX:?set JECT_REX to the ReXGrounding CSV}"
: "${JECT_EMB:?set JECT_EMB to the phrase-embedding .pt}"
: "${JECT_INIT:?set JECT_INIT to the 224 pretrained checkpoint}"

python train.py \
  --init "$JECT_INIT" \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 224 --out_res 224 --hi_ch 64 \
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  --batch 2 --epochs 28 --lr 1e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/muplogit_224}"
