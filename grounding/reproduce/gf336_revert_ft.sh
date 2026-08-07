#!/bin/bash
# =============================================================================
# gf336_revert_ft -- WARM-START harness: continue from gf336_revert's head (0.3103) and change
# exactly one thing about the objective.
#
# The architecture flags below are gf336_revert.sh's, character for character, and they must stay
# that way: --init_grounding refuses ANY non-`art.` key mismatch rather than loading partially
# (train.py:238). Change a head flag here and the run aborts at startup -- which is the intended
# behaviour, not a bug to work around.
#
# WHAT BELONGS IN THIS SLOT. Anything that leaves the head's state_dict identical:
#   loss recalibration (--pos_weight, --tversky_a/b), deep-supervision targets, coverage/ranking
#   terms, --unfreeze, or simply more epochs. Architecture changes -- --stem_stride2,
#   --skip_fusion add, a different --hi_ch, a new head -- cannot warm-start and need their own
#   from-scratch run (see gf336_stride2.sh).
#
# TWO RULES, BOTH LEARNED THE EXPENSIVE WAY:
#
#   REDUCED LR IS NOT OPTIONAL. txthead warm-started at full peak LR and gained +0.001 over its own
#   starting point in 10 h. hires_ext warm-started at QUARTER peak LR and gained +0.010. Default
#   here is 2.5e-4, a quarter of gf336_revert's 1e-3. Raising it discards the warm start.
#
#   ONE VARIABLE PER RUN. tversky(0.3,0.7) and BCE(pos_weight=50) stack two recall biases;
#   unstacking BOTH at once once drove Dice to 0.0000. Change one, in its own run, via EXTRA.
#
# FIRST EXPERIMENT (the one this was written for):
#
#   EXTRA="--pos_weight 10"
#
# Why pos_weight first: the measured signature of the stacked bias is precision 0.172 / recall
# 0.493 at an operating threshold of 0.94, and Tier 0 measured that a PERFECT per-finding threshold
# is worth +0.0187 (0.2986 -> 0.3173 on the 224 model). A threshold that has to sit at 0.94 is a
# calibration failure, and pos_weight=50 is the larger of the two biases producing it. This attacks
# that headroom without waiting for the extent head to be built.
#
# Warm-started runs show their answer in a few epochs rather than 19, so this slot can cycle several
# objective variants in the time one from-scratch skeleton run takes.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

: "${JECT_DATA:?set JECT_DATA (384^3 npz dir; cropped to 336 at load)}"
: "${JECT_REX:?set JECT_REX}"
: "${JECT_EMB:?set JECT_EMB}"
: "${JECT_INIT:?set JECT_INIT to the 336-cooled checkpoint}"
: "${WARM:?set WARM to the run to continue, e.g. /opt/project/cepa2/art/runs/gf336_revert/best.pt}"

export GF_CUDNN_BENCH="${GF_CUDNN_BENCH:-0}"

exec python train.py \
  --init "$JECT_INIT" \
  --init_grounding "$WARM" \
  --data_root "$JECT_DATA" --rex_csv "$JECT_REX" --text_emb "$JECT_EMB" \
  --res 336 --out_res 336 --hires_head --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-20}" --lr "${LR:-2.5e-4}" --weight_decay 1e-4 --clip 5.0 --seed "${SEED:-0}" \
  --out "${OUT:-runs/gf336_revert_ft}"
