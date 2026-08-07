#!/bin/bash
# =============================================================================
# gf336_revert -- BEST SINGLE MODEL: val Dice 0.3103 / hit@0.1 0.7428 at epoch 16 of 19.
#
# First model in the project to pass the 224 muplogit result (0.3070). Only the 3-model
# ensemble + TTA (0.3316) scores higher, and that is not a single network.
#
# It is gf336_hires.sh with three changes, in descending order of what they were worth:
#
#   --enc_skips s2 s3   NOT `s1 s2 s3`. This is the whole experiment. gf336_hires took every
#                       decoder skip from the encoder on the theory that more phrase-carrying
#                       features is better; dropping s1 -- the finest, least semantic scale --
#                       is what the best 224 model (muplogit_224) had always done. Reverting to
#                       it took 336 from 0.2646 to 0.3103. `enc_skips` selects the SOURCE of a
#                       skip, not whether one exists, so this costs nothing in either direction.
#
#   --epochs 19         not 12. gf336_hires was still improving when its schedule ended, and
#                       OneCycleLR cannot be extended after launch. 19 was chosen to fit a day
#                       at the measured ~1.5 h/epoch. It converged with room to spare: the last
#                       four epochs sat in a 0.003 band (0.3103, 0.3061, 0.3031, 0.3040), so the
#                       schedule is no longer the binding constraint at this configuration.
#
#   --lr 1e-3           not 1.2e-3, matching muplogit_224 at the same batch 2. The 1.2e-3 in
#                       gf336_cooled was scaled for batch 3 and was inherited by gf336_hires
#                       without a batch change to justify it.
#
# TRAJECTORY (val best-threshold Dice / hit@0.1). Note ep8: a one-epoch collapse to 0.2177 that
# fully recovered. Do not kill this run on a single bad epoch.
#   ep1  0.1313/0.370   ep5  0.2244/0.583   ep8  0.2177/0.559  <- dip
#   ep10 0.2885/0.727   ep13 0.3016/0.738   ep16 0.3103/0.743  <- best
#   ep17 0.3061         ep18 0.3031         ep19 0.3040        <- converged
#
# The gain over the 224 model is RETRIEVAL: hit@0.1 0.7428 vs 0.7184, while Dice-among-hits
# stayed near 0.40 -- the plateau that every architecture in this project has hit so far.
#
# COST: ~1.5 h/epoch at batch 2, ~29 h total, ~104 GB. Needs a 140 GB card.
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
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-19}" --lr "${LR:-1e-3}" --weight_decay 1e-4 --clip 5.0 --seed "${SEED:-0}" \
  --out "${OUT:-runs/gf336_revert}"
