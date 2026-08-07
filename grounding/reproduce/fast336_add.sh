#!/bin/bash
# =============================================================================
# fast336_add -- fast336 plus PROJECTED-ADD decoder skips. A second skeleton, mainly for ensembling.
#
# WHY A SECOND SKELETON. All eight architectures we have trained share one U-Net body; they differ
# only in the output path, the conditioning, and which priors get added. Their errors are therefore
# about as correlated as they can be -- yet a 3-model ensemble of same-skeleton 224 models still
# gained +0.033 (0.2986 -> 0.3316). Ensemble benefit scales with decorrelation, so a genuinely
# different way of mixing encoder and decoder features is the highest-value diversity available.
#
# WHAT CHANGES. UpBlock3D concatenates the skip (block input = out_ch + skip_ch). AddUpBlock3D
# projects the skip to out_ch with a 1x1x1 and ADDS it, holding peak width at out_ch instead of ~2x.
# Everything else -- transpose-conv upsample, shape guard, ConvBlock3D fuse -- is identical.
#
# THIS ALSO FIXES A CONFOUND IN OUR OWN RECORD. --sep_up4 scored 0.1397 and was abandoned, which
# looks like damning evidence against added skips. But it changed the fusion AND swapped in
# depthwise-separable convolutions. Depthwise convs are much weaker per parameter and are the
# likelier culprit. --skip_fusion add isolates the fusion so the two are attributable separately.
#
# THE GATE IS NOT "WIN". It is "good and different": within ~0.01 of fast336 is enough, because its
# job is to decorrelate, not to top the table. Judge it by the ensemble delta, not its solo Dice.
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
  --res 336 --out_res 168 --stem_stride2 --skip_fusion add --hi_ch 56 \
  --fusion xattn --s2_init pretrained --enc_skips s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  ${EXTRA:-} \
  --batch "${BS:-2}" --epochs "${EPOCHS:-20}" --lr 1e-3 --weight_decay 1e-4 --clip 5.0 --seed 0 \
  --out "${OUT:-runs/fast336_add}"
