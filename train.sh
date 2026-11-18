#!/usr/bin/env bash
# Pretrain the JECT foundation model with the released settings.
#
#   DATA_CSV=manifest.csv DATA_ROOT=/data/cubes_256 SR_ROOT=/data/text/sr TARGET_STATS=/data/text/section_stats.pt \
#   OUT=runs/pretrain GPUS=0,1 bash train.sh
#
# Required: DATA_CSV, DATA_ROOT, SR_ROOT, TARGET_STATS, OUT.
# Optional: GPUS (default 0,1), PORT (default 29500), SMOKE=1 (24 steps of one epoch), DRY_RUN=1 (print the command).
# The effective batch is 32 (batch 4 x grad_accum 4 x 2 GPUs). With a different GPU count, change GRAD_ACCUM to keep it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${DATA_CSV:?set DATA_CSV}" "${DATA_ROOT:?set DATA_ROOT}" "${SR_ROOT:?set SR_ROOT}"
: "${TARGET_STATS:?set TARGET_STATS}" "${OUT:?set OUT}"
GPUS="${GPUS:-0,1}"; PORT="${PORT:-29500}"; NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
EPOCHS="${EPOCHS:-50}"; BATCH_SIZE="${BATCH_SIZE:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"; NUM_WORKERS="${NUM_WORKERS:-12}"

DATA=(--data_csv "$DATA_CSV" --data_root "$DATA_ROOT" --img_col object_id --split_col split --split train
      --output_dir "$OUT")
INPUT=(--depth 224 --height 224 --width 224 --channels 1 --patch_size 4 4 4)
ENCODER=(--variant huge --local_block_type swin --stage3_block_type swin --swin_window_size 7 7 7 --no_prototypes
         --skip_fine --skip_pool_s3 attention --skip_pool_s2s4 attention --no_ct_type_cond)
MASKING=(--mask_ratio 0.485 --num_target_blocks 40 --min_block_size_s3 4 4 4 --max_block_size_s3 7 7 7 --buffer_s3 0
         --no_random_fill --anatomy_sampling_prob 0)
JEPA=(--lambda_s1 0.25 --lambda_s2 0.35 --lambda_s3 0.40 --lambda_s4 0.30 --lambda_region_jepa 0
      --rope_scales s3 s4 --rope_predictor_depth 6)
ANTICOLLAPSE=(--lambda_var 1.0 --lambda_cov 0.04 --anticollapse sigreg --lambda_sigreg 0.02 --sigreg_proj_dim 256)
DENSE=(--use_dense_pred --lambda_dense 0.3 --dense_pred_scale s2 --dense_pred_scales s2 s3 --dense_pred_weights 0.15 0.1
       --dense_warmup_start 0.10 --dense_warmup_end 0.40)
TEXT=(--use_cached_text --cached_dim 2560 --utt_sidecar_suffix _utterances4b.pt --text_dim 512
      --predictor_cond xattn --predictor_xattn_scales s1 s2 s3 s4 --predictor_num_heads 8 --text_drop_prob 0.2
      --require_cached_text_alignment --allow_missing_cached_text)
MSP=(--use_masked_utterance_pred --mup_mask_prob 1.0 --mup_duty 1.0 --lambda_mup 0.5 --mup_scales s3 s4 --mup_depths 6 6
     --mup_spatial none --mup_persistent_routing_query --mup_section_mean_routing_query
     --mup_semantic_output_gate --mup_semantic_output_gate_normalized --mup_binding index
     --mup_loss clinical_innovation --mup_matryoshka_dim 1024 --mup_target_centering conditional
     --mup_condition_key section_name --mup_target_level section --mup_centering_ramp_frac 0.02711
     --mup_image_binding_weight 0.0 --mup_image_binding_margin 0.05
     --mup_conditional_nce_weight 0.0 --mup_conditional_nce_temperature 0.10
     --mup_section_nce_weight 0.0 --mup_section_nce_temperature 0.10
     --mup_section_geometry_weight 0.0 --mup_section_geometry_margin 0.05
     --mup_target_stats "$TARGET_STATS" --sr_sidecar_root "$SR_ROOT")
GROUNDING=(--use_grounding_head --grounding_scale s2 --lambda_weakground 0.5 --wg_max_regions 6 --max_regions 11
           --lambda_region_text 0)
TEXT_HEADS=(--text_heads_pass stochastic --p_clean 0.5 --text_ramp_frac 0.001356)
OPTIM=(--epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --num_workers "$NUM_WORKERS"
       --lr 7e-05 --weight_decay 0.05 --warmup_epochs 0.1354 --amp_dtype bf16
       --log_interval 20 --save_interval 1 --paired_step_seed 20260829)
if [ "${SMOKE:-0}" = 1 ]; then OPTIM+=(--epochs 1 --max_steps_per_epoch 24 --log_interval 1); fi

ARGS=("${DATA[@]}" "${INPUT[@]}" "${ENCODER[@]}" "${MASKING[@]}" "${JEPA[@]}" "${ANTICOLLAPSE[@]}" "${DENSE[@]}"
      "${TEXT[@]}" "${MSP[@]}" "${GROUNDING[@]}" "${TEXT_HEADS[@]}" "${OPTIM[@]}")
CMD=(torchrun --nproc_per_node="$NPROC" --master_port="$PORT" train.py "${ARGS[@]}")
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
CUDA_VISIBLE_DEVICES="$GPUS" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${CMD[@]}"
