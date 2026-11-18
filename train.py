"""Train JECT.

Examples
--------
Quick debug run on small volumes:
  python train.py --data_csv data.csv --data_root /data/preprocessed \
    --output_dir runs/ject_debug --depth 64 --height 64 --width 64 \
    --patch_size 4 4 4 --swin_window_size 2 2 2 --epochs 1 --max_steps_per_epoch 3

Main 224^3 stage:
  python train.py --data_csv data.csv --data_root /data/preprocessed \
    --output_dir runs/ject_224 --depth 224 --height 224 --width 224 \
    --patch_size 8 8 8 --swin_window_size 7 7 7 --variant small \
    --rope_scales s3 --use_aux_segmentation

High-resolution cooldown at 320^3:
  python train.py --data_csv data.csv --data_root /data/preprocessed \
    --output_dir runs/ject_320_cooldown --init_checkpoint runs/ject_224/checkpoint_final.pt \
    --depth 320 --height 320 --width 320 --patch_size 8 8 8 \
    --swin_window_size 5 5 5 --variant small --cooldown --use_aux_segmentation \
    --lr 2.5e-5 --epochs 5
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ject.config.ject import JECTConfig
from ject.data import StructuredReportCTDataset, ject_collate_fn
from ject.backbone.masking_ject import sample_ject_mask_batch
from ject.backbone.model import JECTModel
from ject.losses.ject import JECTLoss
from ject.checkpoint import load_flexible_checkpoint
from ject.backbone.anatomy import ANATOMY_ID_TO_NAME


# ---------------------------------------------------------------------------
# Distributed helpers (torchrun / DDP)
# ---------------------------------------------------------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def set_paired_step_seed(base_seed: int, epoch: int, step: int,
                         steps_per_epoch: int, rank: int, world_size: int) -> int:
    """Reset main-process RNGs to a path-independent per-batch state."""
    paired_index = int(epoch) * int(steps_per_epoch) + int(step)
    seed = (int(base_seed) + paired_index * int(world_size) + int(rank)) % (2**63 - 1)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def is_main():
    return get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Per-rank inverse-frequency weighted sampling (balanced + sharded, with replacement)."""

    def __init__(self, weights, world_size, rank, num_samples=None):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.world_size = world_size
        self.rank = rank
        self.num_samples = num_samples or (len(weights) // world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch * 10007 + self.rank)
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_triple(values, default):
    if values is None:
        return default
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected one value or three values")
    return tuple(values)


def build_config(args) -> JECTConfig:
    cfg = JECTConfig()
    cfg.volume.depth = args.depth
    cfg.volume.height = args.height
    cfg.volume.width = args.width
    cfg.volume.channels = args.channels
    cfg.patch.patch_size = parse_triple(args.patch_size, cfg.patch.patch_size)

    cfg.model.variant = args.variant
    cfg.model.local_block_type = args.local_block_type
    cfg.model.stage3_block_type = args.stage3_block_type
    cfg.model.pos_mode = args.pos_mode
    cfg.model.skip_fine = args.skip_fine
    cfg.model.skip_pool_s3 = args.skip_pool_s3
    cfg.model.skip_pool_s2s4 = args.skip_pool_s2s4
    cfg.model.jepa_target_source = args.jepa_target_source
    cfg.model.skip_legacy_rng = args.skip_legacy_rng
    if not args.skip_fine and (args.skip_pool_s3 != "maxmean" or args.skip_pool_s2s4 != "maxmean" or args.skip_legacy_rng):
        raise SystemExit("--skip_pool_s3/--skip_pool_s2s4/--skip_legacy_rng require --skip_fine")
    if args.pos_mode == "mm_rope" and args.spacing_mm is None:
        raise SystemExit("--pos_mode mm_rope requires --spacing_mm (mm per voxel of --data_root): "
                         "physical coordinates ARE the positional encoding.")
    cfg.model.swin_window_size = parse_triple(args.swin_window_size, cfg.model.swin_window_size)
    cfg.model.use_prototypes = not args.no_prototypes

    cfg.masking.mask_ratio = args.mask_ratio
    cfg.masking.num_target_blocks = args.num_target_blocks
    cfg.masking.min_block_size_s3 = parse_triple(args.min_block_size_s3, cfg.masking.min_block_size_s3)
    cfg.masking.max_block_size_s3 = parse_triple(args.max_block_size_s3, cfg.masking.max_block_size_s3)
    cfg.masking.buffer_s3 = args.buffer_s3
    # Random fill with buffer dilation can hide nearly all encoder tokens at a nominal 50% ratio;
    # --no_random_fill restricts targets to contiguous blocks.
    cfg.masking.allow_random_fill = not args.no_random_fill

    cfg.loss.lambda_s1 = args.lambda_s1
    cfg.loss.lambda_s2 = args.lambda_s2
    cfg.loss.lambda_s3 = args.lambda_s3
    cfg.loss.lambda_s4 = args.lambda_s4
    cfg.loss.lambda_proto = args.lambda_proto
    cfg.loss.lambda_var = args.lambda_var
    cfg.loss.anticollapse = args.anticollapse
    cfg.loss.lambda_cov = args.lambda_cov
    cfg.loss.lambda_sigreg = args.lambda_sigreg
    cfg.loss.use_dense_pred = args.use_dense_pred
    cfg.loss.lambda_dense = args.lambda_dense if args.use_dense_pred else 0.0
    cfg.loss.dense_pred_scale = args.dense_pred_scale
    cfg.loss.dense_pred_scales = tuple(args.dense_pred_scales) if args.dense_pred_scales else ()
    cfg.loss.dense_pred_weights = tuple(args.dense_pred_weights) if args.dense_pred_weights else ()
    cfg.loss.dense_warmup_start = args.dense_warmup_start
    cfg.loss.dense_warmup_end = args.dense_warmup_end
    if cfg.loss.dense_pred_weights and len(cfg.loss.dense_pred_weights) != len(cfg.loss.dense_pred_scales):
        raise SystemExit(f"--dense_pred_weights {args.dense_pred_weights} must match --dense_pred_scales "
                         f"{args.dense_pred_scales} in length")
    cfg.loss.sigreg_proj_dim = args.sigreg_proj_dim

    cfg.text.utterances_json_col = args.utterances_json_col
    cfg.text.findings_col = args.findings_col
    cfg.text.impression_col = args.impression_col
    cfg.text.vocab_size = args.text_vocab_size
    cfg.text.max_tokens = args.text_max_tokens
    cfg.text.text_dim = args.text_dim
    cfg.text.use_cached_embeddings = args.use_cached_text
    cfg.text.require_cached_text_alignment = args.require_cached_text_alignment
    cfg.text.predictor_cond = args.predictor_cond
    cfg.text.predictor_xattn_scales = tuple(args.predictor_xattn_scales)
    cfg.text.predictor_num_heads = args.predictor_num_heads
    cfg.text.text_drop_prob = args.text_drop_prob

    cfg.anatomy.num_anatomies = args.num_anatomies
    cfg.anatomy.anatomy_mask_key = args.anatomy_mask_key
    cfg.anatomy.seg_mask_key = args.seg_mask_key
    cfg.anatomy.use_aux_segmentation = args.use_aux_segmentation
    cfg.anatomy.num_seg_classes = args.num_seg_classes
    cfg.anatomy.use_grounding_head = args.use_grounding_head
    cfg.anatomy.grounding_num_heads = args.grounding_num_heads
    cfg.anatomy.grounding_scale = args.grounding_scale
    cfg.anatomy.wg_max_regions = args.wg_max_regions
    cfg.anatomy.wg_query = args.wg_query
    cfg.anatomy.wg_anatomy_emb = args.wg_anatomy_emb
    cfg.ject_loss.lambda_weakground = args.lambda_weakground
    cfg.anatomy.use_masked_utterance_pred = args.use_masked_utterance_pred
    cfg.anatomy.mup_mask_prob = args.mup_mask_prob
    cfg.anatomy.mup_single_target = args.mup_single_target
    cfg.anatomy.mup_duty = args.mup_duty
    cfg.anatomy.mup_ctx_drop_prob = args.mup_ctx_drop_prob
    cfg.anatomy.mup_scales = tuple(args.mup_scales)
    _mup_depths = args.mup_depths if len(args.mup_depths) != 1 else args.mup_depths * len(args.mup_scales)
    if len(_mup_depths) != len(args.mup_scales):
        raise SystemExit(f"--mup_depths {args.mup_depths} must match --mup_scales {args.mup_scales} in length")
    cfg.anatomy.mup_depths = tuple(_mup_depths)
    cfg.anatomy.mup_cond_pos = args.mup_cond_pos
    # Explicit --mup_spatial wins; if omitted, --mup_cond_pos maps to "additive", else "relbias".
    if args.mup_spatial is not None:
        cfg.anatomy.mup_spatial = args.mup_spatial
    else:
        cfg.anatomy.mup_spatial = "additive" if args.mup_cond_pos else "relbias"
    cfg.anatomy.mup_persistent_routing_query = args.mup_persistent_routing_query
    cfg.anatomy.mup_section_mean_routing_query = args.mup_section_mean_routing_query
    cfg.anatomy.mup_semantic_output_gate = args.mup_semantic_output_gate
    cfg.anatomy.mup_semantic_output_gate_normalized = args.mup_semantic_output_gate_normalized
    cfg.anatomy.joint_predictor = args.joint_predictor
    cfg.anatomy.mup_binding = args.mup_binding
    cfg.anatomy.mup_loss = args.mup_loss
    cfg.anatomy.mup_matryoshka_dim = args.mup_matryoshka_dim
    cfg.anatomy.mup_target_stats = args.mup_target_stats
    cfg.anatomy.mup_target_centering = args.mup_target_centering
    cfg.anatomy.mup_condition_key = args.mup_condition_key
    cfg.anatomy.mup_conditional_mean_shrinkage = args.mup_conditional_mean_shrinkage
    cfg.anatomy.mup_image_binding_weight = args.mup_image_binding_weight
    cfg.anatomy.mup_image_binding_margin = args.mup_image_binding_margin
    cfg.anatomy.mup_conditional_nce_weight = args.mup_conditional_nce_weight
    cfg.anatomy.mup_conditional_nce_temperature = args.mup_conditional_nce_temperature
    cfg.anatomy.mup_section_nce_weight = args.mup_section_nce_weight
    cfg.anatomy.mup_section_nce_temperature = args.mup_section_nce_temperature
    cfg.anatomy.mup_section_geometry_weight = args.mup_section_geometry_weight
    cfg.anatomy.mup_section_geometry_margin = args.mup_section_geometry_margin
    if args.cached_dim is not None:
        cfg.text.cached_dim = args.cached_dim
    if args.utt_sidecar_suffix is not None:
        cfg.text.utterances_sidecar_suffix = args.utt_sidecar_suffix
    if args.mup_target_level is not None:
        cfg.text.mup_target_level = args.mup_target_level
    if args.sr_sidecar_root is not None:
        cfg.text.sr_sidecar_root = args.sr_sidecar_root
    if args.max_regions is not None:
        cfg.anatomy.max_regions = args.max_regions
    cfg.anatomy.text_heads_pass = args.text_heads_pass
    cfg.anatomy.mup_p_clean = args.p_clean
    cfg.anatomy.text_ramp_frac = args.text_ramp_frac
    if args.text_heads_pass == "stochastic" and not (0.0 < args.p_clean <= 1.0):
        raise SystemExit(f"--p_clean must be in (0, 1] for --text_heads_pass stochastic (got {args.p_clean}); "
                         f"otherwise the text heads never run and the encoder trains image-only.")
    cfg.ject_loss.lambda_mup = args.lambda_mup

    cfg.rope.use_rope_predictor = not args.no_rope_predictor
    cfg.rope.rope_scales = tuple(args.rope_scales)
    cfg.rope.predictor_depth = args.rope_predictor_depth

    cfg.ject_loss.lambda_anatomy_region_text = args.lambda_region_text
    cfg.ject_loss.lambda_seg_aux = args.lambda_seg_aux
    cfg.ject_loss.lambda_region_jepa = args.lambda_region_jepa
    cfg.ject_loss.region_jepa_loss = args.region_jepa_loss
    cfg.ject_loss.anatomy_mask_frac = args.anatomy_mask_frac
    if args.cooldown:
        cfg.cooldown.enabled = True
        cfg.loss.lambda_s1 = cfg.ject_loss.cooldown_lambda_s1
        cfg.loss.lambda_s2 = cfg.ject_loss.cooldown_lambda_s2
        cfg.loss.lambda_s3 = cfg.ject_loss.cooldown_lambda_s3
        cfg.ject_loss.lambda_anatomy_region_text = cfg.ject_loss.cooldown_lambda_text
        cfg.ject_loss.lambda_seg_aux = cfg.ject_loss.cooldown_lambda_seg_aux if args.use_aux_segmentation else 0.0

    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = args.num_workers
    cfg.training.total_epochs = args.epochs
    cfg.training.base_lr = args.lr
    cfg.training.weight_decay = args.weight_decay
    cfg.training.mixed_precision = not args.no_amp
    cfg.training.log_interval = args.log_interval
    cfg.training.save_interval = args.save_interval

    cfg.data.csv_path = args.data_csv
    cfg.data.data_root = args.data_root
    cfg.data.img_col = args.img_col
    cfg.data.split_col = args.split_col
    cfg.data.split = args.split
    cfg.data.vary_window = args.vary_window
    cfg.data.aug_intensity = not args.no_aug_intensity

    cfg.__post_init__()
    return cfg


def config_to_dict(cfg: JECTConfig) -> dict:
    return {
        "volume": vars(cfg.volume), "patch": vars(cfg.patch), "model": vars(cfg.model),
        "masking": vars(cfg.masking), "loss": vars(cfg.loss), "ema": vars(cfg.ema),
        "training": vars(cfg.training), "data": vars(cfg.data), "text": vars(cfg.text),
        "anatomy": vars(cfg.anatomy), "rope": vars(cfg.rope), "ject_loss": vars(cfg.ject_loss),
        "cooldown": vars(cfg.cooldown),
    }


def save_checkpoint(path: Path, model: JECTModel, optimizer, epoch: int, step: int, cfg: JECTConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "step": step, "config": config_to_dict(cfg),
    }, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JECT pretraining")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--img_col", default="object_id")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--split", default="train")
    parser.add_argument("--no_ct_type_cond", action="store_true",
                        help="Do not feed the CT-type id to the encoders; the additive CT-type embedding "
                             "is skipped and not trained.")
    parser.add_argument("--balance_by", choices=["none", "ct_type", "source"], default="none",
                        help="Inverse-frequency weighted sampling to equalize the ct_type or source column.")
    parser.add_argument("--vary_window", action="store_true",
                        help="Random HU window per sample (default: fixed general window, which keeps text grounding valid).")
    parser.add_argument("--no_aug_intensity", action="store_true",
                        help="Disable the mild HU jitter and noise augmentation.")
    parser.add_argument("--output_dir", default="runs/ject")

    parser.add_argument("--depth", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--patch_size", type=int, nargs="+", default=[8, 8, 8])
    parser.add_argument("--variant", choices=["tiny", "small", "base", "large", "huge"], default="tiny")
    parser.add_argument("--local_block_type", choices=["swin", "transformer", "convnext"], default="swin")
    parser.add_argument("--stage3_block_type", choices=["global", "swin", "transformer"], default="global")
    parser.add_argument("--skip_fine", action="store_true",
                        help="Add cumulative bottom-up skips into the s2/s3/s4 stage inputs (max+mean pools of every "
                             "finer representation per cell, zero-init projections).")
    parser.add_argument("--skip_pool_s3", choices=["maxmean", "attention"], default="maxmean",
                        help="Replace only the stage-3 fine-skip pooling with local spatial attention.")
    parser.add_argument("--skip_pool_s2s4", choices=["maxmean", "attention"], default="maxmean",
                        help="Pooling for the stage-2 and stage-4 fine-skip entries: max+mean or local spatial attention.")
    parser.add_argument("--jepa_target_source", choices=["decoder", "encoder"], default="decoder",
                        help="Source of the s1-s3 JEPA targets: the EMA decoder outputs or the EMA encoder stages e1-e3.")
    parser.add_argument("--skip_legacy_rng", action="store_true",
                        help="Build the skip modules outside torch.random.fork_rng, so their initialization "
                             "consumes the global RNG stream.")
    parser.add_argument("--pos_mode", choices=["learned", "mm_rope"], default="learned",
                        help="Absolute position encoding. 'learned': per-token tables tied to the grid. "
                             "'mm_rope': physical-millimeter coordinates (CoordEmbed3D at s3 plus RoPE in "
                             "stage 4), so one checkpoint serves any grid size and voxel spacing.")
    parser.add_argument("--spacing_mm", type=float, default=None,
                        help="Isotropic voxel spacing in mm of the volumes in --data_root; required for mm_rope.")
    parser.add_argument("--swin_window_size", type=int, nargs="+", default=[7, 7, 7])

    parser.add_argument("--mask_ratio", type=float, default=0.50)
    parser.add_argument("--num_target_blocks", type=int, default=4)
    parser.add_argument("--min_block_size_s3", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument("--max_block_size_s3", type=int, nargs="+", default=[2, 3, 3])
    parser.add_argument("--buffer_s3", type=int, default=1)
    parser.add_argument("--no_random_fill", action="store_true",
                        help="Do not random-fill scattered tokens up to --mask_ratio; targets are the sampled "
                             "cuboids only (raise --num_target_blocks / --min_block_size_s3 to reach the ratio "
                             "with contiguous blocks). With random fill, buffer dilation can hide nearly all "
                             "encoder tokens at a nominal ratio of 0.5.")
    parser.add_argument("--anatomy_sampling_prob", type=float, default=0.75)

    parser.add_argument("--lambda_s1", type=float, default=0.25)
    parser.add_argument("--lambda_s2", type=float, default=0.35)
    parser.add_argument("--lambda_s3", type=float, default=0.40)
    parser.add_argument("--lambda_s4", type=float, default=0.30, help="Global s4-scale JEPA (encoder e4).")
    parser.add_argument("--lambda_proto", type=float, default=0.10)
    parser.add_argument("--lambda_var", type=float, default=1.0,
                        help="VICReg variance-hinge weight.")
    parser.add_argument("--lambda_cov", type=float, default=0.04, help="VICReg covariance (decorrelation) weight.")
    parser.add_argument("--anticollapse", choices=["vicreg", "sigreg", "both"], default="sigreg",
                        help="Anti-collapse regularizer: SIGReg (LeJEPA), VICReg, or both.")
    parser.add_argument("--sigreg_proj_dim", type=int, default=256,
                        help="SIGReg projector width (0 = regularize the raw latents). The projector protects "
                             "local structure used by the dense and grounding heads.")
    parser.add_argument("--lambda_sigreg", type=float, default=0.05,
                        help="SIGReg (LeJEPA) weight. The Epps-Pulley `* N` factor is kept and the token "
                             "sample is capped at 512, so values match the paper's bstat_lambda (0.01-0.1).")
    parser.add_argument("--use_dense_pred", action="store_true",
                        help="Enable the dense predictive loss (LayerNorm'd L2 on visible tokens at the "
                             "grounding scale, weighted by proximity to the mask boundary).")
    parser.add_argument("--lambda_dense", type=float, default=0.3,
                        help="Weight for the dense predictive loss (only active with --use_dense_pred).")
    parser.add_argument("--dense_pred_scale", choices=["s1", "s2", "s3"], default="s2",
                        help="Single dense-loss scale, used only if --dense_pred_scales is omitted.")
    parser.add_argument("--dense_pred_scales", nargs="+", choices=["s1", "s2", "s3"], default=None,
                        help="Dense-loss scales. s4 is not supported (global scale, no per-token mask).")
    parser.add_argument("--dense_pred_weights", nargs="+", type=float, default=None,
                        help="Per-scale dense weights relative to each scale's masked-JEPA weight, paired "
                             "with --dense_pred_scales. If omitted, --lambda_dense is split evenly.")
    parser.add_argument("--dense_warmup_start", type=float, default=0.10,
                        help="Dense loss stays 0 until this fraction of training, then ramps.")
    parser.add_argument("--dense_warmup_end", type=float, default=0.40,
                        help="Dense loss reaches full weight by this fraction of training.")
    parser.add_argument("--lambda_region_text", type=float, default=0.3,
                        help="Region-text contrastive loss weight.")
    parser.add_argument("--lambda_seg_aux", type=float, default=0.10)
    parser.add_argument("--lambda_region_jepa", type=float, default=0.5,
                        help="Weight of the anatomy-pooled JEPA loss (region-level target).")
    parser.add_argument("--region_jepa_loss", choices=["cosine", "l1", "l2"], default="cosine")
    parser.add_argument("--anatomy_mask_frac", type=float, default=0.5,
                        help="Fraction of the referenced anatomy that is masked (rest stays visible).")
    parser.add_argument("--no_prototypes", action="store_true")

    parser.add_argument("--utterances_json_col", default="utterances_json")
    parser.add_argument("--findings_col", default="findings")
    parser.add_argument("--impression_col", default="impression")
    parser.add_argument("--text_vocab_size", type=int, default=32768)
    parser.add_argument("--text_max_tokens", type=int, default=96)
    parser.add_argument("--text_dim", type=int, default=256)
    parser.add_argument("--use_cached_text", action="store_true",
                        help="Use cached {id}_utterances.pt text embeddings instead of the hashing encoder.")
    parser.add_argument("--predictor_cond", choices=["film", "xattn"], default="film",
                        help="Predictor conditioning: 'film' (spatially uniform) or 'xattn' (gated "
                             "image-to-text cross-attention, spatial; needs --use_cached_text).")
    parser.add_argument("--predictor_xattn_scales", nargs="+", default=["s1", "s2"], choices=["s1", "s2", "s3", "s4"],
                        help="Scales that use cross-attention with --predictor_cond xattn (s3 keeps RoPE + FiLM).")
    parser.add_argument("--predictor_num_heads", type=int, default=8)
    parser.add_argument("--text_drop_prob", type=float, default=0.2,
                        help="Probability of replacing the predictor's text conditioning with the null "
                             "input during training (classifier-free guidance).")
    parser.add_argument("--num_anatomies", type=int, default=32)
    parser.add_argument("--anatomy_mask_key", default="anat_masks")
    parser.add_argument("--seg_mask_key", default="seg_mask")
    parser.add_argument("--use_aux_segmentation", action="store_true")
    parser.add_argument("--num_seg_classes", type=int, default=2)

    # Utterance-query grounding head and weak-grounding loss.
    parser.add_argument("--use_grounding_head", action="store_true",
                        help="Enable the utterance-query grounding head (per-utterance localization maps "
                             "plus a null query). Needs --use_cached_text.")
    parser.add_argument("--grounding_num_heads", type=int, default=8,
                        help="Attention heads in the grounding head (must divide the grounding-scale dim).")
    parser.add_argument("--grounding_scale", choices=["s1", "s2", "s3"], default="s2",
                        help="Feature scale used by the grounding and MUP heads.")
    parser.add_argument("--wg_max_regions", type=int, default=0,
                        help="Size of a separate utterance and anatomy-mask slot set for the grounding head, "
                             "so weak grounding can run with section/section_report MUP targets. "
                             "0 = the grounding head reads the MUP slots.")
    parser.add_argument("--wg_query", choices=["sentence", "anatomy_phrase"], default="sentence",
                        help="Weak-grounding query: the patient-specific sentence embedding or a fixed "
                             "phrase embedding per anatomy id.")
    parser.add_argument("--wg_anatomy_emb", default="",
                        help="Path to the [num_anatomies, cached_dim] phrase-embedding table for --wg_query anatomy_phrase.")
    parser.add_argument("--lambda_weakground", type=float, default=0.5,
                        help="Weak-grounding loss weight (per-utterance map vs anatomy region, null "
                             "query vs region union).")
    parser.add_argument("--use_masked_utterance_pred", action="store_true",
                        help="Masked-utterance prediction (MUP) through the grounder's cross-attention. "
                             "Needs --use_cached_text.")
    parser.add_argument("--mup_mask_prob", type=float, default=0.5,
                        help="Fraction of valid utterances masked+predicted each step.")
    parser.add_argument("--mup_single_target", action="store_true",
                        help="Mask and predict exactly one valid utterance per sample instead of "
                             "per-slot Bernoulli(--mup_mask_prob).")
    parser.add_argument("--mup_ctx_drop_prob", type=float, default=0.0,
                        help="Synced section dropout (reconstructed route only): drop each valid section with "
                             "this probability; the JEPA predictor conditions on the kept sections and MUP "
                             "predicts the dropped ones. 0 = off.")
    parser.add_argument("--mup_duty", type=float, default=1.0,
                        help="For --text_heads_pass corrupted/reconstructed: fraction of optimizer windows "
                             "that run the text heads. 1.0 = every step.")
    parser.add_argument("--lambda_mup", type=float, default=0.5, help="Masked-utterance-prediction loss weight.")
    parser.add_argument("--mup_scales", nargs="+", default=["s2"], choices=["s1", "s2", "s3", "s4"],
                        help="Image scales at which to run masked-utterance prediction.")
    parser.add_argument("--mup_depths", nargs="+", type=int, default=[1],
                        help="MUP head depth (cross-attention + MLP blocks) per --mup_scales entry; a "
                             "single value applies to all scales.")
    parser.add_argument("--mup_cond_pos", action="store_true",
                        help="Add the region-centroid location to the MUP query (same as --mup_spatial "
                             "additive). Leaks the centroid to the output; prefer --mup_spatial relbias.")
    parser.add_argument("--text_heads_pass", default="clean",
                        choices=["clean", "probe", "stochastic", "corrupted", "reconstructed"],
                        help="Feature source for weak grounding and MUP. 'stochastic': JEPA every step, "
                             "unmasked student pass on --p_clean of steps. 'clean': unmasked student pass "
                             "every step. 'reconstructed': one masked pass; image-only JEPA predictions fill "
                             "the masked latents before the text heads. 'probe': detached EMA target, "
                             "readout only.")
    parser.add_argument("--p_clean", type=float, default=0.25,
                        help="Stochastic mode: probability that a step runs the unmasked student pass.")
    parser.add_argument("--text_ramp_frac", type=float, default=0.05,
                        help="Linearly ramp the text losses (MUP, weak grounding) from 0 over this fraction "
                             "of optimizer steps, so the text target does not dominate the encoder early.")
    parser.add_argument("--mup_spatial", default=None,
                        choices=["none", "additive", "relbias", "rope"],
                        help="Anatomy pathway for the MUP cross-attention. relbias: region centroid, extent "
                             "and occupancy enter only the attention logits. rope: factorized 3D RoPE. "
                             "additive: centroid added to the query. none: no spatial mechanism. "
                             "Default: relbias, or additive with --mup_cond_pos.")
    parser.add_argument("--mup_persistent_routing_query", action="store_true",
                        help="For section targets with --mup_spatial none, reapply section identity only "
                             "inside every attention Q projection; it never enters the output value/residual.")
    parser.add_argument("--mup_section_mean_routing_query", action="store_true",
                        help="Use the train-split section mean as the persistent attention-routing query. "
                             "Requires conditional centering; the mean never enters the value path.")
    parser.add_argument("--mup_semantic_output_gate", action="store_true",
                        help="Multiplicatively gate the image-derived MUP state with the section query. "
                             "Adds no parameters.")
    parser.add_argument("--joint_predictor", choices=["off", "shared", "split"], default="off",
                        help="'shared': the JEPA predictor also performs MUP; 'split': a second predictor of "
                             "the same architecture for MUP; 'off': a separate MUP head.")
    parser.add_argument("--mup_semantic_output_gate_normalized", action="store_true",
                        help="Layer-normalize the semantic gate source and bound its multiplicative "
                             "image-state modulation to [0.5, 1.5].")
    parser.add_argument("--cached_dim", type=int, default=None,
                        help="Cached text-embedding dimension (default 1024).")
    parser.add_argument("--utt_sidecar_suffix", default=None,
                        help="Utterance sidecar filename suffix (default '_utterances.pt').")
    parser.add_argument("--mup_centering_ramp_frac", type=float, default=0.0,
                        help="Anneal the SCIR centering mean from 0 to full over this fraction of training "
                             "(0 = off), so training starts close to raw cosine.")
    parser.add_argument("--mup_target_level", default=None,
                        choices=["utterance", "section", "report", "section_report"],
                        help="MUP target granularity. 'section': one slot per canonical report section "
                             "(11 slots) from the section/report sidecars. 'report': one whole-report target "
                             "per study. 'section_report': section slots plus a whole-report slot.")
    parser.add_argument("--sr_sidecar_root", default=None,
                        help="Directory of {id}_sr_targets4b.pt section/report sidecars "
                             "(required for section/report target levels).")
    parser.add_argument("--max_regions", type=int, default=None,
                        help="Region/MUP slot count (default 6; section level needs >= 11).")
    parser.add_argument("--mup_loss", default="raw_cosine",
                        choices=["raw_cosine", "clinical_innovation"],
                        help="MUP loss. clinical_innovation (CIR): residual-direction regression against "
                             "the train-mean prior, weighted by residual norm; requires "
                             "--mup_matryoshka_dim and --mup_target_stats.")
    parser.add_argument("--mup_matryoshka_dim", type=int, default=0,
                        help="Matryoshka prefix for cached targets (slice, then L2-normalize). "
                             "0 = full sidecar dim.")
    parser.add_argument("--mup_target_stats", default="",
                        help="Torch file with per-dim train-split target means. Required for "
                             "clinical_innovation.")
    parser.add_argument("--mup_target_centering", default="global", choices=["global", "conditional"],
                        help="Centering for clinical_innovation targets. global subtracts E[x]; "
                             "conditional subtracts E[x | section_name].")
    parser.add_argument("--mup_condition_key", default="section_name", choices=["section_name"],
                        help="Condition key for --mup_target_centering conditional (section_name only).")
    parser.add_argument("--mup_conditional_mean_shrinkage", type=float, default=0.0,
                        help="Metadata value for train-only conditional target means; use the same value "
                             "used by the stats builder.")
    parser.add_argument("--mup_image_binding_weight", type=float, default=0.0,
                        help="Relative weight of the correct-image vs deranged-image MUP ranking loss. "
                             "0 disables it.")
    parser.add_argument("--mup_image_binding_margin", type=float, default=0.05,
                        help="Required cosine-score advantage of the correct study image over a "
                             "deranged study image for the same masked target.")
    parser.add_argument("--mup_conditional_nce_weight", type=float, default=0.0,
                        help="Relative weight of same-section in-batch patient matching. "
                             "Requires section targets with conditional centering.")
    parser.add_argument("--mup_conditional_nce_temperature", type=float, default=0.10,
                        help="Temperature for same-section in-batch patient matching.")
    parser.add_argument("--mup_section_nce_weight", type=float, default=0.0,
                        help="Relative weight of within-study section matching. Requires section "
                             "targets with conditional centering.")
    parser.add_argument("--mup_section_nce_temperature", type=float, default=0.10,
                        help="Temperature for within-study section matching.")
    parser.add_argument("--mup_section_geometry_weight", type=float, default=0.0,
                        help="Weight for target-calibrated within-study cross-section geometry.")
    parser.add_argument("--mup_section_geometry_margin", type=float, default=0.05,
                        help="Allowed predicted-minus-target pair-cosine slack.")
    parser.add_argument("--mup_binding", default="index", choices=["index", "matched"],
                        help="MUP target binding. index: slot j predicts utterance j. matched: "
                             "optimal-assignment loss over the masked set (exact over k! permutations, k <= 6).")
    parser.add_argument("--allow_missing_cached_text", action="store_true",
                        help="Allow a --use_cached_text run with low cached-text coverage (otherwise "
                             "training aborts after epoch 0).")
    parser.add_argument("--require_cached_text_alignment", action="store_true",
                        help="Reject sidecars without a stored `texts` list, so only sidecars with "
                             "verifiable index alignment to the current utterances are used.")
    parser.add_argument("--collapse_var_threshold", type=float, default=0.9,
                        help="After warmup, abort before checkpointing if the per-scale mean variance hinge "
                             "relu(1 - std) exceeds this (per-channel std collapsing toward 0). "
                             "Override with --allow_collapse.")
    parser.add_argument("--collapse_corr_threshold", type=float, default=0.85,
                        help="Abort after warmup if the mean absolute off-diagonal channel correlation "
                             "(in [0, 1]) exceeds this (correlated collapse). Override with --allow_collapse.")
    parser.add_argument("--allow_collapse", action="store_true",
                        help="Disable the collapse guard (checkpoint even if variance or correlation collapsed).")
    parser.add_argument("--min_weakground_coverage", type=float, default=0.5,
                        help="Abort after epoch 0 if the fraction of samples (across all ranks) with a "
                             "weak-grounding target, or with cached text, is below this. 0 disables.")

    parser.add_argument("--no_rope_predictor", action="store_true")
    parser.add_argument("--rope_scales", nargs="+", default=["s3"], choices=["s1", "s2", "s3", "s4"])
    parser.add_argument("--rope_predictor_depth", type=int, default=1)

    parser.add_argument("--cooldown", action="store_true")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint: restores model, optimizer, EMA and LR-schedule "
                             "position and continues from the saved epoch. --init_checkpoint loads model "
                             "weights only.")
    parser.add_argument("--strict_init", action="store_true")
    parser.add_argument("--allow_conditioner_reset", action="store_true",
                        help="Allow loading an anatomy-conditioned checkpoint into the text-only "
                             "conditioner (conditioning parameters are re-initialized).")
    parser.add_argument("--allow_arch_reset", action="store_true",
                        help="Allow an init/resume where the checkpoint lacks predictor-fusion, grounding or "
                             "MUP parameters the model expects (e.g. adding grounding to a plain JEPA "
                             "backbone, or switching --predictor_cond). Those modules stay randomly initialized.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation micro-steps; effective batch = batch_size x world_size "
                             "x grad_accum. Scheduler and EMA count optimizer steps.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                        help="Autocast dtype. auto = bf16 if supported (fp16 overflows above 65504).")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Cosine floor as a fraction of the base lr. Raise it (e.g. 0.1) for short "
                             "cooldowns so the end of the schedule still updates the weights.")
    parser.add_argument("--warmup_epochs", type=float, default=2.0,
                        help="Linear LR warmup epochs before cosine decay.")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--paired_step_seed", type=int, default=None,
        help="Optional base seed that resets Python/PyTorch RNGs before each training batch, so "
             "runs with different code paths see identical masks and random draws.")
    parser.add_argument("--torch_num_threads", type=int, default=None,
                        help="Number of torch CPU threads.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.torch_num_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_num_threads)))
    setup_distributed()
    rank0 = is_main()
    torch.manual_seed(args.seed + get_rank())      # per-rank seed (independent masks/aug)
    cfg = build_config(args)
    out_dir = Path(args.output_dir)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_to_dict(cfg), f, indent=2, default=str)

    if is_dist():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if rank0:
        print("=" * 88)
        print("JECT")
        print(f"device={device} world_size={get_world_size()} variant={cfg.model.variant} volume={cfg.volume.spatial_size} patch={cfg.patch.patch_size}")
        print(f"grids: S1={cfg.patch.grid_size_s1} S2={cfg.patch.grid_size_s2} S3={cfg.patch.grid_size_s3}")
        print(f"swin_window={cfg.model.swin_window_size} rope_scales={cfg.rope.rope_scales} cooldown={cfg.cooldown.enabled}")
        print(f"loss lambdas: s1={cfg.loss.lambda_s1} s2={cfg.loss.lambda_s2} s3={cfg.loss.lambda_s3} text={cfg.ject_loss.lambda_anatomy_region_text} seg={cfg.ject_loss.lambda_seg_aux}")
        print(f"anti-collapse: mode={cfg.loss.anticollapse} lambda_var={cfg.loss.lambda_var} lambda_cov={cfg.loss.lambda_cov} lambda_sigreg={cfg.loss.lambda_sigreg}")
        print(f"grounding-head: {cfg.anatomy.use_grounding_head} (heads={cfg.anatomy.grounding_num_heads} lambda_weakground={cfg.ject_loss.lambda_weakground})")
        print(f"predictor-cond: {cfg.text.predictor_cond}" + (f" (xattn scales={cfg.text.predictor_xattn_scales})" if cfg.text.predictor_cond == 'xattn' else ""))
        print("=" * 88)

    dataset = StructuredReportCTDataset(cfg)
    world, rank = get_world_size(), get_rank()
    # Inverse-frequency weighting on a chosen column (ct_type / source) equalizes imbalanced groups.
    sampler = None
    if args.balance_by != "none" and args.balance_by in dataset.df.columns:
        counts = collections.Counter(dataset.df[args.balance_by].astype(str))
        weights = dataset.df[args.balance_by].astype(str).map(lambda k: 1.0 / counts[k]).tolist()
        sampler = (DistributedWeightedSampler(weights, world, rank) if is_dist()
                   else torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True))
        if rank0:
            print(f"balanced sampling by '{args.balance_by}': {dict(counts)}")
    elif is_dist():
        sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.training.num_workers, collate_fn=ject_collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=cfg.training.prefetch_factor if cfg.training.num_workers > 0 else None,
    )

    raw_model = JECTModel(cfg).to(device)
    if args.init_checkpoint:
        load_flexible_checkpoint(raw_model, args.init_checkpoint, strict=args.strict_init, verbose=rank0,
                                 allow_conditioner_reset=args.allow_conditioner_reset,
                                 allow_arch_reset=args.allow_arch_reset)
    if is_dist():
        model = DDP(raw_model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    gradient_as_bucket_view=True, find_unused_parameters=True)
    else:
        model = raw_model
    loss_fn = JECTLoss(cfg)
    optimizer = torch.optim.AdamW(raw_model.get_param_groups(cfg.training.base_lr, cfg.training.weight_decay))
    # bf16 has the fp32 exponent range, so activations cannot overflow as in fp16 (max 65504), and it
    # needs no loss scaling. GradScaler is enabled only for the fp16 fallback; disabled, it is a no-op.
    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    if rank0:
        print(f"[amp] enabled={amp_enabled} dtype={amp_dtype} grad_scaler={scaler.is_enabled()}", flush=True)
    steps_per_epoch = len(dataloader)
    # Scheduler and EMA count optimizer steps (committed accumulation windows), not microbatches.
    # Partial end-of-epoch windows are committed, so this is ceil(steps_per_epoch / accum).
    accum = max(1, args.grad_accum)
    optim_steps_per_epoch = max(1, -(-steps_per_epoch // accum))          # ceil division
    total_steps = max(1, optim_steps_per_epoch * cfg.training.total_epochs)
    scheduler = build_scheduler(optimizer, total_steps,
                                int(args.warmup_epochs * optim_steps_per_epoch),
                                min_lr_ratio=args.min_lr_ratio)
    # Tie the EMA momentum schedule to the actual run length instead of the config default.
    raw_model.ema.total_steps = total_steps

    # Resume: restore model + optimizer + EMA-step + LR-schedule position and continue from the saved epoch.
    start_epoch, resume_step = 0, 0
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(rck["model"])                        # exact weights (incl. EMA target)
        optimizer.load_state_dict(rck["optimizer"])
        start_epoch = int(rck.get("epoch", 0))                         # saved as epoch+1, i.e. the next epoch to run
        resume_step = int(rck.get("step", start_epoch * optim_steps_per_epoch))
        raw_model.ema.step = resume_step                              # keep the momentum schedule aligned
        for _ in range(resume_step):                                  # fast-forward the LR schedule
            scheduler.step()
        if rank0:
            print(f"[resume] {args.resume}: continuing from epoch {start_epoch} (step {resume_step})", flush=True)

    if rank0:
        eff_bs = cfg.training.batch_size * get_world_size() * accum
        print(f"[batch] per-gpu={cfg.training.batch_size} × world={get_world_size()} × accum={accum} = eff {eff_bs}")
        print(f"[ema] momentum {cfg.ema.momentum_start}->{cfg.ema.momentum_end} over {total_steps} optim-steps ({cfg.training.total_epochs} ep)")

    def _commit_window(n_micro):
        """Apply one optimizer step for an accumulation window of ``n_micro`` microbatches.

        Returns the EMA momentum if the weights were updated, else None. Gradients can be non-finite
        even when the loss is finite; such windows are skipped, as are fp16 steps that GradScaler
        skips (detected by the scale drop). Skipped windows do not advance the scheduler, EMA or
        global_step.

        Each microbatch loss is divided by ``accum`` at backward time, so a full window sums to the
        mean gradient. A partial window (n_micro < accum, at end of epoch or max-step truncation) is
        rescaled by accum / n_micro, giving the mean over the microbatches that contributed."""
        scaler.unscale_(optimizer)
        if 0 < n_micro < accum:
            corr = accum / n_micro
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(corr)
        # Finite-gradient guard for every amp dtype. The scale-drop check below only covers fp16; with
        # bf16 the scaler is disabled, so non-finite grads from a finite loss would otherwise reach the
        # optimizer. clip_grad_norm_ returns the pre-clip total norm, which is non-finite iff any grad
        # is. All-reduce so every rank makes the same skip decision.
        gmax = cfg.training.grad_clip_norm if cfg.training.grad_clip_norm > 0 else float("inf")
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), gmax)
        gfinite = torch.isfinite(gnorm.detach().to(torch.float32)).float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gfinite, op=dist.ReduceOp.MIN)
        if gfinite.item() < 1.0:                          # non-finite grads: skip without advancing anything
            optimizer.zero_grad(set_to_none=True)
            return None
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:     # scale dropped, so optimizer.step() was skipped
            return None
        scheduler.step()
        return raw_model.update_target()

    global_step = resume_step
    nonfinite_skips = 0
    nonfinite_by_loss = {}
    momentum = cfg.ema.momentum_start
    wg_active_steps = 0                      # steps where weak-grounding had a nonzero (supervised) loss
    for epoch in range(start_epoch, cfg.training.total_epochs):
        model.train()
        micro, window_ok, win_micro = 0, True, 0   # gradient-accumulation window state (reset each epoch)
        text_clean_step = True                     # stochastic clean-view schedule (set per optim window)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        cov = {"n": 0, "ge2": 0, "has_region": 0, "regions": 0, "neg": 0, "unc": 0, "text_ok": 0,
               "mask_loss": 0.0, "mask_enc": 0.0, "mask_n": 0,
               "anat_valid": 0, "var_sum": 0.0, "var_n": 0, "corr_sum": 0.0, "corr_n": 0,
               "anat": collections.Counter()}
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not rank0)
        for step, batch in enumerate(pbar):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            if args.paired_step_seed is not None:
                # Reseed independently of RNG consumed by the previous forward, so different code
                # paths see the same mask and text-drop stream at every batch.
                set_paired_step_seed(
                    args.paired_step_seed, epoch, step, len(dataloader),
                    get_rank(), get_world_size())
            # --- region-text coverage stats (from collated CPU tensors) ---
            cov["n"] += batch["volume"].shape[0]
            cov["neg"] += int(batch["negated"].sum())
            if batch.get("anatomy_mask_valid") is not None:      # anatomy-guided mask activation coverage
                cov["anat_valid"] += int(batch["anatomy_mask_valid"].sum())
            cov["unc"] += int(batch["uncertain"].sum())
            # Cached-text coverage: a missing sidecar yields an all-zero (null) embedding, so track how
            # many samples actually carry text.
            tec = batch.get("text_emb_cached")
            if tec is not None:
                cov["text_ok"] += int((tec.abs().sum(dim=1) > 0).sum())
            rv = batch.get("region_valid")
            if rv is not None:
                per_vol = rv.sum(dim=1)
                cov["ge2"] += int((per_vol >= 2).sum())
                cov["has_region"] += int((per_vol >= 1).sum())      # samples that supervise weak-grounding
                cov["regions"] += int(per_vol.sum())
                rai = batch.get("region_anatomy_ids")
                if rai is not None:
                    cov["anat"].update(int(a) for a in rai[rv].tolist())

            volume = batch["volume"].to(device, non_blocking=True)
            window_id = batch["window_id"].to(device, non_blocking=True)
            ct_type_id = None if args.no_ct_type_cond else batch["ct_type_id"].to(device, non_blocking=True)
            anatomy_id = batch["anatomy_id"].to(device, non_blocking=True)
            anatomy_mask_s3 = batch["anatomy_mask_s3"].to(device, non_blocking=True) if batch["anatomy_mask_s3"] is not None else None
            anatomy_mask_valid = batch["anatomy_mask_valid"].to(device, non_blocking=True) if batch.get("anatomy_mask_valid") is not None else None
            seg_mask = batch["seg_mask"].to(device, non_blocking=True) if batch["seg_mask"] is not None else None
            token_ids, attn_mask = raw_model.tokenize_texts(batch["texts"], device=device)

            # Cached text embeddings, when present, override the hashing text path.
            text_emb_cached = batch["text_emb_cached"].to(device, non_blocking=True) if batch.get("text_emb_cached") is not None else None
            region_text_embs = batch["region_text_embs"].to(device, non_blocking=True) if batch.get("region_text_embs") is not None else None

            # Intra-volume region-utterance grounding inputs.
            region_masks_s3 = batch["region_masks_s3"].to(device, non_blocking=True) if batch["region_masks_s3"] is not None else None
            region_valid = batch["region_valid"].to(device, non_blocking=True) if batch["region_valid"] is not None else None
            region_anatomy_ids = batch["region_anatomy_ids"].to(device, non_blocking=True) if batch["region_anatomy_ids"] is not None else None
            # Weak-grounding slot set (utterances + anatomy masks) when it is separate from the MUP slots.
            wg_region_masks_s3 = batch["wg_region_masks_s3"].to(device, non_blocking=True) if batch.get("wg_region_masks_s3") is not None else None
            wg_region_valid = batch["wg_region_valid"].to(device, non_blocking=True) if batch.get("wg_region_valid") is not None else None
            wg_region_text_embs = batch["wg_region_text_embs"].to(device, non_blocking=True) if batch.get("wg_region_text_embs") is not None else None
            region_token_ids = region_attn = None
            if region_masks_s3 is not None and region_text_embs is None and batch["region_texts"] is not None:
                R = region_masks_s3.shape[1]
                flat = [t for sub in batch["region_texts"] for t in sub]      # B*R, row-major (b,r)
                rtok, ram = raw_model.tokenize_texts(flat, device=device)
                region_token_ids = rtok.view(volume.shape[0], R, -1)
                region_attn = ram.view(volume.shape[0], R, -1)

            masks = sample_ject_mask_batch(
                batch_size=volume.shape[0], grid_s1=cfg.patch.grid_size_s1,
                grid_s2=cfg.patch.grid_size_s2, grid_s3=cfg.patch.grid_size_s3,
                mask_ratio=cfg.masking.mask_ratio, num_target_blocks=cfg.masking.num_target_blocks,
                min_block_size_s3=cfg.masking.min_block_size_s3,
                max_block_size_s3=cfg.masking.max_block_size_s3,
                buffer_s3=cfg.masking.buffer_s3,
                allow_random_fill=cfg.masking.allow_random_fill,
                anatomy_mask_s3=anatomy_mask_s3,
                anatomy_sampling_prob=args.anatomy_sampling_prob,
                anatomy_mask_frac=cfg.ject_loss.anatomy_mask_frac,
                anatomy_mask_valid=anatomy_mask_valid,   # per sample; text-free rows use random masking
                device=device,
            )
            # Log the actual mask fractions (loss targets vs tokens hidden from the encoder); the two
            # differ by the buffer dilation.
            cov["mask_loss"] += float(masks.loss_mask_s3.float().mean()); cov["mask_enc"] += float(masks.encoder_mask_s3.float().mean()); cov["mask_n"] += 1

            if micro == 0:                                   # start of an accumulation window
                optimizer.zero_grad(set_to_none=True)
                window_ok = True
                win_micro = 0                                # microbatches that contribute grads this window
                # Text-supervision schedule for this optimizer window, keyed on global_step so it is
                # identical across ranks and microbatches. Text losses ramp up from 0 over text_ramp_frac
                # of training. In stochastic mode the window runs the unmasked pass with probability
                # p_clean; otherwise the text heads are skipped and only the image JEPA trains.
                loss_fn.text_scale = min(1.0, global_step / max(1, int(args.text_ramp_frac * total_steps)))
                # Dense-loss warm-up: 0 before dense_warmup_start, linear to full by dense_warmup_end.
                _prog = global_step / max(1, total_steps)
                _d0, _d1 = args.dense_warmup_start, args.dense_warmup_end
                loss_fn.lac_loss.dense_scale = float(min(1.0, max(0.0, (_prog - _d0) / max(1e-6, _d1 - _d0))))
                # MUP centering ramp: scale the SCIR-subtracted mean from 0 to 1 over the first
                # mup_centering_ramp_frac of training.
                if args.mup_centering_ramp_frac > 0:
                    raw_model.mup_centering_scale = float(min(1.0, _prog / args.mup_centering_ramp_frac))
                if cfg.anatomy.text_heads_pass == "stochastic":
                    _g = torch.Generator(); _g.manual_seed(0x5EED + int(global_step))
                    text_clean_step = bool(torch.rand((), generator=_g) < args.p_clean)
                elif cfg.anatomy.text_heads_pass in ("corrupted", "reconstructed") and args.mup_duty < 1.0:
                    # Text heads run on mup_duty of windows, using the same rank-identical
                    # generator as the stochastic schedule.
                    _g = torch.Generator(); _g.manual_seed(0x5EED + int(global_step))
                    text_clean_step = bool(torch.rand((), generator=_g) < args.mup_duty)
                else:
                    text_clean_step = True
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(volume, masks=masks, token_ids=token_ids, attention_mask=attn_mask,
                            anatomy_id=anatomy_id, anatomy_mask_s3=anatomy_mask_s3,
                            anatomy_mask_valid=anatomy_mask_valid,
                            window_id=window_id, ct_type_id=ct_type_id,
                            region_masks_s3=region_masks_s3, region_token_ids=region_token_ids,
                            region_attn=region_attn, region_valid=region_valid,
                            text_emb_cached=text_emb_cached, region_text_embs=region_text_embs,
                            region_anatomy_ids=region_anatomy_ids, text_clean_step=text_clean_step,
                            wg_region_masks_s3=wg_region_masks_s3, wg_region_valid=wg_region_valid,
                            wg_region_text_embs=wg_region_text_embs,
                            spacing_mm=args.spacing_mm)
                losses = loss_fn(out, seg_mask=seg_mask)

            # DDP-safe non-finite guard: a NaN/inf loss on any rank would poison the all-reduced
            # grads, so all ranks agree via all_reduce(MIN).
            finite = torch.isfinite(losses["total"]).to(torch.float32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() < 1.0:
                window_ok = False                            # abandon the whole accumulation window
                nonfinite_skips += 1
                bad = [k for k, v in losses.items()
                       if k != "total" and not torch.isfinite(v.detach()).all()]
                for k in bad:
                    nonfinite_by_loss[k] = nonfinite_by_loss.get(k, 0) + 1
                if rank0 and (nonfinite_skips <= 20 or nonfinite_skips % 200 == 0):
                    vals = {k: float(v.detach()) for k, v in losses.items()}
                    print(f"[warn] non-finite loss at step {global_step}; bad={bad or 'total-only'}; "
                          f"vals={vals}; by_loss={nonfinite_by_loss}; skips={nonfinite_skips}", flush=True)
            else:
                scaler.scale(losses["total"] / accum).backward()   # scale for accumulation
                win_micro += 1                                      # this microbatch's grads are in the window
                if "var" in losses:                                # collapse monitor: relu(1-std) hinge in [0,1]
                    cov["var_sum"] += float(losses["var"].detach()); cov["var_n"] += 1
                if "corr" in losses:                               # scale-invariant correlated-collapse monitor
                    cov["corr_sum"] += float(losses["corr"].detach()); cov["corr_n"] += 1
                if "weakground" in losses and float(losses["weakground"].detach()) > 0:
                    wg_active_steps += 1

            micro += 1
            if micro >= accum:                               # end of window
                micro = 0
                if window_ok:                                # finite window: attempt an optimizer step
                    m = _commit_window(win_micro)
                    if m is not None:                        # real update (not a GradScaler skip)
                        momentum = m
                        global_step += 1
                        if rank0 and global_step % cfg.training.log_interval == 0:
                            log = {k: float(v.detach().cpu()) for k, v in losses.items()}
                            log["ema_m"] = momentum
                            log["lr"] = optimizer.param_groups[0]["lr"]
                            # MUP collapse diagnostics: C_pred near 1 with a flat MUP loss
                            # indicates a constant, image-independent predictor.
                            mm = out.get("mup_metrics")
                            if isinstance(mm, dict):
                                for _mk in (
                                        "mup_C_pred", "mup_C_pred_within_slot",
                                        "mup_C_target_within_slot", "mup_G_prior", "mup_image_gap"):
                                    if _mk in mm:
                                        log[_mk.removeprefix("mup_")] = round(float(mm[_mk]), 3)
                            pbar.set_postfix(log)
                else:                                        # poisoned window: drop grads without
                    optimizer.zero_grad(set_to_none=True)    # advancing scaler/scheduler/EMA/global_step

        # Commit any partial end-of-epoch window; otherwise up to accum-1 batches per epoch (or all of
        # them, when steps_per_epoch < accum) would be discarded.
        if micro > 0:
            if window_ok:
                m = _commit_window(win_micro)
                if m is not None:
                    momentum = m
                    global_step += 1
            else:
                optimizer.zero_grad(set_to_none=True)
            micro = 0

        # --- epoch-end region-text coverage report (is the grounding signal active?) ---
        if rank0 and cov["n"]:
            n = cov["n"]
            top = ", ".join(f"{ANATOMY_ID_TO_NAME.get(a, a)}={c}" for a, c in cov["anat"].most_common(8))
            text_frac = cov["text_ok"] / n
            print(f"[epoch {epoch}] coverage: samples={n} "
                  f"frac>=2_regions={cov['ge2']/n:.3f} avg_regions/vol={cov['regions']/n:.2f} "
                  f"negated={cov['neg']/n:.3f} uncertain={cov['unc']/n:.3f} "
                  f"cached_text_ok={text_frac:.3f} anat_mask_ok={cov['anat_valid']/n:.3f} "
                  f"weakground_active_steps={wg_active_steps} "
                  f"mask_loss_frac={cov['mask_loss']/max(1,cov['mask_n']):.3f} "
                  f"mask_encoder_hidden_frac={cov['mask_enc']/max(1,cov['mask_n']):.3f}\n"
                  f"           region anatomy dist: {top}")
            if args.use_cached_text and text_frac < 0.5:
                print(f"[warn] only {text_frac:.1%} of samples have a nonzero cached-text embedding; "
                      f"most train under the null text condition. Check the utterance sidecars.")

        # Fail fast if weak grounding was supervised on too few samples in epoch 0 (missing or corrupt
        # sidecars). Coverage is the fraction of samples, not batches, with at least one valid region.
        # Counts are all-reduced so every rank makes the same decision.
        hr_global, n_global, wg_steps_global = float(cov["has_region"]), float(cov["n"]), float(wg_active_steps)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([hr_global, n_global, wg_steps_global], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            hr_global, n_global, wg_steps_global = t[0].item(), t[1].item(), t[2].item()
        wg_coverage = hr_global / max(1.0, n_global)
        if epoch == 0 and cfg.anatomy.use_grounding_head and cfg.ject_loss.lambda_weakground > 0:
            if wg_coverage < args.min_weakground_coverage:
                raise RuntimeError(
                    f"use_grounding_head is set (lambda_weakground>0) but only {wg_coverage:.1%} of epoch-0 "
                    f"samples (across all ranks) had a valid region for weak grounding, below "
                    f"--min_weakground_coverage={args.min_weakground_coverage:.0%}. Check --use_cached_text and "
                    f"the per-utterance region sidecars (region_text_embs/region_masks_s3), or lower "
                    f"--min_weakground_coverage.")
            # The text heads must have actually run: in stochastic mode a small p_clean over a short epoch
            # can skip them every step while the coverage check above still passes.
            if wg_steps_global == 0:
                raise RuntimeError(
                    f"weak grounding and MUP did not run in epoch 0 (0 supervised text-head steps across all "
                    f"ranks) although lambda_weakground>0. Check --text_heads_pass and --p_clean.")

        # Cached-text runs need the sidecars: missing ones become null embeddings, so the model would
        # train image-only. Abort unless overridden.
        if epoch == 0 and cfg.text.use_cached_embeddings and not args.allow_missing_cached_text:
            to_g, n_g = float(cov["text_ok"]), float(cov["n"])
            if dist.is_available() and dist.is_initialized():
                t2 = torch.tensor([to_g, n_g], device=device, dtype=torch.float64)
                dist.all_reduce(t2, op=dist.ReduceOp.SUM)
                to_g, n_g = t2[0].item(), t2[1].item()
            text_cov = to_g / max(1.0, n_g)
            if text_cov < args.min_weakground_coverage:
                raise RuntimeError(
                    f"--use_cached_text but only {text_cov:.1%} of epoch-0 samples (across all ranks) have "
                    f"a nonzero cached-text embedding, below the --min_weakground_coverage="
                    f"{args.min_weakground_coverage:.0%} floor; samples without sidecars train with null text. "
                    f"Check the utterance sidecars, lower the floor, or pass --allow_missing_cached_text.")

        # Collapse guard (all ranks agree). `var` is the sum of the per-scale variance hinge
        # relu(target_std - std).mean(), each in [0, 1], over the emitted JEPA scales (s1-s3, plus s4
        # when lambda_s4 > 0); dividing by that count gives a per-scale mean that approaches 1 as
        # per-channel std goes to 0. Variance is legitimately high early in training, so the guard is
        # enforced only after warmup. It aborts before saving a checkpoint.
        _N_JEPA_SCALES = 3 + (1 if getattr(cfg.loss, "lambda_s4", 0.0) > 0 else 0)
        vs_g, vn_g = float(cov["var_sum"]), float(cov["var_n"])
        cs_g, cn_g = float(cov["corr_sum"]), float(cov["corr_n"])
        if dist.is_available() and dist.is_initialized():
            tv = torch.tensor([vs_g, vn_g, cs_g, cn_g], device=device, dtype=torch.float64)
            dist.all_reduce(tv, op=dist.ReduceOp.SUM)
            vs_g, vn_g, cs_g, cn_g = tv[0].item(), tv[1].item(), tv[2].item(), tv[3].item()
        var_mean = (vs_g / max(1.0, vn_g)) / _N_JEPA_SCALES       # per-scale hinge mean, in [0,1]
        corr_mean = cs_g / max(1.0, cn_g)                         # mean |off-diag correlation|, in [0,1]
        # Two collapse modes: dimensional (per-channel std goes to 0, high var_mean) and correlated
        # (channels stay high-variance but redundant, high corr_mean). corr is scale-invariant, unlike
        # raw cov, so it supports an absolute threshold.
        collapsed = (var_mean > args.collapse_var_threshold) or (corr_mean > args.collapse_corr_threshold)
        # ceil, not int: with a fractional --warmup_epochs the guard must not run during warmup.
        if epoch >= math.ceil(args.warmup_epochs) and collapsed and not args.allow_collapse:
            raise RuntimeError(
                f"[collapse] epoch {epoch}: per-scale variance-hinge mean={var_mean:.3f} "
                f"(thr {args.collapse_var_threshold}), mean|correlation|={corr_mean:.3f} "
                f"(thr {args.collapse_corr_threshold}): representation collapse. Stopping before the "
                f"checkpoint is written. Increase --lambda_sigreg or pass --allow_collapse.")

        # Rolling latest checkpoint every epoch, independent of save_interval.
        if rank0:
            save_checkpoint(out_dir / "checkpoint_latest.pt", raw_model, optimizer, epoch + 1, global_step, cfg)
        if rank0 and (epoch + 1) % cfg.training.save_interval == 0:
            save_checkpoint(out_dir / f"checkpoint_epoch_{epoch+1}.pt", raw_model, optimizer, epoch + 1, global_step, cfg)

    if rank0:
        save_checkpoint(out_dir / "checkpoint_final.pt", raw_model, optimizer, cfg.training.total_epochs, global_step, cfg)
        print(f"Saved final checkpoint to {out_dir / 'checkpoint_final.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
