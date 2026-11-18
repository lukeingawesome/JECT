"""LAC-JEPA base configuration.

LAC-JEPA (Latent Anatomical Completion JEPA) is the image-only base that JECT extends.
This module defines the base ``LACJEPAConfig`` dataclass (volume / patch / model /
masking / loss / ema / training / data sub-configs) plus the ``Triple`` alias and
``_triple`` helper used by the JECT config.

The model is a hierarchical Swin-UNETR-style dense encoder/decoder producing latents
at four token grids:

    s1  (finest)   = volume / patch_size
    s2             = s1 / 2         (one patch-merge)
    s3             = s1 / 4         (two patch-merges)
    s4  (coarsest) = ceil(s3 / 2)   (global scale)

Dense JEPA predicts the masked-region latents of an EMA teacher at every scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

Triple = Tuple[int, int, int]


def _triple(x) -> Triple:
    """Normalize an int or 1/3-length iterable into a 3-tuple of ints."""
    if isinstance(x, int):
        return (x, x, x)
    vals = tuple(int(v) for v in x)
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    if len(vals) != 3:
        raise ValueError(f"Expected an int or a 1/3-length iterable, got {x!r}")
    return vals


# Per-variant, per-scale (s1, s2, s3, s4) feature dims, attention heads, stage depths and prototype
# counts. s4 is the coarsest, global scale at twice the s3 width. Head dims are kept even (required
# by the axial 3D RoPE predictor).
VARIANTS = {
    "tiny": dict(dims=(32, 64, 128, 256), heads=(2, 4, 8, 8), depths=(2, 2, 2, 2),
                 num_prototypes=(256, 256, 256, 256)),
    "small": dict(dims=(48, 96, 192, 384), heads=(3, 6, 12, 12), depths=(2, 2, 4, 2),
                  num_prototypes=(512, 512, 512, 512)),
    "base": dict(dims=(64, 128, 256, 512), heads=(4, 8, 16, 16), depths=(2, 2, 4, 2),
                 num_prototypes=(1024, 1024, 1024, 1024)),
    # Larger variants scale width and stage depth, not the patch size.
    "large": dict(dims=(96, 192, 384, 768), heads=(6, 12, 24, 24), depths=(2, 2, 8, 4),
                  num_prototypes=(2048, 2048, 2048, 2048)),
    "huge": dict(dims=(128, 256, 512, 1024), heads=(8, 16, 32, 32), depths=(2, 4, 9, 9),
                 num_prototypes=(4096, 4096, 4096, 4096)),
}


@dataclass
class VolumeConfig:
    depth: int = 224
    height: int = 224
    width: int = 224
    channels: int = 1

    @property
    def spatial_size(self) -> Triple:
        return (self.depth, self.height, self.width)


@dataclass
class PatchConfig:
    patch_size: Triple = (4, 4, 4)   # Swin-UNETR operating point
    # Filled in by LACJEPAConfig.__post_init__ from volume + patch_size.
    grid_size_s1: Triple = (0, 0, 0)
    grid_size_s2: Triple = (0, 0, 0)
    grid_size_s3: Triple = (0, 0, 0)
    grid_size_s4: Triple = (0, 0, 0)   # coarsest global scale (s3 // 2)
    num_tokens_s1: int = 0
    num_tokens_s2: int = 0
    num_tokens_s3: int = 0
    num_tokens_s4: int = 0


@dataclass
class ModelConfig:
    variant: str = "tiny"
    local_block_type: str = "swin"      # swin | transformer | convnext
    stage3_block_type: str = "global"   # global | swin | transformer
    # Cumulative bottom-up skips into the s2, s3 and s4 stage inputs: max+mean pools of every finer
    # representation (patch embeddings, e1, e2, e3) over each cell, then LayerNorm and a zero-init Linear.
    skip_fine: bool = False
    # Skip pooling at the s3 entry: "maxmean" or "attention" (local attention pooler).
    skip_pool_s3: str = "maxmean"
    # Skip pooling at the s2 and s4 entries: "maxmean" or "attention" (requires skip_fine; intended
    # together with skip_pool_s3="attention").
    skip_pool_s2s4: str = "maxmean"
    # Source of the s1-s3 JEPA targets on the EMA side: "decoder" (dense decoder outputs) or "encoder"
    # (EMA encoder stage outputs e1-e3, as s4 uses e4). Only the targets change; predictor inputs,
    # grounding, MSP, dense-prediction inputs and readouts use the decoder either way.
    jepa_target_source: str = "decoder"
    # Build the skip modules outside torch.random.fork_rng, so their initialization consumes the
    # global RNG stream.
    skip_legacy_rng: bool = False
    swin_window_size: Triple = (7, 7, 7)
    mlp_ratio: float = 4.0
    drop: float = 0.0
    num_ct_types: int = 3            # non-contrast / contrast / ldct (additive embedding)
    # How absolute position enters the encoder. "learned": per-token tables tied to the grid
    # (pos_embed at the s3 entry and pos_embed_s4). "mm_rope": physical-millimeter coordinates, a
    # continuous CoordEmbed3D at the s3 entry plus RoPE in stage-4 attention, so one set of weights
    # serves any grid size and voxel spacing. "mm_rope" requires spacing_mm at forward time.
    pos_mode: str = "learned"
    use_prototypes: bool = True
    prototype_temperature: float = 0.1
    # Filled in from the variant by __post_init__.
    dim_s1: int = 0
    dim_s2: int = 0
    dim_s3: int = 0
    dim_s4: int = 0
    heads: Tuple[int, ...] = (0, 0, 0, 0)      # length-4 (s1..s4)
    depths: Tuple[int, ...] = (0, 0, 0, 0)     # length-4 (s1..s4)
    stage4_block_type: str = "global"          # s4 is the global semantic scale
    num_prototypes_s1: int = 0
    num_prototypes_s2: int = 0
    num_prototypes_s3: int = 0
    num_prototypes_s4: int = 0


@dataclass
class MaskingConfig:
    mask_ratio: float = 0.50   # masked fraction of the whole volume (anatomy + random fill)
    num_target_blocks: int = 4
    min_block_size_s3: Triple = (1, 1, 1)
    max_block_size_s3: Triple = (2, 3, 3)
    buffer_s3: int = 1
    allow_random_fill: bool = True


@dataclass
class LossConfig:
    lambda_s1: float = 0.25
    lambda_s2: float = 0.35
    lambda_s3: float = 0.40
    lambda_s4: float = 0.30   # global semantic scale JEPA (on encoder e4)
    lambda_proto: float = 0.10
    # Dense predictive loss (V-JEPA 2.1 style): LayerNorm'd L2 on the visible tokens at the grounding
    # scale, weighted by proximity to the mask boundary. Preserves the per-location detail that the
    # grounding head attends to.
    use_dense_pred: bool = False
    lambda_dense: float = 0.0
    dense_pred_scale: str = "s2"        # single scale, used when dense_pred_scales is empty
    dense_pred_scales: Tuple[str, ...] = ()   # dense-loss scales (s4 not supported)
    dense_pred_weights: Tuple[float, ...] = ()  # per-scale weight, relative to that scale's masked-JEPA weight
    dense_warmup_start: float = 0.10    # dense loss is 0 before this fraction of training
    dense_warmup_end: float = 0.40      # then ramps linearly to full weight by this fraction
    dense_pred_dilate: int = 3          # proximity ring width (tokens from the mask boundary)
    # VICReg anti-collapse terms: variance hinge and covariance (decorrelation). Variance alone can
    # give way to a correlated collapse, so the covariance term is used with it.
    lambda_var: float = 1.0
    var_target: float = 1.0
    var_eps: float = 1e-4
    lambda_cov: float = 0.04
    cov_max_tokens: int = 4096   # subsample tokens for the covariance estimate (keeps it cheap)
    # Anti-collapse regularizer: "sigreg" (LeJEPA isotropic Gaussian), "vicreg" (variance +
    # covariance) or "both". SIGReg matches the full 1-D distribution, not only the first two moments.
    anticollapse: str = "sigreg"
    # SIGReg weight on the LeJEPA scale: the Epps-Pulley statistic keeps its `* N` factor and the
    # token sample is capped at sigreg_max_tokens = 512 (LeJEPA's N), so values match the paper's
    # bstat_lambda (0.01-0.1).
    lambda_sigreg: float = 0.05
    # SIGReg: LeJEPA Epps-Pulley test on random 1-D projections (see sigreg_loss). n_proj is the
    # number of projection directions; n_freq is the number of fixed characteristic-function knots
    # on [0, t_max] (odd, for trapezoid symmetry).
    sigreg_n_proj: int = 512     # num_slices (sketch directions)
    sigreg_n_freq: int = 17      # n_points (CF integration knots, odd)
    sigreg_t_max: float = 3.0    # upper integration frequency
    sigreg_max_tokens: int = 512  # N for the `* N` statistic scaling (matches LeJEPA's batch)
    # SIGReg is applied to a separate projector of the latents, as in LeJEPA, rather than to the raw
    # multi-scale latents: forcing an isotropic Gaussian directly on the features used by the dense
    # and grounding heads erodes their local structure. 0 disables the projector (SIGReg on raw
    # latents); the JECT trainer sets it via --sigreg_proj_dim.
    sigreg_proj_dim: int = 0


@dataclass
class EMAConfig:
    momentum_start: float = 0.996
    momentum_end: float = 0.9999
    total_steps: int = 100000
    momentum_schedule: str = "cosine"   # cosine | linear


@dataclass
class TrainingConfig:
    batch_size: int = 1
    num_workers: int = 4
    total_epochs: int = 20
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    log_interval: int = 20
    save_interval: int = 1
    prefetch_factor: int = 2


@dataclass
class DataConfig:
    csv_path: str = ""
    data_root: str = ""
    img_col: str = "object_id"
    split_col: str = "split"
    split: str = "train"
    ct_type_col: str = "ct_type"    # non-contrast / contrast / ldct map to ids 0 / 1 / 2
    # Fixed general window by default: random windows can hide findings outside the chosen window
    # (e.g. a lung nodule in a mediastinal window), corrupting the text-grounding signal.
    # vary_window=True samples windows randomly.
    vary_window: bool = False
    # Mild intensity augmentation (CT only, not masks): HU shift/scale before windowing and small
    # Gaussian noise after.
    aug_intensity: bool = True
    aug_hu_shift: float = 20.0       # +/- HU shift  (~1% of the general window)
    aug_hu_scale: float = 0.03       # x [1-0.03, 1+0.03] HU scale
    aug_noise_std: float = 0.01      # Gaussian std in [0,1] space (~1% of range)
    # Downsampled TotalSegmentator cache, used when the `{id}_ts64.npy` sidecar exists; anatomy
    # masks are cut at 64³ instead of 256³.
    use_ts_cache: bool = True
    ts_cache_suffix: str = "_ts64.npy"


@dataclass
class LACJEPAConfig:
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        self._apply_variant()
        self._compute_grids()
        self._validate()

    # ------------------------------------------------------------------
    def _apply_variant(self):
        mc = self.model
        if mc.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {mc.variant!r}; expected one of {sorted(VARIANTS)}.")
        spec = VARIANTS[mc.variant]
        mc.dim_s1, mc.dim_s2, mc.dim_s3, mc.dim_s4 = spec["dims"]
        mc.heads = tuple(spec["heads"])
        mc.depths = tuple(spec["depths"])
        (mc.num_prototypes_s1, mc.num_prototypes_s2,
         mc.num_prototypes_s3, mc.num_prototypes_s4) = spec["num_prototypes"]

    def _compute_grids(self):
        pc = self.patch
        pc.patch_size = _triple(pc.patch_size)
        D, H, W = self.volume.spatial_size
        pd, ph, pw = pc.patch_size
        for name, vol, p in (("depth", D, pd), ("height", H, ph), ("width", W, pw)):
            if vol % p != 0:
                raise ValueError(f"volume {name}={vol} not divisible by patch_size {p}.")
        s1 = (D // pd, H // ph, W // pw)
        s2 = tuple(g // 2 for g in s1)
        s3 = tuple(g // 4 for g in s1)
        # PatchMerge3D pads odd grids before halving, so s4 = ceil(s3 / 2) (e.g. s3=21 gives s4=11).
        # Using floor here would mismatch the runtime token count and the pos_embed_s4 size.
        s4 = tuple((g + 1) // 2 for g in s3)
        pc.grid_size_s1, pc.grid_size_s2, pc.grid_size_s3, pc.grid_size_s4 = s1, s2, s3, s4
        pc.num_tokens_s1 = s1[0] * s1[1] * s1[2]
        pc.num_tokens_s2 = s2[0] * s2[1] * s2[2]
        pc.num_tokens_s3 = s3[0] * s3[1] * s3[2]
        pc.num_tokens_s4 = s4[0] * s4[1] * s4[2]

    def _validate(self):
        pc = self.patch
        mc = self.model
        # Two patch merges require the S1 grid divisible by 4 on every axis.
        for axis, g in zip(("depth", "height", "width"), pc.grid_size_s1):
            if g < 4 or g % 4 != 0:
                raise ValueError(
                    f"S1 grid must be divisible by 4 (got {axis}={g}). "
                    f"Change volume size or patch size (e.g. 224/patch8 -> S1=28, valid)."
                )
        if getattr(mc, "pos_mode", "learned") not in ("learned", "mm_rope"):
            raise ValueError(f"pos_mode must be 'learned' or 'mm_rope', got {mc.pos_mode!r}.")
        mc.swin_window_size = _triple(mc.swin_window_size)
        # No window-divisibility constraint: SwinBlock3D pads each grid up to a window
        # multiple and crops back, so any window size is valid.
        for scale, dim, heads in (("s1", mc.dim_s1, mc.heads[0]),
                                  ("s2", mc.dim_s2, mc.heads[1]),
                                  ("s3", mc.dim_s3, mc.heads[2])):
            if dim % heads != 0:
                raise ValueError(f"{scale} dim={dim} not divisible by heads={heads}.")
