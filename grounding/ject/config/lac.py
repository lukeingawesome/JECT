"""LAC-JEPA base configuration.

LAC-JEPA = Latent Anatomical Completion JEPA — the image-only pilot that ART-JEPA
extends. This module defines the base ``LACJEPAConfig`` dataclass (volume / patch /
model / masking / loss / ema / training / data sub-configs) plus the ``Triple`` alias
and ``_triple`` helper that the ART layer imports.

The model is a small hierarchical Swin-UNETR-style dense encoder/decoder producing
latents at three token grids:

    s1  (finest)   = volume / patch_size
    s2             = s1 / 2   (one patch-merge)
    s3  (coarsest) = s1 / 4   (two patch-merges)

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


# variant -> per-scale (s1, s2, s3) feature dims, attention heads, and stage depths.
# Head dims are kept even (required by the axial 3D RoPE predictor).
# Per-scale (s1, s2, s3, s4) feature dims / heads / stage depths. s4 is the coarsest, GLOBAL semantic
# scale (7³ = s3//2), 2× the s3 width. `huge` splits the old s3 depth-18 into s3=9 (swin) + s4=9 (global).
VARIANTS = {
    "tiny": dict(dims=(32, 64, 128, 256), heads=(2, 4, 8, 8), depths=(2, 2, 2, 2),
                 num_prototypes=(256, 256, 256, 256)),
    "small": dict(dims=(48, 96, 192, 384), heads=(3, 6, 12, 12), depths=(2, 2, 4, 2),
                  num_prototypes=(512, 512, 512, 512)),
    "base": dict(dims=(64, 128, 256, 512), heads=(4, 8, 16, 16), depths=(2, 2, 4, 2),
                 num_prototypes=(1024, 1024, 1024, 1024)),
    # Foundation-scale (keep patch-4; scale width + stage depth, not the patch).
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
    patch_size: Triple = (4, 4, 4)   # true Swin-UNETR operating point (finer than patch-8)
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
    swin_window_size: Triple = (7, 7, 7)
    mlp_ratio: float = 4.0
    drop: float = 0.0
    num_ct_types: int = 3            # non-contrast / contrast / ldct (AdaLN-free additive embed)
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
    mask_ratio: float = 0.50   # fraction of the WHOLE volume (anatomy + random fill) for all samples
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
    # Dense predictive loss (V-JEPA-2.1 / CEPA): LayerNorm'd L2 on the VISIBLE tokens at the grounding
    # scale, proximity-weighted to the mask boundary — preserves per-location spatial fidelity that the
    # grounding head cross-attends. OFF by default; enable to A/B its effect on grounding sharpness.
    use_dense_pred: bool = False
    lambda_dense: float = 0.0
    dense_pred_scale: str = "s2"        # legacy single-scale (fallback when dense_pred_scales empty)
    dense_pred_scales: Tuple[str, ...] = ()   # dense scales (recommend s2 s3; s1 only in hi-res cooldown; s4 n/a)
    dense_pred_weights: Tuple[float, ...] = ()  # per-scale λ_ctx (relative to masked α_s); effective = α_s·λ_ctx,s
    dense_warmup_start: float = 0.10    # dense loss = 0 before this fraction of training …
    dense_warmup_end: float = 0.40      # … linear ramp to full by here (V-JEPA-2.1: stabilizes global semantics)
    dense_pred_dilate: int = 3          # proximity ring width (tokens from the mask boundary)
    # Anti-collapse (VICReg). lambda_var=0.01 was ~100x too weak: at collapse the var hinge saturates
    # near 3.0 but contributed only 0.03 while collapsing the JEPA terms saved ~0.6, so the model
    # rationally collapsed (~ep12 of run #5). Variance alone can shift to a correlated collapse, so
    # the covariance (decorrelation) term is now on too.
    lambda_var: float = 1.0
    var_target: float = 1.0
    var_eps: float = 1e-4
    lambda_cov: float = 0.04
    cov_max_tokens: int = 4096   # subsample tokens for the covariance estimate (keeps it cheap)
    # anti-collapse regularizer: "sigreg" (LeJEPA isotropic-Gaussian, DEFAULT — used alone at ~O(1)
    # weight per the paper), "vicreg" (var+cov, moment-only), or "both" (ablation). SIGReg matches the
    # full 1-D distribution, not just the 1st/2nd moments, so VICReg is redundant alongside it.
    anticollapse: str = "sigreg"
    # PAPER-FAITHFUL LeJEPA weight: the Epps-Pulley statistic keeps its `* N` factor and the SIGReg
    # token sample is capped to 512 (= LeJEPA's N), so this 0.05 means what bstat_lambda=0.05 means
    # in the paper (they sweep 0.01-0.1). Watch `cov` at ep4-8; raise toward 0.1 if it climbs.
    lambda_sigreg: float = 0.05
    # SIGReg = LeJEPA Epps-Pulley on random 1-D slices (see sigreg_loss). num_slices = sketch
    # directions (LeJEPA uses 512-1024; sigreg_loss chunks them so this is cheap). n_points = FIXED
    # characteristic-function knots on [0, t_max] (odd — trapezoid symmetry; paper default 17).
    # sigreg_max_tokens = N per Epps-Pulley `* N`; 512 matches LeJEPA so lambda transfers.
    sigreg_n_proj: int = 512     # num_slices (sketch directions)
    sigreg_n_freq: int = 17      # n_points (CF integration knots, odd)
    sigreg_t_max: float = 3.0    # upper integration frequency
    sigreg_max_tokens: int = 512  # N for the `* N` statistic scaling (matches LeJEPA's batch)
    # SIGReg is applied to a THROWAWAY PROJECTOR of the latents, not the raw multi-scale token latents:
    # LeJEPA regularizes a projector, and forcing full isotropic-Gaussian directly on the features a
    # dense/grounding head consumes erodes the local structure it needs (VICReg→VICRegL). proj_dim=0
    # disables the projector (SIGReg on raw latents). The BASE default is 0 so image-only LAC callers of
    # LACJEPALoss keep working without emitting projector tensors; the ART trainer sets 256 via
    # --sigreg_proj_dim (ARTJEPAModel emits the projector outputs).
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
    ct_type_col: str = "ct_type"    # maps to non-contrast/contrast/ldct -> 0/1/2
    # Fixed general window by default: random windows would hide findings outside the assigned
    # window (e.g. a lung nodule in the mediastinal window), corrupting the text-grounding signal,
    # and general matches the seg/probe eval window. vary_window=True restores random sampling.
    vary_window: bool = False
    # Slight intensity augmentation (CT only, not masks): HU shift/scale before windowing +
    # small Gaussian noise after. Deliberately mild — do not add severe noise.
    aug_intensity: bool = True
    aug_hu_shift: float = 20.0       # +/- HU shift  (~1% of the general window)
    aug_hu_scale: float = 0.03       # x [1-0.03, 1+0.03] HU scale
    aug_noise_std: float = 0.01      # Gaussian std in [0,1] space (~1% of range)
    # Downsampled-totalseg cache (precompute_art_cache.py) — used automatically when the
    # `{id}_ts64.npy` sidecar exists; cuts anatomy masks over 64³ instead of 256³.
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
        # s4 = merge3(s3): PatchMerge3D PADS odd grids then halves, i.e. ceil(g/2). Compute grid_s4 with
        # the SAME ceil so config (pos_embed_s4 size) matches the runtime token count for odd s3 grids
        # (e.g. 336³: s3=21 -> s4=11, not floor 10) — a plain s1//8 would mismatch and crash the encoder.
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
        mc.swin_window_size = _triple(mc.swin_window_size)
        # NOTE: no window-divisibility constraint — SwinBlock3D pads each grid up to a
        # window multiple internally and crops back, so any window size is valid.
        for scale, dim, heads in (("s1", mc.dim_s1, mc.heads[0]),
                                  ("s2", mc.dim_s2, mc.heads[1]),
                                  ("s3", mc.dim_s3, mc.heads[2])):
            if dim % heads != 0:
                raise ValueError(f"{scale} dim={dim} not divisible by heads={heads}.")
