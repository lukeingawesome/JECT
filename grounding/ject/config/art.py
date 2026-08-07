"""ART/AR-LAC-JEPA configuration.

ART-JEPA = Anatomy-Report-Text guided Latent Anatomical Completion JEPA.
This extends the image-only LAC-JEPA pilot with:
  * structured report utterance conditioning
  * utterance-to-anatomy IDs
  * anatomy-mask-guided target sampling
  * optional 3D RoPE global latent predictor
  * optional auxiliary segmentation loss
  * high-resolution cooldown support via flexible checkpoint loading
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

try:
    from ject.config.lac import LACJEPAConfig, Triple, _triple
except ImportError:
    from ject.config.lac import LACJEPAConfig, Triple, _triple


@dataclass
class ARTTextConfig:
    enabled: bool = True
    vocab_size: int = 32768
    max_tokens: int = 96
    text_dim: int = 256
    text_hidden_dim: int = 512
    dropout: float = 0.05

    # Cached Chest2Vec per-utterance embeddings (recommended over the hashing encoder).
    # Loads `{id}{utterances_sidecar_suffix}` (EOS-pooled 1024-d, index-aligned with the
    # utterances_json). Projected 1024 -> text_dim by `text_in_proj`. Hashing-vs-cached is a
    # run-level choice; within a cached run a MISSING sidecar yields a neutral zero embedding
    # + no valid regions for that sample (it does not flip the batch back to hashing).
    use_cached_embeddings: bool = False
    cached_dim: int = 1024
    utterances_sidecar_suffix: str = "_utterances.pt"
    # When True, a sidecar WITHOUT a stored `texts` list is rejected (treated as no cached text) — set this
    # on rebuilt/positive-only manifests so only sidecars whose index alignment is provable are trusted.
    require_cached_text_alignment: bool = False

    # Predictor conditioning mechanism: "film" (spatially-uniform FiLM, cheap, cannot localize) or
    # "xattn" (gated image→text cross-attention over the per-utterance set — spatial, and its attention
    # doubles as a grounding readout; ~+2-7% step time). xattn applies to predictor_xattn_scales; other
    # scales (and s3's RoPE predictor) keep FiLM. xattn requires cached per-utterance embeddings.
    predictor_cond: str = "film"
    predictor_xattn_scales: Tuple[str, ...] = ("s1", "s2")
    predictor_num_heads: int = 8

    # Classifier-free-guidance text dropout: with this probability (training only) the predictor's
    # text CONDITIONING is dropped to the null (FiLM → neutral vector; xattn → zeroed) — this trains an
    # unconditioned mode so the grounding CFG residual (map|text − map|null) is meaningful, and curbs
    # text over-reliance. Applies to conditioning ONLY, never the grounding supervision targets.
    text_drop_prob: float = 0.2

    # Flexible CSV parsing. Preferred format is a JSON list of utterance dicts.
    # Example: [{"text":"7 mm RUL nodule", "anatomy_id":3, "label":"nodule", "negated":false}]
    utterances_json_col: str = "utterances_json"
    findings_col: str = "findings"
    impression_col: str = "impression"

    # Fallback: section columns where each column is an anatomy bucket.
    use_section_columns: bool = True
    section_columns: Tuple[str, ...] = (
        "Lungs", "Airways & Trachea", "Pleura", "Mediastinum & Hila",
        "Cardiovascular", "Chest Wall", "Bones / Spine", "Upper abdomen",
        "Lower neck", "Breast & Axilla", "Others",
    )


@dataclass
class ARTAnatomyConfig:
    enabled: bool = True
    num_anatomies: int = 32
    background_id: int = 0
    unknown_anatomy_id: int = 0

    # Preferred NPZ keys. anat_masks should be [K,D,H,W] or [1,K,D,H,W].
    anatomy_mask_key: str = "anat_masks"
    alternative_anatomy_mask_keys: Tuple[str, ...] = ("anatomy_masks", "region_masks", "total_masks")
    totalseg_key: str = "totalseg"

    # Optional supervised segmentation key for auxiliary loss.
    seg_mask_key: str = "seg_mask"
    alternative_seg_mask_keys: Tuple[str, ...] = ("mask", "label", "seg", "totalseg")
    num_seg_classes: int = 2
    use_aux_segmentation: bool = False

    # Intra-volume region↔utterance grounding: pool up to `max_regions` anatomy regions
    # per volume and match each to its utterance (batch-1-safe). 0 disables it.
    max_regions: int = 6

    # Stage-0 grounding head: image→text cross-attention producing a per-token localization map,
    # weakly-supervised against the referenced-anatomy region mask (anatomy_mask_s3). See
    # GROUNDING_FOUNDATION_DESIGN.md. Off by default (keeps the base JEPA model unchanged).
    use_grounding_head: bool = False
    grounding_num_heads: int = 8
    # Feature scale the grounding + MUP heads operate on: "s3" (14³/24mm — too coarse for small
    # findings), "s2" (28³/12mm, default) or "s1" (56³/6mm). The s3 region-mask targets are upsampled
    # to this scale; finer features localize better even against an upsampled-coarse weak target.
    grounding_scale: str = "s2"

    # Masked-utterance prediction (CEPA-style "mask + predict text embedding", routed through the
    # grounder's cross-attention so it is spatial). Randomly masks mup_mask_prob of the valid
    # utterances and predicts their embeddings from the image. Needs cached per-utterance embeddings.
    use_masked_utterance_pred: bool = False
    mup_mask_prob: float = 0.5
    # Multi-scale text (utterance) prediction: predict each masked utterance from image features at
    # every scale in `mup_scales`, each head at the matching depth in `mup_depths` (one cross-attn+MLP
    # block per unit). A depth-1 head cannot fit the image→utterance map and saturates at the
    # predict-mean floor (the measured MUP failure); JEPA predictors are depth 6–12 (I-JEPA 12,
    # diffjepa 12/6, CEPA 6). Text prediction lives only where the text carries meaning — s2 (finest
    # meaningful, anatomy granularity) + s3 (coarse semantic); s1 is left image-driven (finding EXTENT
    # is an image property, not a text one). Because the queries are few (~max_regions), cross-attn is
    # cheap even at fine scales. Defaults keep the legacy single-scale s2/depth-1 head.
    mup_scales: Tuple[str, ...] = ("s2",)
    mup_depths: Tuple[int, ...] = (1,)
    # Target-conditioning: add the masked utterance's REGION-CENTROID location to its query so the task
    # is IDENTIFIABLE (else the exchangeable-query loss floors at the per-volume centroid, not phrase-
    # specific — see MaskedUtterancePredictor). Uses region_masks_s3 (already present) as the location.
    # LEGACY (maps to mup_spatial="additive"); prefer mup_spatial below.
    mup_cond_pos: bool = False
    # Anatomy pathway for the MUP cross-attention:
    #   "relbias"  (recommended) — anatomy (region centroid + per-axis extent + downsampled occupancy)
    #              enters ONLY the attention logits as a learned per-head bias; the predictor state is
    #              initialized from the attended IMAGE value, so no centroid/mask residual reaches the
    #              output. Forces region→where, image→what, no p(text|region) shortcut.
    #   "rope"     — factorized 3D RoPE (query rotated by centroid, keys by grid): clean position-match
    #              ablation, same image-initialized state.
    #   "additive" — legacy q ← q + MLP(centroid): identifiable but leaks the centroid to the output.
    #   "none"     — no spatial mechanism (non-identifiable; diagnostic).
    mup_spatial: str = "relbias"
    # Feature source for the text-supervision heads (weak-grounding, MUP):
    #   "stochastic" (recommended, gpt-pro) — JEPA every step; the clean forward + text heads run on a
    #             fraction `mup_p_clean` of steps (Bernoulli), shaping the encoder at ~1.25x avg compute.
    #   "clean" — a 2nd, UNMASKED student forward EVERY step: shapes the encoder, ~2x encoder compute.
    #   "probe" — read the detached EMA target (unmasked): the heads are READOUT PROBES that measure
    #             grounding-readiness without shaping the encoder (the text-conditioned JEPA predictor
    #             does that; the B ablation/control). ~2x cheaper. Never reads the masked student stream.
    text_heads_pass: str = "clean"
    mup_p_clean: float = 0.25          # stochastic mode: P(a step runs the clean encoder-shaping pass)
    text_ramp_frac: float = 0.05       # ramp text losses 0→full over this fraction of total steps
    # Per-scale MUP gradient attenuation (None → default {s1:0, s2:0.25, s3:0.5, s4:1.0}); protects fine
    # scales from the low-rank text target while concentrating report semantics at s3/s4 (gpt-pro).
    mup_stage_gamma: dict | None = None


@dataclass
class ARTRoPEConfig:
    # Use RoPE only in the global latent predictor by default. This avoids destabilizing
    # Swin's native relative-position bias in window attention.
    use_rope_predictor: bool = True
    rope_scales: Tuple[str, ...] = ("s3",)  # allowed: s1, s2, s3
    predictor_depth: int = 1
    predictor_mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class ARTLossConfig:
    # Added on top of the LAC-JEPA image loss.
    lambda_anatomy_region_text: float = 0.05
    lambda_report_labels: float = 0.05
    lambda_seg_aux: float = 0.10
    contrastive_temperature: float = 0.07
    # Weak-grounding loss: the grounding-head localization map vs the referenced-anatomy region.
    # The grounding-prior driver for the predictor-as-grounder design (GROUNDING_FOUNDATION_DESIGN.md).
    lambda_weakground: float = 0.5
    # Masked-utterance prediction loss (CEPA "mask + predict text", made spatial). Complements
    # weak-grounding: self-supervised, scales to all text; bootstrapped by weak-grounding's "where".
    lambda_mup: float = 0.5

    # Anatomy-pooled JEPA residual: predict the pooled representation of the referenced
    # anatomy (text-conditioned) vs the EMA teacher's pooled representation of the same
    # region. Complements the per-patch JEPA — a region-level target that is not diluted
    # by pooling one finding over hundreds of per-token losses. Mean-pool (param-free) so
    # the target stays stop-grad. anatomy_mask_frac = fraction of the anatomy actually masked.
    lambda_region_jepa: float = 0.5
    region_jepa_loss: str = "cosine"     # cosine | l1 | l2
    anatomy_mask_frac: float = 0.5       # partial-mask this fraction of the anatomy

    # During high-resolution cooldown, you usually want more fine-scale dense loss
    # and less global text loss. The train script can override these from CLI.
    cooldown_lambda_s1: float = 0.35
    cooldown_lambda_s2: float = 0.30
    cooldown_lambda_s3: float = 0.20
    cooldown_lambda_text: float = 0.03
    cooldown_lambda_seg_aux: float = 0.15


@dataclass
class ARTCooldownConfig:
    enabled: bool = False
    init_checkpoint: str | None = None
    flexible_load: bool = True
    skip_mismatched_rel_pos_bias: bool = True


@dataclass
class ARTJEPAConfig(LACJEPAConfig):
    text: ARTTextConfig = field(default_factory=ARTTextConfig)
    anatomy: ARTAnatomyConfig = field(default_factory=ARTAnatomyConfig)
    rope: ARTRoPEConfig = field(default_factory=ARTRoPEConfig)
    art_loss: ARTLossConfig = field(default_factory=ARTLossConfig)
    cooldown: ARTCooldownConfig = field(default_factory=ARTCooldownConfig)

    def __post_init__(self):
        super().__post_init__()
        self.rope.rope_scales = tuple(str(s).lower() for s in self.rope.rope_scales)
        for s in self.rope.rope_scales:
            if s not in {"s1", "s2", "s3", "s4"}:
                raise ValueError(f"Invalid rope scale {s!r}; expected s1/s2/s3/s4.")
        self.patch.patch_size = _triple(self.patch.patch_size)
