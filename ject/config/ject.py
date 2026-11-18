"""JECT configuration.

JECT (Joint Embedding Computed Tomography) extends the image-only LAC-JEPA base with:
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

from ject.config.lac import LACJEPAConfig, Triple, _triple


@dataclass
class JECTTextConfig:
    enabled: bool = True
    vocab_size: int = 32768
    max_tokens: int = 96
    text_dim: int = 256
    text_hidden_dim: int = 512
    dropout: float = 0.05

    # Cached per-utterance text embeddings (Chest2Vec, EOS-pooled, 1024-d) loaded from
    # `{id}{utterances_sidecar_suffix}`, index-aligned with utterances_json and projected to
    # text_dim by `text_in_proj`. A missing sidecar yields a zero embedding and no valid
    # regions for that sample.
    use_cached_embeddings: bool = False
    cached_dim: int = 1024
    utterances_sidecar_suffix: str = "_utterances.pt"
    # MUP target granularity:
    #   "utterance":      one slot per positive utterance.
    #   "section":        one slot per canonical report section, loaded from
    #                     `{sr_sidecar_root}/{id}{sr_sidecar_suffix}`. Slot k is the same section in
    #                     every study, so the masked query itself names what to predict.
    #   "report":         one whole-report target (`report_token` in the same sidecar); uses global
    #                     CIR because the query carries no varying text identity.
    #   "section_report": the section slots plus one whole-report slot.
    mup_target_level: str = "utterance"
    sr_sidecar_root: str = ""
    sr_sidecar_suffix: str = "_sr_targets4b.pt"
    # Reject sidecars without a stored `texts` list (treated as no cached text), so only sidecars
    # with verifiable index alignment are used.
    require_cached_text_alignment: bool = False

    # Predictor text conditioning: "film" (spatially uniform FiLM) or "xattn" (gated image-to-text
    # cross-attention over the per-utterance set, spatial). xattn applies to predictor_xattn_scales;
    # other scales and the s3 RoPE predictor keep FiLM. xattn requires cached per-utterance embeddings.
    predictor_cond: str = "film"
    predictor_xattn_scales: Tuple[str, ...] = ("s1", "s2")
    predictor_num_heads: int = 8

    # Classifier-free-guidance text dropout: probability (training only) of replacing the predictor's
    # text conditioning with the null input (neutral FiLM vector, zeroed xattn context). Trains an
    # unconditioned mode for the grounding CFG residual. Does not affect supervision targets.
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
class JECTAnatomyConfig:
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

    # Intra-volume region-utterance grounding: pool up to `max_regions` anatomy regions
    # per volume and match each to its utterance (works at batch size 1). 0 disables it.
    max_regions: int = 6

    # Grounding head: image-to-text cross-attention producing a per-token localization map,
    # weakly supervised by the referenced-anatomy region mask (anatomy_mask_s3).
    use_grounding_head: bool = False
    grounding_num_heads: int = 8
    # Feature scale used by the grounding and MUP heads: "s1" (finest), "s2" or "s3" (coarsest).
    # The s3 region-mask targets are upsampled to this scale.
    grounding_scale: str = "s2"
    # Separate weak-grounding slot set: up to `wg_max_regions` positive utterances with their anatomy
    # masks. Needed with section/report MUP targets, whose slots carry full-volume placeholder masks.
    # 0 = the grounding head reads the MUP region slots.
    wg_max_regions: int = 0
    # Weak-grounding query content: "sentence" uses the patient-specific utterance embedding;
    # "anatomy_phrase" uses a fixed phrase embedding per anatomy id from `wg_anatomy_emb` (same frozen
    # text encoder). Only the valid query vectors are replaced; slots, masks, MUP targets and
    # conditioning are not affected.
    wg_query: str = "sentence"
    wg_anatomy_emb: str = ""

    # Masked-utterance prediction (MUP): mask mup_mask_prob of the valid utterances and predict their
    # embeddings from image features through the grounder's cross-attention. Needs cached embeddings.
    use_masked_utterance_pred: bool = False
    mup_mask_prob: float = 0.5
    # Mask exactly one valid utterance per sample (uniform) instead of per-slot
    # Bernoulli(mup_mask_prob). With mup_spatial="none" the query carries no selector, so several
    # index-bound masked slots are underdetermined and the optimum is a mean over slot candidates.
    mup_single_target: bool = False
    # Duty cycle for text_heads_pass="corrupted"/"reconstructed": run the text heads on this
    # fraction of optimizer windows (Bernoulli from a generator seeded 0x5EED + global_step, so
    # identical across ranks). 1.0 = every step.
    mup_duty: float = 1.0
    # Synced section dropout (reconstructed route only): drop each valid section with this
    # probability. The JEPA predictor conditions on the kept sections and its predictions fill the
    # masked positions; MUP predicts the dropped sections. Replaces the mup_mask_prob draw when > 0.
    mup_ctx_drop_prob: float = 0.0
    # Multi-scale MUP: predict each masked utterance from image features at every scale in
    # `mup_scales`, with a head of the matching depth in `mup_depths` (cross-attention + MLP blocks).
    # A depth-1 head tends to converge to predicting the mean target. Queries are few (~max_regions),
    # so cross-attention stays cheap at fine scales.
    mup_scales: Tuple[str, ...] = ("s2",)
    mup_depths: Tuple[int, ...] = (1,)
    # Add the masked utterance's region centroid (from region_masks_s3) to its query so the target
    # is identifiable. Equivalent to mup_spatial="additive"; prefer mup_spatial.
    mup_cond_pos: bool = False
    # Anatomy pathway for the MUP cross-attention:
    #   "relbias":  region centroid, per-axis extent and downsampled occupancy enter only the attention
    #               logits as a learned per-head bias. The predictor state is initialized from the
    #               attended image value, so location alone cannot predict the text.
    #   "rope":     factorized 3D RoPE (query rotated by centroid, keys by grid position), same
    #               image-initialized state.
    #   "additive": q + MLP(centroid); identifiable but leaks the centroid to the output.
    #   "none":     no spatial mechanism (not identifiable).
    mup_spatial: str = "relbias"
    # Section targets with mup_spatial="none": reapply the fixed section-slot query to every
    # cross-attention Q projection without adding it to the predictor state. Section identity then
    # routes attention while all predicted content comes through image values.
    mup_persistent_routing_query: bool = False
    # Replace the masked token in the routing-only query with the train-split section mean. This gives
    # attention the section identity (e.g. "Lungs") without exposing patient text; the mean never
    # enters the value path. Requires persistent routing and conditional centering.
    mup_section_mean_routing_query: bool = False
    # Multiplicatively gate the image-derived predictor state with the section query. Unlike an
    # additive query residual, the gate cannot produce patient content on its own. Reuses q_in and
    # adds no parameters.
    mup_semantic_output_gate: bool = False
    # Normalize the gate source and bound the modulation to [0.5, 1.5], so a small learned q_in
    # norm cannot switch section conditioning off. Adds no parameters.
    mup_semantic_output_gate_normalized: bool = False
    # "off": separate MUP head; "shared": the RoPE JEPA predictor also performs the MUP role;
    # "split": a second instance of that predictor for the MUP role.
    joint_predictor: str = "off"
    # Target binding for the MUP loss:
    #   "index":   slot j predicts utterance j.
    #   "matched": set prediction; per volume, the loss is the minimum over assignments of masked
    #              predictions to masked targets (exhaustive over k! permutations, k <= 6).
    mup_binding: str = "index"
    # MUP loss:
    #   "raw_cosine":          1 - cos(pred, target).
    #   "clinical_innovation": CIR. Targets are centered by the train-split mean (per Matryoshka
    #                          dim, from mup_target_stats); the model predicts the residual
    #                          direction and each term is weighted by the residual norm, so
    #                          near-prior targets (normal findings, boilerplate) count less.
    mup_loss: str = "raw_cosine"
    # Matryoshka prefix applied to cached target embeddings first (slice, then L2-normalize).
    # 0 = full sidecar dim.
    mup_matryoshka_dim: int = 0
    # Target-stats file (torch.save dict: {"dims": {d: mean-vector}, "encoder", "manifest_hash"}).
    # Required for mup_loss="clinical_innovation"; must contain the mean for mup_matryoshka_dim
    # (means differ per dim because prefix normalization is nonlinear).
    mup_target_stats: str = ""
    # Centering for clinical-innovation targets: "global" subtracts E[x]; "conditional" subtracts
    # E[x | section_name]. The query carries section identity, so the residual target removes the
    # component predictable from it.
    mup_target_centering: str = "global"
    mup_condition_key: str = "section_name"
    mup_conditional_mean_shrinkage: float = 0.0
    # Counterfactual binding regularizer: each masked query is also evaluated against a deranged
    # study's image features, and the correct-image cosine must exceed it by
    # `mup_image_binding_margin`. Penalizes predictions built from batch-shared features. 0 = off.
    mup_image_binding_weight: float = 0.0
    mup_image_binding_margin: float = 0.05
    # Same-section in-batch NCE: each predicted section innovation must identify its own study's
    # target among in-batch targets for the same section. Penalizes patient-independent predictions.
    mup_conditional_nce_weight: float = 0.0
    mup_conditional_nce_temperature: float = 0.10
    # Within-study section NCE: each predicted innovation must identify its own section target
    # among the other sections present in the same study.
    mup_section_nce_weight: float = 0.0
    mup_section_nce_temperature: float = 0.10
    # Cross-section geometry loss: a predicted section pair is penalized when its cosine exceeds
    # the cosine of the matching target-residual pair by more than the margin. Discourages
    # predicting the same vector for every section of a study.
    mup_section_geometry_weight: float = 0.0
    mup_section_geometry_margin: float = 0.05
    # Feature source for the text-supervision heads (weak grounding, MUP):
    #   "stochastic":    JEPA every step; an unmasked student pass and the text heads run on a fraction
    #                    `mup_p_clean` of steps. The draw uses a generator seeded 0x5EED + global_step,
    #                    so every DDP rank makes the same decision each optimizer window.
    #   "clean":         an unmasked student pass every step (about 2x encoder compute).
    #   "reconstructed": one masked student pass; an image-only JEPA predictor fills the masked latents
    #                    before MUP, and gradients reach the encoder through that predictor. No target
    #                    text conditions the reconstructed map.
    #   "probe":         read the detached, unmasked EMA target; the heads are readout probes and do
    #                    not shape the encoder.
    text_heads_pass: str = "clean"
    mup_p_clean: float = 0.25          # "stochastic": probability that a step runs the unmasked pass
    text_ramp_frac: float = 0.05       # linear ramp of text losses from 0 over this fraction of steps
    # Per-scale MUP gradient scaling (None: {s1: 0, s2: 0.25, s3: 0.5, s4: 1.0}); shields fine
    # scales from the low-rank text target and concentrates report semantics at s3/s4.
    mup_stage_gamma: dict | None = None


@dataclass
class JECTRoPEConfig:
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
class JECTLossConfig:
    # Added on top of the LAC-JEPA image loss.
    lambda_anatomy_region_text: float = 0.05
    lambda_report_labels: float = 0.05
    lambda_seg_aux: float = 0.10
    contrastive_temperature: float = 0.07
    # Weak-grounding loss: grounding-head localization map vs the referenced-anatomy region.
    lambda_weakground: float = 0.5
    # Masked-utterance prediction loss.
    lambda_mup: float = 0.5

    # Anatomy-pooled JEPA: predict (text-conditioned) the EMA teacher's mean-pooled representation
    # of the referenced anatomy. A region-level target that is not diluted across per-token losses;
    # mean pooling is parameter-free, so the target stays stop-grad.
    lambda_region_jepa: float = 0.5
    region_jepa_loss: str = "cosine"     # cosine | l1 | l2
    anatomy_mask_frac: float = 0.5       # partial-mask this fraction of the anatomy

    # High-resolution cooldown weights: more fine-scale dense loss, less global text loss.
    cooldown_lambda_s1: float = 0.35
    cooldown_lambda_s2: float = 0.30
    cooldown_lambda_s3: float = 0.20
    cooldown_lambda_text: float = 0.03
    cooldown_lambda_seg_aux: float = 0.15


@dataclass
class JECTCooldownConfig:
    enabled: bool = False
    init_checkpoint: str | None = None
    flexible_load: bool = True
    skip_mismatched_rel_pos_bias: bool = True


@dataclass
class JECTConfig(LACJEPAConfig):
    text: JECTTextConfig = field(default_factory=JECTTextConfig)
    anatomy: JECTAnatomyConfig = field(default_factory=JECTAnatomyConfig)
    rope: JECTRoPEConfig = field(default_factory=JECTRoPEConfig)
    ject_loss: JECTLossConfig = field(default_factory=JECTLossConfig)
    cooldown: JECTCooldownConfig = field(default_factory=JECTCooldownConfig)

    def __post_init__(self):
        super().__post_init__()
        self.rope.rope_scales = tuple(str(s).lower() for s in self.rope.rope_scales)
        for s in self.rope.rope_scales:
            if s not in {"s1", "s2", "s3", "s4"}:
                raise ValueError(f"Invalid rope scale {s!r}; expected s1/s2/s3/s4.")
        self.patch.patch_size = _triple(self.patch.patch_size)
        # MUP option validation (enum membership and cross-field requirements), done at
        # construction rather than after an expensive setup.
        ac = self.anatomy
        _enum = {"mup_loss": ("raw_cosine", "clinical_innovation"),
                 "mup_binding": ("index", "matched"),
                 "mup_spatial": ("none", "additive", "relbias", "rope", None),
                 "text_heads_pass": ("clean", "probe", "stochastic", "corrupted", "reconstructed"),
                 "mup_target_centering": ("global", "conditional")}
        for f, allowed in _enum.items():
            v = getattr(ac, f, None)
            if v not in allowed:
                raise ValueError(f"anatomy.{f}={v!r}; expected one of {allowed}.")
        scales = tuple(getattr(ac, "mup_scales", ()) or ())
        depths = tuple(getattr(ac, "mup_depths", ()) or ())
        if getattr(ac, "use_masked_utterance_pred", False):
            bad = [s for s in scales if s not in ("s1", "s2", "s3", "s4")]
            if bad:
                raise ValueError(f"anatomy.mup_scales contains invalid scales {bad}.")
            if len(depths) not in (1, len(scales)):
                raise ValueError(f"mup_scales {scales} and mup_depths {depths} must have "
                                 f"equal length (or a single broadcastable depth).")
        lvl = str(getattr(self.text, "mup_target_level", "utterance"))
        if lvl not in ("utterance", "section", "report", "section_report"):
            raise ValueError(
                f"text.mup_target_level={lvl!r}; expected 'utterance', 'section', 'report', "
                "or 'section_report'.")
        if lvl in ("section", "report", "section_report"):
            if not str(getattr(self.text, "sr_sidecar_root", "")):
                raise ValueError(
                    f"mup_target_level={lvl!r} requires text.sr_sidecar_root.")
        if lvl == "section":
            if int(getattr(ac, "max_regions", 0)) < 11:
                raise ValueError("mup_target_level='section' needs anatomy.max_regions >= 11 "
                                 "(one slot per canonical section).")
        if lvl == "section_report":
            # 11 section slots plus a whole-report slot (pseudo-id 111); the conditional-centering
            # table has 12 rows: sections 0-10 and the report mean in row 11.
            if int(getattr(ac, "max_regions", 0)) < 12:
                raise ValueError("mup_target_level='section_report' needs anatomy.max_regions >= 12 "
                                 "(11 canonical sections + 1 whole-report slot).")
        if lvl == "report" and int(getattr(ac, "max_regions", 0)) != 1:
            raise ValueError("mup_target_level='report' requires anatomy.max_regions == 1 "
                             "(one whole-report target per study).")
        wg_max = int(getattr(ac, "wg_max_regions", 0) or 0)
        if wg_max < 0:
            raise ValueError("anatomy.wg_max_regions must be >= 0.")
        wq = str(getattr(ac, "wg_query", "sentence"))
        if wq not in ("sentence", "anatomy_phrase"):
            raise ValueError(f"anatomy.wg_query={wq!r}; expected 'sentence' or 'anatomy_phrase'.")
        if wq == "anatomy_phrase":
            import os as _os
            if wg_max == 0:
                raise ValueError("wg_query='anatomy_phrase' requires wg_max_regions > 0 (a separate WG slot set).")
            if not _os.path.isfile(str(getattr(ac, "wg_anatomy_emb", "") or "")):
                raise ValueError("wg_query='anatomy_phrase' requires anatomy.wg_anatomy_emb (a .pt with a [num_anatomies, cached_dim] table).")
        if (bool(getattr(ac, "use_grounding_head", False)) and lvl in ("section", "report", "section_report")
                and wg_max == 0):
            raise ValueError(f"use_grounding_head with mup_target_level={lvl!r} needs "
                             "anatomy.wg_max_regions > 0: section/report slots carry full-volume "
                             "placeholder masks, which would make the weak-grounding target degenerate.")
        persistent_query = bool(getattr(ac, "mup_persistent_routing_query", False))
        if persistent_query and (lvl not in ("section", "section_report")
                                 or getattr(ac, "mup_spatial", None) != "none"):
            raise ValueError("mup_persistent_routing_query requires section targets with "
                             "mup_spatial='none'.")
        mean_routing_query = bool(getattr(ac, "mup_section_mean_routing_query", False))
        if mean_routing_query and not persistent_query:
            raise ValueError("mup_section_mean_routing_query requires "
                             "mup_persistent_routing_query=True.")
        if mean_routing_query and not str(getattr(ac, "mup_target_stats", "") or ""):
            # The routing query needs the conditional means even when the loss is raw cosine.
            raise ValueError("mup_section_mean_routing_query requires mup_target_stats "
                             "(conditional section means).")
        if mean_routing_query and str(getattr(ac, "mup_target_centering", "global")) != "conditional":
            raise ValueError("mup_section_mean_routing_query requires conditional target centering.")
        semantic_output_gate = bool(getattr(ac, "mup_semantic_output_gate", False))
        if semantic_output_gate and not mean_routing_query:
            raise ValueError("mup_semantic_output_gate requires "
                             "mup_section_mean_routing_query=True.")
        normalized_output_gate = bool(getattr(
            ac, "mup_semantic_output_gate_normalized", False))
        if normalized_output_gate and not semantic_output_gate:
            raise ValueError("mup_semantic_output_gate_normalized requires "
                             "mup_semantic_output_gate=True.")
        ctxp = float(getattr(ac, "mup_ctx_drop_prob", 0.0))
        if not (0.0 <= ctxp < 1.0):
            raise ValueError(f"anatomy.mup_ctx_drop_prob={ctxp}; must be in [0, 1).")
        if ctxp > 0 and str(getattr(ac, "text_heads_pass", "clean")) != "reconstructed":
            raise ValueError("anatomy.mup_ctx_drop_prob > 0 is only defined for text_heads_pass='reconstructed' "
                             "(the kept-section-conditioned predictions are the fill MUP reads).")
        if ctxp > 0 and bool(getattr(ac, "mup_single_target", False)):
            # The single-target draw ignores the synced mask and could pick a kept (visible) section,
            # leaking the target.
            raise ValueError("anatomy.mup_ctx_drop_prob > 0 with mup_single_target=True is not defined: the "
                             "single-target draw ignores the synced mask and can select a kept (visible) section.")
        duty = float(getattr(ac, "mup_duty", 1.0))
        if not (0.0 < duty <= 1.0):
            raise ValueError(f"anatomy.mup_duty={duty}; must be in (0, 1].")
        if duty < 1.0 and str(getattr(ac, "text_heads_pass", "clean")) not in ("corrupted", "reconstructed"):
            raise ValueError("anatomy.mup_duty < 1.0 is only defined for "
                             "text_heads_pass='corrupted'/'reconstructed' (stochastic uses mup_p_clean).")
        bind_weight = float(getattr(ac, "mup_image_binding_weight", 0.0))
        bind_margin = float(getattr(ac, "mup_image_binding_margin", 0.05))
        if bind_weight < 0.0:
            raise ValueError("anatomy.mup_image_binding_weight must be >= 0.")
        if not (0.0 < bind_margin <= 2.0):
            raise ValueError("anatomy.mup_image_binding_margin must be in (0, 2].")
        nce_weight = float(getattr(ac, "mup_conditional_nce_weight", 0.0))
        nce_temperature = float(getattr(ac, "mup_conditional_nce_temperature", 0.10))
        if nce_weight < 0.0:
            raise ValueError("anatomy.mup_conditional_nce_weight must be >= 0.")
        if nce_temperature <= 0.0:
            raise ValueError("anatomy.mup_conditional_nce_temperature must be > 0.")
        if nce_weight > 0.0 and lvl != "section":
            raise ValueError("mup_conditional_nce_weight > 0 requires section-level targets.")
        if nce_weight > 0.0 and str(getattr(ac, "mup_target_centering", "global")) != "conditional":
            raise ValueError("mup_conditional_nce_weight > 0 requires conditional target centering.")
        section_nce_weight = float(getattr(ac, "mup_section_nce_weight", 0.0))
        section_nce_temperature = float(getattr(ac, "mup_section_nce_temperature", 0.10))
        if section_nce_weight < 0.0:
            raise ValueError("anatomy.mup_section_nce_weight must be >= 0.")
        if section_nce_temperature <= 0.0:
            raise ValueError("anatomy.mup_section_nce_temperature must be > 0.")
        if section_nce_weight > 0.0 and lvl != "section":
            raise ValueError("mup_section_nce_weight > 0 requires section-level targets.")
        if section_nce_weight > 0.0 and str(
                getattr(ac, "mup_target_centering", "global")) != "conditional":
            raise ValueError("mup_section_nce_weight > 0 requires conditional target centering.")
        section_geometry_weight = float(getattr(ac, "mup_section_geometry_weight", 0.0))
        section_geometry_margin = float(getattr(ac, "mup_section_geometry_margin", 0.05))
        if section_geometry_weight < 0.0:
            raise ValueError("anatomy.mup_section_geometry_weight must be >= 0.")
        if not (0.0 <= section_geometry_margin <= 2.0):
            raise ValueError("anatomy.mup_section_geometry_margin must be in [0, 2].")
        if section_geometry_weight > 0.0 and lvl != "section":
            raise ValueError("mup_section_geometry_weight > 0 requires section-level targets.")
        if section_geometry_weight > 0.0 and str(
                getattr(ac, "mup_target_centering", "global")) != "conditional":
            raise ValueError("mup_section_geometry_weight > 0 requires conditional target centering.")
        d = int(getattr(ac, "mup_matryoshka_dim", 0) or 0)
        if d < 0:
            raise ValueError(f"anatomy.mup_matryoshka_dim={d}; must be >= 0.")
        if getattr(ac, "mup_loss", "raw_cosine") == "clinical_innovation":
            if d <= 0:
                raise ValueError("mup_loss='clinical_innovation' requires "
                                 "mup_matryoshka_dim > 0 (per-dim train means).")
            sp = str(getattr(ac, "mup_target_stats", "") or "")
            if not sp:
                raise ValueError("mup_loss='clinical_innovation' requires mup_target_stats.")
            import os as _os
            if not _os.path.isfile(sp):
                raise ValueError(f"mup_target_stats file not found: {sp}")
            if str(getattr(ac, "mup_binding", "index")) == "matched":
                raise ValueError("mup_loss='clinical_innovation' with mup_binding='matched' "
                                 "is not supported.")
        if str(getattr(ac, "mup_target_centering", "global")) == "conditional":
            # Conditional (per-section) means are used by the CIR loss and/or the section-mean
            # routing query.
            if (getattr(ac, "mup_loss", "raw_cosine") != "clinical_innovation"
                    and not mean_routing_query):
                raise ValueError("mup_target_centering='conditional' is only defined for "
                                 "mup_loss='clinical_innovation' or with "
                                 "mup_section_mean_routing_query.")
            key = str(getattr(ac, "mup_condition_key", "section_name"))
            if key != "section_name":
                raise ValueError("conditional MUP centering currently supports only "
                                 "mup_condition_key='section_name'.")
            if lvl not in ("section", "section_report"):
                raise ValueError("conditional section-name MUP centering requires "
                                 "text.mup_target_level='section' or 'section_report'.")
            tau = float(getattr(ac, "mup_conditional_mean_shrinkage", 0.0))
            if tau < 0.0:
                raise ValueError("mup_conditional_mean_shrinkage must be >= 0.")
        elif str(getattr(ac, "mup_target_centering", "global")) != "global":
            raise ValueError(f"unknown mup_target_centering "
                             f"{getattr(ac, 'mup_target_centering', 'global')!r}.")
