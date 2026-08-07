"""ART/AR-LAC-JEPA model.

The model keeps the image-only LAC-JEPA teacher/student flow but adds:
  * report utterance text encoder
  * utterance(text)-conditioned latent prediction heads (anatomy_id is NOT conditioned on —
    it was a shortcut that let the model skip reading the phrase)
  * optional 3D RoPE global predictor at selected scales
  * local region-text embeddings for contrastive grounding
  * optional auxiliary segmentation head

Note: anatomy_id / anatomy masks still drive WHICH region gets masked + the region-JEPA
pooled target (the completion task), but never the predictor conditioning.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ject.config.art import ARTJEPAConfig
    from ject.backbone.rope import RopeGlobalPredictor3D, make_3d_coords, _even_axis_dims
    from ject.backbone.text import HashingTextTokenizer, HashingTextEncoder
    from ject.backbone.masking import LACMaskBatch, upsample_mask_3d
    from ject.backbone.encoder import DenseSwinUNETREncoder, DenseLatentDecoder, EMAUpdater, PrototypeProjector
    from ject.backbone.region_text import pool_region_embeddings, intra_volume_region_text_loss
except ImportError:
    from ject.config.art import ARTJEPAConfig
    from ject.backbone.rope import RopeGlobalPredictor3D, make_3d_coords, _even_axis_dims
    from ject.backbone.text import HashingTextTokenizer, HashingTextEncoder
    from ject.backbone.masking import LACMaskBatch, upsample_mask_3d
    from ject.backbone.encoder import DenseSwinUNETREncoder, DenseLatentDecoder, EMAUpdater, PrototypeProjector
    from ject.backbone.region_text import pool_region_embeddings, intra_volume_region_text_loss


class ConditionedPredictionHead(nn.Module):
    """MLP latent prediction head with FiLM conditioning from text/anatomy."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None) -> torch.Tensor:
        # Uniform predictor signature; FiLM uses only the cond vector (text_tokens ignored).
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.net(x)


class XAttnPredictionHead(nn.Module):
    """Predictor head conditioned by gated image→text cross-attention instead of FiLM.

    Same MLP body as ConditionedPredictionHead; conditioning is spatial (per-voxel) via the utterance
    SET, and the fusion's attention doubles as a grounding readout. Uniform forward signature so it is
    interchangeable with the FiLM head (it uses text_tokens, ignores the cond vector)."""

    def __init__(self, dim: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.fuse = GatedCrossAttentionFusion(dim, text_dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None) -> torch.Tensor:
        if text_tokens is not None:
            x = self.fuse(x, text_tokens, text_mask)          # spatial text conditioning (gated, zero-init)
        return self.net(self.norm(x))                         # None text -> unconditioned MLP fallback


class ScaleConditioner(nn.Module):
    """Project the report-utterance embedding into a per-scale conditioning vector.

    Text-ONLY by design: the anatomy-ID pathway was removed so the predictor must extract
    location from the utterance itself. Feeding a clean structured anatomy_id let the model
    take a shortcut (measured: swapping anatomy_id moved the prediction 90% vs text 4.4%),
    so it never learned to read the phrase — which is exactly what grounding needs.
    """

    def __init__(self, text_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)


class CrossAttention(nn.Module):
    """Generic multi-head cross-attention: queries attend to key/value tokens.

    Returns updated queries (residual) and the head-averaged attention map [B, Q, K]. Output
    projection is zero-init so the block starts as an identity residual and grows during training.
    """

    def __init__(self, dim: int, kv_dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.h = num_heads
        self.d = dim // num_heads
        self.scale = self.d ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, Q, C = q_in.shape
        K = kv_in.shape[1]
        h, d = self.h, self.d
        q = self.q(self.norm_q(q_in)).view(B, Q, h, d).transpose(1, 2)            # [B,h,Q,d]
        kv = self.kv(self.norm_kv(kv_in)).view(B, K, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                       # [B,h,K,d]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)          # [B,h,Q,K]
        out = (attn @ v).transpose(1, 2).reshape(B, Q, C)
        return q_in + self.proj(out), attn.mean(dim=1)                           # [B,Q,C], [B,Q,K]


class GatedCrossAttentionFusion(nn.Module):
    """Flamingo-style GATED image→text cross-attention (+ gated FFN) — the unified alternative to FiLM.

    Image tokens (queries) attend to per-utterance text tokens (keys/values), so conditioning is
    SPATIAL (per-voxel, unlike FiLM's uniform gamma/beta) and the text→image attention map doubles as a
    grounding readout — one mechanism serves both the predictor and the grounder. Zero-init tanh gates
    make the block start as identity and grow, for stable injection into the (pretrained) predictor.
    Non-trivial because we pass the utterance SET (multiple keys); with a single utterance it would
    reduce to a broadcast (≈ FiLM). Cost is O(N·L) (few text keys), NOT O(N^2). See
    GROUNDING_FOUNDATION_DESIGN.md.
    """

    def __init__(self, dim: int, kv_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.h = num_heads
        self.d = dim // num_heads
        self.scale = self.d ** -0.5
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(kv_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_gate = nn.Parameter(torch.zeros(1))          # tanh -> starts at 0 (identity)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, img: torch.Tensor, txt: torch.Tensor, txt_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False):
        B, N, C = img.shape
        L = txt.shape[1]
        q = self.q(self.nq(img)).view(B, N, self.h, self.d).transpose(1, 2)       # [B,h,N,d]
        kv = self.kv(self.nkv(txt)).view(B, L, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                        # [B,h,L,d]
        attn = (q @ k.transpose(-2, -1)) * self.scale                             # [B,h,N,L]
        if txt_mask is not None:
            attn = attn.masked_fill(~txt_mask.bool().view(B, 1, 1, L), torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        cross = self.proj(out)
        if txt_mask is not None:
            # softmax over an all-masked row is uniform (not zero) → would attend to padded text.
            # Zero the cross-attn contribution for samples with NO valid utterance (unconditioned).
            row_valid = txt_mask.bool().any(dim=1).view(B, 1, 1).to(cross.dtype)
            cross = cross * row_valid
        img = img + torch.tanh(self.attn_gate) * cross                            # gated cross-attn residual
        img = img + torch.tanh(self.ffn_gate) * self.ffn(img)                     # gated FFN residual
        if return_attn:
            return img, attn.mean(dim=1)                                          # [B,N,L] grounding readout
        return img


class UtteranceGroundingHead(nn.Module):
    """MaskFormer/DETR-style predictor-as-grounder, fit to Chest2Vec's per-utterance embeddings.

    Each *utterance* embedding (Chest2Vec's native EOS-pooled output — its trained representation,
    far stronger than a causal LM's per-word states) is a QUERY that attends over the image tokens
    and produces a per-voxel mask via dot-product with the image features. The per-finding mask is
    the grounding signal (agent D: query-based fusion is the SOTA mechanism).

    A learned `null_query` gives a text-agnostic prior (trained to the union of the volume's
    referenced regions), enabling a classifier-free-guidance counterfactual readout:
    `residual = sigmoid(mask | utterance) - sigmoid(mask | null)` — which subtracts the generic
    lung-blob prior that made the earlier zero-shot ART grounding read below chance.
    See GROUNDING_FOUNDATION_DESIGN.md.
    """

    def __init__(self, dim: int, cached_dim: int, num_heads: int = 8):
        super().__init__()
        self.q_in = nn.Linear(cached_dim, dim)                 # utterance emb -> query
        self.null_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.null_query, std=0.02)
        self.cross = CrossAttention(dim, dim, num_heads)       # queries attend to image tokens
        self.q_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.img_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.scale = dim ** -0.5

    def _masks(self, q: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_ref, attn = self.cross(q, img)                       # refine queries by attending to image
        logits = torch.einsum("bqd,bnd->bqn", self.q_out(q_ref), self.img_out(img)) * self.scale
        return logits, attn                                    # [B,Q,N], [B,Q,N]

    def forward(self, utt_emb: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # utt_emb [B,R,cached_dim] (per-utterance EOS-pooled) ; img [B,N,dim] (student S3 tokens)
        B = utt_emb.shape[0]
        logits, attn = self._masks(self.q_in(utt_emb), img)    # [B,R,N] per-utterance masks
        null_logits, _ = self._masks(self.null_query.expand(B, -1, -1), img)   # [B,1,N]
        return logits, null_logits.squeeze(1), attn            # [B,R,N], [B,N], [B,R,N]


def _rope_1d_perbatch(t: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """1D rotary on the last dim, with PER-(batch,token) positions.

    t:   [B, heads, T, dim_even]
    pos: [B, T]  (unlike art_jepa_rope._apply_1d_rope, positions vary per batch element — needed for
         the MUP query, whose 'position' is each sample's own region centroid, not a shared grid axis)
    """
    dim = t.shape[-1]
    if dim == 0:
        return t
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=t.device, dtype=t.dtype) / dim)   # [dim/2]
    freqs = pos.to(t.dtype).unsqueeze(-1) * inv_freq                                        # [B,T,dim/2]
    cos = freqs.cos().unsqueeze(1)                                                          # [B,1,T,dim/2]
    sin = freqs.sin().unsqueeze(1)
    t1, t2 = t[..., 0::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., 0::2] = t1 * cos - t2 * sin
    out[..., 1::2] = t1 * sin + t2 * cos
    return out


def _factorized_cross_rope(q: torch.Tensor, k: torch.Tensor, q_coords: torch.Tensor,
                           k_coords: torch.Tensor, theta: float = 10000.0):
    """Axial 3D RoPE for cross-attention: rotate q by its (per-sample) query coords and k by the shared
    image-grid coords, so the logit depends on the RELATIVE displacement (x_j - c_i).

    q [B,h,Q,d], k [B,h,K,d]; q_coords [B,Q,3] (z,y,x), k_coords [K,3]. Only a dedicated positional
    subspace is rotated (the content subspace is left untouched — the remainder split)."""
    hd = q.shape[-1]
    dz, dy, dx, rem = _even_axis_dims(hd)
    splits = [s for s in (dz, dy, dx, rem) if s > 0]
    qp = list(torch.split(q, splits, dim=-1))
    kp = list(torch.split(k, splits, dim=-1))
    B, K = q.shape[0], k.shape[2]
    kc = k_coords.to(q.dtype)                                                # [K,3]
    oq, ok, pidx = [], [], 0
    for axis, dim in enumerate((dz, dy, dx)):
        if dim <= 0:
            continue
        oq.append(_rope_1d_perbatch(qp[pidx], q_coords[..., axis], theta))
        ok.append(_rope_1d_perbatch(kp[pidx], kc[:, axis].unsqueeze(0).expand(B, K), theta))
        pidx += 1
    if rem > 0:
        oq.append(qp[pidx])
        ok.append(kp[pidx])
    return torch.cat(oq, dim=-1), torch.cat(ok, dim=-1)


class _MUPCrossAttention(nn.Module):
    """Image-only cross-attention for MUP with an optional per-head SPATIAL LOGIT bias and/or factorized
    3D RoPE. Anatomy geometry (centroid/extent/occupancy) may reweight WHICH image tokens are read
    (the logits), but the values are always image tokens — geometry NEVER reaches the output through the
    value path. `residual_q=False` returns the pure attended image value `proj(attn·v)` (used to
    INITIALIZE the predictor state from image evidence, so the mask/centroid query has no residual route
    to the output); `True` returns `q_in + proj(attn·v)`."""

    def __init__(self, dim: int, num_heads: int = 8, residual_q: bool = True, zero_init_proj: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.h, self.d = num_heads, dim // num_heads
        self.scale = self.d ** -0.5
        self.residual_q = residual_q
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if zero_init_proj:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, q_in, kv_in, bias=None, q_coords=None, k_coords=None, rope_theta: float = 10000.0):
        # q_in [B,Q,C]; kv_in [B,K,C]; bias [B,h,Q,K] (added to logits) or None; q/k_coords for RoPE.
        B, Q, C = q_in.shape
        K = kv_in.shape[1]
        h, d = self.h, self.d
        q = self.q(self.norm_q(q_in)).view(B, Q, h, d).transpose(1, 2)            # [B,h,Q,d]
        kv = self.kv(self.norm_kv(kv_in)).view(B, K, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                       # [B,h,K,d]
        if q_coords is not None and k_coords is not None:
            q, k = _factorized_cross_rope(q, k, q_coords, k_coords, rope_theta)
        logits = (q @ k.transpose(-2, -1)) * self.scale                          # [B,h,Q,K]
        if bias is not None:
            logits = logits + bias
        attn = logits.softmax(dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, Q, C))
        if self.residual_q:
            out = q_in + out
        return out, attn.mean(dim=1)                                             # [B,Q,C], [B,Q,K]


class _RegionRelativeBias(nn.Module):
    """Learned per-head spatial attention bias b_ijh = λ_h · MLP[(x_j-c_i)/σ_i, |·|, occupancy_ij, dist_ij].

    A soft, anisotropic anatomy prior over WHERE the query should read (region centroid c_i, per-axis
    extent σ_i, downsampled TotalSegmentator occupancy m_ij, radial distance) — broader than a point
    match, so peripheral / shell-like / elongated findings inside a coarse lobe region are still
    reachable. Enters ONLY the attention logits. The final MLP layer is zero-init and λ_h starts at 1,
    so the bias begins at 0 and GROWS — content routing dominates early, the anatomy prior is learned."""

    def __init__(self, num_heads: int, hidden: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(8, hidden)          # [rel(3), |rel|(3), occupancy(1), dist(1)]
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden, num_heads)
        nn.init.zeros_(self.lin2.weight)          # bias starts exactly 0 → attention is pure content at init
        nn.init.zeros_(self.lin2.bias)
        self.log_lambda = nn.Parameter(torch.zeros(num_heads))   # per-head strength (exp → 1 at init)

    def forward(self, k_coords, centroid, sigma, occupancy):
        # k_coords [K,3]; centroid [B,R,3]; sigma [B,R,3]; occupancy [B,R,K] → bias [B,heads,R,K]
        x = k_coords.to(centroid.dtype).view(1, 1, -1, 3)                        # [1,1,K,3]
        c = centroid.unsqueeze(2)                                                # [B,R,1,3]
        s = sigma.unsqueeze(2).clamp(min=1e-1)                                   # [B,R,1,3]
        rel = (x - c) / s                                                        # [B,R,K,3]
        dist = rel.norm(dim=-1, keepdim=True)                                    # [B,R,K,1]
        occ = occupancy.unsqueeze(-1).to(rel.dtype)                              # [B,R,K,1]
        feats = torch.cat([rel, rel.abs(), occ, dist], dim=-1)                   # [B,R,K,8]
        b = self.lin2(self.act(self.lin1(feats)))                                # [B,R,K,heads]
        b = b * torch.exp(self.log_lambda)
        return b.permute(0, 3, 1, 2).contiguous()                               # [B,heads,R,K]


class MaskedUtterancePredictor(nn.Module):
    """CEPA-style masked-utterance prediction, made SPATIAL via cross-attention (so it doubles as a
    self-supervised grounding objective).

    A random subset of the volume's per-utterance embeddings is replaced by a learned [MASK] token;
    each query cross-attends over the (unmasked, detached) image tokens and must PREDICT its true
    utterance embedding (cosine). To reconstruct the masked finding's text the query has to attend to
    the region that contains it → the cross-attention map is the grounding, and the prediction loss is a
    self-supervised grounding signal.

    IDENTIFIABILITY + no-leak (`spatial` controls the anatomy pathway):
      - Slot IDs are arbitrary (utterance order) → without a spatial mechanism the exchangeable-query
        loss is provably minimized by the per-volume-region text CENTROID regardless of depth (the
        measured ~0.395 study-centroid floor), i.e. NOT phrase-specific.
      - "additive" (legacy ablation): q ← q + MLP(centroid). Identifiable, BUT the residual blocks give
        the centroid a direct path to the output → the head can predict p(t | region) WITHOUT reading
        the image. Kept only as a baseline arm.
      - "relbias" (default, recommended): anatomy enters ONLY the attention logits as a learned
        region-relative bias; the recurrent state is INITIALIZED FROM THE ATTENDED IMAGE VALUE (not the
        raw query), so no centroid/mask residual reaches the predictor. The causal route is forced:
        region says WHERE to read → image says WHAT is there → predict the utterance.
      - "rope": factorized 3D RoPE (query rotated by centroid, keys by grid) — the clean position-
        matching ablation. Same image-initialized state, no query residual.
      - "none": no spatial mechanism (non-identifiable; diagnostic only).

    Values are ALWAYS image tokens; there is NO self-attention over the query set, so a masked query's
    only evidence path is the image cross-attention (never co-occurring-utterance copying).
    Depth 6–12 is required to fit the image→utterance map (I-JEPA 12, diffjepa 12/6, CEPA 6); depth-1
    saturates at the predict-mean floor. See GROUNDING_FOUNDATION_DESIGN.md."""

    def __init__(self, dim: int, cached_dim: int, num_heads: int = 8, max_regions: int = 6,
                 depth: int = 1, mlp_ratio: float = 4.0, spatial: str = "relbias"):
        super().__init__()
        if spatial not in ("none", "additive", "relbias", "rope"):
            raise ValueError(f"unknown mup spatial mode {spatial!r}")
        self.spatial = spatial
        self.q_in = nn.Linear(cached_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # A learned per-slot position embedding gives each masked query a DISTINCT identity (breaks the
        # tie in the FIRST attention). Slot index is arbitrary order → it cannot encode target location.
        self.slot_pos = nn.Parameter(torch.zeros(1, max_regions, dim))
        nn.init.trunc_normal_(self.slot_pos, std=0.02)
        # Legacy additive-centroid query conditioning (ablation arm only).
        self.pos_mlp = (nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
                        if spatial == "additive" else None)
        # Learned anatomy-relative logit bias (recommended arm).
        self.rel_bias = _RegionRelativeBias(num_heads) if spatial == "relbias" else None
        depth = max(1, depth)
        hidden = int(dim * mlp_ratio)
        # Block 0 INITIALIZES the state from the attended image value (residual_q=False, standard-init
        # proj) for relbias/rope so the centroid/mask query never reaches the output; for additive/none
        # the query IS the intended content path, so keep the query residual (residual_q=True, zero-init).
        img_init = self.spatial in ("relbias", "rope")
        self.blocks = nn.ModuleList()
        for i in range(depth):
            first = (i == 0)
            self.blocks.append(nn.ModuleDict({
                "cross": _MUPCrossAttention(
                    dim, num_heads,
                    residual_q=(not (first and img_init)),
                    zero_init_proj=(not (first and img_init)),
                ),
                "norm": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)),
            }))
        self.pred = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cached_dim))   # → predicted utt emb

    def forward(self, utt_emb: torch.Tensor, img: torch.Tensor, mask: torch.Tensor,
                geom: dict = None):
        # utt_emb [B,R,cached_dim]; img [B,N,dim]; mask [B,R] bool (True = masked/predicted).
        # geom (relbias): {k_coords [N,3], centroid [B,R,3], sigma [B,R,3], occupancy [B,R,N]}.
        # geom (rope):    {k_coords [N,3], centroid [B,R,3]}.
        B, R, _ = utt_emb.shape
        q = self.q_in(utt_emb)
        q = torch.where(mask.unsqueeze(-1), self.mask_token.to(q.dtype), q)      # blank masked CONTENT
        q = q + self.slot_pos[:, :R].to(q.dtype)                                # distinct slot id
        if self.pos_mlp is not None and geom is not None and geom.get("centroid") is not None:
            q = q + self.pos_mlp(geom["centroid"].to(q.dtype))                  # additive arm only

        bias = None
        q_coords = k_coords = None
        if self.spatial == "relbias" and self.rel_bias is not None and geom is not None:
            bias = self.rel_bias(geom["k_coords"], geom["centroid"], geom["sigma"], geom["occupancy"])
        elif self.spatial == "rope" and geom is not None:
            q_coords = geom["centroid"].to(q.dtype)                             # [B,R,3]
            k_coords = geom["k_coords"]                                         # [N,3]

        attn = None
        h = q
        for blk in self.blocks:
            # The spatial mechanism (relbias logit / rope) is re-applied at EVERY block so position
            # matching persists through depth; block 0 initialized the state from the attended image
            # value, so from here `h` is image-derived (no centroid/mask residual to the output).
            h, attn = blk["cross"](h, img, bias=bias, q_coords=q_coords, k_coords=k_coords)
            h = h + blk["mlp"](blk["norm"](h))
        return self.pred(h), attn                                              # [B,R,cached_dim], last [B,R,N]


def _masked_utterance_loss(pred: torch.Tensor, target: torch.Tensor,
                           predict: torch.Tensor) -> torch.Tensor:
    """1 - cos(pred, stopgrad(target)) over the MASKED valid utterances (predict = mask & valid)."""
    m = predict.to(pred.dtype)
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    cos = 1.0 - (p * t).sum(dim=-1)                                          # [B,R]
    return (cos * m).sum() / m.sum().clamp(min=1.0)


def _weak_grounding_loss(logits: torch.Tensor, target: torch.Tensor,
                         valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """0.5·BCE + 0.5·soft-Dice between per-utterance mask logits and binary region targets.

    logits/target: [B, R, N] (per utterance, per token); valid: [B, R] (bool). Dice complements BCE
    against the heavy foreground/background imbalance (a referenced region is a small fraction of the
    thoracic tokens); averaged over valid utterances only."""
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)  # [B,R]
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=-1)
    denom = p.sum(dim=-1) + target.sum(dim=-1)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)                                          # [B,R]
    per = 0.5 * bce + 0.5 * dice                                                              # [B,R]
    if valid is None:
        return per.mean()
    w = valid.to(dtype=per.dtype)
    return (per * w).sum() / w.sum().clamp(min=1.0)


def _region_masks_at(masks: torch.Tensor, from_grid, to_grid) -> torch.Tensor:
    """Upsample per-utterance region masks [B,R,N_from] → [B,R,N_to] (nearest) for finer-scale grounding."""
    if tuple(from_grid) == tuple(to_grid):
        return masks
    B, R, N = masks.shape
    up = upsample_mask_3d(masks.reshape(B * R, N), from_grid, to_grid)   # [B*R, N_to]
    return up.reshape(B, R, -1)


def _resize_occupancy(masks: torch.Tensor, src_grid, dst_grid) -> torch.Tensor:
    """Resize per-utterance region masks [B,R,N_src] → SOFT occupancy [B,R,N_dst] in [0,1].

    Unlike _region_masks_at (nearest, boolean), this uses area resampling so a coarser scale gets the
    FRACTION of each cell covered by the region (partial occupancy) — the anatomy-relative bias reads
    occupancy as a continuous feature. Handles both up- and down-sampling."""
    if tuple(src_grid) == tuple(dst_grid):
        return masks.to(torch.float32)
    B, R, N = masks.shape
    x = masks.reshape(B * R, 1, *src_grid).float()
    y = F.interpolate(x, size=tuple(dst_grid), mode="area")             # avg-pool on downsample, nearest-like up
    return y.reshape(B, R, -1)


def masked_average_pool(tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pool [B,N,C] tokens using optional [B,N] bool/float mask."""
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.to(device=tokens.device, dtype=tokens.dtype)
    if weights.ndim != 2 or weights.shape[:2] != tokens.shape[:2]:
        raise ValueError(f"Mask shape {tuple(weights.shape)} does not match tokens {tuple(tokens.shape[:2])}")
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1) / denom


def _region_residual_per_sample(g_pred: torch.Tensor, g_target: torch.Tensor, mode: str = "cosine") -> torch.Tensor:
    """Per-sample [B] pooled-region JEPA residual (so it can be averaged over VALID anatomy samples only)."""
    g_target = g_target.detach()
    if mode == "l1":
        return (g_pred - g_target).abs().mean(dim=-1)
    if mode == "l2":
        return (g_pred - g_target).pow(2).mean(dim=-1)
    gp = F.normalize(g_pred, dim=-1)
    gt = F.normalize(g_target, dim=-1)
    return 1.0 - (gp * gt).sum(dim=-1)


def _region_residual(g_pred: torch.Tensor, g_target: torch.Tensor, mode: str = "cosine",
                     valid: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Pooled-region JEPA residual, averaged over the VALID anatomy samples (a text-free/normal row has a
    zero anatomy mask -> a meaningless pooled region; excluding it is the per-sample-validity fix). Returns
    None if NO sample in the batch has a valid anatomy mask (nothing to supervise)."""
    per = _region_residual_per_sample(g_pred, g_target, mode)          # [B]
    if valid is None:
        return per.mean()
    vf = valid.to(per.device, per.dtype)
    denom = vf.sum()
    if float(denom) == 0.0:
        return None
    return (per * vf).sum() / denom


class ARTJEPAModel(nn.Module):
    """Anatomy-report-text guided dense JEPA model for 3D chest CT."""

    def __init__(self, config: ARTJEPAConfig):
        super().__init__()
        self.config = config
        mc = config.model
        tc = config.text
        ac = config.anatomy
        rc = config.rope

        self.tokenizer = HashingTextTokenizer(vocab_size=tc.vocab_size, max_tokens=tc.max_tokens)
        # Skip the (~vocab*text_dim) hashing embedding table entirely when using cached
        # Chest2Vec embeddings — it would be dead weight.
        self.text_encoder = None if tc.use_cached_embeddings else HashingTextEncoder(
            vocab_size=tc.vocab_size,
            text_dim=tc.text_dim,
            hidden_dim=tc.text_hidden_dim,
            dropout=tc.dropout,
        )
        # Projects cached Chest2Vec utterance embeddings (cached_dim) -> text_dim.
        self.text_in_proj = nn.Linear(tc.cached_dim, tc.text_dim)

        self.student_encoder = DenseSwinUNETREncoder(config)
        self.student_decoder = DenseLatentDecoder(config)
        self.target_encoder = deepcopy(self.student_encoder)
        self.target_decoder = deepcopy(self.student_decoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_decoder.parameters():
            p.requires_grad = False

        cond_dim = tc.text_dim
        self.cond_s1 = ScaleConditioner(tc.text_dim, cond_dim)
        self.cond_s2 = ScaleConditioner(tc.text_dim, cond_dim)
        self.cond_s3 = ScaleConditioner(tc.text_dim, cond_dim)
        self.cond_s4 = ScaleConditioner(tc.text_dim, cond_dim)   # global semantic scale conditioning

        self.pred_s1 = self._make_predictor("s1", mc.dim_s1, mc.heads[0], cond_dim)
        self.pred_s2 = self._make_predictor("s2", mc.dim_s2, mc.heads[1], cond_dim)
        self.pred_s3 = self._make_predictor("s3", mc.dim_s3, mc.heads[2], cond_dim)
        # s4 JEPA predictor on the ENCODER e4 (global, whole-volume). RoPE-global when s4 ∈ rope_scales.
        self.pred_s4 = self._make_predictor("s4", mc.dim_s4, mc.heads[3], cond_dim)
        # For xattn predictor conditioning: project the per-utterance embeddings into K/V dim.
        if getattr(tc, "predictor_cond", "film") == "xattn":
            if not tc.use_cached_embeddings:
                raise ValueError("predictor_cond='xattn' requires cached per-utterance embeddings "
                                 "(--use_cached_text): the cross-attention conditions on region_text_embs.")
            self.pred_text_proj = nn.Linear(tc.cached_dim, tc.text_dim)
        else:
            self.pred_text_proj = None

        self.use_prototypes = mc.use_prototypes
        if self.use_prototypes:
            self.proto_s1 = PrototypeProjector(mc.dim_s1, mc.num_prototypes_s1, mc.prototype_temperature)
            self.proto_s2 = PrototypeProjector(mc.dim_s2, mc.num_prototypes_s2, mc.prototype_temperature)
            self.proto_s3 = PrototypeProjector(mc.dim_s3, mc.num_prototypes_s3, mc.prototype_temperature)

        # Region-text grounding projection: use S3 region token by default.
        self.region_proj = nn.Sequential(nn.LayerNorm(mc.dim_s3), nn.Linear(mc.dim_s3, tc.text_dim), nn.LayerNorm(tc.text_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(tc.text_dim), nn.Linear(tc.text_dim, tc.text_dim), nn.LayerNorm(tc.text_dim))

        if ac.use_aux_segmentation:
            self.seg_head = nn.Conv3d(mc.dim_s1, ac.num_seg_classes, kernel_size=1)
        else:
            self.seg_head = None

        # Stage-0 predictor-as-grounder: each per-utterance Chest2Vec embedding is a query that
        # attends over the student S3 tokens → a per-utterance mask (weakly supervised vs the
        # utterance's anatomy region), plus a learned null query for the counterfactual readout.
        self.grounding_scale = getattr(ac, "grounding_scale", "s3")
        gdim = {"s1": mc.dim_s1, "s2": mc.dim_s2, "s3": mc.dim_s3}[self.grounding_scale]
        if getattr(ac, "use_grounding_head", False):
            if not tc.use_cached_embeddings:
                raise ValueError(
                    "use_grounding_head requires cached per-utterance embeddings (--use_cached_text): "
                    "the utterance-query grounder consumes region_text_embs, which the hashing text "
                    "path does not provide, so weak-grounding would silently train nothing.")
            self.grounding_head = UtteranceGroundingHead(gdim, tc.cached_dim, ac.grounding_num_heads)
        else:
            self.grounding_head = None

        if getattr(ac, "use_masked_utterance_pred", False):
            if not tc.use_cached_embeddings:
                raise ValueError("use_masked_utterance_pred requires cached per-utterance embeddings "
                                 "(--use_cached_text).")
            scales = tuple(getattr(ac, "mup_scales", ("s2",)))
            depths = tuple(getattr(ac, "mup_depths", (1,)))
            if len(depths) == 1 and len(scales) > 1:
                depths = depths * len(scales)            # broadcast a single depth over all scales
            if len(scales) != len(depths):
                raise ValueError(f"mup_scales {scales} and mup_depths {depths} must have equal length")
            dim_of = {"s1": mc.dim_s1, "s2": mc.dim_s2, "s3": mc.dim_s3, "s4": mc.dim_s4}
            # One text predictor per scale, each reading that scale's (unmasked, detached) target
            # features. Image-only (no co-occurring utterances in KV — independent utterances make
            # that inert, per the analysis). s1 is deliberately excluded upstream (config default).
            # `mup_spatial` selects the anatomy pathway (none|additive|relbias|rope); back-compat: an old
            # `mup_cond_pos=True` with no explicit spatial mode maps to the legacy "additive" arm.
            spatial = getattr(ac, "mup_spatial", None)
            if spatial is None:
                spatial = "additive" if getattr(ac, "mup_cond_pos", False) else "none"
            self.mup_spatial = spatial
            self.mup_heads = nn.ModuleDict({
                s: MaskedUtterancePredictor(dim_of[s], tc.cached_dim, ac.grounding_num_heads,
                                            max_regions=ac.max_regions, depth=d, spatial=spatial)
                for s, d in zip(scales, depths)
            })
        else:
            self.mup_heads = None

        # SIGReg projector (LeJEPA-style throwaway head): when SIGReg is active, apply it to a
        # projection of the student latents, NOT the raw latents, so the isotropic-Gaussian constraint
        # doesn't erode the local structure a dense/grounding head consumes. proj_dim=0 -> raw latents.
        lc = config.loss
        proj_dim = getattr(lc, "sigreg_proj_dim", 0)
        if getattr(lc, "anticollapse", "sigreg") in ("sigreg", "both") and proj_dim > 0:
            _sig_scales = [("s1", mc.dim_s1), ("s2", mc.dim_s2), ("s3", mc.dim_s3)]
            # s4 (global semantic scale) must be regularized too when it is an active objective — else it
            # can collapse silently while s1-s3's var/corr/SIGReg stay healthy (it is on the same encoder
            # but has its own stage4 blocks + pos_embed that can degenerate independently).
            if getattr(lc, "lambda_s4", 0.0) > 0 and getattr(mc, "dim_s4", 0) > 0:
                _sig_scales.append(("s4", mc.dim_s4))
            self.sigreg_proj = nn.ModuleDict({
                s: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))
                for s, d in _sig_scales
            })
        else:
            self.sigreg_proj = None

        self.ema = EMAUpdater(
            momentum_start=config.ema.momentum_start,
            momentum_end=config.ema.momentum_end,
            total_steps=config.ema.total_steps,
            schedule=config.ema.momentum_schedule,
        )

    def _make_predictor(self, scale: str, dim: int, num_heads: int, cond_dim: int) -> nn.Module:
        rc = self.config.rope
        tc = self.config.text
        grid = getattr(self.config.patch, f"grid_size_{scale}")
        # Cross-attention conditioning wins on the configured (non-RoPE) scales; the RoPE predictor and
        # unselected scales keep FiLM. xattn conditions on the per-utterance set (needs cached text).
        if (getattr(tc, "predictor_cond", "film") == "xattn"
                and scale in set(getattr(tc, "predictor_xattn_scales", ()))
                and not (rc.use_rope_predictor and scale in set(rc.rope_scales))):
            return XAttnPredictionHead(dim=dim, text_dim=tc.text_dim,
                                       num_heads=getattr(tc, "predictor_num_heads", 8))
        if rc.use_rope_predictor and scale in set(rc.rope_scales):
            # The RoPE predictor keeps FiLM, and ALSO gets image→text cross-attention when this scale is
            # selected for xattn conditioning (spatially-selective phrase conditioning FiLM can't give —
            # needed so pred_s3/s4 are reusable as ART-PGI grounding predictors).
            rope_xattn = (getattr(tc, "predictor_cond", "film") == "xattn"
                          and scale in set(getattr(tc, "predictor_xattn_scales", ())))
            return RopeGlobalPredictor3D(
                dim=dim, num_heads=num_heads, cond_dim=cond_dim, grid_size=grid,
                depth=rc.predictor_depth, mlp_ratio=rc.predictor_mlp_ratio,
                theta=rc.rope_theta, attn_drop=rc.attn_drop, proj_drop=rc.proj_drop,
                use_xattn=rope_xattn, text_dim=tc.text_dim,
            )
        return ConditionedPredictionHead(dim=dim, cond_dim=cond_dim)

    def tokenize_texts(self, texts: list[str | None], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokenizer.batch_encode(texts, device=device)

    def encode_condition(self, token_ids: torch.Tensor | None, attention_mask: torch.Tensor | None,
                         anatomy_id: torch.Tensor | None, batch_size: int, device,
                         text_emb_cached: torch.Tensor | None = None,
                         text_drop: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # NOTE: `anatomy_id` is accepted for call-site compatibility but is deliberately
        # UNUSED for predictor conditioning — the model conditions on the utterance only.
        null = None
        if text_emb_cached is not None:                       # cached Chest2Vec -> text_dim
            text_emb = self.text_in_proj(text_emb_cached)
            # A missing sidecar / text-free (normal) row arrives as an all-zero cached vector. text_in_proj
            # is biased, so text_in_proj(0)=bias — a nonzero "missing-text" condition, NOT null. Mark those
            # rows so they get the SAME exact-zero null as the CFG-dropped rows below; otherwise the coverage
            # logic counts them as image-only while the predictor secretly trains a distinct learned signal.
            null = (text_emb_cached.abs().sum(dim=-1) == 0)
        elif token_ids is not None and self.text_encoder is not None:
            text_emb = self.text_encoder(token_ids, attention_mask)
        else:
            text_emb = torch.zeros(batch_size, self.config.text.text_dim, device=device)
        # CFG dropout applied HERE so it nulls the conditioning on EVERY text path (cached, hashing, and
        # the zero fallback) — not just the cached branch. Without this the hashing path would keep full
        # text conditioning even at text_drop_prob=1.0, and the unconditioned mode would never be learned.
        # Combined with the missing-cached mask so null == exactly-zero text_emb from BOTH sources (one
        # consistent unconditioned representation, which is what the inference-time CFG residual assumes).
        if text_drop is not None:
            null = text_drop if null is None else (null | text_drop)
        if null is not None:
            text_emb = text_emb * (~null).view(-1, 1).to(text_emb.dtype)
        c1 = self.cond_s1(text_emb)
        c2 = self.cond_s2(text_emb)
        c3 = self.cond_s3(text_emb)
        aux = {
            "raw_text_emb": text_emb,
            "text_emb": self.text_proj(text_emb),
        }
        return c1, c2, c3, aux

    @torch.no_grad()
    def update_target(self) -> float:
        return self.ema.update([
            (self.student_encoder, self.target_encoder),
            (self.student_decoder, self.target_decoder),
        ])

    def _seg_logits_from_s1(self, s1_tokens: torch.Tensor, volume_shape: tuple[int, ...]) -> torch.Tensor:
        # Aux seg is a coarse regularizer -> keep it at the s1 grid (56³), NOT full 224³.
        # Full-res [B, num_classes, 224³] logits are memory-prohibitive (OOM even at batch 4);
        # the loss downsamples the target to match. The real segmenter is the finetune decoder.
        B, N, C = s1_tokens.shape
        D, H, W = self.config.patch.grid_size_s1
        x = s1_tokens.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return self.seg_head(x)

    def forward(self, volume: torch.Tensor, masks: LACMaskBatch,
                token_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                anatomy_id: torch.Tensor | None = None,
                anatomy_mask_s3: torch.Tensor | None = None,
                anatomy_mask_valid: torch.Tensor | None = None,
                window_id: torch.Tensor | None = None,
                ct_type_id: torch.Tensor | None = None,
                region_masks_s3: torch.Tensor | None = None,
                region_token_ids: torch.Tensor | None = None,
                region_attn: torch.Tensor | None = None,
                region_valid: torch.Tensor | None = None,
                text_emb_cached: torch.Tensor | None = None,
                region_text_embs: torch.Tensor | None = None,
                region_anatomy_ids: torch.Tensor | None = None,
                text_clean_step: bool = True) -> Dict[str, torch.Tensor]:
        B = volume.shape[0]
        device = volume.device

        # Classifier-free-guidance text dropout (training only): drop the predictor CONDITIONING to null
        # for a random subset of samples, so the model learns an unconditioned mode (needed for the CFG
        # grounding residual). Supervision targets (weak-grounding, region masks) are NOT affected.
        text_drop = None
        tdp = getattr(self.config.text, "text_drop_prob", 0.0)
        if self.training and tdp > 0:
            text_drop = torch.rand(B, device=device) < tdp
            if text_emb_cached is not None:            # null the xattn predictor's chosen-utterance token
                text_emb_cached = text_emb_cached * (~text_drop).view(B, 1).to(text_emb_cached.dtype)

        # `text_drop` also nulls the FiLM/predictor CONDITIONING for every text path inside
        # encode_condition (cached, hashing, and zero) — see the note there.
        c1, c2, c3, cond_aux = self.encode_condition(
            token_ids=token_ids, attention_mask=attention_mask, anatomy_id=anatomy_id,
            batch_size=B, device=device, text_emb_cached=text_emb_cached, text_drop=text_drop,
        )

        with torch.no_grad():
            t_enc = self.target_encoder(volume, window_id=window_id, mask_s1=None,
                                        ct_type_id=ct_type_id)
            t_dec = self.target_decoder(t_enc)

        s_enc = self.student_encoder(volume, window_id=window_id, mask_s1=masks.encoder_mask_s1,
                                     ct_type_id=ct_type_id)
        s_dec = self.student_decoder(s_enc)

        # ---- Feature source for the text-supervision heads (weak-grounding, MUP) ----
        # Neither head may read the MASKED student stream: the anatomy-sampled JEPA mask makes the
        # utterance's region a RECONSTRUCTION TARGET, so masked features carry a "this token was
        # reconstructed" artifact that shortcuts the region-mask supervision, and the gate/finetune
        # evaluate on the UNMASKED encoder (train/eval mismatch). Two allowed sources (text_heads_pass):
        #   "clean" — a 2nd, UNMASKED student forward: gradient SHAPES the encoder, no artifact, matches
        #             downstream. ~2x encoder compute. Skipped (aliased) when no text head is active.
        #   "probe" — the detached EMA target (unmasked): the heads are READOUT PROBES (no encoder
        #             gradient); the text-conditioned JEPA predictor shapes the encoder. ~2x cheaper.
        # "stochastic" (gpt-pro's recommended main model): JEPA runs every step, but the encoder-shaping
        # clean pass + text heads run only on a fraction of steps (text_clean_step, drawn Bernoulli(p_clean)
        # by the trainer). On non-clean steps the text heads are SKIPPED entirely (image-JEPA only), so the
        # expected text gradient is p_clean·∇L_text at ~1.25× (not 2×) avg encoder compute.
        text_pass = getattr(self.config.anatomy, "text_heads_pass", "clean")
        text_heads_active = (
            region_text_embs is not None and region_masks_s3 is not None
            and ((self.grounding_head is not None and self.config.art_loss.lambda_weakground > 0)
                 or (self.mup_heads is not None and self.config.art_loss.lambda_mup > 0))
        )
        run_text = text_heads_active and not (text_pass == "stochastic" and not text_clean_step)
        if not run_text:
            sf_enc, sf_dec = s_enc, s_dec                      # unused (text heads skipped this step)
        elif text_pass == "probe":
            sf_enc, sf_dec = t_enc, t_dec                      # detached EMA target → probe-only (no enc grad)
        else:                                                  # "clean" or ("stochastic" & clean step)
            sf_enc = self.student_encoder(volume, window_id=window_id, mask_s1=None, ct_type_id=ct_type_id)
            sf_dec = self.student_decoder(sf_enc)

        # Predictor conditioning text tokens (only used by xattn heads). PREPEND the CHOSEN utterance
        # (text_emb_cached — the one whose region the JEPA mask targets) to the region set, so the
        # predictor always attends to the utterance it is predicting (the region set is capped/filtered
        # in the loader and may omit it). The rest of the set gives multi-key spatial context.
        pred_txt = pred_txt_mask = None
        if self.pred_text_proj is not None:
            toks, msks = [], []
            if text_emb_cached is not None:
                toks.append(text_emb_cached.unsqueeze(1))                          # [B,1,cached] chosen
                msks.append((text_emb_cached.abs().sum(dim=-1) > 0).view(B, 1))    # valid iff non-null
            if region_text_embs is not None:
                toks.append(region_text_embs)                                     # [B,R,cached] context set
                rv = (region_valid if region_valid is not None
                      else (region_masks_s3.to(torch.bool).any(dim=-1) if region_masks_s3 is not None
                            else torch.ones(B, region_text_embs.shape[1], dtype=torch.bool, device=device)))
                msks.append(rv.to(torch.bool))
            if toks:
                pred_txt = self.pred_text_proj(torch.cat(toks, dim=1))            # [B, 1+R, text_dim]
                pred_txt_mask = torch.cat(msks, dim=1)
                if text_drop is not None:                                        # CFG drop → xattn zeroed
                    pred_txt_mask = pred_txt_mask & (~text_drop).view(B, 1)
        p1 = self.pred_s1(s_dec["s1"], c1, pred_txt, pred_txt_mask)
        p2 = self.pred_s2(s_dec["s2"], c2, pred_txt, pred_txt_mask)
        p3 = self.pred_s3(s_dec["s3"], c3, pred_txt, pred_txt_mask)
        # NULL-TEXT predictions for the DENSE (visible-token) loss only. The visible target is phrase-
        # independent, so supervising the TEXT-conditioned prediction there would teach the predictor to
        # ignore text (a headwind against the phrase-conditioning ART-PGI relies on). Masked JEPA keeps the
        # real phrase (p1/p2/p3 above); dense uses these text-free ones. (text_tokens=None → unconditioned.)
        dense_null = {}
        if getattr(self.config.loss, "use_dense_pred", False):
            _dscales = getattr(self.config.loss, "dense_pred_scales", ()) or (
                getattr(self.config.loss, "dense_pred_scale", "s2"),)
            _dpred = {"s1": (self.pred_s1, s_dec.get("s1"), c1),
                      "s2": (self.pred_s2, s_dec.get("s2"), c2),
                      "s3": (self.pred_s3, s_dec.get("s3"), c3)}
            for ds in _dscales:
                if ds in _dpred and _dpred[ds][1] is not None:
                    head, feat, cnd = _dpred[ds]
                    dense_null[f"dense_pred_{ds}"] = head(feat, cnd, None, None)
        # s4 GLOBAL scale JEPA: predict the target ENCODER e4 from the (masked) student encoder e4.
        # Skip the heavy depth-6 global predictor entirely when disabled, so --lambda_s4 0 is a real
        # no-s4 ablation / OOM-recovery path (not a zero-weighted term that still pays forward compute).
        p4 = None
        if getattr(self.config.loss, "lambda_s4", 0.0) > 0:
            c4 = self.cond_s4(cond_aux["raw_text_emb"])
            p4 = self.pred_s4(s_enc["e4"], c4, pred_txt, pred_txt_mask)

        region_token = masked_average_pool(s_dec["s3"], anatomy_mask_s3)
        region_emb = self.region_proj(region_token)

        # Anatomy-pooled JEPA residual: predict the pooled (text-conditioned) representation
        # of the referenced anatomy vs the EMA teacher's pooled representation of the same
        # region. Mean-pool keeps the target parameter-free / stop-grad (no collapse).
        region_jepa = None
        if anatomy_mask_s3 is not None and self.config.art_loss.lambda_region_jepa > 0:
            g_pred = masked_average_pool(p3, anatomy_mask_s3)
            g_target = masked_average_pool(t_dec["s3"], anatomy_mask_s3)
            region_jepa = _region_residual(g_pred, g_target, self.config.art_loss.region_jepa_loss,
                                           valid=anatomy_mask_valid)   # avg over valid-anatomy samples only

        # Predictor-as-grounder (Stage 0): each per-utterance Chest2Vec embedding is a query that
        # attends over the S3 tokens → a per-utterance mask [B,R,N3]; a learned null query gives the
        # text-agnostic prior for the counterfactual (CFG) residual readout.
        #
        # IMPORTANT: read the UNMASKED target-encoder features (t_dec), NOT the student's masked
        # stream. The encoder mask is sampled from the SAME anatomy region as the weak-ground target,
        # so a head reading the masked stream could minimize the loss by detecting mask-token
        # artifacts instead of learning phrase→region — and that shortcut vanishes at zero-shot eval
        # (no mask). t_dec is unmasked and detached, matching the eval condition; the backbone is
        # shaped by the JEPA objective, and Stage-0 probes whether those features support grounding.
        # Grounding/MUP operate at self.grounding_scale (default s2, finer than s3): read that scale's
        # CLEAN student features (unmasked pass — gradient reaches the ENCODER, no mask artifact, matches
        # the downstream/gate feature source) and upsample the s3 region-mask targets to it.
        gs = self.grounding_scale
        gimg = sf_dec[gs]                                                         # [B, N_gs, gdim] (clean online)
        gmasks = (region_masks_s3 if (region_masks_s3 is None or gs == "s3")
                  else _region_masks_at(region_masks_s3, self.config.patch.grid_size_s3,
                                        getattr(self.config.patch, f"grid_size_{gs}")))

        grounding_logits = grounding_null = grounding_attn = grounding_weak = None
        if self.grounding_head is not None and region_text_embs is not None and run_text:
            grounding_logits, grounding_null, grounding_attn = self.grounding_head(
                region_text_embs, gimg)                                           # [B,R,Ngs],[B,Ngs],[B,R,Ngs]
            if gmasks is not None and self.config.art_loss.lambda_weakground > 0:
                rv = (region_valid if region_valid is not None
                      else gmasks.to(torch.bool).any(dim=-1))
                tgt = gmasks.to(grounding_logits.dtype)
                wl = _weak_grounding_loss(grounding_logits, tgt, rv)
                # Null query → union of the volume's referenced regions (the generic prior it must
                # subtract). Supervise it ONLY on samples that actually have ≥1 valid region — else a
                # missing-sidecar sample (union all-zero) would train the shared null to all-background.
                has_region = rv.any(dim=1)                                        # [B]
                union = (tgt * rv.unsqueeze(-1).to(tgt.dtype)).sum(dim=1).clamp(max=1.0)  # [B,N3]
                wl_null = _weak_grounding_loss(grounding_null.unsqueeze(1), union.unsqueeze(1),
                                               has_region.unsqueeze(1))
                grounding_weak = wl + 0.5 * wl_null

        # Masked-utterance prediction (CEPA "mask + predict text", spatial via cross-attention): mask a
        # random subset of the VALID utterances and predict their embeddings from the UNMASKED image.
        mup_loss = None
        if (self.mup_heads is not None and region_text_embs is not None and region_masks_s3 is not None
                and self.config.art_loss.lambda_mup > 0 and run_text):
            rv = (region_valid if region_valid is not None
                  else region_masks_s3.to(torch.bool).any(dim=-1))
            p = float(getattr(self.config.anatomy, "mup_mask_prob", 0.5))
            predict = (torch.rand(region_text_embs.shape[:2], device=device) < p) & rv  # [B,R] mask & valid
            if bool(predict.any()):
                w = 1.0 / len(self.mup_heads)                                    # split lambda_mup across scales
                # Per-scale gradient attenuation (gpt-pro): the low-rank (~33-d) Chest2Vec target can
                # compress fine features if it pulls early scales too hard. Push report semantics mostly at
                # s4/s3; keep s2 weak and s1 protected (excluded). Loss multiplier attenuates the encoder
                # gradient for that scale.
                mup_gamma = getattr(self.config.anatomy, "mup_stage_gamma", None) \
                    or {"s1": 0.0, "s2": 0.25, "s3": 0.5, "s4": 1.0}
                mup_loss = region_text_embs.new_zeros(())
                # ---- anatomy geometry for the spatial mechanism (shared coord system: make_3d_coords [-1,1]) ----
                # Any spatial mode needs the per-utterance region centroid; relbias additionally needs the
                # per-axis extent (sigma) and the per-scale occupancy. Centroid/sigma are scale-independent
                # (computed on s3 in normalized coords); occupancy + key coords are resized per scale.
                need_geom = any(h.spatial != "none" for h in self.mup_heads.values())
                grid_s3 = self.config.patch.grid_size_s3
                centroid = sigma = None
                if need_geom:
                    coords3 = make_3d_coords(grid_s3, device=device, dtype=torch.float32,
                                             normalize=True)                     # [N3,3] in [-1,1]
                    m = region_masks_s3.to(coords3.dtype)                         # [B,R,N3]
                    msum = m.sum(-1, keepdim=True).clamp(min=1.0)                 # [B,R,1]
                    centroid = torch.einsum("brn,nc->brc", m, coords3) / msum     # [B,R,3]
                    # per-axis extent σ = sqrt(E[(x-c)^2]) over the region mask
                    d2 = (coords3.view(1, 1, -1, 3) - centroid.unsqueeze(2)) ** 2  # [B,R,N3,3]
                    sigma = torch.sqrt((m.unsqueeze(-1) * d2).sum(2) / msum)      # [B,R,3]
                for scale, head in self.mup_heads.items():
                    # KV = CLEAN STUDENT (online, unmasked) features, so MUP gradient reaches the ENCODER
                    # (the whole point of MUP is to make the ENCODER text-aware/localizable) WITHOUT the
                    # masked-stream reconstruction artifact, and matching the downstream feature source.
                    # The detached EMA target (t_enc/t_dec) would train only the head. Text target stays
                    # frozen (external Chest2Vec) → no collapse risk from student KV. s4 lives in encoder.
                    gimg_s = sf_enc["e4"] if scale == "s4" else sf_dec[scale]
                    geom = None
                    if head.spatial == "additive":
                        geom = {"centroid": centroid}
                    elif head.spatial in ("relbias", "rope"):
                        grid_s = getattr(self.config.patch, f"grid_size_{scale}")
                        k_coords = make_3d_coords(grid_s, device=device, dtype=torch.float32,
                                                  normalize=True)                 # [Ns,3] in [-1,1]
                        geom = {"k_coords": k_coords, "centroid": centroid}
                        if head.spatial == "relbias":
                            geom["sigma"] = sigma
                            geom["occupancy"] = _resize_occupancy(region_masks_s3, grid_s3, grid_s)
                    mup_pred, _ = head(region_text_embs, gimg_s, predict, geom=geom)  # image + anatomy prior
                    g = float(mup_gamma.get(scale, 1.0))
                    mup_loss = mup_loss + w * g * _masked_utterance_loss(mup_pred, region_text_embs, predict)

        out: Dict[str, torch.Tensor] = {
            "pred_s1": p1,
            "pred_s2": p2,
            "pred_s3": p3,
            **dense_null,                                    # dense_pred_s{1,2,3}: null-text (for dense loss)
            "target_s1": t_dec["s1"].detach(),
            "target_s2": t_dec["s2"].detach(),
            "target_s3": t_dec["s3"].detach(),
            "loss_mask_s1": masks.loss_mask_s1,
            "loss_mask_s2": masks.loss_mask_s2,
            "loss_mask_s3": masks.loss_mask_s3,
            "encoder_mask_s1": masks.encoder_mask_s1,
            "student_s1": s_dec["s1"],
            "student_s2": s_dec["s2"],
            "student_s3": s_dec["s3"],
            "region_emb": region_emb,
            # NB: pred_s4/target_s4 (+ student_s4) are added below, only when s4 is an active objective.
            "text_emb": cond_aux["text_emb"],
            "raw_text_emb": cond_aux["raw_text_emb"],
        }
        if region_jepa is not None:
            out["region_jepa"] = region_jepa
        if grounding_logits is not None:
            out["grounding_map"] = grounding_logits       # [B,R,N3] per-utterance mask logits
            out["grounding_null"] = grounding_null        # [B,N3] text-agnostic prior (CFG null)
            out["grounding_attn"] = grounding_attn        # [B,R,N3] query→image attention
        if grounding_weak is not None:
            out["weakground"] = grounding_weak
        if mup_loss is not None:
            out["mup"] = mup_loss
        # SIGReg projector outputs (the loss regularizes THESE, not the raw student latents).
        # Subsample tokens to sigreg_max_tokens BEFORE the projector, so we never project/backprop
        # through the hundreds of thousands of tokens the loss would discard (patch-4 224³ → 175k s1
        # tokens): only the ~512 sampled tokens are projected and retained.
        # s4 (global scale) is monitored/regularized only when it is an active objective. student_s4 is
        # the student ENCODER e4 (s4 lives in the encoder, not the decoder); emit it so the anti-collapse
        # var/cov/corr/SIGReg loops cover it (else s4 can collapse without tripping the guard).
        s4_active = getattr(self.config.loss, "lambda_s4", 0.0) > 0 and "e4" in s_enc
        if s4_active:
            out["student_s4"] = s_enc["e4"]
            out["pred_s4"] = p4                                   # p4 is computed only when s4 is active
            out["target_s4"] = t_enc["e4"].detach()
        if self.sigreg_proj is not None:
            max_tok = getattr(self.config.loss, "sigreg_max_tokens", 512)
            sig_srcs = {"s1": s_dec["s1"], "s2": s_dec["s2"], "s3": s_dec["s3"]}
            if s4_active and "s4" in self.sigreg_proj:
                sig_srcs["s4"] = s_enc["e4"]
            for s, src in sig_srcs.items():
                flat = src.reshape(-1, src.shape[-1])                    # [B*N, C]
                if flat.shape[0] > max_tok:
                    idx = torch.randperm(flat.shape[0], device=flat.device)[:max_tok]
                    flat = flat[idx]
                out[f"sigreg_proj_{s}"] = self.sigreg_proj[s](flat)      # [<=max_tok, proj_dim]

        # Intra-volume region↔utterance grounding (batch-1-safe): pool student s3 per
        # region, project, and match to each region's utterance embedding within the volume.
        if region_masks_s3 is not None and (region_token_ids is not None or region_text_embs is not None):
            R = region_masks_s3.shape[1]
            reg_img = self.region_proj(pool_region_embeddings(s_dec["s3"], region_masks_s3))  # [B,R,text_dim]
            if region_text_embs is not None:                                                  # cached Chest2Vec
                reg_txt = self.text_in_proj(region_text_embs)
            else:
                T = region_token_ids.shape[-1]
                reg_txt = self.text_encoder(region_token_ids.reshape(B * R, T),
                                            region_attn.reshape(B * R, T)).reshape(B, R, -1)
            reg_txt = self.text_proj(reg_txt)                                                 # [B,R,text_dim]
            if region_valid is None:
                region_valid = region_masks_s3.to(torch.bool).any(dim=-1)
            out["region_text_intra"] = intra_volume_region_text_loss(
                reg_img, reg_txt, region_valid,
                temperature=self.config.art_loss.contrastive_temperature,
                region_anatomy_ids=region_anatomy_ids)

        if self.use_prototypes:
            logits1, labels1 = self.proto_s1(p1, t_dec["s1"].detach())
            logits2, labels2 = self.proto_s2(p2, t_dec["s2"].detach())
            logits3, labels3 = self.proto_s3(p3, t_dec["s3"].detach())
            out.update({
                "proto_logits_s1": logits1,
                "proto_labels_s1": labels1,
                "proto_logits_s2": logits2,
                "proto_labels_s2": labels2,
                "proto_logits_s3": logits3,
                "proto_labels_s3": labels3,
            })

        if self.seg_head is not None:
            out["seg_logits"] = self._seg_logits_from_s1(s_dec["s1"], volume.shape)

        return out

    def get_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        no_decay_terms = ("bias", "norm", "pos_embed", "mask_token", "prototypes", "embed")
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(term in name for term in no_decay_terms):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]
