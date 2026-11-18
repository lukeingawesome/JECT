"""JECT model.

Image JEPA teacher/student with:
  * report utterance text encoder
  * text-conditioned latent prediction heads (conditioned on the utterance, not on anatomy_id)
  * optional 3D RoPE global predictor at selected scales
  * local region-text embeddings for contrastive grounding
  * optional auxiliary segmentation head

Anatomy masks select the masked region and the pooled region-JEPA target only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ject.config.ject import JECTConfig
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
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None,
                coords: torch.Tensor = None) -> torch.Tensor:
        # Uniform predictor signature; FiLM uses only `cond` (text_tokens/coords are ignored).
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.net(x)


class XAttnPredictionHead(nn.Module):
    """Predictor head conditioned by gated image-to-text cross-attention instead of FiLM.

    Same MLP body as ConditionedPredictionHead. Uses text_tokens and ignores the cond vector.
    """

    def __init__(self, dim: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.fuse = GatedCrossAttentionFusion(dim, text_dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None,
                coords: torch.Tensor = None) -> torch.Tensor:
        # coords is ignored; kept for a uniform predictor signature.
        if text_tokens is not None:
            x = self.fuse(x, text_tokens, text_mask)          # gated, zero-init
        return self.net(self.norm(x))                         # unconditioned if text_tokens is None


class ScaleConditioner(nn.Module):
    """Project the report-utterance embedding into a per-scale conditioning vector.

    Text only: location must be inferred from the utterance, not from an anatomy id.
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

    Returns updated queries (residual) and the head-averaged attention map [B, Q, K]. The output
    projection is zero-initialized so the block starts as an identity residual.
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
    """Flamingo-style gated image-to-text cross-attention followed by a gated FFN.

    Image tokens [B, N, C] attend to per-utterance text tokens [B, L, kv_dim], giving per-voxel
    conditioning. Zero-initialized tanh gates make the block start as identity. Cost is O(N*L).
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
        self.attn_gate = nn.Parameter(torch.zeros(1))          # tanh(0) = 0: identity at init
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
            # Softmax over an all-masked row is uniform, so zero the contribution for samples with
            # no valid utterance.
            row_valid = txt_mask.bool().any(dim=1).view(B, 1, 1).to(cross.dtype)
            cross = cross * row_valid
        img = img + torch.tanh(self.attn_gate) * cross                            # gated cross-attn residual
        img = img + torch.tanh(self.ffn_gate) * self.ffn(img)                     # gated FFN residual
        if return_attn:
            return img, attn.mean(dim=1)                                          # [B,N,L] grounding readout
        return img


class UtteranceGroundingHead(nn.Module):
    """MaskFormer/DETR-style grounding head over per-utterance text embeddings.

    Each utterance embedding is a query that attends over the image tokens and yields a per-token
    mask logit via a dot product with the image features. A learned `null_query`, trained to the
    union of the volume's referenced regions, gives a text-agnostic prior for the counterfactual
    readout `sigmoid(mask | utterance) - sigmoid(mask | null)`.
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
        q_ref, attn = self.cross(q, img)                       # queries attend to image tokens
        logits = torch.einsum("bqd,bnd->bqn", self.q_out(q_ref), self.img_out(img)) * self.scale
        return logits, attn                                    # [B,Q,N], [B,Q,N]

    def forward(self, utt_emb: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # utt_emb [B,R,cached_dim] (per-utterance EOS-pooled) ; img [B,N,dim] (student S3 tokens)
        B = utt_emb.shape[0]
        logits, attn = self._masks(self.q_in(utt_emb), img)    # [B,R,N] per-utterance masks
        null_logits, _ = self._masks(self.null_query.expand(B, -1, -1), img)   # [B,1,N]
        return logits, null_logits.squeeze(1), attn            # [B,R,N], [B,N], [B,R,N]


def _rope_1d_perbatch(t: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """1D rotary embedding on the last dim with per-(batch, token) positions.

    t:   [B, heads, T, dim_even]
    pos: [B, T]
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
    """Axial 3D RoPE for cross-attention; logits depend on the displacement (x_j - c_i).

    q [B,h,Q,d], k [B,h,K,d]; q_coords [B,Q,3] (z,y,x), k_coords [K,3]. The remainder channels
    after the per-axis split are left unrotated."""
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
    """Image-only cross-attention for MUP with an optional per-head spatial logit bias or 3D RoPE.

    Geometry only affects the attention logits; values are always image tokens.
    `residual_q=False` returns `proj(attn @ v)`, `True` returns `q_in + proj(attn @ v)`.
    """

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

    def forward(self, q_in, kv_in, bias=None, q_coords=None, k_coords=None,
                rope_theta: float = 10000.0, routing_q=None):
        # q_in [B,Q,C]; kv_in [B,K,C]; bias [B,h,Q,K] (added to logits) or None; q/k_coords for RoPE.
        B, Q, C = q_in.shape
        K = kv_in.shape[1]
        h, d = self.h, self.d
        q_source = q_in if routing_q is None else q_in + routing_q
        q = self.q(self.norm_q(q_source)).view(B, Q, h, d).transpose(1, 2)        # [B,h,Q,d]
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
    """Learned per-head attention bias b_ijh = lambda_h * MLP[(x_j-c_i)/sigma_i, |.|, occupancy_ij, dist_ij].

    c_i is the region centroid, sigma_i the per-axis extent, occupancy_ij the region's soft occupancy
    of key token j. The last layer is zero-initialized, so the bias starts at 0.
    """

    def __init__(self, num_heads: int, hidden: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(8, hidden)          # [rel(3), |rel|(3), occupancy(1), dist(1)]
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden, num_heads)
        nn.init.zeros_(self.lin2.weight)          # bias is 0 at init
        nn.init.zeros_(self.lin2.bias)
        self.log_lambda = nn.Parameter(torch.zeros(num_heads))   # per-head strength, exp(0) = 1 at init

    def forward(self, k_coords, centroid, sigma, occupancy):
        # k_coords [K,3]; centroid [B,R,3]; sigma [B,R,3]; occupancy [B,R,K] -> bias [B,heads,R,K]
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
    """Masked-utterance prediction (MUP) by cross-attention over image tokens.

    A random subset of the per-utterance embeddings is replaced by a learned mask token; each query
    cross-attends over the image tokens and predicts its utterance embedding.

    `spatial` selects how anatomy enters:
      - "relbias": learned region-relative bias on the attention logits.
      - "rope": factorized 3D RoPE (queries rotated by region centroid, keys by grid position).
      - "additive": q <- q + MLP(centroid); the centroid has a residual path to the output.
      - "none": no spatial mechanism.
    For "relbias", "rope" and "none" the state is initialized from the attended image value rather
    than the query. Values are always image tokens and there is no self-attention over queries.
    """

    def __init__(self, dim: int, cached_dim: int, num_heads: int = 8, max_regions: int = 6,
                 depth: int = 1, mlp_ratio: float = 4.0, spatial: str = "relbias",
                 persistent_routing_query: bool = False,
                 semantic_output_gate: bool = False,
                 semantic_output_gate_normalized: bool = False):
        super().__init__()
        if spatial not in ("none", "additive", "relbias", "rope"):
            raise ValueError(f"unknown mup spatial mode {spatial!r}")
        self.spatial = spatial
        self.persistent_routing_query = bool(persistent_routing_query)
        self.semantic_output_gate = bool(semantic_output_gate)
        self.semantic_output_gate_normalized = bool(semantic_output_gate_normalized)
        self.q_in = nn.Linear(cached_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Per-slot embedding gives otherwise identical masked queries distinct identities.
        self.slot_pos = nn.Parameter(torch.zeros(1, max_regions, dim))
        nn.init.trunc_normal_(self.slot_pos, std=0.02)
        # Additive centroid query conditioning ("additive" mode).
        self.pos_mlp = (nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
                        if spatial == "additive" else None)
        # Learned anatomy-relative logit bias ("relbias" mode).
        self.rel_bias = _RegionRelativeBias(num_heads) if spatial == "relbias" else None
        depth = max(1, depth)
        hidden = int(dim * mlp_ratio)
        # Block 0 initializes the state from the attended image value (no query residual, standard-init
        # proj) except in "additive" mode, where the query residual is the intended content path.
        img_init = self.spatial in ("relbias", "rope", "none")
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
        self.pred = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cached_dim))   # predicted utterance embedding

    def forward(self, utt_emb: torch.Tensor, img: torch.Tensor, mask: torch.Tensor,
                geom: dict = None, routing_emb: torch.Tensor | None = None):
        # utt_emb [B,R,cached_dim]; img [B,N,dim]; mask [B,R] bool (True = masked/predicted).
        # geom (relbias): {k_coords [N,3], centroid [B,R,3], sigma [B,R,3], occupancy [B,R,N]}.
        # geom (rope):    {k_coords [N,3], centroid [B,R,3]}.
        B, R, _ = utt_emb.shape
        q_content = self.q_in(utt_emb)
        slot_pos = self.slot_pos[:, :R].to(q_content.dtype)
        q = torch.where(mask.unsqueeze(-1), self.mask_token.to(q_content.dtype), q_content)
        q = q + slot_pos                                                        # distinct slot id
        if self.pos_mlp is not None and geom is not None and geom.get("centroid") is not None:
            q = q + self.pos_mlp(geom["centroid"].to(q.dtype))                  # "additive" mode

        bias = None
        q_coords = k_coords = None
        if self.spatial == "relbias" and self.rel_bias is not None and geom is not None:
            bias = self.rel_bias(geom["k_coords"], geom["centroid"], geom["sigma"], geom["occupancy"])
        elif self.spatial == "rope" and geom is not None:
            q_coords = geom["centroid"].to(q.dtype)                             # [B,R,3]
            k_coords = geom["k_coords"]                                         # [N,3]

        attn = None
        if routing_emb is not None and not self.persistent_routing_query:
            raise ValueError("routing_emb is only valid with persistent_routing_query=True")
        if routing_emb is not None and routing_emb.shape != utt_emb.shape:
            raise ValueError(f"routing_emb shape {routing_emb.shape} != target shape {utt_emb.shape}")
        routing_q = q if self.persistent_routing_query else None
        semantic_q = None
        if routing_emb is not None:
            # The section mean feeds only the attention queries; detached, never added to the state.
            semantic_q = self.q_in(routing_emb.detach())
            routing_q = torch.where(mask.unsqueeze(-1), semantic_q, q_content) + slot_pos
        # With a routing query the state starts at zero and is built only from attended image values.
        h = torch.zeros_like(q) if routing_q is not None else q
        for blk in self.blocks:
            # The spatial bias / RoPE is applied at every block.
            h, attn = blk["cross"](
                h, img, bias=bias, q_coords=q_coords, k_coords=k_coords,
                routing_q=routing_q)
            h = h + blk["mlp"](blk["norm"](h))
        if self.semantic_output_gate:
            if semantic_q is None:
                raise ValueError("semantic_output_gate requires routing_emb")
            # Multiplicative gate from the section identity; reuses q_in, adds no parameters.
            gate_source = semantic_q + slot_pos
            if self.semantic_output_gate_normalized:
                gate_source = F.layer_norm(gate_source, (gate_source.shape[-1],))
                gate = 1.0 + 0.5 * torch.tanh(gate_source)
            else:
                gate = 1.0 + torch.tanh(gate_source)
            h = h * gate
        return self.pred(h), attn                                              # [B,R,cached_dim], last [B,R,N]


class _JointMupRole(nn.Module):
    """Adapts a RopeGlobalPredictor3D's MUP role to the MaskedUtterancePredictor call signature.

    `core` is either the scale's JEPA predictor itself ("shared") or a separate instance ("split").
    """

    def __init__(self, core):
        super().__init__(); self.core = core; self.spatial = "none"

    def forward(self, utt_emb, img, mask, geom=None, routing_emb=None):
        return self.core.mup_forward(img, utt_emb, mask, routing_emb)


def _masked_utterance_loss(pred: torch.Tensor, target: torch.Tensor,
                           predict: torch.Tensor, binding: str = "index") -> torch.Tensor:
    """1 - cos(pred, stopgrad(target)) over masked valid utterances (predict = mask & valid).

    binding="index": slot j is scored against utterance j.
    binding="matched": per volume, score the masked set under its optimal assignment, found by
    exhaustive search over all k! permutations (k <= max_regions). The assignment index is detached.
    """
    m = predict.to(pred.dtype)
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    if binding == "index":
        cos = 1.0 - (p * t).sum(dim=-1)                                      # [B,R]
        return (cos * m).sum() / m.sum().clamp(min=1.0)
    if binding != "matched":
        raise ValueError(f"unknown mup_binding {binding!r}")
    total = pred.new_zeros(())
    n = 0
    for b in range(pred.shape[0]):
        idx = predict[b].nonzero(as_tuple=True)[0]
        k = int(idx.numel())
        if k == 0:
            continue
        cost = 1.0 - p[b, idx] @ t[b, idx].T                                 # [k,k]
        perms = _mup_perms(k, cost.device)                                   # [k!,k]
        # cost of each permutation = sum_j cost[j, perm[j]]
        per_perm = cost[torch.arange(k, device=cost.device).unsqueeze(0), perms].sum(-1)
        best = perms[per_perm.argmin()]                                      # detached index path
        total = total + cost[torch.arange(k, device=cost.device), best].sum()
        n += k
    return total / max(n, 1)


def _cir_loss(pred: torch.Tensor, t: torch.Tensor, w: torch.Tensor,
              predict: torch.Tensor) -> torch.Tensor:
    """Clinical-innovation regression: cosine loss on mean-centered targets, weighted by residual norm.

    ``t`` [B,R,D]: normalized residual directions (x - mu) / ||x - mu|| (stop-grad targets).
    ``w`` [B,R]: residual norms, so targets close to the corpus mean get less gradient.
    """
    y = F.normalize(pred, dim=-1)
    per = 1.0 - (y * t.detach()).sum(dim=-1)                                # [B,R]
    wm = w.detach() * predict.to(pred.dtype)
    return (per * wm).sum() / wm.sum().clamp(min=1e-6)


def _mup_image_ranking_loss(pred: torch.Tensor, wrong_pred: torch.Tensor,
                            target: torch.Tensor, target_weight: torch.Tensor,
                            predict: torch.Tensor, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Force the corresponding image to beat a deranged image for the same clinical target.

    The target/query stay fixed while only the MUP image K/V tensor is deranged across the batch.
    A query-only or batch-shared solution therefore has zero score gap and pays the full margin.
    Returns ``(margin_loss, weighted_positive_minus_negative_gap)``.
    """
    y_pos = F.normalize(pred, dim=-1)
    y_neg = F.normalize(wrong_pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    gap = (y_pos * t).sum(dim=-1) - (y_neg * t).sum(dim=-1)
    wm = target_weight.detach() * predict.to(pred.dtype)
    denom = wm.sum().clamp(min=1e-6)
    loss = (F.relu(float(margin) - gap) * wm).sum() / denom
    return loss, (gap.detach() * wm).sum() / denom


def _conditional_same_slot_nce(pred: torch.Tensor, target: torch.Tensor,
                               target_weight: torch.Tensor, predict: torch.Tensor,
                               temperature: float) -> torch.Tensor:
    """Match each study to its own target among valid targets for the same canonical slot."""
    y = F.normalize(pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    total = pred.sum() * 0.0
    denom = pred.new_zeros(())
    for slot in range(pred.shape[1]):
        idx = predict[:, slot].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        logits = y[idx, slot] @ t[idx, slot].T / float(temperature)
        labels = torch.arange(idx.numel(), device=pred.device)
        per = F.cross_entropy(logits, labels, reduction="none")
        weight = target_weight[idx, slot].detach().to(per.dtype)
        total = total + (per * weight).sum()
        denom = denom + weight.sum()
    return total / denom.clamp(min=1e-6)


def _within_study_section_nce(pred: torch.Tensor, target: torch.Tensor,
                              target_weight: torch.Tensor, predict: torch.Tensor,
                              temperature: float) -> torch.Tensor:
    """Match every masked prediction to its section target within the same study.

    Conditional NCE distinguishes patients while holding section fixed. This orthogonal
    term distinguishes sections while holding patient fixed, preventing one image-derived
    study vector from being repeated across every canonical section slot.
    """
    y = F.normalize(pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    total = pred.sum() * 0.0
    denom = pred.new_zeros(())
    for batch_index in range(pred.shape[0]):
        indices = predict[batch_index].nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        logits = y[batch_index, indices] @ t[batch_index, indices].T / float(temperature)
        labels = torch.arange(indices.numel(), device=pred.device)
        per = F.cross_entropy(logits, labels, reduction="none")
        weight = target_weight[batch_index, indices].detach().to(per.dtype)
        total = total + (per * weight).sum()
        denom = denom + weight.sum()
    return total / denom.clamp(min=1e-6)


def _within_study_section_geometry(pred: torch.Tensor, target: torch.Tensor,
                                   target_weight: torch.Tensor, predict: torch.Tensor,
                                   margin: float) -> torch.Tensor:
    """Match predicted cross-section cosine geometry to the patient target geometry."""
    y = F.normalize(pred, dim=-1)
    t = F.normalize(target.detach(), dim=-1)
    total = pred.sum() * 0.0
    denom = pred.new_zeros(())
    for batch_index in range(pred.shape[0]):
        indices = predict[batch_index].nonzero(as_tuple=True)[0]
        count = int(indices.numel())
        if count < 2:
            continue
        rows, cols = torch.triu_indices(count, count, offset=1, device=pred.device)
        pred_cos = y[batch_index, indices] @ y[batch_index, indices].T
        target_cos = t[batch_index, indices] @ t[batch_index, indices].T
        per = F.relu(pred_cos[rows, cols] - target_cos[rows, cols] - float(margin))
        weights = target_weight[batch_index, indices].detach().to(per.dtype)
        pair_weight = torch.sqrt(weights[rows] * weights[cols])
        total = total + (per * pair_weight).sum()
        denom = denom + pair_weight.sum()
    return total / denom.clamp(min=1e-6)


def _image_only_jepa_prediction(predictor: nn.Module, features: torch.Tensor,
                                condition: torch.Tensor | None,
                                coords: torch.Tensor | None = None) -> torch.Tensor:
    """Run a JEPA predictor with every text-conditioning route explicitly disabled."""
    null_condition = None if condition is None else torch.zeros_like(condition)
    return predictor(features, null_condition, None, None, coords=coords)


def _merge_reconstructed_tokens(corrupted: torch.Tensor, predicted: torch.Tensor,
                                encoder_mask: torch.Tensor) -> torch.Tensor:
    """Keep visible student tokens and fill encoder-masked positions with JEPA predictions."""
    if corrupted.shape != predicted.shape:
        raise ValueError(f"reconstructed MUP shape mismatch: {corrupted.shape} vs {predicted.shape}")
    if encoder_mask.shape != corrupted.shape[:2]:
        raise ValueError(f"reconstructed MUP mask shape {encoder_mask.shape} does not match "
                         f"tokens {corrupted.shape[:2]}")
    return torch.where(encoder_mask.bool().unsqueeze(-1), predicted, corrupted)


@torch.no_grad()
def _mup_monitor(pred: torch.Tensor, x: torch.Tensor, predict: torch.Tensor,
                 mu: torch.Tensor | None) -> dict:
    """MUP monitoring scalars over the masked set of this step; empty dict when nothing is masked."""
    m = predict
    if not bool(m.any()):
        return {}
    y = F.normalize(pred, dim=-1)[m]
    xt = x[m]
    out = {"mup_pred_norm": pred[m].norm(dim=-1).mean(),
           "mup_C_raw": (y * xt).sum(-1).mean()}
    if mu is not None:
        # mu may be global [D] or conditional [B,R,D]; compare only masked slots.
        mu_m = mu[m] if mu.ndim == x.ndim else mu
        r = xt - mu_m
        w = r.norm(dim=-1, keepdim=True)
        t = r / (w + 1e-6)
        out["mup_C_innov"] = (y * t).sum(-1).mean()
        b = F.normalize(mu_m, dim=-1)
        cp = (xt * b).sum(-1)
        out["mup_G_prior"] = (((y * xt).sum(-1) - cp) / (1.0 - cp + 1e-6)).mean()
        # Tangent-space components of the mean and the residual at the current predictions.
        proj = lambda v: v - (y * v).sum(-1, keepdim=True) * y
        g_mu = proj(mu_m.expand_as(xt))
        g_r = proj(r)
        # Normalize by the sum of norms so shares sum to 1; the norm of the vector sum can approach 0.
        tot = g_mu.norm(dim=-1) + g_r.norm(dim=-1) + 1e-9
        out["mup_gmu_share"] = (g_mu.norm(dim=-1) / tot).mean()
        out["mup_gr_share"] = (g_r.norm(dim=-1) / tot).mean()
    # collapse meter on simultaneously-masked sets (per volume, k>=2)
    sims = []
    for b_ in range(pred.shape[0]):
        idx = predict[b_].nonzero(as_tuple=True)[0]
        if idx.numel() >= 2:
            yy = F.normalize(pred[b_, idx], dim=-1)
            off = ~torch.eye(idx.numel(), dtype=torch.bool, device=yy.device)
            sims.append((yy @ yy.T)[off].mean())
    if sims:
        out["mup_C_pred"] = torch.stack(sims).mean()
    # Same slot across studies: prediction similarity vs the targets' own within-slot similarity.
    pred_slot_sims = []
    target_slot_sims = []
    for slot in range(pred.shape[1]):
        idx = predict[:, slot].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        yp = F.normalize(pred[idx, slot], dim=-1)
        if mu is not None:
            mu_slot = mu[idx, slot] if mu.ndim == x.ndim else mu
            yt = F.normalize(x[idx, slot] - mu_slot, dim=-1)
        else:
            yt = F.normalize(x[idx, slot], dim=-1)
        off = ~torch.eye(idx.numel(), dtype=torch.bool, device=yp.device)
        pred_slot_sims.append((yp @ yp.T)[off].mean())
        target_slot_sims.append((yt @ yt.T)[off].mean())
    if pred_slot_sims:
        out["mup_C_pred_within_slot"] = torch.stack(pred_slot_sims).mean()
        out["mup_C_target_within_slot"] = torch.stack(target_slot_sims).mean()
    return {k: float(v) for k, v in out.items()}


_MUP_PERM_CACHE: dict = {}


def _mup_perms(k: int, device) -> torch.Tensor:
    """All permutations of range(k) as a [k!, k] LongTensor, cached per (k, device). k <= 6 -> 720 rows."""
    key = (k, str(device))
    if key not in _MUP_PERM_CACHE:
        import itertools
        _MUP_PERM_CACHE[key] = torch.tensor(list(itertools.permutations(range(k))),
                                            dtype=torch.long, device=device)
    return _MUP_PERM_CACHE[key]


def _weak_grounding_loss(logits: torch.Tensor, target: torch.Tensor,
                         valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """0.5 * BCE + 0.5 * soft Dice between per-utterance mask logits and binary region targets.

    logits/target: [B, R, N]; valid: [B, R] bool. Dice offsets the foreground/background imbalance.
    Averaged over valid utterances only."""
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
    """Upsample per-utterance region masks [B,R,N_from] -> [B,R,N_to] (nearest)."""
    if tuple(from_grid) == tuple(to_grid):
        return masks
    B, R, N = masks.shape
    up = upsample_mask_3d(masks.reshape(B * R, N), from_grid, to_grid)   # [B*R, N_to]
    return up.reshape(B, R, -1)


def _resize_occupancy(masks: torch.Tensor, src_grid, dst_grid) -> torch.Tensor:
    """Resize per-utterance region masks [B,R,N_src] -> soft occupancy [B,R,N_dst] in [0, 1].

    Area resampling gives the fraction of each cell covered by the region."""
    if tuple(src_grid) == tuple(dst_grid):
        return masks.to(torch.float32)
    B, R, N = masks.shape
    x = masks.reshape(B * R, 1, *src_grid).float()
    y = F.interpolate(x, size=tuple(dst_grid), mode="area")             # averages on downsample
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
    """Per-sample [B] pooled-region JEPA residual."""
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
    """Pooled-region JEPA residual averaged over samples with a valid anatomy mask.

    Returns None if no sample in the batch has a valid anatomy mask."""
    per = _region_residual_per_sample(g_pred, g_target, mode)          # [B]
    if valid is None:
        return per.mean()
    vf = valid.to(per.device, per.dtype)
    denom = vf.sum()
    if float(denom) == 0.0:
        return None
    return (per * vf).sum() / denom


class JECTModel(nn.Module):
    """Anatomy-report-text guided dense JEPA model for 3D chest CT."""

    def __init__(self, config: JECTConfig):
        super().__init__()
        self.config = config
        mc = config.model
        tc = config.text
        ac = config.anatomy
        rc = config.rope

        self.tokenizer = HashingTextTokenizer(vocab_size=tc.vocab_size, max_tokens=tc.max_tokens)
        # The hashing text encoder is unused with cached embeddings.
        self.text_encoder = None if tc.use_cached_embeddings else HashingTextEncoder(
            vocab_size=tc.vocab_size,
            text_dim=tc.text_dim,
            hidden_dim=tc.text_hidden_dim,
            dropout=tc.dropout,
        )
        # Projects cached utterance embeddings (cached_dim) -> text_dim.
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
        self.cond_s4 = ScaleConditioner(tc.text_dim, cond_dim)   # global scale

        self.pred_s1 = self._make_predictor("s1", mc.dim_s1, mc.heads[0], cond_dim)
        self.pred_s2 = self._make_predictor("s2", mc.dim_s2, mc.heads[1], cond_dim)
        self.pred_s3 = self._make_predictor("s3", mc.dim_s3, mc.heads[2], cond_dim)
        # s4 predictor operates on encoder stage e4.
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

        # Region-text projection from the s3 region token.
        self.region_proj = nn.Sequential(nn.LayerNorm(mc.dim_s3), nn.Linear(mc.dim_s3, tc.text_dim), nn.LayerNorm(tc.text_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(tc.text_dim), nn.Linear(tc.text_dim, tc.text_dim), nn.LayerNorm(tc.text_dim))

        if ac.use_aux_segmentation:
            self.seg_head = nn.Conv3d(mc.dim_s1, ac.num_seg_classes, kernel_size=1)
        else:
            self.seg_head = None

        # Grounding head: per-utterance queries produce per-token masks, weakly supervised by the
        # utterance's anatomy region.
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
                depths = depths * len(scales)            # broadcast a single depth
            if len(scales) != len(depths):
                raise ValueError(f"mup_scales {scales} and mup_depths {depths} must have equal length")
            dim_of = {"s1": mc.dim_s1, "s2": mc.dim_s2, "s3": mc.dim_s3, "s4": mc.dim_s4}
            # One MUP head per scale. `mup_spatial` selects the anatomy pathway; if unset,
            # `mup_cond_pos=True` maps to "additive".
            spatial = getattr(ac, "mup_spatial", None)
            if spatial is None:
                spatial = "additive" if getattr(ac, "mup_cond_pos", False) else "none"
            self.mup_spatial = spatial
            # mup_matryoshka_dim > 0: targets are the normalized prefix of the cached embeddings.
            self._mup_dim = int(getattr(ac, "mup_matryoshka_dim", 0) or 0)
            mup_cached = self._mup_dim if self._mup_dim > 0 else tc.cached_dim
            self._mup_loss_kind = str(getattr(ac, "mup_loss", "raw_cosine"))
            self._mup_target_centering = str(getattr(ac, "mup_target_centering", "global"))
            self._mup_condition_key = str(getattr(ac, "mup_condition_key", "section_name"))
            self._mup_image_binding_weight = float(getattr(ac, "mup_image_binding_weight", 0.0))
            self._mup_image_binding_margin = float(getattr(ac, "mup_image_binding_margin", 0.05))
            self._mup_conditional_nce_weight = float(
                getattr(ac, "mup_conditional_nce_weight", 0.0))
            self._mup_conditional_nce_temperature = float(
                getattr(ac, "mup_conditional_nce_temperature", 0.10))
            self._mup_section_nce_weight = float(
                getattr(ac, "mup_section_nce_weight", 0.0))
            self._mup_section_nce_temperature = float(
                getattr(ac, "mup_section_nce_temperature", 0.10))
            self._mup_section_geometry_weight = float(
                getattr(ac, "mup_section_geometry_weight", 0.0))
            self._mup_section_geometry_margin = float(
                getattr(ac, "mup_section_geometry_margin", 0.05))
            self._mup_section_mean_routing_query = bool(
                getattr(ac, "mup_section_mean_routing_query", False))
            if self._mup_loss_kind not in ("clinical_innovation", "raw_cosine"):
                raise ValueError(f"unknown mup_loss {self._mup_loss_kind!r}")
            # Target means are needed by the CIR loss and by the section-mean routing query.
            if self._mup_loss_kind == "clinical_innovation" or self._mup_section_mean_routing_query:
                if self._mup_dim <= 0:
                    raise ValueError("mup_loss='clinical_innovation' / mup_section_mean_routing_query "
                                     "require mup_matryoshka_dim > 0 (per-dim train means; prefix norm "
                                     "is nonlinear).")
                stats_path = str(getattr(ac, "mup_target_stats", "") or "")
                if not stats_path:
                    raise ValueError("mup_loss='clinical_innovation' / mup_section_mean_routing_query "
                                     "require --mup_target_stats.")
                stats = torch.load(stats_path, map_location="cpu", weights_only=False)
                if self._mup_dim not in stats["dims"]:
                    raise ValueError(f"target-stats file lacks a mean for dim {self._mup_dim} "
                                     f"(has {sorted(stats['dims'])}).")
                self.register_buffer("mup_target_mean",
                                     stats["dims"][self._mup_dim].float(), persistent=False)
                if self._mup_target_centering == "conditional":
                    cond = stats.get("conditional_dims", {}).get(self._mup_condition_key)
                    if cond is None or self._mup_dim not in cond:
                        raise ValueError("target-stats file lacks conditional means for "
                                         f"{self._mup_condition_key!r} dim {self._mup_dim}.")
                    means = cond[self._mup_dim].float()
                    if means.ndim != 2 or means.shape[1] != self._mup_dim:
                        raise ValueError("conditional target means must have shape [K, dim]; "
                                         f"got {tuple(means.shape)} for dim {self._mup_dim}.")
                    self.register_buffer("mup_conditional_means", means, persistent=False)
            if (self._mup_loss_kind == "clinical_innovation"
                    and str(getattr(ac, "mup_binding", "index")) == "matched"):
                raise ValueError("mup_loss='clinical_innovation' with mup_binding='matched' is "
                                 "not supported.")
            self.mup_heads = nn.ModuleDict({
                s: MaskedUtterancePredictor(dim_of[s], mup_cached, ac.grounding_num_heads,
                                            max_regions=ac.max_regions, depth=d, spatial=spatial,
                                            persistent_routing_query=bool(getattr(
                                                ac, "mup_persistent_routing_query", False)),
                                            semantic_output_gate=bool(getattr(
                                                ac, "mup_semantic_output_gate", False)),
                                            semantic_output_gate_normalized=bool(getattr(
                                                ac, "mup_semantic_output_gate_normalized", False)))
                for s, d in zip(scales, depths)
            })
            # joint_predictor: "shared" runs MUP through the scale's RoPE JEPA predictor; "split" uses a
            # separate instance of the same architecture. Replaces mup_heads[scale]; mup_depths is ignored.
            jp = str(getattr(ac, "joint_predictor", "off"))
            if jp not in ("off", "shared", "split"):
                raise ValueError(f"joint_predictor must be off|shared|split, got {jp!r}")
            if jp != "off":
                from ject.backbone.rope import RopeGlobalPredictor3D
                if str(spatial) != "none":
                    raise ValueError("joint_predictor needs mup_spatial='none' (sections have no location)")
                if not (bool(getattr(ac, "mup_persistent_routing_query", False)) and bool(getattr(ac, "mup_section_mean_routing_query", False))
                        and bool(getattr(ac, "mup_semantic_output_gate", False))):
                    raise ValueError("joint_predictor implements the routing-only MUP contract: needs mup_persistent_routing_query, "
                                     "mup_section_mean_routing_query and mup_semantic_output_gate")
                for scale in scales:
                    pred = getattr(self, f"pred_{scale}", None)
                    if not isinstance(pred, RopeGlobalPredictor3D) or not pred.use_xattn:
                        raise ValueError(f"joint_predictor needs the xattn RoPE predictor at {scale} (rope_scales + predictor_cond=xattn)")
                    gate_norm = bool(getattr(ac, "mup_semantic_output_gate_normalized", False))
                    if jp == "shared":
                        core = pred
                    else:
                        b0 = pred.blocks[0]
                        core = RopeGlobalPredictor3D(dim=dim_of[scale], num_heads=b0.attn.num_heads, cond_dim=1, grid_size=pred.grid_size,
                                                     depth=len(pred.blocks), mlp_ratio=b0.mlp[0].out_features / dim_of[scale],
                                                     theta=b0.attn.theta, attn_drop=b0.attn.attn_drop.p, proj_drop=b0.attn.proj_drop.p,
                                                     use_xattn=True, text_dim=tc.text_dim)
                    core.enable_mup_role(mup_cached, ac.max_regions, gate_normalized=gate_norm)
                    self.mup_heads[scale] = _JointMupRole(core)
        else:
            self.mup_heads = None

        # SIGReg is applied to a projection of the student latents (LeJEPA-style projector head) so the
        # isotropic-Gaussian constraint does not act on the raw features. proj_dim=0 -> raw latents.
        lc = config.loss
        proj_dim = getattr(lc, "sigreg_proj_dim", 0)
        if getattr(lc, "anticollapse", "sigreg") in ("sigreg", "both") and proj_dim > 0:
            _sig_scales = [("s1", mc.dim_s1), ("s2", mc.dim_s2), ("s3", mc.dim_s3)]
            # Regularize s4 too when it is an active objective.
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
        # xattn head on selected non-RoPE scales; otherwise RoPE predictor or FiLM head.
        if (getattr(tc, "predictor_cond", "film") == "xattn"
                and scale in set(getattr(tc, "predictor_xattn_scales", ()))
                and not (rc.use_rope_predictor and scale in set(rc.rope_scales))):
            return XAttnPredictionHead(dim=dim, text_dim=tc.text_dim,
                                       num_heads=getattr(tc, "predictor_num_heads", 8))
        if rc.use_rope_predictor and scale in set(rc.rope_scales):
            # The RoPE predictor keeps FiLM and adds image-to-text cross-attention on xattn scales.
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
        # `anatomy_id` is accepted for call-site compatibility and is unused.
        null = None
        if text_emb_cached is not None:                       # cached embeddings -> text_dim
            text_emb = self.text_in_proj(text_emb_cached)
            # All-zero cached rows mean no text; mark them null since text_in_proj(0) is its bias.
            null = (text_emb_cached.abs().sum(dim=-1) == 0)
        elif token_ids is not None and self.text_encoder is not None:
            text_emb = self.text_encoder(token_ids, attention_mask)
        else:
            text_emb = torch.zeros(batch_size, self.config.text.text_dim, device=device)
        # CFG dropout and missing text both map to an exactly-zero text_emb on every text path.
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
        # Auxiliary segmentation runs at the s1 grid; full-resolution logits are too memory-heavy.
        # The loss downsamples the target to match.
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
                text_clean_step: bool = True,
                spacing_mm: float | torch.Tensor | None = None,
                wg_region_masks_s3: torch.Tensor | None = None,
                wg_region_valid: torch.Tensor | None = None,
                wg_region_text_embs: torch.Tensor | None = None,
                ctx_drop_override: torch.Tensor | None = None,
                ctx_empty_context: bool = False) -> Dict[str, torch.Tensor]:
        # ctx_drop_override: fixed [B,R] dropped-section set for synced dropout (analysis only).
        # ctx_empty_context: hide every section from the predictor while still predicting the dropped set.
        B = volume.shape[0]
        device = volume.device

        # mm_rope: physical coordinates in mm per predictor scale (stride = spacing * patch * 2^(stage-1)).
        # None in learned positional mode.
        pred_coords: dict[str, torch.Tensor | None] = {"s1": None, "s2": None, "s3": None, "s4": None}
        if getattr(self.config.model, "pos_mode", "learned") == "mm_rope":
            if spacing_mm is None:
                raise ValueError("pos_mode='mm_rope' requires spacing_mm (mm per voxel); physical "
                                 "coordinates are the positional encoding.")
            from ject.backbone.rope import make_mm_coords, run_spacing_tensor
            base = run_spacing_tensor(spacing_mm, device=device) * \
                torch.tensor(self.config.patch.patch_size, dtype=torch.float32, device=device)
            for s, mult in (("s1", 1.0), ("s2", 2.0), ("s3", 4.0), ("s4", 8.0)):
                pred_coords[s] = make_mm_coords(getattr(self.config.patch, f"grid_size_{s}"),
                                                base * mult, device=device)

        # Classifier-free-guidance text dropout (training only): null the predictor conditioning for a
        # random subset of samples. Supervision targets are unaffected.
        text_drop = None
        tdp = getattr(self.config.text, "text_drop_prob", 0.0)
        if self.training and tdp > 0:
            text_drop = torch.rand(B, device=device) < tdp
            if text_emb_cached is not None:            # null the chosen-utterance token
                text_emb_cached = text_emb_cached * (~text_drop).view(B, 1).to(text_emb_cached.dtype)

        c1, c2, c3, cond_aux = self.encode_condition(
            token_ids=token_ids, attention_mask=attention_mask, anatomy_id=anatomy_id,
            batch_size=B, device=device, text_emb_cached=text_emb_cached, text_drop=text_drop,
        )
        # Synced section dropout (mup_ctx_drop_prob > 0): the JEPA predictor is conditioned on the kept
        # sections and MUP predicts the dropped ones. The vector condition and chosen-utterance token
        # are nulled since they may come from a dropped section.
        ctx_drop = None
        ctx_p = float(getattr(self.config.anatomy, "mup_ctx_drop_prob", 0.0))
        if ctx_p > 0 and region_text_embs is not None:
            rv_ctx = (region_valid if region_valid is not None
                      else (region_masks_s3.to(torch.bool).any(dim=-1) if region_masks_s3 is not None
                            else torch.ones(B, region_text_embs.shape[1], dtype=torch.bool, device=device)))
            if ctx_drop_override is not None:
                ctx_drop = ctx_drop_override.to(device=device, dtype=torch.bool) & rv_ctx.to(torch.bool)
            else:
                ctx_drop = (torch.rand(B, region_text_embs.shape[1], device=device) < ctx_p) & rv_ctx.to(torch.bool)
            c1, c2, c3 = torch.zeros_like(c1), torch.zeros_like(c2), torch.zeros_like(c3)

        with torch.no_grad():
            t_enc = self.target_encoder(volume, window_id=window_id, mask_s1=None,
                                        ct_type_id=ct_type_id, spacing_mm=spacing_mm)
            t_dec = self.target_decoder(t_enc)

        s_enc = self.student_encoder(volume, window_id=window_id, mask_s1=masks.encoder_mask_s1,
                                     ct_type_id=ct_type_id, spacing_mm=spacing_mm)
        s_dec = self.student_decoder(s_enc)

        # Feature source for the text heads (weak grounding, MUP), set by text_heads_pass:
        #   "clean": a second, unmasked student forward.
        #   "probe": the detached EMA target; the heads get no encoder gradient.
        #   "corrupted": the masked student stream.
        #   "reconstructed": the masked student stream with masked positions filled by image-only
        #                    JEPA predictions; no second encoder forward.
        #   "stochastic": like "clean", but the text heads run only when text_clean_step is True.
        text_pass = getattr(self.config.anatomy, "text_heads_pass", "clean")
        text_heads_active = (
            region_text_embs is not None and region_masks_s3 is not None
            and ((self.grounding_head is not None and self.config.ject_loss.lambda_weakground > 0)
                 or (self.mup_heads is not None and self.config.ject_loss.lambda_mup > 0))
        )
        # "corrupted" and "reconstructed" also skip the text heads when text_clean_step is False.
        run_text = text_heads_active and not (text_pass in ("stochastic", "corrupted", "reconstructed")
                                              and not text_clean_step)
        if not run_text:
            sf_enc, sf_dec = s_enc, s_dec                      # unused: text heads skipped
        elif text_pass in ("corrupted", "reconstructed"):
            # Text heads read the masked student stream; no extra encoder pass.
            sf_enc, sf_dec = s_enc, s_dec
        elif text_pass == "probe":
            sf_enc, sf_dec = t_enc, t_dec                      # detached EMA target, no encoder gradient
        else:                                                  # "clean" or ("stochastic" & clean step)
            sf_enc = self.student_encoder(volume, window_id=window_id, mask_s1=None, ct_type_id=ct_type_id,
                                          spacing_mm=spacing_mm)
            sf_dec = self.student_decoder(sf_enc)

        # Text tokens for xattn predictors: the chosen utterance (whose region the JEPA mask targets)
        # prepended to the region set, which may not contain it.
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
            if ctx_drop is not None:                                             # synced dropout: kept sections only
                if text_emb_cached is not None:
                    msks[0] = torch.zeros_like(msks[0])                          # chosen utterance hidden
                msks[-1] = torch.zeros_like(msks[-1]) if ctx_empty_context else (msks[-1] & ~ctx_drop)
            if toks:
                pred_txt = self.pred_text_proj(torch.cat(toks, dim=1))            # [B, 1+R, text_dim]
                pred_txt_mask = torch.cat(msks, dim=1)
                if text_drop is not None:                                        # CFG dropout
                    pred_txt_mask = pred_txt_mask & (~text_drop).view(B, 1)
        p1 = self.pred_s1(s_dec["s1"], c1, pred_txt, pred_txt_mask, coords=pred_coords["s1"])
        p2 = self.pred_s2(s_dec["s2"], c2, pred_txt, pred_txt_mask, coords=pred_coords["s2"])
        p3 = self.pred_s3(s_dec["s3"], c3, pred_txt, pred_txt_mask, coords=pred_coords["s3"])
        # Image-only predictions (all text conditioning nulled) for dense supervision and reconstructed
        # MUP, so MUP targets cannot leak through the JEPA predictor.
        reconstruct_text = run_text and text_pass == "reconstructed"
        reconstruct_scales = set(getattr(self.config.anatomy, "mup_scales", ())) if reconstruct_text else set()
        if (reconstruct_text and self.grounding_head is not None
                and self.config.ject_loss.lambda_weakground > 0):
            reconstruct_scales.add(self.grounding_scale)

        # The dense (visible-token) loss uses null-text predictions: its target is phrase-independent,
        # and supervising text-conditioned predictions there would teach the predictor to ignore text.
        dense_null = {}
        image_only_pred = {}
        dense_scales = set()
        if getattr(self.config.loss, "use_dense_pred", False):
            dense_scales.update(getattr(self.config.loss, "dense_pred_scales", ()) or (
                getattr(self.config.loss, "dense_pred_scale", "s2"),))
        # Synced dropout fills with p1..p4 instead, so only the dense scales need null-text calls there.
        null_scales = dense_scales | ({s for s in reconstruct_scales if s != "s4"} if ctx_drop is None else set())
        _dpred = {"s1": (self.pred_s1, s_dec.get("s1"), c1),
                  "s2": (self.pred_s2, s_dec.get("s2"), c2),
                  "s3": (self.pred_s3, s_dec.get("s3"), c3)}
        for ds in sorted(null_scales):
            if ds not in _dpred or _dpred[ds][1] is None:
                raise RuntimeError(f"cannot reconstruct unavailable JEPA scale {ds!r}")
            head, feat, cnd = _dpred[ds]
            image_only_pred[ds] = _image_only_jepa_prediction(
                head, feat, cnd, coords=pred_coords[ds])
            if ds in dense_scales:
                dense_null[f"dense_pred_{ds}"] = image_only_pred[ds]
        # s4 JEPA: predict target encoder e4 from the masked student e4. Skipped when unused.
        p4 = None
        need_s4_jepa = getattr(self.config.loss, "lambda_s4", 0.0) > 0
        if need_s4_jepa or "s4" in reconstruct_scales:
            c4 = self.cond_s4(cond_aux["raw_text_emb"])
            if ctx_drop is not None:
                c4 = torch.zeros_like(c4)
            if need_s4_jepa or (ctx_drop is not None and "s4" in reconstruct_scales):
                p4 = self.pred_s4(s_enc["e4"], c4, pred_txt, pred_txt_mask, coords=pred_coords["s4"])
            if "s4" in reconstruct_scales and ctx_drop is None:
                image_only_pred["s4"] = _image_only_jepa_prediction(
                    self.pred_s4, s_enc["e4"], c4, coords=pred_coords["s4"])

        reconstructed_features = {}
        if reconstruct_text:
            # Fill with kept-section-conditioned predictions under synced dropout, else image-only ones.
            fill = image_only_pred if ctx_drop is None else {"s1": p1, "s2": p2, "s3": p3, "s4": p4}
            for scale in reconstruct_scales:
                if scale == "s4":
                    reconstructed_features[scale] = fill[scale]
                else:
                    reconstructed_features[scale] = _merge_reconstructed_tokens(
                        s_dec[scale], fill[scale],
                        getattr(masks, f"encoder_mask_{scale}"))

        region_token = masked_average_pool(s_dec["s3"], anatomy_mask_s3)
        region_emb = self.region_proj(region_token)

        # Anatomy-pooled JEPA residual: pooled prediction vs the EMA target pooled over the same region.
        region_jepa = None
        if anatomy_mask_s3 is not None and self.config.ject_loss.lambda_region_jepa > 0:
            g_pred = masked_average_pool(p3, anatomy_mask_s3)
            g_target = masked_average_pool(t_dec["s3"], anatomy_mask_s3)
            region_jepa = _region_residual(g_pred, g_target, self.config.ject_loss.region_jepa_loss,
                                           valid=anatomy_mask_valid)

        # Grounding head at self.grounding_scale: per-utterance mask logits [B,R,N] plus a null-query
        # prior. s3 region-mask targets are upsampled to the grounding scale.
        grounding_logits = grounding_null = grounding_attn = grounding_weak = None
        # Use the dedicated weak-grounding slot set (wg_*) when provided, else the MUP region slots.
        wg_embs = wg_region_text_embs if wg_region_text_embs is not None else region_text_embs
        wg_masks = wg_region_masks_s3 if wg_region_text_embs is not None else region_masks_s3
        wg_valid = wg_region_valid if wg_region_text_embs is not None else region_valid
        if self.grounding_head is not None and wg_embs is not None and run_text:
            gs = self.grounding_scale
            # The grounding scale is reconstructed only when lambda_weakground > 0; fall back otherwise.
            gimg = (reconstructed_features.get(gs, sf_dec[gs]) if reconstruct_text else sf_dec[gs])
            gmasks = (wg_masks if (wg_masks is None or gs == "s3")
                      else _region_masks_at(wg_masks, self.config.patch.grid_size_s3,
                                            getattr(self.config.patch, f"grid_size_{gs}")))
            grounding_logits, grounding_null, grounding_attn = self.grounding_head(
                wg_embs, gimg)                                                    # [B,R,Ngs],[B,Ngs],[B,R,Ngs]
            if gmasks is not None and self.config.ject_loss.lambda_weakground > 0:
                rv = (wg_valid if wg_valid is not None
                      else gmasks.to(torch.bool).any(dim=-1))
                tgt = gmasks.to(grounding_logits.dtype)
                wl = _weak_grounding_loss(grounding_logits, tgt, rv)
                # Null query targets the union of referenced regions, only on samples with a valid region.
                has_region = rv.any(dim=1)                                        # [B]
                union = (tgt * rv.unsqueeze(-1).to(tgt.dtype)).sum(dim=1).clamp(max=1.0)  # [B,N3]
                wl_null = _weak_grounding_loss(grounding_null.unsqueeze(1), union.unsqueeze(1),
                                               has_region.unsqueeze(1))
                grounding_weak = wl + 0.5 * wl_null

        # Masked-utterance prediction: mask a random subset of valid utterances and predict them from image.
        mup_loss = None
        mup_scale_losses: dict[str, torch.Tensor] = {}
        mup_metrics: dict = {}
        if (self.mup_heads is not None and region_text_embs is not None and region_masks_s3 is not None
                and self.config.ject_loss.lambda_mup > 0 and run_text):
            rv = (region_valid if region_valid is not None
                  else region_masks_s3.to(torch.bool).any(dim=-1))
            if getattr(self.config.anatomy, "mup_single_target", False):
                # Exactly one masked target per sample, uniform over valid slots; none if no slot is valid.
                scores = torch.rand(region_text_embs.shape[:2], device=device).masked_fill(~rv, -1.0)
                predict = F.one_hot(scores.argmax(dim=1), num_classes=rv.shape[1]).bool() & rv
            elif ctx_drop is not None:
                predict = ctx_drop & rv                        # synced dropout: predict the dropped sections
            else:
                p = float(getattr(self.config.anatomy, "mup_mask_prob", 0.5))
                predict = (torch.rand(region_text_embs.shape[:2], device=device) < p) & rv  # [B,R] mask & valid
            if bool(predict.any()):
                # Matryoshka prefix targets, shared by heads and loss.
                if self._mup_dim > 0:
                    x_t = F.normalize(region_text_embs[..., :self._mup_dim], dim=-1)
                else:
                    x_t = region_text_embs
                cir_t = cir_w = cir_mu = None
                # Target mean (global or per section), used by CIR centering and the routing query.
                if self._mup_loss_kind == "clinical_innovation" or self._mup_section_mean_routing_query:
                    cir_mu = self.mup_target_mean.to(x_t.dtype)
                    if getattr(self, "_mup_target_centering", "global") == "conditional":
                        if self._mup_condition_key != "section_name":
                            raise RuntimeError(f"unsupported MUP condition key {self._mup_condition_key!r}")
                        if region_anatomy_ids is None:
                            raise RuntimeError("conditional section-name MUP centering requires region_anatomy_ids.")
                        means = self.mup_conditional_means.to(x_t.dtype)
                        idx = region_anatomy_ids.to(device=device, dtype=torch.long) - 100
                        bad = (idx < 0) | (idx >= means.shape[0])
                        slot_mu = means[idx.clamp(0, means.shape[0] - 1)]
                        global_mu = self.mup_target_mean.to(x_t.dtype).view(1, 1, -1)
                        cir_mu = torch.where(bad.unsqueeze(-1), global_mu, slot_mu)
                    # CIR centering is scaled by mup_centering_scale (0 = raw cosine, 1 = full centering),
                    # set by the trainer. The routing query always uses the full mean.
                if self._mup_loss_kind == "clinical_innovation":
                    _cscale = float(getattr(self, "mup_centering_scale", 1.0))
                    r_t = x_t - _cscale * cir_mu
                    cir_w = r_t.norm(dim=-1)
                    cir_t = r_t / (cir_w.unsqueeze(-1) + 1e-6)
                w = 1.0 / len(self.mup_heads)                                    # split lambda_mup across scales
                # Per-scale loss weights: report semantics are pushed mostly at coarse scales.
                mup_gamma = getattr(self.config.anatomy, "mup_stage_gamma", None) \
                    or {"s1": 0.0, "s2": 0.25, "s3": 0.5, "s4": 1.0}
                mup_loss = region_text_embs.new_zeros(())
                conditional_nce_loss = None
                if self._mup_conditional_nce_weight > 0.0:
                    conditional_nce_loss = region_text_embs.new_zeros(())
                section_nce_loss = None
                if self._mup_section_nce_weight > 0.0:
                    section_nce_loss = region_text_embs.new_zeros(())
                section_geometry_loss = None
                if self._mup_section_geometry_weight > 0.0:
                    section_geometry_loss = region_text_embs.new_zeros(())
                binding_loss = binding_gap = None
                bind_enabled = self.training and self._mup_image_binding_weight > 0.0 and B > 1
                if bind_enabled:
                    binding_loss = region_text_embs.new_zeros(())
                    binding_gap = region_text_embs.new_zeros(())
                    bind_perm = torch.roll(torch.arange(B, device=device), shifts=1)
                # Region geometry in normalized [-1, 1] coords: centroid and per-axis extent from s3 masks;
                # occupancy and key coords are computed per scale.
                need_geom = any(h.spatial != "none" for h in self.mup_heads.values())
                grid_s3 = self.config.patch.grid_size_s3
                centroid = sigma = None
                if need_geom:
                    coords3 = make_3d_coords(grid_s3, device=device, dtype=torch.float32,
                                             normalize=True)                     # [N3,3] in [-1,1]
                    m = region_masks_s3.to(coords3.dtype)                         # [B,R,N3]
                    msum = m.sum(-1, keepdim=True).clamp(min=1.0)                 # [B,R,1]
                    centroid = torch.einsum("brn,nc->brc", m, coords3) / msum     # [B,R,3]
                    # per-axis extent sigma = sqrt(E[(x-c)^2]) over the region mask
                    d2 = (coords3.view(1, 1, -1, 3) - centroid.unsqueeze(2)) ** 2  # [B,R,N3,3]
                    sigma = torch.sqrt((m.unsqueeze(-1) * d2).sum(2) / msum)      # [B,R,3]
                for scale, head in self.mup_heads.items():
                    # KV source follows text_heads_pass.
                    if text_pass == "reconstructed":
                        gimg_s = reconstructed_features[scale]
                    else:
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
                    routing_emb = cir_mu if self._mup_section_mean_routing_query else None
                    if routing_emb is not None and routing_emb.ndim != 3:
                        raise RuntimeError("semantic section routing requires per-slot conditional means")
                    mup_pred, _ = head(
                        x_t, gimg_s, predict, geom=geom, routing_emb=routing_emb)
                    g = float(mup_gamma.get(scale, 1.0))
                    if self._mup_loss_kind == "clinical_innovation":
                        scale_loss = _cir_loss(mup_pred, cir_t, cir_w, predict)
                        bind_target, bind_target_weight = cir_t, cir_w
                    else:
                        scale_loss = _masked_utterance_loss(
                            mup_pred, x_t, predict,
                            binding=getattr(self.config.anatomy, "mup_binding", "index"))
                        bind_target = x_t
                        bind_target_weight = torch.ones_like(predict, dtype=x_t.dtype)
                    mup_scale_losses[scale] = scale_loss
                    mup_loss = mup_loss + w * g * scale_loss
                    if conditional_nce_loss is not None:
                        conditional_nce_loss = conditional_nce_loss + w * g * _conditional_same_slot_nce(
                            mup_pred, bind_target, bind_target_weight, predict,
                            self._mup_conditional_nce_temperature)
                    if section_nce_loss is not None:
                        section_nce_loss = section_nce_loss + w * g * _within_study_section_nce(
                            mup_pred, bind_target, bind_target_weight, predict,
                            self._mup_section_nce_temperature)
                    if section_geometry_loss is not None:
                        section_geometry_loss = section_geometry_loss + w * g * \
                            _within_study_section_geometry(
                                mup_pred, bind_target, bind_target_weight, predict,
                                self._mup_section_geometry_margin)
                    if bind_enabled:
                        wrong_pred, _ = head(
                            x_t, gimg_s.index_select(0, bind_perm), predict, geom=geom,
                            routing_emb=routing_emb)
                        rank_loss, rank_gap = _mup_image_ranking_loss(
                            mup_pred, wrong_pred, bind_target, bind_target_weight, predict,
                            self._mup_image_binding_margin)
                        binding_loss = binding_loss + w * g * rank_loss
                        binding_gap = binding_gap + w * g * rank_gap
                mup_metrics = _mup_monitor(
                    mup_pred, x_t, predict,
                    cir_mu if self._mup_loss_kind == "clinical_innovation" else None)
                if binding_gap is not None:
                    mup_metrics["mup_image_gap"] = float(binding_gap)

        out: Dict[str, torch.Tensor] = {
            "pred_s1": p1,
            "pred_s2": p2,
            "pred_s3": p3,
            **dense_null,                                    # null-text predictions for the dense loss
            # jepa_target_source: s1-s3 targets from the EMA decoder (default) or raw EMA encoder stages.
            **({"target_s1": t_dec["s1"].detach(), "target_s2": t_dec["s2"].detach(), "target_s3": t_dec["s3"].detach()}
               if getattr(self.config.model, "jepa_target_source", "decoder") == "decoder" else
               {"target_s1": t_enc["e1"].detach(), "target_s2": t_enc["e2"].detach(), "target_s3": t_enc["e3"].detach()}),
            "loss_mask_s1": masks.loss_mask_s1,
            "loss_mask_s2": masks.loss_mask_s2,
            "loss_mask_s3": masks.loss_mask_s3,
            "encoder_mask_s1": masks.encoder_mask_s1,
            "student_s1": s_dec["s1"],
            "student_s2": s_dec["s2"],
            "student_s3": s_dec["s3"],
            "region_emb": region_emb,
            # pred_s4/target_s4/student_s4 are added below when s4 is active.
            "text_emb": cond_aux["text_emb"],
            "raw_text_emb": cond_aux["raw_text_emb"],
        }
        if ctx_drop is not None:
            out["mup_ctx_drop"] = ctx_drop                     # [B,R] dropped (MUP-predicted) sections
        if region_jepa is not None:
            out["region_jepa"] = region_jepa
        if grounding_logits is not None:
            out["grounding_map"] = grounding_logits       # [B,R,N3] per-utterance mask logits
            out["grounding_null"] = grounding_null        # [B,N3] text-agnostic prior (CFG null)
            out["grounding_attn"] = grounding_attn        # [B,R,N3] query-to-image attention
        if grounding_weak is not None:
            out["weakground"] = grounding_weak
        if mup_loss is not None:
            out["mup"] = mup_loss
            for scale, scale_loss in mup_scale_losses.items():
                out[f"mup_scale_{scale}"] = scale_loss
            if conditional_nce_loss is not None:
                out["mup_conditional_nce"] = conditional_nce_loss
            if section_nce_loss is not None:
                out["mup_section_nce"] = section_nce_loss
            if section_geometry_loss is not None:
                out["mup_section_geometry"] = section_geometry_loss
            if binding_loss is not None:
                out["mup_image_binding"] = binding_loss
            if mup_metrics:
                out["mup_metrics"] = mup_metrics
        # SIGReg projector outputs. Tokens are subsampled to sigreg_max_tokens before projection.
        # student_s4 is the student encoder e4, emitted so anti-collapse terms cover s4.
        s4_active = getattr(self.config.loss, "lambda_s4", 0.0) > 0 and "e4" in s_enc
        if s4_active:
            out["student_s4"] = s_enc["e4"]
            out["pred_s4"] = p4
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

        # Intra-volume region-utterance contrastive loss: pool student s3 per region and match to the
        # region's utterance embedding within the volume.
        if region_masks_s3 is not None and (region_token_ids is not None or region_text_embs is not None):
            R = region_masks_s3.shape[1]
            reg_img = self.region_proj(pool_region_embeddings(s_dec["s3"], region_masks_s3))  # [B,R,text_dim]
            if region_text_embs is not None:                                                  # cached embeddings
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
                temperature=self.config.ject_loss.contrastive_temperature,
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
