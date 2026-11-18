"""3D RoPE modules.

RoPE is used in the global latent predictor, not in Swin window attention, which
keeps Swin's relative position bias.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Triple = Tuple[int, int, int]


def run_spacing_tensor(spacing_mm, device=None) -> torch.Tensor:
    """Validate a batch-level voxel spacing and return it as a float32 (z,y,x) triple.

    Accepts a scalar or one 3-vector; per-sample [B, 3] spacing raises ValueError."""
    s = torch.as_tensor(spacing_mm, device=device, dtype=torch.float32)
    if s.ndim > 1 or (s.ndim == 1 and s.numel() not in (1, 3)):
        raise ValueError(f"spacing_mm must be a scalar or one (z,y,x) triple shared by the whole "
                         f"batch (got shape {tuple(s.shape)}); per-sample spacing needs "
                         f"geometry-bucketed batches.")
    return s.expand(3) if s.numel() == 1 else s.reshape(3)


def make_mm_coords(grid_size: Triple, stride_mm, device=None, dtype=torch.float32) -> torch.Tensor:
    """Token-centre coordinates in millimetres, (index + 0.5) * stride_mm per axis, as [N, 3].

    Used by pos_mode="mm_rope" so positions are physical and the same weights serve any
    spacing or grid. stride_mm is a scalar or a per-axis (z, y, x) triple. RoPE logits depend
    only on position differences, so the origin at the volume corner is arbitrary.

    Spacing is shared by the whole batch; a per-sample [B, 3] spacing raises ValueError."""
    s = torch.as_tensor(stride_mm, device=device, dtype=dtype)
    if s.ndim > 1 or (s.ndim == 1 and s.numel() not in (1, 3)):
        raise ValueError(f"stride_mm must be a scalar or one (z,y,x) triple shared by the whole "
                         f"batch (got shape {tuple(s.shape)}); per-sample spacing needs "
                         f"geometry-bucketed batches, not per-sample coords.")
    s = s.expand(3) if s.numel() == 1 else s.reshape(3)
    axes = [torch.arange(g, device=device, dtype=dtype) + 0.5 for g in grid_size]
    zz, yy, xx = torch.meshgrid(axes[0] * s[0], axes[1] * s[1], axes[2] * s[2], indexing="ij")
    return torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1)


def make_3d_coords(grid_size: Triple, device=None, dtype=torch.float32, normalize: bool = True) -> torch.Tensor:
    """Return flattened [N,3] coordinates in z,y,x order."""
    D, H, W = grid_size
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    if normalize:
        # Normalise to [-1, 1] so coordinate ranges match across input resolutions.
        z = z / max(D - 1, 1) * 2.0 - 1.0
        y = y / max(H - 1, 1) * 2.0 - 1.0
        x = x / max(W - 1, 1) * 2.0 - 1.0
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1).reshape(D * H * W, 3)


def _even_axis_dims(head_dim: int) -> tuple[int, int, int, int]:
    """Return even rotary dims for z/y/x plus unrotated remainder."""
    base = max(2, (head_dim // 3) // 2 * 2)
    z = min(base, head_dim // 2 // 2 * 2)
    y = min(base, max(0, head_dim - z) // 2 // 2 * 2)
    x = min(base, max(0, head_dim - z - y) // 2 * 2)
    # Guard for very small heads.
    while z + y + x > head_dim:
        if x >= 2:
            x -= 2
        elif y >= 2:
            y -= 2
        elif z >= 2:
            z -= 2
        else:
            break
    rem = head_dim - z - y - x
    return z, y, x, rem


def _apply_1d_rope(t: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply 1D rotary embedding to the last dimension.

    t:   [B, heads, N, dim_even]
    pos: [N]
    """
    dim = t.shape[-1]
    if dim == 0:
        return t
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")
    # Phases are computed in float32: bf16 cannot resolve high-frequency phases for positions
    # of hundreds of mm. Only the bounded cos/sin are cast back to t.dtype.
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=t.device, dtype=torch.float32) / dim)
    freqs = pos.to(device=t.device, dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = freqs.cos().to(t.dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().to(t.dtype).unsqueeze(0).unsqueeze(0)
    t1 = t[..., 0::2]
    t2 = t[..., 1::2]
    out = torch.empty_like(t)
    out[..., 0::2] = t1 * cos - t2 * sin
    out[..., 1::2] = t1 * sin + t2 * cos
    return out


def apply_3d_rope(q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor, theta: float = 10000.0):
    """Apply axial 3D RoPE to q/k.

    q/k:    [B, heads, N, head_dim]
    coords: [N,3] in z,y,x order, preferably normalized or physical coordinates.
    """
    hd = q.shape[-1]
    dz, dy, dx, rem = _even_axis_dims(hd)
    splits = [d for d in (dz, dy, dx, rem) if d > 0]
    q_parts = list(torch.split(q, splits, dim=-1))
    k_parts = list(torch.split(k, splits, dim=-1))
    out_q, out_k = [], []
    pidx = 0
    for axis, dim in enumerate((dz, dy, dx)):
        if dim <= 0:
            continue
        out_q.append(_apply_1d_rope(q_parts[pidx], coords[:, axis], theta=theta))
        out_k.append(_apply_1d_rope(k_parts[pidx], coords[:, axis], theta=theta))
        pidx += 1
    if rem > 0:
        out_q.append(q_parts[pidx])
        out_k.append(k_parts[pidx])
    return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class RopeSelfAttention3D(nn.Module):
    """Global self-attention with axial 3D RoPE over a flattened token grid."""

    def __init__(self, dim: int, num_heads: int, theta: float = 10000.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.theta = theta
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = apply_3d_rope(q, k, coords=coords, theta=self.theta)
        # SDPA avoids materialising the [B, heads, N, N] attention map; its default scale equals self.scale.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0),
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class _TextCrossAttention(nn.Module):
    """Cross-attention from image tokens (queries) to text tokens (keys/values).

    Gives spatially selective text conditioning, unlike FiLM's global scale/shift. The output
    projection is zero-initialised so the block starts as an identity residual."""

    def __init__(self, dim: int, text_dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.d = num_heads, dim // num_heads
        self.scale = self.d ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(text_dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(text_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, txt: torch.Tensor, txt_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        L = txt.shape[1]
        q = self.q(self.norm_q(x)).view(B, N, self.h, self.d).transpose(1, 2)         # [B,h,N,d]
        kv = self.kv(self.norm_kv(txt)).view(B, L, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                             # [B,h,L,d]
        logits = (q @ k.transpose(-2, -1)) * self.scale                                # [B,h,N,L]
        if txt_mask is not None:
            logits = logits.masked_fill(~txt_mask[:, None, None, :].bool(), float("-inf"))
        attn = torch.nan_to_num(logits.softmax(dim=-1))                                # all-masked rows become 0
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        if txt_mask is not None:
            # Rows with no valid text output exactly zero; otherwise proj's bias would leak a constant.
            out = out * txt_mask.bool().any(dim=1).view(B, 1, 1).to(out.dtype)
        return out


class RopeTransformerBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int,
                 mlp_ratio: float = 4.0, theta: float = 10000.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_xattn: bool = False, text_dim: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RopeSelfAttention3D(dim, num_heads, theta=theta,
                                        attn_drop=attn_drop, proj_drop=proj_drop)
        # Text conditioning is either spatial cross-attention (use_xattn) or global FiLM, not both.
        # A parallel FiLM path would let the predictor ignore the zero-initialised cross-attention.
        self.xattn = _TextCrossAttention(dim, text_dim, num_heads) if use_xattn else None
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden, dim), nn.Dropout(proj_drop),
        )
        self.cond = None if use_xattn else nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 4 * dim))
        if self.cond is not None:
            nn.init.zeros_(self.cond[-1].weight)
            nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, cond: torch.Tensor,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None) -> torch.Tensor:
        if self.cond is not None:                                                      # FiLM path (no xattn)
            g1, b1, g2, b2 = self.cond(cond).chunk(4, dim=-1)
            h = self.norm1(x) * (1.0 + g1.unsqueeze(1)) + b1.unsqueeze(1)
            x = x + self.attn(h, coords)
            h = self.norm2(x) * (1.0 + g2.unsqueeze(1)) + b2.unsqueeze(1)
            x = x + self.mlp(h)
            return x
        # xattn path: image self-attention, text cross-attention, MLP.
        x = x + self.attn(self.norm1(x), coords)
        if text_tokens is not None:
            x = x + self.xattn(x, text_tokens, text_mask)                              # image attends to words
        x = x + self.mlp(self.norm2(x))
        return x


class RopeGlobalPredictor3D(nn.Module):
    """Small global predictor used after the Swin-UNETR decoder at selected scales."""

    def __init__(self, dim: int, num_heads: int, cond_dim: int, grid_size: Triple,
                 depth: int = 1, mlp_ratio: float = 4.0, theta: float = 10000.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_xattn: bool = False, text_dim: int = 0):
        super().__init__()
        self.grid_size = grid_size
        self.use_xattn = use_xattn
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            RopeTransformerBlock3D(dim=dim, num_heads=num_heads, cond_dim=cond_dim,
                                   mlp_ratio=mlp_ratio, theta=theta,
                                   attn_drop=attn_drop, proj_drop=proj_drop,
                                   use_xattn=use_xattn, text_dim=text_dim)
            for _ in range(depth)
        ])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        coords = make_3d_coords(grid_size, device=None, dtype=torch.float32, normalize=True)
        self.register_buffer("coords", coords, persistent=False)
        self.mup_role = False                                   # set by enable_mup_role()

    def enable_mup_role(self, cached_dim: int, max_regions: int, gate_normalized: bool = True) -> None:
        """Add section-slot inputs/outputs so this predictor also predicts masked section embeddings.

        Used as a joint JEPA/MUP predictor through `mup_forward`. Without this call no parameters are added."""
        dim = self.input_norm.normalized_shape[0]
        self.q_in = nn.Linear(cached_dim, dim)                                  # section embedding to query
        self.slot_pos = nn.Parameter(torch.zeros(1, max_regions, dim)); nn.init.trunc_normal_(self.slot_pos, std=0.02)
        self.mup_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, cached_dim))
        self.mup_gate_normalized = bool(gate_normalized); self.mup_role = True
        for blk in self.blocks:
            if blk.cond is not None:
                raise ValueError("joint predictor needs the xattn-style RoPE block (FiLM path has no shared attention for slots)")

    def mup_forward(self, x_img: torch.Tensor, utt_emb: torch.Tensor, mask: torch.Tensor, routing_emb: torch.Tensor):
        """Predict masked section embeddings with the shared predictor blocks.

        R section-slot queries (routing embedding for masked slots, true content for unmasked ones,
        plus a slot position) attend to each block's image states using that block's q/k/v/proj and
        MLP weights. Slots do not attend to each other, image tokens do not attend to slots, and slots
        get no RoPE. Image tokens advance through the blocks without text.

        Args:
            x_img: [B, N, dim] image tokens.
            utt_emb, routing_emb: [B, R, cached_dim].
            mask: [B, R], True for slots to predict.
        Returns:
            ([B, R, cached_dim] predictions, None).
        """
        if not self.mup_role:
            raise RuntimeError("mup_forward on a predictor without enable_mup_role()")
        if routing_emb is None:
            raise ValueError("the joint predictor's MUP role is the routing-only mode: routing_emb is required")
        B, R, _ = utt_emb.shape; dim = self.input_norm.normalized_shape[0]
        q_content = self.q_in(utt_emb); semantic_q = self.q_in(routing_emb.detach()); slot_pos = self.slot_pos[:, :R].to(q_content.dtype)
        routing_q = torch.where(mask.unsqueeze(-1), semantic_q, q_content) + slot_pos                 # [B,R,dim], queries only
        x = self.input_norm(x_img)
        coords = self.coords.to(device=x.device, dtype=torch.float32)
        h = torch.zeros_like(routing_q)
        for blk in self.blocks:
            a = blk.attn; hd = a.head_dim; nh = a.num_heads; W, bvec = a.qkv.weight, a.qkv.bias
            kv_in = blk.norm1(x)                                                                       # same input as image self-attention
            q = F.linear(blk.norm1(h + routing_q), W[:dim], None if bvec is None else bvec[:dim]).view(B, R, nh, hd).transpose(1, 2)
            k = F.linear(kv_in, W[dim:2 * dim], None if bvec is None else bvec[dim:2 * dim]).view(B, -1, nh, hd).transpose(1, 2)
            v = F.linear(kv_in, W[2 * dim:], None if bvec is None else bvec[2 * dim:]).view(B, -1, nh, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=(a.attn_drop.p if self.training else 0.0))
            h = h + a.proj_drop(a.proj(o.transpose(1, 2).reshape(B, R, dim)))
            h = h + blk.mlp(blk.norm2(h))
            x = blk(x, coords=coords, cond=None, text_tokens=None, text_mask=None)                       # image path, no text
        gate_source = semantic_q + slot_pos
        if self.mup_gate_normalized:
            gate_source = F.layer_norm(gate_source, (gate_source.shape[-1],)); gate = 1.0 + 0.5 * torch.tanh(gate_source)
        else:
            gate = 1.0 + torch.tanh(gate_source)
        return self.mup_out(h * gate), None                                                           # [B,R,cached_dim]

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None,
                coords: torch.Tensor = None) -> torch.Tensor:
        # Blocks condition on text via FiLM (cond) or, with use_xattn, via cross-attention to text_tokens.
        # `coords` overrides the buffered normalised-grid coordinates (e.g. physical-mm positions).
        x = self.input_norm(x)
        # Coords stay float32 so bf16 autocast does not quantise physical-mm positions.
        coords = (self.coords if coords is None else coords).to(device=x.device, dtype=torch.float32)
        tt = text_tokens if self.use_xattn else None
        for blk in self.blocks:
            x = blk(x, coords=coords, cond=cond, text_tokens=tt, text_mask=text_mask)
        return self.out(x)
