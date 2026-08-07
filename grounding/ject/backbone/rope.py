"""3D RoPE modules for ART-JEPA.

The pilot uses RoPE in the global latent predictor, not in Swin window attention.
This is deliberate: it gives long-range 3D position awareness without replacing
Swin's tested relative-position-bias machinery.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Triple = Tuple[int, int, int]


def make_3d_coords(grid_size: Triple, device=None, dtype=torch.float32, normalize: bool = True) -> torch.Tensor:
    """Return flattened [N,3] coordinates in z,y,x order."""
    D, H, W = grid_size
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    if normalize:
        # Keep coordinate ranges comparable across 224->320/352 cooldown.
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
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=t.device, dtype=t.dtype) / dim)
    freqs = pos.to(device=t.device, dtype=t.dtype).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
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
        # Memory-efficient / flash SDPA instead of a materialized [B,heads,N,N] softmax. Mathematically
        # identical (default scale = head_dim**-0.5, matching self.scale); this attention returns FEATURES
        # only (no attn map is read out), so SDPA is a drop-in with no quality change — it just avoids the
        # N×N (N=2744 at s3) materialization that dominated the depth-6 predictor's backward memory.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0),
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class _TextCrossAttention(nn.Module):
    """Image tokens (queries) attend to the per-utterance text tokens (keys/values). Gives the RoPE
    predictor SPATIALLY-SELECTIVE text conditioning — each image position pulls on the relevant words —
    which FiLM (a global per-channel scale/shift) cannot. Values are text; output projection is zero-init
    so the block starts as a pure identity residual (self-attn + FiLM establish first, x-attn grows)."""

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
        logits = (q @ k.transpose(-2, -1)) * self.scale                                # [B,h,N,L] (L tiny)
        if txt_mask is not None:
            logits = logits.masked_fill(~txt_mask[:, None, None, :].bool(), float("-inf"))
        attn = torch.nan_to_num(logits.softmax(dim=-1))                                # all-masked row → 0
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        if txt_mask is not None:
            # A row with NO valid text (CFG-dropped / missing sidecar) must be TRULY null: proj(0)=bias
            # (a learned constant) would otherwise leak a spurious condition into the null path that ART-PGI
            # and the CFG residual assume is exactly zero. Zero those rows.
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
        # Text conditioning is EITHER spatial cross-attention (use_xattn) OR global FiLM — never both.
        # When xattn is on we deliberately DROP FiLM so ALL text conditioning is routed through the spatial
        # mechanism: a global FiLM path would let the predictor satisfy the JEPA loss globally and leave the
        # (zero-init) xattn dormant → no spatial phrase-selectivity (the shortcut ART-PGI must avoid, same
        # philosophy as leak-free relbias MUP). FiLM kept only for the non-xattn RoPE case.
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
        # xattn path: image self-attn → spatial text cross-attn → MLP (no FiLM).
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None,
                text_tokens: torch.Tensor = None, text_mask: torch.Tensor = None) -> torch.Tensor:
        # Conditions via FiLM (cond) always; ADDITIONALLY via image→text cross-attention when use_xattn
        # (spatially-selective text conditioning — needed for the reusable grounding predictor).
        x = self.input_norm(x)
        coords = self.coords.to(device=x.device, dtype=x.dtype)
        tt = text_tokens if self.use_xattn else None
        for blk in self.blocks:
            x = blk(x, coords=coords, cond=cond, text_tokens=tt, text_mask=text_mask)
        return self.out(x)
