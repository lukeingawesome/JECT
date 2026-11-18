"""Image backbone: dense Swin-UNETR-style encoder and latent decoder.

The student (masked input) and EMA target (full input) share this architecture:

    volume -> PatchEmbed3D (s1 grid) -> [mask tokens] -> local stage1
           -> PatchMerge -> local stage2 -> PatchMerge -> global stage3
           -> PatchMerge -> global stage4 (e4)
    encoder skips {e1, e2, e3} -> DenseLatentDecoder -> latents {s1, s2, s3}
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ject.backbone.rope import RopeSelfAttention3D, make_mm_coords, run_spacing_tensor

Triple = Tuple[int, int, int]


# ----------------------------------------------------------------------------
# Small shared pieces
# ----------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def _to_grid(x: torch.Tensor, grid: Triple) -> torch.Tensor:
    """[B, N, C] -> [B, C, D, H, W]."""
    B, N, C = x.shape
    D, H, W = grid
    return x.transpose(1, 2).reshape(B, C, D, H, W)


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, D, H, W] -> [B, N, C]."""
    return x.flatten(2).transpose(1, 2)


# ----------------------------------------------------------------------------
# Attention blocks
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    """Standard multi-head self-attention over all tokens (no positional bias)."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop_p = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # SDPA avoids fp16 score overflow and never materializes the [B, heads, N, N] matrix.
        # Its default scale equals self.scale.
        attn_mask = bias.unsqueeze(0).to(q.dtype) if bias is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock3D(nn.Module):
    """Pre-norm global transformer block (used for full-attention local / global stages)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, grid: Triple) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _pad_to_window(x: torch.Tensor, window: Triple):
    """Pad a [B, D, H, W, C] grid up to a multiple of the window on each spatial axis."""
    _, D, H, W, _ = x.shape
    wd, wh, ww = window
    pd, ph, pw = (wd - D % wd) % wd, (wh - H % wh) % wh, (ww - W % ww) % ww
    if pd or ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))   # pads W, H, D (C last, untouched)
    return x, (D, H, W)


def _win_partition(x: torch.Tensor, window: Triple) -> torch.Tensor:
    """[B, D, H, W, C] (D/H/W already multiples of window) -> [B*num_windows, ws, C]."""
    B, D, H, W, C = x.shape
    wd, wh, ww = window
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, C)


def _win_merge(windows: torch.Tensor, gridp: Triple, window: Triple, B: int) -> torch.Tensor:
    """[B*num_windows, ws, C] -> [B, Dp, Hp, Wp, C]."""
    Dp, Hp, Wp = gridp
    wd, wh, ww = window
    C = windows.shape[-1]
    x = windows.view(B, Dp // wd, Hp // wh, Wp // ww, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, Dp, Hp, Wp, C)


class WindowAttention3D(nn.Module):
    """Windowed multi-head attention with a 3D relative-position-bias table."""

    def __init__(self, dim: int, num_heads: int, window: Triple,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window = window
        wd, wh, ww = window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"))  # [3,wd,wh,ww]
        coords_flat = coords.reshape(3, -1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [3, ws, ws]
        rel = rel.permute(1, 2, 0).contiguous()                  # [ws, ws, 3]
        rel[..., 0] += wd - 1
        rel[..., 1] += wh - 1
        rel[..., 2] += ww - 1
        rel[..., 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[..., 1] *= (2 * ww - 1)
        self.register_buffer("relative_position_index", rel.sum(-1).view(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop_p = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _bias(self) -> torch.Tensor:
        ws = self.relative_position_index.numel()
        ws = int(round(ws ** 0.5))
        table = self.relative_position_bias_table[self.relative_position_index]
        return table.view(ws, ws, -1).permute(2, 0, 1).contiguous()  # [heads, ws, ws]

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        Bn, ws, C = windows.shape
        qkv = self.qkv(windows).reshape(Bn, ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Relative-position bias is passed to SDPA as an additive attn_mask.
        attn_mask = self._bias().unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(Bn, ws, C)
        return self.proj_drop(self.proj(out))


class SwinBlock3D(nn.Module):
    """Shifted-window 3D attention block with internal padding (arbitrary grid sizes).

    ``shift`` alternates across a stage (regular / shifted windows) to give cross-window
    connectivity; the grid is padded up to a window multiple before partitioning and
    cropped back after, so no divisibility constraint reaches the caller.
    """

    def __init__(self, dim: int, num_heads: int, window: Triple,
                 mlp_ratio: float = 4.0, drop: float = 0.0, shift: bool = False):
        super().__init__()
        self.window = window
        self.shift_size = tuple(w // 2 for w in window) if shift else (0, 0, 0)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, grid: Triple) -> torch.Tensor:
        B, N, C = x.shape
        D, H, W = grid
        h = self.norm1(x).view(B, D, H, W, C)
        h, _ = _pad_to_window(h, self.window)
        Dp, Hp, Wp = h.shape[1:4]
        if any(s > 0 for s in self.shift_size):
            h = torch.roll(h, shifts=tuple(-s for s in self.shift_size), dims=(1, 2, 3))
        windows = self.attn(_win_partition(h, self.window))
        h = _win_merge(windows, (Dp, Hp, Wp), self.window, B)
        if any(s > 0 for s in self.shift_size):
            h = torch.roll(h, shifts=self.shift_size, dims=(1, 2, 3))
        h = h[:, :D, :H, :W].reshape(B, N, C)                  # crop padding
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ConvNeXtBlock3D(nn.Module):
    """Depthwise 3D ConvNeXt block operating on a token grid."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor, grid: Triple) -> torch.Tensor:
        y = _to_tokens(self.dwconv(_to_grid(x, grid)))
        y = self.pw2(self.act(self.pw1(self.norm(y))))
        return x + self.gamma * y


def build_block(block_type: str, dim: int, num_heads: int, window: Triple,
                mlp_ratio: float, drop: float, shift: bool = False) -> nn.Module:
    if block_type == "swin":
        return SwinBlock3D(dim, num_heads, window, mlp_ratio, drop, shift=shift)
    if block_type in ("transformer", "global"):
        return TransformerBlock3D(dim, num_heads, mlp_ratio, drop)
    if block_type == "convnext":
        return ConvNeXtBlock3D(dim, mlp_ratio, drop)
    raise ValueError(f"Unknown block type {block_type!r}")


# ----------------------------------------------------------------------------
# Patch embed / merge
# ----------------------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, channels: int, dim: int, patch_size: Triple):
        super().__init__()
        self.proj = nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.norm(_to_tokens(self.proj(volume)))


def _pool_fine(x: torch.Tensor, grid: Triple, grid_to: Triple) -> torch.Tensor:
    """Max- and mean-pool z-major tokens on `grid` into the cells of `grid_to`.

    [B, N, C] -> [B, N_to, 2C], pooling the k^3 fine tokens of each coarse cell (k = grid / grid_to)."""
    B, N, C = x.shape
    D, H, W = grid
    k = D // grid_to[0]
    if (D, H, W) != (k * grid_to[0], k * grid_to[1], k * grid_to[2]):
        raise ValueError(f"skip_fine needs grid {grid} to be an integer multiple of the coarser grid {grid_to}")
    v = x.transpose(1, 2).reshape(B, C, D, H, W)
    mx = F.max_pool3d(v, k)
    mn = F.avg_pool3d(v, k)
    return torch.cat([mx, mn], dim=1).flatten(2).transpose(1, 2)


class LocalAttentionPool3D(nn.Module):
    """Coarse-query attention over one cell's fine tokens, with a learned 3D-offset bias.

    Alternative to max/mean skip pooling: four heads, 128 internal channels, and the same
    2C output width. Token order within each cell is z/y/x, so each bias index is a distinct
    relative 3D position. Attention never crosses cells.
    """

    def __init__(self, fine_dim: int, coarse_dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.heads, self.head_dim = 4, 32
        self.fine_norm = nn.LayerNorm(fine_dim)
        self.query_norm = nn.LayerNorm(coarse_dim)
        self.query = nn.Linear(coarse_dim, 128)
        self.key_value = nn.Linear(fine_dim, 256)
        self.relative_bias = nn.Parameter(torch.zeros(4, factor ** 3))
        self.output = nn.Linear(128, 2 * fine_dim)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor, grid: Triple,
                grid_to: Triple) -> torch.Tensor:
        B, N, C = fine.shape
        D, H, W = grid_to
        k = self.factor
        if tuple(grid) != (D * k, H * k, W * k) or N != D * H * W * k ** 3:
            raise ValueError("attention skip requires exactly nested fine/coarse grids")
        if coarse.shape[:2] != (B, D * H * W):
            raise ValueError("attention skip query count does not match its coarse grid")
        cells = fine.reshape(B, D, k, H, k, W, k, C)
        cells = cells.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * D * H * W, k ** 3, C)
        kv = self.key_value(self.fine_norm(cells))
        kv = kv.reshape(-1, k ** 3, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.query(self.query_norm(coarse)).reshape(-1, self.heads, 1, self.head_dim)
        bias = self.relative_bias.to(q.dtype)[None, :, None, :]
        # The masked SDPA kernel fails on more than 65,535 batch entries per call, so large inputs are
        # chunked. Cells attend only within themselves, so chunking does not change the result.
        n = q.shape[0]; chunk = 32768
        if n <= chunk:
            pooled = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=bias, dropout_p=0.0)
        else:
            pooled = torch.cat([F.scaled_dot_product_attention(q[i:i + chunk], kv[0][i:i + chunk], kv[1][i:i + chunk],
                                                               attn_mask=bias, dropout_p=0.0) for i in range(0, n, chunk)], dim=0)
        return self.output(pooled.reshape(B, D * H * W, 128))


class PatchMerge3D(nn.Module):
    """2x2x2 patch merging: [B, N, C] -> [B, N/8, out_dim]."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(8 * in_dim)
        self.reduce = nn.Linear(8 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, grid: Triple) -> Tuple[torch.Tensor, Triple]:
        B, N, C = x.shape
        D, H, W = grid
        if D % 2 or H % 2 or W % 2:                     # pad odd grids to even before merging
            x = x.view(B, D, H, W, C)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
            D, H, W = D + D % 2, H + H % 2, W + W % 2
            x = x.reshape(B, D * H * W, C)
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, (D // 2) * (H // 2) * (W // 2), 8 * C)
        x = self.reduce(self.norm(x))
        return x, (D // 2, H // 2, W // 2)


# ----------------------------------------------------------------------------
# Encoder / decoder
# ----------------------------------------------------------------------------
class CoordEmbed3D(nn.Module):
    """Absolute-position embedding from Fourier features of mm coordinates, followed by an MLP.

    Used with pos_mode="mm_rope" in place of the learned `pos_embed`. Absolute position carries
    anatomical signal in chest CT that relative RoPE alone would discard, and a coordinate MLP
    needs no interpolation when the grid or spacing changes. Coordinates are divided by a fixed
    `ref_mm`, so a physical location always maps to the same feature. The output layer is
    zero-initialized."""

    def __init__(self, dim: int, n_freq: int = 6, ref_mm: float = 400.0):
        super().__init__()
        self.ref_mm = ref_mm
        freqs = (2.0 ** torch.arange(n_freq)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = 3 * 2 * n_freq
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords_mm: torch.Tensor) -> torch.Tensor:
        # coords_mm [N, 3] -> [N, dim]
        x = (coords_mm.to(self.freqs.dtype) / self.ref_mm).unsqueeze(-1) * self.freqs  # [N,3,F]
        feat = torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)                        # [N,3*2F]
        return self.net(feat)


class RopeEncoderBlock3D(nn.Module):
    """Pre-norm transformer block with axial 3D RoPE, used for stage 4 with pos_mode="mm_rope".

    Position enters attention as physical-mm rotary phases instead of a grid-sized table, so the
    same weights serve any grid or spacing."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 theta: float = 10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RopeSelfAttention3D(dim, num_heads, theta=theta, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, coords_mm: torch.Tensor) -> torch.Tensor:
        # coords stay fp32; _apply_1d_rope does phase math in fp32 and casts cos/sin to x.dtype.
        x = x + self.attn(self.norm1(x), coords_mm)
        return x + self.mlp(self.norm2(x))


class DenseSwinUNETREncoder(nn.Module):
    """Hierarchical encoder producing skip features at the s1/s2/s3 grids."""

    def __init__(self, config):
        super().__init__()
        mc = config.model
        pc = config.patch
        self.grid_s1 = pc.grid_size_s1
        self.grid_s2 = pc.grid_size_s2
        self.grid_s3 = pc.grid_size_s3
        window = mc.swin_window_size

        self.patch_embed = PatchEmbed3D(config.volume.channels, mc.dim_s1, pc.patch_size)
        self.window_embed = nn.Embedding(4, mc.dim_s1)
        nn.init.zeros_(self.window_embed.weight)
        self.ct_type_embed = nn.Embedding(mc.num_ct_types, mc.dim_s1)
        nn.init.zeros_(self.ct_type_embed.weight)   # no-op at init, like window_embed
        self.mask_token = nn.Parameter(torch.zeros(1, 1, mc.dim_s1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.stage1 = nn.ModuleList([
            build_block(mc.local_block_type, mc.dim_s1, mc.heads[0], window, mc.mlp_ratio, mc.drop,
                        shift=(i % 2 == 1))
            for i in range(mc.depths[0])
        ])
        self.merge1 = PatchMerge3D(mc.dim_s1, mc.dim_s2)
        self.stage2 = nn.ModuleList([
            build_block(mc.local_block_type, mc.dim_s2, mc.heads[1], window, mc.mlp_ratio, mc.drop,
                        shift=(i % 2 == 1))
            for i in range(mc.depths[1])
        ])
        self.merge2 = PatchMerge3D(mc.dim_s2, mc.dim_s3)
        # skip_fine (default off) adds bottom-up skips of pooled finer features into each coarser stage:
        # the s2 entry gets [p0, e1], s3 gets [p0, e1, e2], s4 gets [p0, e1, e2, e3], where p0 is the patch
        # embedding. Each skip is a LayerNorm + zero-initialized Linear, so it is a no-op at init.
        self.skip_fine = bool(getattr(mc, "skip_fine", False))
        self.skip_pool_s3 = getattr(mc, "skip_pool_s3", "maxmean")
        self.skip_pool_s2s4 = getattr(mc, "skip_pool_s2s4", "maxmean")
        if self.skip_pool_s3 not in ("maxmean", "attention"):
            raise ValueError(f"unknown skip_pool_s3: {self.skip_pool_s3}")
        if self.skip_pool_s2s4 not in ("maxmean", "attention"):
            raise ValueError(f"unknown skip_pool_s2s4: {self.skip_pool_s2s4}")
        if not self.skip_fine and (self.skip_pool_s3 != "maxmean" or self.skip_pool_s2s4 != "maxmean"
                                   or getattr(mc, "skip_legacy_rng", False)):
            raise ValueError("pooling comparison requires skip_fine")
        if self.skip_fine:
            # Built under a forked RNG so the skip modules do not shift the initialization of later
            # weights; shared weights match a skip_fine=False model with the same seed.
            rng_context = nullcontext() if getattr(mc, "skip_legacy_rng", False) else torch.random.fork_rng(devices=[])
            with rng_context:
                self.skip2_norm = nn.LayerNorm(2 * (mc.dim_s1 + mc.dim_s1))
                self.skip2_proj = nn.Linear(2 * (mc.dim_s1 + mc.dim_s1), mc.dim_s2)
                self.skip3_norm = nn.LayerNorm(2 * (mc.dim_s1 + mc.dim_s1 + mc.dim_s2))
                self.skip3_proj = nn.Linear(2 * (mc.dim_s1 + mc.dim_s1 + mc.dim_s2), mc.dim_s3)
                self.skip4_norm = nn.LayerNorm(2 * (mc.dim_s1 + mc.dim_s1 + mc.dim_s2 + mc.dim_s3))
                self.skip4_proj = nn.Linear(2 * (mc.dim_s1 + mc.dim_s1 + mc.dim_s2 + mc.dim_s3), mc.dim_s4)
            for proj in (self.skip2_proj, self.skip3_proj, self.skip4_proj):
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        # pos_mode: "learned" uses per-token position tables sized to the grid; "mm_rope" uses physical-mm
        # coordinates (CoordEmbed3D at the s3 entry, RoPE in stage-4 attention) and works for any grid/spacing.
        self.pos_mode = getattr(mc, "pos_mode", "learned")
        self.patch_size = pc.patch_size
        if self.pos_mode == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, pc.num_tokens_s3, mc.dim_s3))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.coord_embed = None
        else:
            self.pos_embed = None
            self.coord_embed = CoordEmbed3D(mc.dim_s3)
        self.stage3 = nn.ModuleList([
            build_block(mc.stage3_block_type, mc.dim_s3, mc.heads[2], window, mc.mlp_ratio, mc.drop,
                        shift=(i % 2 == 1))
            for i in range(mc.depths[2])
        ])
        # s4: coarsest global scale; halves the s3 grid and maps dim_s3 to dim_s4.
        self.grid_s4 = pc.grid_size_s4
        self.merge3 = PatchMerge3D(mc.dim_s3, mc.dim_s4)
        if self.pos_mode == "learned":
            self.pos_embed_s4 = nn.Parameter(torch.zeros(1, pc.num_tokens_s4, mc.dim_s4))
            nn.init.trunc_normal_(self.pos_embed_s4, std=0.02)
            self.stage4 = nn.ModuleList([
                build_block(mc.stage4_block_type, mc.dim_s4, mc.heads[3], window, mc.mlp_ratio, mc.drop,
                            shift=(i % 2 == 1))
                for i in range(mc.depths[3])
            ])
        else:
            self.pos_embed_s4 = None
            self.stage4 = nn.ModuleList([
                RopeEncoderBlock3D(mc.dim_s4, mc.heads[3], mc.mlp_ratio, mc.drop)
                for _ in range(mc.depths[3])
            ])

        if self.skip_pool_s3 == "attention":
            # Forked RNG keeps the initialization of all other modules unchanged.
            with torch.random.fork_rng(devices=[]):
                self.skip3_attention = nn.ModuleList([
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s3, 4),
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s3, 4),
                    LocalAttentionPool3D(mc.dim_s2, mc.dim_s3, 2),
                ])
        if self.skip_pool_s2s4 == "attention":
            # Attention pooling for the s2 entry [p0, e1] (factor 2) and the s4 entry [p0, e1, e2, e3]
            # (factors 8/8/4/2). Built in its own forked RNG after the s3 modules so their initialization
            # is unchanged.
            with torch.random.fork_rng(devices=[]):
                self.skip2_attention = nn.ModuleList([
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s2, 2),
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s2, 2),
                ])
                self.skip4_attention = nn.ModuleList([
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s4, 8),
                    LocalAttentionPool3D(mc.dim_s1, mc.dim_s4, 8),
                    LocalAttentionPool3D(mc.dim_s2, mc.dim_s4, 4),
                    LocalAttentionPool3D(mc.dim_s3, mc.dim_s4, 2),
                ])

    def forward(self, volume: torch.Tensor, window_id: torch.Tensor | None = None,
                mask_s1: torch.Tensor | None = None,
                ct_type_id: torch.Tensor | None = None,
                spacing_mm: float | torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if self.pos_mode != "learned" and spacing_mm is None:
            raise ValueError("pos_mode='mm_rope' requires spacing_mm (mm per voxel); physical "
                             "coordinates are the positional encoding.")
        x = self.patch_embed(volume)                                 # [B, N1, dim_s1]
        if window_id is not None:
            x = x + self.window_embed(window_id).unsqueeze(1)
        if ct_type_id is not None:
            x = x + self.ct_type_embed(ct_type_id).unsqueeze(1)
        if mask_s1 is not None:
            x = torch.where(mask_s1.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        p0 = x if self.skip_fine else None

        for blk in self.stage1:
            x = blk(x, self.grid_s1)
        e1 = x
        x, _ = self.merge1(x, self.grid_s1)
        if self.skip_fine:
            if self.skip_pool_s2s4 == "attention":
                sources = ((p0, self.grid_s1), (e1, self.grid_s1))
                fine = torch.cat([pool(f, x, grid, self.grid_s2)
                                  for pool, (f, grid) in zip(self.skip2_attention, sources)], dim=-1)
            else:
                fine = torch.cat([_pool_fine(p0, self.grid_s1, self.grid_s2), _pool_fine(e1, self.grid_s1, self.grid_s2)], dim=-1)
            x = x + self.skip2_proj(self.skip2_norm(fine)).to(x.dtype)
        for blk in self.stage2:
            x = blk(x, self.grid_s2)
        e2 = x
        x, _ = self.merge2(x, self.grid_s2)
        if self.skip_fine:
            if self.skip_pool_s3 == "attention":
                sources = ((p0, self.grid_s1), (e1, self.grid_s1), (e2, self.grid_s2))
                fine = torch.cat([pool(f, x, grid, self.grid_s3)
                                  for pool, (f, grid) in zip(self.skip3_attention, sources)], dim=-1)
            else:
                fine = torch.cat([_pool_fine(p0, self.grid_s1, self.grid_s3), _pool_fine(e1, self.grid_s1, self.grid_s3),
                                  _pool_fine(e2, self.grid_s2, self.grid_s3)], dim=-1)
            x = x + self.skip3_proj(self.skip3_norm(fine)).to(x.dtype)
        if self.pos_mode == "learned":
            x = x + self.pos_embed
        else:
            # Token stride in mm is spacing * patch * 2^(stage-1); s4 uses twice the s3 stride. For odd
            # s3 grids PatchMerge pads before halving, so the last s4 centre is off by up to half a stride.
            stride_s3 = run_spacing_tensor(spacing_mm, device=x.device) * torch.tensor(
                self.patch_size, dtype=torch.float32, device=x.device) * 4.0
            coords_s3 = make_mm_coords(self.grid_s3, stride_s3, device=x.device)
            x = x + self.coord_embed(coords_s3).to(x.dtype).unsqueeze(0)
        for blk in self.stage3:
            x = blk(x, self.grid_s3)
        e3 = x
        x, _ = self.merge3(x, self.grid_s3)                          # [B, N3/8, dim_s4]
        if self.skip_fine:
            if self.skip_pool_s2s4 == "attention":
                sources = ((p0, self.grid_s1), (e1, self.grid_s1), (e2, self.grid_s2), (e3, self.grid_s3))
                fine = torch.cat([pool(f, x, grid, self.grid_s4)
                                  for pool, (f, grid) in zip(self.skip4_attention, sources)], dim=-1)
            else:
                fine = torch.cat([_pool_fine(p0, self.grid_s1, self.grid_s4), _pool_fine(e1, self.grid_s1, self.grid_s4),
                                  _pool_fine(e2, self.grid_s2, self.grid_s4), _pool_fine(e3, self.grid_s3, self.grid_s4)], dim=-1)
            x = x + self.skip4_proj(self.skip4_norm(fine)).to(x.dtype)
        if self.pos_mode == "learned":
            x = x + self.pos_embed_s4
            for blk in self.stage4:
                x = blk(x, self.grid_s4)
        else:
            coords_s4 = make_mm_coords(self.grid_s4, stride_s3 * 2.0, device=x.device)
            for blk in self.stage4:
                x = blk(x, coords_s4)                                # RoPE global attention over mm positions
        e4 = x
        return {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


class DenseLatentDecoder(nn.Module):
    """Upsampling decoder yielding dense latents at the s3/s2/s1 grids."""

    def __init__(self, config):
        super().__init__()
        mc = config.model
        pc = config.patch
        self.grid_s1 = pc.grid_size_s1
        self.grid_s2 = pc.grid_size_s2
        self.grid_s3 = pc.grid_size_s3

        self.block3 = Mlp(mc.dim_s3, mc.mlp_ratio, mc.drop)
        self.norm3 = nn.LayerNorm(mc.dim_s3)
        self.up3to2 = nn.Linear(mc.dim_s3, mc.dim_s2)
        self.fuse2 = nn.LayerNorm(mc.dim_s2)
        self.block2 = Mlp(mc.dim_s2, mc.mlp_ratio, mc.drop)
        self.up2to1 = nn.Linear(mc.dim_s2, mc.dim_s1)
        self.fuse1 = nn.LayerNorm(mc.dim_s1)
        self.block1 = Mlp(mc.dim_s1, mc.mlp_ratio, mc.drop)
        self.out1 = nn.LayerNorm(mc.dim_s1)
        self.out2 = nn.LayerNorm(mc.dim_s2)
        self.out3 = nn.LayerNorm(mc.dim_s3)

    @staticmethod
    def _upsample(x: torch.Tensor, src_grid: Triple, dst_grid: Triple) -> torch.Tensor:
        y = F.interpolate(_to_grid(x, src_grid), size=tuple(dst_grid),
                          mode="trilinear", align_corners=False)
        return _to_tokens(y)

    def forward(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        e1, e2, e3 = enc["e1"], enc["e2"], enc["e3"]
        d3 = e3 + self.block3(self.norm3(e3))
        u2 = self.up3to2(self._upsample(d3, self.grid_s3, self.grid_s2))
        d2 = u2 + e2
        d2 = d2 + self.block2(self.fuse2(d2))
        u1 = self.up2to1(self._upsample(d2, self.grid_s2, self.grid_s1))
        d1 = u1 + e1
        d1 = d1 + self.block1(self.fuse1(d1))
        return {"s1": self.out1(d1), "s2": self.out2(d2), "s3": self.out3(d3)}


# ----------------------------------------------------------------------------
# Prototypes + EMA
# ----------------------------------------------------------------------------
class PrototypeProjector(nn.Module):
    """Online-vs-target prototype assignment (DINO/SwAV-style, hard target labels)."""

    def __init__(self, dim: int, num_prototypes: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = max(temperature, 1e-4)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, student: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = F.normalize(self.prototypes, dim=-1)
        logits = F.normalize(student, dim=-1) @ w.t() / self.temperature   # [B, N, K]
        with torch.no_grad():
            labels = (F.normalize(target, dim=-1) @ w.t()).argmax(dim=-1)   # [B, N]
        return logits, labels


class EMAUpdater:
    """Momentum update of a target network from a student, with a scheduled momentum."""

    def __init__(self, momentum_start: float = 0.996, momentum_end: float = 0.9999,
                 total_steps: int = 100000, schedule: str = "cosine"):
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.total_steps = max(1, int(total_steps))
        self.schedule = schedule
        self.step = 0

    def _momentum(self) -> float:
        import math
        p = min(self.step / self.total_steps, 1.0)
        if self.schedule == "linear":
            return self.momentum_start + (self.momentum_end - self.momentum_start) * p
        # cosine: raised-cosine schedule from start to end.
        return self.momentum_end - (self.momentum_end - self.momentum_start) * (math.cos(math.pi * p) + 1) / 2

    @torch.no_grad()
    def update(self, pairs: List[Tuple[nn.Module, nn.Module]]) -> float:
        m = self._momentum()
        for student, target in pairs:
            for ps, pt in zip(student.parameters(), target.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
            for bs, bt in zip(student.buffers(), target.buffers()):
                bt.data.copy_(bs.data)
        self.step += 1
        return m
