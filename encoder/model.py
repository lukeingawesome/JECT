"""JECT image encoder: standalone architecture definition.

Depends only on ``torch``. Defines the 3D Swin-UNETR-style hierarchical encoder,
the dense latent decoder on top of it, and the config needed to rebuild both from
a training checkpoint. Training-only components (EMA, predictors, text heads,
losses) are not included.

Encoder layout (224 input, patch 4, `huge` variant)::

    volume [B,1,224,224,224]
      -> PatchEmbed3D (conv4x4x4)            -> e0  56^3  x 128
      -> stage1  (2x swin, window 7)         -> e1  56^3  x 128
      -> PatchMerge -> stage2 (4x swin)      -> e2  28^3  x 256
      -> PatchMerge -> +pos_embed -> stage3 (9x swin)  -> e3  14^3 x 512
      -> PatchMerge -> +pos_embed_s4 -> stage4 (9x global attn) -> e4 7^3 x 1024

At 336 input the grids are 84/42/21/11. The learned s3/s4 absolute position
embeddings are resolution-specific, so each input size has its own checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Triple = Tuple[int, int, int]

# Per-variant (dims, heads, depths) for scales (s1, s2, s3, s4).
VARIANTS = {
    "tiny":  dict(dims=(32, 64, 128, 256),    heads=(2, 4, 8, 8),    depths=(2, 2, 2, 2)),
    "small": dict(dims=(48, 96, 192, 384),    heads=(3, 6, 12, 12),  depths=(2, 2, 4, 2)),
    "base":  dict(dims=(64, 128, 256, 512),   heads=(4, 8, 16, 16),  depths=(2, 2, 4, 2)),
    "large": dict(dims=(96, 192, 384, 768),   heads=(6, 12, 24, 24), depths=(2, 2, 8, 4)),
    "huge":  dict(dims=(128, 256, 512, 1024), heads=(8, 16, 32, 32), depths=(2, 4, 9, 9)),
}

# HU windows the encoder was trained with. `window_id` is an additive embedding on the
# patch tokens, so the id passed at inference must match the window actually applied.
WINDOW_PARAMS = {
    0: {"center": 450,  "width": 2100},   # lung:        -600 .. 1500
    1: {"center": 195,  "width": 310},    # mediastinal:   40 ..  350
    2: {"center": 0,    "width": 2000},   # general:    -1000 .. 1000  (default)
    3: {"center": 1500, "width": 5000},   # full:       -1000 .. 4000
}
GENERAL_WINDOW_ID = 2

# ct_type to additive embedding index.
CT_TYPE_MAP = {"non-contrast": 0, "contrast": 1, "ldct": 2}
DEFAULT_CT_TYPE = 0


def _triple(x) -> Triple:
    if isinstance(x, int):
        return (x, x, x)
    vals = tuple(int(v) for v in x)
    if len(vals) == 1:
        return (vals[0],) * 3
    if len(vals) != 3:
        raise ValueError(f"expected an int or a 1/3-length iterable, got {x!r}")
    return vals


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EncoderConfig:
    """Encoder geometry and width; normally built with ``from_training_config``."""

    input_size: Triple = (224, 224, 224)
    patch_size: Triple = (4, 4, 4)
    in_channels: int = 1
    variant: str = "huge"

    local_block_type: str = "swin"     # stages 1-2
    stage3_block_type: str = "swin"
    stage4_block_type: str = "global"  # full attention over the 7^3 / 11^3 grid
    window: Triple = (7, 7, 7)
    mlp_ratio: float = 4.0
    drop: float = 0.0
    num_ct_types: int = 3
    # Cumulative bottom-up skips of pooled finer features into the s2/s3/s4 stage entries.
    skip_fine: bool = False
    # Pooling for the s3 skip: "maxmean" ([max, mean] cell pools) or "attention" (LocalAttentionPool3D).
    skip_pool_s3: str = "maxmean"
    # Pooling for the s2 and s4 skips: "maxmean" or "attention".
    skip_pool_s2s4: str = "maxmean"

    # filled by __post_init__
    dims: Tuple[int, ...] = field(default_factory=tuple)
    heads: Tuple[int, ...] = field(default_factory=tuple)
    depths: Tuple[int, ...] = field(default_factory=tuple)
    grid_s1: Triple = (0, 0, 0)
    grid_s2: Triple = (0, 0, 0)
    grid_s3: Triple = (0, 0, 0)
    grid_s4: Triple = (0, 0, 0)

    def __post_init__(self):
        self.input_size = _triple(self.input_size)
        self.patch_size = _triple(self.patch_size)
        self.window = _triple(self.window)
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {sorted(VARIANTS)}")
        spec = VARIANTS[self.variant]
        if self.skip_pool_s3 not in ("maxmean", "attention"):
            raise ValueError(f"unknown skip_pool_s3 {self.skip_pool_s3!r}; expected 'maxmean' or 'attention'")
        if self.skip_pool_s3 != "maxmean" and not self.skip_fine:
            raise ValueError("skip_pool_s3='attention' requires skip_fine (mirrors pretrain)")
        if self.skip_pool_s2s4 not in ("maxmean", "attention"):
            raise ValueError(f"unknown skip_pool_s2s4 {self.skip_pool_s2s4!r}; expected 'maxmean' or 'attention'")
        if self.skip_pool_s2s4 != "maxmean" and not self.skip_fine:
            raise ValueError("skip_pool_s2s4='attention' requires skip_fine (mirrors pretrain)")
        self.dims = tuple(spec["dims"])
        self.heads = tuple(spec["heads"])
        self.depths = tuple(spec["depths"])
        for name, v, p in zip("DHW", self.input_size, self.patch_size):
            if v % p:
                raise ValueError(f"input_size {name}={v} not divisible by patch_size {p}")
        s1 = tuple(v // p for v, p in zip(self.input_size, self.patch_size))
        s2 = tuple(g // 2 for g in s1)
        s3 = tuple(g // 4 for g in s1)
        # PatchMerge3D pads odd grids before halving, so s4 = ceil(s3 / 2) (e.g. s3 21 gives s4 11).
        s4 = tuple((g + 1) // 2 for g in s3)
        self.grid_s1, self.grid_s2, self.grid_s3, self.grid_s4 = s1, s2, s3, s4

    # -- convenience ---------------------------------------------------------
    @property
    def num_tokens(self) -> Dict[str, int]:
        return {f"s{i}": g[0] * g[1] * g[2]
                for i, g in enumerate((self.grid_s1, self.grid_s2, self.grid_s3, self.grid_s4), 1)}

    @property
    def grids(self) -> Dict[str, Triple]:
        return {"s1": self.grid_s1, "s2": self.grid_s2, "s3": self.grid_s3, "s4": self.grid_s4}

    @property
    def channels(self) -> Dict[str, int]:
        return {f"s{i}": d for i, d in enumerate(self.dims, 1)}

    @classmethod
    def from_training_config(cls, saved: dict) -> "EncoderConfig":
        """Rebuild from the `config` dict stored inside a JECT training checkpoint.

        The input size is derived from ``grid_size_s1 * patch_size`` rather than the ``volume``
        block, and the derived grids and widths are checked against the saved ones.
        """
        patch = saved.get("patch", {})
        model = saved.get("model", {})
        psize = _triple(patch.get("patch_size", (4, 4, 4)))
        g1 = _triple(patch["grid_size_s1"])
        cfg = cls(
            input_size=tuple(g * p for g, p in zip(g1, psize)),
            patch_size=psize,
            in_channels=int(saved.get("volume", {}).get("channels", 1)),
            variant=str(model.get("variant", "huge")),
            local_block_type=str(model.get("local_block_type", "swin")),
            stage3_block_type=str(model.get("stage3_block_type", "swin")),
            stage4_block_type=str(model.get("stage4_block_type", "global")),
            window=_triple(model.get("swin_window_size", (7, 7, 7))),
            mlp_ratio=float(model.get("mlp_ratio", 4.0)),
            drop=0.0,                       # no dropout at inference
            num_ct_types=int(model.get("num_ct_types", 3)),
            skip_fine=bool(model.get("skip_fine", False)),
            skip_pool_s3=str(model.get("skip_pool_s3", "maxmean")),
            skip_pool_s2s4=str(model.get("skip_pool_s2s4", "maxmean")),
        )
        for name, derived in (("grid_size_s2", cfg.grid_s2), ("grid_size_s3", cfg.grid_s3),
                              ("grid_size_s4", cfg.grid_s4)):
            if name in patch and _triple(patch[name]) != derived:
                raise ValueError(f"geometry mismatch: checkpoint {name}={tuple(patch[name])} "
                                 f"but input_size {cfg.input_size} implies {derived}")
        for key, want in (("dim_s1", cfg.dims[0]), ("dim_s2", cfg.dims[1]),
                          ("dim_s3", cfg.dims[2]), ("dim_s4", cfg.dims[3])):
            if key in model and int(model[key]) != want:
                raise ValueError(f"width mismatch: checkpoint {key}={model[key]} but "
                                 f"variant {cfg.variant!r} implies {want}")
        return cfg


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
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


def to_grid(x: torch.Tensor, grid: Triple) -> torch.Tensor:
    """[B, N, C] -> [B, C, D, H, W]."""
    B, N, C = x.shape
    D, H, W = grid
    if N != D * H * W:
        raise ValueError(f"token count {N} does not match grid {grid} ({D * H * W})")
    return x.transpose(1, 2).reshape(B, C, D, H, W)


def to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, D, H, W] -> [B, N, C]."""
    return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention over all tokens (no positional bias)."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop_p = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_mask = bias.unsqueeze(0).to(q.dtype) if bias is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock3D(nn.Module):
    """Pre-norm global transformer block (stage 4)."""

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
    B, D, H, W, C = x.shape
    wd, wh, ww = window
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, C)


def _win_merge(windows: torch.Tensor, gridp: Triple, window: Triple, B: int) -> torch.Tensor:
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
        if dim % num_heads:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        wd, wh, ww = window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"))
        coords_flat = coords.reshape(3, -1)
        rel = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
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
        ws = int(round(self.relative_position_index.numel() ** 0.5))
        table = self.relative_position_bias_table[self.relative_position_index]
        return table.view(ws, ws, -1).permute(2, 0, 1).contiguous()   # [heads, ws, ws]

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        Bn, ws, C = windows.shape
        qkv = self.qkv(windows).reshape(Bn, ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_mask = self._bias().unsqueeze(0).to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(Bn, ws, C)
        return self.proj_drop(self.proj(out))


class SwinBlock3D(nn.Module):
    """Shifted-window 3D attention block; pads internally so any grid size works."""

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
        h = h[:, :D, :H, :W].reshape(B, N, C)          # crop the window padding
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ConvNeXtBlock3D(nn.Module):
    """Depthwise 3D ConvNeXt block."""

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
        y = to_tokens(self.dwconv(to_grid(x, grid)))
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
    raise ValueError(f"unknown block type {block_type!r}")


class PatchEmbed3D(nn.Module):
    def __init__(self, channels: int, dim: int, patch_size: Triple):
        super().__init__()
        self.proj = nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.norm(to_tokens(self.proj(volume)))


def _pool_fine(x: torch.Tensor, grid: Triple, grid_to: Triple) -> torch.Tensor:
    """Max- and mean-pool z-major tokens on `grid` into the cells of `grid_to`.

    [B, N, C] -> [B, N_to, 2C], pooling the k^3 fine tokens of each coarse cell (k = grid / grid_to)."""
    B, N, C = x.shape
    D, H, W = grid
    k = D // grid_to[0]
    if (D, H, W) != (k * grid_to[0], k * grid_to[1], k * grid_to[2]):
        raise ValueError(f"skip_fine needs grid {grid} to be an integer multiple of the coarser grid {grid_to}")
    v = x.transpose(1, 2).reshape(B, C, D, H, W)
    return torch.cat([F.max_pool3d(v, k), F.avg_pool3d(v, k)], dim=1).flatten(2).transpose(1, 2)


class LocalAttentionPool3D(nn.Module):
    """Coarse-query attention over one cell's fine tokens, with a learned 3D-offset bias.

    Four heads, 128 internal channels, and the same 2C output width as max/mean pooling.
    Token order within each cell is z/y/x, so each bias index is a distinct relative 3D
    position. Attention never crosses cells.
    """

    SDPA_CHUNK = 32768   # max cells per SDPA call

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
        # One SDPA batch entry per cell (B*D*H*W). Above ~65k entries the fused kernel exceeds the CUDA
        # launch-grid limit, so large inputs are processed in chunks; the result is unchanged.
        n = q.shape[0]
        if n <= self.SDPA_CHUNK:
            pooled = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=bias, dropout_p=0.0)
        else:
            pooled = torch.cat([F.scaled_dot_product_attention(q[i:i + self.SDPA_CHUNK], kv[0][i:i + self.SDPA_CHUNK],
                                                               kv[1][i:i + self.SDPA_CHUNK], attn_mask=bias, dropout_p=0.0)
                                for i in range(0, n, self.SDPA_CHUNK)], dim=0)
        return self.output(pooled.reshape(B, D * H * W, 128))


class PatchMerge3D(nn.Module):
    """2x2x2 patch merging: [B, N, C] -> [B, N/8, out_dim] (odd grids are padded first)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(8 * in_dim)
        self.reduce = nn.Linear(8 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, grid: Triple) -> Tuple[torch.Tensor, Triple]:
        B, N, C = x.shape
        D, H, W = grid
        if D % 2 or H % 2 or W % 2:
            x = x.view(B, D, H, W, C)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
            D, H, W = D + D % 2, H + H % 2, W + W % 2
            x = x.reshape(B, D * H * W, C)
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, (D // 2) * (H // 2) * (W // 2), 8 * C)
        return self.reduce(self.norm(x)), (D // 2, H // 2, W // 2)


# ---------------------------------------------------------------------------
# Encoder / dense decoder
# ---------------------------------------------------------------------------
class JECTEncoder(nn.Module):
    """Hierarchical 3D encoder producing token features at the s1/s2/s3/s4 grids.

    Module and parameter names match the pretraining checkpoint exactly, so
    ``state_dict`` keys load without remapping (strip the ``target_encoder.``
    or ``student_encoder.`` prefix first).
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        d1, d2, d3, d4 = config.dims
        h1, h2, h3, h4 = config.heads
        n1, n2, n3, n4 = config.depths
        win = config.window
        self.grid_s1, self.grid_s2 = config.grid_s1, config.grid_s2
        self.grid_s3, self.grid_s4 = config.grid_s3, config.grid_s4

        self.patch_embed = PatchEmbed3D(config.in_channels, d1, config.patch_size)
        self.window_embed = nn.Embedding(4, d1)            # HU-window id (see WINDOW_PARAMS)
        self.ct_type_embed = nn.Embedding(config.num_ct_types, d1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d1))   # pretraining only; unused here

        self.stage1 = nn.ModuleList([
            build_block(config.local_block_type, d1, h1, win, config.mlp_ratio, config.drop,
                        shift=(i % 2 == 1)) for i in range(n1)])
        self.merge1 = PatchMerge3D(d1, d2)
        self.stage2 = nn.ModuleList([
            build_block(config.local_block_type, d2, h2, win, config.mlp_ratio, config.drop,
                        shift=(i % 2 == 1)) for i in range(n2)])
        self.merge2 = PatchMerge3D(d2, d3)
        # skip_fine: at each stage entry, a projection of pooled features from every finer representation
        # (patch embeddings, e1, e2, e3).
        self.skip_fine = bool(config.skip_fine)
        if self.skip_fine:
            self.skip2_norm = nn.LayerNorm(2 * (d1 + d1)); self.skip2_proj = nn.Linear(2 * (d1 + d1), d2)
            self.skip3_norm = nn.LayerNorm(2 * (d1 + d1 + d2)); self.skip3_proj = nn.Linear(2 * (d1 + d1 + d2), d3)
            self.skip4_norm = nn.LayerNorm(2 * (d1 + d1 + d2 + d3)); self.skip4_proj = nn.Linear(2 * (d1 + d1 + d2 + d3), d4)
        # skip_pool_s3="attention": one LocalAttentionPool3D per s3 skip source instead of max/mean pooling.
        self.skip_pool_s3 = str(config.skip_pool_s3)
        if self.skip_fine and self.skip_pool_s3 == "attention":
            self.skip3_attention = nn.ModuleList([
                LocalAttentionPool3D(d1, d3, 4),
                LocalAttentionPool3D(d1, d3, 4),
                LocalAttentionPool3D(d2, d3, 2),
            ])
        # skip_pool_s2s4="attention": the same per-cell attention pooling at the s2 and s4 entries.
        self.skip_pool_s2s4 = str(getattr(config, "skip_pool_s2s4", "maxmean"))
        if self.skip_fine and self.skip_pool_s2s4 == "attention":
            self.skip2_attention = nn.ModuleList([
                LocalAttentionPool3D(d1, d2, 2),
                LocalAttentionPool3D(d1, d2, 2),
            ])
            self.skip4_attention = nn.ModuleList([
                LocalAttentionPool3D(d1, d4, 8),
                LocalAttentionPool3D(d1, d4, 8),
                LocalAttentionPool3D(d2, d4, 4),
                LocalAttentionPool3D(d3, d4, 2),
            ])
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens["s3"], d3))
        self.stage3 = nn.ModuleList([
            build_block(config.stage3_block_type, d3, h3, win, config.mlp_ratio, config.drop,
                        shift=(i % 2 == 1)) for i in range(n3)])
        self.merge3 = PatchMerge3D(d3, d4)
        self.pos_embed_s4 = nn.Parameter(torch.zeros(1, config.num_tokens["s4"], d4))
        self.stage4 = nn.ModuleList([
            build_block(config.stage4_block_type, d4, h4, win, config.mlp_ratio, config.drop,
                        shift=(i % 2 == 1)) for i in range(n4)])

    def trunk(self, volume: torch.Tensor,
              window_id: Optional[torch.Tensor] = None,
              ct_type_id: Optional[torch.Tensor] = None,
              mask_s1: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run stages 1-3 and merge3 (with skips), stopping before pos_embed_s4 and stage 4.

        Useful for fine-tuning only stage 4. Returns ``(x_s4_in, e1, e2, e3)``.
        """
        expected = self.config.input_size
        if tuple(volume.shape[-3:]) != tuple(expected):
            raise ValueError(f"expected volume {expected}, got {tuple(volume.shape[-3:])}. "
                             f"This checkpoint is fixed-resolution (learned s3/s4 position "
                             f"embeddings); use the checkpoint matching your input size.")
        x = self.patch_embed(volume)
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
        x = x + self.pos_embed
        for blk in self.stage3:
            x = blk(x, self.grid_s3)
        e3 = x
        x, _ = self.merge3(x, self.grid_s3)
        if self.skip_fine:
            if self.skip_pool_s2s4 == "attention":
                sources = ((p0, self.grid_s1), (e1, self.grid_s1), (e2, self.grid_s2), (e3, self.grid_s3))
                fine = torch.cat([pool(f, x, grid, self.grid_s4)
                                  for pool, (f, grid) in zip(self.skip4_attention, sources)], dim=-1)
            else:
                fine = torch.cat([_pool_fine(p0, self.grid_s1, self.grid_s4), _pool_fine(e1, self.grid_s1, self.grid_s4),
                                  _pool_fine(e2, self.grid_s2, self.grid_s4), _pool_fine(e3, self.grid_s3, self.grid_s4)], dim=-1)
            x = x + self.skip4_proj(self.skip4_norm(fine)).to(x.dtype)
        return x, e1, e2, e3

    def forward(self, volume: torch.Tensor,
                window_id: Optional[torch.Tensor] = None,
                ct_type_id: Optional[torch.Tensor] = None,
                mask_s1: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode a volume.

        Args:
            volume: [B, 1, D, H, W], HU-windowed to [0, 1] (see ``preprocess``).
            window_id: [B] long, the HU window applied (see WINDOW_PARAMS).
            ct_type_id: [B] long, optional (see CT_TYPE_MAP).
            mask_s1: [B, N_s1] bool, optional; replaces tokens with the mask token.

        Returns:
            dict of token features ``{e1, e2, e3, e4}``, each [B, N_s, C_s].
        """
        x, e1, e2, e3 = self.trunk(volume, window_id, ct_type_id, mask_s1)
        x = x + self.pos_embed_s4
        for blk in self.stage4:
            x = blk(x, self.grid_s4)
        e4 = x
        return {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


class DenseLatentDecoder(nn.Module):
    """Upsampling decoder producing dense latents at the s3/s2/s1 grids.

    These latents are the JEPA targets during pretraining. Optional at inference; the
    encoder features are the recommended readout for probing and retrieval.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        d1, d2, d3, _ = config.dims
        self.grid_s1, self.grid_s2, self.grid_s3 = config.grid_s1, config.grid_s2, config.grid_s3
        self.block3 = Mlp(d3, config.mlp_ratio, config.drop)
        self.norm3 = nn.LayerNorm(d3)
        self.up3to2 = nn.Linear(d3, d2)
        self.fuse2 = nn.LayerNorm(d2)
        self.block2 = Mlp(d2, config.mlp_ratio, config.drop)
        self.up2to1 = nn.Linear(d2, d1)
        self.fuse1 = nn.LayerNorm(d1)
        self.block1 = Mlp(d1, config.mlp_ratio, config.drop)
        self.out1 = nn.LayerNorm(d1)
        self.out2 = nn.LayerNorm(d2)
        self.out3 = nn.LayerNorm(d3)

    @staticmethod
    def _upsample(x: torch.Tensor, src_grid: Triple, dst_grid: Triple) -> torch.Tensor:
        y = F.interpolate(to_grid(x, src_grid), size=tuple(dst_grid),
                          mode="trilinear", align_corners=False)
        return to_tokens(y)

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


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _substate(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    p = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(p):]: v for k, v in state.items() if k.startswith(p)}


def load_pretrained(checkpoint_path, stream: str = "target", with_decoder: bool = False,
                    map_location: str = "cpu"):
    """Rebuild the encoder (and optionally the dense decoder) from a training checkpoint.

    Args:
        checkpoint_path: path to ``checkpoint_epoch_*.pt``.
        stream: ``"target"`` (EMA teacher, recommended) or ``"student"`` (online weights).
        with_decoder: also build and load the dense latent decoder.
        map_location: device passed to ``torch.load``.

    Returns:
        ``(encoder, config, info)``, or ``(encoder, decoder, config, info)`` when
        ``with_decoder``. Modules are in eval mode. ``info`` holds the epoch, step,
        stream, checkpoint path and the raw training config.
    """
    if stream not in ("target", "student"):
        raise ValueError(f"stream must be 'target' or 'student', got {stream!r}")
    ck = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    if "model" not in ck or "config" not in ck:
        raise ValueError(f"{checkpoint_path} is not an JECT training checkpoint "
                         f"(missing 'model'/'config'; found {sorted(ck)[:6]})")
    cfg = EncoderConfig.from_training_config(ck["config"])

    encoder = JECTEncoder(cfg)
    enc_state = _substate(ck["model"], f"{stream}_encoder")
    if not enc_state:
        raise ValueError(f"checkpoint has no '{stream}_encoder.*' weights")
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    # `relative_position_index` is a non-persistent buffer rebuilt in __init__.
    missing = [k for k in missing if not k.endswith("relative_position_index")]
    if missing or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    encoder.eval()

    info = {"epoch": ck.get("epoch"), "step": ck.get("step"), "stream": stream,
            "checkpoint": str(checkpoint_path), "training_config": ck["config"]}
    if not with_decoder:
        return encoder, cfg, info

    decoder = DenseLatentDecoder(cfg)
    dec_state = _substate(ck["model"], f"{stream}_decoder")
    if not dec_state:
        raise ValueError(f"checkpoint has no '{stream}_decoder.*' weights")
    missing, unexpected = decoder.load_state_dict(dec_state, strict=True)
    decoder.eval()
    return encoder, decoder, cfg, info
