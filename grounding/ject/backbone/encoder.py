"""LAC-JEPA base model: dense Swin-UNETR-style teacher/student encoder + decoder.

This provides the image-only backbone that ART-JEPA builds on. Both the student
(masked input) and EMA target (full input) share this architecture:

    volume -> PatchEmbed3D (s1 grid) -> [mask tokens] -> local stage1
           -> PatchMerge -> local stage2 -> PatchMerge -> global stage3
    encoder skips {e1, e2, e3} -> DenseLatentDecoder -> latents {s1, s2, s3}

The three latents are the dense JEPA targets/predictions at increasing resolution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # SDPA accumulates the QKᵀ scores + softmax in fp32 (no fp16 overflow → no NaN) and is
        # memory-efficient (never materializes the full [.,.,N,N] fp32 score matrix, which OOM'd
        # the manual fp32 path). Default scale = head_dim**-0.5, matching self.scale.
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
        # SDPA (fp32-accumulated softmax, memory-efficient) with the relative-position bias passed
        # as an additive attn_mask. See Attention.forward for why this avoids the fp16-overflow NaN.
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
        nn.init.zeros_(self.ct_type_embed.weight)   # identity at init (like window_embed)
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
        self.pos_embed = nn.Parameter(torch.zeros(1, pc.num_tokens_s3, mc.dim_s3))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.stage3 = nn.ModuleList([
            build_block(mc.stage3_block_type, mc.dim_s3, mc.heads[2], window, mc.mlp_ratio, mc.drop,
                        shift=(i % 2 == 1))
            for i in range(mc.depths[2])
        ])
        # s4: coarsest GLOBAL semantic scale (whole-volume). 14³ -> 7³, dim_s3 -> dim_s4.
        self.grid_s4 = pc.grid_size_s4
        self.merge3 = PatchMerge3D(mc.dim_s3, mc.dim_s4)
        self.pos_embed_s4 = nn.Parameter(torch.zeros(1, pc.num_tokens_s4, mc.dim_s4))
        nn.init.trunc_normal_(self.pos_embed_s4, std=0.02)
        self.stage4 = nn.ModuleList([
            build_block(mc.stage4_block_type, mc.dim_s4, mc.heads[3], window, mc.mlp_ratio, mc.drop,
                        shift=(i % 2 == 1))
            for i in range(mc.depths[3])
        ])

    def forward(self, volume: torch.Tensor, window_id: torch.Tensor | None = None,
                mask_s1: torch.Tensor | None = None,
                ct_type_id: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(volume)                                 # [B, N1, dim_s1]
        if window_id is not None:
            x = x + self.window_embed(window_id).unsqueeze(1)
        if ct_type_id is not None:
            x = x + self.ct_type_embed(ct_type_id).unsqueeze(1)
        if mask_s1 is not None:
            x = torch.where(mask_s1.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        for blk in self.stage1:
            x = blk(x, self.grid_s1)
        e1 = x
        x, _ = self.merge1(x, self.grid_s1)
        for blk in self.stage2:
            x = blk(x, self.grid_s2)
        e2 = x
        x, _ = self.merge2(x, self.grid_s2)
        x = x + self.pos_embed
        for blk in self.stage3:
            x = blk(x, self.grid_s3)
        e3 = x
        x, _ = self.merge3(x, self.grid_s3)                          # 14³ -> 7³, dim_s3 -> dim_s4
        x = x + self.pos_embed_s4
        for blk in self.stage4:
            x = blk(x, self.grid_s4)                                 # GLOBAL whole-volume semantic scale
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
        # cosine: start -> end following a raised cosine.
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
