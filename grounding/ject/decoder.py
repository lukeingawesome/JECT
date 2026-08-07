"""Full-resolution Swin-UNETR segmentation decoder for ART/LAC-JEPA.

The pretraining backbone (``DenseSwinUNETREncoder``) is a shifted-window Swin encoder
producing skip features at the s1/s2/s3 grids (e.g. 56³/28³/14³ at 224³/patch-4). This
module adds the UNETR-style decoder that turns those skips + raw-CT conv stems into
**full-resolution** logits — the actual downstream segmenter (the aux seg head used
during pretraining is only a 1×1 conv on 28³ and is not a real decoder).

    volume ──► [Swin encoder] ──► e1 (56³) e2 (28³) e3 (14³, bottleneck)
       │                                │        │        │
       ├─ stem0 (224³) ─────────────────┼────────┼────────┤ skips
       └─ stem1 (112³) ─────────────────┘        │        │
                                                  ▼
    14³ → up→28 (+e2) → up→56 (+e1) → up→112 (+stem1) → up→224 (+stem0) → 1×1 conv → logits

Weights load from an ART/LAC-JEPA checkpoint via :func:`load_encoder_from_art_checkpoint`
(the pretrained ``student_encoder.*`` params). Encoder is frozen by default (exp_054
showed full unfreeze is unstable); flip ``freeze_encoder=False`` for end-to-end fine-tune.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ject.backbone.encoder import DenseSwinUNETREncoder
except ImportError:
    from ject.backbone.encoder import DenseSwinUNETREncoder


class ConvBlock3D(nn.Module):
    """Two 3×3×3 conv + InstanceNorm + LeakyReLU, with a residual (nnU-Net style)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


class UpBlock3D(nn.Module):
    """Transpose-conv upsample, concat skip, fuse (UNETR up block)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:                       # guard off-by-one from padded grids
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class AddUpBlock3D(nn.Module):
    """UpBlock3D with a projected-ADDED skip instead of a concatenated one.

    Concat momentarily materialises `out_ch + skip_ch` channels at the FINER grid, which is both the
    tensor that OOMs at 336³ and the one that pins hi_ch at 56 against the 2³¹ cuDNN index limit.
    Projecting the skip to out_ch and adding holds peak width at out_ch instead of ~2x that.

    NOT interchangeable with SepUpBlock3D, which ALSO swaps the fuse for depthwise-separable convs.
    The single run that used SepUpBlock3D (--sep_up4) scored 0.1397, but it changed the fusion AND
    the convolution type together; this block isolates the fusion so the two can be attributed
    separately. Everything else -- transpose-conv upsample, shape guard, ConvBlock3D fuse -- is
    identical to UpBlock3D, so `--skip_fusion add` differs from the baseline in exactly one respect.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv3d(skip_ch, out_ch, 1, bias=False) if skip_ch != out_ch else nn.Identity()
        self.block = ConvBlock3D(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:                       # guard off-by-one from padded grids
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(x + self.skip_proj(skip))


class SepConvBlock3D(nn.Module):
    """Depthwise-separable version of ConvBlock3D for full-resolution decoding.

    Each 3x3x3 dense conv (C*C*27 MACs/voxel) becomes depthwise 3x3x3 (C*27) + pointwise 1x1x1
    (C*C), which at C=56 is ~24x fewer MACs. Measured at 336^3/C=56: the dense pair is 12.85 TFLOP,
    the separable pair 0.35 TFLOP.

    Unlike a trilinear feature lift, this computes REAL convolutions on the full-resolution grid, so
    the output can express detail the res/2 features do not contain -- the reason to prefer it when
    the res/2 output ceiling (0.826 median on our val GT, 0.749 for findings <500 voxels) is what
    limits precision. Same "decomposed 3D convolutions" idea Text3DSAM uses.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw1 = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.dw2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw2 = nn.Conv3d(out_ch, out_ch, 1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.act(self.norm1(self.pw1(self.dw1(x))))
        x = self.norm2(self.pw2(self.dw2(x)))
        return self.act(x + r)


class SepUpBlock3D(nn.Module):
    """UpBlock3D with a separable fuse and an ADDED (not concatenated) skip.

    Concatenation would materialise a 2C-channel tensor at full resolution -- 15.8 GiB at 336^3,
    which is what OOMs before any convolution runs. Projecting the skip to C and adding keeps peak
    memory at C channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_proj = nn.Conv3d(skip_ch, out_ch, 1, bias=False) if skip_ch != out_ch else nn.Identity()
        self.block = SepConvBlock3D(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(x + self.skip_proj(skip))


class SwinUNETRSegModel(nn.Module):
    """Pretrained Swin encoder + full-resolution UNETR decoder -> voxel logits."""

    def __init__(self, config, num_classes: int, freeze_encoder: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        mc = config.model
        self.encoder = DenseSwinUNETREncoder(config)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.dim_s1, self.dim_s2, self.dim_s3 = mc.dim_s1, mc.dim_s2, mc.dim_s3
        self.grid_s1 = config.patch.grid_size_s1
        self.grid_s2 = config.patch.grid_size_s2
        self.grid_s3 = config.patch.grid_size_s3
        cs = mc.dim_s1                                          # decoder stem channels
        in_ch = config.volume.channels

        # Raw-CT conv stems for the high-res skips the transformer doesn't produce.
        self.stem0 = ConvBlock3D(in_ch, cs)                    # full resolution
        self.stem1 = nn.Sequential(                            # half resolution
            nn.Conv3d(cs, cs, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(cs, affine=True), nn.LeakyReLU(0.01, inplace=True))

        self.bottleneck = ConvBlock3D(mc.dim_s3, mc.dim_s3)
        self.up1 = UpBlock3D(mc.dim_s3, mc.dim_s2, mc.dim_s2)   # 14³ -> 28³ (+e2)
        self.up2 = UpBlock3D(mc.dim_s2, mc.dim_s1, mc.dim_s1)   # 28³ -> 56³ (+e1)
        self.up3 = UpBlock3D(mc.dim_s1, cs, cs)                 # 56³ -> 112³ (+stem1)
        self.up4 = UpBlock3D(cs, cs, cs)                        # 112³ -> 224³ (+stem0)
        self.out = nn.Conv3d(cs, num_classes, 1)

        self.deep_supervision = deep_supervision
        if deep_supervision:                                   # aux heads at 56³ and 112³
            self.aux56 = nn.Conv3d(mc.dim_s1, num_classes, 1)
            self.aux112 = nn.Conv3d(cs, num_classes, 1)

    def _to_grid(self, tok: torch.Tensor, grid, dim: int) -> torch.Tensor:
        B = tok.shape[0]
        return tok.transpose(1, 2).reshape(B, dim, *grid)

    def forward(self, volume: torch.Tensor, window_id: Optional[torch.Tensor] = None,
                ct_type_id: Optional[torch.Tensor] = None):
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            enc = self.encoder(volume, window_id=window_id, mask_s1=None, ct_type_id=ct_type_id)
        e1 = self._to_grid(enc["e1"], self.grid_s1, self.dim_s1)
        e2 = self._to_grid(enc["e2"], self.grid_s2, self.dim_s2)
        e3 = self._to_grid(enc["e3"], self.grid_s3, self.dim_s3)

        f0 = self.stem0(volume)
        f1 = self.stem1(f0)
        x = self.bottleneck(e3)
        x = self.up1(x, e2)
        d56 = self.up2(x, e1)
        d112 = self.up3(d56, f1)
        d224 = self.up4(d112, f0)
        logits = self.out(d224)
        if self.deep_supervision and self.training:
            return {"logits": logits, "aux": [self.aux56(d56), self.aux112(d112)]}
        return logits


def load_encoder_from_art_checkpoint(seg_model: SwinUNETRSegModel, checkpoint_path: str,
                                     verbose: bool = True) -> dict:
    """Load the pretrained ``student_encoder.*`` weights into the seg model's encoder."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    prefix = "student_encoder."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:  # fall back: maybe a bare encoder state dict was passed
        enc_state = state
    missing, unexpected = seg_model.encoder.load_state_dict(enc_state, strict=False)
    # Loud-fail guard: if the frozen encoder was left ~entirely random (wrong checkpoint schema, or
    # no student_encoder.* keys), the fine-tune would run on an uninitialized encoder while only
    # printing a skip report. Require most encoder tensors to actually match.
    enc_total = len(seg_model.encoder.state_dict())
    matched = enc_total - len(missing)
    if matched < max(1, enc_total // 2):
        raise RuntimeError(
            f"load_encoder_from_art_checkpoint matched only {matched}/{enc_total} encoder tensors "
            f"from {checkpoint_path} (checkpoint top-level keys={list(state.keys())[:8]}). "
            f"Wrong checkpoint schema? Expected torch.save({{'model': model.state_dict()}}) with "
            f"'{prefix}*' params.")
    report = {"loaded": len(enc_state) - len(unexpected), "missing": list(missing),
              "unexpected": list(unexpected)}
    if verbose:
        print(f"Seg encoder load: matched={report['loaded']} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    return report
