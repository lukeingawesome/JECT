"""Masking for LAC-JEPA (image-only base).

Targets are sampled as cuboid blocks on the coarsest ``s3`` token grid, then the
encoder mask (targets dilated by a buffer) and both masks are projected up to the
finer ``s2`` / ``s1`` grids. Sampling at s3 and upsampling keeps the multi-scale
masks hierarchy-consistent (a masked s3 token corresponds to whole s2/s1 blocks),
which is what prevents masked content leaking through patch merging.

Mask convention (all tensors are ``[B, N]`` boolean):
  * ``loss_mask_*``    True  -> token is a prediction target.
  * ``encoder_mask_*`` True  -> token is hidden from the student encoder
                                (target region dilated by ``buffer_s3``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from typing import Tuple

Triple = Tuple[int, int, int]


@dataclass
class LACMaskBatch:
    encoder_mask_s1: torch.Tensor
    encoder_mask_s2: torch.Tensor
    encoder_mask_s3: torch.Tensor
    loss_mask_s1: torch.Tensor
    loss_mask_s2: torch.Tensor
    loss_mask_s3: torch.Tensor

    def to(self, device: torch.device | str) -> "LACMaskBatch":
        return LACMaskBatch(
            encoder_mask_s1=self.encoder_mask_s1.to(device),
            encoder_mask_s2=self.encoder_mask_s2.to(device),
            encoder_mask_s3=self.encoder_mask_s3.to(device),
            loss_mask_s1=self.loss_mask_s1.to(device),
            loss_mask_s2=self.loss_mask_s2.to(device),
            loss_mask_s3=self.loss_mask_s3.to(device),
        )


def _dilate_mask_3d(mask: torch.Tensor, grid: Triple, radius: int) -> torch.Tensor:
    """Morphologically dilate a flattened 3D boolean mask by ``radius`` tokens."""
    if radius is None or radius <= 0:
        return mask.clone()
    D, H, W = grid
    B = mask.shape[0]
    x = mask.view(B, 1, D, H, W).float()
    k = 2 * radius + 1
    y = F.max_pool3d(x, kernel_size=k, stride=1, padding=radius)
    return (y.view(B, D * H * W) > 0.5)


def upsample_mask_3d(mask: torch.Tensor, src_grid: Triple, dst_grid: Triple) -> torch.Tensor:
    """Nearest-neighbour project a flattened boolean mask from ``src_grid`` to ``dst_grid``."""
    if tuple(src_grid) == tuple(dst_grid):
        return mask.clone()
    B = mask.shape[0]
    x = mask.view(B, 1, *src_grid).float()
    y = F.interpolate(x, size=tuple(dst_grid), mode="nearest")
    n = dst_grid[0] * dst_grid[1] * dst_grid[2]
    return (y.view(B, n) > 0.5)


def _sample_block_mask(
    grid: Triple,
    mask_ratio: float,
    num_blocks: int,
    min_block: Triple,
    max_block: Triple,
    allow_random_fill: bool = True,
) -> torch.Tensor:
    """Sample a single ``[N]`` boolean target mask of cuboid blocks on ``grid``."""
    D, H, W = grid
    n = D * H * W
    target_count = max(1, min(n - 1, int(round(mask_ratio * n))))
    mask = torch.zeros(D, H, W, dtype=torch.bool)

    for _ in range(max(1, num_blocks)):
        sd = random.randint(max(1, min_block[0]), max(1, min(max_block[0], D)))
        sh = random.randint(max(1, min_block[1]), max(1, min(max_block[1], H)))
        sw = random.randint(max(1, min_block[2]), max(1, min(max_block[2], W)))
        d0 = random.randint(0, max(0, D - sd))
        h0 = random.randint(0, max(0, H - sh))
        w0 = random.randint(0, max(0, W - sw))
        mask[d0:d0 + sd, h0:h0 + sh, w0:w0 + sw] = True
        if int(mask.sum()) >= target_count:
            break

    flat = mask.view(-1)
    if allow_random_fill and int(flat.sum()) < target_count:
        candidates = (~flat).nonzero(as_tuple=True)[0]
        if candidates.numel() > 0:
            extra = target_count - int(flat.sum())
            choose = candidates[torch.randperm(candidates.numel())[:extra]]
            flat[choose] = True

    if int(flat.sum()) == 0:
        flat[random.randint(0, n - 1)] = True
    if bool(flat.all()):
        flat[random.randint(0, n - 1)] = False
    return flat


def sample_lac_mask_batch(
    batch_size: int,
    grid_s1: Triple,
    grid_s2: Triple,
    grid_s3: Triple,
    mask_ratio: float = 0.45,
    num_target_blocks: int = 4,
    min_block_size_s3: Triple = (1, 1, 1),
    max_block_size_s3: Triple = (2, 3, 3),
    buffer_s3: int = 1,
    allow_random_fill: bool = True,
    device: torch.device | str | None = None,
) -> LACMaskBatch:
    """Sample image-only LAC-JEPA masks for a batch.

    Targets are cuboid blocks on ``grid_s3``; the encoder mask is the targets
    dilated by ``buffer_s3`` (kept from being fully opaque), and both masks are
    upsampled to the finer ``s2`` / ``s1`` grids.
    """
    loss_rows = [
        _sample_block_mask(
            grid=grid_s3, mask_ratio=mask_ratio, num_blocks=num_target_blocks,
            min_block=min_block_size_s3, max_block=max_block_size_s3,
            allow_random_fill=allow_random_fill,
        )
        for _ in range(batch_size)
    ]
    loss_s3 = torch.stack(loss_rows, dim=0)
    encoder_s3 = _dilate_mask_3d(loss_s3, grid_s3, radius=buffer_s3)
    # Never hide the whole volume from the encoder.
    for b in range(batch_size):
        if bool(encoder_s3[b].all()):
            visible = (~loss_s3[b]).nonzero(as_tuple=True)[0]
            if visible.numel() > 0:
                encoder_s3[b, visible[torch.randint(0, visible.numel(), (1,)).item()]] = False

    batch = LACMaskBatch(
        encoder_mask_s1=upsample_mask_3d(encoder_s3, grid_s3, grid_s1),
        encoder_mask_s2=upsample_mask_3d(encoder_s3, grid_s3, grid_s2),
        encoder_mask_s3=encoder_s3,
        loss_mask_s1=upsample_mask_3d(loss_s3, grid_s3, grid_s1),
        loss_mask_s2=upsample_mask_3d(loss_s3, grid_s3, grid_s2),
        loss_mask_s3=loss_s3,
    )
    return batch.to(device) if device is not None else batch
