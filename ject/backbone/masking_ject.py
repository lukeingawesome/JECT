"""Anatomy-aware masking for JECT."""

from __future__ import annotations

from typing import Tuple
import random

import torch

from ject.backbone.masking import LACMaskBatch, _dilate_mask_3d, upsample_mask_3d, sample_lac_mask_batch

Triple = Tuple[int, int, int]


def _flat_to_zyx(idx: int, grid: Triple) -> tuple[int, int, int]:
    D, H, W = grid
    z = idx // (H * W)
    r = idx % (H * W)
    y = r // W
    x = r % W
    return int(z), int(y), int(x)


def _sample_anatomy_single_s3(
    anatomy_mask_s3: torch.Tensor,
    grid_s3: Triple,
    mask_ratio: float,
    num_blocks: int,
    min_block: Triple,
    max_block: Triple,
    allow_random_fill: bool = True,
    anat_frac: float = 0.5,
) -> torch.Tensor:
    """Partially mask the referenced anatomy, then top up with random tokens outside it.

    Masks ``anat_frac`` of the anatomy's tokens (the rest stay visible to anchor the
    prediction), then random-fills tokens outside the anatomy until the total reaches
    ``mask_ratio`` of the whole volume.
    """
    D, H, W = grid_s3
    n = D * H * W
    anatomy = anatomy_mask_s3.detach().cpu().bool().flatten()
    valid = anatomy.nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        # No anatomy tokens: fall back to the image-only LAC block mask.
        return sample_lac_mask_batch(
            batch_size=1, grid_s1=grid_s3, grid_s2=grid_s3, grid_s3=grid_s3,
            mask_ratio=mask_ratio, num_target_blocks=num_blocks,
            min_block_size_s3=min_block, max_block_size_s3=max_block,
            buffer_s3=0, allow_random_fill=allow_random_fill,
        ).loss_mask_s3[0].cpu()

    target_total = max(1, min(n - 1, int(round(mask_ratio * n))))    # fraction of the whole volume
    anat_n = int(valid.numel())
    anat_target = max(1, min(target_total, int(round(anat_frac * anat_n))))
    mask = torch.zeros(n, dtype=torch.bool)
    anat_pick = valid[torch.randperm(anat_n)[:anat_target]]          # partial anatomy
    mask[anat_pick] = True
    need = target_total - int(mask.sum())
    if need > 0:
        outside = (~anatomy).nonzero(as_tuple=True)[0]              # random-fill outside the anatomy
        if outside.numel() > 0:
            fill = outside[torch.randperm(outside.numel())[:need]]
            mask[fill] = True
    return mask


def sample_ject_mask_batch(
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
    anatomy_mask_s3: torch.Tensor | None = None,
    anatomy_sampling_prob: float = 0.75,
    anatomy_mask_frac: float = 0.5,
    anatomy_mask_valid: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> LACMaskBatch:
    """Sample JECT masks.

    If anatomy_mask_s3 is supplied, most samples target cuboids inside the selected
    report/anatomy region. Otherwise falls back to LAC-JEPA masks.
    """
    if anatomy_mask_s3 is None or anatomy_sampling_prob <= 0:
        return sample_lac_mask_batch(
            batch_size=batch_size, grid_s1=grid_s1, grid_s2=grid_s2, grid_s3=grid_s3,
            mask_ratio=mask_ratio, num_target_blocks=num_target_blocks,
            min_block_size_s3=min_block_size_s3, max_block_size_s3=max_block_size_s3,
            buffer_s3=buffer_s3, allow_random_fill=allow_random_fill, device=device,
        )

    loss_rows = []
    for b in range(batch_size):
        # Samples without a valid anatomy mask use block masking.
        valid_b = anatomy_mask_valid is None or bool(anatomy_mask_valid[b])
        if valid_b and random.random() < anatomy_sampling_prob:
            loss_rows.append(_sample_anatomy_single_s3(
                anatomy_mask_s3=anatomy_mask_s3[b], grid_s3=grid_s3,
                mask_ratio=mask_ratio, num_blocks=num_target_blocks,
                min_block=min_block_size_s3, max_block=max_block_size_s3,
                allow_random_fill=allow_random_fill, anat_frac=anatomy_mask_frac,
            ))
        else:
            fallback = sample_lac_mask_batch(
                batch_size=1, grid_s1=grid_s3, grid_s2=grid_s3, grid_s3=grid_s3,
                mask_ratio=mask_ratio, num_target_blocks=num_target_blocks,
                min_block_size_s3=min_block_size_s3, max_block_size_s3=max_block_size_s3,
                buffer_s3=0, allow_random_fill=allow_random_fill,
            ).loss_mask_s3[0].cpu()
            loss_rows.append(fallback)
    loss_s3 = torch.stack(loss_rows, dim=0)
    encoder_s3 = _dilate_mask_3d(loss_s3, grid_s3, radius=buffer_s3)
    for b in range(batch_size):
        if encoder_s3[b].all():
            non_loss = (~loss_s3[b]).nonzero(as_tuple=True)[0]
            if non_loss.numel() > 0:
                encoder_s3[b, non_loss[torch.randint(0, non_loss.numel(), (1,)).item()]] = False

    loss_s2 = upsample_mask_3d(loss_s3, grid_s3, grid_s2)
    loss_s1 = upsample_mask_3d(loss_s3, grid_s3, grid_s1)
    encoder_s2 = upsample_mask_3d(encoder_s3, grid_s3, grid_s2)
    encoder_s1 = upsample_mask_3d(encoder_s3, grid_s3, grid_s1)
    batch = LACMaskBatch(
        encoder_mask_s1=encoder_s1, encoder_mask_s2=encoder_s2, encoder_mask_s3=encoder_s3,
        loss_mask_s1=loss_s1, loss_mask_s2=loss_s2, loss_mask_s3=loss_s3,
    )
    return batch.to(device) if device is not None else batch
