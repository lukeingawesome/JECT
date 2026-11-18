"""Intra-volume region/utterance matching loss.

Within a single CT, each anatomy region's pooled image embedding is matched to its
assigned report utterance embedding and contrasted against the other regions of the
same volume. The contrastive axis is the number of regions per case (R), not the
batch, so the loss is informative at batch size 1.

Inputs (per batch item, R regions):
  * s3 tokens            [B, N3, C]      encoder/decoder region features
  * region masks         [B, R, N3] bool token membership of each region (s3 grid)
  * region text emb      [B, R, D]       one utterance vector per region
  * region valid         [B, R]   bool   which region slots are real (not padding)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_region_embeddings(s3_tokens: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
    """Masked-mean pool per region.

    s3_tokens:    [B, N, C]
    region_masks: [B, R, N]  (bool / float; True = token in region)
    returns:      [B, R, C]  (empty regions pool to 0)
    """
    B, N, C = s3_tokens.shape
    w = region_masks.to(dtype=s3_tokens.dtype)                 # [B, R, N]
    denom = w.sum(dim=-1, keepdim=True).clamp(min=1.0)         # [B, R, 1]
    pooled = torch.bmm(w, s3_tokens) / denom                   # [B, R, C]
    return pooled


def intra_volume_region_text_loss(
    region_img_emb: torch.Tensor,
    region_txt_emb: torch.Tensor,
    region_valid: torch.Tensor,
    temperature: float = 0.07,
    region_anatomy_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric region/utterance matching, computed independently per volume.

    region_img_emb: [B, R, D]
    region_txt_emb: [B, R, D]
    region_valid:   [B, R] bool  (regions with both a mask and an utterance)
    region_anatomy_ids: [B, R] optional; other slots with the same anatomy are excluded from a
        slot's negatives, since they pool the same image region.
    Returns a scalar averaged over volumes that have >= 2 valid regions. Volumes with < 2 valid
    regions contribute 0 (nothing to contrast against).
    """
    B, R, D = region_img_emb.shape
    img = F.normalize(region_img_emb, dim=-1)
    txt = F.normalize(region_txt_emb, dim=-1)
    valid = region_valid.bool()

    total = region_img_emb.sum() * 0.0
    counted = 0
    for b in range(B):
        vmask = valid[b]                                       # [R]
        if int(vmask.sum()) < 2:
            continue
        # Skip volumes whose slots all share one anatomy: with same-anatomy negatives masked
        # the loss would be a constant 0.
        if region_anatomy_ids is not None:
            aid_all = region_anatomy_ids[b][vmask]
            if aid_all.unique().numel() < 2:
                continue
        vi = img[b][vmask]                                     # [r, D]
        vt = txt[b][vmask]                                     # [r, D]
        logits = vi @ vt.t() / max(temperature, 1e-6)         # [r, r]
        r = vi.shape[0]
        if region_anatomy_ids is not None:
            aid = region_anatomy_ids[b][vmask]                # [r]
            same = aid.unsqueeze(0) == aid.unsqueeze(1)       # [r, r] same anatomy
            off = ~torch.eye(r, dtype=torch.bool, device=logits.device)
            # Fill with logits.dtype's min: fp32's finfo.min overflows bf16 under autocast.
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(same & off, neg_inf)  # keep diagonal (positive), drop same-anat negs
        labels = torch.arange(r, device=vi.device)
        loss_b = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        total = total + loss_b
        counted += 1
    if counted == 0:
        return total                                          # 0, grad-safe
    return total / counted


class IntraVolumeRegionText(nn.Module):
    """Pool s3 region embeddings, project them and the utterance vectors, and match them.

    img_proj maps the s3 dim to the shared dim; txt_proj maps the text dim to the shared dim.
    """

    def __init__(self, s3_dim: int, text_dim: int, shared_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.img_proj = nn.Sequential(nn.LayerNorm(s3_dim), nn.Linear(s3_dim, shared_dim),
                                      nn.LayerNorm(shared_dim))
        self.txt_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, shared_dim),
                                      nn.LayerNorm(shared_dim))

    def forward(self, s3_tokens: torch.Tensor, region_masks: torch.Tensor,
                region_txt_emb: torch.Tensor, region_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if region_valid is None:
            # A region is valid if it has >=1 token and a non-zero utterance vector.
            has_tok = region_masks.to(torch.bool).any(dim=-1)          # [B, R]
            has_txt = region_txt_emb.abs().sum(dim=-1) > 0             # [B, R]
            region_valid = has_tok & has_txt
        img = self.img_proj(pool_region_embeddings(s3_tokens, region_masks))
        txt = self.txt_proj(region_txt_emb)
        return intra_volume_region_text_loss(img, txt, region_valid, self.temperature)
