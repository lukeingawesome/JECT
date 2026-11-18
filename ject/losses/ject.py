"""Losses for JECT."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ject.config.ject import JECTConfig
from ject.losses.lac import LACJEPALoss


def symmetric_contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if a.shape[0] < 2:
        return a.sum() * 0.0
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / max(temperature, 1e-6)
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Simple mean Dice loss over foreground + background classes."""
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long().clamp(min=0, max=logits.shape[1] - 1)
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)
    dims = (0, 2, 3, 4)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class JECTLoss(nn.Module):
    def __init__(self, cfg: JECTConfig):
        super().__init__()
        self.cfg = cfg
        self.lac_loss = LACJEPALoss(cfg)
        # Warm-up scale on the text-supervision losses, set by the trainer to min(1, step / ramp_steps)
        # so text targets do not dominate the encoder early in training. 1.0 at eval.
        self.text_scale = 1.0

    def forward(self, out: Dict[str, torch.Tensor], seg_mask: Optional[torch.Tensor] = None,
                report_labels: Optional[torch.Tensor] = None,
                report_label_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses = self.lac_loss(out)
        total = losses["total"]

        # Anatomy-pooled JEPA residual (region-level target; complements per-patch JEPA).
        if "region_jepa" in out and self.cfg.ject_loss.lambda_region_jepa > 0:
            losses["region_jepa"] = out["region_jepa"]
            total = total + self.cfg.ject_loss.lambda_region_jepa * out["region_jepa"]
        else:
            losses["region_jepa"] = total.detach() * 0.0

        # Weak grounding: predictor localization map vs the referenced anatomy region.
        ts = float(self.text_scale)
        lam_wg = getattr(self.cfg.ject_loss, "lambda_weakground", 0.0)
        if "weakground" in out and lam_wg > 0:
            losses["weakground"] = out["weakground"]
            total = total + ts * lam_wg * out["weakground"]
        else:
            losses["weakground"] = total.detach() * 0.0

        # Masked-utterance prediction (spatial, via the grounder's cross-attention).
        lam_mup = getattr(self.cfg.ject_loss, "lambda_mup", 0.0)
        if "mup" in out and lam_mup > 0:
            losses["mup"] = out["mup"]
            total = total + ts * lam_mup * out["mup"]
        else:
            losses["mup"] = total.detach() * 0.0

        bind_weight = float(getattr(self.cfg.anatomy, "mup_image_binding_weight", 0.0))
        if "mup_image_binding" in out and lam_mup > 0 and bind_weight > 0:
            losses["mup_image_binding"] = out["mup_image_binding"]
            total = total + ts * lam_mup * bind_weight * out["mup_image_binding"]
        else:
            losses["mup_image_binding"] = total.detach() * 0.0

        nce_weight = float(getattr(self.cfg.anatomy, "mup_conditional_nce_weight", 0.0))
        if "mup_conditional_nce" in out and lam_mup > 0 and nce_weight > 0:
            losses["mup_conditional_nce"] = out["mup_conditional_nce"]
            total = total + ts * lam_mup * nce_weight * out["mup_conditional_nce"]
        else:
            losses["mup_conditional_nce"] = total.detach() * 0.0

        section_nce_weight = float(getattr(self.cfg.anatomy, "mup_section_nce_weight", 0.0))
        if "mup_section_nce" in out and lam_mup > 0 and section_nce_weight > 0:
            losses["mup_section_nce"] = out["mup_section_nce"]
            total = total + ts * lam_mup * section_nce_weight * out["mup_section_nce"]
        else:
            losses["mup_section_nce"] = total.detach() * 0.0

        section_geometry_weight = float(getattr(
            self.cfg.anatomy, "mup_section_geometry_weight", 0.0))
        if ("mup_section_geometry" in out and lam_mup > 0
                and section_geometry_weight > 0):
            losses["mup_section_geometry"] = out["mup_section_geometry"]
            total = total + ts * lam_mup * section_geometry_weight * out["mup_section_geometry"]
        else:
            losses["mup_section_geometry"] = total.detach() * 0.0

        lam_rt = self.cfg.ject_loss.lambda_anatomy_region_text
        if "region_text_intra" in out and lam_rt > 0:
            # Intra-volume region/utterance matching; works at batch size 1.
            region_text = out["region_text_intra"]
            losses["region_text"] = region_text
            total = total + lam_rt * region_text
        elif "region_emb" in out and "text_emb" in out and lam_rt > 0:
            # Batch-level contrastive; zero at batch size < 2.
            region_text = symmetric_contrastive_loss(
                out["region_emb"], out["text_emb"], temperature=self.cfg.ject_loss.contrastive_temperature,
            )
            losses["region_text"] = region_text
            total = total + lam_rt * region_text
        else:
            losses["region_text"] = total.detach() * 0.0

        if "report_logits" in out and report_labels is not None and self.cfg.ject_loss.lambda_report_labels > 0:
            logits = out["report_logits"]
            labels = report_labels.to(device=logits.device, dtype=logits.dtype)
            if report_label_mask is None:
                mask = torch.ones_like(labels)
            else:
                mask = report_label_mask.to(device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            report_loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
            losses["report_labels"] = report_loss
            total = total + self.cfg.ject_loss.lambda_report_labels * report_loss

        if "seg_logits" in out and seg_mask is not None and self.cfg.ject_loss.lambda_seg_aux > 0:
            logits = out["seg_logits"]
            seg_mask = seg_mask.to(device=logits.device)
            # Aux seg logits are on the s1 grid; downsample the target to match.
            if seg_mask.shape[-3:] != logits.shape[-3:]:
                seg_mask = F.interpolate(seg_mask[:, None].float(), size=logits.shape[-3:],
                                         mode="nearest")[:, 0]
            seg_mask = seg_mask.long()
            ce = F.cross_entropy(logits, seg_mask)
            dice = multiclass_dice_loss(logits, seg_mask)
            seg_loss = 0.5 * ce + 0.5 * dice
            losses["seg_aux"] = seg_loss
            total = total + self.cfg.ject_loss.lambda_seg_aux * seg_loss
        else:
            losses["seg_aux"] = total.detach() * 0.0

        losses["total"] = total
        return losses
