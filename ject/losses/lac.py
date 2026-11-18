"""Losses for LAC-JEPA (image-only base).

Combines, over the three latent scales:
  * masked latent-prediction loss   L_s = 1 - cos(pred[masked], stopgrad(target[masked]))
  * optional prototype cross-entropy (student assignment vs teacher hard code)
  * an anti-collapse regularizer selected by cfg.loss.anticollapse: SIGReg (LeJEPA
    Epps-Pulley, default), VICReg variance+covariance, or both
"""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ject.config.lac import LACJEPAConfig

_SCALES = ("s1", "s2", "s3")
# Anti-collapse terms and monitors also cover s4 when the model emits it; the s4 prediction loss is
# added separately.
_MONITOR_SCALES = ("s1", "s2", "s3", "s4")


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """1 - cosine between L2-normalized pred and stop-grad target, averaged over masked tokens."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    per_token = 1.0 - (pred * target).sum(dim=-1)                 # [B, N]
    m = loss_mask.to(dtype=per_token.dtype)
    denom = m.sum().clamp(min=1.0)
    return (per_token * m).sum() / denom


def _proximity_weights(loss_mask: torch.Tensor, grid, n_dilate: int = 3) -> torch.Tensor:
    """Per-token weights on visible tokens, highest next to the masked region.

    loss_mask [B, N] (True = masked token) is reshaped to ``grid`` (D, H, W) and dilated
    ``n_dilate`` times with a 3x3x3 max-pool; a token first reached at dilation step i gets weight
    1 - (i-1)/n_dilate. Masked tokens and tokens beyond ``n_dilate`` get 0.
    Returns [B, N] float."""
    B = loss_mask.shape[0]
    D, H, W = grid
    m = loss_mask.to(torch.float32).view(B, 1, D, H, W)
    visible = 1.0 - m
    prox = torch.zeros_like(m)
    cur = m
    for i in range(n_dilate):
        dil = F.max_pool3d(cur, kernel_size=3, stride=1, padding=1)
        ring = (dil - cur).clamp(min=0.0)                        # tokens first reached at distance i+1
        prox = prox + ring * (1.0 - i / max(1, n_dilate))        # closer rings get higher weight
        cur = dil
    return (prox * visible).view(B, -1)


def dense_predictive_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
                          grid, n_dilate: int = 3) -> torch.Tensor:
    """Dense loss on visible tokens, in the style of V-JEPA 2.1.

    LayerNorm'd L2 between the predictor output and the stop-grad target on visible tokens,
    weighted toward the mask boundary by ``_proximity_weights``."""
    d = pred.shape[-1]
    pn = F.layer_norm(pred, (d,))
    tn = F.layer_norm(target.detach(), (d,))
    per_token = (pn - tn).pow(2).mean(dim=-1)                     # [B, N]
    w = _proximity_weights(loss_mask, grid, n_dilate).to(per_token.dtype)
    return (per_token * w).sum() / w.sum().clamp(min=1.0)


def variance_loss(feats: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge: penalize per-channel std below ``target_std``."""
    x = feats.reshape(-1, feats.shape[-1]).float()
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(feats: torch.Tensor, max_tokens: int = 4096) -> torch.Tensor:
    """VICReg covariance term: penalize off-diagonal feature covariances."""
    x = feats.reshape(-1, feats.shape[-1]).float()
    n = x.shape[0]
    if n < 2:
        return x.sum() * 0.0
    if n > max_tokens:                                  # subsample tokens to bound cost
        idx = torch.randperm(n, device=x.device)[:max_tokens]
        x = x[idx]
        n = max_tokens
    x = x - x.mean(dim=0, keepdim=True)
    d = x.shape[-1]
    cov = (x.T @ x) / (n - 1)                           # [d, d]
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def feature_correlation(feats: torch.Tensor, max_tokens: int = 4096) -> torch.Tensor:
    """Mean absolute off-diagonal channel correlation, in [0, 1].

    Scale-invariant collapse monitor (raw covariance grows with feature magnitude); values near 1
    indicate redundant channels. Diagnostic only, not part of the loss."""
    x = feats.reshape(-1, feats.shape[-1]).float()
    n, d = x.shape
    if n < 2 or d < 2:
        return x.sum() * 0.0
    if n > max_tokens:
        idx = torch.randperm(n, device=x.device)[:max_tokens]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    xn = x / std
    corr = (xn.T @ xn) / xn.shape[0]                    # [d, d] correlation matrix
    off = corr - torch.diag(torch.diagonal(corr))
    return off.abs().sum() / (d * (d - 1))


def sigreg_loss(feats: torch.Tensor, num_slices: int = 512, n_points: int = 17,
                t_max: float = 3.0, max_tokens: int = 512, slice_chunk: int = 64) -> torch.Tensor:
    """SIGReg (LeJEPA): Sketched Isotropic Gaussian Regularization via the Epps-Pulley test.

    Port of LeJEPA's `EppsPulley` + `SlicingUnivariateTest` (github.com/rbalestr-lab/lejepa).
    Along random 1-D unit projections y = x.a, compares the empirical characteristic function
    with the N(0, 1) CF phi(t) = exp(-t^2/2):

        T = N * integral_0^inf |phi_emp(t) - phi(t)|^2 w(t) dt   (averaged over slices)

    The integral uses a fixed grid t = linspace(0, t_max, n_points) with trapezoidal weights
    folded with w(t) = exp(-t^2/2), exploiting the +/-t symmetry.

    The `* N` factor makes strength grow with the sample count, so tokens are subsampled to
    `max_tokens` (512, the N at which LeJEPA calibrates its weight). The ECF is not all-reduced
    across ranks; DDP averages gradients instead.

    Projections are computed in chunks of `slice_chunk`, so peak memory is
    [min(n, max_tokens), slice_chunk, n_points] regardless of `num_slices`.
    """
    x = feats.reshape(-1, feats.shape[-1]).float()
    n, d = x.shape
    if n < 2:
        return x.sum() * 0.0
    if n > max_tokens:
        idx = torch.randperm(n, device=x.device)[:max_tokens]
        x = x[idx]
    n_eff = x.shape[0]                                            # sample count N (post-subsample)
    n_points = int(n_points) | 1                                  # force odd (trapezoid symmetry)
    dirs = F.normalize(torch.randn(d, num_slices, device=x.device, dtype=x.dtype), dim=0)  # [d,P] unit

    t = torch.linspace(0.0, t_max, n_points, device=x.device, dtype=x.dtype)   # [K] fixed knots (t>=0)
    phi = torch.exp(-0.5 * t ** 2)                               # [K] N(0,1) CF (real; imag=0)
    dt = t_max / (n_points - 1)
    weights = torch.full((n_points,), 2.0 * dt, device=x.device, dtype=x.dtype)
    weights[0] = weights[-1] = dt                                # trapezoid endpoints (half weight)
    weights = weights * phi                                      # fold in the Gaussian weight w(t)

    # Sum the per-slice integral over chunks so the [n, chunk, K] expansion never materializes
    # the full [n, num_slices, K] tensor; divide by num_slices at the end (mean over slices).
    total = x.new_zeros(())
    chunk = max(1, int(slice_chunk))
    for c0 in range(0, num_slices, chunk):
        yc = x @ dirs[:, c0:c0 + chunk]                          # [n, cs] 1-D projections
        ty = yc.unsqueeze(-1) * t                               # [n, cs, K]
        cos_mean = ty.cos().mean(dim=0)                          # [cs, K] Re ECF
        sin_mean = ty.sin().mean(dim=0)                          # [cs, K] Im ECF
        err = (cos_mean - phi) ** 2 + sin_mean ** 2             # [cs, K] |phi_emp - phi|^2
        total = total + (err @ weights).sum()                   # integrate over K, sum the chunk
    return (total / num_slices) * n_eff                          # mean over slices, * N (Epps-Pulley)


class LACJEPALoss(nn.Module):
    def __init__(self, cfg: LACJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.dense_scale = 1.0     # dense-loss warm-up scale in [0, 1], set per step by the trainer

    def forward(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lc = self.cfg.loss
        lambdas = {"s1": lc.lambda_s1, "s2": lc.lambda_s2, "s3": lc.lambda_s3}
        losses: Dict[str, torch.Tensor] = {}

        ref = out["pred_s1"]
        total = ref.sum() * 0.0

        for s in _SCALES:
            l = masked_cosine_loss(out[f"pred_{s}"], out[f"target_{s}"], out[f"loss_mask_{s}"])
            losses[f"jepa_{s}"] = l
            total = total + lambdas[s] * l

        # s4: coarsest global scale, predicted from the encoder output without a per-token loss mask.
        if "pred_s4" in out and getattr(lc, "lambda_s4", 0.0) > 0:
            l4 = (1.0 - F.cosine_similarity(out["pred_s4"].float(),
                                            out["target_s4"].float(), dim=-1)).mean()
            losses["jepa_s4"] = l4
            total = total + lc.lambda_s4 * l4
        else:
            losses["jepa_s4"] = total.detach() * 0.0

        # Dense predictive loss on visible tokens (off by default). With dense_pred_weights the weight for
        # scale s is lambda_s * dense_pred_weights[s]; otherwise lambda_dense is split evenly over scales.
        # Scaled by dense_scale for warm-up. s4 has no per-token loss mask and is skipped.
        if getattr(lc, "use_dense_pred", False):
            scales = tuple(getattr(lc, "dense_pred_scales", ()) or (getattr(lc, "dense_pred_scale", "s2"),))
            ctx_w = tuple(getattr(lc, "dense_pred_weights", ()))
            ramp = float(getattr(self, "dense_scale", 1.0))
            dloss = total.detach() * 0.0
            dweighted = total.detach() * 0.0
            n = 0
            for i, ds in enumerate(scales):
                if f"pred_{ds}" not in out or f"loss_mask_{ds}" not in out:
                    continue                                    # s4 (no loss_mask) or disabled scale
                grid = getattr(self.cfg.patch, f"grid_size_{ds}")
                # Use the null-text prediction when available: visible-token targets do not depend on the text.
                dpred = out.get(f"dense_pred_{ds}", out[f"pred_{ds}"])
                d_s = dense_predictive_loss(
                    dpred, out[f"target_{ds}"], out[f"loss_mask_{ds}"],
                    tuple(grid), n_dilate=getattr(lc, "dense_pred_dilate", 3))
                if ctx_w:                                       # lambda_s * dense_pred_weights[i]
                    w_s = float(lambdas.get(ds, 1.0)) * (float(ctx_w[i]) if i < len(ctx_w) else 0.0)
                else:                                           # lambda_dense split over scales
                    w_s = float(getattr(lc, "lambda_dense", 0.0)) / max(1, len(scales))
                dloss = dloss + d_s
                dweighted = dweighted + w_s * d_s
                n += 1
            losses["dense"] = dloss / max(1, n)                 # logged as the unweighted mean
            total = total + ramp * dweighted
        else:
            losses["dense"] = total.detach() * 0.0

        # Prototype assignment loss (present only when the model uses prototypes).
        if "proto_logits_s1" in out and lc.lambda_proto > 0:
            proto = ref.sum() * 0.0
            for s in _SCALES:
                logits = out[f"proto_logits_{s}"]
                labels = out[f"proto_labels_{s}"]
                proto = proto + F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
            losses["proto"] = proto
            total = total + lc.lambda_proto * proto
        else:
            losses["proto"] = total.detach() * 0.0

        mode = getattr(lc, "anticollapse", "sigreg")
        use_vicreg = mode in ("vicreg", "both")
        use_sigreg = mode in ("sigreg", "both")

        # VICReg variance/covariance are always computed as collapse monitors; they are added to the
        # loss only under vicreg/both and run without autograd otherwise.
        var_active = use_vicreg and lc.lambda_var > 0
        cov_active = use_vicreg and getattr(lc, "lambda_cov", 0.0) > 0

        # Variance hinge / diagnostic.
        var = ref.sum() * 0.0
        with torch.no_grad() if not var_active else _nullcontext():
            for s in _MONITOR_SCALES:
                if f"student_{s}" in out:
                    var = var + variance_loss(out[f"student_{s}"], lc.var_target, lc.var_eps)
        losses["var"] = var
        if var_active:
            total = total + lc.lambda_var * var

        # Covariance decorrelation / diagnostic.
        cov = ref.sum() * 0.0
        with torch.no_grad() if not cov_active else _nullcontext():
            for s in _MONITOR_SCALES:
                if f"student_{s}" in out:
                    cov = cov + covariance_loss(out[f"student_{s}"], getattr(lc, "cov_max_tokens", 4096))
        losses["cov"] = cov
        if cov_active:
            total = total + lc.lambda_cov * cov

        # Scale-invariant correlation monitor averaged over scales, in [0, 1]; diagnostic only.
        with torch.no_grad():
            corr = ref.sum() * 0.0
            n_corr = 0
            for s in _MONITOR_SCALES:
                if f"student_{s}" in out:
                    corr = corr + feature_correlation(out[f"student_{s}"], getattr(lc, "cov_max_tokens", 4096))
                    n_corr += 1
            corr = corr / max(1, n_corr)
        losses["corr"] = corr

        # SIGReg is averaged over scales so lambda_sigreg keeps LeJEPA's per-application calibration.
        if use_sigreg and getattr(lc, "lambda_sigreg", 0.0) > 0:
            sig = ref.sum() * 0.0
            n_sig = 0
            proj_dim = getattr(lc, "sigreg_proj_dim", 0)
            for s in _MONITOR_SCALES:
                # Regularize the SIGReg projector output when proj_dim > 0 (raises if it is missing),
                # otherwise the raw latents.
                if proj_dim > 0:
                    if f"sigreg_proj_{s}" not in out:
                        if f"student_{s}" in out:
                            raise RuntimeError(
                                f"sigreg_proj_dim={proj_dim}>0 but out['sigreg_proj_{s}'] is missing. The "
                                f"model must emit SIGReg projector outputs; set sigreg_proj_dim=0 to apply "
                                f"SIGReg to the raw latents.")
                        continue
                    feats = out[f"sigreg_proj_{s}"]
                else:
                    feats = out.get(f"student_{s}")
                if feats is not None:
                    sig = sig + sigreg_loss(feats, lc.sigreg_n_proj, lc.sigreg_n_freq,
                                            getattr(lc, "sigreg_t_max", 3.0),
                                            getattr(lc, "sigreg_max_tokens", 512))
                    n_sig += 1
            sig = sig / max(1, n_sig)
            losses["sigreg"] = sig
            total = total + lc.lambda_sigreg * sig
        else:
            losses["sigreg"] = total.detach() * 0.0

        losses["total"] = total
        return losses
