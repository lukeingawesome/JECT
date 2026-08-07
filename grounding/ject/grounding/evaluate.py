"""Threshold-swept Dice evaluation.

Dice is strongly threshold-dependent, so we sweep a fixed grid and report the best-threshold
mean Dice plus a per-sample oracle upper bound. The reported figure is therefore VAL-TUNED --
for a held-out number, fix the threshold on val and apply it unchanged to test.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
import ject.grounding.paths as paths


def dice_score(pred, gt):
    a = pred > 0; b = gt > 0; s = a.sum() + b.sum()
    return 0.0 if s == 0 else float(2 * (a & b).sum() / s)


EVAL_THRS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98)


@torch.no_grad()
def evaluate(model, loader, dev, rank0):
    model.eval()
    sums = {t: 0.0 for t in EVAL_THRS}; hits = {t: 0 for t in EVAL_THRS}; oracle = 0.0; tot = 0
    for batch in loader:                       # not `b`: the inner per-sample loop below owns that
        vol, q, gt = batch["vol"], batch["q"], batch["gt"]
        vol, q = vol.to(dev), q.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(vol, q, window_id=torch.full((vol.shape[0],), 2, device=dev),
                          ct_type_id=torch.zeros(vol.shape[0], dtype=torch.long, device=dev))  # non-contrast
        logit = F.interpolate(logit[:, None].float(), size=paths.T, mode="trilinear", align_corners=False)[:, 0]
        prob = torch.sigmoid(logit).cpu().numpy()                     # pred -> 224^3 for scoring
        gtn = gt.numpy()
        for b in range(vol.shape[0]):
            p = prob[b]; g = gtn[b] > 0; best_d = 0.0
            for t in EVAL_THRS:
                d = dice_score(p >= t, g)
                sums[t] += d; hits[t] += int(d >= 0.1); best_d = max(best_d, d)
            oracle += best_d; tot += 1
    model.train()
    n = max(tot, 1)
    md = {t: sums[t] / n for t in EVAL_THRS}
    best_t = max(md, key=md.get)
    if rank0:
        curve = "  ".join(f"{t:.2f}:{md[t]:.3f}" for t in EVAL_THRS)
        print(f"    [thr sweep] {curve}  | best_thr={best_t:.2f} oracle={oracle / n:.3f}", flush=True)
    return hits[best_t] / n, md[best_t], tot


