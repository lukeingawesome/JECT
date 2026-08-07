"""Loss terms for phrase grounding.

Tversky+BCE is the overlap objective. The ranking terms (mil_peak, voxel_infonce) exist because
a set-overlap loss is nearly invariant to WHERE the probability mass sits once the prediction is
near-empty: measured on val, the peak lands outside the lesion for 73% of missed findings while a
volume-oracle buys only +0.012 Dice -- a ranking failure, not a thresholding one.

SIZE RANKING TERMS BY GRADIENT, NOT LOSS VALUE. They act on raw logits over a sparse support, so
their gradient share vastly exceeds their loss share: at weight 0.1, voxel_infonce contributed
22% of the loss value but 99.9% of the gradient norm on the logit.
"""
from __future__ import annotations
import torch, torch.nn.functional as F


def mil_peak_loss(logit, gt, k=32, margin=1.0, eps=1e-6):
    """Max-in / max-out ranking hinge with top-k pooling inside the target.

    Nothing in Tversky/BCE penalises a confident BACKGROUND peak: a 320-voxel target is 0.003% of
    224^3, so the objective is invariant to whether the global argmax lands inside the mask. Measured:
    the peak is OUTSIDE the lesion for 73% of findings, while a volume-oracle buys only +0.012 -- a
    ranking failure, not a thresholding one. Venice-H1 finds the same pathology dominates RIS error
    (7-18% of samples, 40-68% of total error).

        L = relu( max_{v not in M} S_v  -  mean-topk_{v in M} S_v  +  margin )

    TOP-K inside, not single-max: TKP-PCL's headline ablation is that single-max is dominated by one
    outlier voxel; with ~320-voxel targets a k~32 mean is far more stable. Operates on raw logits so
    the margin is in logit units. Empty targets are skipped.
    """
    B = logit.shape[0]
    S = logit.flatten(1)
    g = gt.flatten(1) > 0.5
    has = g.any(dim=1)
    if not has.any():
        return logit.sum() * 0.0
    neg = S.masked_fill(g, float("-inf")).amax(dim=1)          # best background score
    kk = int(max(1, min(k, S.shape[1])))
    top = S.masked_fill(~g, float("-inf")).topk(kk, dim=1).values
    # PER-SAMPLE k: a batch-min k would collapse to 1 whenever any finding in the batch is a single
    # voxel, silently degenerating to the single-max form this is meant to replace.
    valid = torch.isfinite(top)
    pos = (top.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp(min=1))
    return F.relu(neg[has] - pos[has] + margin).mean()


def attn_kl_loss(simmap, gt, eps=1e-8):
    """KL(normalised GT || softmax over positions of the text-image similarity map).

    A dense localisation signal on 100% of voxels, unlike Dice which is dominated by the 99.997%
    background. Reported to 'promote separation between the referred object and spatially distant
    repetitive objects that are semantically similar' -- i.e. anatomical look-alikes.
    """
    B = simmap.shape[0]
    a = F.log_softmax(simmap.flatten(1), dim=1)
    g = gt.flatten(1).float()
    tot = g.sum(dim=1, keepdim=True)
    has = (tot > 0).squeeze(1)
    if not has.any():
        return simmap.sum() * 0.0
    P = g / tot.clamp(min=1.0)
    return (P[has] * (torch.log(P[has] + eps) - a[has])).sum(dim=1).mean()


def voxel_infonce_loss(logit, gt, n_pos=512, n_neg=4096, tau=0.1, eps=1e-8):
    """Voxel-level InfoNCE: make in-target voxels OUT-RANK background. Measured motivation --
    on the 28.2% of val findings we miss, the peak is inside the target only 14% of the time and a
    median 979 background voxels out-rank the best in-target voxel. Tversky/Dice cannot fix that:
    it is a SET-OVERLAP loss, so for a near-zero prediction (tp~0, fp~0, fn~|g|) its value is
    essentially identical whether the top activations sit on the lesion or on the opposite lung. It
    carries almost no ranking information; this term is pure ranking.

        L = -[ logsumexp(s_pos/tau) - logsumexp(cat(s_pos, s_neg)/tau) ]

    A strict generalisation of mil_peak_loss: softmax over MANY sampled negatives instead of the
    single max, so it is not dominated by one outlier voxel. Positives and negatives are subsampled
    (targets range 12 -> 78k voxels, and 224^3 background is 11.2M) which also keeps it O(n_pos+n_neg)
    per sample. Empty targets are skipped.
    """
    B = logit.shape[0]
    S = logit.flatten(1).float() / tau
    G = gt.flatten(1) > 0.5
    terms = []
    for b in range(B):
        pi = G[b].nonzero(as_tuple=True)[0]
        if pi.numel() == 0:
            continue
        ni = (~G[b]).nonzero(as_tuple=True)[0]
        if ni.numel() == 0:
            continue
        if pi.numel() > n_pos:
            pi = pi[torch.randint(pi.numel(), (n_pos,), device=pi.device)]
        ni = ni[torch.randint(ni.numel(), (min(n_neg, ni.numel()),), device=ni.device)]
        sp, sn = S[b][pi], S[b][ni]
        terms.append(-(torch.logsumexp(sp, 0) - torch.logsumexp(torch.cat([sp, sn]), 0)))
    if not terms:
        return logit.sum() * 0.0
    return torch.stack(terms).mean()


def tversky_bce_loss(logit, gt, alpha=0.3, beta=0.7, eps=1.0, pos_weight=50.0):
    """Tversky + weighted BCE. NOTE alpha/beta/pos_weight are now CLI-configurable: the defaults
    stack TWO recall biases (beta>alpha AND pos_weight=50), which no surveyed system does, and the
    measured precision 0.172 / recall 0.493 is exactly that signature."""
    p = torch.sigmoid(logit).flatten(1); g = gt.flatten(1)
    tp = (p * g).sum(1); fp = (p * (1 - g)).sum(1); fn = ((1 - p) * g).sum(1)
    tversky = (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()
    bce = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=torch.tensor(pos_weight, device=logit.device))
    return tversky + bce


