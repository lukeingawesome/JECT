"""Helpers shared by the CT-RATE and RAD-ChestCT evaluations."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def bare_id(x) -> str:
    """Strip a trailing .npz / .nii.gz so ids from different tables compare equal."""
    s = str(x)
    for ext in (".npz", ".nii.gz"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def load_features(path: str, key: str = "s4_mean") -> Dict[str, torch.Tensor]:
    """Read a feature file into ``{id: float32 vector}``.

    Accepted layouts:
      * ``{"ids": [...], "<key>": Tensor[N, D]}``         (inference/extract_features.py)
      * ``{"ids": [...], "embeddings": Tensor[N, D]}``    (one readout per file; ``key`` ignored)
      * ``{"feats": {id: {"<key>": Tensor[D]}}}``          (per-case dictionaries)
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "feats" in blob:
        return {bare_id(i): v[key].float() for i, v in blob["feats"].items()}
    mat = blob[key] if key in blob else blob.get("embeddings")
    if mat is None:
        raise KeyError(f"{path}: no '{key}' or 'embeddings' entry")
    if len(blob["ids"]) != len(mat):
        raise ValueError(f"{path}: {len(blob['ids'])} ids for {len(mat)} rows")
    return {bare_id(i): mat[k].float() for k, i in enumerate(blob["ids"])}


def merge_features(paths: Sequence[str], key: str) -> Dict[str, torch.Tensor]:
    """Union of several feature files; an id present twice must carry the same vector."""
    out: Dict[str, torch.Tensor] = {}
    for p in paths:
        for i, v in load_features(p, key).items():
            if i in out and not torch.equal(out[i], v):
                raise ValueError(f"id {i} has different features in {p}")
            out[i] = v
    return out


class LinearProbe(nn.Module):
    """One linear layer over a frozen feature vector."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def load_probe(path: str, dim: int):
    """Load a ``best_linear_probe.pt``; returns the probe in eval mode and its label columns."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    probe = LinearProbe(dim, len(ck["label_cols"]))
    probe.load_state_dict(ck["model_state_dict"])
    return probe.eval(), list(ck["label_cols"])


def macro_auroc(y: np.ndarray, p: np.ndarray, min_count: int = 1):
    """Mean AUROC over labels with at least ``min_count`` positives and negatives; returns (mean, n_labels)."""
    aucs = []
    for j in range(y.shape[1]):
        pos = y[:, j].sum()
        if pos >= min_count and len(y) - pos >= min_count:
            aucs.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(aucs)), len(aucs)


def cluster_draws(groups: Sequence, n_draws: int, seed: int) -> List[np.ndarray]:
    """Cluster bootstrap: resample groups (patients) with replacement, keep all their rows.

    ``groups[k]`` is the cluster key of row k. Groups are enumerated in order of first
    appearance, so the draws depend on the row order as well as on the seed.
    """
    members: Dict = {}
    for k, g in enumerate(groups):
        members.setdefault(g, []).append(k)
    lists = [np.asarray(v) for v in members.values()]
    rng = np.random.default_rng(seed)
    return [np.concatenate([lists[g] for g in rng.integers(0, len(lists), len(lists))])
            for _ in range(n_draws)]


def percentile_ci(values) -> List[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
