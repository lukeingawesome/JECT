"""Train-split section target means for --mup_target_stats.

For each Matryoshka dim d, every section token of a train case is sliced to its
first d values and L2-normalised (as in the model), then averaged globally and
per canonical section. A section with no train tokens gets the global mean.

Output (torch.save):
  dims              {d: [d]}            global mean
  conditional_dims  {"section_name": {d: [11, d]}}
  condition_counts  {"section_name": [11]}
  condition_names   {"section_name": [...]}
  shrinkage, n, encoder, split, manifest_sha256, mean_norms, section_mean_norms

  python section_stats.py --manifest manifest.csv --sr_root targets/sr --out section_target_stats.pt
"""
from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd
import torch
import torch.nn.functional as F

from encode_chest2vec import SR_SUFFIX, stem_of
from sections import SECTIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with object_id, split; rows whose split starts with 'train' are used")
    ap.add_argument("--sr_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", nargs="+", type=int, default=[1024, 512, 256])
    ap.add_argument("--shrinkage_tau", type=float, default=0.0,
                    help="shrink each section mean towards the global mean with weight tau / (count + tau)")
    ap.add_argument("--encoder", default="chest2vec/chest2vec_4B", help="recorded in the output only")
    a = ap.parse_args()
    if a.shrinkage_tau < 0:
        raise SystemExit("--shrinkage_tau must be >= 0")

    df = pd.read_csv(a.manifest, usecols=["object_id", "split"])
    train = sorted({stem_of(o) for o, s in zip(df["object_id"], df["split"])
                    if str(s).lower().startswith("train")})
    files = [p for p in (os.path.join(a.sr_root, s + SR_SUFFIX) for s in train) if os.path.exists(p)]
    print(f"{len(train)} train cases, {len(files)} with a section sidecar", flush=True)

    dims = tuple(a.dims)
    pos = {name: k for k, name in enumerate(SECTIONS)}
    sums = {d: torch.zeros(d, dtype=torch.float64) for d in dims}
    sec_sums = {d: torch.zeros(len(SECTIONS), d, dtype=torch.float64) for d in dims}
    counts = torch.zeros(len(SECTIONS), dtype=torch.long)
    n = 0
    for f in files:
        sc = torch.load(f, map_location="cpu", weights_only=True)
        toks = sc.get("section_tokens")
        if toks is None:
            continue
        for j, name in enumerate(sc.get("section_names") or []):
            k = pos.get(str(name))
            if k is None or j >= toks.shape[0]:
                continue
            raw = toks[j].double()
            if raw.numel() < max(dims):
                raise SystemExit(f"{f}: token dim {raw.numel()} < {max(dims)}")
            for d in dims:
                x = F.normalize(raw[:d], dim=-1)
                sums[d] += x
                sec_sums[d][k] += x
            counts[k] += 1
            n += 1
    if n == 0:
        raise SystemExit("no section targets found")

    tau = float(a.shrinkage_tau)
    means = {d: (sums[d] / n).float() for d in dims}
    sec_means = {}
    for d in dims:
        out = torch.zeros(len(SECTIONS), d)
        for k in range(len(SECTIONS)):
            c = int(counts[k])
            m = (sec_sums[d][k] / c).float() if c else means[d]
            if tau > 0 and c > 0:
                m = ((c / (c + tau)) * m + (tau / (c + tau)) * means[d]).float()
            out[k] = m
        sec_means[d] = out

    with open(a.manifest, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    torch.save({
        "dims": means,
        "conditional_dims": {"section_name": sec_means},
        "condition_counts": {"section_name": counts},
        "condition_names": {"section_name": list(SECTIONS)},
        "shrinkage": {"section_name": tau},
        "n": int(n),
        "encoder": a.encoder,
        "split": "train",
        "manifest_sha256": sha,
        "mean_norms": {d: float(means[d].norm()) for d in dims},
        "section_mean_norms": {d: [float(v.norm()) for v in sec_means[d]] for d in dims},
    }, a.out)
    print("counts: " + ", ".join(f"{s}={int(counts[k])}" for k, s in enumerate(SECTIONS)))
    print(f"{n} section targets written to {a.out}")


if __name__ == "__main__":
    main()
