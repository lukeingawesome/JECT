#!/usr/bin/env python
"""RAD-ChestCT label efficiency: frozen features, 0-100 % of the training labels.

Labels: the 16 CT-RATE-18 abnormalities that RAD-ChestCT can express (all but 'Medical material'
and 'Lymphadenopathy'), mapped onto RAD-ChestCT findings (``data/label_map_ctrate18_to_radchest16.json``;
a label is positive if any mapped finding is). Every fit is scored once on radchest_test (360 scans).

  0 %    no RAD-ChestCT labels: the model's five CT-RATE-18 linear probes (seeds 41-45, as written
         by evaluate/ctrate_probe/run_probe.py) applied unchanged and averaged; only entries
         given a probe name get this point.
  f > 0  a linear head fitted on a patient-level subset of radchest_train (f % of its patients,
         all their scans): z-scored features, class-balanced BCE (positive weight capped at 50),
         AdamW, 50 epochs, epoch picked on the full radchest_valid (984 scans, never subsampled),
         seeds 41-43 averaged. 5 repeats per fraction (1 at 100 %), identical subsets for every entry.
  metric macro AUROC over labels with >= 5 positives and >= 5 negatives, averaged over repeats;
         95 % CI and paired deltas against ``--ours`` from a patient-cluster bootstrap whose draws
         are shared across entries and fractions.

    python evaluate/radchest/label_efficiency.py --cache_dir feats/ --probe_dir probes/ct-rate-18 \
        --subsets_in evaluate/radchest/ids/train_subsets_seed3026.json --ours mine \
        --entries mine=radchest_feats.pt:s4_mean:mine other=other_feats.pt:emb --out results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bare_id, cluster_draws, load_features, load_probe, macro_auroc, percentile_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_COUNT = 5
PROBE_SEEDS = (41, 42, 43, 44, 45)
FIT_SEEDS = (41, 42, 43)


def macro(y, p):
    return macro_auroc(y, p, min_count=MIN_COUNT)


def read_labels(path):
    """Split and MRN_DEID per object_id, and the 16-label table."""
    df = pd.read_csv(path)
    df.index = df.object_id.map(bare_id)
    return df[["split", "MRN_DEID"]], df.iloc[:, 3:].astype(int)


def zero_shot_probs(probe_dir, probe_name, X, label_names):
    """Mean sigmoid of the CT-RATE-18 probes, restricted to the RAD-ChestCT label columns."""
    outs, cols = [], None
    for s in PROBE_SEEDS:
        probe, cols = load_probe(os.path.join(probe_dir, f"{probe_name}_s{s}", "best_linear_probe.pt"), X.shape[1])
        with torch.inference_mode():
            outs.append(torch.sigmoid(probe(torch.from_numpy(X))).numpy())
    return np.mean(outs, 0)[:, [cols.index(k) for k in label_names]]


def fit_probe(Xtr, Ytr, Xva, Yva, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    head = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    pos = torch.tensor(Ytr, dtype=torch.float32).sum(0)
    pw = ((len(Ytr) - pos) / pos.clamp(min=1)).clamp(max=50)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state = -1, None
    for _ in range(50):
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 128):
            i = perm[b:b + 128]
            opt.zero_grad()
            lossf(head(xt[i]), yt[i]).backward()
            opt.step()
        with torch.inference_mode():
            m, _ = macro(Yva, torch.sigmoid(head(xv)).numpy())
        if m > best:
            best, best_state = m, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return lambda X: torch.sigmoid(head(torch.tensor((X - mu) / sd, dtype=torch.float32))).detach().numpy()


def fit_predict(Xtr, Ytr, Xva, Yva, Xte):
    return np.mean([fit_probe(Xtr, Ytr, Xva, Yva, s)(Xte) for s in FIT_SEEDS], 0)


def draw_subsets(meta, pos, fractions, repeats, seed):
    """Patient-level training subsets drawn from the full radchest_train patient list."""
    mt = meta[meta.split == "radchest_train"]
    patients = np.unique(mt.MRN_DEID.values)
    rng = np.random.default_rng(seed)
    subsets = {}
    for f in fractions:
        if f <= 0:
            continue
        subsets[f] = []
        for _ in range(1 if f >= 100 else repeats):
            k = max(1, int(round(len(patients) * f / 100.0)))
            chosen = set(rng.choice(patients, size=k, replace=False)) if f < 100 else set(patients)
            subsets[f].append(np.array([pos[o] for o, p in zip(mt.index, mt.MRN_DEID.values)
                                        if p in chosen and o in pos]))
    return subsets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="+", required=True,
                    help="name=feature_file:key[:ctrate18_probe_name]; feature_file is relative to --cache_dir")
    ap.add_argument("--ours", required=True, help="entry paired against every other entry")
    ap.add_argument("--cache_dir", default=".", help="directory of the feature files")
    ap.add_argument("--probe_dir", default=".", help="directory of <probe_name>_s41..45/best_linear_probe.pt")
    ap.add_argument("--labels", default=str(HERE / "data" / "radchest_labels16.csv"))
    ap.add_argument("--fractions", nargs="+", type=float, default=[0, 1, 2, 5, 10, 25, 50, 100])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--boot_seed", type=int, default=5150)
    ap.add_argument("--seed", type=int, default=3026, help="subset draw seed (ignored with --subsets_in)")
    ap.add_argument("--subsets_in", default="", help="json of training-subset object_ids per fraction")
    ap.add_argument("--subsets_out", default="", help="write the training subsets used to this json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta, Y16 = read_labels(a.labels)
    label_names = list(Y16.columns)
    t0 = time.time()
    spec, feats = {}, {}
    for e in a.entries:
        name, rest = e.split("=", 1)
        parts = rest.split(":")
        spec[name] = (parts[0], parts[1], parts[2] if len(parts) > 2 else None)
        feats[name] = load_features(os.path.join(a.cache_dir, parts[0]), parts[1])
    ids = [i for i in meta.index if all(i in feats[n] for n in spec)]
    m, Y = meta.loc[ids], Y16.loc[ids].values
    tr, va, te = ((m.split == s).values for s in ("radchest_train", "radchest_valid", "radchest_test"))
    X = {n: torch.stack([feats[n][i] for i in ids]).numpy().astype(np.float32) for n in spec}
    Yte, mrn_te = Y[te], m.MRN_DEID.values[te]
    print(f"[data] {len(ids)} scans: train {tr.sum()} ({len(np.unique(m.MRN_DEID.values[tr]))} patients), "
          f"valid {va.sum()}, test {te.sum()} | entries {list(spec)}", flush=True)

    pos = {i: k for k, i in enumerate(ids)}
    if a.subsets_in:
        raw = json.load(open(a.subsets_in))
        subsets = {float(f): [np.array([pos[bare_id(o)] for o in rep if bare_id(o) in pos]) for rep in reps]
                   for f, reps in raw.items()}
    else:
        subsets = draw_subsets(meta, pos, a.fractions, a.repeats, a.seed)
    if a.subsets_out:
        json.dump({str(f): [[ids[k] for k in rep] for rep in reps] for f, reps in subsets.items()},
                  open(a.subsets_out, "w"))

    P = {}   # P[f][name] = test probabilities, one array per repeat
    for f in a.fractions:
        P[f] = {}
        for n, (_, _, probe_name) in spec.items():
            if f <= 0:
                if probe_name:
                    P[f][n] = [zero_shot_probs(a.probe_dir, probe_name, X[n][te], label_names)]
            else:
                P[f][n] = [fit_predict(X[n][idx], Y[idx], X[n][va], Y[va], X[n][te]) for idx in subsets[f]]
        pts = {n: np.mean([macro(Yte, p)[0] for p in ps]) for n, ps in P[f].items()}
        print(f"[f={f:g}%] " + " | ".join(f"{n} {v:.4f}" for n, v in pts.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    draws = cluster_draws(mrn_te, a.n_boot, a.boot_seed)
    res = {"recipe": {"repeats": a.repeats, "n_boot": a.n_boot, "boot_seed": a.boot_seed, "seed": a.seed,
                      "n_train_scans": int(tr.sum()), "n_test": int(te.sum())},
           "subset_sizes": {str(f): [int(len(s)) for s in ss] for f, ss in subsets.items()}, "fractions": {}}
    for f in a.fractions:
        B = {n: np.array([np.mean([macro(Yte[d], p[d])[0] for p in ps]) for d in draws]) for n, ps in P[f].items()}
        ent = {n: {"macro_auroc": float(np.mean([macro(Yte, p)[0] for p in ps])),
                   "repeat_sd": float(np.std([macro(Yte, p)[0] for p in ps])), "ci95": percentile_ci(B[n])}
               for n, ps in P[f].items()}
        pairs = {}
        if a.ours in B:
            for n in B:
                if n == a.ours:
                    continue
                lo, hi = percentile_ci(B[a.ours] - B[n])
                pairs[f"{a.ours} - {n}"] = {"delta": ent[a.ours]["macro_auroc"] - ent[n]["macro_auroc"],
                                            "ci95": [lo, hi], "separated": bool(lo > 0 or hi < 0)}
        res["fractions"][f"{f:g}"] = {"entries": ent, "pairs": pairs}
        print(f"[boot f={f:g}%] " + " | ".join(f"{n} {v['macro_auroc']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
                                               for n, v in ent.items()), flush=True)
        for k, v in pairs.items():
            print(f"    {k} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                  f"{' separated' if v['separated'] else ''}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[saved] {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
