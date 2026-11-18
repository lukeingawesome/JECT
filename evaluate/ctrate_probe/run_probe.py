#!/usr/bin/env python
"""CT-RATE linear-probe benchmark: train on CT-RATE train features, score on validation.

Two label sets, both scored on the same validation cohort:
  ct-rate-18     the 18 CT-RATE abnormality labels
  chest2vec-44   44 report-derived findings (the "_others" columns and the study-limitation
                 column of the label file are not targets)

Cohorts, built from the label files alone:
  validation     one scan per CT-RATE validation study: the lowest-sorted scan that has features.
                 When the chest2vec-44 file is present, both label sets use its 1,544 validation
                 scans instead, so the two scores refer to the same scans
  train          chest2vec-44: every train row; CT-RATE-18: the lowest-sorted scan of each study

Per label set: five linear probes (seeds 41-45), each trained for 50 epochs and kept at the
epoch with the best validation macro AUROC; the reported score is the macro AUROC of the mean
sigmoid of the five, with a 95 % CI from a patient-cluster bootstrap (CT-RATE patient number,
3,000 draws).

    python evaluate/ctrate_probe/run_probe.py --features feats_train.pt feats_val.pt \
        --key s4_mean --name ject_s4 --out_dir probes/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (LinearProbe, bare_id, cluster_draws, load_probe, macro_auroc,  # noqa: E402
                    merge_features, percentile_ci)

HERE = Path(__file__).resolve().parent
TASKS = {
    "ct-rate-18": {"csv": "ct_rate_labels.csv", "n_labels": 18},
    "chest2vec-44": {"csv": "ct_rate_chest2vec_probe.csv", "n_labels": 44},
}
META_COLS = {"object_id", "VolumeName", "split"}
CHEST2VEC_EXCLUDED = {"Study limitation / limited evaluation (non-motion)"}


def study_of(scan_id: str) -> str:
    return scan_id.rsplit("_", 1)[0]


def patient_of(scan_id: str) -> str:
    m = re.match(r"^(?:valid|train)_(\d+)_", scan_id)
    return m.group(1) if m else scan_id


def read_labels(task: str, labels_dir: Path):
    """Label table indexed by scan id (split and label columns) and the label column names."""
    df = pd.read_csv(labels_dir / TASKS[task]["csv"])
    df.index = df.object_id.map(bare_id)
    cols = [c for c in df.columns if c not in META_COLS]
    if task == "chest2vec-44":
        cols = [c for c in cols if not c.endswith("_others") and c not in CHEST2VEC_EXCLUDED]
    if len(cols) != TASKS[task]["n_labels"]:
        raise ValueError(f"{task}: expected {TASKS[task]['n_labels']} label columns, found {len(cols)}")
    # uncertain (-1) and missing labels count as negative
    df[cols] = df[cols].fillna(0).clip(lower=0).astype(np.float32)
    return df, cols


def build_cohorts(labels_dir: Path, available=None):
    """Per task: (train ids, validation ids, label columns, label table).

    Train ids are sorted. Validation ids keep the row order of the CT-RATE-18 file, which fixes
    the order patients enter the bootstrap and so the draws.
    """
    frames = {t: read_labels(t, labels_dir) for t in TASKS if (labels_dir / TASKS[t]["csv"]).exists()}
    if "ct-rate-18" not in frames:
        raise FileNotFoundError(f"{labels_dir / TASKS['ct-rate-18']['csv']} is required")
    c18, _ = frames["ct-rate-18"]
    if "chest2vec-44" in frames:
        c44, _ = frames["chest2vec-44"]
        c44_val = set(c44.index[c44.split == "val"])
        val = [i for i in c18.index[c18.split == "val"] if i in c44_val]
        if len(val) != len(c44_val):
            raise ValueError("chest2vec-44 validation scans are not all CT-RATE-18 validation rows")
    else:
        # without the chest2vec-44 file: per validation study, the lowest-sorted scan that has features
        first = {}
        for i in sorted(c18.index[c18.split == "val"]):
            if available is None or i in available:
                first.setdefault(study_of(i), i)
        keep = set(first.values())
        val = [i for i in c18.index[c18.split == "val"] if i in keep]
    out = {}
    for task, (df, cols) in frames.items():
        train = sorted(df.index[df.split == "train"])
        if task == "ct-rate-18":
            first, seen = [], set()
            for i in train:
                if study_of(i) not in seen:
                    seen.add(study_of(i))
                    first.append(i)
            train = first
        missing = [i for i in val if i not in df.index]
        if missing:
            raise ValueError(f"{task}: {len(missing)} validation scans have no labels")
        out[task] = (train, val, cols, df)
    return out


def stack(ids, feats, df, cols):
    x = torch.stack([feats[i] for i in ids]).float()
    y = torch.from_numpy(df.loc[ids, cols].to_numpy(np.float32))
    return x, y


@torch.no_grad()
def val_macro_auroc(model, x, y):
    model.eval()
    logits = np.concatenate([model(x[b:b + 2048]).numpy() for b in range(0, len(x), 2048)])
    return macro_auroc(y, 1.0 / (1.0 + np.exp(-logits)))[0]


def draw_seed() -> int:
    return int(torch.empty((), dtype=torch.int64).random_().item())


def train_probe(xtr, ytr, xva, yva, seed, epochs, threads):
    """Class-balanced BCE, Adam with cosine decay, epoch picked by validation macro AUROC."""
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LinearProbe(xtr.shape[1], ytr.shape[1])
    pos = ytr.sum(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(ytr) - pos) / pos.clamp_min(1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    yva_np = yva.numpy()
    best_auc, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        # Five draws per epoch from the seeded global generator, the fourth seeding the shuffle.
        # This is the generator stream of the trainer behind the published probes, so seeds
        # 41-45 reproduce them bit for bit.
        shuffle_seed = [draw_seed() for _ in range(5)][3]
        perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(shuffle_seed))
        model.train()
        for b in range(0, len(xtr), 512):
            i = perm[b:b + 512]
            opt.zero_grad()
            criterion(model(xtr[i]), ytr[i]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        auc = val_macro_auroc(model, xva, yva_np)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return seed, best_epoch, best_auc, best_state


def run_task(task, cohort, feats, args):
    train_ids, val_ids, cols, df = cohort
    absent = [i for i in val_ids if i not in feats]
    if absent:
        raise SystemExit(f"{task}: {len(absent)} of {len(val_ids)} validation scans have no features")
    kept = [i for i in train_ids if i in feats]
    print(f"[{task}] train {len(kept)} of {len(train_ids)} scans with features, "
          f"val {len(val_ids)}, {len(cols)} labels", flush=True)
    xtr, ytr = stack(kept, feats, df, cols)
    xva, yva = stack(val_ids, feats, df, cols)

    task_dir = Path(args.out_dir) / task
    jobs = [(xtr, ytr, xva, yva, s, args.epochs, args.threads) for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
        fits = list(pool.map(train_probe, *zip(*jobs)))
    per_seed = {}
    for seed, epoch, auc, state in fits:
        d = task_dir / f"{args.name}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": state, "label_cols": cols, "epoch": epoch,
                    "macro_auc": auc}, d / "best_linear_probe.pt")
        per_seed[seed] = {"best_epoch": epoch, "val_macro_auroc": auc}
        print(f"  seed {seed}: best epoch {epoch}, macro AUROC {auc:.4f}", flush=True)

    probs = []
    for seed in args.seeds:
        probe, _ = load_probe(str(task_dir / f"{args.name}_s{seed}" / "best_linear_probe.pt"), xva.shape[1])
        with torch.inference_mode():
            probs.append(torch.sigmoid(probe(xva)).numpy())
    p = np.mean(probs, axis=0)
    y = yva.numpy()
    point, _ = macro_auroc(y, p)
    auprc = float(np.mean([average_precision_score(y[:, j], p[:, j])
                           for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]))
    draws = cluster_draws([patient_of(i) for i in val_ids], args.n_boot, args.boot_seed)
    boot = np.array([macro_auroc(y[d], p[d])[0] for d in draws])
    per_label = {c: (float(roc_auc_score(y[:, j], p[:, j])) if 0 < y[:, j].sum() < len(y) else None)
                 for j, c in enumerate(cols)}
    ci = percentile_ci(boot)
    print(f"[{task}] ensemble macro AUROC {point:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  AUPRC {auprc:.4f}  "
          f"({len(set(map(patient_of, val_ids)))} patients, {args.n_boot} draws)", flush=True)
    return {"n_train": len(kept), "n_val": len(val_ids), "macro_auroc": point, "ci95": ci,
            "macro_auprc": auprc, "per_seed": per_seed, "per_label_auroc": per_label,
            "bootstrap": {"cluster": "patient", "n_draws": args.n_boot, "seed": args.boot_seed,
                          "macro_auroc_per_draw": boot.tolist()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True,
                    help="feature files covering the train and validation scans (merged by id)")
    ap.add_argument("--key", default="s4_mean", help="readout to probe, e.g. s4_mean or s3_mean")
    ap.add_argument("--name", default="model", help="prefix of the saved probe folders")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--labels_dir", default=str(HERE / "labels"))
    ap.add_argument("--tasks", nargs="+", default=["ct-rate-18"], choices=list(TASKS),
                    help="label sets to score (chest2vec-44 needs ct_rate_chest2vec_probe.csv in --labels_dir)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--boot_seed", type=int, default=3026)
    ap.add_argument("--jobs", type=int, default=3, help="probes trained in parallel")
    ap.add_argument("--threads", type=int, default=8, help="torch threads per probe")
    args = ap.parse_args()

    t0 = time.time()
    feats = merge_features(args.features, args.key)
    print(f"[features] {len(feats)} scans, dim {next(iter(feats.values())).numel()}", flush=True)
    cohorts = build_cohorts(Path(args.labels_dir), set(feats))
    results = {"key": args.key, "name": args.name, "seeds": args.seeds, "epochs": args.epochs, "tasks": {}}
    for task in args.tasks:
        if task not in cohorts:
            raise SystemExit(f"{task}: {Path(args.labels_dir) / TASKS[task]['csv']} not found")
        results["tasks"][task] = run_task(task, cohorts[task], feats, args)
    os.makedirs(args.out_dir, exist_ok=True)
    out = Path(args.out_dir) / f"{args.name}_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[saved] {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
