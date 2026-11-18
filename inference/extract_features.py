#!/usr/bin/env python
"""Extract per-case JECT encoder features from preprocessed npz cubes.

Input contract, per case: ``<case>.npz`` holding key ``ct``, a raw-HU int16 cube at 1.5 mm
isotropic spacing, 256^3, in the layout produced by ``preprocess/``. Each cube is centre-cropped
to the encoder input (224^3), clipped to the general window (id 2, -1000..1000 HU), scaled to
[0, 1] and encoded with ``window_id`` 2.

Output (``torch.save``):
    {"ids": [N case ids],
     "s3_mean": Tensor[N, 512],    mean of the stage-3 tokens (14^3 grid)
     "s4_mean": Tensor[N, 1024],   mean of the stage-4 tokens (7^3 grid)
     "meta": {...}}
With ``--tokens_dir``, the stage-3 and stage-4 token grids of every case are also written to
``<tokens_dir>/<id>.pt`` as float16 ``[N_tokens, C]`` tensors.

    python inference/extract_features.py --checkpoint encoder.pt --input cases.csv --out feats.pt
    python inference/extract_features.py --checkpoint encoder.pt --input cubes/ --out feats.pt --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.model import GENERAL_WINDOW_ID, load_pretrained  # noqa: E402
from encoder.preprocess import load_npz_volume  # noqa: E402


def list_cases(src: str, path_col: str, id_col: str):
    """Case ids and npz paths from a CSV (``path_col``, optional ``id_col``) or a directory."""
    if os.path.isdir(src):
        names = sorted(f for f in os.listdir(src) if f.endswith(".npz"))
        return [n[:-4] for n in names], [os.path.join(src, n) for n in names]
    df = pd.read_csv(src)
    if path_col not in df.columns:
        raise SystemExit(f"{src}: no '{path_col}' column (found {list(df.columns)})")
    paths = df[path_col].astype(str).tolist()
    if id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [Path(p).name[:-4] if p.endswith(".npz") else Path(p).name for p in paths]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{src}: case ids are not unique")
    return ids, paths


class Cubes(Dataset):
    def __init__(self, paths, input_size, key):
        self.paths, self.input_size, self.key = paths, input_size, key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, load_npz_volume(self.paths[i], self.input_size, key=self.key, window_id=GENERAL_WINDOW_ID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="encoder checkpoint (encoder.pt)")
    ap.add_argument("--input", required=True, help="CSV of npz paths, or a directory of .npz cubes")
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument("--path_col", default="path")
    ap.add_argument("--id_col", default="id", help="optional CSV column with case ids (default: file stem)")
    ap.add_argument("--npz_key", default="ct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default=None,
                    help="default: bf16 autocast on CUDA, fp32 on CPU")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tokens_dir", default=None, help="also save stage-3/4 tokens per case here")
    args = ap.parse_args()

    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    encoder, cfg, info = load_pretrained(args.checkpoint, stream="target")
    encoder = encoder.to(device).eval()
    ids, paths = list_cases(args.input, args.path_col, args.id_col)
    print(f"[encoder] input {cfg.input_size}, epoch {info['epoch']}, {precision} on {device} | {len(ids)} cases",
          flush=True)
    if args.tokens_dir:
        os.makedirs(args.tokens_dir, exist_ok=True)

    loader = DataLoader(Cubes(paths, cfg.input_size, args.npz_key), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    s3 = [None] * len(ids)
    s4 = [None] * len(ids)
    t0, done = time.time(), 0
    with torch.no_grad():
        for idx, vol in loader:
            vol = vol.to(device, dtype=torch.float32)
            wid = torch.full((len(idx),), GENERAL_WINDOW_ID, dtype=torch.long, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                out = encoder(vol, window_id=wid)
            e3, e4 = out["e3"].float(), out["e4"].float()
            m3, m4 = e3.mean(1).cpu(), e4.mean(1).cpu()
            for j, i in enumerate(idx.tolist()):
                s3[i], s4[i] = m3[j], m4[j]
                if args.tokens_dir:
                    torch.save({"s3": e3[j].half().cpu(), "s4": e4[j].half().cpu(),
                                "grid_s3": cfg.grid_s3, "grid_s4": cfg.grid_s4},
                               os.path.join(args.tokens_dir, f"{ids[i]}.pt"))
            done += len(idx)
            if done % (args.batch_size * 25) < args.batch_size or done == len(ids):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(ids)}  {rate:.2f} cases/s", flush=True)

    meta = {"checkpoint": os.path.basename(args.checkpoint), "epoch": info["epoch"],
            "input_size": tuple(cfg.input_size), "window_id": GENERAL_WINDOW_ID,
            "precision": precision, "npz_key": args.npz_key}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"ids": ids, "s3_mean": torch.stack(s3), "s4_mean": torch.stack(s4), "meta": meta}, args.out)
    print(f"[saved] {args.out}  s3_mean {tuple(torch.stack(s3).shape)}  s4_mean {tuple(torch.stack(s4).shape)}")


if __name__ == "__main__":
    main()
