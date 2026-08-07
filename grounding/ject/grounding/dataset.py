"""Dataset, distributed setup and config loading for JECT phrase grounding."""
from __future__ import annotations
import os, json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
import torch.distributed as dist

from ject.config.art import ARTJEPAConfig
from ject.data import make_crop_pad_transform, apply_crop_pad_3d
import ject.grounding.paths as paths   # module, not `from ... import DATA` (binds by value)


def gwin(ct):
    return np.clip((ct.astype(np.float32) + 1000.0) / 2000.0, 0.0, 1.0)

class RexGrounding(Dataset):
    def __init__(self, split, embs, augment=False):
        df = pd.read_csv(paths.REX_CSV); df = df[df["rex_split"] == split]
        self.items = []
        for _, r in df.iterrows():
            oid = str(r["object_id"])
            for k, txt in json.loads(r["rex_findings"]).items():
                if txt in embs:
                    self.items.append((oid, int(k), txt))
        self.embs = embs; self.augment = augment
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        oid, k, txt = self.items[i]
        p = os.path.join(paths.DATA, oid);  p = p if os.path.exists(p) else p + ".npz"
        z = np.load(p); ct = z["ct"]; rex = z["rex"]
        if ct.ndim == 4: ct = ct[0]
        if rex.ndim == 5: rex = rex[:, 0]
        gtm = (rex[k] > 0).astype(np.float32) if k < rex.shape[0] else np.zeros_like(ct, np.float32)
        # augmentation: random crop only (train). NO SPATIAL FLIPS for text-conditioned grounding: the
        # phrase embedding is a FIXED pooled Chest2Vec vector, so mirroring the CT+mask (LPS: axis0 sup↔inf
        # = "upper/lower lobe", axis1 L↔R = laterality, axis2 ant↔post) while keeping the phrase would train
        # directional phrases against the WRONG-side mask → phrase-invariant / wrong-location grounding. A
        # random crop is a translation (laterality/direction preserved), so it is safe and kept.
        sl, pd_ = make_crop_pad_transform(tuple(ct.shape), paths.T, random_crop=self.augment)
        ct = apply_crop_pad_3d(ct, sl, pd_, constant=-1000.0)
        gtm = apply_crop_pad_3d(gtm, sl, pd_, constant=0)
        vol = torch.from_numpy(gwin(ct)).float().unsqueeze(0)
        return vol, self.embs[txt].float(), torch.from_numpy(gtm.astype(np.float32))

def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def build_cfg(init_ckpt):
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    cfg = ARTJEPAConfig()
    for sub, d in ck.get("config", {}).items():
        if hasattr(cfg, sub) and isinstance(d, dict):
            t = getattr(cfg, sub)
            for kk, vv in d.items():
                if hasattr(t, kk): setattr(t, kk, tuple(vv) if isinstance(vv, list) else vv)
    cfg.__post_init__()
    return cfg, ck


# Dice is highly threshold-dependent (CEPA's best Dice needs ~0.75-0.8). Sweep a fixed grid and report the
# best-threshold mean-Dice, plus the per-sample oracle as an upper bound, instead of one heuristic threshold.
