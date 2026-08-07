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

def _stem(name):
    """`train_1741_b_2` from either `...npz` (the CSV's object_id) or `...nii.gz` (the JSON's name).

    The two sources disagree on suffix, which is exactly the kind of mismatch that yields a lookup
    silently returning the -1 sentinel for every finding instead of failing.
    """
    n = str(name)
    for suf in (".nii.gz", ".npz", ".nii"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return n


def _entity_counts():
    """{(object_id, finding_index): entity_count} from the challenge JSON, or {} if unset.

    OPTIONAL, unlike the four required paths: only extent/censoring mechanisms read it, and nothing
    in the current training path does. Absent, every lookup returns -1 and a consumer that needs it
    must say so itself rather than silently training on a sentinel.

    Why it matters: `entity_count == 3` is the annotation CAP, not a count -- 29.5% of train
    findings pile up there -- so a volume target read off those masks is a lower bound, not a label.
    Built across ALL splits because the challenge JSON's split names need not match `rex_split`.
    """
    path = os.environ.get("JECT_CHALLENGE_JSON", "")
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    for studies in json.load(open(path)).values():
        for s in studies:
            oid = _stem(s.get("name", ""))
            for k, c in (s.get("entity_counts") or {}).items():
                out[(oid, int(k))] = int(c)
    return out


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
        self.ent = _entity_counts()
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
        # A DICT, not a tuple. Mechanisms that condition on the target's EXTENT (a predicted
        # threshold or predicted volume) need |Y| and entity_count alongside the mask, and growing a
        # positional tuple each time one arrives means re-editing every unpack site with a fresh
        # chance to mis-order it. Keys are additive; unpacks are not.
        return {"vol": vol,
                "q": self.embs[txt].float(),
                "gt": torch.from_numpy(gtm.astype(np.float32)),
                "gt_vox": int(gtm.sum()),                  # |Y| AFTER the crop, i.e. what the loss sees
                "entity_count": self.ent.get((_stem(oid), k), -1),
                "oid": oid, "k": k}

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
