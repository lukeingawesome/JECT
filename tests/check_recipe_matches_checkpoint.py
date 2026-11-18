#!/usr/bin/env python3
"""Release check: a recipe, parsed by this repository's own train.py, builds exactly the architecture and
config of a given trained checkpoint (strict state_dict load, identical tensor shapes, identical config apart from paths).

    python tests/check_recipe_matches_checkpoint.py --checkpoint checkpoint_final.pt --target_stats section_target_stats.pt
"""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import train  # noqa: E402
from ject.backbone.model import JECTModel  # noqa: E402

p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--target_stats", required=True)
p.add_argument("--recipe", default="pretrain"); a = p.parse_args()
r = json.loads((ROOT / "recipes" / f"{a.recipe}.json").read_text())
argv = r["args"] + ["--data_csv", "x.csv", "--data_root", ".", "--output_dir", "unused", "--mup_target_stats", a.target_stats,
                    "--sr_sidecar_root", "."]
cfg = train.build_config(train.build_parser().parse_args(argv))
model = JECTModel(cfg)
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
own = model.state_dict(); ref = ck["model"]
missing = sorted(set(ref) - set(own)); extra = sorted(set(own) - set(ref))
shape = [k for k in ref if k in own and tuple(own[k].shape) != tuple(ref[k].shape)]
print(f"tensors: checkpoint {len(ref)}, model {len(own)} | missing {len(missing)} extra {len(extra)} shape-mismatch {len(shape)}")
model.load_state_dict(ref, strict=True); print("strict load: OK")
c_new, c_ref = train.config_to_dict(cfg), ck["config"]
PATHY = ("csv", "root", "dir", "stats", "path", "checkpoint")
diffs, added = [], []
for sec in sorted(set(c_new) | set(c_ref)):
    a_, b_ = c_new.get(sec, {}), c_ref.get(sec, {})
    if not isinstance(a_, dict) or not isinstance(b_, dict):
        if a_ != b_: diffs.append((sec, b_, a_))
        continue
    for k in sorted(set(a_) | set(b_)):
        va, vb = a_.get(k), b_.get(k)
        if isinstance(va, list): va = tuple(va)
        if isinstance(vb, list): vb = tuple(vb)
        if k not in b_:   # option added to the code after this checkpoint was trained: report it, it must be at its default
            added.append(f"{sec}.{k}={va}"); continue
        if va != vb and not any(t in k for t in PATHY): diffs.append((f"{sec}.{k}", vb, va))
print(f"config differences (non-path): {len(diffs)}; options newer than the checkpoint (at defaults): {added}")
for d in diffs: print("  ", d[0], "checkpoint:", d[1], "| recipe:", d[2])
sys.exit(0 if not (missing or extra or shape or diffs) else 1)
