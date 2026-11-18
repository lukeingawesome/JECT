#!/usr/bin/env python3
"""Launch a released JECT training recipe with your own data paths.

    python launch.py --recipe pretrain      --data_csv manifest.csv --data_root cubes/ --sr_root sections/ \
        --target_stats section_target_stats.pt --out runs/pretrain --gpus 0,1
    python launch.py --recipe cooldown_320 --init_checkpoint runs/pretrain/checkpoint_final.pt --data_csv manifest_384.csv \
        --data_root cubes_384/ --sr_root sections/ --target_stats section_target_stats.pt --out runs/cooldown --gpus 0,1

Recipes live in recipes/*.json (the exact argument lists used for the released models, data paths removed).
--smoke runs 24 steps of one epoch to check memory and data before a full run; --dry-run only prints the command.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--recipe", required=True, choices=sorted(x.stem for x in (ROOT / "recipes").glob("*.json")))
p.add_argument("--data_csv", required=True, help="manifest: object_id, split and the text columns (see README)")
p.add_argument("--data_root", required=True, help="directory of <object_id>.npz cubes")
p.add_argument("--sr_root", required=True, help="directory of per-case section/report target sidecars")
p.add_argument("--target_stats", required=True, help="train-split section target statistics (.pt)")
p.add_argument("--out", required=True, help="new, empty output directory")
p.add_argument("--gpus", required=True, help="e.g. 0,1")
p.add_argument("--init_checkpoint", help="full training checkpoint to continue from (required by cooldown recipes)")
p.add_argument("--smoke", action="store_true"); p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
r = json.loads((ROOT / "recipes" / f"{a.recipe}.json").read_text()); args = list(r["args"])
if r["needs_init_checkpoint"] and not a.init_checkpoint: p.error(f"{a.recipe} needs --init_checkpoint (a full training checkpoint)")
for path in [a.data_csv, a.data_root, a.sr_root, a.target_stats] + ([a.init_checkpoint] if a.init_checkpoint else []):
    if not Path(path).exists(): p.error(f"not found: {path}")
def setarg(k, v):
    if k in args: args[args.index(k) + 1] = str(v)
    else: args.extend([k, str(v)])
for k, v in [("--data_csv", a.data_csv), ("--data_root", a.data_root), ("--sr_sidecar_root", a.sr_root),
             ("--mup_target_stats", a.target_stats), ("--output_dir", a.out)]:
    setarg(k, Path(v).resolve())
if a.init_checkpoint: setarg("--init_checkpoint", Path(a.init_checkpoint).resolve())
n = len(a.gpus.split(","))
if a.smoke:
    setarg("--epochs", 1); setarg("--max_steps_per_epoch", 24); setarg("--log_interval", 1)
elif n != r["gpus"]:
    p.error(f"{a.recipe} is specified for {r['gpus']} GPUs (the effective batch depends on it); adjust --grad_accum in the recipe explicitly for {n}")
if Path(a.out).exists() and any(Path(a.out).iterdir()): p.error("choose a new, empty --out")
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", str(ROOT / "train.py"), *args]
print(json.dumps({"recipe": a.recipe, "description": r["description"], "command": cmd}, indent=1), flush=True)
if not a.dry_run:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus); env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, env=env, cwd=ROOT, check=True)
