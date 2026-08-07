# Setup — getting a first run going

Onboarding path for someone picking this repo up fresh. Follow it in order; each step verifies the
previous one. `README.md` is the reference manual — come here first, go there for detail.

Requires one CUDA GPU. The 224 config needs ~64 GB, the 336 configs ~124 GB.

---

## 1. What you need to copy

Five things. Sizes are for the grounding subset only — **you do not need the full pretraining corpus.**

| # | what | size | where it is now |
|---|---|---|---|
| 1 | code | — | this repo |
| 2 | volumes, 256³ @ 1.5 mm — for **224** models | **60 GB** (3,492 files) | `/opt/project/final_preprocessed_half` |
| 3 | volumes, 384³ @ 1.0 mm — for **336** models | **195 GB** (3,492 files) | `/opt/project/final_preprocessed_full` |
| 4 | annotations CSV | 800 KB | `/opt/project/cepa2/dataframe/final_rex_miccai.csv` |
| 5 | phrase embeddings | 29 MB | `/opt/project/cepa2/art/runs/rex_all_embs.pt` |
| 6 | pretrained backbone — 224 | 5.0 GB | `/opt/project/cepa2/art/runs/stage0_huge_foundation_v2/checkpoint_epoch_40.pt` |
| 7 | pretrained backbone — 336-cooled | 5.0 GB | `/opt/project/cepa2/art/runs/stage0_huge_foundation_v2_cd336/checkpoint_epoch_6.pt` |

Copy **only the 3,492 volumes named in the CSV**, not the whole directory — the source dirs hold 32k–73k
files (1.9 TB / 848 GB) because they also carry the pretraining corpus. To pull just what you need:

```bash
python - <<'EOF'
import os, shutil, pandas as pd
SRC = "/opt/project/final_preprocessed_full"     # or ..._half for the 224 pipeline
DST = "/your/data/ject_336"
os.makedirs(DST, exist_ok=True)
for oid in pd.read_csv("/opt/project/cepa2/dataframe/final_rex_miccai.csv").object_id.astype(str):
    p = os.path.join(SRC, oid)
    p = p if os.path.exists(p) else p + ".npz"
    shutil.copy2(p, DST)
EOF
```

You only need one of #2 / #3 to start. Take **#3 (384³)** if you want to continue current work; take
**#2 (256³)** if you want to reproduce the best published single model, which is a 224 run.

### Pair the data with the backbone correctly

This is the one mistake that fails silently rather than loudly:

| backbone | `--res` | data dir | volume shape |
|---|---|---|---|
| `..._v2/checkpoint_epoch_40.pt` | 224 | `final_preprocessed_half` | 256³ @ 1.5 mm |
| `..._v2_cd336/checkpoint_epoch_6.pt` | 336 | `final_preprocessed_full` | 384³ @ 1.0 mm |

Volumes are **centre-cropped** to `--res` at load. Pointing a 224 run at the 384³ data will run
without error and quietly throw away most of the chest — the crop is in voxels, and at 1.0 mm 224 voxels
is a 224 mm field of view instead of 384 mm.

---

## 2. Put it anywhere, point at it with env vars

No path is hardcoded and there are no machine-specific defaults — an unset path fails loudly at startup
rather than training on the wrong data.

```bash
export JECT_DATA=/your/data/ject_336                 # dir of <object_id>.npz
export JECT_REX=/your/data/final_rex_miccai.csv      # annotations
export JECT_EMB=/your/data/rex_all_embs.pt           # {phrase: embedding}
export JECT_INIT=/your/weights/backbone_cd336.pt     # pretrained checkpoint
export JECT_RES=336                                  # only needed outside train.py — see below
```

`JECT_RES` sets the crop size **only** when you load the dataset yourself, as in step 3a. During
training it is irrelevant: `train.py` overwrites it from `--res`, so the CLI flag always wins. Set it
anyway, or the standalone check in 3a silently uses the 224 default and reports the wrong volume shape.

A convenient layout, already git-ignored:

```
JECT/grounding/
├── train.py
├── ject/
└── weights/
    ├── backbone_224.pt
    └── backbone_cd336.pt
```

```bash
pip install -r requirements.txt
```

---

## 3. Verify before you train

Three checks, ~2 minutes, in order. Each one catches a different silent failure.

### 3a. The data loads and the splits are right

```bash
cd JECT/grounding && python - <<'EOF'
import torch, ject.grounding.paths as paths
from ject.grounding.dataset import RexGrounding
embs = torch.load(paths.TEXT_EMB, map_location="cpu", weights_only=False)
for split in ("train", "val"):
    ds = RexGrounding(split, embs)
    print(f"{split}: {len(ds)} findings")
vol, q, gt = ds[0]
print("volume", tuple(vol.shape), "| embedding", tuple(q.shape), "| mask", tuple(gt.shape),
      "| positive voxels", int(gt.sum()))
EOF
```

Expect `train: 7687 findings`, `val: 381 findings`, a volume of `(1, res, res, res)`, an embedding of
`(1024,)`, and a non-zero positive count. A finding count far below these means phrases are missing from
the embedding dict — phrases absent from it are **silently skipped**, so `JECT_EMB` must cover everything
you intend to train on.

### 3b. The checkpoint loads cleanly

Watch the loader line at startup:

```
Flexible checkpoint load: loaded=1382 resized=4 skipped=0
```

**`skipped` must be 0.** A skipped `pos_embed*` means that scale silently starts from random weights.
`resized=4` is expected and correct whenever `--res` differs from the checkpoint's native resolution —
position embeddings are interpolated. If you see `skipped > 0`, stop and check `--res` against the
checkpoint instead of training on.

### 3c. One real optimizer step

Syntax checks and imports do not prove the training loop runs. Launch the real config with
`--epochs 1` and confirm you get a couple of `ep0 step…` lines with a finite loss, then kill it.

---

## 4. First run

```bash
bash reproduce/muplogit_224.sh      # best published single model — val Dice 0.3070 at ep15/28
```

Expect roughly **0.10–0.12 at ep1**, crossing **0.25 around ep8**, peaking **0.3070 at ep15**. If ep1 is
far below 0.10, something in §3 is wrong — stop rather than waiting for ep15.

Outputs land in `--out`: `best.pt` (highest val Dice so far), `latest.pt`, and one val line per epoch.
`best.pt` strips the frozen backbone, since it is byte-identical to `--init` — so **loading a run always
needs both `--init` and the run's `best.pt`**.

### The other configs

| script | backbone | res | out_res | hi_ch | batch | result |
|---|---|---|---|---|---|---|
| `muplogit_224.sh` | 224 | 224 | 224 | 64 | 2 | **0.3070** — best single model |
| `gf336_cooled.sh` | 336-cooled | 336 | 168 | 56 | 3 | 0.2203 @ ep7 |
| `gf336_hires.sh` | 336-cooled | 336 | 336 | 56 | 2 | 0.2480 @ ep7, still climbing |
| `gf336_sepup4.sh` | 336-cooled | 336 | 336 | 56 | 1 ×3 accum | 0.1397 @ ep1, abandoned |

All four take `OUT=` to redirect the run directory; `gf336_cooled.sh` also takes `EXTRA="--txt_head"`.

Best overall result to date is **0.3316** — a 3-model ensemble with TTA, not a single run.

---

## 5. Reading the numbers

Per-epoch val Dice is **threshold-swept on val**, so it is tuned on the set it reports. For a held-out
figure, fix the threshold on val and apply it unchanged to test. `ject/submission/` does the round trip to
original `.nii.gz` geometry and is the only path that produces a challenge-comparable number — see the
Submission section of `README.md`, and run its three steps in order, since each gates the next.

Two conventions worth knowing before they surprise you:

- The decoder emits **`res/2`** or **`res`** and nothing between. Any other `--out_res` is rejected at
  startup.
- At `res=336`, `hi_ch` is capped at **56**. The stem holds a `res³ × hi_ch` tensor and crossing 2³¹
  elements drops cuDNN onto a ~30× slower path. This is measured, not theoretical.

---

## 6. Where to make changes

```
train.py                  entry point -> ject.grounding.train:main
ject/grounding/
  paths.py                dataset locations (single source of truth)
  dataset.py              RexGrounding, distributed setup, config loading
  model.py                TextGroundingJEPA — the decoder and text conditioning
  losses.py               Tversky+BCE, MIL hinge, attention-KL, voxel InfoNCE
  evaluate.py             threshold-swept Dice
  train.py                argument parsing and the training loop
ject/backbone/            the pretrained model, imported so it can be rebuilt exactly
ject/submission/          model space -> original nii.gz, scoring and packing
```

`paths.py` is imported as a **module** everywhere (`import ject.grounding.paths as paths`), never
`from ject.grounding.paths import DATA` — the latter binds the value at import time, so a CLI override
would be silently ignored. Keep it that way.

Full flag reference, memory tables, and the submission pipeline are in `README.md`.
