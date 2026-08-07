# Setup — getting a first run going

Onboarding path for someone picking this repo up fresh. Follow it in order; each step verifies the
previous one. `README.md` is the reference manual — come here first, go there for detail.

Requires one CUDA GPU. The 224 config needs ~64 GB, the 336 configs ~124 GB.

---

## 1. What you need to copy

Seven things to train, and three more if you also want to produce a submission. Sizes are for the
grounding subset only — **you do not need the full pretraining corpus.**

| # | what | size | where it is now |
|---|---|---|---|
| 1 | code | — | this repo |
| 2 | volumes, 256³ @ 1.5 mm — for **224** models | **60 GB** (3,492 files) | `/opt/project/final_preprocessed_half` |
| 3 | volumes, 384³ @ 1.0 mm — for **336** models | **195 GB** (3,492 files) | `/opt/project/final_preprocessed_full` |
| 4 | annotations CSV | 800 KB | `/opt/project/cepa2/dataframe/final_rex_miccai.csv` |
| 5 | phrase embeddings | 29 MB | `/opt/project/cepa2/art/runs/rex_all_embs.pt` — or rebuild it, § below |
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

### If you also want to produce a submission

Training and validation need nothing beyond the table above. A **challenge submission does**, because
scoring happens on the original CT grid: `ject/submission/` inverts each prediction back to the
`.nii.gz` geometry it came from, and it reconstructs that geometry from the original files rather than
storing it. Three more directories, one file per `case_id`, named `<case_id>.nii.gz` throughout:

| # | what | flag | val | test | where it is now |
|---|---|---|---|---|---|
| 8 | original CT | `--orig_dir` | 200 files, 22 GB | 300 files, 33 GB | `/opt/project/miccai/{val,test}/original/` |
| 9 | TotalSegmentator labels | `--totalseg_dir` | 200, 397 MB | 300, 599 MB | `/opt/project/miccai/{val,test}/totalseg/` |
| 10 | ReXGrounding masks, **original geometry** | `--rex_dir` / `--gt_dir` | 200, 161 MB | — | `/opt/project/miccai/{val,test}/rexgrounding/` |

**TotalSegmentator is not optional and not obvious.** Grounding never reads it — `dataset.py` loads
only `ct` and `rex` (the *pretraining* dataset does use it, `ject/data.py:302`). The *inversion* needs it:
the preprocessing crop was centred on the lung bounding box (labels 10–14), so recovering where the
`T³` cube was cut from means recomputing that box (`invert.py:73-84`, where the offset becomes the
affine's translation term). The `totalseg` array inside the `.npz` cannot substitute — it is already
cropped, so it cannot say where the crop came from. Omit this directory and predictions invert to the
wrong place in the original volume, with no error.

**The test split has no ground-truth masks** — that is the point of it. `--mode score` therefore works
only on `val`; for `test` you run `--mode zip` and submit.

### Rebuilding the phrase embeddings

Unlike the volumes, item #5 is **reproducible from this repo in one command** — a few minutes on one
GPU, no preprocessing pipeline involved:

```bash
pip install transformers accelerate          # not in requirements.txt: needed to make this file, not to train
python tools/make_phrase_embeddings.py \
  --rex_csv /path/to/final_rex_miccai.csv \
  --out weights/rex_all_embs.pt \
  --verify /path/to/existing/rex_all_embs.pt   # optional
```

It embeds every distinct `rex_findings` phrase with **Chest2Vec-0.6B**
(`lukeingawesome/chest2vec_0.6b_chest`) and writes `{phrase: 1024-d float32}`, L2-normalised. The
published model has its LoRA merged into the weights and bundles its own tokenizer and modeling
code, so `trust_remote_code=True` is the whole story — no `chest2vec` package, no `peft`, and no
separate download of the base Qwen3-Embedding.

Nothing here is machine-specific: the model is **public and ungated** (no HF token), both paths are
CLI arguments, and it runs on **CPU** if you have no GPU free — 0.6B parameters, a few minutes
either way. This is the one input a new machine can produce without being handed anything but the
CSV.

Verified against the shipped file across all **6,926** phrases: cosine **min 0.9993, mean 0.9997**
(bf16 rounding in the merged weights — the model card quotes ≥ 0.9996 for the same comparison).
`--verify` fails below 0.999, which is where a different encoder or pooling would land.

Two ways to get this wrong, both silent:

* **Keys must match the CSV exactly.** `dataset.py` does `if txt in embs` and skips any phrase it
  cannot find, so a stray character drops that finding from training with no error. Re-run §3a and
  confirm `train: 7687 / val: 381`.
* **Do not pass an instruction.** Chest2Vec follows the Qwen3-Embedding convention where queries are
  wrapped in an instruction template and documents are not; these phrases were embedded as
  documents, and an instruction yields a different vector for the same string.

Swapping the encoder is allowed — the grounding head's `query_proj` (1024 → 256) trains from
scratch, so only *consistency* matters. Rebuild the whole file and retrain the head; the failure to
avoid is two encoders inside one file.

### The `.npz` format, if you have to produce it yourself

Three arrays, all `(1, T, T, T)`, one file per `case_id`:

| key | dtype | contents |
|---|---|---|
| `ct` | `int16` | Hounsfield units. `gwin()` in `dataset.py` maps them to [0,1] at load |
| `totalseg` | `uint16` | TotalSegmentator label map. Not read by grounding training; the pretraining dataset cuts anatomy masks from it (`ject/data.py:302`) to drive region masking and the region-JEPA loss |
| `rex` | `uint16` | ReXGrounding finding masks, index `k` = finding `k` in the CSV's `rex_findings` |

Produced by: reorient to **LPS** → resample to isotropic spacing → crop to `T³` around the lungs
(TotalSegmentator labels 10–14). Two `(T, spacing)` pairs are in use, covering the same 384 mm field
of view — `(256, 1.5 mm)` for 224 models and `(384, 1.0 mm)` for 336 models. Pass the matching pair to
`invert.py`/`pack.py` as `--crop`/`--spacing`; **guessing it from the prediction size is not safe.**

Each preprocessed directory also carries a `preprocessing_metadata.csv` recording, per case, the three
source `.nii.gz` paths, the original shape and spacing, and the processed shapes. That file is the
provenance record — check it before assuming a volume was built the way you expect.

> **The preprocessing script itself is not in this repo.** It lives in the tree this package was
> extracted from. Everything above is enough to reimplement it or to verify an existing `.npz`, and
> `python -m ject.submission.invert` is the check: it inverts stored ground truth back to the original
> grid and fails below `--gate` (default 0.85) median Dice on large findings.

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
| `gf336_hires.sh` | 336-cooled | 336 | 336 | 56 | 2 | 0.2646 (final, ep12) |
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
