# JECT — Phrase Grounding

Train a text-conditioned 3D segmentation head on a frozen JECT/ART-JEPA backbone: given a chest-CT
volume and a free-text finding (e.g. *"Subcentimeter nodule in the right lower lobe"*), predict the
voxel mask of that finding.

The pretrained backbone stays frozen by default. What is trained is a multi-scale decoder plus the
text-conditioning path — about 35M parameters against the backbone's 546M.

> **New here? Read [SETUP.md](SETUP.md) first.** It is the step-by-step path: which files to copy,
> where to put them, how to verify the setup before training, and what the first run should print.
> This file is the reference manual for flags, memory limits and the submission pipeline.

---

## Quickstart

```bash
git clone <repo> && cd JECT/grounding
pip install -r requirements.txt
```

`train.py` can be launched from any working directory — Python puts the script's own directory on
`sys.path`, so the `ject` package resolves either way:

```bash
cd JECT/grounding && python train.py ...        # or
python /path/to/JECT/grounding/train.py ...     # also fine
```

Only `--out` is interpreted relative to your working directory, so prefer an absolute path for it
when launching from elsewhere. The scripts in `reproduce/` `cd` to the repo root themselves.

### Where the pretrained weights go

Anywhere — the path is passed with `--init`. There is no expected filename or location. A `weights/`
directory here is convenient and already git-ignored:

```
JECT/grounding/
├── train.py
├── ject/
└── weights/
    └── ject_backbone_224.pt      <- pass this to --init
```

`--init` expects a checkpoint saved as:

```python
torch.save({"model": model.state_dict(), "config": {...}}, "ject_backbone_224.pt")
```

The architecture is rebuilt from the checkpoint's own `config`, so you do not pass any architecture
flags — a 224 checkpoint builds a 224 model, a 336 one builds a 336 model. Absolute position
embeddings are interpolated automatically if `--res` differs from the checkpoint's resolution.
The loader prints what it did:

```
Flexible checkpoint load: loaded=1382 resized=4 skipped=0
  resized: ['student_encoder.pos_embed', 'student_encoder.pos_embed_s4', ...]
```

`skipped` should be **0**. A skipped `pos_embed*` means that scale silently starts from random
weights — stop and check `--res` against the checkpoint rather than training on.

### Smallest run that works

```bash
export JECT_DATA=/data/volumes          # dir of <object_id>.npz
export JECT_REX=/data/rex.csv           # annotations
export JECT_EMB=/data/phrase_emb.pt     # {phrase: embedding}

python train.py --init weights/ject_backbone_224.pt --out runs/first
```

Everything else has a working default. Outputs land in `--out`: `best.pt` (highest val Dice),
`latest.pt`, and per-epoch val lines on stdout. To reproduce a published configuration instead, see
**Reproducing published runs** below.

Requires a CUDA GPU. Memory scales steeply with `--res` and `--hi_ch`; see **Resolution & memory**.

---

## Data layout

Three inputs, all set by CLI flag or environment variable. There are **no machine-specific defaults** —
an unset or missing path fails immediately rather than silently training on the wrong data.

| flag | env | what |
|---|---|---|
| `--data_root` | `JECT_DATA` | directory of per-volume `.npz` files |
| `--rex_csv` | `JECT_REX` | grounding annotations (CSV) |
| `--text_emb` | `JECT_EMB` | `{phrase: embedding}` dict (`.pt`) |

**`--data_root`** — one `.npz` per volume, named by `object_id`:

```
<data_root>/<object_id>.npz
    ct   float array (D,H,W) or (1,D,H,W)   Hounsfield units
    rex  uint8 array (F,D,H,W) or (F,1,D,H,W)   F binary masks, one per finding
```

`ct` is windowed internally; store raw HU. Volumes may be any cube size ≥ `--res` — they are
centre-cropped (train: random-cropped) to `--res`.

**`--rex_csv`** — one row per volume:

| column | meaning |
|---|---|
| `object_id` | matches the `.npz` filename |
| `rex_split` | `train` / `val` / `test` |
| `rex_findings` | JSON `{"<index into rex>": "<finding text>"}` |

**`--text_emb`** — a `torch.save`d `dict[str, Tensor]` mapping each finding string to its embedding.
Phrases absent from this dict are skipped, so it must cover every phrase you intend to train on.
Embedding width must match the backbone's `cfg.text.cached_dim` (1024 for the released weights).

---

## Train

```bash
python train.py \
  --init weights/ject_backbone_224.pt \
  --data_root /data/volumes --rex_csv /data/rex.csv --text_emb /data/phrase_emb.pt \
  --res 224 --out_res 224 --hi_ch 64 \
  --fusion xattn --enc_skips s1 s2 s3 \
  --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 --s2_init pretrained \
  --batch 2 --epochs 28 --lr 1e-3 --out runs/grounding
```

`--init` accepts a pretrained checkpoint saved as `{"model": state_dict, "config": {...}}`. The
architecture is rebuilt from the checkpoint's own `config`, and absolute position embeddings are
interpolated if the checkpoint's resolution differs from `--res`.

Each epoch reports best-threshold mean Dice and hit@0.1 on the val split, and writes `best.pt`
(highest val Dice so far) and `latest.pt`. Frozen backbone weights are stripped from `best.pt`,
since they are identical to `--init`.

---

## What the flags do

The backbone is not just a feature extractor — three of these reuse mechanisms learned during
pretraining.

| flag | effect |
|---|---|
| `--enc_skips s1 s2 s3` | take decoder skips from the **encoder** stages rather than the pretrained dense decoder. The dense decoder attenuates phrase-discriminative signal; the encoder retains it. |
| `--predictor_prior` | inject the frozen predictor's classifier-free-guidance residual `pred(phrase) − pred(null)` into the decoder. Largest single contributor in our ablations. |
| `--mup_prior s4` | invert the frozen masked-utterance-prediction head into a spatial prior: score each location by how well the text it predicts matches the query phrase. Requires no finetuning. |
| `--mup_logit` | add that prior directly to the output logit instead of gating the bottleneck, so the decoder learns a residual over a working prior rather than being free to ignore it. |
| `--fusion xattn` | phrase attends over image tokens; `--s2_init pretrained` warm-starts it from the pretrained grounding head. |
| `--film` | per-channel text modulation at every decoder stage (zero-init identity). |
| `--deep_sup` | auxiliary losses at intermediate decoder scales. |
| `--use_s4` | route the coarsest encoder stage into the bottleneck as global context. |
| `--txt_head` | make the full-resolution logit text-dependent via a dot product between a text-derived vector and per-voxel embeddings (zero-init gate). |

Losses: `--tversky_a/b`, `--pos_weight` (overlap); `--mil_w` (max-in/max-out ranking hinge);
`--attn_kl_w` (KL between the normalised mask and the text-image similarity map); `--infonce_w`
or `--dense_loss infonce` (voxel-level ranking).

> Ranking losses (`--mil_w`, `--infonce_w`) act on **raw logits** and are far sparser than the
> overlap terms, so their *gradient* share is orders of magnitude above their *loss* share. In our
> setup `--infonce_w 0.1` contributed 22% of the loss value but **99.9% of the gradient**. Size these
> by gradient norm, not by loss magnitude.

---

## Resolution & memory

The decoder emits **`res/2`** (after `up3`) or **`res`** (after `up4`) — nothing in between. Any
other `--out_res` is rejected at startup with an explanatory message, since it would otherwise
produce a logit/target shape mismatch at the first step.

Two hardware cliffs are worth knowing, both from CUDA kernels that index with 32-bit integers:

* the decoder stem holds a `res³ × hi_ch` tensor. Once that exceeds **2³¹ elements**, cuDNN falls
  back to a 64-bit path roughly **30× slower**. At `res=336` this caps `hi_ch` at **56**;
  at `res=224` any `hi_ch ≤ 170` is safe.
* emitting at full input resolution is far more expensive than one rung down — at `res=336`,
  `out_res=336` measured ~48× the cost of `out_res=168`.

Measured on a 140 GB GPU:

| res | out_res | hi_ch | batch | fwd | VRAM |
|---|---|---|---|---|---|
| 224 | 224 | 64 | 2 | ~0.4 s | 64 GB |
| 336 | 168 | 56 | 3 | ~0.3 s | 124 GB |
| 336 | 336 | 64 | 1 | ~40 s | 50 GB |

---

## Evaluation

Dice is highly threshold-dependent, so `evaluate()` sweeps a fixed threshold grid and reports the
best-threshold mean Dice plus a per-sample oracle upper bound. **The reported number is therefore
val-tuned** — for a held-out figure, pick the threshold on val and apply it unchanged to test.

---

## Reproducing published runs

`reproduce/` holds the exact configurations behind the reported numbers. Each script requires
four environment variables and fails immediately if any is unset:

```bash
export JECT_DATA=/data/volumes        # .npz dir
export JECT_REX=/data/rex.csv         # annotations
export JECT_EMB=/data/phrase_emb.pt   # {phrase: embedding}
export JECT_INIT=/weights/backbone.pt # pretrained checkpoint

bash reproduce/muplogit_224.sh        # best single model, val Dice 0.3070
bash reproduce/gf336_cooled.sh        # high-res variant on a 336-cooled backbone
```

| script | backbone | res | out_res | hi_ch | batch | result |
|---|---|---|---|---|---|---|
| `muplogit_224.sh` | 224 pretrained | 224 | 224 | 64 | 2 | **0.3070** val Dice (ep15/28) |
| `gf336_cooled.sh` | 336-cooled | 336 | 168 | 56 | 3 | 0.2203 (ep7) |
| `gf336_hires.sh` | 336-cooled | 336 | 336 | 56 | 2 | 0.2646 (final, ep12) |
| `gf336_sepup4.sh` | 336-cooled | 336 | 336 | 56 | 1 ×3 accum | 0.1397 (ep1, abandoned) |

`gf336_cooled.sh` accepts `EXTRA="--txt_head"` for the text-conditioned-output variant; all four
accept `OUT=` to redirect the run directory, and the 336 scripts also take `BS=` and `EPOCHS=`.

Best result to date is **0.3316**, a 3-model ensemble with test-time augmentation — not a single run.
The single-model figures above are what these scripts reproduce.

---

## Submission

Training and evaluation run on preprocessed `.npz` cubes, but a challenge submission must be on the
**original `.nii.gz` grid**. `ject/submission/` handles that round trip. Run the three steps in order
— each one gates the next.

### 0. Verify the geometry inversion

```bash
python -m ject.submission.invert \
  --npz_dir /data/preprocessed --orig_dir /data/orig_ct \
  --totalseg_dir /data/totalseg --rex_dir /data/orig_masks \
  --crop 256 --spacing 1.5
```

Inverts the ground truth from model space and Dices it against the original mask. `--crop`/`--spacing`
must match the preprocessing that produced your `.npz`:

| pipeline | `--crop` | `--spacing` | field of view |
|---|---|---|---|
| 224-model data | 256 | 1.5 | 384 mm |
| 336-model data | 384 | 1.0 | 384 mm |

> **Small findings do not round-trip, and that is the data pipeline, not a bug.** Measured on val at
> 1.5 mm: findings ≥1000 original voxels return at median Dice **0.898**, findings <1000 voxels at
> **0.479** — a 165-voxel finding survives preprocessing as **8 voxels**. The test therefore gates on
> large findings only. It also means small-lesion Dice is capped before the model is involved, which
> is an argument for the 1.0 mm pipeline independent of model quality.

### 1. Predict

```bash
python -m ject.submission.predict \
  --init weights/backbone.pt --weights runs/grounding/best.pt \
  --split val --out_dir preds/val \
  --res 224 --out_res 224 --hi_ch 64 \
  --fusion xattn --enc_skips s1 s2 s3 --predictor_prior --mup_prior s4 --mup_logit \
  --film --deep_sup --use_s4 \
  --thr_mode relmax --thr 0.75
```

Architecture flags must match the ones the weights were trained with — a mismatch aborts rather than
loading partially. Writes one `<id>_pred.nii.gz` per case, shape `(F, S, S, S)`.

Thresholding happens here, since everything downstream is binary. `relmax` cuts at a fraction of each
finding's own peak; `abs` uses a fixed probability. Pick one, validate with it, submit with it.

### 2. Score on val, in original geometry

```bash
python -m ject.submission.pack --mode score \
  --pred_dir preds/val --gt_dir /data/orig_masks \
  --orig_dir /data/orig_ct --totalseg_dir /data/totalseg \
  --crop 256 --spacing 1.5 --closing 0
```

**Do this before generating a test submission.** It is the only number measured on the same grid the
challenge scores on. Compare it against your model-space val Dice: a large drop means the inversion
or thresholding is losing mask, not that the model got worse.

### 3. Pack the test set

```bash
python -m ject.submission.pack --mode zip \
  --pred_dir preds/test --out_dir submission/test \
  --orig_dir /data/test_ct --totalseg_dir /data/test_totalseg \
  --crop 256 --spacing 1.5 --closing 0 --expect 300
```

Writes one file per case named exactly like the CT, shape `(F, H, W, D)`, then zips them.

> `--closing` **must be identical** in steps 2 and 3, or the submitted masks are not the ones you
> measured. `--expect` triggers a loud warning if the zip is short — missing cases score zero.

---

## Layout

```
train.py                     entry point -> ject.grounding.train:main
reproduce/                   exact configs behind the reported numbers
ject/
  config/  art.py lac.py     dataclass configs; rebuilt from the checkpoint's own `config`
  backbone/                  the pretrained model, imported so it can be reconstructed exactly
    model.py                 ARTJEPAModel: encoder + predictors + MUP / grounding heads
    encoder.py               Swin-UNETR-style 4-scale encoder
    rope.py                  3D rotary embeddings (parameter-free, resolution-agnostic)
    masking.py  text.py  anatomy.py  region_text.py
  data.py                    volume loading, crop/pad transforms
  checkpoint.py              flexible loader; interpolates position embeddings across resolutions
  decoder.py                 ConvBlock3D / UpBlock3D
  submission/
    invert.py                model space -> original nii.gz geometry (+ round-trip self-test)
    predict.py               run a trained model over a split -> per-case masks
    pack.py                  invert then score (val) or zip (test)
  grounding/
    paths.py                 dataset locations + input cube size (single source of truth)
    dataset.py               RexGrounding, distributed setup, config loading
    model.py                 TextGroundingJEPA, SpatialTextFuse, FiLM3d
    losses.py                Tversky+BCE, MIL hinge, attention-KL, voxel InfoNCE
    evaluate.py              threshold-swept Dice
    train.py                 argument parsing and the training loop
```

`paths.py` is imported as a **module** everywhere (`import ject.grounding.paths as paths`), never as
`from ject.grounding.paths import DATA` — the latter binds the value at import time, so a CLI
override would be silently ignored.
