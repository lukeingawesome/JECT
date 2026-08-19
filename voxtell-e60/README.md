# VoxTell E60 — text-prompted 3D grounding on ReXGroundingCT

Fine-tune [VoxTell](https://github.com/MIC-DKFZ/VoxTell) v1.0 (440M, text-prompted, pretrained
on anatomy) for referring segmentation: given a chest CT and a free-text finding — *"Subcentimeter
nodule in the right lower lobe"* — predict that finding's voxel mask.

This is the **E60** arm and nothing else. The sibling [`../grounding/`](../grounding/) folder is
the JECT/ART-JEPA approach to the same task; this one starts from a different pretrained model and
is here so the two are comparable side by side.

> **New here? Read [SETUP.md](SETUP.md) first.** It is the step-by-step path: what to install,
> what to copy, how to verify each step before spending GPU hours, and what the first run should
> print. This file is the reference manual.

---

## The number

`mean_global_dice_per_finding`, the challenge's official metric, over the MICCAI val split
(200 cases / 381 findings), at the submission operating point `--tile-step 0.25 --thresholds 0.5`:

| run | val dice | what changed |
|---|---|---|
| **E60** | **0.34334** | arm E, per-volume z-score, 60,000 steps |
| E60_Hu | 0.34076 | `--hu-norm ct` and nothing else |
| — | 0.1538 | the bar this had to clear (`BAR` in `train_voxtell.py`) |

Every number here is produced by the challenge's own `rexrank_eval_fast.py`, imported as a module
by [`scripts/scorer.py`](scripts/scorer.py) and pinned by content hash. Nothing in this folder
re-implements a metric.

**E60 vs E60_Hu is a null result, and is reported as one.** The −0.00258 gap is inside the
pre-registered ±0.008 detection floor, and the sign flips between operating points (at threshold
0.6 it is −0.00231; at the in-training sweep's tile 0.5 it is **+0.00196**). There is no
gradient-side effect either: step-25 `grad_norm` is 20.603 under z-score against 20.387 under ct.

---

## What "arm E" means

VoxTell fuses the text embedding into the decoder. Arm E routes that fused feature **back into the
encoder trunk** as well, through a PWAM module at encoder stages 2 and 3, so the image features
downstream of the fusion have already seen the prompt.

```
                   --fuse off  (arms B / C)        --fuse trunk  (arm E, E60)

  image ──► encoder ──────────────► decoder      image ──► encoder ──┬──► decoder
                                       ▲                       ▲     │        ▲
  text  ─────────────────────────────┘                         └─PWAM┤        │
                                                  text  ──────────────┴────────┘
```

`--fuse skip` (arm S) sends the fused feature to the decoder skips only. E and S differ by that
one flag and nothing else, which is what makes the comparison mean something.

| | parameters |
|---|---|
| stock VoxTell v1.0 (`--fuse off`) | 440,029,541 |
| arm E (`--fuse trunk`) | 442,243,301 |
| the PWAM fusion module itself | 2,213,760 |

`tests/test_e60_real.py` pins all three, because a change here does not crash — it produces a
model the published checkpoints fail to load into, or one that loads and means something else.

---

## Layout

```
voxtell-e60/
├── scripts/
│   ├── train_voxtell.py      training, the in-training official sweep, both text banks
│   ├── voxtell_fusion.py     the PWAM module arm E adds
│   ├── predict_split.py      sharded tiled inference, N thresholds from one forward, TTA
│   ├── scorer.py             the challenge scorer, hash-pinned + the transforms it needs
│   ├── make_submission.py    prediction dir -> submission zip + provenance manifest
│   ├── paths.py              the three host-specific paths, one place, env-overridable
│   ├── build_npy_cache.py    paired NIfTI -> the native-grid cache the loader reads
│   └── build_native_stats.py per-volume HU statistics + per-finding centroids
├── src/rexg_data/           the YAML config loader, the CT-RATE geometry/HU join,
│                            and the two transforms build_npy_cache.py bakes in
├── configs/rexg_native.yaml  the ONLY config; see "the config that lies" below
├── reproduce/                the exact launches, with the values the published runs used
└── tests/                   107 tests, ~45 s; only test_e60_real.py needs the real assets
```

## Quickstart

```bash
pip install -r requirements.txt          # after installing VoxTell itself — see SETUP.md
python scripts/train_voxtell.py --self-test          # gates only, no GPU, no data
python -m pytest tests/ -q

reproduce/text_banks.sh                  # once, ~8 min/bank on a free GPU
reproduce/e60.sh                         # ~15.5 h on one H200
reproduce/submit.sh runs/E60/checkpoint_step60000.pth zscore
```

`train_voxtell.py` can be launched from any directory — it puts its own directory and `src/` on
`sys.path`. `--out-dir` is interpreted relative to your working directory, so prefer an absolute
path when launching from elsewhere. The `reproduce/` scripts `cd` to this folder themselves.

---

## Training

```
python scripts/train_voxtell.py --fuse trunk --out-dir runs/E60 \
       --steps 60000 --val-at 30000,60000 --ckpt-every 5000
```

| flag | default | notes |
|---|---|---|
| `--fuse {off,trunk,skip}` | `off` | `trunk` is arm E (E60). `off` is stock VoxTell. |
| `--random-init` | off | arm C: skip the pretrained load, change nothing else. |
| `--hu-norm {zscore,ct}` | `zscore` | see below. Recorded in the checkpoint; inference refuses to disagree. |
| `--steps` | 60000 | poly `(1 − t)^0.9` LR over exactly this many steps. |
| `--val-at` | `30000,60000` | steps at which the official sweep runs. Each costs ~60–85 min of the training GPU. |
| `--val-limit` | 0 | truncate the sweep for a smoke run. Any nonzero value marks the result PARTIAL and not comparable. |
| `--batch-size` / `--queries` | 2 / 2 | one item = one 192³ patch + 2 sentences of that CT. |
| `--lr` / `--momentum` / `--clip` | 1e-4 / 0.99 / 12.0 | SGD nesterov, nnU-Net's clip. |
| `--fg-prob` | 0.66 | probability the patch is centred on a finding's centroid (+ jitter), else uniform. |
| `--ckpt-every` | 0 | write a checkpoint every N steps. Sweeps are hours apart; this bounds what a crash costs. |
| `--resume` / `--init-from` | — | continue everything / load weights only. Mutually exclusive, and refused together. |

**Cost, measured on one H200 (E60_Hu, the full 60,000 steps):** 0.95 s/step, peak 41.3 GiB,
15 h 30 min wall including both sweeps (85 min at step 30,000 and 64 min at step 60,000).

**Patch size is not a flag.** VoxTell's `pos_embed` is a persistent `(1728, 1, 2048)` buffer —
12³ tokens at stride 16 — so 192³ is hard-locked, asserted against `plans.json` at startup
(G-SHAPE). It lives in code, not in the YAML, for that reason.

### The two text banks

Sentences are encoded **once**, offline, by Qwen3-Embedding-4B into an `.npz` bank; training and
inference read the bank and never load the encoder. Arm E needs **two** banks, and they are not
interchangeable:

* the **pooled** bank — one 2560-d vector per sentence, what the stock model consumes;
* the **token** bank — the per-token sequence the PWAM fusion attends over.

They must describe the same sentence set. `--build-token-bank-only` therefore reads its sentence
list *off the pooled bank on disk*, in the pooled bank's own order, rather than recomputing it —
recomputing has to reproduce the exact selection logic, and drifting from it even slightly is how
this broke once: a pooled bank extended for the test split while the token bank stayed at its
original size, so 2,675 of 6,467 sentences were paired against the wrong text.
`token_bank_path_for()` is the one place the token bank's filename is derived; never write it out.

A **submission** needs a different pair. The default bank covers findings that have a mask
(`has_seg == 1`), and the test split has none — so its prompts are simply absent, and
`gate_bank_covers` refuses at predict time. `--bank-splits train,val,test` builds the `_all` pair;
`reproduce/text_banks.sh` builds both pairs.

### `--hu-norm`

| | what it does |
|---|---|
| `zscore` (default) | per-volume statistics on RAW HU, read from `stats.csv`. What E60 trained under. |
| `ct` | nnU-Net's CT scheme: clip to [−1000, 1000], then a FIXED affine (mean −540.9803, std 482.0892). |

The constants were measured over all 2,992 MICCAI train volumes / 292.7G voxels *after* clipping,
so no test-split intensity leaks into the normalisation. The motivation is real even though the
result was null: per-volume z-score lets a handful of voxels set the whole volume's brightness
scale — 504 of 3,192 volumes have `hu_std > 600` against a median of 483, and one volume's std
jumped from 492 to 1601 because of the 0.1% of voxels above 1000 HU. The same tissue at +30 HU
therefore lands on a different input value in different patients.

It is resolved **once** (`hu_norm_spec`) and the same dict reaches the training patches and the
in-training sweep, so they cannot disagree. At inference `gate_hu_norm` **SystemExits** when the
requested scheme differs from the one the checkpoint records. That gate is not decoration: masks
come out either way and the scorer scores them either way, so without it no metric would ever show
the mismatch.

---

## Inference and submission

```
python scripts/predict_split.py --checkpoint runs/E60/checkpoint_step60000.pth \
       --fuse trunk --split test --out runs/pred --tile-step 0.25 --thresholds 0.5,0.6 \
       --shard 0/6 --device cuda:0 --save-prob 0.05 --text-bank .../text_emb_qwen3_all.npz
```

Three things it does that the in-training sweep does not:

* **`--shard i/n`** splits the case list across processes and GPUs. Cases are independent, so
  this shards perfectly; prediction, not scoring, is the bottleneck (a 200-case pass is ~60 min
  of prediction against ~7 min of scoring with the fast scorer).
* **`--thresholds`** writes one directory per threshold from a **single** forward pass. The stock
  `predict_single_image` bakes in `sigmoid > 0.5`; this calls the public
  `predict_sliding_window_return_logits` underneath instead, so N thresholds cost N cheap
  binarisations rather than N inference passes.
* **`--split test`**, where there are no masks to score against — which is what a leaderboard
  submission needs.

`--save-prob FLOOR` additionally stores the sigmoid sparsely above `FLOOR`, so re-thresholding
later costs minutes instead of a second 40-minute pass over 300 volumes.

`--tile-step 0.25` (against the predictor's default 0.5) is the operating point both submissions
used: it quadruples the sliding-window overlap, so each voxel is seen at four offsets inside a
patch.

**`--tta SHIFT`** is translation-jitter TTA ported from JECT's `infer.py:tta_fwd` — 7 forwards,
the volume shifted by ±SHIFT voxels along each axis, averaged **after** the sigmoid. It costs 7×
the wall clock. Note it uses edge replication, not `torch.roll`: JECT's volume is a lung-anchored
cube whose borders are air, so a wrap joins air to air; here the volume is a full chest CT and a
wrap joins the top of the lungs to the abdomen. `tests/test_predict_split_tta.py` checks the
inverse algebraically, against a translation-equivariant stand-in network — a wrong sign there
does not crash, it just blurs the prediction by `SHIFT` voxels and scores slightly worse.

**There is deliberately no flip TTA.** 85.7% of test prompts name left or right and 55.3% name a
lobe, so mirroring the image while the prompt still says "right middle lobe" contradicts the
request for most cases — and thoracic anatomy is not symmetric anyway (three lobes right, two
left, plus the heart and the arch). `tests/test_no_flip_augment.py` is an executable assert
rather than a comment, because the first person who wants more augmentation will reach for flips.

`make_submission.py` writes the zip **and a manifest next to it** carrying `checkpoint_sha`,
`text_bank_sha`, `tile_step`, `code_sha` and `shards_agree_on_checkpoint`. That manifest is the
only provenance a submitted zip has.

---

## The gates

Every one is a literal `assert` or a `SystemExit`, and `main()` refuses to start under `python -O`
— that flag deletes asserts, and it was the one thing that could skip all of them at once.

| gate | what it stops |
|---|---|
| `G-INDEX` | the sample index comes from `findings.csv[miccai_split, has_seg]`, never from globbing `npy/ct/*.npy` — 100 cached test CTs have no mask at all. |
| `G-BINARY` | 49 of the 70 native seg channels are entity-**instance** maps carrying values 2..5. Raw BCE against target=3 is unbounded below (measured: loss −1.045 and still falling) and the run completes normally with a garbage checkpoint. |
| `G-SHAPE` | the patch is exactly 192³ after padding. 104 of 3,192 volumes have nz < 192 (min 104). |
| `G-AXIS` | the training patch is in nnU-Net's (nz, ny, nx) order, the same order the sweep feeds the predictor. A **cube** patch hides an axis swap from every shape assert, so this one is enforced by a test on an anisotropic fixture, not a runtime check. |
| `G-TEXT` | the bank is raw last-token-pooled Qwen3, wrapper included. An L2-normalised bank fed to a pretrained `Linear(2560, 2048)` is a 138× scale error. |
| `G-BANK-CONSISTENT` | every token bank row's last token is bit-identical to the pooled bank's vector for the same sentence. |
| `G-LOSS-POSITIVE` | loss > 0 every step. Non-positive means G-BINARY was bypassed. |
| `G-LOAD-KEYS` | arm E relaxes the v1.0 load to `strict=False`; this asserts the only unsupplied keys are exactly the fusion module's, so a silently-partial load cannot make arm B a slightly-warmer arm C. |
| `G-HU-NORM` | inference refuses a normalisation the checkpoint was not trained under. |
| `G-WRAPPED-CKPT` | a `net.`-prefixed checkpoint (from a wrapped arm this folder does not ship) is refused rather than silently loaded into the stock network. `tests/test_gates.py` additionally **scans** `scripts/` for any `load_state_dict(..., strict=True)` that reaches `network_weights` without passing through it. |
| `G-DEAD-CONFIG` | see below. |
| `G-SCORER` | the challenge scorer is present and byte-identical to the version the fast/reference equivalence was measured against. |

Two of those are **static scans over this folder's own source**, not runtime checks, because
the defect they catch is a line that never runs: `tests/test_no_undefined_names.py` fails on any
global name a function reads and the module never binds, and `test_no_loader_bypasses_gate_unwrapped`
fails on any `load_state_dict(..., strict=True)` that reaches `network_weights` without passing
through the gate. Both were written after a `NameError` on `val_sweep`'s fused branch — the branch
E60 actually takes — survived 94 green tests and surfaced only in a real run, after the training
steps.

### The config that lies

`configs/rexg_native.yaml` is the only config here, and every preprocessing key in it is `null` on
purpose. `train_voxtell.py` reads **none** of `spatial_size`, `target_spacing`, `cube_size`,
`iso_spacing`, `hu_window`, `cache2mm_root` — it cuts 192³ in **voxels** off the native grid, and
`VoxTellPredictor.preprocess` is `crop_to_nonzero` + normalisation with `plans.json` spacing set
to `None`. A config naming a spacing records preprocessing that does not happen, and B60 and E60
each ran 60,000 steps under a config claiming 2 mm isotropic resampling before anyone noticed.

`gate_dead_config` refuses to load such a file, naming the offending keys. A written note cannot
fix this, because the note is not what the next session reads — the config is. So the config stops
being loadable while it lies.

**What this means for the numbers, stated plainly:** 192³ on the native grid spans a different
physical volume per patient — 58.3–187.5 mm in plane (median 133.9) and 96–576 mm in z (median
192) across the 3,492 cached volumes. A 10 mm lesion is 10.2–32.9 voxels in plane depending on the
patient. Whether that costs Dice is **not measured**.

---

## What is not here

This folder is the E60 path only. The upstream work also ran an auxiliary per-category head, a
retrieval-adapter arm, and a dual-path arm that feeds a whole-volume cube alongside the patch;
their flags, code and tests were removed rather than shipped dark. `gate_unwrapped` is what makes
a checkpoint from one of those arms fail loudly here instead of loading into the stock network and
predicting plausible masks with every gate still green.

The 60-step `--hu-norm` smoke pair showed step-25 `grad_norm` 54.457 (z-score) against 16.241 (ct)
with the clip at 12.0, which is what motivated running E60_Hu at all. It does **not** reproduce at
60,000 steps (20.603 vs 20.387). Quoting the smoke number as if it were the run is the mistake
this paragraph exists to prevent.
