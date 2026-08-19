# Setup — getting a first run going

Onboarding path for someone picking this folder up fresh. Follow it in order; each step verifies
the previous one, and every step before step 5 is cheap. [README.md](README.md) is the reference
manual — come here first, go there for detail.

Requires one CUDA GPU with ≥48 GB (measured peak 41.3 GiB at `--batch-size 2 --queries 2`).

---

## 1. Install VoxTell first, then this

`voxtell` is not on PyPI. Install it from source **before** `requirements.txt`, because it is what
pulls in `nnunetv2`, `acvl_utils`, `batchgenerators` and `dynamic_network_architectures`, and
letting it resolve those itself is what keeps the versions consistent.

```bash
git clone https://github.com/MIC-DKFZ/VoxTell && pip install -e VoxTell
pip install -r requirements.txt
```

Verify — this needs no data, no GPU and no checkpoint:

```bash
python scripts/train_voxtell.py --self-test
python -m pytest tests/test_gates.py tests/test_no_undefined_names.py -q
```

Expected: `self-test ok`, then `91 passed`. If `pytest` reports collection errors mentioning
`voxtell`, `nibabel` or `monai`, the environment is the problem, not the code.

## 2. Point at the three external assets

Everything host-specific lives in [`scripts/paths.py`](scripts/paths.py), overridable by
environment variable. Print what is resolved and whether it is there:

```bash
python scripts/paths.py
```

| variable | what belongs there | size |
|---|---|---|
| `REXG_VOXTELL_MODEL_DIR` | VoxTell v1.0's nnU-Net model dir: `plans.json` + `fold_0/checkpoint_final.pth` | 1.7 GB |
| `REXG_QWEN_DIR` | `Qwen3-Embedding-4B` — needed **only** in step 4 | 7.6 GB |
| `REXG_CHALLENGE_DIR` | the ReXGroundingCT release: `rexrank_eval_fast.py`, `rexrank_eval.py`, `segmentations/`, `MICCAI_challenge_dataset.json` | 3.4 GB |

The architecture is rebuilt from `plans.json`, so `REXG_VOXTELL_MODEL_DIR` is required even with
`--random-init`. Check the scorer specifically — it is hash-pinned, so a version drift shows up
here rather than as a number that quietly means something else:

```bash
python scripts/scorer.py
```

## 3. Point at the data, and build the cache

Edit **`configs/rexg_native.yaml`** — the four `paths:` entries are the only thing in it you
change. Every `preprocess:` key must stay `null`; `gate_dead_config` refuses the file otherwise,
and [README.md § the config that lies](README.md#the-config-that-lies) says why.

| key | what it points at |
|---|---|
| `paired_root` | the paired ReXGroundingCT release: `ct/`, `findings.csv`, `volume_labels.csv` |
| `seg_root` | the GT masks the scorer reads (same dir as `REXG_CHALLENGE_DIR/segmentations`) |
| `npy_root` | where the cache below is written — **budget 640 GB** |
| `cache2mm_root` | `null`. Non-null and the config is refused. |

Then build the cache. The loader reads it and nothing else; going back to the NIfTI costs ~2.5 s
per volume in gzip decode and HU rescale, every epoch.

```bash
python scripts/build_npy_cache.py --workers 8      # ~639 GB of CT + 840 MB of masks
python scripts/build_native_stats.py --workers 12  # -> npy_root/stats.csv
```

`build_npy_cache.py` is resumable (it skips volumes already written) and bakes only the two
settled transforms — the CT-RATE HU rescale and the LPS→RAS reorientation. Spacing, cropping and
the intensity window stay open decisions, which is why choosing one later does not mean rebuilding
640 GB.

`build_native_stats.py` writes per-volume HU statistics and per-finding centroids. Both are
required: volume statistics are the correct normalisation (patch-level z-score is a *different*
number — measured median skew 0.49 σ/voxel), and without the precompute the loader runs 3.2×
data-bound at 1.339 s/step against a 0.423 s/step compute budget.

Verify the whole of step 3 at once:

```bash
python -m pytest tests/test_e60_real.py -q
```

That builds the real 442M-parameter model on CPU (~1 min), asserts it is the architecture the
published checkpoints fit, and checks `findings.csv` / `volume_labels.csv` / `index.csv` /
`stats.csv` are all where the config says.

## 4. Build the text banks

```bash
reproduce/text_banks.sh                            # ~8 min per bank on a free GPU
```

Four files, ~4.4 GB, and the last time this GPU is needed for text at all. Two **pairs** — a
pooled `.npz` and its per-token `.npy` — one pair for training (sentences with a mask) and one
`_all` pair that also covers the test split's prompts. README § *the two text banks* explains why
both halves of a pair are required and why the token bank derives its own filename.

## 5. Train

```bash
reproduce/e60.sh                                   # arm E, 60,000 steps, ~15.5 h on one H200
```

The first 60 seconds tell you whether the next 15 hours are worth starting. Expected, in order:

```
G-DEAD-CONFIG passes silently
G-INDEX: dropped 10 findings on leak patients ('train_1841', 'train_2936')
G-INDEX: 2990 volumes / 7677 findings ...; 99 of them are smaller than 192 on some axis
text encoder: qwen3 (.../Qwen3-Embedding-4B) | pooling last_token | padding left | wrapper True
G-TEXT ok: 6467 sentences, ||e||_2 min 128.1 / median 137.5 / max 139.9, dim 2560
G-TEXT-TOK[qwen3] ok: 6467 sentences x 64 x 2560, prefix 23 dropped, lengths 3-61 (median 17)
G-BANK-CONSISTENT ok: every last token is bit-identical to the pooled bank
loaded .../checkpoint_final.pth strict=False, G-LOAD-KEYS ok: ... 442.24M params (arm E)
hu_norm: zscore -> None

arm E (fusion into the encoder trunk): 60000 steps, batch 2 x 2 queries, ...
[1/60000] loss 1.2... lr 1.000e-04 gn 0.46 ...
```

Then roughly `0.95 s/step` and `peak 41.3GiB`. If the first sweep is scheduled at step 30,000,
nothing else prints for about eight hours — that is expected, and `runs/E60/train_log.jsonl` is
the file to watch.

**Smoke it first if this is a new box.** `--steps 60 --val-at 60 --val-limit 4` exercises the
whole path including a (deliberately PARTIAL, not comparable) sweep in a few minutes.

Two flags that keep a crash cheap: `--ckpt-every 5000` writes a checkpoint independently of the
sweeps, and `--resume <ckpt>` continues optimizer momentum, step and sampler position exactly —
sample *i* is the same sample either way, because `__getitem__(i)` seeds its own RNG from
`(seed, i)`. `--init-from` is the other thing: weights only, fresh optimizer, schedule restarts at
step 0. They are mutually exclusive and refused together.

## 6. Predict and submit

```bash
reproduce/submit.sh runs/E60/checkpoint_step60000.pth zscore
reproduce/submit.sh runs/E60_Hu/checkpoint_step60000.pth ct 6      # with translation TTA
```

The second argument must match the `--hu-norm` the checkpoint was trained under. Getting it wrong
is a `SystemExit` naming both schemes, not a slightly worse score — see README § `--hu-norm`.

It scores val first and prints the number before touching the test split, and refuses to zip if
either split came back short. Expect `VAL mean_global_dice_per_finding = 0.34334` for E60 and
`0.34076` for E60_Hu; the zip lands in `runs/` with its provenance manifest beside it.

---

## Where things go wrong

| symptom | cause |
|---|---|
| `SystemExit: <config>: sets [...], and this script reads NONE of them` | a `preprocess:` key in the YAML is not `null`. That is `G-DEAD-CONFIG`; set it back. |
| `SystemExit: REXG_* points at ... which does not exist` | step 2. The message names the variable and what belongs there. |
| `SystemExit: G-BANK-COVERS: N of M sweep prompts are not in the bank` | predicting the test split with the training bank. Use the `_all` pair (step 4). |
| `SystemExit: ... trained with --hu-norm 'X' but this run asks for 'Y'` | `G-HU-NORM`. The checkpoint knows what it was trained under; believe it. |
| `SystemExit: G-WRAPPED-CKPT: N weight keys are net.-prefixed` | a checkpoint from an arm this folder does not implement. |
| `SystemExit: G-TEXT[qwen3]: median \|\|e\|\|_2 = 1.000, outside (130, 145)` | an L2-normalised bank. VoxTell wants raw last-token-pooled vectors — a 138× scale error into a pretrained `Linear(2560, 2048)`. |
| `refusing to run under python -O` | `-O` deletes every `assert`, which is every gate here. |
| `ModuleNotFoundError: nibabel` (at pytest **collection**) | environment, not code. Step 1. |
