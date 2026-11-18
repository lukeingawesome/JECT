# Training

JECT pretrains a 3D chest-CT encoder with a student/teacher joint-embedding objective. The student sees a masked volume and
predicts the teacher's latents of the masked regions at four scales. Masked section prediction (MSP) adds a second target:
for each report section, the predictor fills in the chest2vec embedding of that section's findings from the image alone.
Weak grounding ties report utterances to anatomy regions.

## Pretraining

`train.sh` runs pretraining with the settings of the released model. Set the data paths and the GPUs:

```bash
DATA_CSV=manifest.csv DATA_ROOT=cubes_256/ SR_ROOT=text_targets/sr/ TARGET_STATS=text_targets/section_stats.pt \
    OUT=runs/pretrain GPUS=0,1 bash train.sh
```

`SMOKE=1` runs 24 steps first; `DRY_RUN=1` prints the full command. Batch size, gradient accumulation, epochs and workers
can be overridden with `BATCH_SIZE`, `GRAD_ACCUM`, `EPOCHS` and `NUM_WORKERS`.

## Running a recipe

`recipes/*.json` hold the exact argument lists of the released runs, without data paths. `launch.py` adds your paths and
starts `torchrun`:

```bash
python launch.py --recipe pretrain --data_csv manifest.csv --data_root cubes_256/ \
    --sr_root text_targets/sr/ --target_stats text_targets/section_stats.pt --out runs/pretrain --gpus 0,1 --smoke
```

Drop `--smoke` for the full run. `--dry-run` prints the command without running it. Each recipe is specified for two GPUs;
with a different count, change `--grad_accum` in the recipe so the effective batch stays 32.

| recipe | input | schedule | init |
|---|---|---|---|
| `pretrain` | 224³ crops of 1.5 mm, 256³ cubes | 50 epochs, lr 7e-5, batch 4 × accum 4 × 2 GPUs | random |
| `cooldown_320` | 320³ crops of 1 mm, 384³ cubes | 10 epochs, lr 2.5e-5, batch 2 × accum 8 × 2 GPUs | full training checkpoint |

The cooldown needs `checkpoint_final.pt` from a pretraining run (student, teacher, decoder and predictors), not an
encoder-only export.

`--epochs` is fixed at launch because the learning-rate schedule is derived from it. Crops are taken in voxels, so the
crop size must match the cube spacing: 224 with 1.5 mm cubes, 320 with 1 mm cubes. A mismatch still trains, on the wrong
field of view.

`train.py` can also be called directly; `python train.py --help` lists every option. Options not used by the recipes
reproduce the ablations in the paper.

## Inputs

**Manifest** (`--data_csv`): one row per volume.

| column | meaning |
|---|---|
| `object_id` | file name of the cube under `--data_root`, without `.npz` |
| `split` | rows with `split == train` are used |
| `utterances_json` | JSON list, one entry per report finding, with its section and negation/uncertainty flags |

**Cubes** (`--data_root`): `<object_id>.npz` with `ct`, raw HU as int16, shape `[1, D, H, W]`. `totalseg` (TotalSegmentator
labels, same shape) is used for anatomy regions when present. See `preprocess/README.md`.

**Text targets**: `<object_id>_utterances4b.pt` next to each cube, `<object_id>_sr_targets4b.pt` under `--sr_root`, and the
train-split section statistics passed as `--target_stats`. See `text/README.md`.

## Outputs

`--output_dir` receives `config.json`, `checkpoint_latest.pt`, `checkpoint_epoch_N.pt` and `checkpoint_final.pt`. Multi-GPU
runs use DDP through `torchrun`; without `RANK`/`LOCAL_RANK` the script runs as a single process.

## Code layout

```
train.py                     entry point: config, DDP, training loop, checkpoints
ject/config/                 model and loss configuration
ject/data.py                 manifest and cube dataset, text sidecars, HU windows
ject/checkpoint.py           checkpoint loading for warm starts and cooldowns
ject/losses/                 latent prediction, anti-collapse, MSP and grounding losses
ject/backbone/model.py       student/teacher model, predictors and heads
ject/backbone/encoder.py     hierarchical Swin encoder, skip pooling, dense decoder
ject/backbone/masking*.py    block masks and anatomy-aware mask sampling
ject/backbone/rope.py        3D rotary position embeddings for the predictors
ject/backbone/text.py        report text encoder
ject/backbone/region_text.py region and utterance alignment
ject/backbone/anatomy.py     anatomy id tables
```
