# JECT: Addressing the Context Shortcut in Joint-Embedding Predictive Learning for Chest CT

Training and evaluation code for JECT, a 3D chest-CT foundation model. JECT pretrains a hierarchical encoder with a
student/teacher latent-prediction objective. Masked section prediction (MSP) adds a report-based target: the predictor
fills in the embedding of each report section from the masked image. Weak grounding ties report findings to anatomy.

Weights of the 224³ model with a report-aligned readout are released separately.

## Pipeline

| step | folder | what it does |
|---|---|---|
| 1. Volumes | `preprocess/` | NIfTI to npz cubes: reorient, resample, lung-centred crop, raw HU int16 |
| 2. Report targets | `text/` | parse section-structured findings, encode them with chest2vec, compute section statistics, write the training manifest |
| 3. Training | `train.sh`, `train.py`, `ject/`, `launch.py` | pretraining at 224³ (`train.sh`, released settings by default) and the cooldown at 320³ |
| 4. Inference | `inference/`, `encoder/` | stage-3 and stage-4 features from a trained encoder |
| 5. Evaluation | `evaluate/` | CT-RATE linear probes and the RAD-ChestCT label-efficiency test |

Each folder has its own README with inputs, outputs and commands. Training details are in `docs/TRAINING.md`.

## Quickstart

```bash
pip install -r requirements.txt

# 1. cubes (1.5 mm, 256^3 for the 224 model)
python preprocess/cli.py run --data-root /data/ct --out /data/cubes_256 --config preprocess/configs/ject_half_256.yaml

# 2. report targets
python text/make_manifest.py --help
python text/encode_chest2vec.py --help
python text/section_stats.py --help

# 3. pretraining with the released settings, then the cooldown from the full training checkpoint
DATA_CSV=manifest.csv DATA_ROOT=/data/cubes_256 SR_ROOT=/data/text/sr TARGET_STATS=/data/text/section_stats.pt \
    OUT=runs/pretrain GPUS=0,1 bash train.sh
python launch.py --recipe cooldown_320 --init_checkpoint runs/pretrain/checkpoint_final.pt --data_csv manifest_384.csv \
    --data_root /data/cubes_384 --sr_root /data/text/sr --target_stats /data/text/section_stats.pt --out runs/cooldown --gpus 0,1

# 4. features
python inference/extract_features.py --help

# 5. benchmarks
python evaluate/ctrate_probe/run_probe.py --help
python evaluate/radchest/label_efficiency.py --help
```

`--epochs` is fixed at launch because the learning-rate schedule depends on it. Crops are in voxels: use 224 with
1.5 mm cubes and 320 with 1 mm cubes.

## Checks

`tests/check_recipe_matches_checkpoint.py` confirms that a recipe, parsed by this `train.py`, builds exactly the
architecture and configuration of a trained checkpoint (strict load of every tensor, identical config apart from paths).

## Requirements

Python 3.10 or newer, PyTorch 2.4 or newer with CUDA for training, and the packages in `requirements.txt`. The text stage
needs `transformers` and access to the chest2vec models on Hugging Face.

## Citation

```bibtex
@misc{ject2026,
  title  = {{JECT}: Addressing the Context Shortcut in Joint-Embedding Predictive Learning for Chest {CT}},
  author = {TBD},
  year   = {2026}
}
```

## License

The code in this repository is released under the MIT License (`LICENSE`). Released model weights are licensed
separately under CC BY-NC-SA 4.0 and may not be used commercially.
