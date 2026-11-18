# Evaluation

Two frozen-feature benchmarks. Both read feature files written by `inference/extract_features.py`
(or any file in the layouts listed in `common.py`), run on CPU, and need only `torch`, `numpy`,
`pandas` and `scikit-learn`.

| path | what |
|---|---|
| `common.py` | feature loading, linear probe, macro AUROC, patient-cluster bootstrap |
| `ctrate_probe/run_probe.py` | CT-RATE linear-probe benchmark on the 18 CT-RATE abnormality labels |
| `ctrate_probe/labels/` | place `ct_rate_labels.csv` here (see below) |
| `radchest/label_efficiency.py` | RAD-ChestCT label-efficiency curve, 0 to 100 % of the training labels |
| `radchest/data/`, `radchest/ids/` | RAD-ChestCT labels, splits, training subsets and bootstrap draws |

## 1. CT-RATE linear probe

### Labels

Use `ct_rate_labels.csv`: the 18 abnormality labels distributed with CT-RATE. Build it once from the
multi-abnormality label files of the CT-RATE release (train and validation) and save it to
`evaluate/ctrate_probe/labels/`:

```python
import pandas as pd
parts = []
for split, path in [("train", "train_predicted_labels.csv"), ("val", "valid_predicted_labels.csv")]:
    df = pd.read_csv(path)                                     # VolumeName + 18 label columns
    df.insert(0, "split", split)
    df.insert(0, "object_id", df.VolumeName.str.replace(".nii.gz", ".npz", regex=False))
    parts.append(df)
pd.concat(parts).to_csv("evaluate/ctrate_probe/labels/ct_rate_labels.csv", index=False)
```

Columns: `object_id`, `VolumeName`, `split` (`train` or `val`), then the 18 labels.

### Features

Extract features for every CT-RATE train and validation scan listed in the label files (one cube per
`object_id`, e.g. `train_1_a_1.npz`). Train and validation may be in one file or several:

```bash
python inference/extract_features.py --checkpoint encoder.pt --input ctrate_cubes/ --out ctrate_feats.pt
```

### Run

```bash
python evaluate/ctrate_probe/run_probe.py --features ctrate_feats.pt --key s4_mean \
    --name ject_s4 --out_dir probes/
```

| step | setting |
|---|---|
| labels | `ct_rate_labels.csv`, the 18 CT-RATE abnormality labels. Uncertain (-1) and missing values count as negative. |
| validation | one scan per CT-RATE validation study: the lowest-sorted scan that has features (1,564 studies, 1,304 patients) |
| train | the lowest-sorted scan of each training study |
| probe | one linear layer, class-balanced BCE (positive weight = negatives / positives), Adam lr 1e-3, weight decay 1e-4, batch 512, cosine decay over 50 epochs, gradient clip 1.0; the epoch with the best validation macro AUROC is kept |
| seeds | 41 to 45; the reported score uses the mean sigmoid of the five probes |
| metric | macro AUROC over labels; 95 % CI from a patient-cluster bootstrap (CT-RATE patient number, 3,000 draws, seed 3026) |

Train scans without features are dropped and counted in the log; a missing validation scan is an error.

### Outputs

```
probes/<task>/<name>_s<seed>/best_linear_probe.pt    {"model_state_dict", "label_cols", "epoch", "macro_auc"}
probes/<name>_results.json                           per task: macro AUROC, CI, AUPRC, per-seed and per-label results,
                                                     and the macro AUROC of every bootstrap draw
```

The draws depend only on the cohort and `--boot_seed`, so the per-draw values of two runs are paired:
the CI of their difference is the percentile interval of the per-draw differences.

### Expected result

Released encoder, `s4_mean`, with the default settings: CT-RATE-18 macro AUROC 0.872 [0.865, 0.878].

With the default seeds the probes, and so these numbers, are reproduced exactly on CPU; a different
thread count or BLAS library changes the probe weights only at float rounding.

### Optional second label set

`--tasks ct-rate-18 chest2vec-44` also scores 44 report-derived findings when `ct_rate_chest2vec_probe.csv`
is present in the labels folder (not distributed here). Both label sets are then scored on the 1,544
validation scans that file covers; the released encoder reaches 0.848 [0.840, 0.855] on chest2vec-44 and
the same 0.872 on CT-RATE-18.

## 2. RAD-ChestCT label efficiency

How well frozen features adapt to RAD-ChestCT when only part of its training labels is available.

| item | content |
|---|---|
| `data/radchest_labels16.csv` | object_id, split, MRN_DEID (patient key) and the 16 labels, 3,629 scans |
| `data/label_map_ctrate18_to_radchest16.json` | CT-RATE-18 label to RAD-ChestCT findings; positive if any mapped finding is positive |
| `ids/splits.csv` | radchest_train 2,285 / radchest_valid 984 / radchest_test 360 |
| `ids/train_subsets_seed3026.json` | training object_ids per fraction; 5 repeats at 1 to 50 %, 1 at 100 % |
| `ids/bootstrap_draws_seed5150.json` | the 1,000 test resamples as object_id lists, as the script draws them |

The 16 labels are the CT-RATE-18 set without 'Medical material' and 'Lymphadenopathy'.

### Protocol

| fraction | setting |
|---|---|
| 0 % | no RAD-ChestCT labels: the five CT-RATE-18 probes from section 1 applied unchanged and averaged, 16 columns kept |
| 1 to 100 % | per repeat, a linear head on the training subset (patients drawn at random, all their scans): z-scored features, class-balanced BCE with the positive weight capped at 50, AdamW lr 1e-3, weight decay 1e-4, batch 128, 50 epochs; the epoch with the best macro AUROC on the full radchest_valid is kept; seeds 41 to 43 averaged |
| metric | macro AUROC on radchest_test over labels with at least 5 positives and 5 negatives, averaged over repeats; 95 % CI from 1,000 patient-cluster draws (seed 5150), shared by every entry and fraction so deltas are paired |

### Run

Each entry is `name=feature_file:key[:probe_name]`. `feature_file` is read from `--cache_dir`;
`probe_name` (optional, needed for the 0 % point) refers to `<probe_dir>/<probe_name>_s41..45/`.

```bash
python inference/extract_features.py --checkpoint encoder.pt --input radchest_cubes/ --out feats/radchest_ject.pt

python evaluate/radchest/label_efficiency.py --cache_dir feats/ --probe_dir probes/ct-rate-18 \
    --subsets_in evaluate/radchest/ids/train_subsets_seed3026.json --n_boot 1000 \
    --ours ject --out results/radchest_label_efficiency.json \
    --entries ject=radchest_ject.pt:s4_mean:ject_s4 other=radchest_other.pt:emb
```

RAD-ChestCT cubes must follow the same input contract as CT-RATE (LPS, 1.5 mm, lung-centred 256^3,
see `docs/PREPROCESSING.md`). A full curve takes a few minutes per entry on CPU.

### Outputs

The log prints the curve point per fraction, its CI and the paired deltas against `--ours`. The JSON
holds, per fraction, `entries` (macro AUROC, SD over repeats, CI) and `pairs` (delta, CI, separated
when the CI excludes 0).

### Expected result

Features of the CT-RATE-aligned image tower (pooled readout) with its CT-RATE-18 probes:

| labels used | 0 % | 1 % | 2 % | 5 % | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|---|---|---|---|
| macro AUROC | 0.791 | 0.669 | 0.699 | 0.752 | 0.771 | 0.792 | 0.801 | 0.810 |
