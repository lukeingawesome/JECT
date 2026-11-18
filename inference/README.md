# Inference: per-case encoder features

`extract_features.py` runs the pretrained JECT encoder over preprocessed CT cubes and writes one
feature vector per case for each of two readouts.

| readout | what | dim |
|---|---|---|
| `s3_mean` | mean of the stage-3 tokens (14 x 14 x 14 grid) | 512 |
| `s4_mean` | mean of the stage-4 tokens (7 x 7 x 7 grid) | 1024 |

`s4_mean` is the readout used for the linear-probe results.

## Input

One `.npz` per case with key `ct`: a raw-HU `int16` cube at 1.5 mm isotropic spacing, 256^3, in the
layout written by `preprocess/` (see `docs/PREPROCESSING.md`). The script centre-crops 224^3, clips
to the general window (-1000 to 1000 HU, window id 2), scales to [0, 1] and passes window id 2 to the
encoder. Other spacings, cube sizes or orientations run without error but give wrong features.

Cases are listed either as a directory of `.npz` files or as a CSV:

```
id,path
valid_1_a_1,/data/cubes/valid_1_a_1.npz
```

`id` is optional (default: the file name without `.npz`).

## Run

```bash
python inference/extract_features.py --checkpoint encoder.pt --input cases.csv --out feats.pt
python inference/extract_features.py --checkpoint encoder.pt --input cubes/ --out feats.pt --device cpu
```

| flag | default | |
|---|---|---|
| `--device` | `cuda` | `cpu` also works (a few seconds per case on 16 threads) |
| `--precision` | `bf16` on CUDA, `fp32` on CPU | the reference features were extracted with bf16 autocast |
| `--batch_size` | 4 | about 9 GB of GPU memory at 4 in bf16 |
| `--num_workers` | 4 | data loading processes |
| `--tokens_dir` | none | also save the stage-3 and stage-4 tokens of each case as `<id>.pt` (float16) |

## Output

```python
{"ids": [...], "s3_mean": Tensor[N, 512], "s4_mean": Tensor[N, 1024],
 "meta": {"checkpoint", "epoch", "input_size", "window_id", "precision", "npz_key"}}
```

Rows follow the input order. This file is read directly by `evaluate/ctrate_probe/run_probe.py` and
`evaluate/radchest/label_efficiency.py` (`--key s4_mean` or `s3_mean`).

## Precision

With the defaults (bf16 autocast on CUDA, batch 4) the features match the ones behind the reported
results bit for bit. fp32 features differ from them by a cosine similarity above 0.99999; that is
well inside probe seed noise, but use bf16 when comparing against the published numbers.
