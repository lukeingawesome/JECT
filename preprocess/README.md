# Preprocessing

Turns CT-RATE NIfTI volumes and their TotalSegmentator / ReXGrounding masks into
one npz per case. Run every command from the repository root.

## 1. Stage the files

Everything downstream addresses the data through one dataset root:

```
<data-root>/original/<case_id>.nii.gz       chest CT               (required)
<data-root>/totalseg/<case_id>.nii.gz       TotalSegmentator mask  (required)
<data-root>/rexgrounding/<case_id>.nii.gz   ReXGrounding masks     (optional)
```

`stage` collects files from wherever they were produced into that layout,
symlinking by default so nothing is duplicated on disk:

```bash
python preprocess/cli.py stage \
  --data-root /data_1/ct \
  --ct        /raw/ct_niftis \
  --totalseg  /raw/totalseg_output \
  --grounding /raw/rexgrounding

python preprocess/cli.py coverage --data-root /data_1/ct
```

Sources may be directories (searched recursively) or single files; the case id
is the filename without its `.nii`/`.nii.gz` suffix. Use `--link-mode
{symlink,hardlink,copy}`, `--dry-run` to preview, `--overwrite` to replace.
If the data is already laid out this way, skip straight to step 2.

## 2. Keep one volume per study

CT-RATE names volumes `{split}_{patient}_{study}_{reconstruction}.nii.gz` and
reconstructs every scan with several kernels — "25,692 non-contrast chest CT
volumes, expanded to 50,188 through various reconstructions". Those siblings are
different renderings of one acquisition: a smooth mediastinal kernel and a sharp
lung kernel, often at different matrices and slice thicknesses. Training on all
of them means training on the same scan several times over, so `select` keeps
one volume per study:

```bash
python preprocess/cli.py select \
  --data-root /data_1/ct \
  --metadata  /data_1/ct/validation_metadata.csv \
  --manifest  /data_1/preprocessed_full/selected_volumes.csv \
  --rejected  /data_1/preprocessed_full/rejected_volumes.csv
```

Per study, in order: prefer the `--native-matrix 512` in-plane matrix → prefer an
axial series over a reformat → rank the rest with `--rule`. `soft512` (default)
ranks contrast-phase series last, then soft-tissue kernel over lung over bone,
standard matrix, thinnest slice, chest-limited z-coverage, and the lowest
reconstruction index as a tie-break. `--rule recon1` instead keeps the `_1`
volume of every study.

Neither preference can empty a study: one that offers only a non-512 matrix —
116 of the 1564 validation studies — still contributes it, flagged by the
manifest's `native_matrix` column, and no study is lost for being a reformat.
`--native-matrix 0` disables the matrix check.

A volume carrying a ReXGrounding mask is never dropped, since the mask annotates
that one reconstruction. When a sibling wins the study, both rows appear and the
`kept_for` column tells them apart (`study_pick` / `grounding`) — on train that
is 392 extra rows, and it is the difference between keeping 2386 and all 2778
grounding masks.

**Know what the matrix preference costs you.** CT-RATE's 768/1024 volumes are
genuine sharp-kernel reconstructions with high in-plane resolution, not
interpolated copies — and they are the anisotropic ones (every 768/1024 volume
in the validation split has 1.5 mm slices, against a median of 1.0 mm at 512).
Preferring 512 buys through-plane resolution and a smoother, mediastinal-window
rendering at the cost of in-plane detail. A task that wants the opposite should
set `--native-matrix 1024` or `0`.

Without `--metadata` the axial test falls back to the NIfTI affine, which still
catches reformatted and oblique volumes. Headers are read straight from the
first 348 bytes of the gzip stream, so scanning never decompresses an image —
the 3039-volume validation split takes about 10 s.

The manifest records what was kept and why (`kernel`, `nx`, `dz`, `z_extent_mm`,
`native_matrix`, `n_recons_in_study`, `dropped_siblings`); `--rejected` records
what was dropped, with a `reason` of `non_native_matrix`, `duplicate_recon`,
`not_axial` or `unreadable`. Feed the manifest to step 3 with `--id-list`.

## 3. Convert to npz

```bash
python preprocess/cli.py run \
  --data-root /data_1/ct \
  --out       /data_1/preprocessed_full \
  --config    preprocess/configs/ctrate_full.yaml \
  --id-list   /data_1/preprocessed_full/selected_volumes.csv
```

Per case the pipeline: loads the three volumes → puts the multi-channel
grounding masks in channel-first layout and gives them the CT's affine →
reorients to LPS → resamples to the target spacing (trilinear for the CT,
nearest for masks) → optionally windows HU to `[0, 1]` → pads (CT with air HU,
masks with 0) → crops to the target shape around the **lung bounding box** taken
from the TotalSegmentator lung lobes → casts and saves.

Output: `<out>/<case_id>.npz` holding `ct`, `totalseg` and, when available,
`rex`, all `(C, D, H, W)` — plus `preprocessing_metadata.csv` (one row per case
with source shape/spacing, output shapes and status) and
`preprocess_config.yaml` (the exact settings used).

Cases are skipped rather than fatal: a missing TotalSegmentator mask, a scan
below `min_source_shape`, or a mask without lung labels is recorded in the
metadata `status` column and the run continues.

Useful flags: `--id-list` to restrict to a whitelist — a `select` manifest CSV,
a JSON list, a `split -> ids` mapping (with `--splits train valid`), or one id
per line — plus `--limit N` for a smoke test, `--num-workers`, `--overwrite`,
and any config key as a flag (`--target-shape 256 256 224`,
`--hu-window -1000 200`, …). Precedence: CLI flag > YAML config > defaults in
`settings.py`.

## 4. Check the result

```bash
python preprocess/visualize.py --npz-dir /data_1/preprocessed_full --num-cases 2 --out figures
```

Samples cases (preferring ones with grounding masks) and writes one figure each:
axial / coronal / sagittal slices of the CT alone, with the TotalSegmentator
labels, and with the ReXGrounding phrases. Slices are chosen where the grounding
mask is largest, falling back to the lung bounding box.

## End to end

`run.sh` runs stage → select → convert → visualize; the paths
are environment variables at the top of the file:

```bash
DATA_ROOT=/data_1/ct OUT_DIR=/data_1/preprocessed_full \
METADATA=/data_1/ct/validation_metadata.csv \
bash preprocess/run.sh
```

## Configuration

`preprocess/configs/*.yaml` holds the presets (`ctrate_full`: 1 mm, 384³;
`ctrate_half`: 1.5 mm, 256×256×224). Every key is documented in
`preprocess/settings.py` — target spacing/shape, orientation, HU window, storage
dtypes, padding values, the lung labels used for cropping, the minimum accepted
source shape, compression and worker count.


## Files

| file | role |
| --- | --- |
| `cli.py` | CLI: `stage`, `select`, `coverage`, `run` |
| `run.sh` | end-to-end example run |
| `visualize.py` | sanity-check figures from the npz output |
| `settings.py` | `PreprocessConfig`: every setting, YAML/CLI merge, validation |
| `layout.py` | modalities, dataset layout, case discovery, id whitelists |
| `staging.py` | step 1: collect NIfTI files into the layout |
| `selection.py` | step 2: one native reconstruction per study, NIfTI header reader |
| `transforms.py` | MONAI transforms: reference geometry, lung-bbox crop |
| `pipeline.py` | step 3: transform chain and per-case conversion |
| `runner.py` | multiprocessing driver, metadata, run summary |
| `geometry.py` | dependency-free cropping maths |
