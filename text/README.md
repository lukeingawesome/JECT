# Text targets

This stage turns section-structured report findings into the text inputs the trainer reads:

| output | trainer flag | produced by |
|---|---|---|
| manifest CSV (`object_id`, `split`, `utterances_json`, optional `npz_root`, `utt_root`) | `--data_csv` | `make_manifest.py` |
| `<stem>_utterances4b.pt` per case | read from `utt_root` (or next to the npz), `--utt_sidecar_suffix _utterances4b.pt` | `encode_chest2vec.py` |
| `<stem>_sr_targets4b.pt` per case | `--sr_sidecar_root` | `encode_chest2vec.py` |
| train-split section statistics | `--mup_target_stats` | `section_stats.py` |

`<stem>` is `object_id` without a trailing `.npz`.

## Files

| file | purpose |
|---|---|
| `sections.py` | parser: findings text to utterances; canonical section list, aliases, anatomy names, negation and uncertainty patterns |
| `build_targets.py` | utterances to per-case section targets (`positive`, `normal`, `not_assessed`) and the positive report text; optional JSONL dump |
| `make_manifest.py` | table of (object_id, split, findings) to the training manifest |
| `encode_chest2vec.py` | chest2vec embeddings of utterances, sections and report; writes both sidecars; resumable |
| `section_stats.py` | global and per-section means of the section embeddings over the train split |

## Input format

The parser expects section-structured findings: one `[Section]` header line per anatomical section, followed by one
`- ` bullet per finding sentence.

```
[Lungs]
- Millimetric nonspecific nodules in both lungs.
- No mass or infiltrative lesion in either lung.
[Pleura]
- No pleural effusion.
```

Recognised headers are `Lungs`, `Airways & Trachea`, `Pleura`, `Mediastinum & Hila`, `Cardiovascular`, `Chest Wall`,
`Bones / Spine`, `Upper abdomen`, `Lower neck`, `Breast & Axilla`, `Others` and the spellings `Trachea & Airways`,
`Upper Abdomen`, `Lower Neck`, `Breast`. Any other header is kept as the utterance's section and counted as `Others`
when section targets are built. CT-RATE's raw reports are free text and are not in this format, so they must first be
restructured into section headers and one-sentence bullets (for example with a language model) before this stage.

## Utterances

Each bullet becomes `{"text", "section", "anatomy", "negated", "uncertain"}`. The text is kept verbatim; `anatomy` is
derived from the section; `negated` and `uncertain` are set by the regular expressions in `sections.py`. A manifest row
keeps at most 32 utterances, the number of slots in an utterance sidecar, because the trainer
(`--require_cached_text_alignment`) only uses a sidecar that covers every manifest utterance.

`encode_chest2vec.py` and `build_targets.py` read `utterances_json` as given. A manifest built by other means, for
example one that keeps only positive findings, is encoded the same way.

## Section targets

Per case and per canonical section:

- `positive`: at least one utterance that is neither negated nor uncertain; target text = those utterances joined.
- `normal`: no positive utterance, at least one negated one; target text = the negated utterances joined.
- `not_assessed`: not mentioned, or only uncertain utterances; no target, the slot is masked in the loss.

`report_pos` joins the positive section texts in canonical order, or is `No significant abnormality was detected.` when
there are none.

## Sidecar contents

`<stem>_utterances4b.pt`: `tokens` [32, 2560] float32 (zero rows past the last utterance), `attn_mask` [32] bool,
`texts`, `sections`, `negated`, `uncertain`.

`<stem>_sr_targets4b.pt`: `section_names`, `section_status`, `section_tokens` [n, 2560] float32 for the positive and
normal sections in canonical order, `section_texts`, `report_token` [2560] float32, `report_text`.

Embeddings come from `chest2vec/chest2vec_4B` (`embed_texts`: end-of-text token appended, left padding, last-token
pooling, L2 normalisation), computed in bfloat16 and stored at full width. The trainer slices a Matryoshka prefix
(`--mup_matryoshka_dim`) and re-normalises it.

## Statistics

`section_stats.py` slices each section token of every train case to 1024, 512 and 256 dimensions, L2-normalises each
prefix and averages, globally (`dims`) and per section (`conditional_dims["section_name"]`, shape [11, d]). Rows count
as train when `split` starts with `train`.

## Commands

```bash
# 1. manifest from a table with section-structured findings
python text/make_manifest.py --table reports.csv --id_col object_id --split_col split \
    --findings_col findings --utt_root targets/utt --out manifest.csv

# optional: inspect the section targets
python text/build_targets.py --manifest manifest.csv --out targets.jsonl

# 2. embeddings and sidecars (one GPU; rerun the same command to resume)
python text/encode_chest2vec.py --manifest manifest.csv --utt_root targets/utt --sr_root targets/sr \
    --device cuda --batch 256

# 3. train-split statistics
python text/section_stats.py --manifest manifest.csv --sr_root targets/sr --out section_target_stats.pt
```

Then train with `--data_csv manifest.csv --sr_sidecar_root targets/sr --mup_target_stats section_target_stats.pt`.
Without `--utt_root` in step 1, the utterance sidecars must be written next to the npz files (`--utt_root` in step 2
set to the cube directory).
