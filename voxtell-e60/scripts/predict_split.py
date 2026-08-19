#!/usr/bin/env python
"""Predict a split's masks, sharded across GPUs, at several thresholds from ONE forward pass.

This is the submission path and the fast-validation path at once, because they want the same
thing: masks on the original CT grid, written exactly as rexrank_eval and the challenge zip
expect them.

THREE THINGS IT DOES THAT `val_sweep` DOES NOT

  * `--shard i/n` splits the case list, so two processes on two GPUs halve the wall clock.
    Prediction is the bottleneck now -- with the fast scorer a 200-case sweep is ~60 min of
    prediction against ~7 min of scoring -- and it shards perfectly because cases are
    independent.
  * `--thresholds` writes one prediction directory per threshold from a SINGLE forward pass.
    `VoxTellPredictor.predict_single_image` bakes in `sigmoid > 0.5`; this calls the public
    `predict_sliding_window_return_logits` underneath it instead, so N thresholds cost N cheap
    binarisations rather than N inference passes. Sweeping the threshold is then free, which
    matters because the metric's own measured behaviour -- global dice rises monotonically
    with predicted-instance count -- says 0.5 is unlikely to be the best operating point.
  * it runs on the `test` split, where there are no masks to score against, which is what a
    leaderboard submission actually needs.

WHY THERE IS NO FLIP TTA HERE, WHICH IS THE OBVIOUS THING TO ADD

85.7% of test prompts name left or right, 91.8% contain some directional word and 55.3% name
a lobe. Mirroring the image while the prompt still says "right middle lobe" contradicts the
request for the large majority of cases, and thoracic anatomy is not symmetric anyway -- three
lobes on the right, two on the left, plus the heart and the arch -- so a mirrored chest CT is
out of distribution even if the prompt were rewritten. Checkpoint ensembling is the TTA that
does not break the task; add it here if it earns its cost on val.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _predict_split_source_record() -> dict[str, str]:
    """Bind a prediction record to the exact producer bytes executing it."""
    source = Path(__file__).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "predict_split_source_path": str(source),
        "predict_split_source_sha256": digest.hexdigest(),
    }


def select_prediction_rows(findings_csv: Path, *, split: str):
    """The findings of one MICCAI split. `test` has no masks -- that is the submission set."""
    import pandas as pd

    if split not in {"val", "test"}:
        raise ValueError(f"prediction split must be val or test, got {split!r}")
    findings = pd.read_csv(Path(findings_csv), dtype={"VolumeName": str})
    return findings[findings["miccai_split"] == split].copy()


def postprocess_bbox(binary, ev, P):
    """`P.postprocess` restricted to the prediction's bounding box, which is the same trick
    rexrank_eval_fast uses on the scoring side and is exact for the same reason: everything
    outside the box is background, so connected components cannot cross it, sizes cannot
    change, and the top-K cap sees the same components.

    It matters because the full array is 512x512xZ and cc_filter runs once per finding per
    THRESHOLD -- with 4 thresholds the postprocess, not the GPU, became the bottleneck at
    73.8 s/case.
    """
    import numpy as np
    if not binary.any():
        return np.zeros_like(binary, dtype=np.uint8)
    sl = []
    for ax in range(binary.ndim):
        nz = np.any(binary, axis=tuple(i for i in range(binary.ndim) if i != ax))
        lo = int(nz.argmax()); hi = len(nz) - int(nz[::-1].argmax())
        sl.append(slice(lo, hi))
    out = np.zeros_like(binary, dtype=np.uint8)
    out[tuple(sl)] = P.postprocess(binary[tuple(sl)], ev)
    return out


def jitter(x: torch.Tensor, s: int, dim: int) -> torch.Tensor:
    """Translate `x` by `s` voxels along `dim`, REPLICATING the edge rather than wrapping.

    JECT's `infer.py:tta_fwd` uses `torch.roll`, and there it is harmless: its volume is a
    336^3 lung-anchored cube whose borders are air, so a wrap joins air to air. Here
    `predictor.preprocess` has already cropped to the nonzero box, so a wrap would join the
    apex to the abdomen and every tile crossing that seam would see anatomy that does not
    exist. Clamping the index instead duplicates `s` boundary slices -- the only voxels where
    the jitter is not exactly invertible, and they are outside the body, not through it.
    """
    idx = (torch.arange(x.shape[dim]) - s).clamp(0, x.shape[dim] - 1)
    return x.index_select(dim, idx)


def predict_prob(predictor, prepped: torch.Tensor, emb, tta: int) -> torch.Tensor:
    """Sigmoid probabilities (F, X, Y, Z) on the preprocessed grid; `tta` voxels of jitter.

    The port of JECT's translation TTA: 1 + 3 axes x 2 signs = 7 forwards, each un-jittered
    before it is averaged so that every pass votes on the same voxel. `prepped` is (C,X,Y,Z)
    and the logits are (F,X,Y,Z), so the spatial axes are 1..3 on both and the inverse jitter
    uses the same dim -- unlike JECT, where the batch axis makes it `ax - 1`.

    Averaged AFTER the sigmoid, as JECT does: the thresholds downstream act on probabilities,
    so that is the quantity whose mean is being asked for.

    NOT free and not obviously worth it here: 7 forwards is 7x the wall clock, and VoxTell's
    sliding window at `--tile-step 0.25` already sees each voxel at four different offsets
    inside a patch, which is part of what JECT's jitter averages over. JECT measured
    +0.0055-0.0091; expect less. Measure on val before spending it on a submission.
    """
    def fwd(x):
        return torch.sigmoid(
            predictor.predict_sliding_window_return_logits(x, emb).to("cpu").float())

    prob = fwd(prepped)
    if not tta:
        return prob
    for dim in (1, 2, 3):
        for s in (tta, -tta):
            prob += jitter(fwd(jitter(prepped, s, dim)), -s, dim)
    return prob / 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--out", required=True, help="parent dir; one subdir per threshold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="i/n -- this process takes cases i::n")
    ap.add_argument("--thresholds", default="0.5",
                    help="comma separated, e.g. 0.5,0.4,0.3,0.2")
    ap.add_argument("--fuse", choices=("off", "trunk", "skip"), default="off")
    ap.add_argument("--text-bank", default=None, help="defaults to <paired_root>/<TEXT_BANK_NAME>")
    ap.add_argument("--save-prob", type=float, default=None, metavar="FLOOR",
                    help="also store the sigmoid probabilities above FLOOR, sparsely, so any "
                         "threshold above it can be re-applied later WITHOUT re-predicting. "
                         "The leaderboard ranks your best of several submissions, so this is "
                         "the difference between a re-threshold costing minutes and costing a "
                         "40-minute pass over 300 volumes. 0.1 is a sensible floor.")
    ap.add_argument("--hu-norm", choices=("zscore", "ct"), default="zscore",
                    help="must match what the checkpoint was TRAINED under -- "
                         "train_voxtell.gate_hu_norm refuses a mismatch, because predicting "
                         "under a different input distribution than the weights were fitted "
                         "to produces masks that score wrong with every gate still green.")
    ap.add_argument("--tta", type=int, default=0, metavar="SHIFT",
                    help="translation-jitter TTA ported from JECT (external_weights/infer.py:"
                         "tta_fwd): average 7 forwards, the volume shifted by +-SHIFT voxels "
                         "along each axis. 6 is JECT's value. COSTS 7x THE WALL CLOCK and "
                         "changes the numbers -- validate on val before a submission. 0 = off.")
    ap.add_argument("--tile-step", type=float, default=None,
                    help="override the predictor's 0.5 sliding-window step. CHANGES THE "
                         "NUMBERS -- validate on val before using it for a submission.")
    a = ap.parse_args()

    import nibabel as nib
    import pandas as pd
    import scorer as P
    import train_voxtell as T
    from rexg_data.config import load_data_config
    from voxtell.inference.predictor import VoxTellPredictor
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    i, n = (int(x) for x in a.shard.split("/"))
    thresholds = [float(x) for x in a.thresholds.split(",")]
    assert all(0.0 < t < 1.0 for t in thresholds), f"thresholds must be in (0,1): {thresholds}"
    device = torch.device(a.device)

    cfg = load_data_config(REPO / "configs/rexg_res_2mm.yaml")
    paired, npy_root = Path(cfg.paired_root), Path(cfg.npy_root)
    bank_path = Path(a.text_bank) if a.text_bank else paired / T.TEXT_BANK_NAME

    rows_all = select_prediction_rows(paired / "findings.csv", split=a.split)
    names = sorted(rows_all["VolumeName"].unique())[i::n]
    index = pd.read_csv(npy_root / "index.csv", dtype={"VolumeName": str}).set_index("VolumeName")

    # The prompts must be present in the bank BEFORE any GPU time: the test split was never in
    # the training bank (it is built from has_seg == 1, and test has no masks), so a missing
    # embedding here is the difference between a submission and 6 wasted hours.
    bank, _, _ = T.load_text_bank(bank_path)
    need = {str(t).lower() for t in rows_all["text"]}
    missing = sorted(need - set(bank))
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(need)} {a.split} prompts are not in {bank_path.name}, "
            f"e.g. {missing[:2]}. Build a bank that covers this split first.")

    fuse = {"off": None, "trunk": True, "skip": False}[a.fuse]
    tok_bank = None
    if fuse is not None:
        arr, tok_index, lengths, meta = T.load_token_bank(T.token_bank_path_for(bank_path))
        T.gate_token_bank(arr, tok_index, lengths, meta, bank, list(rows_all["text"]),
                          enc=T.encoder_for(bank_path))
        tok_bank = (arr, tok_index, lengths)

    predictor = VoxTellPredictor(model_dir=T.MODEL_DIR, device=device,
                                 text_encoding_model=T.QWEN, embedding_bank=str(bank_path),
                                 use_precomputed_embeddings=False)
    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    hu_norm = T.gate_hu_norm(blob, a.hu_norm, f"predict_split {Path(a.checkpoint).name}")
    if hu_norm is not None:
        predictor.normalization = T.FixedAffineNormalization(hu_norm)
        print(f"*** hu_norm = ct, fixed affine {hu_norm} (not per-volume z-score) ***")
    sd = blob["network_weights"]

    # ONE definition of "this checkpoint is loadable", shared with train_voxtell.val_sweep,
    # so the two eval paths cannot disagree about what they refuse.
    sd = T.gate_unwrapped(sd)
    if fuse is None:
        predictor.network.load_state_dict(sd, strict=True)
    else:
        net, _ = T.build_network(random_init=True, deep_supervision=False, fuse_into_trunk=fuse)
        net.load_state_dict(sd, strict=True)
        predictor.network = net.to(device)
    predictor.network.eval()
    if a.tile_step is not None:
        predictor.tile_step_size = a.tile_step
        print(f"*** tile_step_size = {a.tile_step} (not the predictor default 0.5) ***")

    out_dirs = {t: Path(a.out) / f"thr{t:g}" for t in thresholds}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    prob_dir = Path(a.out) / "prob"
    if a.save_prob is not None:
        prob_dir.mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / f"predict_shard{i}of{n}.jsonl", "a", buffering=1)
    ev = P._load_evaluator(P.EVALUATOR)
    print(f"shard {i}/{n}: {len(names)} cases, thresholds {thresholds}, "
          f"scorer {P.EVALUATOR.name}", flush=True)

    # ---- provenance ------------------------------------------------------------------
    # A directory of .npz probabilities is unusable a week later if nobody can say which
    # checkpoint, which text bank and which fuse mode produced it. Every shard writes its own
    # record and CROSS-CHECKS the ones already there: shards that disagree on the checkpoint
    # would silently produce a prediction set that is half one model and half another, and no
    # metric would reveal it.
    import subprocess
    def _sha(path, n=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(n):
                h.update(chunk)
        return h.hexdigest()[:16]

    rec = {
        "checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_sha": _sha(a.checkpoint),
        "checkpoint_step": int(torch.load(a.checkpoint, map_location="cpu",
                                          weights_only=False).get("step", -1)),
        "arm": a.fuse, "split": a.split, "shard": a.shard, "device": a.device,
        "text_bank": str(bank_path.resolve()), "text_bank_sha": _sha(bank_path),
        "thresholds": thresholds, "save_prob_floor": a.save_prob,
        "tile_step": a.tile_step if a.tile_step is not None else 0.5,
        # Present ONLY when it is not the default: an absent key means off, so a TTA or
        # fixed-affine run is loudly not the same record as a plain one.
        **({"tta": a.tta} if a.tta else {}),
        **({"hu_norm": a.hu_norm} if a.hu_norm != "zscore" else {}),
        "scorer": P.EVALUATOR.name, "n_cases_this_shard": len(names),
        "started": time.strftime("%F %T"),
        "code_sha": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        **_predict_split_source_record(),
    }
    KEYS = ("checkpoint_sha", "arm", "split", "text_bank_sha",
            "predict_split_source_path", "predict_split_source_sha256",
            "tile_step", "save_prob_floor", "tta", "hu_norm")
    def _shard_produced(shard: str) -> int:
        """How many cases the shard behind `run_shard*.json` actually finished.

        Its per-case jsonl is the only per-shard evidence: masks land in a shared `thr*/`
        directory and cannot be attributed back to a shard.
        """
        try:
            i, n = (int(x) for x in str(shard).split("/"))
        except (ValueError, AttributeError):
            return -1          # unreadable record: treat as occupied, never as disposable
        log = Path(a.out) / f"predict_shard{i}of{n}.jsonl"
        if not log.is_file():
            return 0
        return sum(1 for line in log.read_text().splitlines() if line.strip())

    for sib in sorted(Path(a.out).glob("run_shard*.json")):
        other = json.loads(sib.read_text())
        # rec.get, not rec[k]: `tta` is absent when off, and half a prediction set averaged
        # over 7 forwards while the other half was not is exactly what this loop exists for.
        bad = {k: (other.get(k), rec.get(k)) for k in KEYS if other.get(k) != rec.get(k)}
        if not bad:
            continue
        # A disagreeing record is a conflict only if that shard PRODUCED something. A run that
        # died before its first case leaves the record behind with nothing beside it, and the
        # relaunch that corrects the settings then looks like the violation -- which cost three
        # restarts on 2026-08-18 before the message was read as "clear the directory". Judge by
        # output, not by the file existing.
        produced = _shard_produced(other.get("shard", ""))
        if produced != 0:
            raise SystemExit(
                f"shard disagreement with {sib.name} on {sorted(bad)}: {bad}\n"
                f"That shard already wrote {produced} case(s), so keeping both would make this "
                f"prediction set half one configuration and half another -- and no metric would "
                f"reveal it.\nEither finish this set with the ORIGINAL settings, or start a "
                f"clean one: rm -rf {a.out}   (or pass a different --out).")
        print(f"*** discarding {sib.name}: it disagrees on {sorted(bad)} but produced 0 cases, "
              f"so it is the record of an aborted run, not a sibling shard ***", flush=True)
        sib.unlink()
    (Path(a.out) / f"run_shard{i}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"  provenance: ckpt {rec['checkpoint_sha']} step {rec['checkpoint_step']} "
          f"arm {rec['arm']} bank {rec['text_bank_sha']} code {rec['code_sha']}", flush=True)

    t0 = time.time()
    for k, name in enumerate(names, 1):
        rows = rows_all[rows_all["VolumeName"] == name].sort_values("seg_channel")
        prompts = [str(t) for t in rows["text"]]
        ct = np.load(npy_root / "ct" / f"{name}.npy")
        data = np.ascontiguousarray(ct.transpose(2, 1, 0))

        emb = predictor.embed_text_prompts(prompts)
        if fuse is not None:
            arr, tok_index, lengths = tok_bank
            r = [tok_index[p.lower()] for p in prompts]
            predictor.network.set_tokens(
                torch.from_numpy(np.stack([np.asarray(arr[x]) for x in r]))[None].to(device),
                # arr.shape[1], not a module constant: the pad length is a property of the
                # bank that was actually loaded, and a constant can disagree with it.
                torch.stack([torch.arange(arr.shape[1]) < int(lengths[x])
                             for x in r])[None].to(device),
                pooling=T.encoder_for(bank_path).pooling)
        try:
            prepped, bbox, orig_shape = predictor.preprocess(data)
            prob = predict_prob(predictor, prepped, emb, a.tta)
        finally:
            # Always, even on an exception: a stale token set would score the NEXT volume
            # against this one's sentences, and no metric would show it.
            if fuse is not None:
                predictor.network.clear_tokens()
        pos = str(index.loc[name, "patient_position"])

        if a.save_prob is not None:
            # Sparse because the volume is ~0 almost everywhere: findings occupy 0.03-0.8% of
            # it, so coordinates + fp16 values are ~1 MB against 175 MB dense per finding.
            # Stored in MODEL space (pre-insert_crop), with the bbox and the original shape,
            # because that is the only form from which any threshold can be re-applied exactly.
            pk = (prob >= a.save_prob).numpy()
            idx = np.nonzero(pk)
            import pickle
            np.savez_compressed(
                prob_dir / f"{name}.npz",
                coords=np.stack(idx).astype(np.int32),
                values=prob.numpy()[idx].astype(np.float32),   # NOT fp16: see rethreshold.py
                prob_shape=np.array(prob.shape, np.int32),
                orig_shape=np.array(orig_shape, np.int32),
                bbox=np.frombuffer(pickle.dumps(bbox), np.uint8),
                floor=np.array([a.save_prob], np.float32),
                position=np.frombuffer(pos.encode(), np.uint8))

        for thr in thresholds:
            seg = np.zeros([prob.shape[0], *orig_shape], dtype=np.uint8)
            seg = insert_crop_into_image(seg, (prob > thr).numpy().astype(np.uint8), bbox)
            seg = seg.transpose(0, 3, 2, 1)                      # back to RAS (nx, ny, nz)
            packed = np.ascontiguousarray(np.stack(
                [P.ras_to_lps(postprocess_bbox(seg[c] > 0, ev, P), pos) for c in range(len(rows))]
            )).astype(np.uint8)
            nib.save(nib.Nifti1Image(packed, np.eye(4)), str(out_dirs[thr] / name))

        dt = (time.time() - t0) / k
        log.write(json.dumps({"name": name, "findings": len(rows), "i": k, "n": len(names),
                              "s_per_case": round(dt, 2)}) + "\n")
        if k % 10 == 0 or k == len(names):
            print(f"  [{k}/{len(names)}] {dt:.1f}s/case  eta {(len(names)-k)*dt/60:.0f} min",
                  flush=True)
    print(f"shard {i}/{n} done in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
