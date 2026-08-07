"""Run a trained grounding model over a split and write per-case masks in MODEL space.

Output is one `<object_id>_pred.nii.gz` per case, shape (F, S, S, S) with F = the case's finding
count in the order given by the annotations CSV. `pack.py` then inverts these to the original CT
geometry.

Predictions are binarised here, with the threshold rule stated explicitly, because everything
downstream (inversion, scoring, zipping) operates on binary masks. Whatever rule you use MUST be
the one you validated with -- see `validate.py`.
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from ject.checkpoint import load_flexible_checkpoint
from ject.grounding.dataset import build_cfg, gwin
from ject.grounding.model import TextGroundingJEPA
from ject.data import make_crop_pad_transform, apply_crop_pad_3d
import ject.grounding.paths as paths


def binarise(prob: np.ndarray, mode: str, value: float) -> np.ndarray:
    """`abs`: prob >= value.   `relmax`: prob >= value * prob.max() (per finding).

    relmax adapts to each finding's confidence rather than assuming a shared scale, which matters
    when per-case peak probability varies. Choose one, validate with it, and submit with it.
    """
    if mode == "abs":
        return (prob >= value).astype(np.uint8)
    if mode == "relmax":
        pk = float(prob.max())
        return (prob >= value * pk).astype(np.uint8) if pk > 0 else np.zeros_like(prob, np.uint8)
    raise ValueError(f"unknown threshold mode {mode!r}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="backbone checkpoint (architecture source)")
    ap.add_argument("--weights", required=True, help="trained grounding best.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default=None, help="overrides $JECT_DATA")
    ap.add_argument("--rex_csv", default=None, help="overrides $JECT_REX")
    ap.add_argument("--text_emb", default=None, help="overrides $JECT_EMB")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out_res", type=int, default=224)
    ap.add_argument("--hi_ch", type=int, default=64)
    ap.add_argument("--fusion", default="xattn")
    ap.add_argument("--enc_skips", nargs="*", default=["s1", "s2", "s3"])
    ap.add_argument("--predictor_prior", action="store_true")
    ap.add_argument("--mup_prior", default=None, choices=[None, "s2", "s3", "s4"])
    ap.add_argument("--mup_logit", action="store_true")
    ap.add_argument("--txt_head", action="store_true")
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--deep_sup", action="store_true")
    ap.add_argument("--use_s4", action="store_true")
    ap.add_argument("--thr_mode", default="relmax", choices=["abs", "relmax"])
    ap.add_argument("--thr", type=float, default=0.75,
                    help="abs: probability cutoff. relmax: fraction of the per-finding peak.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse existing predictions in --out_dir. Only permitted when the run "
                         "signature (weights, split, res, out_res, threshold) is identical; "
                         "otherwise the run aborts rather than mixing masks from two configs.")
    a = ap.parse_args()

    if a.data_root: paths.DATA = a.data_root
    if a.rex_csv:   paths.REX_CSV = a.rex_csv
    if a.text_emb:  paths.TEXT_EMB = a.text_emb
    paths.T = (a.res,) * 3
    for nm, v, kind in (("--data_root", paths.DATA, "dir"), ("--rex_csv", paths.REX_CSV, "file"),
                        ("--text_emb", paths.TEXT_EMB, "file")):
        if not v or not (os.path.isdir(v) if kind == "dir" else os.path.isfile(v)):
            raise SystemExit(f"{nm} = {v!r} is unset or missing.")
    os.makedirs(a.out_dir, exist_ok=True)

    # RUN SIGNATURE. Bare file-existence is not evidence a prediction belongs to THIS run: change
    # --weights or --thr, rerun, and you would silently ship a mixture. Record what produced these
    # masks and refuse to mix.
    import hashlib
    sig = {"weights": os.path.abspath(a.weights), "weights_mtime": os.path.getmtime(a.weights),
           "split": a.split, "res": a.res, "out_res": a.out_res, "hi_ch": a.hi_ch,
           "thr_mode": a.thr_mode, "thr": a.thr, "fusion": a.fusion,
           "enc_skips": sorted(a.enc_skips), "predictor_prior": a.predictor_prior,
           "mup_prior": a.mup_prior, "mup_logit": a.mup_logit, "txt_head": a.txt_head,
           "film": a.film, "deep_sup": a.deep_sup, "use_s4": a.use_s4}
    sig_path = os.path.join(a.out_dir, "_run.json")
    existing = sorted(glob.glob(os.path.join(a.out_dir, "*_pred.nii.gz")))
    if existing:
        prev = json.load(open(sig_path)) if os.path.exists(sig_path) else None
        if not a.resume:
            raise SystemExit(
                f"{len(existing)} prediction(s) already in {a.out_dir}. Pass --resume to continue "
                f"an interrupted run with the SAME settings, or use a fresh --out_dir. Reusing "
                f"masks from a different checkpoint or threshold silently corrupts the result.")
        if prev != sig:
            diff = [k for k in sig if prev is None or prev.get(k) != sig[k]]
            raise SystemExit(
                f"--resume refused: {a.out_dir} was produced with different settings "
                f"(differs in: {diff}). Use a fresh --out_dir.")
    json.dump(sig, open(sig_path, "w"), indent=1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, _ = build_cfg(a.init)
    model = TextGroundingJEPA(cfg, out_res=a.out_res, fusion=a.fusion, hi_ch=a.hi_ch,
                              film=a.film, deep_sup=a.deep_sup, enc_skips=tuple(a.enc_skips),
                              predictor_prior=a.predictor_prior, mup_prior=a.mup_prior,
                              mup_logit=a.mup_logit, txt_head=a.txt_head,
                              use_s4=a.use_s4).to(dev).eval()
    load_flexible_checkpoint(model.art, a.init, verbose=True, allow_conditioner_reset=True)
    model.art.student_encoder.load_state_dict(model.art.target_encoder.state_dict())
    model.art.student_decoder.load_state_dict(model.art.target_decoder.state_dict())
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # FAIL CLOSED IN BOTH DIRECTIONS. `missing` = the model wants a tensor the weights lack.
    # `unexpected` = the weights carry a trained tensor this model has no slot for, i.e. training
    # used a head (--txt_head, --film, --mup_logit, ...) that this invocation omitted. Ignoring the
    # second case silently predicts with a DIFFERENT architecture and still writes submittable masks.
    live_missing = [k for k in missing if not k.startswith("art.")]
    live_unexpected = [k for k in unexpected if not k.startswith("art.")]
    if live_missing or live_unexpected:
        raise SystemExit(
            f"architecture mismatch between --weights and the flags given here.\n"
            f"  missing (model needs, weights lack): {len(live_missing)} e.g. {live_missing[:4]}\n"
            f"  unexpected (weights have, model lacks): {len(live_unexpected)} e.g. {live_unexpected[:4]}\n"
            f"Pass the same architecture flags the weights were trained with.")
    print("[predict] weights loaded; architecture matches exactly", flush=True)

    import pandas as pd
    df = pd.read_csv(paths.REX_CSV)
    df = df[df["rex_split"] == a.split]
    embs = torch.load(paths.TEXT_EMB, map_location="cpu", weights_only=True)
    print(f"[predict] {len(df)} {a.split} cases -> {a.out_dir} "
          f"(threshold {a.thr_mode} {a.thr})", flush=True)

    done = skipped = 0
    for _, row in df.iterrows():
        oid = str(row["object_id"])
        stem = oid[:-4] if oid.endswith(".npz") else oid
        dst = os.path.join(a.out_dir, f"{stem}_pred.nii.gz")
        if os.path.exists(dst):
            done += 1
            continue
        p = os.path.join(paths.DATA, oid)
        p = p if os.path.exists(p) else p + ".npz"
        if not os.path.exists(p):
            print(f"  [skip] {stem}: no volume at {p}")
            skipped += 1
            continue
        ct = np.load(p)["ct"]
        if ct.ndim == 4:
            ct = ct[0]
        sl, pd_ = make_crop_pad_transform(tuple(ct.shape), paths.T, random_crop=False)
        vol = torch.from_numpy(gwin(apply_crop_pad_3d(ct, sl, pd_, constant=-1000.0))).float()
        vol = vol.unsqueeze(0).unsqueeze(0).to(dev)

        findings = json.loads(row["rex_findings"])
        n_find = max(int(k) for k in findings) + 1          # keep the annotation's index order
        out = np.zeros((n_find, a.res, a.res, a.res), np.uint8)
        for k, txt in findings.items():
            if txt not in embs:
                print(f"  [skip finding] {stem}[{k}]: phrase absent from --text_emb")
                continue
            q = embs[txt].float().unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = model(vol, q, window_id=torch.full((1,), 2, device=dev),
                              ct_type_id=torch.zeros(1, dtype=torch.long, device=dev))
            logit = logit[0] if isinstance(logit, tuple) else logit
            logit = F.interpolate(logit[:, None].float(), size=paths.T, mode="trilinear",
                                  align_corners=False)[:, 0]
            out[int(k)] = binarise(torch.sigmoid(logit)[0].cpu().numpy(), a.thr_mode, a.thr)
        nib.save(nib.Nifti1Image(out, np.eye(4)), dst)      # affine is set during inversion
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(df)}", flush=True)
    print(f"[predict] wrote {done}, skipped {skipped}")
    if skipped:
        print(f"[predict] *** {skipped} case(s) have no volume and will be ABSENT downstream ***")


if __name__ == "__main__":
    main()
