#!/usr/bin/env python3
"""Rebuild the phrase-embedding file that `$JECT_EMB` points at.

    python tools/make_phrase_embeddings.py --rex_csv final_rex_miccai.csv --out rex_all_embs.pt

Output is a plain `.pt` dict `{phrase_string: 1024-d float32 tensor}`, L2-normalised, keyed by the
EXACT strings in the CSV's `rex_findings` JSON. That exactness matters: `dataset.py` does
`if txt in embs` and **silently skips** any phrase it cannot find, so a key that differs by one
character removes that finding from training with no error. Verify the counts afterwards --
`train: 7687 / val: 381` (SETUP.md §3a).

The encoder is Chest2Vec-0.6B, published with its LoRA already merged into the weights and its
tokenizer and modeling code bundled, so `trust_remote_code=True` is all it takes -- no `chest2vec`
package, and no separate download of the base Qwen3-Embedding. `model.embed()` owns the
tokenisation, the EOS/last-token pooling and the batching, which is what makes this script short.

Needs `transformers` (and `accelerate` for device placement), which are NOT in requirements.txt --
they are a one-off cost for producing this file, not a training dependency:

    pip install transformers accelerate

Pass no instruction. Chest2Vec follows the Qwen3-Embedding convention where a query is wrapped in
an instruction template and a document is not; the phrases here were embedded as documents, and
adding an instruction would produce a different vector for the same string.

The choice of encoder is not load-bearing as long as it is CONSISTENT: the grounding head's
`query_proj` (Linear 1024 -> 256) trains from scratch. Swap encoders and you must rebuild this file
for every phrase and retrain the head; mixing two encoders in one file is the failure to avoid.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import torch

DEFAULT_MODEL = "lukeingawesome/chest2vec_0.6b_chest"


def phrases_from_csv(path: str) -> list[str]:
    """Every distinct finding phrase in the CSV, sorted so reruns are byte-stable."""
    df = pd.read_csv(path)
    seen: set[str] = set()
    for row in df["rex_findings"]:
        for _, txt in json.loads(row).items():
            seen.add(str(txt))
    return sorted(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rex_csv", required=True, help="ReXGrounding CSV (column `rex_findings`)")
    ap.add_argument("--out", required=True, help="destination .pt")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF repo id or local snapshot dir (default {DEFAULT_MODEL})")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N phrases (smoke test)")
    ap.add_argument("--verify", default=None,
                    help="compare the result against an existing .pt and report cosine similarity")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    phrases = phrases_from_csv(a.rex_csv)
    if a.limit:
        phrases = phrases[:a.limit]
    print(f"[emb] {len(phrases)} unique phrases from {a.rex_csv}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    # No attn_implementation= here: the wrapper intercepts that kwarg and rejects it (it does not
    # declare SDPA support), while the inner Qwen3 encoder reads it from its own config and already
    # defaults to sdpa. Passing it raises rather than doing anything useful.
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).eval().to(a.device)

    embs: dict[str, torch.Tensor] = {}
    for i in range(0, len(phrases), a.batch):
        chunk = phrases[i:i + a.batch]
        with torch.no_grad():
            vecs = model.embed(chunk, tokenizer=tok, batch_size=len(chunk))
        for phrase, v in zip(chunk, vecs):
            embs[phrase] = v.detach().float().cpu()
        if i % (a.batch * 50) == 0:
            print(f"  {i + len(chunk)}/{len(phrases)}  dim={vecs.shape[1]}", flush=True)

    dim = next(iter(embs.values())).shape[0]
    norm = next(iter(embs.values())).norm().item()
    print(f"[emb] {len(embs)} phrases, dim {dim}, L2 norm {norm:.4f}", flush=True)

    if a.verify:
        ref = torch.load(a.verify, map_location="cpu", weights_only=True)
        shared = [p for p in embs if p in ref]
        if not shared:
            print(f"[verify] no shared keys with {a.verify} — different CSV?", file=sys.stderr)
            return 1
        cos = torch.stack([torch.dot(embs[p], ref[p].float()) /
                           (embs[p].norm() * ref[p].float().norm()) for p in shared])
        print(f"[verify] {len(shared)} shared phrases vs {a.verify}: "
              f"cosine min {cos.min():.6f}  mean {cos.mean():.6f}", flush=True)
        # Merged-LoRA weights reproduce the original adapter path to cosine >= 0.9996 (bf16
        # rounding only), so anything below 0.999 means a genuinely different encoder or pooling.
        if cos.min() < 0.999:
            print("[verify] FAILED — not the same encoder/pooling as the reference.", file=sys.stderr)
            return 1
        print("[verify] OK", flush=True)

    if not a.limit:
        torch.save(embs, a.out)
        print(f"[emb] wrote {a.out}", flush=True)
    else:
        print(f"[emb] --limit set, not writing {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
