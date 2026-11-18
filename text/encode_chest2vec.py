"""Encode utterance and section targets with chest2vec and write the trainer's sidecars.

Per case (stem = object_id without ".npz"):
  <utt_root>/<stem>_utterances4b.pt
      tokens [32, H] float32, attn_mask [32] bool, texts, sections, negated, uncertain
  <sr_root>/<stem>_sr_targets4b.pt
      section_names, section_status, section_tokens [n, H] float32, section_texts,
      report_token [H] float32, report_text

Embeddings use the model's embed_texts: EOS appended, left padding, last-token
pooling, L2 normalisation. The full hidden size is stored; Matryoshka slicing
happens in the trainer. Cases whose two sidecars both exist are skipped, so an
interrupted run can be restarted with the same command.

  HF_HOME=... python encode_chest2vec.py --manifest manifest.csv \
      --utt_root targets/utt --sr_root targets/sr --device cuda --batch 256
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import torch

from build_targets import build_case, encodable_sections

S_MAX = 32  # utterance slots per sidecar
UTT_SUFFIX = "_utterances4b.pt"
SR_SUFFIX = "_sr_targets4b.pt"


def stem_of(object_id: str) -> str:
    s = str(object_id)
    return s[:-4] if s.endswith(".npz") else s


def save_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with object_id, utterances_json")
    ap.add_argument("--utt_root", required=True)
    ap.add_argument("--sr_root", required=True)
    ap.add_argument("--model", default="chest2vec/chest2vec_4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--batch", type=int, default=256, help="texts per forward pass")
    ap.add_argument("--chunk", type=int, default=2000, help="cases pooled and length-sorted per round")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="encode at most this many cases (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.utt_root, exist_ok=True)
    os.makedirs(a.sr_root, exist_ok=True)

    df = pd.read_csv(a.manifest, usecols=["object_id", "utterances_json"])
    todo = []
    for r in df.itertuples(index=False):
        stem = stem_of(r.object_id)
        if (os.path.exists(os.path.join(a.utt_root, stem + UTT_SUFFIX))
                and os.path.exists(os.path.join(a.sr_root, stem + SR_SUFFIX))):
            continue
        todo.append((stem, json.loads(r.utterances_json)))
        if a.limit and len(todo) >= a.limit:
            break
    print(f"{len(todo)} cases to encode, {len(df)} in manifest", flush=True)
    if not todo:
        return

    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
    hidden = int(model.config.hidden_size)

    def embed(texts: list[str]) -> torch.Tensor:
        """[N, H] float32 in input order; batches are formed over length-sorted texts."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), hidden)
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            e = model.embed_texts([texts[i] for i in idx], max_len=a.max_tokens, batch_size=a.batch).float()
            out[idx] = e
        return out

    t0, done = time.time(), 0
    for c0 in range(0, len(todo), a.chunk):
        metas, flat = [], []
        for stem, utts in todo[c0:c0 + a.chunk]:
            case = build_case(utts)  # sections see every utterance; the sidecar keeps S_MAX
            utts = utts[:S_MAX]
            secs = encodable_sections(case)
            metas.append((stem, utts, secs, case["report_pos"], len(flat)))
            flat += [u["text"] for u in utts] + [s[2] for s in secs] + [case["report_pos"]]
        emb = embed(flat)
        for stem, utts, secs, report, base in metas:
            n_u, n_s = len(utts), len(secs)
            tokens = torch.zeros(S_MAX, hidden)
            tokens[:n_u] = emb[base:base + n_u]
            mask = torch.zeros(S_MAX, dtype=torch.bool)
            mask[:n_u] = True
            save_atomic({"tokens": tokens, "attn_mask": mask, "texts": [u["text"] for u in utts],
                         "sections": [u.get("section") for u in utts],
                         "negated": [bool(u.get("negated")) for u in utts],
                         "uncertain": [bool(u.get("uncertain")) for u in utts]},
                        os.path.join(a.utt_root, stem + UTT_SUFFIX))
            save_atomic({"section_names": [s[0] for s in secs], "section_status": [s[1] for s in secs],
                         "section_tokens": emb[base + n_u:base + n_u + n_s].clone(),
                         "section_texts": [s[2] for s in secs],
                         "report_token": emb[base + n_u + n_s].clone(), "report_text": report},
                        os.path.join(a.sr_root, stem + SR_SUFFIX))
            done += 1
        rate = done / max(time.time() - t0, 1e-9)
        print(f"{done}/{len(todo)} cases | {rate:.1f} cases/s", flush=True)


if __name__ == "__main__":
    main()
