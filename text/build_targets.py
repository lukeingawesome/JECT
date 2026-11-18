"""Utterances to per-case section targets.

For each of the canonical sections a case gets one status:
  positive      at least one utterance that is neither negated nor uncertain;
                text = those utterances joined
  normal        no positive utterance but at least one negated one; text = the negated ones
  not_assessed  section not mentioned (or only uncertain utterances); excluded from the loss
Uncertain utterances never enter a section text and are kept separately.
`report_pos` joins the positive section texts in canonical order.

  python build_targets.py --manifest manifest.csv --out targets.jsonl
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from sections import SECTIONS, canonical_section

NORMAL_REPORT = "No significant abnormality was detected."


def build_case(utterances: list[dict]) -> dict:
    """Utterance dicts (text, section, negated, uncertain) to {sections, report_pos, n_pos}."""
    by_sec: dict = {}
    for it in utterances:
        b = by_sec.setdefault(canonical_section(it.get("section")), {"pos": [], "neg": [], "unc": []})
        k = "neg" if it.get("negated") else ("unc" if it.get("uncertain") else "pos")
        b[k].append(it["text"])
    secs = []
    for name in SECTIONS:
        b = by_sec.get(name)
        if b and b["pos"]:
            status, text = "positive", " ".join(b["pos"])
        elif b and b["neg"]:
            status, text = "normal", " ".join(b["neg"])
        else:
            status, text = "not_assessed", ""
        secs.append({"name": name, "status": status, "text": text,
                     "uncertain_texts": b["unc"] if b else []})
    pos = [s["text"] for s in secs if s["status"] == "positive"]
    return {"sections": secs, "report_pos": " ".join(pos) or NORMAL_REPORT, "n_pos": len(pos)}


def encodable_sections(case: dict) -> list[tuple[str, str, str]]:
    """(name, status, text) of the sections that get an embedding: positive or normal with text."""
    return [(s["name"], s["status"], s["text"]) for s in case["sections"]
            if s["status"] in ("positive", "normal") and s["text"].strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with object_id, split, utterances_json")
    ap.add_argument("--out", required=True, help="output JSONL, one record per case")
    a = ap.parse_args()
    df = pd.read_csv(a.manifest, usecols=["object_id", "split", "utterances_json"])
    counts = {"positive": 0, "normal": 0, "not_assessed": 0}
    with open(a.out, "w") as f:
        for r in df.itertuples(index=False):
            case = build_case(json.loads(r.utterances_json))
            for s in case["sections"]:
                counts[s["status"]] += 1
            f.write(json.dumps({"object_id": str(r.object_id), "split": r.split, **case}) + "\n")
    print(f"{len(df)} cases written to {a.out} | section states {counts}")


if __name__ == "__main__":
    main()
