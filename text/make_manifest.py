"""Training manifest from a table of (object_id, split, section-structured findings).

Writes object_id, split, utterances_json (plus npz_root / utt_root when given).
Rows whose findings have no section header, or yield no bullet, are skipped and
counted. At most 32 utterances are kept per case: the trainer requires the
utterance sidecar, which has 32 slots, to cover every manifest utterance.

  python make_manifest.py --table reports.csv --findings_col findings --out manifest.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from encode_chest2vec import S_MAX
from sections import has_header, parse


def read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="CSV or parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id_col", default="object_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--findings_col", default="findings")
    ap.add_argument("--npz_root", default="", help="written as a per-row npz_root column if set")
    ap.add_argument("--utt_root", default="", help="written as a per-row utt_root column if set; "
                                                    "otherwise the trainer looks for sidecars next to the npz")
    a = ap.parse_args()

    df = read_table(a.table)
    rows, skipped = [], {"no_findings": 0, "no_header": 0, "no_bullet": 0}
    for oid, split, text in zip(df[a.id_col], df[a.split_col], df[a.findings_col]):
        if not isinstance(text, str) or not text.strip():
            skipped["no_findings"] += 1
            continue
        if not has_header(text):
            skipped["no_header"] += 1
            continue
        utts = parse(text)
        if not utts:
            skipped["no_bullet"] += 1
            continue
        row = {"object_id": str(oid), "split": str(split),
               "utterances_json": json.dumps(utts[:S_MAX], ensure_ascii=False)}
        if a.npz_root:
            row["npz_root"] = a.npz_root
        if a.utt_root:
            row["utt_root"] = a.utt_root
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    n_utt = sum(len(json.loads(x)) for x in out["utterances_json"]) if len(out) else 0
    print(f"{len(out)} cases, {n_utt} utterances written to {a.out} | skipped {skipped}")
    if len(out):
        print("split counts: " + json.dumps(out["split"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
