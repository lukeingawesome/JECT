"""Assign each refined-report utterance to an anatomy grounded in TotalSegmentator.

The refined reports for CT-RATE, d2 and d3 share the same structure::

    [Lungs]
    - A subcentimeter nodule in the right lower lobe (RLL).
    [Mediastinum & Hila]
    - No lymph node enlargement.

This module turns that into ART-JEPA's ``utterances_json`` — one dict per bullet::

    {"text": "...", "anatomy_id": 8, "anatomy": "right_lower_lobe",
     "section": "Lungs", "negated": false, "uncertain": false}

The anatomy id comes from the ``[Section]`` header, then — for lung/pleura
findings — is **refined to a specific lobe/side** when the sentence names one
("right lower lobe", "(RLL)", "lingula", ...). Every anatomy id maps to a set of
TotalSegmentator-v2 labels (``ANATOMY_TO_TOTALSEG``) so a binary anatomy mask can
be cut from an npz ``totalseg`` field via :func:`cut_anat_masks`.

Anatomy ids are aligned with ``art_jepa_data.ANATOMY_NAME_TO_ID`` so the strings
and integers agree across the two modules.

CLI (adds/overwrites a ``utterances_json`` column, writes a new CSV)::

    python anatomy_assign.py --csv dataframe/final_d2_findings.csv \
        --img_col object_id --out dataframe/final_d2_utt.csv
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Anatomy taxonomy (ids match art_jepa_data.ANATOMY_NAME_TO_ID)
# ---------------------------------------------------------------------------
UNKNOWN = 0
LUNG, LEFT_LUNG, RIGHT_LUNG = 1, 2, 3
LUL, LLL, RUL, RML, RLL = 4, 5, 6, 7, 8          # lobes
AIRWAYS, PLEURA, MEDIASTINUM = 9, 10, 11
CARDIOVASCULAR, CHEST_WALL, BONES = 12, 13, 14
UPPER_ABDOMEN, LOWER_NECK, BREAST, OTHER = 15, 16, 17, 18

ANATOMY_ID_TO_NAME = {
    0: "unknown", 1: "lung", 2: "left_lung", 3: "right_lung",
    4: "left_upper_lobe", 5: "left_lower_lobe", 6: "right_upper_lobe",
    7: "right_middle_lobe", 8: "right_lower_lobe", 9: "airways",
    10: "pleura", 11: "mediastinum", 12: "cardiovascular", 13: "chest_wall",
    14: "bones_spine", 15: "upper_abdomen", 16: "lower_neck", 17: "breast_axilla",
    18: "other",
}

# ---------------------------------------------------------------------------
# Section header -> base anatomy id.  Keyed by a normalized header string
# (lowercased, '&'/'/'/whitespace collapsed to single spaces).  Covers the
# CT-RATE and d2/d3 spelling variants ("Airways & Trachea" vs "Trachea &
# Airways", "Upper abdomen" vs "Upper Abdomen", "Breast & Axilla" vs "Breast").
# ---------------------------------------------------------------------------
SECTION_TO_ANATOMY: Dict[str, int] = {
    "lungs": LUNG, "lung": LUNG, "lungs pleura": LUNG, "lung pleura": LUNG,
    "airways trachea": AIRWAYS, "trachea airways": AIRWAYS, "airways": AIRWAYS,
    "trachea": AIRWAYS, "trachea bronchi": AIRWAYS,
    "pleura": PLEURA,
    "mediastinum hila": MEDIASTINUM, "mediastinum": MEDIASTINUM, "hila": MEDIASTINUM,
    "hila mediastinum": MEDIASTINUM, "mediastinum hilum": MEDIASTINUM,
    "cardiovascular": CARDIOVASCULAR, "heart": CARDIOVASCULAR, "cardiac": CARDIOVASCULAR,
    "chest wall": CHEST_WALL, "musculoskeletal chest wall": CHEST_WALL,
    "bones spine": BONES, "bones": BONES, "spine": BONES, "thoracic msk": BONES,
    "musculoskeletal": BONES, "bone": BONES,
    "upper abdomen": UPPER_ABDOMEN, "abdominal": UPPER_ABDOMEN, "abdomen": UPPER_ABDOMEN,
    "lower neck": LOWER_NECK, "neck": LOWER_NECK, "thyroid": LOWER_NECK,
    "breast axilla": BREAST, "breast": BREAST, "axilla": BREAST,
    "others": OTHER, "other": OTHER,
}


def _norm_section(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", " ").replace("/", " ").replace("+", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def section_to_anatomy(section: str) -> int:
    return SECTION_TO_ANATOMY.get(_norm_section(section), OTHER)


# ---------------------------------------------------------------------------
# Anatomy id -> TotalSegmentator v2 ("total" task) label ids.
# Lung lobes 10-14 are confirmed by grounding_decoder.py. Entries flagged
# "proxy" have no exact TotalSeg structure (pleura, hila lymph nodes, breast);
# they use the nearest structural surrogate so a mask can still be cut. Edit
# freely — regions may overlap by design.
# ---------------------------------------------------------------------------
_TS_LUNG = {10, 11, 12, 13, 14}
_TS_VERT = set(range(26, 51))                     # sacrum..C1 (thoracic present ~30-44)
_TS_RIBS = set(range(92, 116))                    # rib_left_1..rib_right_12
_TS_GREAT_VESSELS = {52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63}  # aorta, PV, SVC/IVC, brachioceph...

ANATOMY_TO_TOTALSEG: Dict[int, set] = {
    UNKNOWN: set(),
    LUNG: set(_TS_LUNG),
    LEFT_LUNG: {10, 11},
    RIGHT_LUNG: {12, 13, 14},
    LUL: {10}, LLL: {11}, RUL: {12}, RML: {13}, RLL: {14},
    AIRWAYS: {16},                                # trachea (bronchi not separate in TSv2)
    PLEURA: set(_TS_LUNG),                        # proxy: pleura wraps the lungs
    MEDIASTINUM: {15, 16} | _TS_GREAT_VESSELS,    # esophagus, trachea, great vessels (hila LNs = proxy)
    CARDIOVASCULAR: {51, 52, 61, 53, 62, 63},     # heart, aorta, atrial appendage, PV, SVC/IVC
    CHEST_WALL: {86, 87, 71, 72, 73, 74, 69, 70}, # autochthon, scapula, clavicle, humerus
    BONES: set(_TS_VERT) | _TS_RIBS | {25, 116},  # vertebrae, ribs, sacrum, sternum
    UPPER_ABDOMEN: {1, 2, 3, 4, 5, 6, 7, 8, 9},   # spleen, kidneys, gallbladder, liver, stomach...
    LOWER_NECK: {17, 57, 58},                     # thyroid, common carotids
    BREAST: set(),                                # proxy: no breast structure in TSv2
    OTHER: set(),
}

# ---------------------------------------------------------------------------
# Utterance text parsing
# ---------------------------------------------------------------------------
_LOBE_ABBR = {
    "lul": LUL, "lll": LLL, "rul": RUL, "rml": RML, "rll": RLL,
}
_NEG_RE = re.compile(
    r"^\s*(no\b|no evidence|without\b|absence of|negative for|"
    r"not\b|unremarkable|none\b|free of|resolved)", re.I)
_NEG_INLINE_RE = re.compile(
    r"\b(no evidence of|without|absence of|not seen|not identified|"
    r"negative for|ruled out)\b", re.I)
_UNCERTAIN_RE = re.compile(
    r"\b(may\b|maybe|possible|possibly|probable|probably|likely|"
    r"suspicious|suspected|cannot exclude|can not exclude|cannot be excluded|"
    r"equivocal|indeterminate|questionable|suggestive of|concerning for)\b", re.I)


def _detect_negated(text: str) -> bool:
    return bool(_NEG_RE.search(text) or _NEG_INLINE_RE.search(text))


def _detect_uncertain(text: str) -> bool:
    return bool(_UNCERTAIN_RE.search(text))


def refine_lung_lobe(text: str, base: int) -> int:
    """Refine a lung/pleura utterance to a specific lobe/side when named.

    Returns the most specific anatomy id supported by the sentence, else ``base``.
    """
    p = text.lower()
    # Explicit lobe abbreviations, e.g. "(RLL)" / " rll " (word-bounded).
    for abbr, aid in _LOBE_ABBR.items():
        if re.search(rf"\b{abbr}\b", p):
            return aid

    has_left = ("left" in p) or ("lingula" in p) or ("lingular" in p)
    has_right = "right" in p
    bilateral = ("bilateral" in p) or ("both" in p)
    upper = ("upper lobe" in p) or ("upper lobes" in p) or ("lingula" in p) or ("lingular" in p)
    middle = "middle lobe" in p
    lower = ("lower lobe" in p) or ("lower lobes" in p) or ("basal" in p) or ("basilar" in p)

    if bilateral:
        return base  # keep generic lung/pleura
    if has_right and not has_left:
        if upper:
            return RUL
        if middle:
            return RML
        if lower:
            return RLL
        return RIGHT_LUNG
    if has_left and not has_right:
        if upper:
            return LUL
        if lower:
            return LLL
        return LEFT_LUNG
    # Side not disambiguated: keep the base (generic lung / pleura).
    return base


def assign_utterance_anatomy(text: str, section: str) -> int:
    """Anatomy id for one utterance given its section header."""
    base = section_to_anatomy(section)
    if base in (LUNG, PLEURA) or ("lobe" in text.lower()) or ("lung" in text.lower()):
        refined = refine_lung_lobe(text, base if base in (LUNG, PLEURA) else LUNG)
        # Only override to a lung/lobe id; never turn a non-lung section into a lobe
        # unless the base was lung/pleura.
        if base in (LUNG, PLEURA):
            return refined
    return base


def parse_report(text: str) -> List[Tuple[str, str]]:
    """Parse a ``[Section]`` + ``- bullet`` report into (section, utterance) pairs.

    Bullets before any section header get section ``""`` (-> OTHER). Robust to the
    numbered-impression fallback already normalized to ``- `` bullets upstream.
    """
    if not isinstance(text, str):
        return []
    pairs: List[Tuple[str, str]] = []
    current = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\[([^\]]+)\]\s*$", line)
        if m:
            current = m.group(1).strip()
            continue
        if line.startswith("- "):
            u = line[2:].strip()
            if u:
                pairs.append((current, u))
        elif line.startswith(("-", "*", "•")):
            u = line[1:].strip()
            if u:
                pairs.append((current, u))
    return pairs


def assign_utterances(text: str) -> List[dict]:
    """Full pipeline: report text -> list of ART ``utterances_json`` dicts."""
    out: List[dict] = []
    for section, utt in parse_report(text):
        aid = assign_utterance_anatomy(utt, section)
        out.append({
            "text": utt,
            "anatomy_id": int(aid),
            "anatomy": ANATOMY_ID_TO_NAME.get(aid, "other"),
            "section": section,
            "negated": _detect_negated(utt),
            "uncertain": _detect_uncertain(utt),
        })
    return out


# ---------------------------------------------------------------------------
# Anatomy mask cutting (helper — not run at scale here)
# ---------------------------------------------------------------------------
def cut_anat_masks(totalseg: np.ndarray, anatomy_id: int) -> Optional[np.ndarray]:
    """Binary mask (float32, same DHW as ``totalseg``) of an anatomy's TotalSeg labels.

    Returns None when the anatomy has no mapped TotalSeg structure (e.g. breast).
    Accepts ``totalseg`` shaped [D,H,W] or [1,D,H,W]/[K,D,H,W] (first channel used).
    """
    labels = ANATOMY_TO_TOTALSEG.get(int(anatomy_id))
    if not labels:
        return None
    ts = totalseg
    while ts.ndim > 3:
        ts = ts[0]
    return np.isin(ts, list(labels)).astype(np.float32)


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------
_TEXT_COL_CANDIDATES = ("refined_findings", "refined_text", "report_text", "findings")


def resolve_text_col(columns, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"--text_col {explicit!r} not in CSV columns.")
        return explicit
    for c in _TEXT_COL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"No report column found among {_TEXT_COL_CANDIDATES}; pass --text_col.")


def build_utterances_json(csv_path: str, out_path: str, text_col: Optional[str] = None,
                          img_col: str = "object_id") -> dict:
    import pandas as pd
    df = pd.read_csv(csv_path)
    col = resolve_text_col(df.columns, text_col)
    jsons, counts, n_empty = [], [], 0
    anat_hist: Dict[int, int] = {}
    for text in df[col].astype(object):
        utts = assign_utterances(text if isinstance(text, str) else "")
        jsons.append(json.dumps(utts, ensure_ascii=False))
        counts.append(len(utts))
        if not utts:
            n_empty += 1
        for u in utts:
            anat_hist[u["anatomy_id"]] = anat_hist.get(u["anatomy_id"], 0) + 1
    df["utterances_json"] = jsons
    df.to_csv(out_path, index=False)
    total = int(sum(counts))
    stats = {
        "rows": len(df), "text_col": col, "out": out_path,
        "total_utterances": total,
        "mean_utterances_per_row": round(total / max(1, len(df)), 2),
        "rows_with_no_utterances": n_empty,
        "anatomy_histogram": {ANATOMY_ID_TO_NAME[k]: v for k, v in sorted(anat_hist.items())},
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Assign refined-report utterances to anatomy (TotalSeg).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_col", default=None,
                    help=f"Report column; auto-detected among {_TEXT_COL_CANDIDATES}.")
    ap.add_argument("--img_col", default="object_id")
    args = ap.parse_args()
    stats = build_utterances_json(args.csv, args.out, args.text_col, args.img_col)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
