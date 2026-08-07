"""Structured report + anatomy-mask dataset utilities for ART-JEPA.

Preferred data layout
---------------------
CSV columns:
  object_id, split, utterances_json

utterances_json example:
  [
    {"text":"7 mm solid nodule in the right upper lobe", "anatomy_id":3, "label":"nodule", "negated":false},
    {"text":"No pleural effusion", "anatomy_id":7, "label":"pleural_effusion", "negated":true}
  ]

NPZ keys:
  ct:         [1,D,H,W] or [D,H,W]
  anat_masks: [K,D,H,W] optional, binary/soft masks for anatomy IDs
  seg_mask:   [D,H,W] optional, supervised segmentation target for auxiliary loss

Fallbacks are intentionally permissive so you can connect your existing CEPA CSV.
"""

from __future__ import annotations

import json
import os
import random
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from ject.config.art import ARTJEPAConfig
    from .data import apply_window
except ImportError:
    from ject.config.art import ARTJEPAConfig
    from data import apply_window

# TOTALSEG_TO_REGION was removed from data.py in the architecture cleanse. It only
# powers the *optional* TotalSegmentator anatomy-mask fallback; when it is absent we
# disable that fallback (the README already recommends supplying your own anat_masks).
try:
    from .data import TOTALSEG_TO_REGION  # type: ignore
except ImportError:
    try:
        from data import TOTALSEG_TO_REGION  # type: ignore
    except ImportError:
        TOTALSEG_TO_REGION: dict = {}

# Curated anatomy_id -> TotalSegmentator-v2 label sets (replaces the removed
# region map). Lets anatomy masks be cut from an npz `totalseg` field on the fly.
try:
    from ject.backbone.anatomy import ANATOMY_TO_TOTALSEG
except ImportError:
    from ject.backbone.anatomy import ANATOMY_TO_TOTALSEG

Triple = Tuple[int, int, int]

# CT-type conditioning ids (matches the main cepa CT_TYPE_MAP). Normalizes the
# underscore/hyphen and phase-string variants seen across CT-RATE / d2 / d3.
CT_TYPE_MAP = {
    "non-contrast": 0, "non_contrast": 0, "noncontrast": 0, "nc": 0, "plain": 0,
    "contrast": 1, "ce": 1, "cect": 1, "arterial": 1, "venous": 1, "delayed": 1,
    "ldct": 2, "low-dose": 2, "low_dose": 2, "screening": 2,
}


def ct_type_to_id(value) -> int:
    if value is None:
        return 0
    key = str(value).strip().lower()
    return CT_TYPE_MAP.get(key, 0)


ANATOMY_NAME_TO_ID = {
    "background": 0,
    "unknown": 0,
    "lung": 1,
    "lungs": 1,
    "left_lung": 2,
    "right_lung": 3,
    "lung_left_upper": 4,
    "left_upper_lobe": 4,
    "lul": 4,
    "lung_left_lower": 5,
    "left_lower_lobe": 5,
    "lll": 5,
    "lung_right_upper": 6,
    "right_upper_lobe": 6,
    "rul": 6,
    "lung_right_middle": 7,
    "right_middle_lobe": 7,
    "rml": 7,
    "lung_right_lower": 8,
    "right_lower_lobe": 8,
    "rll": 8,
    "airways": 9,
    "airways_trachea": 9,
    "trachea_bronchi": 9,
    "pleura": 10,
    "mediastinum": 11,
    "mediastinum_hila": 11,
    "hila": 11,
    "heart": 12,
    "cardiovascular": 12,
    "chest_wall": 13,
    "bones": 14,
    "spine": 14,
    "upper_abdomen": 15,
    "lower_neck": 16,
    "breast_axilla": 17,
    "other": 18,
    "others": 18,
}

SECTION_TO_ANATOMY = {
    "Lungs": 1,
    "Lungs + Pleura": 1,
    "Airways & Trachea": 9,
    "Pleura": 10,
    "Mediastinum & Hila": 11,
    "Cardiovascular": 12,
    "Chest Wall": 13,
    "Bones / Spine": 14,
    "Thoracic MSK": 14,
    "Upper abdomen": 15,
    "Lower neck": 16,
    "Breast & Axilla": 17,
    "Others": 18,
}


def normalize_anatomy_id(value: Any, default: int = 0, num_anatomies: int = 32) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        out = int(value)
    elif isinstance(value, float) and np.isfinite(value):
        out = int(value)
    else:
        key = str(value).strip().lower().replace(" ", "_").replace("/", "_").replace("&", "_")
        key = key.replace("__", "_")
        out = ANATOMY_NAME_TO_ID.get(key, default)
    if out < 0 or out >= num_anatomies:
        out = default
    return out


def _is_empty_text(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    return str(x).strip() == ""


def _norm_text(x: Any) -> str:
    """Whitespace-collapsed, lowercased text — used to match sidecar-stored utterance strings against the
    manifest so cached embeddings can only be used when they provably belong to the current utterances."""
    return " ".join(str(x).split()).strip().lower()


def _safe_json_loads(x: Any):
    if _is_empty_text(x):
        return None
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return None


def split_report_sentences(text: str) -> list[str]:
    if _is_empty_text(text):
        return []
    # Conservative splitter; reports often have semicolon-separated findings.
    cleaned = str(text).replace("\n", ". ")
    pieces = []
    for chunk in cleaned.replace(";", ".").split("."):
        s = chunk.strip()
        if s:
            pieces.append(s)
    return pieces


def diversity_first_order(aids: list) -> list[int]:
    """Order indices so that ONE slot per distinct anatomy comes before any duplicate-anatomy slot.

    e.g. anatomies [lung, lung, lung, mediastinum, heart] -> [0, 3, 4, 1, 2], so truncating to
    max_regions still covers distinct anatomies (giving the region-text contrastive real negatives)
    instead of filling every slot with the first anatomy. Report order is preserved within each group.
    """
    seen: set = set()
    first, rest = [], []
    for i, a in enumerate(aids):
        (rest if a in seen else first).append(i)
        seen.add(a)
    return first + rest


def parse_utterances_from_row(row: pd.Series, cfg: ARTJEPAConfig) -> list[dict]:
    """Return list of {text, anatomy_id, label, negated, uncertain}."""
    out: list[dict] = []
    col = cfg.text.utterances_json_col
    parsed = _safe_json_loads(row[col]) if col in row.index else None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                txt = item.get("text", item.get("sentence", item.get("utterance", item.get("content", ""))))
                if _is_empty_text(txt):
                    continue
                anatomy_raw = item.get("anatomy_id", item.get("anatomy", item.get("region", item.get("section", None))))
                out.append({
                    "text": str(txt),
                    "anatomy_id": normalize_anatomy_id(anatomy_raw, cfg.anatomy.unknown_anatomy_id, cfg.anatomy.num_anatomies),
                    "label": str(item.get("label", item.get("finding", ""))),
                    "negated": bool(item.get("negated", False)),
                    "uncertain": bool(item.get("uncertain", False)),
                })
            elif not _is_empty_text(item):
                out.append({"text": str(item), "anatomy_id": cfg.anatomy.unknown_anatomy_id,
                            "label": "", "negated": False, "uncertain": False})
    elif isinstance(parsed, dict):
        # Accept anatomy -> text dictionary.
        for anatomy_raw, txt in parsed.items():
            for sent in split_report_sentences(str(txt)):
                out.append({
                    "text": sent,
                    "anatomy_id": normalize_anatomy_id(anatomy_raw, cfg.anatomy.unknown_anatomy_id, cfg.anatomy.num_anatomies),
                    "label": "", "negated": False, "uncertain": False,
                })

    if out:
        return out

    # Fallback: structured anatomy section columns.
    if cfg.text.use_section_columns:
        for sec in cfg.text.section_columns:
            if sec in row.index and not _is_empty_text(row[sec]):
                anatomy_id = SECTION_TO_ANATOMY.get(sec, cfg.anatomy.unknown_anatomy_id)
                for sent in split_report_sentences(str(row[sec])):
                    out.append({"text": sent, "anatomy_id": anatomy_id,
                                "label": "", "negated": False, "uncertain": False})
    if out:
        return out

    # Fallback: findings/impression as a global report.
    for fallback_col in (cfg.text.findings_col, cfg.text.impression_col):
        if fallback_col in row.index and not _is_empty_text(row[fallback_col]):
            for sent in split_report_sentences(str(row[fallback_col])):
                out.append({"text": sent, "anatomy_id": cfg.anatomy.unknown_anatomy_id,
                            "label": "", "negated": False, "uncertain": False})
            if out:
                return out

    # Synthetic placeholder for a row with NO usable text (positive-only manifest emits utterances_json=[]
    # for normal/no-finding studies). Tagged `_synthetic` so the loader keeps it truly text-free and never
    # reads a (possibly stale) cached sidecar against it.
    return [{"text": "no report available", "anatomy_id": cfg.anatomy.unknown_anatomy_id,
             "label": "", "negated": False, "uncertain": False, "_synthetic": True}]


def mask_to_flat_grid(mask: np.ndarray | torch.Tensor, grid_size: Triple, threshold: float = 0.01) -> torch.Tensor:
    """Resize a full-resolution binary/soft mask to a flattened token grid."""
    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask.astype(np.float32))
    else:
        m = mask.detach().float().cpu()
    while m.ndim > 3:
        m = m[0]
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {tuple(m.shape)}")
    x = m.unsqueeze(0).unsqueeze(0)
    # max-ish pooling via interpolation + threshold is robust when dimensions are not integer multiples.
    y = F.interpolate(x, size=grid_size, mode="trilinear", align_corners=False)
    return (y[0, 0] > threshold).flatten()


def extract_selected_anatomy_mask(data: np.lib.npyio.NpzFile, anatomy_id: int, cfg: ARTJEPAConfig) -> np.ndarray | None:
    keys = [cfg.anatomy.anatomy_mask_key, *cfg.anatomy.alternative_anatomy_mask_keys]
    for key in keys:
        if key in data.files:
            arr = data[key]
            # Accept [K,D,H,W], [1,K,D,H,W], or [D,H,W] binary selected mask.
            if arr.ndim == 5 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 4:
                if anatomy_id < arr.shape[0]:
                    return arr[anatomy_id]
                if arr.shape[0] > 0:
                    return arr[0] * 0
            elif arr.ndim == 3:
                return arr

    if cfg.anatomy.totalseg_key in data.files:
        tot = data[cfg.anatomy.totalseg_key]
        if tot.ndim == 4:
            tot = tot[0]
        # Prefer the curated anatomy->TotalSeg-v2 map; fall back to any legacy region map.
        labels = sorted(ANATOMY_TO_TOTALSEG.get(int(anatomy_id), set()))
        if not labels:
            labels = [label_id for label_id, region_id in TOTALSEG_TO_REGION.items() if region_id == anatomy_id]
        if labels:
            return np.isin(tot, labels).astype(np.float32)
    return None


def anat_cropped_from_ts64(ts64: np.ndarray, anatomy_id: int, orig_shape, slices) -> np.ndarray | None:
    """Cut an anatomy mask from a downsampled totalseg (n³) and apply the crop, scaled PER AXIS.

    `slices` are in the original grid; each axis is scaled by ts64.shape[axis]/orig_shape[axis]
    (handles non-cubic volumes). Only valid when the CT crop needs no padding (the caller guards
    that — a padded crop falls back to the exact full-res path to keep mask/CT alignment).
    """
    labels = sorted(ANATOMY_TO_TOTALSEG.get(int(anatomy_id), set()))
    if not labels:
        return None
    m = np.isin(ts64, labels).astype(np.float32)
    sl = []
    for axis, s in enumerate(slices):
        sf = ts64.shape[axis] / float(orig_shape[axis])
        a, b = int(s.start * sf), int(s.stop * sf)
        sl.append(slice(a, max(a + 1, b)))
    m = m[tuple(sl)]
    return m if m.size else None


def extract_seg_mask(data: np.lib.npyio.NpzFile, cfg: ARTJEPAConfig) -> np.ndarray | None:
    for key in [cfg.anatomy.seg_mask_key, *cfg.anatomy.alternative_seg_mask_keys]:
        if key in data.files:
            arr = data[key]
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
    return None


def make_crop_pad_transform(src_shape: Triple, dst_shape: Triple, random_crop: bool = True):
    """Create a crop/pad transform for 3D arrays."""
    slices = []
    pads = []
    for src, dst in zip(src_shape, dst_shape):
        if src >= dst:
            start = random.randint(0, src - dst) if random_crop and src > dst else (src - dst) // 2
            slices.append(slice(start, start + dst))
            pads.append((0, 0))
        else:
            slices.append(slice(0, src))
            before = (dst - src) // 2
            after = dst - src - before
            pads.append((before, after))
    return tuple(slices), tuple(pads)


def apply_crop_pad_3d(arr: np.ndarray, slices: tuple[slice, slice, slice], pads: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], constant: float = 0.0) -> np.ndarray:
    """Apply crop/pad to the last three dimensions of arr."""
    cropped = arr[(...,) + slices]
    pad_width = [(0, 0)] * (cropped.ndim - 3) + list(pads)
    if any(a > 0 or b > 0 for a, b in pads):
        cropped = np.pad(cropped, pad_width=pad_width, mode="constant", constant_values=constant)
    return cropped


class StructuredReportCTDataset(Dataset):
    """CT dataset returning one structured utterance/anatomy target per sample."""

    def __init__(self, cfg: ARTJEPAConfig, split: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.data_root = cfg.data.data_root
        df = pd.read_csv(cfg.data.csv_path)
        split = cfg.data.split if split is None else split
        if cfg.data.split_col and cfg.data.split_col in df.columns and split is not None:
            df = df[df[cfg.data.split_col] == split].reset_index(drop=True)
        if cfg.data.img_col not in df.columns:
            raise ValueError(f"CSV does not contain image column {cfg.data.img_col!r}.")
        self.df = df
        # Foundation manifests carry a per-row `npz_root` (CT-RATE / d2 / d3 live in different
        # dirs); use it when present, else the single `data_root`. Sidecars resolve next to it.
        ids = df[cfg.data.img_col].astype(str)
        if "npz_root" in df.columns:
            self.paths = (df["npz_root"].astype(str) + os.sep + ids).tolist()
        else:
            self.paths = (self.data_root + os.sep + ids).tolist()
        # Text sidecars (`_utterances.pt`) can live in a SEPARATE dir from the npz — e.g. CT-RATE's
        # positive-report sidecars go to `utt_root` (fp_positive_utt) while its npz stays in npz_root.
        # Use per-row `utt_root` when the manifest carries it, else co-locate with the npz.
        if "utt_root" in df.columns:
            self.utt_paths = (df["utt_root"].astype(str) + os.sep + ids).tolist()
        else:
            self.utt_paths = self.paths

    def __len__(self):
        return len(self.df)

    def _load_npz(self, path: str):
        if os.path.exists(path):
            return np.load(path)
        if os.path.exists(path + ".npz"):
            return np.load(path + ".npz")
        raise FileNotFoundError(path)

    def _load_sidecar(self, path: str):
        """Load `{id}{suffix}` cached utterance embeddings -> (tokens[S,D], attn_mask[S], texts|None) or None.

        `texts` (when the sidecar carries it) is the list of utterance strings the vectors were built from,
        index-aligned with `tokens`. The loader validates it against the manifest so a stale/reordered
        sidecar in a reused utt_root is rejected instead of silently pairing the wrong phrase with a region."""
        suffix = getattr(self.cfg.text, "utterances_sidecar_suffix", "_utterances.pt")
        base = path[:-4] if path.endswith(".npz") else path
        sc = base + suffix
        if os.path.exists(sc):
            try:
                # weights_only=True: sidecars are pure tensor dicts loaded per-sample from the
                # data root, so restrict deserialization (no arbitrary pickle code execution).
                # (A "texts" list is plain Python and is allowed under weights_only.)
                d = torch.load(sc, map_location="cpu", weights_only=True)
                if isinstance(d, dict) and "tokens" in d and "attn_mask" in d:
                    return d["tokens"], d["attn_mask"], d.get("texts")
            except Exception:
                return None
        return None

    def _anat_cropped(self, data, ts64, anatomy_id, orig_shape, slices, pads):
        """Cropped anatomy mask (numpy) from the ts64 cache when present, else the npz totalseg.

        The coarse cache is only used when the crop needs NO padding (cube-cropped, vol >= crop);
        a padded crop falls back to the exact full-res path so the mask stays aligned to the CT.
        """
        if ts64 is not None and not any(a or b for a, b in pads):
            return anat_cropped_from_ts64(ts64, anatomy_id, orig_shape, slices)
        m = extract_selected_anatomy_mask(data, anatomy_id, self.cfg)
        if m is None:
            return None
        return apply_crop_pad_3d(m, slices, pads, constant=0.0)

    def _load_one(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.paths[idx]
        utterances = parse_utterances_from_row(row, self.cfg)
        n_utt = len(utterances)
        use_cached = getattr(self.cfg.text, "use_cached_embeddings", False)
        cached = self._load_sidecar(self.utt_paths[idx]) if use_cached else None   # sidecar from utt_root
        # Text-free rows (positive-only manifest turns normal/no-finding studies into utterances_json=[],
        # which parse_utterances_from_row backfills with a single `_synthetic` "no report available" entry)
        # must stay text-free. A reused utt_root may still hold a STALE `{id}_utterances.pt` from an earlier
        # full-report/normal-placeholder extraction; without this guard chosen_idx=0 would read cached[0][0]
        # and silently condition a supposedly image-only volume (and inflate cached_text_ok coverage).
        if not any(not u.get("_synthetic", False) for u in utterances):
            cached = None

        # Provable index alignment: a sidecar carries the utterance `texts` its vectors were built from.
        # Validate them against the manifest (position by position). A stale/reordered/subset sidecar in a
        # reused utt_root — e.g. an old full-report sidecar whose slot 0 is a DIFFERENT phrase than the
        # positive-only manifest's slot 0 — is rejected outright rather than silently pairing the wrong
        # phrase with a region mask (which would corrupt region-text / weak-grounding / MUP supervision).
        if cached is not None:
            sc_texts = cached[2]
            require = getattr(self.cfg.text, "require_cached_text_alignment", False)
            if sc_texts is None:
                if require:                                   # rebuilt-manifest safety: demand provable texts
                    cached = None
            else:
                am = cached[1]
                S = int(am.shape[0])
                valid_i = [i for i in range(S) if bool(am[i])]
                # (1) every VALID slot must match a manifest utterance at the same index (no reorder/stale text)
                aligned = all(i < n_utt and i < len(sc_texts)
                              and _norm_text(sc_texts[i]) == _norm_text(utterances[i].get("text", ""))
                              for i in valid_i)
                extra = any(i >= len(sc_texts) or i >= n_utt for i in valid_i)   # stale/longer sidecar
                if (not aligned) or extra:
                    cached = None
                elif require:
                    # (2) COVERAGE: a PARTIAL sidecar (valid prefix/subset) would otherwise pass (1) while
                    # silently dropping the uncached manifest utterances from text/region supervision. Under
                    # require_cached_text_alignment, demand the sidecar covers ALL n_utt manifest utterances
                    # with a valid, text-matching slot — a SHORT sidecar (S < n_utt) must be REJECTED, not
                    # accepted on its prefix (which would silently drop the tail utterances / positive findings).
                    covered = (S >= n_utt and len(sc_texts) >= n_utt and all(
                        bool(am[i]) and _norm_text(sc_texts[i]) == _norm_text(utterances[i].get("text", ""))
                        for i in range(n_utt)))
                    if not covered:
                        cached = None

        # Choose one utterance (bias to non-negated); when using cached embeddings, restrict
        # to indices that actually have a valid sidecar vector (index-aligned with utterances).
        pos_idx = [i for i in range(n_utt)
                   if not utterances[i].get("negated", False) and not utterances[i].get("uncertain", False)]
        pool = pos_idx if pos_idx and random.random() < 0.8 else list(range(n_utt))
        if cached is not None:
            valid_sc = {i for i in range(min(n_utt, cached[0].shape[0])) if bool(cached[1][i])}
            pool = [i for i in pool if i in valid_sc] or list(valid_sc) or pool
        chosen_idx = random.choice(pool) if pool else 0
        chosen = utterances[chosen_idx] if n_utt else {"anatomy_id": self.cfg.anatomy.unknown_anatomy_id}
        anatomy_id = int(chosen.get("anatomy_id", self.cfg.anatomy.unknown_anatomy_id))

        data = self._load_npz(path)
        ct = data["ct"]
        if ct.ndim == 4:
            ct = ct[0]
        orig_shape = tuple(ct.shape)               # original grid (per-axis) — for scaled ts-cache crop
        # Downsampled-totalseg cache: cut anatomy masks over 64³ instead of decompressing 256³.
        ts64 = None
        if getattr(self.cfg.data, "use_ts_cache", True):
            tsp = (path[:-4] if path.endswith(".npz") else path) + getattr(self.cfg.data, "ts_cache_suffix", "_ts64.npy")
            if os.path.exists(tsp):
                try:
                    ts64 = np.load(tsp)
                except Exception:
                    ts64 = None
        target_shape = self.cfg.volume.spatial_size
        slices, pads = make_crop_pad_transform(tuple(ct.shape), target_shape, random_crop=True)
        ct = apply_crop_pad_3d(ct, slices, pads, constant=-1000.0)
        dc = self.cfg.data
        # Slight HU jitter (shift + scale) in HU space, before windowing.
        if getattr(dc, "aug_intensity", False):
            ct = ct.astype(np.float32)
            if dc.aug_hu_scale > 0:
                ct = ct * float(np.random.uniform(1.0 - dc.aug_hu_scale, 1.0 + dc.aug_hu_scale))
            if dc.aug_hu_shift > 0:
                ct = ct + float(np.random.uniform(-dc.aug_hu_shift, dc.aug_hu_shift))
        # General window only by default (id 2): random windows would clip findings outside the
        # assigned window and break text grounding; general matches the seg/probe eval window.
        window_id = np.random.randint(0, 4) if getattr(dc, "vary_window", False) else 2
        vol = apply_window(ct, int(window_id))
        # Slight Gaussian noise in [0,1] space, after windowing.
        if getattr(dc, "aug_intensity", False) and dc.aug_noise_std > 0:
            vol = np.clip(vol + np.random.randn(*vol.shape).astype(np.float32) * dc.aug_noise_std, 0.0, 1.0)
        volume = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)

        ct_col = getattr(self.cfg.data, "ct_type_col", "ct_type")
        ct_type_id = ct_type_to_id(row[ct_col]) if ct_col in row.index else 0

        output: dict = {
            "volume": volume,
            "window_id": torch.tensor(window_id, dtype=torch.long),
            "ct_type_id": torch.tensor(ct_type_id, dtype=torch.long),
            "text": str(chosen.get("text", "")),
            "anatomy_id": torch.tensor(anatomy_id, dtype=torch.long),
            "utterance_label": str(chosen.get("label", "")),
            "negated": torch.tensor(bool(chosen.get("negated", False))),
            "uncertain": torch.tensor(bool(chosen.get("uncertain", False))),
            "path": path,
        }

        # When --use_cached_text is set, ALWAYS emit a tensor (zeros if the sidecar is missing
        # or the chosen utterance has no valid vector) so a partially-cached corpus never flips
        # the whole batch back to hashing. A zero vector is a neutral text embedding.
        cached_dim = getattr(self.cfg.text, "cached_dim", 1024)
        if use_cached:
            if cached is not None and chosen_idx < cached[0].shape[0] and bool(cached[1][chosen_idx]):
                output["text_emb_cached"] = cached[0][chosen_idx].float()
            else:
                output["text_emb_cached"] = torch.zeros(cached_dim)
        else:
            output["text_emb_cached"] = None

        anat_mask = self._anat_cropped(data, ts64, anatomy_id, orig_shape, slices, pads)
        if anat_mask is not None:
            output["anatomy_mask_s1"] = mask_to_flat_grid(anat_mask, self.cfg.patch.grid_size_s1)
            output["anatomy_mask_s2"] = mask_to_flat_grid(anat_mask, self.cfg.patch.grid_size_s2)
            output["anatomy_mask_s3"] = mask_to_flat_grid(anat_mask, self.cfg.patch.grid_size_s3)
        else:
            output["anatomy_mask_s1"] = None
            output["anatomy_mask_s2"] = None
            output["anatomy_mask_s3"] = None

        seg = extract_seg_mask(data, self.cfg) if self.cfg.anatomy.use_aux_segmentation else None
        if seg is not None:
            if seg.ndim == 4 and seg.shape[0] > 1:
                # Multi-channel one-hot -> label map.
                seg = np.argmax(seg, axis=0)
            seg = apply_crop_pad_3d(seg, slices, pads, constant=0.0)
            output["seg_mask"] = torch.from_numpy(seg.astype(np.int64))
        else:
            output["seg_mask"] = None

        # Intra-volume region↔utterance grounding: ONE slot PER POSITIVE UTTERANCE (padded to
        # max_regions). Multiple findings in the same lobe each get a slot — same anatomy mask,
        # different phrase/embedding — so the grounding head sees every phrase→region pair (not a
        # per-anatomy reduction). The anatomy mask is cached per aid to avoid recomputation, and
        # region_anatomy_ids lets the region-text contrastive skip same-anatomy false negatives.
        max_regions = getattr(self.cfg.anatomy, "max_regions", 0)
        if max_regions > 0:
            N3 = self.cfg.patch.num_tokens_s3
            masks_t = torch.zeros(max_regions, N3, dtype=torch.bool)
            valid_t = torch.zeros(max_regions, dtype=torch.bool)
            anat_ids = torch.zeros(max_regions, dtype=torch.long)
            texts = [""] * max_regions
            reg_embs = torch.zeros(max_regions, cached_dim) if use_cached else None
            mask_cache: dict = {}
            eligible = []                                              # (uidx, aid, mask3) that pass all filters
            for uidx, u in enumerate(utterances):
                aid = int(u.get("anatomy_id", 0))
                if aid <= 0 or aid == 18 or u.get("negated", False):   # skip unknown/other/negated
                    continue
                if use_cached:                                         # a slot needs a valid vector
                    if cached is None or not (uidx < cached[0].shape[0] and bool(cached[1][uidx])):
                        continue                                       # missing -> no valid slot
                if aid not in mask_cache:
                    m = self._anat_cropped(data, ts64, aid, orig_shape, slices, pads)
                    m3 = mask_to_flat_grid(m, self.cfg.patch.grid_size_s3) if m is not None else None
                    mask_cache[aid] = m3 if (m3 is not None and int(m3.sum()) > 0) else None
                if mask_cache[aid] is None:                            # anatomy not cuttable
                    continue
                eligible.append((uidx, aid, mask_cache[aid]))
            # DIVERSITY-FIRST: take one slot per distinct anatomy before any duplicate, so a lung-heavy
            # report (many same-anatomy findings first) doesn't fill all max_regions slots with one
            # anatomy and leave the region-text contrastive with no cross-anatomy negative.
            order = diversity_first_order([aid for _, aid, _ in eligible])
            for r, ei in enumerate(order[:max_regions]):
                uidx, aid, m3 = eligible[ei]
                masks_t[r] = m3
                valid_t[r] = True
                anat_ids[r] = aid
                texts[r] = str(utterances[uidx].get("text", ""))
                if use_cached:
                    reg_embs[r] = cached[0][uidx].float()
            output["region_masks_s3"] = masks_t
            output["region_valid"] = valid_t
            output["region_anatomy_ids"] = anat_ids
            output["region_texts"] = texts
            output["region_text_embs"] = reg_embs
        else:
            output["region_masks_s3"] = None
            output["region_valid"] = None
            output["region_anatomy_ids"] = None
            output["region_texts"] = None
            output["region_text_embs"] = None
        return output

    def __getitem__(self, idx: int) -> dict:
        max_retries = 5
        last = None
        for _ in range(max_retries):
            try:
                return self._load_one(idx)
            except (ValueError, OSError, EOFError, zipfile.BadZipFile, FileNotFoundError) as e:
                last = e
                idx = np.random.randint(0, len(self.df))
        raise RuntimeError(f"Failed to load sample after {max_retries} retries. Last error: {last}")


def art_collate_fn(batch: list[dict]) -> dict:
    out = {
        "volume": torch.stack([b["volume"] for b in batch]),
        "window_id": torch.stack([b["window_id"] for b in batch]),
        "ct_type_id": torch.stack([b["ct_type_id"] for b in batch]),
        "texts": [b["text"] for b in batch],
        "anatomy_id": torch.stack([b["anatomy_id"] for b in batch]),
        "negated": torch.stack([b["negated"] for b in batch]),
        "uncertain": torch.stack([b["uncertain"] for b in batch]),
        "paths": [b.get("path", "") for b in batch],
    }
    # Anatomy masks: PER-SAMPLE validity. A text-free/normal row has no anatomy mask; the old all-or-
    # nothing stack nulled anatomy_mask for the WHOLE batch if ANY sample lacked one (~57% of batch-10
    # batches at 8% text-free), disabling anatomy-guided masking + region-JEPA for every sample. Instead
    # stack zeros for the missing ones and carry an `anatomy_mask_valid` [B] flag so masking/region-JEPA
    # fall back per sample, not per batch.
    amask_valid = torch.tensor([b.get("anatomy_mask_s3") is not None for b in batch])
    for key in ("anatomy_mask_s1", "anatomy_mask_s2", "anatomy_mask_s3"):
        vals = [b.get(key) for b in batch]
        ref = next((v for v in vals if v is not None), None)
        out[key] = None if ref is None else torch.stack(
            [v if v is not None else torch.zeros_like(ref) for v in vals])
    out["anatomy_mask_valid"] = amask_valid
    for key in ("region_masks_s3", "region_valid", "region_anatomy_ids",
                "text_emb_cached", "region_text_embs"):
        vals = [b.get(key) for b in batch]
        out[key] = torch.stack(vals) if all(v is not None for v in vals) else None
    rtexts = [b.get("region_texts") for b in batch]
    out["region_texts"] = rtexts if all(v is not None for v in rtexts) else None
    segs = [b.get("seg_mask") for b in batch]
    out["seg_mask"] = torch.stack(segs) if all(s is not None for s in segs) else None
    return out
