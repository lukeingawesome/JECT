"""Checkpoint loading that tolerates resolution changes and architecture differences."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


def _infer_grid_from_tokens(n: int) -> tuple[int, int, int] | None:
    # Only cubic grids can be inferred.
    c = round(n ** (1.0 / 3.0))
    if c ** 3 == n:
        return (c, c, c)
    return None


def _resize_factorized_pos(key: str, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor | None:
    # Shapes are [1,D,1,1,C], [1,1,H,1,C], [1,1,1,W,C].
    if src.ndim != 5 or dst.ndim != 5 or src.shape[-1] != dst.shape[-1]:
        return None
    if src.shape == dst.shape:
        return src
    # Move the non-singleton spatial axis to depth and interpolate as 1D.
    axis = [i for i in range(1, 4) if src.shape[i] != 1 or dst.shape[i] != 1]
    if len(axis) != 1:
        return None
    ax = axis[0]
    src_len, dst_len, C = src.shape[ax], dst.shape[ax], src.shape[-1]
    x = src.squeeze(0)
    if ax == 1:
        x = x[:, 0, 0, :].t().view(1, C, src_len)
    elif ax == 2:
        x = x[0, :, 0, :].t().view(1, C, src_len)
    else:
        x = x[0, 0, :, :].t().view(1, C, src_len)
    y = F.interpolate(x, size=dst_len, mode="linear", align_corners=False)
    y = y.view(C, dst_len).t()
    if ax == 1:
        return y.view(1, dst_len, 1, 1, C)
    if ax == 2:
        return y.view(1, 1, dst_len, 1, C)
    return y.view(1, 1, 1, dst_len, C)


def _resize_full_pos(src: torch.Tensor, dst: torch.Tensor, old_grid=None, new_grid=None) -> torch.Tensor | None:
    # Shapes [1,N,C].
    if src.ndim != 3 or dst.ndim != 3 or src.shape[-1] != dst.shape[-1]:
        return None
    if src.shape == dst.shape:
        return src
    old_grid = old_grid or _infer_grid_from_tokens(src.shape[1])
    new_grid = new_grid or _infer_grid_from_tokens(dst.shape[1])
    if old_grid is None or new_grid is None:
        return None
    B, N, C = src.shape
    x = src.view(1, *old_grid, C).permute(0, 4, 1, 2, 3)
    y = F.interpolate(x, size=new_grid, mode="trilinear", align_corners=False)
    return y.permute(0, 2, 3, 4, 1).reshape(1, math.prod(new_grid), C)


def load_flexible_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path,
                             strict: bool = False, verbose: bool = True,
                             allow_conditioner_reset: bool = False,
                             allow_arch_reset: bool = False) -> dict:
    """Load a checkpoint, interpolating or skipping tensors whose shapes changed.

    Handles:
      * factorized absolute position embeddings
      * full S3/S4 absolute position embeddings when cubic grids can be inferred
      * skips mismatched relative-position-bias tables/window buffers
      * skips any remaining shape mismatch

    Raises if an anatomy-conditioned checkpoint (anatomy_embed + text+anatomy
    ScaleConditioner) is loaded into the text-only conditioner, unless
    ``allow_conditioner_reset=True``, and if predictor/grounding/text/s4 parameters the
    model expects are absent, unless ``allow_arch_reset=True``.

    Returns:
        dict with counts of loaded, resized, skipped, missing and unexpected keys.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    loaded = {}
    skipped = []
    resized = []

    for k, v in state.items():
        if k not in current:
            skipped.append((k, "missing_in_model", tuple(v.shape) if hasattr(v, "shape") else None, None))
            continue
        dst = current[k]
        vv = v
        if hasattr(vv, "shape") and vv.shape != dst.shape:
            new_v = None
            if any(k.endswith(suffix) for suffix in ("pos_d", "pos_h", "pos_w")):
                new_v = _resize_factorized_pos(k, vv, dst)
            elif k.endswith("pos_embed") or k.endswith("pos_embed_s4"):
                new_v = _resize_full_pos(vv, dst)
            # Relative-position bias depends on window size; safer to reinitialize.
            if new_v is not None and new_v.shape == dst.shape:
                vv = new_v
                resized.append(k)
            else:
                skipped.append((k, "shape_mismatch", tuple(v.shape), tuple(dst.shape)))
                continue
        loaded[k] = vv

    # A valid checkpoint ({"model": state_dict}) always has student_encoder.* params.
    enc_loaded = sum(1 for k in loaded if k.startswith("student_encoder."))
    if enc_loaded == 0:
        raise RuntimeError(
            f"load_flexible_checkpoint matched 0 'student_encoder.*' params from {checkpoint_path} "
            f"(loaded={len(loaded)}, checkpoint top-level keys={list(state.keys())[:8]}). "
            f"Wrong checkpoint schema? Expected torch.save({{'model': model.state_dict()}}).")

    # Anatomy-conditioned checkpoints have `anatomy_embed.*` and `cond_s{1,2,3}.*` with input dim
    # text_dim + anatomy_dim; these land in `skipped` and would leave the conditioner at random init.
    cond_reset_skips = [
        s for s in skipped
        if ("anatomy_embed" in s[0])
        or (s[0].startswith(("cond_s1.", "cond_s2.", "cond_s3.")) and s[1] == "shape_mismatch")
    ]
    if cond_reset_skips and not allow_conditioner_reset:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} uses the anatomy-conditioned predictor schema (anatomy_embed / "
            f"text+anatomy ScaleConditioner), which is incompatible with the text-only conditioner. "
            f"{len(cond_reset_skips)} conditioning parameter(s) would be reset to random initialisation. "
            f"Examples: {[s[0] for s in cond_reset_skips[:6]]}. To accept the reset, pass "
            f"allow_conditioner_reset=True (CLI: --allow_conditioner_reset).")
    if cond_reset_skips:
        # On reset, drop every conditioning tensor, including shape-compatible ones, so the whole
        # conditioning stack keeps its fresh init instead of mixing in stale weights.
        def _is_conditioning_key(k: str) -> bool:
            return (k.startswith(("anatomy_embed", "cond_s1.", "cond_s2.", "cond_s3."))
                    or ".cond." in k)
        dropped = [k for k in list(loaded) if _is_conditioning_key(k)]
        for k in dropped:
            del loaded[k]
        if verbose:
            print(f"  [conditioner-reset] old anatomy-conditioned checkpoint: dropped {len(dropped)} "
                  f"shape-compatible conditioning tensor(s) from the load so the text-only conditioner "
                  f"+ predictor FiLM start fresh (mismatched: {len(cond_reset_skips)}).")

    missing, unexpected = model.load_state_dict(loaded, strict=False)

    # Architecture guard: flag parameters the model expects but did not receive (missing or skipped)
    # in schema-defining modules. Unexpected checkpoint keys are allowed, so models that omit heads
    # (e.g. for linear probing) still load.
    def _is_arch_key(k: str) -> bool:
        # Modules whose absence would leave a whole capability at random init: grounding/MUP heads,
        # the text path, region_proj, and the s4 conditioner/predictor.
        if k.startswith(("grounding_head", "mup_heads", "mup_head", "pred_text_proj",
                         "text_in_proj", "text_encoder", "text_proj", "region_proj",
                         "cond_s4", "pred_s4")):
            return True
        # s4 encoder stage, in either the student or target encoder.
        if any(tok in k for tok in ("merge3", "stage4", "pos_embed_s4")):
            return True
        return (k.startswith(("pred_s1.", "pred_s2.", "pred_s3."))
                and (".fuse." in k or ".cond." in k))
    arch_gaps = [k for k in missing if _is_arch_key(k)] + [s[0] for s in skipped if _is_arch_key(s[0])]
    if allow_conditioner_reset:                       # `.cond.` gaps are expected after a conditioner reset
        arch_gaps = [k for k in arch_gaps if ".cond." not in k]
    if arch_gaps and not allow_arch_reset:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing {len(arch_gaps)} predictor/grounding parameter(s) "
            f"that the requested model expects. The architecture flags probably differ from the ones the "
            f"checkpoint was trained with (predictor_cond film vs xattn, --use_grounding_head, "
            f"--use_masked_utterance_pred). These parameters would stay randomly initialised. Examples: "
            f"{arch_gaps[:6]}. To accept this, pass allow_arch_reset=True (CLI: --allow_arch_reset).")

    report = {"loaded": len(loaded), "resized": resized, "skipped": skipped,
              "missing_after_load": list(missing), "unexpected_after_load": list(unexpected)}
    if verbose:
        print(f"Flexible checkpoint load: loaded={report['loaded']} resized={len(resized)} skipped={len(skipped)}")
        if resized:
            print("  resized:", resized[:20], "..." if len(resized) > 20 else "")
        if skipped:
            print("  skipped examples:", skipped[:10])
    if strict and (missing or unexpected or skipped):
        raise RuntimeError(f"Strict flexible load failed: {report}")
    return report
