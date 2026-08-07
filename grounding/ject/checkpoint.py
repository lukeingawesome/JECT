"""Checkpoint utilities for ART-JEPA, including high-resolution cooldown loading."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


def _infer_grid_from_tokens(n: int) -> tuple[int, int, int] | None:
    # Common cooldown is cubic, so try cube first.
    c = round(n ** (1.0 / 3.0))
    if c ** 3 == n:
        return (c, c, c)
    # Fallback: no safe inference.
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
    """Load checkpoint with safe interpolation/skipping for resolution cooldown.

    Handles:
      * factorized absolute position embeddings
      * full S3 absolute position embeddings when cubic grids can be inferred
      * skips mismatched relative-position-bias tables/window buffers
      * skips any remaining shape mismatch

    Guards against loading an OLD anatomy-conditioned checkpoint (anatomy_embed +
    text+anatomy ScaleConditioner) into the current text-only conditioner, which would
    silently reset the predictor's conditioning to random while the encoder loads
    (a half-initialized model that looks fine). Pass ``allow_conditioner_reset=True``
    (CLI ``--allow_conditioner_reset``) only when that reset is intentional.
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
                # pos_embed_s4 does NOT end with "pos_embed", so before this it fell through to
                # `skipped` and was silently left at RANDOM init on any resolution change. At
                # 224->336 the s4 grid goes 7^3 (343 tok) -> 11^3 (1331 tok), so it always
                # mismatches -- which would have started the cooldown with a random s4 position
                # code, the one scale --use_s4 and the s4 MUP prior (+0.163 swap-gap) depend on.
                new_v = _resize_full_pos(vv, dst)
            # Relative-position bias depends on window size; safer to reinitialize.
            if new_v is not None and new_v.shape == dst.shape:
                vv = new_v
                resized.append(k)
            else:
                skipped.append((k, "shape_mismatch", tuple(v.shape), tuple(dst.shape)))
                continue
        loaded[k] = vv

    # Loud-fail guard: a real ART checkpoint (saved as {"model": state_dict}) has student_encoder.*
    # params. If ZERO matched, we're almost certainly handed the wrong schema (e.g. a bare/foreign
    # dict) and would otherwise "succeed" while loading nothing → cooldown from random weights.
    enc_loaded = sum(1 for k in loaded if k.startswith("student_encoder."))
    if enc_loaded == 0:
        raise RuntimeError(
            f"load_flexible_checkpoint matched 0 'student_encoder.*' params from {checkpoint_path} "
            f"(loaded={len(loaded)}, checkpoint top-level keys={list(state.keys())[:8]}). "
            f"Wrong checkpoint schema? Expected torch.save({{'model': model.state_dict()}}).")

    # Conditioner-schema guard: an OLD anatomy-conditioned checkpoint has `anatomy_embed.*`
    # (now absent) and `cond_s{1,2,3}.*` whose input dim was text_dim+anatomy_dim (now text_dim).
    # Those keys land in `skipped`, so the text-only conditioner would silently start RANDOM while
    # the encoder loads fine — a half-initialized predictor that "looks successful" on cooldown.
    cond_reset_skips = [
        s for s in skipped
        if ("anatomy_embed" in s[0])
        or (s[0].startswith(("cond_s1.", "cond_s2.", "cond_s3.")) and s[1] == "shape_mismatch")
    ]
    if cond_reset_skips and not allow_conditioner_reset:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} uses the OLD anatomy-conditioned schema (anatomy_embed / "
            f"text+anatomy ScaleConditioner), incompatible with the current text-only conditioner. "
            f"{len(cond_reset_skips)} conditioning param(s) would be silently reset to random while "
            f"the encoder loads → a half-initialized predictor that looks fine. Examples: "
            f"{[s[0] for s in cond_reset_skips[:6]]}. If this reset is intentional (fresh re-pretrain "
            f"or migration off an old backbone), pass allow_conditioner_reset=True "
            f"(CLI: --allow_conditioner_reset).")
    if cond_reset_skips:
        # Reset is authorized → the ENTIRE conditioning stack must start fresh, not just the
        # shape-mismatched front. Shape-COMPATIBLE tensors (cond_s*.net.1.bias / net.3.*, and the
        # predictor FiLM pred_s*.[blocks.N.]cond.*) would otherwise load stale old-schema weights,
        # leaving a predictor that is neither old nor a clean text-only reset. Drop them all so
        # they retain their fresh (zero/default) init — honoring the "conditioner reset" contract.
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

    # New conditioning/grounding schema guard. The predictor FUSION (FiLM `.cond.` vs xattn `.fuse.`), the
    # predictor text projection, and the grounding / MUP heads are all schema-DEFINING. If the resume/
    # cooldown flags disagree with the checkpoint (e.g. an xattn+grounding checkpoint loaded into a FiLM/
    # no-grounding model, or vice-versa), the MODEL's params for the requested architecture land in
    # `missing`/`skipped` and load_state_dict(strict=False) still "succeeds" with a half-random predictor
    # or a randomly-initialized grounding head. We flag only params the MODEL EXPECTS but did not receive
    # (missing/skipped) — NOT `unexpected` checkpoint keys — so a downstream that legitimately omits the
    # heads (e.g. the linear probe) is never falsely blocked.
    def _is_arch_key(k: str) -> bool:
        # Schema-defining modules whose absence silently randomizes a whole capability. Includes the TEXT
        # path (text_in_proj / text_encoder / text_proj) and region_proj: a text_dim/version skew shape-
        # mismatches these, and load_state_dict(strict=False) would restore the encoder while leaving text
        # conditioning + region-text alignment RANDOM — even under allow_conditioner_reset.
        # `mup_heads` = current multi-scale text predictors; `mup_head` = legacy single-scale name.
        # List both explicitly rather than relying on `mup_head` being a string-prefix of `mup_heads`.
        # `cond_s4`/`pred_s4` = the 4-scale (s4 global) additions — a pre-s4 checkpoint lacks them, so
        # treat them as schema-defining too (else a 3-scale ckpt loads with s4 silently random).
        if k.startswith(("grounding_head", "mup_heads", "mup_head", "pred_text_proj",
                         "text_in_proj", "text_encoder", "text_proj", "region_proj",
                         "cond_s4", "pred_s4")):
            return True
        # s4 ENCODER stage (merge3 / stage4 / pos_embed_s4) under either the student or target encoder:
        # a 3-scale checkpoint has none of these, and load_state_dict(strict=False) would leave the whole
        # global stage random while reporting success.
        if any(tok in k for tok in ("merge3", "stage4", "pos_embed_s4")):
            return True
        return (k.startswith(("pred_s1.", "pred_s2.", "pred_s3."))
                and (".fuse." in k or ".cond." in k))
    arch_gaps = [k for k in missing if _is_arch_key(k)] + [s[0] for s in skipped if _is_arch_key(s[0])]
    if allow_conditioner_reset:                       # `.cond.` gaps are the intentional conditioner reset
        arch_gaps = [k for k in arch_gaps if ".cond." not in k]
    if arch_gaps and not allow_arch_reset:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing {len(arch_gaps)} predictor/grounding param(s) the "
            f"requested model expects — the resume/cooldown flags likely disagree with how the checkpoint "
            f"was trained (predictor_cond film vs xattn, or --use_grounding_head / --use_masked_utterance_pred "
            f"mismatch). load_state_dict would otherwise leave these RANDOM while reporting success. Examples: "
            f"{arch_gaps[:6]}. If this architecture change is intentional, pass allow_arch_reset=True "
            f"(CLI: --allow_arch_reset).")

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
