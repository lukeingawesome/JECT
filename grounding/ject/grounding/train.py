"""Training entry point for JECT phrase grounding."""
from __future__ import annotations
import os, sys, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from ject.checkpoint import load_flexible_checkpoint
from ject.grounding.dataset import RexGrounding, setup, build_cfg
from ject.grounding.evaluate import dice_score, evaluate
from ject.grounding.losses import (mil_peak_loss, attn_kl_loss, voxel_infonce_loss,
                                   tversky_bce_loss, coverage_loss)
from ject.grounding.model import TextGroundingJEPA
import ject.grounding.paths as paths   # module, so CLI overrides are visible everywhere


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--fusion", choices=["xattn", "none", "s2only"], default="xattn")
    # The s2 spatial-fusion head is structurally identical to the pretrained grounding head. THE thesis A/B:
    # 'pretrained' loads its weights, 'random' leaves it fresh. With a matched --seed, that is the ONLY diff.
    ap.add_argument("--s2_init", choices=["pretrained", "random"], default="random")
    ap.add_argument("--init_grounding", default=None,
                    help="Warm-start the TRAINED head (decoder + fusion + priors) from a previous "
                         "run's best.pt/latest.pt. Optimizer and LR-schedule state are NOT restored, "
                         "so this is a warm start, not a resume: training restarts at epoch 0 with a "
                         "fresh schedule. Use it to continue an interrupted run, or to start a new "
                         "config from a stronger point than random. Architecture must match exactly "
                         "-- a mismatch aborts rather than loading partially.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed torch/numpy/random + the DataLoader, so the pretrained-vs-random A/B differs "
                         "ONLY in the s2 head init (not in decoder init or data order).")
    ap.add_argument("--hi_ch", type=int, default=64, help="high-res decoder channels (112^3/224^3 stages)")
    ap.add_argument("--film", action="store_true", help="CEPA-style FiLM text conditioning at every scale")
    ap.add_argument("--deep_sup", action="store_true", help="deep supervision (aux mask heads at 28^3/56^3)")
    ap.add_argument("--predictor_prior", action="store_true",
                    help="reflect-the-training: inject the FROZEN predictor's CFG residual (pred_s2 phrase-null) at s2")
    ap.add_argument("--out_res", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=2)
    # CEPA's proven grounding recipe: FROZEN backbone, decoder lr 1e-3 constant-ish, wd 1e-4 (got 0.30-0.625).
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--unfreeze", action="store_true",
                    help="finetune the pretrained ART backbone too (default: FROZEN, like CEPA). "
                         "The grounding head stays trainable either way (for the prior A/B).")
    ap.add_argument("--use_s4", action="store_true",
                    help="add an encoder-e4 (7^3/1024) global-context skip into the decoder bottleneck. "
                         "e4 is 74%% of encoder params and carries the heaviest text supervision (MUP gamma=1.0) "
                         "but is otherwise computed and DISCARDED by the grounding decoder.")
    ap.add_argument("--enc_skips", nargs="*", default=[], choices=["s1", "s2", "s3"],
                    help="take these decoder SKIPS from the raw encoder stage (eX) instead of the "
                         "DenseLatentDecoder. Measured: the decoder destroys phrase specificity at "
                         "s2 (enc +0.042/8.8se -> dec +0.001/n.s.) and weakens it at s3.")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="Accumulate gradients over N micro-batches. Effective batch = --batch * N. "
                         "Use when VRAM caps --batch below the value the LR was tuned for: at full "
                         "resolution --batch is forced to 1, and running the batch-3 LR at batch 1 "
                         "means a 3x larger per-sample step. Costs no extra wall-time (same samples "
                         "processed), only fewer optimizer steps.")
    ap.add_argument("--sep_up4", action="store_true",
                    help="Full-resolution up-block with DEPTHWISE-SEPARABLE convs and an added "
                         "(not concatenated) skip: ~18x fewer FLOPs and half the peak memory of the "
                         "dense block, while still computing real convolutions at --res. Prefer over "
                         "--hires_head when output precision, not cost, is the constraint.")
    ap.add_argument("--hires_head", action="store_true",
                    help="Emit at full --res by interpolating the res/2 feature map and applying "
                         "only the 1x1x1 output conv (+ --txt_head dot product) there, skipping the "
                         "expensive up4 convolutions. ~1600x cheaper than native-res decoding and "
                         "lifts the res/2 Dice ceiling (~0.69 median). Set --out_res to --res.")
    ap.add_argument("--stem_stride2", action="store_true",
                    help="Downsample in the raw-CT stem's FIRST conv (res -> res/2) so no tensor "
                         "wider than the 1-channel input is ever held at full resolution. Ours "
                         "convolves at res^3 x hi_ch = 2.12e9 elements at 336/56 -- 0.99x the 2^31 "
                         "cuDNN limit and 6.54 TFLOP -- purely to feed up4's skip. CEPA's decoder "
                         "does this and trains ~6x faster at the same input resolution. Removes the "
                         "res-resolution skip, so pair with --out_res <res/2> or --hires_head.")
    ap.add_argument("--skip_fusion", choices=["concat", "add"], default="concat",
                    help="How decoder skips enter the fuse. 'concat' (default) stacks channels; "
                         "'add' projects the skip to the block width with a 1x1x1 and adds, halving "
                         "peak width at every fusion point. NOTE --sep_up4 already used an added "
                         "skip but ALSO swapped in depthwise-separable convs and scored 0.1397 -- "
                         "this flag isolates the fusion so the two are attributable separately.")
    ap.add_argument("--txt_head", action="store_true",
                    help="make the FULL-RESOLUTION logit text-conditioned via a dot product "
                         "between a projected phrase vector and the per-voxel embedding "
                         "(SAT/VoxTell/ESICA operator). Added to the existing conv head.")
    ap.add_argument("--censor_bg_w", type=float, default=1.0,
                    help="multiplier on the FALSE-POSITIVE penalty for CENSORED findings "
                         "(entity_count >= --censor_at). The annotation caps at 3 entities: 29.5%% "
                         "of TRAIN findings pile up there while val runs out to 36, so on those "
                         "samples unlabelled real lesions sit in the background and the FP term "
                         "trains the model to suppress true findings. 1.0 = current behaviour. "
                         "Needs $JECT_CHALLENGE_JSON, and aborts without it rather than silently "
                         "doing nothing.")
    ap.add_argument("--censor_at", type=int, default=3,
                    help="entity_count value at which the annotation is capped (see --censor_bg_w).")
    ap.add_argument("--coverage_w", type=float, default=0.0,
                    help="weight of the COVERAGE term softplus(m + s_N - Q_tau(z_P)): lift the LOW "
                         "quantile of positive logits above the hard negatives. Targets "
                         "foreground-coverage collapse -- the measured failure where the model "
                         "raises a confident seed but not the target's full support (oracle top-k "
                         "peaks at 1.5x|Y|, quality-among-hits pinned at 0.405). Unlike --mil_w, "
                         "which is satisfied by ONE positive out-ranking the negatives and so "
                         "strengthens the seed the model already has.")
    ap.add_argument("--coverage_tau", type=float, default=0.15,
                    help="quantile of the positive logits the coverage term lifts (0.10-0.25).")
    ap.add_argument("--mil_w", type=float, default=0.0,
                    help="weight of the CATMIL peak term -log(max p inside target). Targets the "
                         "measured failure directly: the peak is OUTSIDE the lesion for 73%% of "
                         "findings and a volume-oracle buys only +0.012.")
    ap.add_argument("--dense_loss", choices=["tversky", "infonce"], default="tversky",
                    help="PRIMARY dense loss. 'infonce' SUBSTITUTES voxel InfoNCE for Tversky "
                         "(a ranking objective instead of a set-overlap one). Note InfoNCE is "
                         "shift-invariant per sample -> the logit is NOT calibrated, so a small BCE "
                         "(--infonce_bce) is kept to anchor the sigmoid for fixed-threshold scoring.")
    ap.add_argument("--infonce_w", type=float, default=0.0,
                    help="ADDITIVE voxel-InfoNCE weight, on top of the primary loss.")
    ap.add_argument("--infonce_bce", type=float, default=1.0,
                    help="BCE weight retained when --dense_loss infonce, purely to calibrate the "
                         "logit scale (InfoNCE alone fixes only the ORDER of voxels).")
    ap.add_argument("--infonce_tau", type=float, default=0.1)
    ap.add_argument("--attn_kl_w", type=float, default=0.0,
                    help="weight of KL(normalised GT || softmax(text-image similarity)). Needs "
                         "--txt_head. Dense localisation signal on all voxels.")
    ap.add_argument("--tversky_a", type=float, default=0.3)
    ap.add_argument("--tversky_b", type=float, default=0.7)
    ap.add_argument("--pos_weight", type=float, default=50.0)
    ap.add_argument("--mup_logit", action="store_true",
                    help="add the MUP field to the OUTPUT LOGIT (not just the bottleneck gate), so "
                         "the decoder learns a residual over a working prior instead of being free "
                         "to ignore it. Needs --mup_prior.")
    ap.add_argument("--predictor_prior_s3", action="store_true",
                    help="CFG residual from the FROZEN pred_s3 (depth-6 RoPE, WITH image "
                         "self-attention) fused at s3. This is the predictor every audit "
                         "diagnostic measured; pred_s2 has no image self-attention.")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="grad-norm clip. Measured on real batches at ep22: |g| median 0.76, p90 2.18, max 5.4, and |g| tracks batch foreground fraction -- so clip=1.0 fires on 37.5%% of steps and preferentially shrinks the lesion-rich (informative) ones. 5.0 keeps a net for the rare blow-up without throttling signal.")
    ap.add_argument("--mup_prior", default=None, choices=["s2", "s3", "s4"],
                    help="feed the MUP head's phrase cross-attention map into the decoder "
                         "bottleneck. s4 is the strongest (gamma_mup=1.0, hit@0.1=0.53 frozen).")
    ap.add_argument("--data_root", default=None,
                    help="Dir of per-volume .npz files, each with `ct` and `rex`. Overrides $JECT_DATA.")
    ap.add_argument("--rex_csv", default=None,
                    help="ReXGrounding CSV: object_id, rex_split, rex_findings. Overrides $JECT_REX.")
    ap.add_argument("--text_emb", default=None,
                    help=".pt dict {phrase -> embedding} from the text encoder. Overrides $JECT_EMB.")
    ap.add_argument("--res", type=int, default=224,
                    help="Cube side the volume is cropped to. out_res must be res/2^k for some k>=0.")
    ap.add_argument("--out", default="runs/gf")
    args = ap.parse_args()

    lr_local, rank, world = setup(); dev = f"cuda:{lr_local}"; rank0 = rank == 0
    if world > 1:
        raise SystemExit("single-process only — launch with plain python (CUDA_VISIBLE_DEVICES), not torchrun.")
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Resolve dataset paths (CLI > env) and FAIL LOUDLY if unset/missing. A silent default here
    # would train on the wrong data and still report a plausible Dice.
    if args.data_root: paths.DATA = args.data_root
    if args.rex_csv:   paths.REX_CSV = args.rex_csv
    if args.text_emb:  paths.TEXT_EMB = args.text_emb
    paths.T = (args.res,) * 3
    for name, val, kind in (("--data_root/$JECT_DATA", paths.DATA, "dir"),
                            ("--rex_csv/$JECT_REX", paths.REX_CSV, "file"),
                            ("--text_emb/$JECT_EMB", paths.TEXT_EMB, "file")):
        if not val:
            raise SystemExit(f"{name} is not set. See README.md for the expected dataset layout.")
        if not (os.path.isdir(val) if kind == "dir" else os.path.isfile(val)):
            raise SystemExit(f"{name} = {val!r} does not exist ({kind}).")
    if args.out_res not in (args.res // 2, args.res):
        raise SystemExit(f"--out_res {args.out_res} invalid for --res {args.res}. The decoder emits "
                         f"only {args.res // 2} (after up3) or {args.res} (after up4).")

    cfg, ck = build_cfg(args.init)
    # Autotune is a big win at a FIXED 224^3 input, but at 336^3 the search itself runs full convs on
    # 37.9M-voxel tensors for every distinct shape in the decoder, which can stall the first step for
    # many minutes. GF_CUDNN_BENCH=0 disables it (default stays on, so 224 runs are unchanged).
    torch.backends.cudnn.benchmark = os.environ.get("GF_CUDNN_BENCH", "1") != "0"
    model = TextGroundingJEPA(cfg, out_res=args.out_res, fusion=args.fusion, hi_ch=args.hi_ch, use_s4=args.use_s4, enc_skips=tuple(args.enc_skips),
                              pred_frozen_input=(args.unfreeze and args.predictor_prior),
                              mup_prior=args.mup_prior,
                              predictor_prior_s3=args.predictor_prior_s3,
                              mup_logit=args.mup_logit, txt_head=args.txt_head,
                              hires_head=args.hires_head, sep_up4=args.sep_up4,
                              stem_stride2=args.stem_stride2, skip_fusion=args.skip_fusion,
                              film=args.film, deep_sup=args.deep_sup,
                              predictor_prior=args.predictor_prior).to(dev)
    load_flexible_checkpoint(model.art, args.init, verbose=rank0, allow_conditioner_reset=True)
    # align the trainable student path to the EMA target the features were shaped on (same as finetune).
    model.art.student_encoder.load_state_dict(model.art.target_encoder.state_dict())
    model.art.student_decoder.load_state_dict(model.art.target_decoder.state_dict())
    # THE prior A/B: the s2 spatial-fusion head is structurally identical to the pretrained grounding head.
    # 'pretrained' loads its weights (the thesis); 'random' leaves it fresh. Everything else is seed-matched.
    if args.s2_init == "pretrained":
        if not hasattr(model, "fuse2"):
            raise SystemExit("--s2_init pretrained needs a fuse2 (use --fusion xattn or s2only).")
        if model.art.grounding_head is None:
            raise SystemExit("--s2_init pretrained needs a pretrained grounding_head in the checkpoint.")
        assert model.art.grounding_scale == "s2", \
            f"grounding_head is at {model.art.grounding_scale}, but fuse2 is s2 — dims won't match."
        model.fuse2.load_from_grounding_head(model.art.grounding_head)
        if rank0: print("[s2_init] fuse2 loaded from PRETRAINED grounding_head", flush=True)
    if rank0:
        print(f"[init] {args.init} loaded | fusion={args.fusion} out_res={args.out_res} "
              f"s2_init={args.s2_init} seed={args.seed}", flush=True)

    embs = torch.load(paths.TEXT_EMB, map_location="cpu", weights_only=True)
    from torch.utils.data import DataLoader
    tr = RexGrounding("train", embs, augment=True); va = RexGrounding("val", embs)
    if rank0: print(f"[data] train={len(tr)} val={len(va)} findings", flush=True)
    gen = torch.Generator().manual_seed(args.seed)                # seeded shuffle -> identical data order across arms
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, pin_memory=True,
                     drop_last=True, generator=gen)
    val = DataLoader(va, batch_size=args.batch, num_workers=4)

    # FREEZE the whole pretrained ART backbone by default (CEPA freezes encoder+predictor -> its 0.30-0.625
    # recipe). The trainable prior now lives in model.fuse2 (a SpatialTextFuse outside art, warm-started from
    # the pretrained grounding head), so nothing inside art needs to stay trainable.
    # --unfreeze must mean "encoder + dense decoder", NOT "everything in art". Blanket-unfreezing
    # also makes pred_s2 trainable at the FULL lr, which destroys the predictor prior: its whole
    # contract (_predictor_delta) is a CFG residual from the FROZEN pretrained predictor. It would
    # also allocate optimizer state for mup_heads / pred_s1,s3,s4 (~226M) that are not even in the
    # forward graph.
    for p in model.art.parameters():
        p.requires_grad_(False)
    if args.unfreeze:
        for mod in (model.art.student_encoder, model.art.student_decoder):
            for p in mod.parameters():
                p.requires_grad_(True)
    steps = (len(trl) // max(1, args.grad_accum)) * args.epochs   # optimizer steps, not micro-batches
    if args.unfreeze:
        enc_params = list(model.art.student_encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
        opt = torch.optim.AdamW([{"params": other, "lr": args.lr},
                                 {"params": enc_params, "lr": args.lr * args.enc_lr_mult}],
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * args.enc_lr_mult],
                                                    total_steps=steps, pct_start=0.05)
    else:
        train_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    if rank0:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[train] frozen_backbone={not args.unfreeze} trainable={n_tr:.1f}M lr={args.lr:g} "
              f"wd={args.weight_decay:g} hi_ch={args.hi_ch}", flush=True)
    if args.init_grounding:
        gsd = torch.load(args.init_grounding, map_location="cpu", weights_only=False)
        gsd = gsd["model"] if isinstance(gsd, dict) and "model" in gsd else gsd
        gmiss, gunexp = model.load_state_dict(gsd, strict=False)
        # `art.` is the frozen backbone, already loaded from --init and stripped when best.pt was
        # written, so missing art.* keys are expected. Anything else differing means the head
        # architecture does not match and a partial load would train a silently different model.
        gm = [k for k in gmiss if not k.startswith("art.")]
        gu = [k for k in gunexp if not k.startswith("art.")]
        if gm or gu:
            raise SystemExit(
                f"--init_grounding architecture mismatch.\n"
                f"  missing: {len(gm)} e.g. {gm[:4]}\n  unexpected: {len(gu)} e.g. {gu[:4]}\n"
                f"Pass the same head flags the checkpoint was trained with.")
        if rank0:
            _ep = gsd.get("epoch") if isinstance(gsd, dict) else None
            print(f"[warm-start] head loaded from {args.init_grounding}"
                  f"{f' (was ep{_ep})' if _ep is not None else ''}; "
                  f"optimizer/schedule restart from scratch", flush=True)

    if rank0: os.makedirs(args.out, exist_ok=True)
    def _primary(lg, g, pw, bgw=None):
        """Primary dense loss at one scale. 'infonce' SUBSTITUTES ranking for set-overlap; the BCE
        term is kept only because InfoNCE is shift-invariant per sample, so without it
        sigmoid(logit) is uncalibrated and fixed-threshold Dice is meaningless."""
        if args.dense_loss == "infonce":
            l = voxel_infonce_loss(lg, g, tau=args.infonce_tau)
            if args.infonce_bce > 0:
                l = l + args.infonce_bce * F.binary_cross_entropy_with_logits(
                    lg, g, pos_weight=torch.tensor(pw, device=lg.device))
            return l
        return tversky_bce_loss(lg, g, alpha=args.tversky_a, beta=args.tversky_b, pos_weight=pw,
                                bg_w=bgw)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for i, batch in enumerate(trl):
            vol, q, gt = batch["vol"].to(dev), batch["q"].to(dev), batch["gt"].to(dev)
            bgw = None
            if args.censor_bg_w != 1.0:
                ec = batch["entity_count"].to(dev).float()
                if bool((ec < 0).all()):
                    raise SystemExit(
                        "--censor_bg_w needs entity_count, which is -1 for every sample. Set "
                        "$JECT_CHALLENGE_JSON to MICCAI_challenge_dataset.json (see SETUP.md §2).")
                bgw = torch.where(ec >= args.censor_at,
                                  torch.full_like(ec, args.censor_bg_w), torch.ones_like(ec))
            if i % args.grad_accum == 0:
                opt.zero_grad(set_to_none=True)
            # adaptive_max_pool3d, NOT a fixed integer kernel: a fixed kernel needs out_res to
            # divide T exactly, which forbids e.g. out_res=224 at T=336 (336/224 = 1.5). Adaptive
            # pooling handles any target and still MAX-pools, so a sub-cell lesion survives.
            gtr = F.adaptive_max_pool3d(gt.unsqueeze(1), (args.out_res,) * 3)[:, 0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(vol, q, window_id=torch.full((vol.shape[0],), 2, device=dev),
                            ct_type_id=torch.zeros(vol.shape[0], dtype=torch.long, device=dev))  # non-contrast
                if isinstance(out, tuple):                       # deep supervision
                    logit, aux = out
                    loss = _primary(logit.float(), gtr, args.pos_weight, bgw)
                    for a, w in aux:                             # downsample GT to each aux scale
                        ga = F.max_pool3d(gtr.unsqueeze(1), kernel_size=gtr.shape[-1] // a.shape[-1])[:, 0]
                        # scale pos_weight by the pooling factor: the positive FRACTION is
                        # ~k^3 higher at the aux scale, so reusing 50 grossly over-weights it.
                        k = gtr.shape[-1] // a.shape[-1]
                        loss = loss + w * _primary(a.float(), ga, max(1.0, args.pos_weight / k ** 3),
                                                   bgw)
                else:
                    logit = out
                    loss = _primary(logit.float(), gtr, args.pos_weight, bgw)
                if args.infonce_w > 0:
                    loss = loss + args.infonce_w * voxel_infonce_loss(logit.float(), gtr,
                                                                      tau=args.infonce_tau)
                if args.mil_w > 0:
                    loss = loss + args.mil_w * mil_peak_loss(logit.float(), gtr)
                if args.coverage_w > 0:
                    loss = loss + args.coverage_w * coverage_loss(logit.float(), gtr,
                                                                  tau=args.coverage_tau)
                if args.attn_kl_w > 0 and getattr(model, "_txt_sim", None) is not None:
                    loss = loss + args.attn_kl_w * attn_kl_loss(model._txt_sim.float(), gtr)
            (loss / args.grad_accum).backward()          # scale so the accumulated grad is a mean
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step(); sched.step()
            if rank0 and i % 5 == 0:
                g2 = float(model.fuse2.gamma) if hasattr(model, "fuse2") else float("nan")
                print(f"ep{ep} step{i} loss={float(loss):.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"fuse2.gamma={g2:.3f}", flush=True)
        hr, md, tot = evaluate(model, val, dev, rank0)
        if rank0:
            best = max(best, md)
            # md is the BEST-of-swept-threshold val Dice (val-tuned, exploratory — not a fixed-threshold score).
            print(f"=== [ep{ep+1}] val best-thr Dice={md:.4f}  hit@0.1={hr:.4f}  (best={best:.4f}, n={tot}) "
                  f"[fusion={args.fusion} s2={args.s2_init}] ===", flush=True)
            # Strip `art.` when the backbone is frozen: it is byte-identical to --init, and 546M of
            # 578M params would otherwise be rewritten every epoch (2.31 GB -> 127 MB). That is what
            # makes keeping a separate best.pt free. ALSO keep best.pt -- previously only latest.pt was
            # written, so a dip after the peak silently destroyed the best model.
            sd = model.state_dict()
            if not args.unfreeze:
                sd = {k: v for k, v in sd.items() if not k.startswith("art.")}
            ck = {"model": sd, "epoch": ep + 1, "mean_dice": md, "art_from": args.init}
            torch.save(ck, os.path.join(args.out, "latest.pt"))
            if md >= best:
                torch.save(ck, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
