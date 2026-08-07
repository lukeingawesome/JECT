"""Text-conditioned segmentation head over a frozen JECT backbone."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

from ject.backbone.model import ARTJEPAModel, CrossAttention
from ject.backbone.rope import make_3d_coords
from ject.decoder import ConvBlock3D, UpBlock3D, SepUpBlock3D


class SpatialTextFuse(nn.Module):
    """Genuinely SPATIAL text fusion, structurally identical to UtteranceGroundingHead so the PRETRAINED
    grounding head's weights (incl. null_query) load directly (the s2 prior).

    The phrase is a QUERY that attends OVER the image tokens (KV=image, softmax over N>1 -> non-degenerate).
    A learned null_query gives the text-agnostic prior, and the gate is the VALIDATED CFG residual
    `sigmoid(text) - sigmoid(null)` (same readout art_grounding_gate scored) -> ~0 in generic tissue, positive
    on the finding. Features are modulated `img * (1 + gamma * residual)`.

    `gamma` starts LIVE (gamma_init, same in both A/B arms) so the copied pretrained map affects the forward
    from step 0 — a zero-init would leave the pretrained weights gradient-free and bf16-invisible for a while."""

    def __init__(self, dim, cached_dim, num_heads=8, gamma_init=1.0):
        super().__init__()
        self.q_in = nn.Linear(cached_dim, dim)
        self.null_query = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.null_query, std=0.02)
        self.cross = CrossAttention(dim, dim, num_heads)          # query attends OVER image tokens
        self.q_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.img_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def load_from_grounding_head(self, gh):
        """Copy the pretrained UtteranceGroundingHead (q_in, null_query, cross, q_out, img_out)."""
        self.q_in.load_state_dict(gh.q_in.state_dict())
        self.null_query.data.copy_(gh.null_query.data)
        self.cross.load_state_dict(gh.cross.state_dict())
        self.q_out.load_state_dict(gh.q_out.state_dict())
        self.img_out.load_state_dict(gh.img_out.state_dict())

    def _logit(self, q, img):                                    # q [B,Q,dim], img [B,N,dim] -> [B,Q,N]
        q_ref, _ = self.cross(q, img)
        return torch.einsum("bqd,bnd->bqn", self.q_out(q_ref), self.img_out(img)) * self.scale

    def forward(self, img_tok, phrase):                          # img_tok [B,N,dim], phrase [B,cached]
        B = img_tok.shape[0]
        text = self._logit(self.q_in(phrase).unsqueeze(1), img_tok)          # [B,1,N]
        null = self._logit(self.null_query.expand(B, -1, -1), img_tok)       # [B,1,N]
        residual = (torch.sigmoid(text) - torch.sigmoid(null)).transpose(1, 2)  # [B,N,1] CFG readout
        return img_tok * (1 + self.gamma * residual)             # spatial CFG gating, live from step 0


class FiLM3d(nn.Module):
    """CEPA-style channel-wise text conditioning at a decoder stage: x*(1+gamma)+beta from the phrase,
    zero-init -> identity warm-start. Complements the spatial CFG gate with global per-channel modulation."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gb.weight); nn.init.zeros_(self.to_gb.bias)

    def forward(self, x, phrase):                                # x [B,C,D,H,W], phrase [B,cond]
        g, b = self.to_gb(phrase).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None]


class TextGroundingJEPA(nn.Module):
    """ART Swin encoder+decoder + multi-scale SPATIAL text fusion (SpatialTextFuse) + UNETR upsampler.
    Optional CEPA-style FiLM-every-scale (--film) and deep supervision (--deep_sup)."""

    def __init__(self, cfg, out_res=224, fusion="xattn", hi_ch=32, film=False, deep_sup=False,
                 predictor_prior=False, use_s4=False, enc_skips=(), pred_frozen_input=False,
                 mup_prior=None, predictor_prior_s3=False, mup_logit=False, txt_head=False,
                 hires_head=False, sep_up4=False):
        super().__init__()
        self.out_res = out_res
        # Input cube side implied by the backbone geometry (grid_s1 * patch). The decoder emits
        # res/2 after up3 and res after up4, so `out_res` is only valid at those two values --
        # gating on a literal 224 silently mismatches at any other input resolution.
        _ps = cfg.patch.patch_size
        self._res = int(cfg.patch.grid_size_s1[0]) * int(_ps[0] if isinstance(_ps, (tuple, list)) else _ps)
        self.fusion = fusion
        self.predictor_prior = predictor_prior
        # ENCODER SKIPS (--enc_skips). Measured on the ep40 checkpoint with a native-grid
        # phrase->voxel probe (GT max-pooled, 96 findings, 6 splits): the phrase-discriminative
        # signal is in the ENCODER at every scale, and the DenseLatentDecoder WIPES IT OUT at s2 --
        #   s2: enc +0.042 (8.8se)  vs  dec +0.001 (n.s.)   <- total loss
        #   s3: enc +0.047 (4.9se)  vs  dec +0.035 (3.1se)
        #   s1: enc +0.025 (5.6se)  vs  dec +0.024 (7.0se)  <- equal, leave alone
        # s2 is where the grounding head, weak-grounding, fuse2 and the predictor prior ALL read,
        # so they consume the one stream where the signal is gone. This swaps the decoder SKIP to
        # the encoder stage. NOTE the predictor prior still takes s_dec['s2'] as ITS input -- pred_s2
        # was pretrained on decoder features, so feeding it e2 would be off-distribution.
        self.enc_skips = tuple(enc_skips)
        # PREDICTOR INPUT SOURCE. _predictor_delta feeds s_dec['s2'] into the FROZEN pred_s2, and
        # the CFG residual is only meaningful because pred_s2 sees the feature distribution it was
        # pretrained to complete. With --unfreeze the student encoder/decoder become trainable, so
        # that input DRIFTS and the prior silently decays over training -- undermining the exact
        # mechanism the prior exists to exploit. When the backbone is trainable we therefore read
        # the predictor's input from the FROZEN EMA target stream (target_encoder/target_decoder,
        # still the untouched ep40 weights) instead. Costs one extra no_grad encoder forward.
        # With a frozen backbone student==target exactly (main() copies target->student at init and
        # it never updates), so the extra pass is skipped.
        self.pred_frozen_input = pred_frozen_input
        # MUP PRIOR (--mup_prior s2|s3|s4). The MUP head was trained so that predicting a masked
        # utterance REQUIRES attending to the region containing it -- its docstring calls the
        # cross-attention map 'the grounding'. Measured on the frozen ep40 head (value-path
        # readout, 80 CTs), the phrase-localization signal tracks the MUP weight exactly:
        #   s2 (gamma=0.25): gap -0.002 n.s.   s3 (0.50): +0.017   s4 (1.00): +0.033 [2.2se],
        #   hit@0.1 = 0.53 at s4 with ZERO finetuning.
        # Here the phrase is fed as the QUERY through q_in. q_in is UNTRAINED in the checkpoint
        # (masked slots overwrite it with mask_token and only masked slots enter the MUP loss, so
        # it never saw a gradient) -- so it is free to learn here at no cost to pretrained weights.
        # geom=None ON PURPOSE: relbias needs a per-phrase anatomy region, which ReXGrounding does
        # not provide, and using the GT mask's own centroid would leak the answer. Without it the
        # phrase alone must drive attention -- the purest form of the test.
        self.mup_prior = mup_prior
        self.mup_logit = mup_logit
        self.txt_head_on = txt_head
        # HIGH-RES OUTPUT WITHOUT CONVOLVING AT FULL RESOLUTION.
        # up4 runs two CxC 3x3x3 convs on the res^3 grid: 12.85 TFLOP at 336^3/C=56, which is why
        # native-resolution output cost ~50 h/epoch. But `out` is a 1x1x1 conv and the txt_head
        # readout is a dot product -- both are O(C) per voxel, ~1600x cheaper. So we can lift the
        # res/2 feature map to full resolution by interpolation and apply only those.
        # This matters because a res/2 output caps Dice at ~0.69 median regardless of the model
        # (measured on GT: 0.517 for findings <500 voxels), and that ceiling, not capacity, is what
        # limits quality-among-hits. Features are still res/2-derived, so it lifts the ceiling
        # rather than removing it.
        self.hires_head = hires_head
        # SEPARABLE full-resolution up-block. Unlike hires_head (which lifts res/2 features by
        # interpolation), this runs REAL convolutions on the res^3 grid at ~18x fewer FLOPs and half
        # the peak memory of the dense block (skip is ADDED, not concatenated). Use when the res/2
        # output ceiling is the binding constraint on precision.
        self.sep_up4 = sep_up4
        # PREDICTOR PRIOR AT s3 (--predictor_prior_s3). EVERY audit diagnostic that showed the
        # predictor grounds was measured on pred_s3 -- Dregion +0.0475 (5.9se), Dswap +0.0188
        # (2.3se) -- yet the finetune only ever used pred_s2. They are different architectures:
        # pred_s2 (XAttnPredictionHead) is a PER-TOKEN MLP + text cross-attn with NO image
        # self-attention, so its residual can only be a per-token text modulation; pred_s3
        # (RopeGlobalPredictor3D, depth 6) has RoPE self-attention over image tokens, so its
        # residual can carry genuinely spatial phrase evidence.
        self.predictor_prior_s3 = predictor_prior_s3
        self.art = ARTJEPAModel(cfg)
        if mup_prior:
            if getattr(self.art, 'mup_heads', None) is None or mup_prior not in self.art.mup_heads:
                raise SystemExit(f'--mup_prior {mup_prior} needs mup_heads[{mup_prior}] in the '
                                 f'checkpoint (got {list(getattr(self.art, "mup_heads", {}) or {})})')
        mc = cfg.model
        self.g3 = tuple(cfg.patch.grid_size_s3); self.g2 = tuple(cfg.patch.grid_size_s2); self.g1 = tuple(cfg.patch.grid_size_s1)
        self.d3, self.d2, self.d1 = mc.dim_s3, mc.dim_s2, mc.dim_s1
        # TAPER channels at high resolution (nnU-Net style). Running d1=128-ch 3D convs at 112^3/224^3
        # cost ~21s/forward for up4 alone (~3e13 FLOPs on a slow kernel); hi_ch=32 cuts that ~16x and
        # frees the activation memory that was pinning batch=2. Mask quality comes from the skips, not
        # from wide high-res channels.
        cs = hi_ch
        H = 8
        CACHED = cfg.text.cached_dim                              # 1024 (phrase = Chest2Vec)
        # per-scale SPATIAL phrase fusion (phrase-query attends OVER image tokens -> per-voxel gate).
        if fusion == "xattn":
            self.fuse3 = SpatialTextFuse(self.d3, CACHED, H)
            self.fuse2 = SpatialTextFuse(self.d2, CACHED, H)
            self.fuse1 = SpatialTextFuse(self.d1, CACHED, H)
        elif fusion == "s2only":
            self.fuse2 = SpatialTextFuse(self.d2, CACHED, H)      # single-scale ablation
        # The s2 fuse is structurally identical to the pretrained grounding head (both at grounding_scale=s2):
        # --s2_init pretrained loads its weights (the thesis), random leaves it fresh. main() loads AFTER
        # the checkpoint is restored.
        # S4 SKIP (--use_s4). The pretrained encoder's global stage e4 (7^3, 1024-d) is 74% of the
        # encoder's parameters and carries the HEAVIEST text supervision in pretraining (MUP gamma=1.0,
        # plus the s4 JEPA at lambda=0.30) -- the spec calls it "the utterance's semantic text home".
        # DenseLatentDecoder emits only {s1,s2,s3}, so without this the bottleneck starts COLD at s3 with
        # no global context and e4 is computed and discarded on every forward. 7^3 is coarse (each token
        # ~32^3 voxels) so it is used as CONTEXT into the bottleneck, not for precision -- the fine detail
        # still comes from the s1/stem skips downstream.
        self.use_s4 = use_s4
        if use_s4:
            self.g4 = tuple(cfg.patch.grid_size_s4); self.d4 = mc.dim_s4
            # 1x1x1 convs ON PURPOSE. 3x3x3 ConvBlock3Ds here cost +43.5M trainable params on a 26.3M
            # decoder -- that would confound "does e4 INFORMATION help" with "does 2.6x more decoder
            # CAPACITY help". 1x1 keeps the added capacity ~1M so the ablation stays interpretable, and
            # spatial mixing is not needed anyway: e4 is a 7^3 global summary used as context.
            self.s4_proj = nn.Conv3d(self.d4, self.d3, 1)             # 1024 -> 512, at 7^3
            self.s4_fuse = nn.Conv3d(self.d3 + self.d3, self.d3, 1)   # concat with s3 at 14^3
        if mup_prior:
            # LIVE multiplicative gate, matching SpatialTextFuse's `img * (1 + gamma * residual)`
            # (:77) and its rationale (:47-48): a zero/identity-init leaves the pretrained signal
            # gradient-free and bf16-invisible for many epochs. Measured: with an identity-init concat
            # conv the arm matched its own baseline exactly at ep1 (0.1061 vs 0.1039) -- the map was
            # contributing nothing. The field is a STRONG frozen signal (swap-gap +0.163, 5.3se,
            # hit@0.1 0.969), so it should steer the bottleneck from step 0.
            self.mup_gamma = nn.Parameter(torch.tensor(1.0))
            # LOGIT-LEVEL prior (--mup_logit). The gate above sits 4 conv stages upstream of the
            # output, so the decoder can wash it out -- measured: an identity-init concat reproduced
            # the baseline exactly at ep1 (0.1061 vs 0.1039). Adding the field to the LOGIT makes it
            # unavoidable: the decoder starts from a working prior and learns only the RESIDUAL over
            # it. The field scores Dice 0.403 / hit@0.1 0.969 standalone at 7^3, so this makes that a
            # floor rather than a suggestion. w init 1.0 (never 0), same live-from-step-0 rationale.
            self.mup_logit_w = nn.Parameter(torch.tensor(1.0))
        if predictor_prior_s3:
            assert getattr(self.art, 'pred_s3', None) is not None, \
                '--predictor_prior_s3 needs pred_s3 in the checkpoint'
            # Concat + 3x3x3 ConvBlock3D, mirroring the WORKING s2 prior (:214). Concat is right
            # here (and was wrong for the MUP map) because the delta is d3-dimensional -- a full
            # 512-ch tensor entering at unit scale through a randomly-init block -- not a single
            # identity-initialised channel among 513.
            self.pred_fuse3 = ConvBlock3D(self.d3 + self.d3, self.d3)
        # UNETR decoder
        self.bottleneck = ConvBlock3D(self.d3, self.d3)
        self.up1 = UpBlock3D(self.d3, self.d2, self.d2)          # 14->28, skip s2
        self.up2 = UpBlock3D(self.d2, self.d1, self.d1)          # 28->56, skip s1
        self.stem0 = ConvBlock3D(cfg.volume.channels, cs)                                       # 224^3
        self.stem1 = nn.Sequential(nn.Conv3d(cs, cs, 3, 2, 1, bias=False),
                                   nn.InstanceNorm3d(cs, affine=True), nn.LeakyReLU(0.01, inplace=True))  # 112^3
        self.up3 = UpBlock3D(self.d1, cs, cs)                    # 56->112, skip stem1
        self.up4 = (SepUpBlock3D(cs, cs, cs) if sep_up4 else
                    UpBlock3D(cs, cs, cs))                       # res/2 -> res, skip stem0
        self.out = nn.Conv3d(cs, 1, 1)
        # TEXT-CONDITIONED OUTPUT (--txt_head). The final decision was text-AGNOSTIC: all fusion
        # stopped at 56^3 and `self.out` is a plain conv, so the highest-resolution logit never saw
        # the phrase. SAT / VoxTell / ESICA / CLIP-Driven all converge on the same fix -- the mask
        # logit IS a dot product between a text-derived vector and the per-voxel embedding. VoxTell's
        # source does exactly this at full resolution with NO conv head at all, from a single POOLED
        # 1024-d vector (last_token_pool), i.e. the same form as Chest2Vec. ESICA's isolated ablation
        # of this operator: 0.659 -> 0.585 DSC, attributed to small/weakly-contrasted structures.
        # Kept ADDITIVE over the existing conv (w init 1.0) rather than replacing it, so the working
        # head survives and the text path is a measurable delta.
        self.txt_head = nn.Sequential(nn.LayerNorm(CACHED), nn.Linear(CACHED, cs),
                                      nn.GELU(), nn.Linear(cs, cs))
        # ZERO-init (NOT 1.0). txt_head is RANDOMLY initialised, so w=1.0 adds a random cosine map
        # to the final logit at full strength: measured at init, std(txt_w*sim)/std(base_logit) = 0.32,
        # i.e. a third of the logit variance is noise -- on a decoder whose fuse2 is deliberately
        # warm-started from the pretrained grounding head. Arms that work start at ep1 Dice
        # 0.104-0.115; with w=1.0 this arm started at 0.044. The w=1.0 rationale was copied from
        # mup_logit_w, where it IS right because that field is pretrained and validated (Dice 0.403
        # standalone) -- a random projection is the opposite case. Zero-init also matches this file's
        # own convention (FiLM is zero-init identity).
        # NOTE this does NOT disable the text path: attn_kl_loss trains txt_head through `_txt_sim`
        # directly, independent of txt_w, and dL/d(txt_w) = <dL/dlogit, sim> != 0, so the gate opens
        # on its own once the map carries signal.
        self.txt_w = nn.Parameter(torch.tensor(0.0))
        # CEPA-style FiLM at every decoder stage (global per-channel text conditioning, zero-init identity).
        self.film = film
        if film:
            self.film3 = FiLM3d(CACHED, self.d3); self.film2 = FiLM3d(CACHED, self.d2)
            self.film1 = FiLM3d(CACHED, self.d1); self.film0 = FiLM3d(CACHED, cs)
        # Deep supervision: aux mask heads at 28^3 (post up1) and 56^3 (post up2), CEPA weights 0.25/0.5.
        self.deep_sup = deep_sup
        if deep_sup:
            self.aux28 = nn.Conv3d(self.d2, 1, 1); self.aux56 = nn.Conv3d(self.d1, 1, 1)
        # PREDICTOR-as-prior (reflect-the-training): read grounding from the FROZEN pretrained predictor's
        # CFG residual  delta = pred_s2(phrase) - pred_s2(null)  [B,N2,d2] at s2, and fuse it into the decoder.
        # This is the trained mechanism (utterance-guided masked prediction), not a fresh head.
        self.predictor_prior = predictor_prior
        if predictor_prior:
            assert getattr(self.art, "pred_s2", None) is not None, \
                "--predictor_prior needs pred_s2 in the checkpoint (predictor_cond=xattn, s2 in xattn scales)"
            self.pred_fuse = ConvBlock3D(self.d2 + self.d2, self.d2)   # fuse s2 decoder feats + predictor delta

    def _to3d(self, tok, grid, dim):
        return tok.transpose(1, 2).reshape(tok.shape[0], dim, *grid)

    def _predictor_delta(self, s2_tok, qemb):
        """CFG residual from the FROZEN predictor at s2: pred_s2(phrase) - pred_s2(null).  [B,N2,d2]."""
        art = self.art; B = s2_tok.shape[0]; dev = s2_tok.device
        drop_all = torch.ones(B, dtype=torch.bool, device=dev)
        c_pos = art.encode_condition(None, None, None, B, dev, text_emb_cached=qemb, text_drop=None)
        c_null = art.encode_condition(None, None, None, B, dev, text_emb_cached=qemb, text_drop=drop_all)
        c2_pos, c2_null = c_pos[1], c_null[1]                         # s2 FiLM/AdaLN conditioning
        pred_txt = art.pred_text_proj(qemb.unsqueeze(1))             # [B,1,text_dim]
        m_pos = torch.ones(B, 1, dtype=torch.bool, device=dev)
        m_null = torch.zeros(B, 1, dtype=torch.bool, device=dev)     # null path: xattn attends to nothing
        p_pos = art.pred_s2(s2_tok, c2_pos, pred_txt, m_pos)         # [B,N2,d2]
        p_null = art.pred_s2(s2_tok, c2_null, pred_txt, m_null)
        return p_pos - p_null                                        # [B,N2,d2]

    def _predictor_delta_s3(self, s3_tok, qemb):
        """CFG residual from the FROZEN predictor at s3: pred_s3(phrase) - pred_s3(null).  [B,N3,d3].

        Mirrors _predictor_delta exactly but at s3, where encode_condition's 3rd element is the s3
        conditioning (it returns c1, c2, c3, aux -- art_jepa_model.py:740) and pretraining calls
        pred_s3(s_dec["s3"], c3, pred_txt, mask) (:854). The null path uses an ALL-FALSE text mask,
        not text_tokens=None, so both calls traverse the identical block stack -- the same choice that
        keeps the s2 prior clear of the gated-FFN null-path inconsistency."""
        art = self.art; B = s3_tok.shape[0]; dev = s3_tok.device
        drop_all = torch.ones(B, dtype=torch.bool, device=dev)
        c_pos = art.encode_condition(None, None, None, B, dev, text_emb_cached=qemb, text_drop=None)
        c_null = art.encode_condition(None, None, None, B, dev, text_emb_cached=qemb, text_drop=drop_all)
        c3_pos, c3_null = c_pos[2], c_null[2]
        pred_txt = art.pred_text_proj(qemb.unsqueeze(1))
        m_pos = torch.ones(B, 1, dtype=torch.bool, device=dev)
        m_null = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        return art.pred_s3(s3_tok, c3_pos, pred_txt, m_pos) - art.pred_s3(s3_tok, c3_null, pred_txt, m_null)

    @torch.no_grad()
    def _mup_field(self, img):
        """Frozen full-depth 'text-prediction field': t_hat[j] = MUP(mask_query, KV={z_j}) per site.

        MUP was trained image->text; its only text->image port (q_in) never received a gradient, so
        DRIVING it with a phrase trains a new module. INVERTING it -- asking what text each site
        predicts, then scoring that against the phrase -- runs the TRAINED direction and needs no
        parameters at all. K=1 makes the softmax exactly delta_j, so block 0 emits proj(v_j) and
        blocks 1-5 apply their real MLPs (they are live: proj std grew from zero-init to ~70% of
        block 0). Measured frozen on 64 val findings at s4: SWAP-GAP +0.163 +/-0.031 (5.3se),
        hit@0.1 0.969 -- vs +0.033 n.s. for a block-0 linearisation, and vs +0.042/+0.047 for
        *trained* bilinear probes on raw encoder features. The field is PHRASE-INDEPENDENT."""
        head = self.art.mup_heads[self.mup_prior]
        B, N, C = img.shape
        out = []
        for b in range(B):
            kv1 = img[b].reshape(N, 1, C)                                  # one key per query
            q0 = torch.zeros(N, 1, head.q_in.in_features, device=img.device, dtype=img.dtype)
            mk = torch.ones(N, 1, dtype=torch.bool, device=img.device)     # mask_token = trained query
            p, _ = head(q0, kv1, mk, geom=None)
            out.append(p[:, 0])
        return torch.stack(out)                                            # [B,N,cached_dim]

    def _fuse(self, x, mod, phrase):
        B, C, D, Hh, W = x.shape
        tok = x.flatten(2).transpose(1, 2)                       # [B,N,C] image tokens
        ref = mod(tok, phrase)                                   # phrase-query attends OVER image -> gated [B,N,C]
        return ref.transpose(1, 2).reshape(B, C, D, Hh, W)

    def forward(self, vol, qemb, window_id=None, ct_type_id=None):
        s_enc = self.art.student_encoder(vol, window_id=window_id, mask_s1=None, ct_type_id=ct_type_id)
        s_dec = self.art.student_decoder(s_enc)
        pick = lambda sc: (s_enc["e" + sc[1:]] if sc in self.enc_skips else s_dec[sc])
        f3 = self._to3d(pick("s3"), self.g3, self.d3)
        f2 = self._to3d(pick("s2"), self.g2, self.d2)
        f1 = self._to3d(pick("s1"), self.g1, self.d1)
        if self.use_s4:                                          # global context from the encoder's e4
            f4 = self._to3d(s_enc["e4"], self.g4, self.d4)       # [B,1024,7,7,7]
            f4 = F.interpolate(self.s4_proj(f4), size=self.g3, mode="trilinear", align_corners=False)
            f3 = self.s4_fuse(torch.cat([f3, f4.to(f3.dtype)], dim=1))
        if self.predictor_prior_s3:
            d3 = self._to3d(self._predictor_delta_s3(s_dec["s3"], qemb), self.g3, self.d3)
            f3 = self.pred_fuse3(torch.cat([f3, d3.to(f3.dtype)], dim=1))
        if self.mup_prior:
            if self.pred_frozen_input:      # backbone trainable -> read the FROZEN EMA stream,
                with torch.no_grad():       # else the +0.163 field silently drifts off-distribution
                    t_e = self.art.target_encoder(vol, window_id=window_id, mask_s1=None,
                                                  ct_type_id=ct_type_id)
                    mimg = t_e["e4"] if self.mup_prior == "s4" else self.art.target_decoder(t_e)[self.mup_prior]
            else:
                mimg = s_enc["e4"] if self.mup_prior == "s4" else s_dec[self.mup_prior]
            fld = self._mup_field(mimg)                                   # [B,N,cached] FROZEN
            qn = F.normalize(qemb.float(), dim=-1)
            sc = torch.einsum("bnd,bd->bn", F.normalize(fld.float(), dim=-1), qn)   # [B,N]
            sc = sc - sc.amin(1, keepdim=True)
            sc = sc / sc.amax(1, keepdim=True).clamp(min=1e-6)            # per-sample [0,1]
            mg = tuple(getattr(self.art.config.patch, f"grid_size_{self.mup_prior}"))
            mp = sc.reshape(sc.shape[0], 1, *mg)
            self._mup_map = mp                                            # native grid, for diagnostics
            if mg != self.g3:
                mp = F.interpolate(mp, size=self.g3, mode="trilinear", align_corners=False)
            mp = mp - mp.mean(dim=(2, 3, 4), keepdim=True)        # centre -> gate is +/- around 1
            f3 = f3 * (1 + self.mup_gamma * mp.to(f3.dtype))
        x = self.bottleneck(f3)
        if self.film: x = self.film3(x, qemb)
        if self.fusion == "xattn":
            x = self._fuse(x, self.fuse3, qemb)
        x = self.up1(x, f2)                                      # 28^3
        if self.film: x = self.film2(x, qemb)
        if self.predictor_prior:                                 # inject the FROZEN predictor's CFG residual
            if self.pred_frozen_input:                           # backbone trainable -> use the EMA stream
                with torch.no_grad():
                    t_enc = self.art.target_encoder(vol, window_id=window_id, mask_s1=None,
                                                    ct_type_id=ct_type_id)
                    src2 = self.art.target_decoder(t_enc)["s2"]
            else:
                src2 = s_dec["s2"]
            delta = self._to3d(self._predictor_delta(src2, qemb), self.g2, self.d2)
            x = self.pred_fuse(torch.cat([x, delta.to(x.dtype)], dim=1))
        if self.fusion in ("xattn", "s2only"):
            x = self._fuse(x, self.fuse2, qemb)                  # s2 fuse = the prior (pretrained-init) mechanism
        aux28 = self.aux28(x).squeeze(1) if self.deep_sup else None
        x = self.up2(x, f1)                                      # 56^3
        if self.film: x = self.film1(x, qemb)
        if self.fusion == "xattn":
            x = self._fuse(x, self.fuse1, qemb)
        aux56 = self.aux56(x).squeeze(1) if self.deep_sup else None
        st0 = self.stem0(vol); st1 = self.stem1(st0)
        x = self.up3(x, st1)                                     # 112^3
        if self.film: x = self.film0(x, qemb)
        if self.sep_up4:
            x = self.up4(x, st0)                                 # real convs at full resolution
        elif self.hires_head:
            # trilinear lift only -- no convolution above res/2
            x = F.interpolate(x, size=(self._res,) * 3, mode="trilinear", align_corners=False)
        elif self.out_res >= self._res:
            x = self.up4(x, st0)                                 # 224^3
        logit = self.out(x).squeeze(1)
        if self.txt_head_on:
            tv = self.txt_head(qemb.to(x.dtype))                       # [B,cs] phrase -> voxel space
            tv = F.normalize(tv, dim=-1)
            xn = F.normalize(x, dim=1)
            sim = torch.einsum("bcdhw,bc->bdhw", xn, tv)
            self._txt_sim = sim                       # raw cosine map, for attn_kl_loss
            logit = logit + self.txt_w * sim
        if self.mup_prior and self.mup_logit and getattr(self, "_mup_map", None) is not None:
            # centre so the prior shifts the logit +/- rather than adding a constant bias, then
            # upsample the native-grid field (7^3 at s4) to the output resolution.
            pm = self._mup_map - self._mup_map.mean(dim=(2, 3, 4), keepdim=True)
            pm = F.interpolate(pm, size=logit.shape[-3:], mode="trilinear", align_corners=False)
            logit = logit + self.mup_logit_w * pm[:, 0].to(logit.dtype)
        if self.deep_sup and self.training:
            return logit, [(aux56, 0.5), (aux28, 0.25)]         # (aux, weight) at 56^3 and 28^3
        return logit


