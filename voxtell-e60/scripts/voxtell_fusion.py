#!/usr/bin/env python
"""v13: LAVT-lineage encoder fusion for VoxTell, as an ablation PAIR.

The stock model's encoder is language-blind: `skips = self.encoder(img)` runs outside the
prompt loop, and text enters exactly once, as `tgt` against the 12^3 bottleneck. Whatever
reaches the decoder is one global vector per prompt, so "this voxel goes with that word"
has no representation anywhere in the network.

PWAM (Pixel-Word Attention Module, LAVT, arXiv:2112.02244) inverts the attention: voxels
are queries, the sentence's tokens are keys, so the output is shaped like the feature map
and varies per position. It is inserted at encoder stages 2 (48^3) and 3 (24^3).

THE PAIR. Both arms build the same modules with the same parameters and run the same code
path; one boolean decides whether the fused tensor re-enters the encoder trunk:

    fused = x + gate(pwam(x, tokens))
    x     = fused if fuse_into_trunk else x      # <- arm E vs arm S
    skips.append(fused)

  * arm E (`fuse_into_trunk=True`)  -- conditioning PROPAGATES: stages 3,4,5 and therefore
    the prompt decoder's own memory all become text-dependent, with no extra module.
  * arm S (`fuse_into_trunk=False`) -- conditioning is DELIVERED to the decoder and cut
    there; the trunk past the fusion point is numerically identical to arm B.

Arm S could skip the per-prompt encoder passes entirely, since its trunk does not depend on
the text. It does not. Running both arms folded keeps parameters, FLOPs and code path
identical so the only free variable is the line above -- the same discipline as arm C's
`--random-init changes exactly one line`.

TWO THINGS THAT WOULD SILENTLY BREAK THE COMPARISON, AND WHAT STOPS THEM

1. Folding the prompt axis into the batch is not free. The stock prompt decoder gives the N
   prompts SELF-attention over each other, and naive folding (one prompt per batch row)
   deletes that interaction -- the model would differ from arm B even with the gates at
   zero, and "v13 vs B30" would carry a second, unstated change. Instead the N per-prompt
   memories are concatenated along the sequence axis and a block-diagonal `memory_mask`
   keeps prompt n attending only to its own block, so tgt self-attention survives intact.
   `self_test` asserts the whole forward reproduces the stock model.

2. The gate is zero-initialised, so at step 0 the network IS arm B, bit for bit. That is
   what keeps the pretrained checkpoint valid and makes any later difference attributable.
   It also means a "does scrambling the text change the output?" preflight FAILS here, and
   correctly so. The mechanism gate must therefore be on the gradient, not the output.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from voxtell.model.voxtell_model import VoxTellModel

FUSE_STAGES = (2, 3)          # 48^3 and 24^3. Stage 4 is the prompt decoder's own memory:
                              # fusing there bundles "make the bottleneck text-aware" into a
                              # run that is meant to test conditioning BEFORE the bottleneck.


class PWAM(nn.Module):
    """Voxel-word cross-attention: every position attends over the sentence's tokens.

    Cost is linear in the number of voxels -- voxels are the query side and the 64 tokens
    are the key side, so positions never attend to each other. `scaled_dot_product_attention`
    additionally never materialises the (V x L) score matrix.
    """

    def __init__(self, c_vis: int, c_txt: int = 2560, dim: int | None = None, heads: int = 8):
        super().__init__()
        dim = dim or c_vis
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads, self.dim = heads, dim
        self.q = nn.Conv3d(c_vis, dim, 1)
        self.kv = nn.Linear(c_txt, 2 * dim)
        self.o = nn.Conv3d(dim, c_vis, 1)
        self.gate = nn.Conv3d(c_vis, c_vis, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, tok: torch.Tensor, tok_mask: torch.Tensor):
        """x (B,C,D,H,W); tok (B,L,c_txt); tok_mask (B,L) bool, True = a real token."""
        b, _, d, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)                       # (B, V, dim)
        k, v = self.kv(tok).chunk(2, dim=-1)                           # (B, L, dim)
        shape = lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        # SDPA's bool convention is the OPPOSITE of nn.MultiheadAttention's: here True MAY
        # attend. Getting it backwards would attend to padding only; self_test catches it.
        att = F.scaled_dot_product_attention(
            shape(q), shape(k), shape(v), attn_mask=tok_mask[:, None, None, :])
        att = att.transpose(1, 2).flatten(2).transpose(1, 2).view(b, self.dim, d, h, w)
        return self.gate(self.o(att))


class FusedVoxTellModel(VoxTellModel):
    """VoxTell plus PWAM at `fuse_stages`. `fuse_into_trunk` is the ablation's free variable.

    The vendored `voxtell` package is NOT modified: the encoder's stage loop
    (`stem` then `for s in self.stages`) is re-walked here so PWAM can sit between stages.
    """

    def __init__(self, *args, fuse_stages=FUSE_STAGES, fuse_into_trunk: bool = True,
                 text_token_dim: int = 2560, pwam_heads: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_stages = tuple(int(s) for s in fuse_stages)
        self.fuse_into_trunk = bool(fuse_into_trunk)
        self.pwam = nn.ModuleDict({
            str(s): PWAM(self.DECODER_CONFIGS[s]["channels"], text_token_dim, heads=pwam_heads)
            for s in self.fuse_stages})
        self._tok = self._tok_mask = None
        self._pooling = "last_token"
        self._paired = False

    # -- token plumbing ---------------------------------------------------------------
    # The official eval goes through VoxTellPredictor, whose sliding window calls
    # `network(img, text_embedding)` and has no slot for token features. So the caller sets
    # them per volume and clears them after. clear_tokens() is the point: without it a
    # missing set_tokens would silently score the PREVIOUS volume's sentences, which no
    # metric would reveal.

    POOLINGS = ("last_token", "mean")

    def set_tokens(self, tok: torch.Tensor, tok_mask: torch.Tensor,
                   pooling: str = "last_token") -> None:
        """tok (B,N,L,c_txt), tok_mask (B,N,L) bool.

        `pooling` is how the POOLED bank was built, and it exists so `_check_pairing` can
        reconstruct the pooled vector from the tokens exactly instead of approximating it.
        Approximating is not good enough: a mean over a last-token bank does not identify
        its own sentence in general, and hardcoding the last token rejected a correct
        mean-pooled bank outright (the gatortron arm, step 0, 0.6328 vs 0.6016). It defaults
        to last_token because that is what every caller written before this argument meant.
        """
        assert tok.ndim == 4 and tok_mask.shape == tok.shape[:3], (tok.shape, tok_mask.shape)
        assert tok_mask.any(-1).all(), "a prompt with zero real tokens would attend to nothing"
        assert pooling in self.POOLINGS, f"unknown pooling {pooling!r}, have {self.POOLINGS}"
        self._tok, self._tok_mask, self._pooling = tok, tok_mask, pooling
        self._paired = False

    def clear_tokens(self) -> None:
        self._tok = self._tok_mask = None
        # _pooling is left as set: it describes the BANK, not the volume, and clearing it
        # to a default would let the next set_tokens inherit a silent last_token assumption.

    def _check_pairing(self, text_embedding: torch.Tensor) -> None:
        """Prompt i's tokens must BE prompt i's sentence, checked against the pooled vector.

        The two banks are looked up by two different code paths -- the trainer indexes the
        token bank itself, the eval sweep lets VoxTellPredictor resolve the pooled one -- so
        a reordering in either would hand prompt i another prompt's words. Nothing about the
        output would look wrong; the dice would simply be a number for a model nobody built.

        The probe vector derived from block i's tokens must match pooled i more closely than
        it matches any other prompt here. Comparing by rank rather than by value keeps this
        immune to the dtype each path happens to use. Duplicate sentences are common (35.3%
        of volumes hold one finding, so the same sentence is drawn twice) and are exempt by
        construction.

        The probe is RECONSTRUCTED from the tokens using the pooling the bank was built with
        (`set_tokens(..., pooling=...)`), not approximated. Both approximations were tried
        and both are wrong: hardcoding the last real token -- correct for qwen3, where the
        pooled vector IS that position -- rejected a correct mean-pooled bank at step 0
        (gatortron, 0.6328 vs 0.6016); and using the masked mean for both fails the qwen3
        case, because the mean of a sentence's tokens does not in general sit closer to that
        sentence's last token than to another's. Reconstructing exactly costs one branch and
        removes the assumption.
        """
        b, n = text_embedding.shape[:2]
        # autocast OFF for the whole check. This is a diagnostic, not model math, and under
        # bfloat16 it cannot do its job: bf16's spacing at 1.0 is 2^-7 = 0.0078, so two
        # DISTINCT sentences at cos 0.9988 round to exactly 1.0 and tie with a prompt's own
        # self-similarity. argmax then returns the lower index, j != i, and the gate reports a
        # crossing on a correctly-paired batch -- which is what killed the gatortron arm at
        # step 76 (sim[1->0] = sim[1->1] = 1.000000000, gap 0, while the two pooled vectors
        # differed by 4.2e-03). It is NOT a gatortron property: 16/6467 of its sentences have
        # a near-twin that collapses in bf16 against 18/6467 of qwen3's, so the qwen3 arms
        # carried the same latent defect and merely never drew the pair. In float32 the
        # relation is exact -- probe IS the pooled vector, so sim[i,i] == 1.0 -- and the check
        # resolves 0.9988 from 1.0 instead of blurring them. Strictly stronger, not looser.
        with torch.autocast(device_type=text_embedding.device.type, enabled=False):
            pooled = F.normalize(text_embedding.squeeze(2).float(), dim=-1)
            if self._pooling == "mean":
                m = self._tok_mask.unsqueeze(-1).float()
                probe = (self._tok.float() * m).sum(-2) / m.sum(-2).clamp(min=1)
            else:                                # last_token: the pooled vector IS this slot
                idx = self._tok_mask.sum(-1) - 1
                probe = self._tok[torch.arange(b)[:, None],
                                  torch.arange(n)[None, :], idx].float()
            sim = F.normalize(probe, dim=-1) @ pooled.transpose(1, 2)          # (b, n, n)
        for bi in range(b):
            for i in range(n):
                j = int(sim[bi, i].argmax())
                if j == i or torch.equal(pooled[bi, j], pooled[bi, i]):
                    continue
                # The message carries the EVIDENCE, not just the verdict. A bare "(1.0000 vs
                # 1.0000)" is unactionable: it cannot distinguish a duplicate-sentence tie the
                # exemption should have caught from a real reordering, and the two need
                # opposite fixes. Full precision plus the raw vector delta separates them.
                d = (pooled[bi, j] - pooled[bi, i]).abs().max().item()
                raise AssertionError(
                    f"token/pooled pairing is crossed: sample {bi} prompt {i}'s tokens match "
                    f"prompt {j}'s sentence better than its own "
                    f"({sim[bi, i, j]:.4f} vs {sim[bi, i, i]:.4f}). One of the two bank "
                    "lookups is ordered differently from the other.\n"
                    f"  sim[{i}->{j}] = {sim[bi, i, j].item():.9f}   "
                    f"sim[{i}->{i}] = {sim[bi, i, i].item():.9f}   "
                    f"gap = {(sim[bi, i, j] - sim[bi, i, i]).item():.3e}\n"
                    f"  max |pooled[{j}] - pooled[{i}]| = {d:.3e}  "
                    f"(equal={torch.equal(pooled[bi, j], pooled[bi, i])}, "
                    f"allclose={torch.allclose(pooled[bi, j], pooled[bi, i])})\n"
                    f"  finite: pooled {bool(torch.isfinite(pooled[bi]).all())}, "
                    f"probe {bool(torch.isfinite(probe[bi]).all())}, "
                    f"tok {bool(torch.isfinite(self._tok[bi]).all())}; "
                    f"lengths {self._tok_mask[bi].sum(-1).tolist()}; pooling {self._pooling!r}")
        self._paired = True

    # -- forward ----------------------------------------------------------------------

    def _encode(self, img, tok, tok_mask):
        enc = self.encoder
        x = enc.stem(img) if enc.stem is not None else img
        skips = []
        for i, stage in enumerate(enc.stages):
            x = stage(x)
            if str(i) in self.pwam:
                fused = x + self.pwam[str(i)](x, tok, tok_mask)
                skips.append(fused)
                if self.fuse_into_trunk:
                    x = fused
            else:
                skips.append(x)
        return skips

    def forward(self, img: torch.Tensor, text_embedding: torch.Tensor = None):
        b, n = text_embedding.shape[:2]
        assert self._tok is not None, (
            "set_tokens() was not called for this volume. Refusing to run rather than "
            "silently reusing the previous volume's sentences.")
        assert self._tok.shape[:2] == (b, n), (
            f"tokens are for {tuple(self._tok.shape[:2])} but text_embedding is for {(b, n)}")
        if not self._paired:                 # once per set_tokens, not once per sliding tile
            self._check_pairing(text_embedding)

        # Prompts fold into the batch so the encoder can be text-conditioned per prompt.
        img_f = img.repeat_interleave(n, dim=0)
        tok_f = self._tok.flatten(0, 1)
        mask_f = self._tok_mask.flatten(0, 1)
        skips = self._encode(img_f, tok_f, mask_f)                     # each (b*n, C, ...)

        sel = self.project_bottleneck_embed(
            rearrange(skips[self.selected_decoder_layer], 'x c d h w -> x h w d c'))
        mem = rearrange(sel, 'x h w d c -> (h w d) x c')               # (V, b*n, q)
        v, q = mem.shape[0], mem.shape[-1]
        # (V, b, n, q) -> (n, V, b, q) -> (n*V, b, q): prompt n's memory is block n.
        mem = mem.view(v, b, n, q).permute(2, 0, 1, 3).reshape(n * v, b, q)
        pos = self.pos_embed.repeat(n, 1, 1)

        text_embed = self.project_text_embed(
            repeat(text_embedding.squeeze(2), 'b n dim -> n b dim'))   # (n, b, q)
        # nn.MultiheadAttention: True == NOT allowed to attend. Block-diagonal, so prompt n
        # sees only its own memory and the tgt self-attention among prompts is untouched.
        blocks = torch.arange(n, device=mem.device).repeat_interleave(v)
        memory_mask = blocks[None, :] != torch.arange(n, device=mem.device)[:, None]

        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed, memory=mem, pos=pos, memory_mask=memory_mask,
            memory_key_padding_mask=None)
        mask_embedding = repeat(mask_embedding, 'n x dim -> x n dim')  # (b, n, q)

        prompt_embeds = [p(mask_embedding).flatten(0, 1).unsqueeze(1)
                         for p in self.project_to_decoder_channels]    # each (b*n, 1, C)
        outs = self.decoder(skips, prompt_embeds)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        outs = [o.view(b, n, *o.shape[2:]) for o in outs]
        return outs if self.deep_supervision else outs[0]


def fusion_param_names(net: nn.Module) -> set[str]:
    """The keys a pretrained VoxTell checkpoint cannot supply. G-LOAD-KEYS compares against
    exactly this set, which is what replaces the strict=True load arm B/C rely on."""
    return {f"pwam.{k}" for k, _ in net.pwam.named_parameters()} | \
           {f"pwam.{k}" for k, _ in net.pwam.named_buffers()}


# ---------------------------------------------------------------------------- self test

def self_test(device: str = "cuda:0") -> None:
    """Three properties, on the real 440M model and the real checkpoint.

    1. zero-init gates => the fused forward reproduces the STOCK forward. This is the whole
       identity claim, and it simultaneously proves the block-diagonal memory_mask, the
       fold/unfold index arithmetic and the SDPA mask polarity, because getting any of them
       wrong moves the logits.
    2. arm E and arm S agree at step 0 and have identical parameter counts.
    3. the PWAM parameters receive non-zero gradients on the first backward -- the mechanism
       IS in the training path even though its output is currently zero.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_voxtell as T

    torch.manual_seed(0)
    stock, _ = T.build_network(random_init=False, deep_supervision=False)
    stock = stock.to(device).eval()

    def fused(into_trunk):
        torch.manual_seed(0)
        net, _ = T.build_network(random_init=False, deep_supervision=False,
                                 fuse_into_trunk=into_trunk)
        return net.to(device).eval()

    e, s = fused(True), fused(False)
    n_e = sum(p.numel() for p in e.parameters())
    n_s = sum(p.numel() for p in s.parameters())
    assert n_e == n_s, f"arm parity broken: E {n_e} vs S {n_s}"
    print(f"  params: stock {sum(p.numel() for p in stock.parameters())/1e6:.2f}M  "
          f"E {n_e/1e6:.2f}M  S {n_s/1e6:.2f}M  (+{(n_e - sum(p.numel() for p in stock.parameters()))/1e6:.2f}M)")

    b, n, L, real = 1, 2, 64, 17
    img = torch.randn(b, 1, *T.PATCH, device=device)
    tok = torch.randn(b, n, L, T.EMB_DIM, device=device)
    tok_mask = torch.zeros(b, n, L, dtype=torch.bool, device=device)
    tok_mask[..., :real] = True                     # median real length; the rest is padding
    # The fixture reproduces a LAST-TOKEN bank's relation (pooled IS the last real token),
    # so `pooling="last_token"` is passed explicitly at every set_tokens below rather than
    # inherited from the default -- otherwise the pairing gate here would be testing nothing,
    # and a later change to that default would quietly turn this self-test into a no-op.
    # .clone() because basic indexing returns a VIEW: without it a fixture that perturbs tok
    # would perturb txt in lockstep and stay consistent.
    txt = tok[:, :, real - 1:real].clone()

    def run(net, t, m):
        net.set_tokens(t, m, pooling="last_token")
        try:
            return net(img, txt)
        finally:
            net.clear_tokens()

    with torch.inference_mode():
        ref = stock(img, txt)
        # The threshold is measured, not guessed. Running the STOCK model at a different
        # batch shape already moves its logits -- different kernels, different reduction
        # order -- and no re-implementation can agree more closely than that. Measured on
        # this checkpoint: 3.6e-2 absolute on logits whose |max| is 177, i.e. 2e-4 relative.
        floor = (ref - stock(img.repeat(2, 1, 1, 1, 1), txt.repeat(2, 1, 1, 1))[:1]) \
            .abs().max().item()
        outs = {name: run(net, tok, tok_mask) for name, net in (("E", e), ("S", s))}

    scale = ref.abs().max().item()
    print(f"  logits |max| = {scale:.1f} ; stock-vs-stock batch-shape floor = {floor:.3e}")
    for name, out in outs.items():
        assert out.shape == ref.shape, f"{name} {out.shape} != stock {ref.shape}"
        d = (out - ref).abs().max().item()
        print(f"  arm {name} vs stock, max |dlogit| = {d:.3e}  (rel {d / scale:.2e})")
        assert d <= max(3 * floor, 1e-6), (
            f"arm {name} is NOT the stock model at zero-init: {d:.3e} against a "
            f"batch-shape floor of {floor:.3e}. The gates are zero, so this is a "
            "fold/mask/index bug, not a modelling difference.")
    assert (outs["E"] - outs["S"]).abs().max().item() == 0.0, "E and S differ at step 0"

    # 4. Per-prompt isolation, WITH THE GATES OPENED.
    # The identity test above cannot see a wrong memory_mask: at zero-init every prompt's
    # memory block is a copy of every other, so attending to the wrong block returns the
    # right answer. Opening the gates makes the blocks differ, and then prompt 0's output
    # must not move when only prompt 1's tokens change. This catches an inverted mask
    # polarity, a leaked block, and a fold index that pairs a prompt with another's tokens.
    torch.manual_seed(1)
    saved = copy.deepcopy(e.pwam.state_dict())
    for p in e.pwam.parameters():
        nn.init.normal_(p, std=0.02)
    other = tok.clone()
    other[:, 1] = torch.randn_like(other[:, 1])

    def run_unpaired(net, t, m):
        """Only the TOKENS change here, not the pooled vector -- changing the pooled one too
        would move prompt 0 legitimately, through the prompt decoder's tgt self-attention,
        and the isolation test would be measuring that instead. The pairing gate is bypassed
        because this fixture is deliberately crossed."""
        net.set_tokens(t, m, pooling="last_token")
        net._paired = True
        try:
            return net(img, txt)
        finally:
            net.clear_tokens()

    with torch.inference_mode():
        a, bb = run_unpaired(e, tok, tok_mask), run_unpaired(e, other, tok_mask)
    leak = (a[:, 0] - bb[:, 0]).abs().max().item()
    moved = (a[:, 1] - bb[:, 1]).abs().max().item()
    print(f"  gates opened: prompt-1 tokens changed -> prompt 0 moves {leak:.3e}, "
          f"prompt 1 moves {moved:.3e}")
    assert leak == 0.0, (
        f"prompt 0 moved by {leak:.3e} when only prompt 1's tokens changed -- the "
        "block-diagonal memory_mask is leaking, or the fold pairs the wrong tokens.")
    assert moved > 1e-3 * scale, (
        f"prompt 1 barely moved ({moved:.3e}) when its own tokens were replaced -- the "
        "tokens are not reaching the output, so PWAM is inert.")
    e.pwam.load_state_dict(saved)                      # back to the real init, gates at zero

    # 6. The pairing gate must FIRE on the case it was written for. A detector that has
    # never been shown a crossed pairing is not a validated detector.
    with torch.inference_mode():
        try:
            run(e, tok.flip(1), tok_mask)
        except AssertionError as exc:
            assert "pairing is crossed" in str(exc), f"wrong assertion fired: {exc}"
            print("  pairing gate fires when prompt 0 and 1's token blocks are swapped")
        else:
            raise AssertionError(
                "the token/pooled pairing gate did NOT fire on swapped token blocks")

    # 5. G-FUSE-GRAD: the mechanism is in the training path, and its bootstrap is stated
    # rather than discovered. At zero-init ONLY the gate can receive gradient: everything
    # upstream of it is multiplied by a zero weight on the way to the loss. That is the
    # ControlNet/LoRA bootstrap, not a dead module -- one step makes the gate non-zero and
    # the rest starts learning. A gate written as "every PWAM parameter must have gradient"
    # would fail here for a correct model.
    e.train()
    e.set_tokens(tok, tok_mask, pooling="last_token")
    e(img, txt).float().pow(2).mean().backward()
    e.clear_tokens()
    gate = [p for k, p in e.pwam.named_parameters() if ".gate." in k]
    upstream = [(k, p) for k, p in e.pwam.named_parameters() if ".gate." not in k]
    assert all(p.grad is not None and p.grad.any() for p in gate), \
        "the gate receives no gradient: PWAM is not in the training path at all"
    live = [k for k, p in upstream if p.grad is not None and p.grad.any()]
    assert not live, f"upstream weights have gradient at zero-init, so the gate is not zero: {live[:4]}"
    print(f"  step 0: gate |g|max = {max(p.grad.abs().max().item() for p in gate):.3e}, "
          f"upstream zero as the bootstrap predicts")

    with torch.no_grad():                              # what the first optimiser step does
        for p in gate:
            p.add_(0.01 * torch.randn_like(p))
    e.zero_grad(set_to_none=True)
    e.set_tokens(tok, tok_mask, pooling="last_token")
    e(img, txt).float().pow(2).mean().backward()
    e.clear_tokens()
    dead = [k for k, p in upstream if p.grad is None or not p.grad.any()]
    assert not dead, f"after the gate opens these stay inert: {dead[:6]}"
    print("  after the gate opens, every PWAM parameter receives gradient")
    print("self_test ok")


if __name__ == "__main__":
    self_test(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
