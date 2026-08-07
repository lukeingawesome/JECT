"""Lightweight report/utterance text utilities for ART-JEPA.

This pilot intentionally avoids a hard dependency on HuggingFace or Chest2Vec.
The HashingTextEncoder is trainable and stable for code testing. For paper-grade
training, replace it with a radiology language encoder and keep the same output
interface: text_emb [B, text_dim].
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]+")


class HashingTextTokenizer:
    def __init__(self, vocab_size: int = 32768, max_tokens: int = 96):
        self.vocab_size = int(vocab_size)
        self.max_tokens = int(max_tokens)
        self.pad_id = 0
        self.unk_id = 1

    def _hash_token(self, token: str) -> int:
        # Reserve 0 pad and 1 unk.
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return 2 + (int.from_bytes(h, byteorder="little") % max(1, self.vocab_size - 2))

    def encode(self, text: str | None) -> list[int]:
        if text is None:
            text = ""
        text = str(text).lower()
        toks = _TOKEN_RE.findall(text)
        ids = [self._hash_token(t) for t in toks[: self.max_tokens]]
        if not ids:
            ids = [self.unk_id]
        return ids

    def batch_encode(self, texts: Sequence[str | None], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        T = min(self.max_tokens, max(len(x) for x in encoded))
        token_ids = torch.zeros(len(encoded), T, dtype=torch.long)
        attn_mask = torch.zeros(len(encoded), T, dtype=torch.bool)
        for i, ids in enumerate(encoded):
            ids = ids[:T]
            token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :len(ids)] = True
        if device is not None:
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
        return token_ids, attn_mask


class HashingTextEncoder(nn.Module):
    """Small trainable text encoder based on hashed token embeddings."""

    def __init__(self, vocab_size: int, text_dim: int, hidden_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.text_dim = text_dim
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(token_ids)
        if attention_mask is None:
            attention_mask = token_ids.ne(0)
        weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.net(pooled)
