"""Piku-Brain: tiny char-level correction transformer for Piku OCR v1.

Denoising setup: full-attention transformer, input = corrupted char ids
(reader-style errors), target = clean char ids at the same position.
No causality needed -> simpler, faster, and accurate with 4 small layers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..charset import PikuCharset


def corrupt(text: str, rng: torch.Generator, cer: float = 0.10) -> str:
    if not text:
        return text
    out = []
    for ch in text:
        p = float(torch.rand(1, generator=rng))
        if p < cer * 0.30:
            continue
        if p < cer * 0.55:
            out.append(ch)
            out.append(ch)
        elif p < cer * 0.80:
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out) or text


class _Layer(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PikuBrain(nn.Module):
    def __init__(self, charset: PikuCharset, cfg):
        super().__init__()
        d = cfg.brain_dim
        self.charset = charset
        self.cfg = cfg
        self.emb = nn.Embedding(charset.size + 1, d)
        self.pos = nn.Embedding(cfg.brain_max_len, d)
        self.layers = nn.ModuleList([_Layer(d, cfg.brain_heads) for _ in range(cfg.brain_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, charset.size + 1)

    def forward(self, noisy_ids: torch.Tensor) -> torch.Tensor:
        b, t = noisy_ids.shape
        x = self.emb(noisy_ids) + self.pos(torch.arange(t, device=noisy_ids.device))[None]
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    def loss(self, noisy_ids: torch.Tensor, clean_ids: list[list[int]]):
        t = noisy_ids.shape[1]
        clean = torch.full((noisy_ids.shape[0], t), -100, device=noisy_ids.device, dtype=torch.long)
        for r, seq in enumerate(clean_ids):
            seq = seq[:t]
            clean[r, : len(seq)] = torch.tensor(seq, device=noisy_ids.device, dtype=torch.long)
        logit = self.forward(noisy_ids)
        return F.cross_entropy(
            logit.reshape(-1, logit.shape[-1]), clean.reshape(-1), ignore_index=-100
        )

    @torch.no_grad()
    def correct(self, noisy_ids: torch.Tensor) -> list[str]:
        logit = self.forward(noisy_ids).argmax(-1)
        return [self.charset.decode(row.tolist()) for row in logit]
