"""Small autoregressive decoder producing structured JSON over image memory.

Task routing uses reserved prompt tokens (<task_ocr>, <task_describe>,
<task_detect>, <task_form>) instead of a bigger general-purpose LM, matching
the micro-VLM philosophy: perception lives in the encoder + heads; this
decoder only *formats* them into JSON. Its own token vocabulary is the
multi-script grapheme inventory plus JSON/control specials, so one 8k vocab
serves Bangla, English, Russian and 210+ other languages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CraftlyConfig


class HadamardMLP(nn.Module):
    """Gated MLP whose hidden mixing uses a fixed Hadamard transform (cheap, sparse)."""

    def __init__(self, dim: int, mult: float = 3.0):
        super().__init__()
        hidden = int(dim * mult) // 8 * 8 or 8
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        blk = 64
        h = torch.ones(blk, blk)
        while h.shape[0] < blk:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = h / (blk ** 0.5)
        self.register_buffer("H", h, persistent=False)
        self.stride = blk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(x) * torch.sigmoid(self.gate(x))
        n, rem = u.shape[-1] // self.stride, u.shape[-1] % self.stride
        main = u[..., : n * self.stride].unflatten(-1, (n, self.stride))
        main = F.linear(main, self.H) * self.stride ** 0.5
        main = main.flatten(-2)
        return self.down(torch.cat([main, u[..., n * self.stride:]], -1))


class CraftlyDecoder(nn.Module):
    def __init__(self, cfg: CraftlyConfig, enc_dim: int | None = None):
        super().__init__()
        d = cfg.decoder_dim
        self.mem_proj = nn.Linear(enc_dim, d, bias=False) if enc_dim and enc_dim != d else None
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.dec = nn.ModuleList(
            [_Layer(d, cfg) for _ in range(cfg.decoder_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying keeps params down

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """tokens: (B, T) input ids; memory: (B, N, D_enc) image cells."""
        if self.mem_proj is not None:
            memory = self.mem_proj(memory)
        h = self.embed(tokens)
        t = tokens.size(1)
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1
        )
        for layer in self.dec:
            h = layer(h, memory, causal, attn_mask)
        return self.lm_head(self.norm(h))

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )


class _Layer(nn.Module):
    def __init__(self, d: int, cfg: CraftlyConfig):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.decoder_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, cfg.decoder_heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = HadamardMLP(d)

    def forward(self, h, memory, causal, mem_mask):
        x = self.n1(h)
        h = h + self.self_attn(x, x, x, attn_mask=causal, need_weights=False)[0]
        x = self.n2(h)
        h = h + self.cross_attn(x, memory, memory, need_weights=False)[0]
        return h + self.mlp(self.n3(h))
