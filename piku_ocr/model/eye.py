"""Piku-Eye: line-image reader. New architecture for Piku OCR v1.

Line crops (3x64x512) -> depthwise stem -> height collapse -> flat token row
-> narrow-band transformer: full self-attention restricted to a sliding width
window ("lens"), so long Bangla conjunct clusters get tall receptive fields
with short-range cost -> CTC head over the compact 3-language charset.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..charset import PikuCharset


class PikuStem(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_conv = nn.Conv2d(3, dim, 5, stride=2, padding=2)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(dim // 16, dim)
        self.grn = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.gelu(self.in_conv(x))
        y = self.dw(x)
        y = self.pw(y)
        g = y.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(1e-6).rsqrt()
        y = y * g * (1 + self.grn.view(1, -1, 1, 1)) + x
        return self.norm(y)


class BandAttention(nn.Module):
    def __init__(self, dim: int, heads: int, band: int = 33):
        super().__init__()
        self.heads = heads
        self.band = band
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.relu(self.qkv(x)).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        half = self.band // 2
        pad = F.pad(k, (0, 0, half, half, 0, 0))
        vpad = F.pad(v, (0, 0, half, half, 0, 0))
        outs = []
        for s in range(-half, half + 1):
            ks = pad[:, :, half + s: half + s + t]
            vs = vpad[:, :, half + s: half + s + t]
            outs.append((q * ks).sum(-1, keepdim=True))
            outs.append(vs)
        attn_a = torch.cat(outs[0::2], dim=-1)
        vals = torch.stack(outs[1::2], dim=0)
        w = torch.softmax(attn_a.permute(3, 0, 1, 2) / math.sqrt(d // self.heads), dim=0)
        out = (w.unsqueeze(-1) * vals).sum(0)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class EyeBlock(nn.Module):
    def __init__(self, dim, heads, dropout, band=33):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = BandAttention(dim, heads, band)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h)
        return x + self.mlp(self.n2(x))


def _sinusoid(t: int, d: int, device, dtype):
    pos = torch.arange(t, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(t, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div).to(dtype)
    pe[:, 1::2] = torch.cos(pos * div).to(dtype)
    return pe


class PikuEye(nn.Module):
    def __init__(self, charset: PikuCharset, cfg):
        super().__init__()
        self.charset = charset
        d = cfg.eye_dim
        self.stem = PikuStem(d)
        self.down = nn.Conv2d(d, d, 3, stride=2, padding=1, groups=d)
        self.blocks = nn.ModuleList(
            [EyeBlock(d, cfg.eye_heads, cfg.eye_dropout) for _ in range(cfg.eye_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, charset.size)

    def tokens(self, px: torch.Tensor) -> torch.Tensor:
        x = self.stem(px)
        x = self.down(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        x = x + _sinusoid(x.shape[1], x.shape[2], x.device, x.dtype)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def logits(self, px: torch.Tensor) -> torch.Tensor:
        return self.head(self.tokens(px))

    def loss(self, px, ids_list: list[list[int]]):
        logit = self.logits(px)
        logp = logit.log_softmax(-1).transpose(0, 1)
        tgt = torch.tensor([i for s in ids_list for i in s], device=px.device, dtype=torch.long)
        tl = torch.tensor([len(s) for s in ids_list], device=px.device, dtype=torch.long)
        il = torch.full((len(ids_list),), logp.shape[0], device=px.device, dtype=torch.long)
        return F.ctc_loss(logp, tgt, il, tl, blank=0, zero_infinity=True)

    @torch.no_grad()
    def decode(self, px: torch.Tensor) -> list[str]:
        ids = self.logits(px).argmax(-1)
        out = []
        for row in ids:
            seq, prev = [], -1
            for i in row.tolist():
                if i != prev:
                    seq.append(i)
                    prev = i
            out.append(self.charset.decode(seq))
        return out
