"""Shared vision encoder: HConvGLU body + gated attention blocks.

Channel mixing follows the Mirage recipe: fixed Walsh–Hadamard transforms
surround small learned *banded* convolutions with a gate, so all channels
interact while parameter count stays ~8x smaller than a dense matrix.
Global Response Normalisation balances channel responses.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def hadamard_matrix(n: int) -> torch.Tensor:
    """Walsh–Hadamard (Sylvester) matrix, normalised orthonormal; H is its own inverse."""
    assert n & (n - 1) == 0, "n must be a power of two"
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h / math.sqrt(n)


class BandedMix(nn.Module):
    """y = H^T @ (grad-step-free banded(u * g)) where u = H @ x.

    The learned part is d x band instead of d x d. For d=512, band=65 the
    weight matrix has 512x65 entries -- about 8x fewer than dense.
    """

    def __init__(self, dim: int, band: int = 33):
        super().__init__()
        assert band % 2 == 1
        self.dim = dim
        self.band = band
        # pad dim to nearest power of two for the transform
        self.padded = 1 << (dim - 1).bit_length()
        self.register_buffer("H", hadamard_matrix(self.padded), persistent=False)
        self.weight = nn.Parameter(torch.zeros(self.padded, band))
        nn.init.normal_(self.weight, std=0.02)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, self.padded - self.dim))

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.dim]

    def bandmul(self, u: torch.Tensor) -> torch.Tensor:
        """Apply the learned banded kernel over the channel axis (..., d)."""
        half = self.band // 2
        padded = F.pad(u, (half, half))
        u_b = padded.unfold(-1, self.band, 1)          # (..., padded, band)
        return (u_b * self.weight).sum(-1)             # (..., padded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (..., d)
        hp = self._pad(x)
        u = F.linear(hp, self.H)          # fixed transform: H @ x^T
        v = self.bandmul(u)
        return self._trim(F.linear(v, self.H))  # H^-1 = H (orthonormal)


class HConvGLU(nn.Module):
    """Gated banded mixing with residual + Global Response Normalisation (ConvNeXtV2-style)."""

    def __init__(self, dim: int, band: int = 33):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mix_a = BandedMix(dim, band)
        self.mix_b = BandedMix(dim, band)
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate = torch.sigmoid(self.mix_b(h))
        y = self.mix_a(h) * gate + x
        g = (y.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return y / g * (self.beta + g)


def _downsample(dim_in: int, dim_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(dim_out),
    )


class VisionEncoder(nn.Module):
    """Hierarchical conv stem -> HConvGLU body -> attention blocks with registers.

    Returns (memory, fine): `memory` has (B, N+R, D) rows for decoder /
    detection cross-attention; `fine` preserves stride-8 detail for the
    detection and mask heads without running the whole body at that
    resolution.
    """

    def __init__(self, cfg):
        super().__init__()
        dims, depths = cfg.encoder_dims, cfg.encoder_depths
        band = cfg.hadamard_band
        self.in_proj = nn.Conv2d(3, dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.stages = nn.ModuleList()
        for i, d in enumerate(depths):
            self.stages.append(
                nn.Sequential(*[HConvGLU(dims[i], band) for _ in range(d)])
            )
            if i < len(depths) - 1:
                self.stages.append(_downsample(dims[i], dims[i + 1]))
        self.per_dim = nn.Identity()
        self.fine_dim = dims[2]  # stage-3 output for dense heads (stride 8)

        d = dims[-1]
        self.attn_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(
            [
                GQABlock(d, cfg.attn_heads, cfg.attn_kv_heads)
                for _ in range(cfg.attn_blocks)
            ]
        )
        self.registers = nn.Parameter(torch.zeros(1, cfg.num_registers, d))
        nn.init.normal_(self.registers, std=0.02)
        self.out_norm = nn.LayerNorm(d)
        self.mem_dim = d

    def forward(self, pixels: torch.Tensor):
        x = self.in_proj(pixels)
        fine = None
        for i, layer in enumerate(self.stages):
            if isinstance(layer, nn.Sequential) and any(isinstance(m, HConvGLU) for m in layer):
                # attention over flattened spatial cells for this stage
                b, c, h, w = x.shape
                y = layer(x.flatten(2).transpose(1, 2))
                x = y.transpose(1, 2).reshape(b, c, h, w)
            else:
                x = layer(x)
            if i == 4:  # after stage-3 blocks => stride 8 map
                fine = x
        b, c, h, w = x.shape
        toks = x.flatten(2).transpose(1, 2)
        regs = self.registers.expand(b, -1, -1)
        toks = torch.cat([regs, toks], dim=1)
        for blk in self.blocks:
            toks = blk(toks)
        return self.out_norm(toks), (fine, (h, w))


class GQABlock(nn.Module):
    """Grouped-query self-attention with a gated (GLU) value projection."""

    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.hd = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, self.kv_heads * self.hd, bias=False)
        self.v = nn.Linear(dim, self.kv_heads * self.hd, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, hd, kh = self.heads, self.hd, self.kv_heads
        q = self.q(x).view(b, n, h, hd)
        k = self.k(x).view(b, n, kh, hd)
        v = self.v(x).view(b, n, kh, hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if kh != h:
            rep = h // kh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, n, d)
        g = torch.sigmoid(self.gate(x))
        return x + self.gamma * self.proj(o * g)
