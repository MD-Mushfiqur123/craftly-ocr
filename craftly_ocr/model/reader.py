"""CTC text reader for detected crops / overlapping windows.

Follows the GOT + Mirage pattern: the shared encoder compresses the page, a
cheap head reads each text region. Real datasets for 200+ languages are
scarce, so this head is designed to be trained almost entirely on the
synthetic engine in `craftly_ocr.data.synthetic` (any text + augmentation =
labelled line image), then fine-tuned on real multi-script benchmarks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CraftlyConfig
from .encoder import HConvGLU


class OCRReader(nn.Module):
    def __init__(self, cfg: CraftlyConfig, vocab_size: int):
        super().__init__()
        d = cfg.reader_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
            nn.Conv2d(48, d, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.Hardswish(),
        )
        self.body = nn.Sequential(*[HConvGLU(d, band=15) for _ in range(cfg.reader_depth)])
        self.pos = nn.Parameter(torch.zeros(1, 128, d))
        nn.init.normal_(self.pos, std=0.02)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """crops: (B, 3, H, W) -> CTC logits (B, T, V). Height is pooled away."""
        x = self.stem(crops)
        x = self.body(x.mean(2).transpose(1, 2))  # pool height -> (B, W', d)
        t = x.size(1)
        if t > self.pos.size(1):
            x = x + F.interpolate(
                self.pos.transpose(1, 2), size=t, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            x = x + self.pos[:, :t]
        return self.head(x)

    def loss(self, logits: torch.Tensor, targets: list[list[int]], target_lens: list[int]) -> torch.Tensor:
        b, t, v = logits.shape
        logp = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        flat: list[int] = []
        for seq in targets:
            flat.extend(seq)
        tgt = torch.tensor(flat, dtype=torch.long)
        tgt_lens = torch.tensor(target_lens, dtype=torch.long)
        in_lens = torch.full((b,), t, dtype=torch.long)
        return F.ctc_loss(
            logp, tgt, in_lens, tgt_lens, blank=0, zero_infinity=True, reduction="mean"
        )

    @torch.no_grad()
    def decode(self, logits: torch.Tensor, charset) -> list[str]:
        ids = logits.argmax(-1).tolist()
        return [charset.decode(row) for row in ids]
