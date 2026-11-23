"""Dense text/object detection head over stride-8 encoder features.

DBNet-style: predicts a textness heatmap plus box offsets; objects reuse the
same head with a second channel group so one fine feature map serves both.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.encoder_dims[2]
        self.conv = nn.Sequential(
            nn.Conv2d(d, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(),
            nn.Conv2d(96, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
        )
        self.text_heat = nn.Conv2d(48, 1, 1)
        self.obj_heat = nn.Conv2d(48, 1, 1)
        self.box_offsets = nn.Conv2d(48, 4, 1)   # (dl, dt, dr, db) in cells

    def forward(self, fine: torch.Tensor):
        z = self.conv(fine)
        return {
            "text_heat": torch.sigmoid(self.text_heat(z)),
            "obj_heat": torch.sigmoid(self.obj_heat(z)),
            "box_offsets": self.box_offsets(z),
        }

    def loss(self, out: dict, heat_target: torch.Tensor) -> torch.Tensor:
        """heat_target: (B, 1, H, W) positive map with distance-based weights."""
        pred = out["text_heat"]
        pos = (heat_target > 0.5).float()
        neg = 1.0 - pos
        n_pos = pos.sum().clamp(min=1.0)
        wce = -(
            pos * heat_target * pred.clamp_min(1e-4).log()
            + neg * (1 - heat_target) * (1 - pred).clamp_min(1e-4).log()
        ).sum() / n_pos
        dice = 1.0 - (2 * (pred * pos).sum() + 1) / (pred.sum() + pos.sum() + 1)
        return wce + dice
