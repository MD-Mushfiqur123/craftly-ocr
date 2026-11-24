"""Piku OCR configuration: fixed settings tuned for one RTX Pro 6000 96GB."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PikuConfig:
    line_height: int = 64
    line_width: int = 512
    eye_dim: int = 256
    eye_layers: int = 8
    eye_heads: int = 8
    eye_dropout: float = 0.1
    brain_dim: int = 256
    brain_layers: int = 4
    brain_heads: int = 8
    brain_max_len: int = 160

    max_perf = {
        "batch": 1024,
        "bf16": True,
        "tf32": True,
        "compile": True,
        "fused_adamw": True,
        "bucket": True,
        "workers": 8,
        "minutes_per_day": 120,
    }
