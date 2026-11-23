"""Central configuration for Craftly OCR.

All sizes are tuned for on-device deployment (< 50 MB at 4-bit, ~120M params).
Scaling knobs live here so stage trainers do not hardcode shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CraftlyConfig:
    # ---- image handling -------------------------------------------------
    image_size: int = 512            # base side length; dynamic resolution multiplies by 1.5x
    patch_stride: int = 32           # 5 stem stages /2 each -> 16x16 = 256 image cells
    dynamic_resolution: bool = True  # long side up to 1.5x, cell grid grows accordingly

    # ---- shared vision encoder (HConvGLU, Mirage-style) -----------------
    encoder_dims: tuple = (64, 128, 192, 320, 384)
    encoder_depths: tuple = (2, 3, 5, 6, 2)
    hadamard_band: int = 33          # banded conv width inside HConvGLU
    num_registers: int = 4           # working-memory tokens for attention blocks
    attn_blocks: int = 12
    attn_heads: int = 8
    attn_kv_heads: int = 2           # grouped-query attention

    # ---- OCR reader (CTC over crops) ------------------------------------
    reader_dim: int = 320
    reader_depth: int = 4
    reader_window: int = 512         # px window for overlapping crops
    reader_beam: int = 1             # greedy CTC default; beam may raise accuracy

    # ---- text detection head -------------------------------------------
    det_stride: int = 8              # dense head reads fine stem features
    det_max_lines: int = 64

    # ---- decoder (structured JSON over image memory) --------------------
    decoder_dim: int = 512
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_kv_heads: int = 2
    decoder_mhc_lanes: int = 4       # multi-hyper-candidate residual lanes
    vocab_size: int = 8192           # grapheme + special tokens (multi-script)
    max_out_tokens: int = 512

    # ---- tasks ----------------------------------------------------------
    tasks: tuple = ("ocr", "detect", "describe", "form")

    # ---- training -------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    batch_size: int = 64
    accumulation_steps: int = 2
    bf16: bool = True

    # extra metadata carried through checkpoints
    extras: dict = field(default_factory=dict)


def default_config() -> CraftlyConfig:
    return CraftlyConfig()
