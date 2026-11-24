"""Two-model inference for Piku OCR: Eye reads the line, Brain cleans it."""

from __future__ import annotations

import torch

from .charset import build_charset
from .config import PikuConfig
from .model.brain import PikuBrain
from .model.eye import PikuEye


def build_default(device: str | None = None, eye_ckpt: str | None = None, brain_ckpt: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, charset = PikuConfig(), build_charset()
    eye, brain = PikuEye(charset, cfg).to(device).eval(), PikuBrain(charset, cfg).to(device).eval()
    if eye_ckpt:
        ck = torch.load(eye_ckpt, map_location="cpu", weights_only=False)
        eye.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    if brain_ckpt:
        ck2 = torch.load(brain_ckpt, map_location="cpu", weights_only=False)
        brain.load_state_dict(ck2["model"] if isinstance(ck2, dict) and "model" in ck2 else ck2)
    return eye, brain, charset, device


@torch.no_grad()
def read_lines(images: torch.Tensor, eye_ckpt: str | None = None, brain_ckpt: str | None = None,
               apply_brain: bool = True) -> list[str]:
    """images: (B,3,64,512) float 0-1. Returns cleaned text lines."""
    eye, brain, charset, device = build_default(None, eye_ckpt, brain_ckpt)
    images = images.to(device)
    lines = eye.decode(images)
    if apply_brain:
        ids = [charset.encode(t)[: brain.cfg.brain_max_len] for t in lines]
        t = max(1, max(len(i) for i in ids))
        noisy = torch.full((len(ids), t), 1, dtype=torch.long)
        for r, s in enumerate(ids):
            noisy[r, : len(s)] = torch.tensor(s, device="cpu", dtype=torch.long)
        fixed = brain.correct(noisy.to(device))
        return fixed
    return lines
