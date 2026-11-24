"""Shared trainer utilities for Piku OCR: perf stack, budget, checkpointing."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch


def max_perf(device: str) -> None:
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def seconds_left(deadline: float) -> float:
    return deadline - time.time()


def save_checkpoint(path: Path, model, opt, scaler, step: int, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: Path, model, opt=None, scaler=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["opt"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    return ck["step"], ck.get("extra", {})


def git_ready() -> None:
    subprocess.run([sys.executable, "-c", "import torch"], check=True)
