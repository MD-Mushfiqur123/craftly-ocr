"""Shared trainer utilities for Piku OCR: perf stack, budget, checkpointing."""

from __future__ import annotations

import os
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


def auto_vram_batch(run_step, device: str, base: int = 1024, cap: int = 16384, frac: float = 0.85) -> tuple[int, float]:
    """Probe VRAM by doubling batch until ~frac of total memory is used.
    run_step(batch_size) must run one tiny fwd+bwd on the given batch.
    Returns (chosen_batch, peak_bytes). CPU returns (base, 0)."""
    if device != "cuda":
        return base, 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    target = total * frac
    b = base
    last_peak = 0.0
    while True:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            run_step(b)
            last_peak = torch.cuda.max_memory_allocated()
            if b >= cap or last_peak * 1.6 >= target:
                scale = min(1.6, max(1.0, target / max(last_peak, 1.0)))
                chosen = min(cap, max(base, int(b * scale) // 512 * 512))
                torch.cuda.empty_cache()
                return chosen, last_peak
            b = min(cap, b * 2)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return max(base, int(b / 1.8) // 512 * 512), last_peak


def hf_upload(files: dict, repo_id: str, token: str, msg: str, private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    for name, path in files.items():
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo_id, commit_message=msg)
    return f"https://huggingface.co/{repo_id}"


def maybe_hf_upload(args, files: dict, msg: str, model_name: str, step: int, summary: dict) -> str:
    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    user = args.hf_user or os.environ.get("HF_USER", "")
    if not token or not user:
        print("(HF upload skipped — set HF_TOKEN and HF_USER env or --hf-token/--hf-user)")
        return ""
    import json
    import tempfile

    meta = {"model": model_name, "global_steps": step, **summary,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(meta, f, indent=2)
        meta_path = f.name
    payload = dict(files)
    payload[f"logs/{model_name}_progress.json"] = meta_path
    try:
        url = hf_upload(payload, f"{user}/piku-ocr", token, msg, args.hf_private)
        print(f"☁️ uploaded {model_name} progress to HF: {url}")
    except Exception as e:
        print(f"❌ HF upload failed: {type(e).__name__}: {e}")
    finally:
        os.unlink(meta_path)
    return ""
