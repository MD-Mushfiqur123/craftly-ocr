"""Piku-Eye trainer — stage 1. 2h/day resumable CTC training, auto-VRAM filling.

Fixed protocol for one RTX Pro 6000 96GB:
  batch = auto-probed at session start (fills ~85% of available VRAM,
  up to 16384), bf16 autocast, TF32, cudnn.benchmark, torch.compile,
  fused AdamW, 8 workers, live step/loss/it/s/eta every minute.
  Session end (or Ctrl+C) saves ckpt and — if HF_TOKEN + HF_USER are set —
  uploads checkpoint + progress to huggingface.co/<HF_USER>/piku-ocr.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .charset import build_charset
from .config import PikuConfig
from .data.synthetic import EyeDataset, eye_collate
from .model.eye import PikuEye
from .train_util import auto_vram_batch, load_checkpoint, max_perf, maybe_hf_upload, save_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="corpora")
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--ckpt", default="ckpt/eye_last.pt")
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--hf-user", default="")
    ap.add_argument("--hf-private", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_perf(device)
    cfg = PikuConfig()
    charset = build_charset()
    model = PikuEye(charset, cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=False)
    total_prev = 0
    ckpt_p = Path(args.ckpt)
    if ckpt_p.exists():
        total_prev, _ = load_checkpoint(ckpt_p, model, opt, scaler)
        print(f"resumed from {ckpt_p} at global step {total_prev:,}")

    if args.no_compile is False and device == "cuda":
        model = torch.compile(model)

    def _probe(b: int):
        px = torch.rand(b, 3, cfg.line_height, cfg.line_width, device=device)
        ids = [[4, 5, 6, 7] * 4] * b
        model.loss(px, ids).backward()
        for p in model.parameters():
            p.grad = None
        model.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)

    if device == "cuda":
        batch, peak = auto_vram_batch(_probe, device, base=1024, cap=16384)
        gb = peak / (1 << 30)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        print(f"auto batch: {batch:,} — measured {gb:.1f} GB at 1024 -> target ~{total_gb * 0.85:.0f}/{total_gb:.0f} GB VRAM")
    else:
        batch = 16
    workers = args.workers if device == "cuda" else 0

    ds = EyeDataset(charset, args.corpora, args.fonts)
    if not ds.lines:
        raise SystemExit(f"no clean corpus in {args.corpora} — run scripts/fetch_piku.py first")
    print(f"dataset: {len(ds.lines):,} lines, charset {charset.size}, device {device}")
    loader = DataLoader(
        ds, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=device == "cuda", persistent_workers=device == "cuda",
        prefetch_factor=4 if device == "cuda" else None, collate_fn=eye_collate,
    )

    amp = device == "cuda"
    model.train()
    it, loss_hist = iter(loader), []
    t_start, deadline = time.time(), time.time() + args.minutes * 60
    step_in_session, eta_print = 0, time.time() + 60

    def _finish(status: str):
        el = time.time() - t_start
        mean_last = sum(loss_hist[-500:]) / max(1, len(loss_hist[-500:]))
        summary = {"device": device, "session_steps": step_in_session,
                   "batch": batch, "session_minutes": round(el / 60, 1),
                   "mean_loss_last500": round(mean_last, 3)}
        if loss_hist:
            summary["first_loss"] = round(loss_hist[0], 2)
            summary["last_loss"] = round(loss_hist[-1], 3)
        print(f"{status}: +{step_in_session:,} steps in {el / 60:.1f} min, "
              f"mean_loss_last500={mean_last:.3f}, checkpoint={ckpt_p}")
        maybe_hf_upload(
            args,
            {"ckpts/eye_last.pt": str(ckpt_p)},
            f"Piku-Eye daily session: +{step_in_session:,} steps (global {total_prev + step_in_session:,})",
            "eye", total_prev + step_in_session, summary,
        )

    try:
        while time.time() < deadline:
            try:
                px, ids = next(it)
            except StopIteration:
                it = iter(loader)
                px, ids = next(it)
            px = px.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                loss = model.loss(px, ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_hist.append(loss.item())
            step_in_session += 1
            if time.time() >= eta_print:
                eta_print = time.time() + 60
                el = time.time() - t_start
                its = step_in_session / max(el, 1e-6)
                print(f"step {total_prev + step_in_session:,} loss={loss_hist[-1]:.3f} "
                      f"it/s={its:.1f} session_min_left={(deadline - time.time()) / 60:.0f}")
            if step_in_session % 500 == 0:
                save_checkpoint(ckpt_p, model, opt, scaler, total_prev + step_in_session)
    except KeyboardInterrupt:
        _finish("interrupted — progress saved")
        return
    _finish("session done")


if __name__ == "__main__":
    main()
