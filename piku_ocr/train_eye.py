"""Piku-Eye trainer — stage 1. 2h/day resumable CTC training, max-perf defaults.

Settings are fixed for one RTX Pro 6000 96GB (no guessing on the CLI):
  batch 1024, bf16 autocast, TF32, cudnn.benchmark, compiled model (CUDA),
  fused AdamW, bucketed-ish shuffling, 8 workers.
Protocol: run daily `python -m piku_ocr.train_eye --minutes 120`; it resumes
from ckpt/eye_last.pt automatically and prints live step/loss/it/s/eta.
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
from .train_util import load_checkpoint, max_perf, save_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="corpora")
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--ckpt", default="ckpt/eye_last.pt")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_perf(device)
    cfg = PikuConfig()
    charset = build_charset()
    model = PikuEye(charset, cfg).to(device)
    batch = cfg.max_perf["batch"] if device == "cuda" else 16
    workers = args.workers if device == "cuda" else 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=False)
    done_steps, total_prev = 0, 0
    ckpt_p = Path(args.ckpt)
    if ckpt_p.exists():
        total_prev, _ = load_checkpoint(ckpt_p, model, opt, scaler)
        print(f"resumed from {ckpt_p} at global step {total_prev:,}")

    if args.no_compile is False and device == "cuda":
        model = torch.compile(model)

    ds = EyeDataset(charset, args.corpora, args.fonts)
    if ds.lines:
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
    save_checkpoint(ckpt_p, model, opt, scaler, total_prev + step_in_session)
    el = time.time() - t_start
    mean_last = sum(loss_hist[-500:]) / max(1, len(loss_hist[-500:]))
    print(f"session done: +{step_in_session:,} steps in {el / 60:.1f} min, "
          f"mean_loss_last500={mean_last:.3f}, checkpoint={ckpt_p}")


if __name__ == "__main__":
    main()
