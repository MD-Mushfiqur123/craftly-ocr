"""Piku-Brain trainer — stage 2. 2h/day resumable denoising training.

Runs after an eye checkpoint exists. Noise mimics real reader errors; the
brain learns to map corrupted char ids back to clean corpus text.
Protocol: `python -m piku_ocr.train_brain --minutes 120` daily.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .charset import build_charset
from .config import PikuConfig
from .data.synthetic import BrainDataset, brain_collate
from .model.brain import PikuBrain
from .train_util import auto_vram_batch, load_checkpoint, max_perf, maybe_hf_upload, save_checkpoint


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="corpora")
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", default="ckpt/brain_last.pt")
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--hf-user", default="")
    ap.add_argument("--hf-private", action="store_true")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_perf(device)
    cfg = PikuConfig()
    charset = build_charset()
    model = PikuBrain(charset, cfg).to(device)
    if device == "cuda":
        def _probe(b):
            noisy = torch.randint(3, charset.size, (b, 64), device=device)
            clean = [[4, 5, 6, 7] * 4] * b
            model.loss(noisy, clean).backward()
            model.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)

        batch, peak = auto_vram_batch(_probe, device, base=1024, cap=16384)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
        print(f"auto batch: {batch:,} — measured {peak / (1 << 30):.1f} GB at 1024 -> target ~{total_gb * 0.85:.0f}/{total_gb:.0f} GB VRAM")
    else:
        batch = 16
    workers = args.workers if device == "cuda" else 0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=device == "cuda")

    done, total_prev = 0, 0
    ckpt_p = Path(args.ckpt)
    if ckpt_p.exists():
        total_prev, _ = load_checkpoint(ckpt_p, model, opt)
        print(f"resumed from {ckpt_p} at global step {total_prev:,}")

    ds = BrainDataset(charset, args.corpora)
    if not ds.lines:
        raise SystemExit(f"no clean corpus in {args.corpora} — run fetch first")
    print(f"dataset: {len(ds.lines):,} lines, charset {charset.size}, device {device}")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=brain_collate)

    model.train()
    it, loss_hist = iter(loader), []
    t_start, deadline = time.time(), time.time() + args.minutes * 60
    step_in_session, eta_print = 0, time.time() + 60

    def _finish(status: str):
        mean_last = sum(loss_hist[-500:]) / max(1, len(loss_hist[-500:]))
        summary = {"device": device, "session_steps": step_in_session, "batch": batch,
                   "session_minutes": round((time.time() - t_start) / 60, 1),
                   "mean_loss_last500": round(mean_last, 3)}
        print(f"{status}: +{step_in_session:,} steps, mean_loss_last500={mean_last:.3f}, ckpt={ckpt_p}")
        maybe_hf_upload(
            args, {"ckpts/brain_last.pt": str(ckpt_p)},
            f"Piku-Brain daily session: +{step_in_session:,} steps (global {total_prev + step_in_session:,})",
            "brain", total_prev + step_in_session, summary,
        )

    try:
        while time.time() < deadline:
            try:
                noisy, clean = next(it)
            except StopIteration:
                it = iter(loader)
                noisy, clean = next(it)
            noisy = noisy.to(device, non_blocking=True)
            loss = model.loss(noisy, clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_hist.append(loss.item())
            step_in_session += 1
            if time.time() >= eta_print:
                eta_print = time.time() + 60
                print(f"step {total_prev + step_in_session:,} loss={loss_hist[-1]:.3f} "
                      f"session_min_left={(deadline - time.time()) / 60:.0f}")
            if step_in_session % 1000 == 0:
                save_checkpoint(ckpt_p, model, opt, None, total_prev + step_in_session)
    except KeyboardInterrupt:
        _finish("interrupted — progress saved")
        return
    _finish("session done")


if __name__ == "__main__":
    main()
