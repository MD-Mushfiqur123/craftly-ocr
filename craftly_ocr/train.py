"""Stage trainer for Craftly OCR.

Usage (on the RTX Pro 6000 / 96GB):
    python -m craftly_ocr.train --stage reader  --steps 100000
    python -m craftly_ocr.train --stage e2e    --data data/pages  --epochs 3

Stages (mirror the GOT recipe, adapted for a micro model):
  reader    synthetic pretrain of encoder+CTC reader (pure multi-script lines)
  readerft  fine-tune reader on real line data (data/lines, pairs or AnnotationFolder)
  det       detection head on data/pages (boxes -> DBNet heatmaps)
  decoder   decoder alignment on data/jsonl ({image, prompt, target})
  e2e       joint fine-tune, all heads, low LR
"""
from __future__ import annotations

import argparse
import math

import torch

from .charset import build_charset
from .config import default_config
from .data.synthetic import SyntheticLineDataset
from .model import CraftlyOCR


def _collate_lines(batch):
    px, ids = zip(*batch)
    lens = [len(s) for s in ids]
    if max(lens) < 2:
        ids = [s + [1] for s in ids]
        lens = [len(s) for s in ids]
    return torch.stack(px), list(ids), lens


def _pad_list(seqs, pad=1):
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [pad] * (m - len(s)) for s in seqs], dtype=torch.long)


def make_loader(ds, bs, workers=8):
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=workers,
        collate_fn=_collate_lines, drop_last=True, pin_memory=True,
    )


def build_model(cfg, device):
    charset = build_charset()
    model = CraftlyOCR(cfg, charset).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, charset: {len(charset)}")
    return model


def run_reader_stage(args, cfg, device):
    """Stage 0: encoder body + CTC reader on synthetic lines."""
    charset = build_charset()
    cfg.batch_size = args.batch
    model = build_model(cfg, device).float()
    ds = SyntheticLineDataset(charset, corpora_dir=args.corpora)
    dl = make_loader(ds, args.batch, args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = math.ceil(cfg.warmup_steps + args.steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=cfg.warmup_steps / max(total, 1))

    model.train()
    step = 0
    running = 0.0
    import time
    t0 = time.time()
    for px, ids, lens in dl:
        px = px.to(device, non_blocking=True)
        logits = model.reader(px)
        loss = model.reader.loss(logits, ids, lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        running += loss.item()
        step += 1
        if step % 50 == 0:
            rate = step / (time.time() - t0)
            eta_min = (args.steps - step) / max(rate, 1e-6) / 60
            print(
                f"step {step}/{args.steps} reader_loss={running/50:.3f} "
                f"{rate:.1f} it/s eta={eta_min:.0f} min",
                flush=True,
            )
            running = 0.0
        if step >= args.steps:
            break
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, args.out)
    print(f"saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["reader", "readerft", "det", "decoder", "e2e"], required=True)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--corpora", default=None, help="dir of <lang>.txt corpora for the renderer")
    ap.add_argument("--data", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="craftly_stage0.pt")
    args = ap.parse_args()

    cfg = default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; running smoke-training on CPU (use a tiny --steps)")
    if args.stage in ("reader", "readerft"):
        run_reader_stage(args, cfg, device)
    else:
        raise SystemExit(
            f"stage {args.stage!r} requires data packs; see docs/TRAINING.md for the pack format "
            "(stages 2-4 wire the same trainer to page-level JSONL)."
        )


if __name__ == "__main__":
    main()
