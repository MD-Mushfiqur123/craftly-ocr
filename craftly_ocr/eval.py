"""Evaluation: CER / WER with Levenshtein distance."""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyps: list[str], refs: list[str]) -> float:
    total = sum(levenshtein(h, r) for h, r in zip(hyps, refs))
    return total / max(sum(len(r) for r in refs), 1)


def wer(hyps: list[str], refs: list[str]) -> float:
    n = 0
    tot = 0
    for h, r in zip(hyps, refs):
        hw, rw = h.split(), r.split()
        tot += len(rw)
        n += levenshtein(" ".join(hw), " ".join(rw))
    return n / max(tot, 1)


def _cli():
    import argparse
    from pathlib import Path

    import torch

    from .charset import build_charset
    from .config import default_config
    from .model import CraftlyOCR

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", dest="set_dir", required=True,
                    help="dir with images/ and ground_truth.txt (path<TAB>transcription lines)")
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = default_config()
    cs = build_charset()
    model = CraftlyOCR(cfg, cs)
    model.load_state_dict(ckpt["model"])
    model.eval()
    hyps, refs = [], []
    root = Path(args.set_dir)
    for ln in (root / "ground_truth.txt").read_text(encoding="utf-8").splitlines():
        rel, ref = ln.split("\t", 1)
        from PIL import Image
        img = Image.open(root / "images" / rel).convert("RGB").resize((256, 48))
        buf = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        px = buf.view(48, 256, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        hyp = model.read_crops(px.unsqueeze(0))[0]
        hyps.append(hyp); refs.append(ref)
    print(f"CER {cer(hyps, refs):.4f}   WER {wer(hyps, refs):.4f}   n={len(hyps)}")


if __name__ == "__main__":
    _cli()
