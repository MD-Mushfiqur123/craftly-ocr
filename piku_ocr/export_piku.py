"""Export Piku OCR checkpoints to safetensors (no tied weights) + config.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .charset import build_charset
from .config import PikuConfig
from .model.brain import PikuBrain
from .model.eye import PikuEye


def export(eye_ckpt: str, brain_ckpt: str | None, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg, charset = PikuConfig(), build_charset()
    eye = PikuEye(charset, cfg)
    ck = torch.load(eye_ckpt, map_location="cpu", weights_only=False)
    eye.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    sd = {f"eye.{k}": v.contiguous().cpu() for k, v in eye.state_dict().items()}
    if brain_ckpt and Path(brain_ckpt).exists():
        brain = PikuBrain(charset, cfg)
        ck2 = torch.load(brain_ckpt, map_location="cpu", weights_only=False)
        brain.load_state_dict(ck2["model"] if isinstance(ck2, dict) and "model" in ck2 else ck2)
        sd.update({f"brain.{k}": v.contiguous().cpu() for k, v in brain.state_dict().items()})
    save_file(sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({
        "line_height": cfg.line_height, "line_width": cfg.line_width,
        "eye_dim": cfg.eye_dim, "eye_layers": cfg.eye_layers,
        "charset_size": charset.size, "languages": ["bn", "en", "ru"],
    }, indent=2))
    print(f"exported {len(sd):,} tensors -> {out / 'model.safetensors'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", default="ckpt/eye_last.pt")
    ap.add_argument("--brain", default="ckpt/brain_last.pt")
    ap.add_argument("--out", default="export/piku-ocr")
    args = ap.parse_args()
    export(args.eye, args.brain, args.out)


if __name__ == "__main__":
    main()
