"""Export checkpoints: safetensors + dynamic-int8 quantised state dict."""
from __future__ import annotations

import argparse

import torch


def export_int8(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if v.dim() >= 2 and "norm" not in k and v.abs().max() > 0:
            scale = v.abs().max() / 127.0
            out["q" + k] = (v / scale).round().to(torch.int8)
            out["scale" + k] = scale
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="craftly_ocr")
    args = ap.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.contiguous() for k, v in ckpt["model"].items()}
    sd.pop("decoder.lm_head.weight", None)
    from safetensors.torch import save_file
    save_file(sd, f"{args.out}.safetensors")
    save_file(export_int8(sd), f"{args.out}.int8.safetensors")
    print(f"wrote {args.out}.safetensors and {args.out}.int8.safetensors")


if __name__ == "__main__":
    main()
