"""Fetch high-quality mono corpora (CC-100) for the priority languages.

Streams per-language CC-100 `.xz` files and stops at a byte budget, so the
same script serves a laptop (small budget) or a 1-exabyte-scale lab
(budget=None for full). Line-cleaning happens in clean_corpus.py.

Usage:
    python scripts/fetch_corpora.py --langs bn,en,ru --mb 5000 --out corpora
    python scripts/fetch_corpora.py --langs bn --full --out /mnt/data/cc100
"""
from __future__ import annotations

import argparse
import lzma
import sys
import urllib.request
from pathlib import Path

CC100 = "https://data.statmt.org/cc-100/{lang}.txt.xz"


def fetch(lang: str, path: Path, budget: int | None) -> int:
    url = CC100.format(lang=lang)
    tmp = path.with_suffix(".part")
    written = 0
    req = urllib.request.Request(url, headers={"User-Agent": "craftly-ocr/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dec = lzma.LZMADecompressor()
        with open(tmp, "wb") as f:
            while True:
                comp = r.read(1 << 20)
                if not comp:
                    break
                data = dec.decompress(comp)
                if not data:
                    continue
                if budget is not None and written + len(data) >= budget:
                    f.write(data[: budget - written])
                    written = budget
                    break
                f.write(data)
                written += len(data)
    tmp.replace(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="bn,en,ru")
    ap.add_argument("--mb", type=int, default=None, help="per-language budget in MB")
    ap.add_argument("--full", action="store_true", help="download everything (no cap)")
    ap.add_argument("--out", default="corpora")
    args = ap.parse_args()
    if args.full and args.mb:
        sys.exit("--full and --mb are mutually exclusive")
    budget = None if args.full else (args.mb or 0) * (1 << 20)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang in args.langs.split(","):
        tgt = out / f"{lang}.raw.txt"
        print(f"[{lang}] {url_size(lang)} -> {tgt}", flush=True)
        n = fetch(lang, tgt, budget)
        print(f"[{lang}] wrote {n / (1 << 20):.1f} MB", flush=True)


def url_size(lang: str) -> str:
    return CC100.format(lang=lang)


if __name__ == "__main__":
    main()
