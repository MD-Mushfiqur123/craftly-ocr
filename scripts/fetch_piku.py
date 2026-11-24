"""Fetch + clean CC-100 corpora (bn/en/ru) for Piku OCR.

Usage:
  python scripts/fetch_piku.py --langs bn,en,ru --mb 2000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from piku_ocr.data.corpus import clean_corpus, fetch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="bn,en,ru")
    ap.add_argument("--mb", type=int, default=2000)
    ap.add_argument("--dir", default="corpora")
    args = ap.parse_args()
    d = Path(args.dir)
    d.mkdir(exist_ok=True)
    for lang in args.langs.split(","):
        raw, clean = d / f"{lang}.raw.txt", d / f"{lang}.txt"
        t0 = time.time()
        if not raw.exists() or raw.stat().st_size < args.mb * (1 << 20):
            n = fetch(lang, raw, args.mb * (1 << 20))
            print(f"fetch {lang}: {n / (1 << 20):.1f} MB in {time.time() - t0:.1f}s")
        else:
            print(f"fetch {lang}: cached")
        t1 = time.time()
        lines = clean_corpus(raw, clean, lang, args.mb)
        print(f"clean {lang}: {lines:,} lines in {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
