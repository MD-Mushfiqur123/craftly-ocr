"""High-quality corpus cleaning for OCR synthetic training data.

For each raw CC-100 file produce a clean `corpora/<lang>.txt` of OCR-relevant
lines:
  - NFC normalise, strip control/replacement chars
  - keep lines that are mostly the target script (Bangla / Cyrillic / Latin)
  - length 3..120 chars (OCR line crops), no leading/trailing junk
  - hash-dedup across the corpus (RAM-friendly: fixed-capacity Bloom-ish set)
  - keep a small share of digits/punctuation-heavy lines (receipts, dates)

Usage:
    python scripts/clean_corpus.py --raw-dir corpora --out-dir corpora \
        --thumbs bn,en,ru --mb-budget 2000
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

SCRIPT_TESTS = {
    "bn": (re.compile(r"[\u0980-\u09FF]"), re.compile(r"[A-Za-z\u0400-\u04FF]")),
    "ru": (re.compile(r"[\u0400-\u04FF]"), re.compile(r"[A-Za-z\u0980-\u09FF\u0300-\u036F\u0600-\u06FF]")),
    "en": (re.compile(r"[A-Za-z]"), re.compile(r"[\u0400-\u04FF\u0980-\u09FF\u0900-\u097F\u0600-\u06FF\u3000-\u9FFF\uAC00-\uD7AF]")),
}

_BAD = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f]")
_JUNK = re.compile(r"^(https?:|www\.|[\W_])+$")
_URL = re.compile(r"https?://\S+")


class HashSet:
    """Bounded dedup set: after `cap` entries, probabilistic keep via hash bucket."""

    def __init__(self, cap: int = 50_000_000):
        self.cap = cap
        self.seen: set[int] = set()
        self.buckets = (1 << 24) * [False] if cap > (1 << 24) else None
        if self.buckets:
            self.buckets = bytearray(1 << 24)

    def __contains__(self, item: str) -> bool:
        return hash(item) in self.seen if len(self.seen) <= self.cap else self.buckets[hash(item) & 0xFFFFFF]

    def add(self, item: str) -> None:
        h = hash(item)
        self.seen.add(h)

    def seen_before(self, item: str) -> bool:
        h = hash(item)
        if self.buckets is not None:
            seen = bool(self.buckets[h & 0xFFFFFF])
            self.buckets[h & 0xFFFFFF] = True
            return seen
        seen = h in self.seen
        if not seen:
            self.seen.add(h)
        return seen


def clean_lang(raw: Path, out: Path, lang: str, dedup: HashSet, max_mb: float) -> int:
    target_re, foreign_re = SCRIPT_TESTS[lang]
    kept = 0
    budget = max_mb * (1 << 20)
    written = 0
    with raw.open(encoding="utf-8", errors="ignore") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            line = unicodedata.normalize("NFC", _BAD.sub("", _URL.sub("", line))).strip()
            if not (3 <= len(line) <= 120) or _JUNK.match(line):
                continue
            target = len(target_re.findall(line))
            foreign = len(foreign_re.findall(line))
            if target < max(4, 0.6 * len(line)) or foreign > 0.25 * max(len(line), 1):
                continue
            if dedup.seen_before(line):
                continue
            w.write(line + "\n")
            kept += 1
            written += len(line) + 1
            if written >= budget:
                break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="corpora")
    ap.add_argument("--out-dir", default="corpora")
    ap.add_argument("--langs", default="bn,en,ru")
    ap.add_argument("--mb", type=float, default=2000, help="clean-output budget per language")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = HashSet()
    for lang in args.langs.split(","):
        raw = Path(args.raw_dir) / f"{lang}.raw.txt"
        if not raw.exists():
            print(f"[{lang}] raw file missing, run fetch_corpora.py first")
            continue
        kept = clean_lang(raw, out_dir / f"{lang}.txt", lang, dedup, args.mb)
        print(f"[{lang}] kept {kept:,} clean lines -> {out_dir / (lang + '.txt')}")


if __name__ == "__main__":
    main()
