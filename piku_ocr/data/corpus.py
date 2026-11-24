"""Corpus fetch (CC-100, streaming) + per-script cleaning for Piku OCR."""

from __future__ import annotations

import lzma
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

CC100 = "https://data.statmt.org/cc-100/{lang}.txt.xz"
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SCRIPTS = {
    "bn": (0x0980, 0x09FF, 2),
    "en": ((0x0041, 0x007A), None, 1),
    "ru": (0x0400, 0x04FF, 2),
}


def _script_ratio(ch: str, lang: str) -> float:
    if lang == "bn":
        lo, hi = _SCRIPTS["bn"][0], _SCRIPTS["bn"][1]
        ok = lo <= ord(ch) <= hi
    elif lang == "ru":
        lo, hi = _SCRIPTS["ru"][0], _SCRIPTS["ru"][1]
        ok = lo <= ord(ch) <= hi
    else:
        ok = ("a" <= ch <= "z") or ("A" <= ch <= "Z")
    return float(ok)


def clean_line(line: str, lang: str) -> bool:
    line = unicodedata.normalize("NFC", line).strip()
    if not 3 <= len(line) <= 120 or _URL_RE.search(line):
        return False
    letters = [c for c in line if not c.isspace()]
    if not letters:
        return False
    good = sum(_script_ratio(c, lang) for c in letters)
    need = _SCRIPTS[lang][2] if isinstance(_SCRIPTS[lang][2], int) and lang != "en" else 1
    return good / len(letters) >= (0.5 if lang != "en" else 0.9) and \
        all(("\\ufffd" != c) and (c.isprintable() or c == " ") for c in line)


def clean_corpus(raw_path: Path, out_path: Path, lang: str, mb_budget: int) -> int:
    seen: set[str] = set()
    n = 0
    budget = mb_budget * (1 << 20)
    with open(out_path, "w", encoding="utf-8") as f:
        with open(raw_path, encoding="utf-8", errors="ignore") as r:
            for line in r:
                if not clean_line(line, lang):
                    continue
                line = unicodedata.normalize("NFC", line).strip()
                if line in seen:
                    continue
                seen.add(line)
                f.write(line + "\n")
                n += 1
                if f.tell() > budget:
                    break
    return n


def fetch(lang: str, path: Path, budget: int) -> int:
    data = Path(str(path) + ".tmp")
    total = 0
    decompressor = lzma.LZMADecompressor()
    written = 0
    with urlopen(CC100.format(lang=lang)) as resp, open(data, "wb") as f:
        while written < budget:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(decompressor.decompress(chunk, max_length=budget - written))
            written = f.tell()
            total += len(chunk)
    data.replace(path)
    return written
