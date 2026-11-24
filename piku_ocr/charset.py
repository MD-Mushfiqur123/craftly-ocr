"""Piku OCR charset: Bangla + English + Russian only, compact glyph set."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

BLANK = 0
PAD = 1
UNK = 2

_BENGALI_RANGE = range(0x0980, 0x0A00)
_CYRILLIC_RANGE = range(0x0400, 0x0500)
_EXTRAS = "’‘“”–—…«»§°€£¥¢"


def _keep(ch: str) -> bool:
    if ch == " ":
        return True
    cat = unicodedata.category(ch)
    return cat[0] in "LM" or cat in ("Nd", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sc")


@lru_cache(maxsize=1)
def _glyphs() -> tuple[str, ...]:
    out: list[str] = []

    def add(seq):
        for ch in seq:
            if ch not in out:
                out.append(ch)

    add(chr(c) for c in range(0x20, 0x7F))
    add(chr(c) for c in _CYRILLIC_RANGE if _keep(chr(c)))
    add(chr(c) for c in _BENGALI_RANGE if _keep(chr(c)))
    add(_EXTRAS)
    return tuple(out)


@dataclass
class PikuCharset:
    glyphs: tuple[str, ...]

    def __post_init__(self):
        self.size = len(self.glyphs) + 3
        self.stoi = {g: i + 3 for i, g in enumerate(self.glyphs)}

    def encode(self, text: str) -> list[int]:
        out = [self.stoi.get(ch, UNK) for ch in unicodedata.normalize("NFC", text)]
        return out if out else [UNK]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i < 3:
                chars.append(" ")
            else:
                chars.append(self.glyphs[i - 3])
        return unicodedata.normalize("NFC", "".join(chars)).strip()


def build_charset() -> PikuCharset:
    return PikuCharset(_glyphs())
