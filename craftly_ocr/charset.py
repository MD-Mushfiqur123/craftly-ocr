"""Multi-script glyph inventory for the CTC OCR reader.

Design note (see docs/DESIGN.md, section "Reading 210+ languages"):
the reader is CTC over *graphemes*, not bytes. A single shared inventory of
~6k graphemes covers every living script in the Unicode BMP plus the BMP Han
range used by CJK. This keeps the CTC head cheap (one linear projection) while
letting one set of reader weights serve every language. Low-resource languages
are bootstrapped by the synthetic renderer; no per-language heads are used.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# (start, end, note) ranges, inclusive-exclusive.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0020, 0x007F, "ASCII"),
    (0x00A0, 0x0180, "Latin-1 supplement + Latin Ext-A (accents, eszett)"),
    (0x0180, 0x0250, "Latin Extended-B (Vietnamese, Polish, Turkish)"),
    (0x0250, 0x02AF, "IPA common"),
    (0x0370, 0x0400, "Greek"),
    (0x0400, 0x0500, "Cyrillic + historic"),
    (0x0500, 0x0530, "Cyrillic supplement"),
    (0x0530, 0x0590, "Armenian"),
    (0x0590, 0x0600, "Hebrew"),
    (0x0600, 0x0700, "Arabic"),
    (0x0700, 0x0750, "Syriac"),
    (0x0780, 0x07C0, "Thaana"),
    (0x0900, 0x0980, "Devanagari"),
    (0x0980, 0x0A00, "Bengali (Bangla + Assamese)"),
    (0x0A00, 0x0A80, "Gurmukhi"),
    (0x0A80, 0x0B00, "Gujarati"),
    (0x0B00, 0x0B80, "Oriya"),
    (0x0B80, 0x0C00, "Tamil"),
    (0x0C00, 0x0C80, "Telugu"),
    (0x0C80, 0x0D00, "Kannada"),
    (0x0D00, 0x0D80, "Malayalam"),
    (0x0D80, 0x0E00, "Sinhala"),
    (0x0E00, 0x0E80, "Thai"),
    (0x0E80, 0x0F00, "Lao"),
    (0x0F00, 0x1000, "Tibetan"),
    (0x1000, 0x10A0, "Myanmar"),
    (0x10A0, 0x1100, "Georgian"),
    (0x1100, 0x1200, "Hangul Jamo"),
    (0x1200, 0x1380, "Ethiopic"),
    (0x13A0, 0x1400, "Cherokee"),
    (0x1400, 0x1680, "Canadian Aboriginal"),
    (0x1680, 0x17A0, "Ogham + Mongolian"),
    (0x1780, 0x1800, "Khmer"),
    (0x3040, 0x30A0, "Hiragana"),
    (0x30A0, 0x3100, "Katakana"),
    (0x4E00, 0x9FD6, "CJK Unified (BMP)"),
    (0xAC00, 0xD7A4, "Hangul syllables"),
    (0xF900, 0xFB00, "CJK compatibility"),
    (0xFB50, 0xFDFF, "Arabic presentation A"),
    (0x2000, 0x2070, "punctuation, super/subscripts"),
    (0x20A0, 0x20C0, "currency symbols"),
    (0x2100, 0x2150, "letterlike symbols"),
    (0x2190, 0x2300, "arrows"),
    (0x2500, 0x27C0, "box drawing, geometric, dingbats subset"),
    (0xFB00, 0xFB50, "Latin ligatures"),
)

Special = tuple[int, str]
_SPECIALS: tuple[Special, ...] = (
    (0x0000, "<blank>"),  # CTC blank, index == spec index
    (1, "<pad>"),
    (2, "<unk>"),
    (3, "<bos>"),
    (4, "<eos>"),
)


@dataclass
class Charset:
    itos: tuple[str, ...]
    stoi: dict[str, int]
    blank: int

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        return [self.stoi.get(ch, 2) for ch in text]

    def decode(self, ids: list[int], collapse: bool = True) -> str:
        gtts = ids if not collapse else _ctc_collapse(ids, self.blank)
        return unicodedata.normalize("NFC", "".join(self.itos[i] for i in gtts))


def _ctc_collapse(ids: list[int], blank: int) -> list[int]:
    out: list[int] = []
    prev = -1
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def build_charset() -> Charset:
    itos: list[str] = []
    itos.extend(name for _, name in _SPECIALS)
    seen: set[str] = set(itos)
    for start, end, _ in _SCRIPT_RANGES:
        for cp in range(start, end):
            ch = chr(cp)
            if ch.isprintable() and ch not in seen:
                seen.add(ch)
                itos.append(ch)
    return Charset(tuple(itos), {c: i for i, c in enumerate(itos)}, blank=0)
