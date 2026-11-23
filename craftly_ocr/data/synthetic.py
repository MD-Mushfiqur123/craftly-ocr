"""Synthetic multi-script line renderer -- the bootstrap engine for 210+ languages.

For 200+ languages there is no labelled OCR corpus; the practical path (used
by GOT/TrOCR lineage pretraining and industry OCR pretraining) is to render
corpus text (CC-100, NLLB dumps, Wikipedia) over textured backgrounds with
Noto fonts per script, plus geometric/photometric augmentation. Every sample
carries a perfect label, so one engine serves any language.

Setup for production use:
    apt install fonts-noto fonts-noto-core fonts-noto-ui-core   # ~200 scripts
Provide `corpora/<lang>.txt` files; fall back to the demo corpus otherwise.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..charset import Charset

_DEMO_CORPUS = {
    "bn": ["পাঠ প্রযুক্তি", "বাংলা ভাষা ও সাহিত্য", "ঢাকা বিশ্ববিদ্যালয়", "এক দুই তিন", "ক্রমবর্ধমান"],
    "en": ["The quick brown fox", "Craftware 2026", "RECEIPT #0042", "Hello, world!"],
    "ru": ["Торговый центр", "Привет, мир", "Москва 2026", "РусскаяПочта-СДС"],
}

AUG_BLUR = (0.0, 1.2)
AUG_ROT = (-4, 4)
AUG_SHEAR = (-8, 8)


class SyntheticLineDataset(torch.utils.data.Dataset):
    """Infinite-style dataset yielding (image_tensor, text, char_ids)."""

    def __init__(
        self,
        charset: Charset,
        corpora_dir: str | None = None,
        image_h: int = 48,
        image_w: int = 512,
        fonts_dir: str | None = None,
    ):
        self.charset = charset
        self.image_h = image_h
        self.image_w = image_w
        self.lines = list(_DEMO_CORPUS["bn"]) + list(_DEMO_CORPUS["en"]) + list(_DEMO_CORPUS["ru"])
        if corpora_dir:
            p = Path(corpora_dir)
            cleaned = [f for f in sorted(p.glob("*.txt")) if not f.name.endswith(".raw.txt")]
            for f in cleaned:
                self.lines.extend(
                    ln
                    for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if 3 <= len(ln.strip()) <= 120
                )
        fonts: list[str] = []
        repo_fonts = Path(__file__).resolve().parents[2] / "fonts"
        for d in ([Path(fonts_dir)] if fonts_dir else [repo_fonts, Path("/usr/share/fonts")]):
            for pat in ("*.ttf", "**/*.ttf", "**/*.otf"):
                fonts.extend(str(f) for f in d.glob(pat))
        self.fonts = [
            f
            for f in fonts
            if any(k in f.lower() for k in ("beng", "bangla", "cyril", "notosans", "dejavu", "liberation"))
        ] or fonts
        self.default_font = None  # Pillow >= 10.1 ships a scalable fallback
        try:
            ImageFont.load_default(size=24)
            self.default_font = True if not self.fonts else None
        except TypeError:
            pass

    def __len__(self) -> int:
        return 1_000_000

    def __getitem__(self, idx: int):
        text = random.choice(self.lines)
        size = random.randint(18, 44)
        font = (
            ImageFont.load_default(size=size)
            if not self.fonts
            else ImageFont.truetype(random.choice(self.fonts), size)
        )
        img = Image.new("RGB", (self.image_w, self.image_h), (random.randint(200, 255),) * 3)
        d = ImageDraw.Draw(img)
        d.text((random.randint(4, 40), random.randint(0, 8)), text, font=font, fill=(random.randint(0, 60),) * 3)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*AUG_BLUR)))
        img = img.rotate(random.uniform(*AUG_ROT), expand=False, fillcolor=(240, 240, 240))
        buf = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        px = buf.view(self.image_h, self.image_w, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        ids = self.charset.encode(text)
        return px, ids


def collate_lines(batch):
    px, ids = zip(*batch)
    lens = [len(s) for s in ids]
    if max(lens) < 2:
        ids = [s + [1] for s in ids]
        lens = [len(s) for s in ids]
    return torch.stack(px), list(ids), lens
