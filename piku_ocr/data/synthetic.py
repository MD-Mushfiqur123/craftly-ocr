"""Synthetic line rendering + noisy/clean datasets for Piku OCR training."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from ..charset import PikuCharset
from ..model.brain import corrupt


def _fonts(fonts_dir: str) -> list:
    paths = sorted(list(Path(fonts_dir).glob("*.ttf")) + list(Path(fonts_dir).glob("*.otf")))
    if not paths:
        paths = list(Path("/usr/share/fonts").rglob("*.ttf"))[:8]
    if not paths:
        raise RuntimeError("no fonts found — run scripts/fetch_fonts.py or add fonts/")
    return [ImageFont.truetype(str(p), 32) for p in paths]


class LineRenderer:
    def __init__(self, fonts_dir: str):
        self.fonts = _fonts(fonts_dir)
        self.rng = random.Random(7)

    def render(self, text: str, height: int, width: int) -> torch.Tensor:
        img = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(img)
        font = self.fonts[self.rng.randrange(len(self.fonts))]
        size = min(44, max(16, int((width - 20) / max(1, len(text)) * 1.9)))
        if size < 32:
            font = font.font_variant(size=size)
        y = (height - font.size) // 2
        x = self.rng.randint(4, 16)
        draw.text((x, y), text, fill=0, font=font)
        if self.rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.2, 1.2)))
        t = torch.frombuffer(bytearray(img.convert("RGB").tobytes()), dtype=torch.uint8)
        return (t.reshape(height, width, 3).permute(2, 0, 1).float() / 255.0)


def _load_lines(corpora_dir: str, langs: list[str], cap: int = 2_000_000) -> list[tuple[str, str]]:
    per_lang = max(1, cap // len(langs))
    out: list[tuple[str, str]] = []
    for lang in langs:
        if len(out) >= cap:
            break
        p = Path(corpora_dir) / f"{lang}.txt"
        if not p.exists():
            continue
        seen = set()
        quota = per_lang if len(out) + per_lang <= cap else cap - len(out)
        n_lang = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if 3 <= len(line) <= 120 and line not in seen:
                    seen.add(line)
                    out.append((lang, line))
                    n_lang += 1
                    if n_lang >= quota:
                        break
    return out


class EyeDataset(Dataset):
    """clean text -> rendered line image + ids (for Piku-Eye CTC)."""

    def __init__(self, charset: PikuCharset, corpora_dir: str, fonts_dir: str, langs=("bn", "en", "ru"), cap: int = 2_000_000):
        self.charset = charset
        self.lines = _load_lines(corpora_dir, list(langs), cap=cap)
        self.renderer = LineRenderer(fonts_dir)

    def __len__(self):
        return max(1, len(self.lines))

    def __getitem__(self, idx):
        lang, text = self.lines[idx % len(self.lines)]
        return self.renderer.render(text, 64, 512), self.charset.encode(text)


class BrainDataset(Dataset):
    """corrupted (reader-style) ids -> clean ids (for Piku-Brain)."""

    def __init__(self, charset: PikuCharset, corpora_dir: str, langs=("bn", "en", "ru"), seed: int = 11, cap: int = 2_000_000):
        self.charset = charset
        self.lines = _load_lines(corpora_dir, list(langs), cap=cap)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return max(1, len(self.lines))

    def __getitem__(self, idx):
        _, text = self.lines[idx % len(self.lines)]
        clean = self.charset.encode(text)[:158]
        noisy = self.charset.encode(corrupt(text, self.rng, cer=0.10))[:158]
        return torch.tensor(noisy, dtype=torch.long), clean, len(clean)


def eye_collate(batch):
    px, ids = zip(*batch)
    return torch.stack(px), list(ids)


def brain_collate(batch):
    noisy, clean, lens = zip(*batch)
    t = max(n.shape[0] for n in noisy)
    noisy_p = torch.full((len(noisy), t), 1, dtype=torch.long)
    for i, n in enumerate(noisy):
        noisy_p[i, : n.shape[0]] = n
    return noisy_p, list(clean)
