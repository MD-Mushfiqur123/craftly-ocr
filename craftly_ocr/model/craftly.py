"""Full Craftly OCR model: shared encoder + detector + CTC reader + decoder."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CraftlyConfig
from .encoder import VisionEncoder
from .detector import DetectionHead
from .reader import OCRReader
from .decoder import CraftlyDecoder


class CraftlyOCR(nn.Module):
    def __init__(self, cfg: CraftlyConfig, charset):
        super().__init__()
        self.cfg = cfg
        self.charset = charset
        self.encoder = VisionEncoder(cfg)
        self.detector = DetectionHead(cfg)
        self.reader = OCRReader(cfg, len(charset))
        self.decoder = CraftlyDecoder(cfg, enc_dim=cfg.encoder_dims[-1])

    def encode(self, pixels: torch.Tensor):
        mem, (fine, grid) = self.encoder(pixels)
        return mem, fine, grid

    def forward(self, pixels: torch.Tensor, task_tokens: torch.Tensor | None = None, **kw):
        """Training forward; each sub-loss is returned and stage trainers pick them."""
        out: dict = {}
        mem, fine, grid = self.encode(pixels)
        det = self.detector(fine)
        out["det"] = det

        if "reader_targets" in kw:
            crops: torch.Tensor = kw["reader_crops"]
            logits = self.reader(crops)
            out["reader_loss"] = self.reader.loss(logits, kw["reader_targets"], kw["reader_lens"])
            out["reader_logits"] = logits

        if "labels" in kw:
            task = task_tokens if task_tokens is not None else torch.full(
                (pixels.size(0), 1), 3, device=pixels.device, dtype=torch.long
            )
            tok = torch.cat([task, kw["tokens"]], dim=1)
            logits = self.decoder(tok, mem)
            out["decoder_loss"] = self.decoder.loss(logits, torch.cat([task, kw["labels"]], dim=1))
        return out

    @torch.no_grad()
    def read_crops(self, crops: torch.Tensor) -> list[str]:
        return self.reader.decode(self.reader(crops), self.charset)

    @torch.no_grad()
    def describe(self, pixels: torch.Tensor, prompt_ids: torch.Tensor, max_new_tokens: int = 128):
        mem, _, _ = self.encode(pixels)
        out = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.decoder(out, mem)[:, -1]
            out = torch.cat([out, logits.argmax(-1, keepdim=True)], dim=1)
        return list(out[:, prompt_ids.size(1):])
