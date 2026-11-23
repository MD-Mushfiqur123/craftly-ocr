import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from craftly_ocr import build_charset, default_config  # noqa: E402
from craftly_ocr.model import CraftlyOCR  # noqa: E402


def test_charset_roundtrip_multiscript():
    cs = build_charset()
    for text in ("পাঠ প্রযুক্তি", "Торговый центр", "Hello, world! 2026", "¥12 €–«ARRÉT»"):
        assert cs.decode(cs.encode(text), collapse=False) == text


def test_model_forward_shapes_and_losses():
    cfg = default_config()
    cfg.attn_blocks = 2
    cfg.decoder_layers = 2
    cfg.encoder_depths = (1, 1, 1, 1, 1)
    cs = build_charset()
    model = CraftlyOCR(cfg, cs)
    px = torch.randn(2, 3, 512, 512)
    out = model(
        px,
        reader_crops=torch.randn(2, 3, 48, 256),
        reader_targets=[[5, 6, 7], [8, 9]],
        reader_lens=[3, 2],
        tokens=torch.randint(5, 100, (2, 12)),
        labels=torch.randint(5, 100, (2, 12)),
    )
    assert out["reader_loss"].isfinite()
    assert out["decoder_loss"].isfinite()
    mem, fine, grid = model.encode(px)
    assert mem.shape == (2, cfg.num_registers + 256, cfg.encoder_dims[-1])
    assert fine.shape[-2:] == (64, 64)


def test_small_training_step_decreases_reader_loss():
    cfg = default_config()
    cfg.attn_blocks = 1
    cfg.encoder_depths = (1, 1, 1, 1, 1)
    cs = build_charset()
    model = CraftlyOCR(cfg, cs)
    px = torch.randn(4, 3, 48, 256)
    ids = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15], [16, 17]]
    lens = [4, 4, 3, 2]
    opt = torch.optim.AdamW(model.reader.parameters(), lr=3e-3)
    first = None
    for i in range(8):
        logits = model.reader(px)
        loss = model.reader.loss(logits, ids, lens)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
