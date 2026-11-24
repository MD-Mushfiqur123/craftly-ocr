import torch

from piku_ocr.charset import build_charset
from piku_ocr.config import PikuConfig
from piku_ocr.data.synthetic import LineRenderer
from piku_ocr.model.brain import PikuBrain
from piku_ocr.model.eye import PikuEye


def test_charset_roundtrip():
    cs = build_charset()
    for s in ["পাঠ প্রযুক্তি", "Торговый центр", "Hello, world! 2026"]:
        assert cs.decode(cs.encode(s)) == s
    return cs


def test_eye_forward_loss_decode():
    cs = test_charset_roundtrip()
    cfg = PikuConfig()
    m = PikuEye(cs, cfg)
    px = torch.rand(2, 3, 64, 512)
    logits = m.logits(px)
    assert logits.shape[:2] == (2, 128) and torch.isfinite(logits).all()
    ids = [cs.encode("পাঠ প্রযুক্তি"), cs.encode("Hello")]
    loss = m.loss(px, ids)
    assert torch.isfinite(loss)
    text = m.decode(px)
    assert len(text) == 2


def test_eye_loss_decreases():
    cs, cfg = build_charset(), PikuConfig()
    cfg.eye_layers = 2
    m = PikuEye(cs, cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    torch.manual_seed(0)
    texts = ["পাঠ প্রযুক্তি লিমিটেড", "Торговый центр открыт", "Hello, world! 2026"]
    px = torch.stack([LineRenderer("fonts").render(t, 64, 512) for t in texts])
    ids = [cs.encode(t) for t in texts]
    first = None
    for i in range(60):
        loss = m.loss(px, ids)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5, f"no training progress: {first} -> {loss.item()}"


def test_brain_denoises():
    cs, cfg = build_charset(), PikuConfig()
    m = PikuBrain(cs, cfg)
    from piku_ocr.data.synthetic import BrainDataset, brain_collate
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "en.txt"
        p.write_text("Hello, world! 2026\nТорговый центр открыт\nপাঠ প্রযুক্তি লিমিটেড\n", encoding="utf-8")
        ds = BrainDataset(cs, d)
        noisy, clean = brain_collate([ds[i] for i in range(3)])
        loss = m.loss(noisy, clean)
        assert torch.isfinite(loss)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        first = None
        for i in range(60):
            loss = m.loss(noisy, clean)
            opt.zero_grad(); loss.backward(); opt.step()
            first = first if first is not None else loss.item()
        assert loss.item() < first * 0.5
