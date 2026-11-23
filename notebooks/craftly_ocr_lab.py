import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🔍 Craftly OCR — Research Lab
        Reactive notebook for building a **Mirage-inspired micro VLM OCR** (63.4M
        params) for **Bangla 🇧🇩 · English 🇬🇧 · Russian 🇷🇺**. Everything is
        reactive: move a slider and dependent cells re-run.

        ## Research ground truth (what this architecture is built on)
        | Paper / system | Key finding we use |
        |---|---|
        | **Mirage** (Cactus, 2026) — 51.3M/22MB all-in-one vision model | HConvGLU: fixed Walsh–Hadamard transforms around small banded convs ≈ 8× fewer weights; heads do perception, a small decoder only formats JSON |
        | **GOT-OCR 2.0** (arXiv:2409.01704) | high-compression encoder (1024px → 256 tokens); multi-stage recipe: synthetic pretrain → real fine-tune → joint |
        | **TrOCR** (arXiv:2109.10282) | synthetic-text pretraining transfers strongly to real OCR |
        | **PARSeq** (CVPR 2022) | iterative refinement beats naive attention decoding; we keep CTC for on-device cost |
        | **DBNet / DBNet++** (AAAI 2020/2022) | differentiable binarisation detection heads |
        | **ConvNeXtV2** (arXiv:2401.10571) | Global Response Normalisation inside HConvGLU |

        **Why CTC over 210+ languages:** one shared Unicode inventory
        (`craftly_ocr.charset`, 39.6k graphemes) — no per-language heads;
        low-resource scripts are bootstrapped by the synthetic renderer with
        per-script Noto fonts.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 1 · Data budget (reactive)")
    return


@app.cell
def _(mo):
    langs = mo.ui.multiselect(
        options=["bn", "en", "ru"], value=["bn", "en", "ru"], label="languages"
    )
    mb = mo.ui.slider(10, 20000, value=200, label="MB per language (CC-100 budget)")
    mo.hstack([langs, mb], justify="start")
    return langs, mb


@app.cell
def _(langs, mb):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path.cwd()))
    from scripts.fetch_corpora import fetch as _fetch

    corpora_dir = _Path("corpora")
    corpora_dir.mkdir(exist_ok=True)
    budget = mb.value * (1 << 20)
    sizes = {}
    for lang in langs.value:
        tgt = corpora_dir / f"{lang}.raw.txt"
        if not tgt.exists() or tgt.stat().st_size < budget:
            n = _fetch(lang, tgt, budget)
        else:
            n = tgt.stat().st_size
        sizes[lang] = round(n / (1 << 20), 1)
    sizes
    return (corpora_dir, sizes,)


@app.cell
def _(mo):
    mo.md("## 2 · Clean: script filter + dedup")
    return


@app.cell
def _(corpora_dir, langs):
    import subprocess as _subprocess
    import sys as _sys

    out = _subprocess.run(
        [_sys.executable, "scripts/clean_corpus.py",
         "--raw-dir", str(corpora_dir), "--out-dir", str(corpora_dir),
         "--langs", ",".join(langs.value), "--mb", "500"],
        capture_output=True, text=True,
    )
    stats = out.stdout.strip().splitlines()
    lines = {
        lang: sum(1 for _ in open(corpora_dir / f"{lang}.txt", encoding="utf-8"))
        for lang in langs.value
        if (corpora_dir / f"{lang}.txt").exists()
    }
    stats, lines
    return (lines,)


@app.cell
def _(mo):
    mo.md("## 3 · Synthetic render preview (TrOCR-style bootstrapping)")
    return


@app.cell
def _(corpora_dir):
    from pathlib import Path as _Path

    from craftly_ocr import build_charset as _build_charset
    from craftly_ocr.data.synthetic import SyntheticLineDataset

    ds = SyntheticLineDataset(_build_charset(), corpora_dir=str(corpora_dir), fonts_dir="fonts")
    import torch as _torch

    batch = _torch.utils.data.DataLoader(
        ds, batch_size=8, num_workers=0,
        collate_fn=lambda b: (_torch.stack([px for px, _ in b]), [ids for _, ids in b]),
    )
    imgs, ids = next(iter(batch))
    f"corpus lines: {len(ds.lines):,} · crops: {tuple(imgs.shape)}"
    return (imgs, ds, ids,)


@app.cell
def _(ds, ids, imgs, mo):
    cs = ds.charset
    texts = [cs.decode(seq, collapse=False) for seq in ids[:6]]
    mo.md("### Rendered training pairs (image ⇄ exact label)\n\n" + "\n\n".join(f"`{t}`" for t in texts))
    return (cs, texts,)


@app.cell
def _(mo):
    mo.md("## 4 · Model (smoke-scale) · full config = 63.4M params")
    return


@app.cell
def _(mo):
    attn_blocks = mo.ui.slider(1, 12, value=1, label="attention blocks")
    steps = mo.ui.slider(10, 2000, value=60, label="smoke train steps")
    lr = mo.ui.number(1e-3, 1e-2, value=3e-3, step=5e-4, label="reader LR")
    mo.hstack([attn_blocks, steps, lr], justify="start")
    return attn_blocks, lr, steps


@app.cell
def _(attn_blocks, ds):
    import torch as _torch

    from craftly_ocr.config import default_config
    from craftly_ocr.model import CraftlyOCR
    cfg = default_config()
    cfg.attn_blocks = attn_blocks.value
    cfg.encoder_depths = (1, 1, 1, 1, 1)
    model = CraftlyOCR(cfg, ds.charset)
    pcount = {n: round(sum(p.numel() for p in m.parameters()) / 1e6, 2) for n, m in model.named_children()}
    torch = _torch
    (pcount, f"SMOKE total {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return (model, torch,)


@app.cell
def _(mo):
    mo.md("## 5 · CTC training (stage 0) — loss should fall")
    return


@app.cell
def _(ds, lr, model, steps, torch):
    opt = torch.optim.AdamW(model.reader.parameters(), lr=lr.value)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, num_workers=0, shuffle=True,
        collate_fn=lambda b: (torch.stack([px for px, _ in b]), [ids_b for _, ids_b in b]),
    )
    it = iter(loader)
    loss_history = []
    for step in range(steps.value):
        px_b, ids_b = next(it)
        loss = model.reader.loss(model.reader(px_b), ids_b, [len(s) for s in ids_b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            loss_history.append(round(loss.item(), 2))
    loss_history
    return (loss_history,)


@app.cell
def _(loss_history, mo):
    improved = len(loss_history) > 1 and loss_history[-1] < loss_history[0]
    mo.md(
        ("✅" if improved else "⚠️")
        + f" loss {loss_history[0]} → {loss_history[-1]} (every 10 steps: `{loss_history}`)"
    )
    return


@app.cell
def _(mo):
    mo.md("## 6 · Greedy CTC decode sanity (unseen renders; improves with training)")
    return


@app.cell
def _(ds, model, mo, torch):
    loader_ck = torch.utils.data.DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=lambda b: torch.stack([px for px, _ in b]),
    )
    hyps = model.read_crops(next(iter(loader_ck)))
    mo.md("\n\n".join(f"`{h}`" for h in hyps))
    return


@app.cell
def _(mo):
    mo.md(
        """## Next steps (lab plan)
        1. crank the budget sliders (your 150 GB RAM machine: bn/en/ru × GB-scale)
        2. `python -m craftly_ocr.train --stage reader --steps 200000` on the RTX Pro 6000
        3. stage 1: real data (BanglaWriting / Cyrillic Handwriting / ICDAR MLT)
        4. stages 2–4: detector → decoder → joint e2e → int8/2-bit export
        *We will win.* 🚀"""
    )
    return


if __name__ == "__main__":
    app.run()
