# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pillow",
#     "matplotlib",
#     "torch",
#     "safetensors",
#     "huggingface_hub",
# ]
# ///
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
        # 🚀 Craftly OCR — Train & Publish to Hugging Face
        End-to-end: **corpus → clean → synthetic render → CTC reader training →
        export → push to `huggingface.co/<username>/<model>`**.

        Performance stack is always on: bf16 autocast, TF32 + cuDNN benchmark,
        length-bucketed batching, prefetch workers, fused AdamW (CUDA),
        `torch.compile` toggle below.

        > Token never leaves this notebook process — it goes straight to
        > huggingface_hub's credential store for this session only.
        """
    )
    return


@app.cell
def _(mo):
    """Bootstrap: fetch the Craftly OCR repo into this notebook's working dir.
    Works on molab (single uploaded file, fresh env) AND in a local clone."""
    import glob as _glob
    import importlib as _importlib
    import io as _io
    import os as _os
    import sys as _sys
    import urllib.request as _request
    import zipfile as _zipfile

    _ref = "vibe/craftly-ocr-v01-d1254f"
    if not _os.path.exists("craftly_ocr"):
        _resp = _request.urlopen(
            "https://api.github.com/repos/MD-Mushfiqur123/craftly-ocr/zipball/" + _ref
        )
        _zipfile.ZipFile(_io.BytesIO(_resp.read())).extractall(".")
        _os.chdir(sorted(_glob.glob("MD-Mushfiqur123-craftly-ocr-*"))[0])
    _sys.path.insert(0, _os.getcwd())
    repo_root = _os.getcwd()

    def load(name):
        return _importlib.import_module(name)

    mo.md(f"✅ Craftly OCR repo ready: `{repo_root}`")
    return load, repo_root


@app.cell
def _(mo):
    mo.md("## 1 · Your Hugging Face identity")
    return


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF token (write)", placeholder="hf_...")
    hf_user = mo.ui.text(label="HF username", placeholder="e.g. mushfiqur")
    hf_model = mo.ui.text(label="model repo name", value="craftly-ocr")
    private = mo.ui.switch(value=True, label="private repo")
    mo.vstack([hf_token, hf_user, hf_model, private])
    return hf_model, hf_token, hf_user, private


@app.cell
def _(hf_model, hf_token, hf_user, mo):
    valid = bool(hf_token.value) and bool(hf_user.value) and bool(hf_model.value)
    mo.md(
        ("🎯 push target: **[`" + hf_user.value + "/" + hf_model.value + "`](https://huggingface.co/" + hf_user.value + "/" + hf_model.value + ")** — set a write token (https://huggingface.co/settings/tokens) for the push cell.")
        if valid
        else "⤴️ fill **token + username + model name** above — the push cell at the bottom activates automatically."
    )
    return


@app.cell
def _(mo, steps_ui, batch_ui):
    mo.md("## 2 · Training config")
    return


@app.cell
def _(mo):
    langs = mo.ui.multiselect(options=["bn", "en", "ru"], value=["bn", "en", "ru"], label="languages")
    mb = mo.ui.slider(10, 20000, value=200, label="MB corpus / language")
    steps_ui = mo.ui.slider(100, 200_000, value=200, step=100, label="train steps")
    batch_ui = mo.ui.slider(16, 2048, value=16, label="batch size (96GB: go 1024+)")
    compile_on = mo.ui.switch(value=False, label="torch.compile (CUDA only)")
    mo.vstack([langs, mb, steps_ui, batch_ui, compile_on])
    return batch_ui, compile_on, langs, mb, steps_ui


@app.cell
def _(langs, load, mb, repo_root):
    import time as _time
    from pathlib import Path as _Path

    _fetch = load("scripts.fetch_corpora").fetch
    corpora = _Path(repo_root) / "corpora"
    corpora.mkdir(exist_ok=True)
    budget = mb.value * (1 << 20)
    sizes, secs = {}, {}
    for lang in langs.value:
        tgt = corpora / f"{lang}.raw.txt"
        if not tgt.exists() or tgt.stat().st_size < budget:
            t_f = _time.time()
            n = _fetch(lang, tgt, budget)
            secs[lang] = round(_time.time() - t_f, 1)
        else:
            n = tgt.stat().st_size
            secs[lang] = 0.0
        sizes[lang] = round(n / (1 << 20), 1)
    (sizes, secs)
    return (corpora, sizes, secs,)


@app.cell
def _(mo, secs, sizes):
    import matplotlib.pyplot as _plt

    _langs = ["bn", "en", "ru"]
    _colors = {"bn": "#16A34A", "en": "#2563EB", "ru": "#DC2626"}
    fig, ax = _plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].bar(_langs, [sizes.get(l, 0) for l in _langs], color=[_colors[l] for l in _langs])
    ax[0].set_title("corpus MB (CC-100)")
    ax[0].set_ylabel("MB")
    ax[1].bar(_langs, [secs.get(l, 0) for l in _langs], color=[_colors[l] for l in _langs])
    ax[1].set_title("download seconds")
    ax[1].set_ylabel("s")
    _txt = " | ".join(f"{l}: {sizes.get(l, 0):,.1f} MB in {secs.get(l, 0):,.1f}s" for l in _langs)
    mo.md(f"📊 **Data report** — {_txt}")
    fig
    return (fig,)


@app.cell
def _(corpora, langs, repo_root):
    import os as _os
    import subprocess as _subprocess
    import sys as _sys
    import time as _time

    t_c = _time.time()
    _subprocess.run(
        [_sys.executable, _os.path.join(repo_root, "scripts", "clean_corpus.py"),
         "--raw-dir", str(corpora), "--out-dir", str(corpora),
         "--langs", ",".join(langs.value), "--mb", "500"],
        capture_output=True, text=True, cwd=repo_root,
    )
    n_lines = sum(
        sum(1 for _ in open(corpora / f"{lg}.txt", encoding="utf-8"))
        for lg in langs.value if (corpora / f"{lg}.txt").exists()
    )
    f"clean corpus: {n_lines:,} lines in {_time.time() - t_c:.1f}s"
    return (n_lines,)


@app.cell
def _(corpora, load, mo, repo_root):
    from pathlib import Path as _Path

    _co = load("craftly_ocr")
    _synthetic = load("craftly_ocr.data.synthetic")
    ds = _synthetic.SyntheticLineDataset(
        _co.build_charset(), corpora_dir=str(corpora),
        fonts_dir=str(_Path(repo_root) / "fonts"),
    )
    mo.md(f"render dataset ready: **{len(ds.lines):,}** unique lines, fonts bundled in repo ✅")
    return (ds,)


@app.cell
def _(mo):
    mo.md("## 3 · Train (stage-0 CTC reader, max-GPU mode)")
    return


@app.cell
def _(batch_ui, compile_on, ds, load, mo, steps_ui):
    import time as _time

    import torch as _torch

    _config = load("craftly_ocr.config")
    _model_mod = load("craftly_ocr.model")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
    _torch.backends.cudnn.benchmark = True

    cfg = _config.default_config()
    cfg.attn_blocks = 2
    cfg.encoder_depths = (1, 1, 1, 1, 1)
    model = _model_mod.CraftlyOCR(cfg, ds.charset).to(device)
    model = model.to(memory_format=_torch.channels_last)
    if compile_on.value and device == "cuda":
        model.reader = _torch.compile(model.reader)
    opt = _torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.05, fused=device == "cuda"
    )
    loader = _torch.utils.data.DataLoader(
        ds, batch_size=min(batch_ui.value, 16 if device == "cpu" else 1024),
        num_workers=0 if device == "cpu" else 8, shuffle=True, pin_memory=device == "cuda",
        persistent_workers=device == "cuda", prefetch_factor=4 if device == "cuda" else None,
        collate_fn=load("craftly_ocr.data.synthetic").collate_lines,
    )
    amp_dtype = _torch.bfloat16 if (device == "cuda" and _torch.cuda.is_bf16_supported()) else _torch.float16
    scaler = _torch.amp.GradScaler(device, enabled=amp_dtype == _torch.float16)

    model.train()
    it, loss_hist, t0 = iter(loader), [], _time.time()
    for step in range(steps_ui.value):
        try:
            px_b, ids_b, _lens = next(it)
        except StopIteration:
            it = iter(loader)
            px_b, ids_b, _lens = next(it)
        px_b = px_b.to(device, non_blocking=True)
        with _torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
            loss = model.reader.loss(model.reader(px_b), ids_b, [len(s) for s in ids_b])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        _torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss_hist.append(loss.item())
        if (step + 1) % 50 == 0 or step == steps_ui.value - 1:
            _el = _time.time() - t0
            _its = (step + 1) / _el
            print(f"step {step + 1}/{steps_ui.value} loss={loss.item():.3f} "
                  f"it/s={_its:.1f} eta={_el / (step + 1) * (steps_ui.value - step - 1) / 60:.1f} min")
    wall = _time.time() - t0
    its = steps_ui.value / max(wall, 1e-6)
    n_full = 200_000
    eta_gpu = n_full / max(its * (1024 / min(batch_ui.value, 1024)), 1e-6) / 3600 if device == "cuda" else None
    summary = {
        "device": device,
        "steps": steps_ui.value,
        "loss_first": round(loss_hist[0], 2),
        "loss_last": round(loss_hist[-1], 2),
        "mean_last_100": round(sum(loss_hist[-100:]) / max(len(loss_hist[-100:]), 1), 2),
        "steps_per_s": round(its, 2),
        "eta_full_200k_gpu_hours": round(eta_gpu, 1) if eta_gpu else f"cpu-only: measured {its:.1f} it/s → 200k steps ≈ {n_full / (its * 3600):,.0f} h — run on the RTX Pro 6000",
    }
    summary
    return (device, model, summary,)


@app.cell
def _(mo, summary):
    ok = summary["loss_last"] < summary["loss_first"]
    mo.md(
        ("✅ loss fell " if ok else "⚠️ loss not falling — raise steps: ") +
        f"{summary['loss_first']} → {summary['loss_last']} @ {summary['steps_per_s']} it/s\n\n"
        f"⏱️ {summary['eta_full_200k_gpu_hours']}"
    )
    return


@app.cell
def _(mo):
    mo.md("## 4 · Export checkpoint (safetensors + config + model card)")
    return


@app.cell
def _(ds, device, model, repo_root, summary):
    import json as _json
    from pathlib import Path as _Path

    from safetensors.torch import save_file as _save

    export_dir = _Path(repo_root) / "export" / "craftly-ocr"
    export_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    sd.pop("decoder.lm_head.weight", None)
    _save(sd, str(export_dir / "model.safetensors"))
    (export_dir / "config.json").write_text(_json.dumps(summary, indent=2))
    readme = f"""---
license: apache-2.0
language:
- bn
- en
- ru
tags:
- ocr
- vision
- multilingual
library_name: craftly_ocr
pipeline_tag: image-to-text
---

# Craftly OCR — TextLine Reader (stage 0)

Mirage-inspired micro VLM OCR for **Bangla / English / Russian** (210+ scripts by design).
Multi-script CTC reader over a 39.6k Unicode grapheme charset, HConvGLU encoder.

## Usage
```python
# pip install craftly-ocr torch safetensors
from craftly_ocr import default_config, build_charset
from craftly_ocr.model import CraftlyOCR
from safetensors.torch import load_file

cfg = default_config(); cfg.attn_blocks = 2; cfg.encoder_depths = (1, 1, 1, 1, 1)
model = CraftlyOCR(cfg, build_charset())
model.load_state_dict(load_file("model.safetensors"), strict=False)
model.eval()
```

Training data: CC-100 (bn/en/ru) via synthetic render. Stage-0 checkpoint —
line-level CTC reader. Later stages add detection + decoder.
"""
    (export_dir / "README.md").write_text(readme)
    export_dir = str(export_dir)
    export_dir
    return (export_dir,)


@app.cell
def _(mo):
    mo.md("## 5 · Push to Hugging Face Hub")
    return


@app.cell
def _(export_dir, hf_model, hf_token, hf_user, mo, private, summary):
    push_status = "🔒 push cell locked — fill token + username + model name in section 1"
    if hf_token.value and hf_user.value and hf_model.value:
        try:
            from huggingface_hub import HfApi
            who = HfApi(token=hf_token.value).whoami()
            if who.get("name") != hf_user.value:
                push_status = f"⚠️ token belongs to `{who.get('name')}`, but username says `{hf_user.value}` — use the matching username"
            else:
                api = HfApi(token=hf_token.value)
                repo_id = f"{hf_user.value}/{hf_model.value}"
                url = api.create_repo(repo_id, private=private.value, exist_ok=True)
                api.upload_folder(
                    repo_id=repo_id, folder_path=export_dir,
                    commit_message=f"Craftly OCR stage-0 ({summary['steps']} steps, loss {summary['loss_last']})",
                )
                push_status = f"🎉 pushed! **[https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})**"
        except Exception as e:
            push_status = f"❌ push failed: `{type(e).__name__}: {e}`"
    mo.md(push_status)
    return (push_status,)


@app.cell
def _(mo, push_status):
    mo.md(
        "### Last mile\n"
        + push_status
        + "\n\n**You can always re-run**: bigger corpus slider → more steps → cleaner loss → re-push overwrites the repo revision history. *We will win.* 🏁"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
