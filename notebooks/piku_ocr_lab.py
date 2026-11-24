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
        """
        # 👁️🧠 Piku OCR — molab lab
        Two-model OCR for **Bangla / English / Russian**, built from scratch for the
        RTX Pro 6000 96 GB. This notebook runs the full daily protocol:
        **data → eye session → brain session → live demo → export → push to
        Hugging Face** (checkpoints auto-resume across sessions).
        """
    )
    return


@app.cell
def _(mo):
    """Bootstrap: fetch the Piku OCR repo into this notebook's working dir."""
    import glob as _glob
    import importlib as _importlib
    import io as _io
    import os as _os
    import sys as _sys
    import urllib.request as _request
    import zipfile as _zipfile

    _ref = "vibe/craftly-ocr-v01-d1254f"
    if not _os.path.exists("piku_ocr"):
        _resp = _request.urlopen(
            "https://api.github.com/repos/MD-Mushfiqur123/craftly-ocr/zipball/" + _ref
        )
        _zipfile.ZipFile(_io.BytesIO(_resp.read())).extractall(".")
        _os.chdir(sorted(_glob.glob("MD-Mushfiqur123-craftly-ocr-*"))[0])
    _sys.path.insert(0, _os.getcwd())
    repo_root = _os.getcwd()

    def load(name):
        return _importlib.import_module(name)

    mo.md(f"✅ Piku OCR repo ready: `{repo_root}`")
    return load, repo_root


@app.cell
def _(mo):
    mo.md("## 1 · Hugging Face identity (for the auto-push at the bottom)")
    return


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF token (write)", placeholder="hf_...")
    hf_user = mo.ui.text(label="HF username", placeholder="e.g. mushfiqur")
    hf_model = mo.ui.text(label="model repo name", value="piku-ocr")
    hf_private = mo.ui.switch(value=True, label="private repo")
    mo.vstack([hf_token, hf_user, hf_model, hf_private])
    return hf_model, hf_private, hf_token, hf_user


@app.cell
def _(hf_model, hf_private, hf_token, hf_user, mo):
    _valid = bool(hf_token.value) and bool(hf_user.value) and bool(hf_model.value)
    mo.md(
        (f"🎯 push target: **{hf_user.value}/{hf_model.value}**"
         f"{' (private)' if hf_private.value else ''}")
        if _valid else "⬆️ fill token + username above — export/push cell activates instantly"
    )
    return


@app.cell
def _(mo):
    mo.md("## 2 · Session settings")
    return


@app.cell
def _(mo):
    langs_p = mo.ui.multiselect(options=["bn", "en", "ru"], value=["bn", "en", "ru"], label="languages")
    mb_p = mo.ui.slider(50, 20000, value=1000, label="MB corpus / language")
    minutes_p = mo.ui.slider(5, 180, value=30, label="eye session minutes (2h/day protocol: 120)")
    brain_minutes_p = mo.ui.slider(5, 120, value=30, label="brain session minutes")
    compile_p = mo.ui.switch(value=True, label="torch.compile (CUDA)")
    run_eye_p = mo.ui.switch(value=True, label="run eye session now")
    run_brain_p = mo.ui.switch(value=True, label="run brain session now")
    mo.vstack([langs_p, mb_p, minutes_p, brain_minutes_p, compile_p, run_eye_p, run_brain_p])
    return brain_minutes_p, compile_p, langs_p, mb_p, minutes_p, run_brain_p, run_eye_p


@app.cell
def _(langs_p, load, mb_p, mo, repo_root):
    import time as _f_time
    from pathlib import Path as _f_p

    _corpora_mod = load("piku_ocr.data.corpus")
    corpora_p = _f_p(repo_root) / "corpora"
    corpora_p.mkdir(exist_ok=True)
    _budget = mb_p.value * (1 << 20)
    _stat = {}
    for _lang in langs_p.value:
        _raw, _clean = corpora_p / f"{_lang}.raw.txt", corpora_p / f"{_lang}.txt"
        _t0 = _f_time.time()
        if not _raw.exists() or _raw.stat().st_size < _budget:
            _corpora_mod.fetch(_lang, _raw, _budget)
        _t1 = _f_time.time()
        _n = _corpora_mod.clean_corpus(_raw, _clean, _lang, mb_p.value)
        _stat[_lang] = f"{_n:,} lines (fetch+clean {_t1 - _t0:.1f}s)"
    mo.md("📊 " + " · ".join(f"**{k}**: {v}" for k, v in _stat.items()))
    return (corpora_p,)


@app.cell
def _(corpora_p, load, mo, repo_root):
    from pathlib import Path as _d_p

    _charset = load("piku_ocr.charset").build_charset()
    _syn = load("piku_ocr.data.synthetic")
    eye_ds = _syn.EyeDataset(_charset, str(corpora_p), str(_d_p(repo_root) / "fonts"), cap=1_200_000)
    eye_stats = {"lines": len(eye_ds.lines), "charset": _charset.size}
    mo.md("🖼️ render dataset ready: **{}** lines · charset **{}** glyphs ✅".format(
        f"{eye_stats['lines']:,}", eye_stats["charset"]))
    return eye_ds, eye_stats


@app.cell
def _(corpora_p, eye_ds, load, mo):
    import matplotlib.pyplot as _plt
    from pathlib import Path as _v_p

    _syn = load("piku_ocr.data.synthetic")
    _render = _syn.LineRenderer(str(corpora_p.parent / "fonts"))
    _samples = [eye_ds.lines[min(i * max(1, len(eye_ds.lines) // 3), len(eye_ds.lines) - 1)] for i in range(3)]
    fig, ax = _plt.subplots(len(_samples), 1, figsize=(8, 1.4 * len(_samples)))
    if len(_samples) == 1:
        ax = [ax]
    for _a, (_lang, _text) in zip(ax, _samples):
        _a.imshow(_render.render(_text, 64, 512)[0], cmap="gray")
        _a.set_title(f"{_lang}: {_text[:48]}", fontsize=8)
        _a.axis("off")
    mo.md("## 3 · Data preview")
    return (fig,)


@app.cell
def _(mo):
    mo.md("## 4 · Train sessions (resumable — the daily 2h protocol)")
    return


@app.cell
def _(compile_p, eye_ds, hf_model, hf_token, hf_user, load, minutes_p, mo, repo_root, run_eye_p):
    import time as _time
    from pathlib import Path as _e_p

    import torch as _torch

    _util = load("piku_ocr.train_util")
    _syn = load("piku_ocr.data.synthetic")
    _cfg = load("piku_ocr.config").PikuConfig()
    _char = load("piku_ocr.charset").build_charset()

    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    _util.max_perf(_device)
    eye_model = load("piku_ocr.model.eye").PikuEye(_char, _cfg).to(_device)
    _ckpt = _e_p(repo_root) / "ckpt" / "eye_last.pt"
    if not _ckpt.exists() and hf_token.value and hf_user.value and hf_model.value:
        if _util.hf_ensure_ckpt(f"{hf_user.value}/{hf_model.value}", hf_token.value, "eye_last.pt", _ckpt):
            mo.output.append("☁️ pulled eye checkpoint from Hugging Face — resuming the daily streak!")
    _opt = _torch.optim.AdamW(eye_model.parameters(), lr=3e-4, weight_decay=0.01, fused=_device == "cuda")

    _gsteps = 0
    if _ckpt.exists():
        _gsteps = _util.load_checkpoint(_ckpt, eye_model, _opt)[0]
        mo.output.append(f"↩️ resumed eye at global step {_gsteps:,}")

    def _probe(b):
        _px = _torch.rand(b, 3, 64, 512, device=_device)
        eye_model.loss(_px, [[4, 5, 6, 7] * 4] * b).backward()
        eye_model.zero_grad(set_to_none=True)

    if _device == "cuda":
        _batch, _peak = _util.auto_vram_batch(_probe, _device)
        mo.output.append(f"🔧 auto batch **{_batch:,}** (~{_peak / (1 << 30):.1f} GB @1024 → fills ~85% VRAM)")
        if compile_p.value:
            eye_model = _torch.compile(eye_model)
    else:
        _batch = 16
        mo.output.append("(cpu mode — batch 16; attach a GPU in specs for the monster run)")

    _loader = _torch.utils.data.DataLoader(
        eye_ds, batch_size=_batch, shuffle=True,
        num_workers=8 if _device == "cuda" else 0,
        pin_memory=_device == "cuda", persistent_workers=_device == "cuda",
        prefetch_factor=4 if _device == "cuda" else None,
        collate_fn=_syn.eye_collate,
    )
    _amp = _device == "cuda"
    eye_model.train()
    _it = iter(_loader)
    _t0 = _time.time()
    _deadline = _t0 + (minutes_p.value if run_eye_p.value else 0.01) * 60
    session_steps, _last = 0, 0.0
    _hist = []
    try:
        while _time.time() < _deadline:
            try:
                px, ids = next(_it)
            except StopIteration:
                _it = iter(_loader)
                px, ids = next(_it)
            px = px.to(_device, non_blocking=True)
            with _torch.autocast(device_type=_device, dtype=_torch.bfloat16, enabled=_amp):
                loss = eye_model.loss(px, ids)
            _opt.zero_grad(set_to_none=True)
            loss.backward()
            _torch.nn.utils.clip_grad_norm_(eye_model.parameters(), 1.0)
            _opt.step()
            _hist.append(loss.item())
            session_steps += 1
            _now = _time.time()
            if _now - _last > 30:
                _last = _now
                mo.output.append(
                    f"step **{_gsteps + session_steps:,}** · loss **{_hist[-1]:.3f}** · "
                    f"{session_steps / max(_now - _t0, 1e-6):.1f} it/s · left {(_deadline - _now) / 60:.0f} min"
                )
            if session_steps % 500 == 0:
                _util.save_checkpoint(_ckpt, eye_model, _opt, None, _gsteps + session_steps)
    except KeyboardInterrupt:
        mo.output.append("⏸️ interrupted — saving…")

    eye_summary = {
        "device": _device,
        "global_steps": _gsteps + session_steps,
        "session_steps": session_steps,
        "batch": _batch,
        "first_loss": round(_hist[0], 2) if _hist else None,
        "last_loss": round(_hist[-1], 3) if _hist else None,
        "minutes": round((_time.time() - _t0) / 60, 1),
    }
    _util.save_checkpoint(_ckpt, eye_model, _opt, None, _gsteps + session_steps)
    _ok = _hist and _hist[-1] < _hist[0]
    mo.output.append(
        ("✅ eye session: loss " if _ok else "⚠️ eye session: ")
        + (f"{_hist[0]:.2f} → {_hist[-1]:.2f}" if _hist else "(no steps)")
    )
    return eye_model, eye_summary


@app.cell
def _(brain_minutes_p, corpora_p, hf_model, hf_token, hf_user, load, mo, repo_root, run_brain_p):
    import time as _b_time
    from pathlib import Path as _b_p

    import torch as _b_torch

    _util = load("piku_ocr.train_util")
    _syn = load("piku_ocr.data.synthetic")
    _b_cfg = load("piku_ocr.config").PikuConfig()
    _b_char = load("piku_ocr.charset").build_charset()

    _b_device = "cuda" if _b_torch.cuda.is_available() else "cpu"
    _util.max_perf(_b_device)
    brain_model = load("piku_ocr.model.brain").PikuBrain(_b_char, _b_cfg).to(_b_device)
    _b_ckpt = _b_p(repo_root) / "ckpt" / "brain_last.pt"
    if not _b_ckpt.exists() and hf_token.value and hf_user.value and hf_model.value:
        if _util.hf_ensure_ckpt(f"{hf_user.value}/{hf_model.value}", hf_token.value, "brain_last.pt", _b_ckpt):
            mo.output.append("☁️ pulled brain checkpoint from Hugging Face — resuming!")
    _b_opt = _b_torch.optim.AdamW(brain_model.parameters(), lr=3e-4, weight_decay=0.01, fused=_b_device == "cuda")

    _b_steps0 = 0
    if _b_ckpt.exists():
        _b_steps0 = _util.load_checkpoint(_b_ckpt, brain_model, _b_opt)[0]
        mo.output.append(f"↩️ resumed brain at global step {_b_steps0:,}")

    _b_ds = _syn.BrainDataset(_b_char, str(corpora_p), cap=1_200_000)
    if not _b_ds.lines:
        mo.output.append("⚠️ no corpus — run the data cell first")
        brain_summary = {"device": _b_device, "global_steps": _b_steps0, "session_steps": 0, "last_loss": None}
    else:
        if _b_device == "cuda":
            mo.output.append("🧠 brain: full-attention denoiser · batch **1024** (~2.5M params — light but wide)")
        _b_loader = _b_torch.utils.data.DataLoader(
            _b_ds, batch_size=1024 if _b_device == "cuda" else 16, shuffle=True,
            num_workers=4 if _b_device == "cuda" else 0,
            pin_memory=_b_device == "cuda", persistent_workers=_b_device == "cuda",
            prefetch_factor=4 if _b_device == "cuda" else None,
            collate_fn=_syn.brain_collate,
        )
        brain_model.train()
        _b_it, _b_hist = iter(_b_loader), []
        _b_t0 = _b_time.time()
        _b_deadline = _b_t0 + (brain_minutes_p.value if run_brain_p.value else 0.01) * 60
        session_b, _b_last = 0, 0.0
        try:
            while _b_time.time() < _b_deadline:
                try:
                    noisy, clean = next(_b_it)
                except StopIteration:
                    _b_it = iter(_b_loader)
                    noisy, clean = next(_b_it)
                noisy = noisy.to(_b_device, non_blocking=True)
                loss_b = brain_model.loss(noisy, list(clean))
                _b_opt.zero_grad(set_to_none=True)
                loss_b.backward()
                _b_torch.nn.utils.clip_grad_norm_(brain_model.parameters(), 1.0)
                _b_opt.step()
                _b_hist.append(loss_b.item())
                session_b += 1
                _b_now = _b_time.time()
                if _b_now - _b_last > 30:
                    _b_last = _b_now
                    mo.output.append(
                        f"brain step **{_b_steps0 + session_b:,}** · loss **{_b_hist[-1]:.3f}** · "
                        f"left {(_b_deadline - _b_now) / 60:.0f} min"
                    )
                if session_b % 1000 == 0:
                    _util.save_checkpoint(_b_ckpt, brain_model, _b_opt, None, _b_steps0 + session_b)
        except KeyboardInterrupt:
            mo.output.append("⏸️ interrupted — saving…")

        brain_summary = {
            "device": _b_device,
            "global_steps": _b_steps0 + session_b,
            "session_steps": session_b,
            "last_loss": round(_b_hist[-1], 3) if _b_hist else None,
        }
        _util.save_checkpoint(_b_ckpt, brain_model, _b_opt, None, _b_steps0 + session_b)
        mo.output.append(
            f"✅ brain session: loss → **{brain_summary['last_loss']}**"
            if _b_hist else "⚠️ brain idle"
        )
    return brain_model, brain_summary


@app.cell
def _(corpora_p, eye_ds, eye_model, load, mo):
    import torch as _r_torch

    _syn = load("piku_ocr.data.synthetic")
    _render = _syn.LineRenderer(str(corpora_p.parent / "fonts"))
    _pairs = []
    with _r_torch.no_grad():
        for _i in range(4):
            _lang, _text = eye_ds.lines[_i * 977 % max(1, len(eye_ds.lines))]
            _px = _render.render(_text, 64, 512).unsqueeze(0)
            _guess = eye_model.decode(_px)[0]
            _pairs.append((_lang, _text, _guess))
    _rows = "\n".join(
        f"| `{_t[:38]}` (`{_l}`) | `{_g[:38] if _g else '(…)'}` |" for _l, _t, _g in _pairs
    )
    mo.md(
        "## 5 · Live read demo (eye only — gets sharper every daily session)\n\n"
        "| truth | eye reads |\n|---|---|\n" + _rows
    )
    return


@app.cell
def _(mo):
    mo.md("## 6 · Export + push to Hugging Face 🚀")
    return


@app.cell
def _(brain_model, brain_summary, eye_model, eye_summary, hf_model, hf_private, hf_token, hf_user, load, mo, repo_root):
    import json as _x_json
    from pathlib import Path as _x_p

    from safetensors.torch import save_file as _x_save

    out_dir = _x_p(repo_root) / "export" / "piku-ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    _sd = {}
    for _k, _v in eye_model.state_dict().items():
        _sd["eye." + _k.replace("_orig_mod.", "")] = _v.contiguous().cpu()
    for _k, _v in brain_model.state_dict().items():
        _sd["brain." + _k.replace("_orig_mod.", "")] = _v.contiguous().cpu()
    _x_save(_sd, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(_x_json.dumps({
        "line_height": 64, "line_width": 512, "charset_size": 457,
        "languages": ["bn", "en", "ru"], "params": sum(v.numel() for v in _sd.values()),
    }, indent=2))
    _readme = (
        "---\nlicense: apache-2.0\nlanguage:\n- bn\n- en\n- ru\ntags:\n- ocr\n- vision\n---\n\n"
        "# Piku OCR 👁️🧠\n\nTwo-model OCR (Piku-Eye CTC reader + Piku-Brain corrector) built from scratch\n"
        "for **Bangla / English / Russian**. Training progress: "
        f"eye @ {eye_summary.get('global_steps', 0):,} steps, "
        f"brain @ {brain_summary.get('global_steps', 0):,} steps.\n"
    )
    (out_dir / "README.md").write_text(_readme)
    push_result = ""
    if hf_token.value and hf_user.value and hf_model.value:
        try:
            _api = load("huggingface_hub").HfApi(token=hf_token.value)
            _who = _api.whoami()
            if _who.get("name") != hf_user.value:
                push_result = f"⚠️ token belongs to `{_who.get('name')}` — write that username instead"
            else:
                _rid = f"{hf_user.value}/{hf_model.value}"
                _api.create_repo(_rid, private=hf_private.value, exist_ok=True)
                _api.upload_folder(repo_id=_rid, folder_path=str(out_dir), commit_message="Piku OCR progress push")
                for _name in ("eye_last.pt", "brain_last.pt"):
                    _f = _x_p(repo_root) / "ckpt" / _name
                    if _f.exists():
                        _api.upload_file(path_or_fileobj=str(_f), path_in_repo=f"ckpts/{_name}", repo_id=_rid, commit_message="session checkpoints")
                push_result = f"🎉 pushed → **https://huggingface.co/{_rid}** (model + checkpoints — next session auto-resumes from here)"
        except Exception as e_x:
            push_result = f"❌ {type(e_x).__name__}: {e_x}"
    else:
        push_result = "🔒 fill token + username in section 1 to push"
    mo.md(push_result)
    return out_dir, push_result


@app.cell
def _(brain_summary, eye_summary, mo, push_result):
    mo.md(
        "### Session report\n"
        f"- eye: **{eye_summary.get('session_steps', 0):,} steps this session** · "
        f"global **{eye_summary.get('global_steps', 0):,}** @ batch {eye_summary.get('batch')} · "
        f"device **{eye_summary.get('device')}** · loss {eye_summary.get('first_loss')} → {eye_summary.get('last_loss')}\n"
        f"- brain: **{brain_summary.get('session_steps', 0):,} steps** · "
        f"global **{brain_summary.get('global_steps', 0):,}** · loss {brain_summary.get('last_loss')}\n"
        f"- {push_result}\n\n"
        "*Daily protocol: re-run this notebook each day — checkpoints resume automatically. "
        "We will win.* 🏁"
    )
    return


if __name__ == "__main__":
    app.run()
