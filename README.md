# Piku OCR 👁️🧠

Accurate, two-model OCR built from scratch for **Bangla 🇧🇩 + English 🇬🇧 + Russian 🇷🇺**.
No inherited architecture — designed clean, trained on one RTX Pro 6000 96GB,
budgeted at **2 hours per day** with resumable checkpoints.

## The two models

| Model | Role | Size | License |
|---|---|---|---|
| **Piku-Eye** 👁️ | reads a line image (3×64×512) → chars via CTC over a compact 457-glyph bn/en/ru charset | ~6.5M | Apache-2.0 |
| **Piku-Brain** 🧠 | tiny full-attention denoiser that fixes reader errors (char-id → char-id) | ~2.5M | Apache-2.0 |

Why two models: with a daily 2h budget, the eye converges in a few sessions,
and the brain adds the accuracy layer plain readers miss (error-correction
decoder at 2.5M params).

## New-architecture highlights (Piku-Eye)

- **Narrow-band attention**: a sliding width-window of 33 anchors per query
  (padded band attention over the time axis) — long-range bangla conjunct
  clusters without full quadratic attention
- **GRN residual stem** (ConvNeXt-V2-style global response norm on the stem)
- Flat token row (T=128) with sinusoidal positions; 8-layer pre-LN encoder
- **Compact charset** (457 glyphs vs region-wide ~40k) → sharper per-glyph
  decision boundaries, faster CTC

## Daily protocol (the only commands you need)

```bash
# Day 0 (once): data  — 2,000 MB per language is plenty for stage 1-2
python scripts/fetch_piku.py --langs bn,en,ru --mb 2000

# every day: one 2h eye session (auto-resumes from ckpt/eye_last.pt)
python -m piku_ocr.train_eye --minutes 120

# after a few eye sessions: one 2h brain session
python -m piku_ocr.train_brain --minutes 120

# anytime: export both to a single safetensors bundle
python -m piku_ocr.export_piku --out export/piku-ocr
```

All max-performance settings are pre-fixed in `piku_ocr/config.py`
(batch 1024, bf16, TF32, torch.compile, fused AdamW, 8 workers) —
no CLI guessing needed on the 96GB box.

## Fixed hardware profile (RTX Pro 6000, 96GB)

- bf16 autocast + TF32 matmul + cuDNN benchmark
- `torch.compile` on the eye trainer (disable with `--no-compile`)
- fused AdamW, batch 1024, 8 dataloader workers, prefetch 4
- checkpoints every 500 steps (eye) / 1000 (brain) + at session end

## Repository layout

```
piku_ocr/
  charset.py         bn/en/ru glyph builder (blank/pad/unk + 457 glyphs)
  config.py          PikuConfig + fixed max-perf settings
  model/eye.py       PikuEye (stem, band attention, CTC head)
  model/brain.py     PikuBrain (denoising corrector) + error simulator
  data/corpus.py     CC-100 streaming fetch + script-ratio cleaner
  data/synthetic.py  LineRenderer, EyeDataset, BrainDataset, collates
  train_eye.py       stage-1 trainer (2h/day, resumable)
  train_brain.py     stage-2 trainer (2h/day, resumable)
  infer.py           two-model inference: read_lines()
  export_piku.py     safetensors export (eye + brain bundle)
scripts/fetch_piku.py  data CLI
tests/test_piku.py     charset roundtrip, eye task learning, brain denoise
```

## Data

Synthetic-first (same proven recipe as modern OCR systems): clean text
corpora rendered with Noto fonts + blur/geometry augmentation means zero
labelling cost. Handwriting fine-tuning datasets (real-image stage 3) are
catalogued in [docs/DATASETS.md](docs/DATASETS.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Piku OCR is built from scratch; no
third-party model code or weights are included.
