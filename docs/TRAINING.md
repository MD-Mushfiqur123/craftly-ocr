# Piku OCR — Training guide (RTX Pro 6000 96GB, 2h/day)

## Day 0 — data (one time)
```bash
python scripts/fetch_piku.py --langs bn,en,ru --mb 2000
```
Shows per-language fetch MB + seconds (time accounting) and clean line counts.

## Daily session — eye (stage 1)
```bash
python -m piku_ocr.train_eye --minutes 120
```
- Auto-resumes from `ckpt/eye_last.pt`
- Live every minute: `step N loss=… it/s=… session_min_left=…`
- Stop-safe: checkpoints every 500 steps and at session end
- Fixed defaults: batch 1024, bf16, TF32, torch.compile, fused AdamW, 8 workers

## Daily session — brain (stage 2, after a few eye sessions)
```bash
python -m piku_ocr.train_brain --minutes 120
```

## Export / inference
```bash
python -m piku_ocr.export_piku --out export/piku-ocr
python - <<'PY'
import torch
from piku_ocr.data.synthetic import LineRenderer
from piku_ocr.infer import read_lines
px = torch.stack([LineRenderer("fonts").render("পাঠ প্রযুক্তি", 64, 512)])
print(read_lines(px, "ckpt/eye_last.pt", "ckpt/brain_last.pt"))
PY
```

## Targets
- Stage-0 synthetic eval: CER < 3% on held-out lines per language
- After handwriting fine-tune (stage 1): CER < 5% real data

## Expected GPU time (batch 1024, bf16)
- ~20-60 it/s measured on the 96GB box → each 2h session ≈ 150k+ steps
- Eye convergence typically 3-6 daily sessions; brain 1-2 sessions
