# Training on RTX Pro 6000 (96 GB)

## Stages
| Stage | What trains | Data | Loss |
|---|---|---|---|
| 0 `reader` | encoder stems reused later + CTC reader | synthetic lines (all scripts) | CTC |
| 1 `readerft` | reader only, low LR | real line datasets (bn/en/ru) | CTC |
| 2 `det` | detector head (frozen encoder starts) | page images + boxes | DBNet WCE+Dice |
| 3 `decoder` | decoder joining reader/det results | image + prompt + JSON pairs | CE |
| 4 `e2e` | everything | mix of 0–3 | weighted sum |

## Data pipeline (bn / en / ru)
```bash
# 1. stream CC-100 corpora (budget per language; drop --mb for full downloads)
python scripts/fetch_corpora.py --langs bn,en,ru --mb 2000     # 2 GB each
# 2. clean: script filter, dedup, length filter -> corpora/<lang>.txt
python scripts/clean_corpus.py --langs bn,en,ru --mb 2000
# fonts for the three priority scripts are bundled in fonts/ (SIL OFL)

# 3. stage 0 — synthetic multi-script reader pretrain on your corpus
python -m craftly_ocr.train --stage reader --steps 200000 --batch 512 --corpora corpora/
```

## Commands
```bash
pip install -e ".[train]"
# stage 1 — real data fine-tune
python -m craftly_ocr.train --stage readerft --steps 40000 --batch 256 --data data/lines
# eval
python -m craftly_ocr.eval --ckpt craftly_stage0.pt --set data/lines/val
```

## 96 GB VRAM budget (stage 0, bf16)
- model 63.4M params × ~18 bytes (AdamW fp32 master + grads) ≈ 1.2 GB
- activations dominate: batch 512 × 512px crops fits in < 40 GB
- large headroom → use it for bigger crops (768px) or larger reader batch;
  accuracy on small/complex scripts scales with crop resolution

## Targets
- Bangla line CER < 3% (BanglaWriting-style eval)
- English scene CER < 3% (COCO-Text style)
- Russian CER < 3%
- on-device: < 100 ms per 512px page on flagship NPU-class hardware, < 50 MB at 4-bit
