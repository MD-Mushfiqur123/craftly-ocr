# Dataset roadmap

## Stage 0 — synthetic (unbounded, no labelling cost)
- Render engine: `craftly_ocr/data/synthetic.py`
- Fonts: [Noto family](https://fonts.google.com/noto) per script — Noto Sans
  Bengali / Noto Sans (Cyrillic/Latin covers bn/en/ru). Linux: `apt install
  fonts-noto fonts-noto-core`
- Corpora (plain text, any size — verified sources):
  - **CC-100** (XLM-R monolingual corpus, 100+ languages incl. bn/en/ru):
    [HF: statmt/cc100](https://huggingface.co/datasets/statmt/cc100), files at
    `data.statmt.org/cc-100/`
  - **BdNC** — Bangladesh govt Bangla corpus, 40 GB / 3B words:
    [corpus.bangla.gov.bd](https://corpus.bangla.gov.bd/)
  - **OPUS** — free multilingual parallel corpora: [opus.nlpl.eu](https://opus.nlpl.eu/)
  - Bengali aggregate: [HF: munzurul/bangla-corpus](https://huggingface.co/datasets/munzurul/bangla-corpus) (CC-100-bn mirror, CC-BY-4.0)
- Augmentation: blur, rotation, shear, contrast, JPEG artifacts

## Stage 1 — real line-level recognition (bn / en / ru)
| Dataset | Language | Where | Notes |
|---|---|---|---|
| **BanglaWriting** | Bangla (handwritten) | [Mendeley r43wkvdk4w](https://data.mendeley.com/datasets/r43wkvdk4w/1) · [GitHub](https://github.com/QuwsarOhi/BanglaWriting) | 260 writers, 21,234 words / 32,787 chars, manual word boxes + unicode |
| **Ekush** | Bangla (handwritten chars) | [rabby.dev/ekush](https://rabby.dev/ekush/) | largest handwritten Bangla character set |
| **BN-HTRd** | Bangla (handwritten lines) | [Mendeley](https://data.mendeley.com/) (search BN-HTRd) | 788 pages / 150 writers / ~108k words, page+line+word level |
| **Cyrillic Handwriting Dataset** | Russian (handwritten) | [Kaggle](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) | ~73.8k text crops (v4), 95/5 train/test |
| **ICDAR 2019 MLT** | multi-script scene text (Cyrillic, Bangla? check split) | [Kaggle mirror](https://www.kaggle.com/datasets/zubairalibhutto/mlt-19-ocr-dataset) | standard detection+recognition benchmark |
| **COCO-Text v2 / TextOCR** | English (scene) | textvqa.org / ocrd.github.io refs | large natural-scene sets |
| **OCR datasets index** | all kinds | [xinke-wang/OCRDatasets](https://github.com/xinke-wang/OCRDatasets) | curated list: scene/doc/handwritten/historical |
| **awesome-bangla** | Bangla everything | [banglakit/awesome-bangla](https://github.com/banglakit/awesome-bangla) | tools + datasets catalogue |

> Rule: **re-verify license + script coverage before shipping any dataset**
> (the table is a verified-starting map, sizes quoted from the sources above).

## Stage 2 — page-level detection + structure
- ICDAR MLT page images (boxes), shop receipts (Bangladeshi market data —
  self-collected safest), photos from scanner apps
- Self-distillation: stage-0 reader over own photos → high-confidence
  agreement as pseudo-labels for the detector

## Stage 3 — structured JSON (decoder)
- Image + reader/detector outputs → JSON pairs (auto-generated)
- Manual correction batches (receipts, ID cards, signage)

## Low-resource tail (languages 4–210+)
- Synthetic-first with per-language CC-100 corpora (all in `statmt/cc100`)
- Per-language manifest in `data/langs/<iso3>.yaml`: source, license, size,
  CER-target
