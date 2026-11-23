# Dataset roadmap

## Stage 0 — synthetic (unbounded, no labelling cost)
- Render engine: `craftly_ocr/data/synthetic.py`
- Fonts: Noto family per script (`fonts-noto` + Noto Sans Bengali / Noto Sans
  Cyrillic subsets)
- Corpora: CC-100 / NLLB-200 dumps / Wikipedia per language; demo corpus is
  built in (`bn`, `en`, `ru`)
- Augmentation: blur, rotation, shear, contrast, JPEG artifacts; later phases
  add textured backgrounds, layout fields, mixed-script lines

## Stage 1 — real line-level recognition ( Bangla / English / Russian focus )
| Dataset | Language(s) | Notes |
|---|---|---|
| ICDAR 2017 MLT | ~9–10 scripts incl. Cyrillic, Bangla? (check split) | scene-text line crops, standard benchmark |
| ICDAR 2019 MLT | multi-script | successor, includes more languages |
| COCO-Text v2 / TextOCR | English | large natural-scene text |
| SynthText, SynthAdd, MJSynth | English | synthetic pre-mixed sets (skip if stage 0 covers) |
| BanglaWriting | Bangla (handwritten) | paragraph setImage; handwritten difficulty |
| Ekush | Bangla (handwritten) | ~369k character crops; verify license |
| Cyrillic Handwriting Dataset | Russian (handwritten) | Kaggle; verify lineage |

> Rule of the project: **never ship a dataset without re-verifying its
> license and script coverage.** The table is a starting map, not a claim.

## Stage 2 — page-level detection + structure
- ICDAR MLT page images (boxes), shop receipts (Bangla Dhaka dataset? verify),
  document photos from scanner apps (self-collected, own data safest)
- Self-distillation: run the stage-0 reader over own photos, keep
  high-confidence agreements as pseudo-labels for the detector

## Stage 3 — structured JSON (decoder)
- Any image + its reader/detector outputs → JSON pairs (auto-generated)
- Manual correction batches for quality slips (receipts, ID cards, signage)

## Low-resource tail (languages 4–210+)
- Synthetic-first (stage 0 corpora per language)
- Community transcription sets once verified; keep a per-language manifest in
  `data/langs/<iso3>.yaml` with source, license, size, CER-target
