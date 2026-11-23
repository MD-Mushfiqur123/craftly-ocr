# Craftly OCR — Design

A Mirage-inspired **micro vision-language OCR model** for on-device text
reading in **210+ languages** (Bangla, English, Russian first).

## Research basis

| Work | What we take |
|---|---|
| **Mirage** (Cactus Compute, 2026, Apache-2.0) — 51.3M params, 22 MB: one pass gives scene/detect/OCR/segment/relations/decisions | Shared encoder; HConvGLU (fixed Walsh–Hadamard transforms around small banded convs, ~8x fewer weights than dense); 256 image cells; register tokens; a CTC text reader over crops; small decoder that only *formats* JSON while heads do perception; heavy 2-bit quantization |
| **GOT-OCR 2.0** (Wei et al., 2024) — 580M unified encoder–decoder, 1024px→256 tokens | High-compression encoder token budget; multi-stage training (synthetic pretrain → real fine-tune → joint); dynamic resolution |
| **TrOCR** (Li et al., 2021) | Line-image recognizer as encoder–decoder; we replace the decoder with CTC for on-device cost |
| **PARSeq** (Bautista & Atienza, CVPR 2022) | Permutation-aware decoding ideas for optional future accuracy head |
| **DBNet / DBNet++** (Liao et al., AAAI 2020/2022) | Differentiable binarization heatmap + offsets text detection |
| **ConvNeXtV2 GRN** (Liu et al., 2023) | Global response normalisation inside HConvGLU blocks |

## Composition

```
512px image
  └─ conv stem (stride 2) ─ 5 HConvGLU stages ──► 16×16 = 256 cells (d=384)
  │        └─ stride-8 fine map ──► DetectionHead (text/obj heat + boxes)
  ├─ OCRReader: crop/window → CTC over 39.6k multi-script graphemes
  └─ 12 GQA attention blocks + 4 register tokens → image memory
       └─ CraftlyDecoder (8 layers, d=512, Hadamard MLP):
          <task_ocr> <task_detect> <task_describe> <task_form> → JSON
```

**Parameter budget (v0.1):** encoder 10.1M · reader 13.0M · detector 0.2M ·
decoder 40.1M · **total 63.4M** → ~15 MB at 2-bit, ~32 MB at 4-bit.

## Reading 210+ languages

1. **One shared CTC reader, one shared charset** — a single Unicode-script
   inventory (~39.6k graphemes: Latin, Cyrillic, Greek, Arabic, Bangla,
   Devanagari, all Indic, Thai/Lao/Khmer/Myanmar, CJK BMP, Hangul, Ethiopic,
   …). No per-language heads.
2. **Synthetic bootstrap**: labelled data for low-resource scripts comes from
   `craftly_ocr.data.synthetic` — any corpus text (CC-100, NLLB dumps,
   Wikipedia) rendered over backgrounds with per-script Noto fonts +
   augmentation. Scale is unbounded and free.
3. **Script-priority curriculum**: Bangla → English → Russian head the
   stage-0 mix; the remaining 200+ languages enter as a long-tail replay mix
   so the priority three keep ≥ 40% of each batch.
4. **Decoder stays small** by *referencing* reader output (`text_refs`) in
   JSON instead of re-emitting raw text.

## Why not just fine-tune an existing VLM?

GOT-OCR (580M) and PaddleOCR-VL are excellent but heavy for phones; Mirage
has no public training code or data. Craftly OCR targets the Mirage point on
the curve (sub-100M, sub-100 ms on NPUs) with a fully open pipeline.
