# Piku OCR v1 — Design notes

Fresh from-scratch design (no inherited code) for **bn / en / ru**, tuned for
one RTX Pro 6000 96GB and a daily 2-hour training budget.

## Research grounding
- GOT-OCR (arXiv 2409.01704): synthetic-corpus-first training, compact image
  tokenisation, encoder→decoder reading — motivates our Stage-0 recipe
- ConvNeXt-V2: GRN (global response norm) — used in Piku-Eye stem
- Error-correction / noisy-channel decoding (classic LM post-correction):
  motivates Piku-Brain, a tiny denoising transformer over reader outputs

## Pipeline
```
corpus (CC-100 bn/en/ru) --clean--> clean text lines
        --render (Noto fonts + aug)--> line images (3x64x512) ~ zero label cost
                     |
             [Piku-Eye] CTC reader  (457-glyph compact charset, T=128)
                     |  greedy decode
             reader-style errors
                     |
             [Piku-Brain] denoiser (char-id -> char-id, 4 layers, d=256)
                     |
             corrected text
```

## Why this wins on "accurate + good" per 2h/day
1. Line-level task + tiny charset -> CTC converges within a few 2h sessions
2. Band attention (33-anchor sliding lens) keeps cost ~O(T * band) instead of
   O(T^2) full attention, yet covers long Bangla conjunct clusters
3. Brain stage is cheap (~2.5M params) but lifts real-world accuracy the way
   noisy-channel rescuing does in ASR literature
4. Everything resumable: daily sessions add steps without losing optimiser state

## Roadmap (stages)
- Stage 0 (done, this repo): line reading + correction on synthetic data
- Stage 1: real handwriting fine-tune (BanglaWriting, Ekush, BN-HTRd, Cyrillic
  Handwriting — see docs/DATASETS.md)
- Stage 2: page segmentation model (line finding), then full-page composing
- Stage 3: HF Hub publishing + int8/quantised on-device export
