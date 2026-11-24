# 👁️ Piku OCR

**Bangla · English · Russian OCR — on-device fast, deeply rebranded, fully open.**

Piku OCR is a multilingual OCR toolkit built around the proven PikuOCR
dynasty of models (text detection + recognition + direction classify +
document parsing), with **bn / en / ru as first-class languages**.

> **Provenance:** Piku OCR is a deep rebrand of
> [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0).
> The full training engine, architectures, configs and deployment tooling
> originate upstream — see [NOTICE](NOTICE) and [LICENSE](LICENSE).

## Why Piku OCR

- **bn / en / ru first-class** — dedicated dicts, corpus tooling, fine-tune
  recipes for Bangla, English and Russian
- **Micro to server** — mobile models run on-device in a few MB; server
  models rail the full accuracy curve
- **Complete pipeline** — text detection → cropping → recognition →
  post-processing; plus layout, table, formula and doc-VL pipelines
- **One-command inference & export** — Python API, CLI, PNG→TXT in one line

## Quickstart (inference)

```bash
pip install -e .
pikuocr ocr -i your_image.png
```

```python
from pikuocr import PikuOCR

pipeline = PikuOCR(device="cpu")
result = pipeline.predict("your_image.png")
result.print()
result.save_to_img("output")
```

## Training (full pipeline included)

```bash
# single-GPU training (works on your RTX Pro 6000 96GB)
python tools/train.py -c configs/rec/PikuOCRv4/rec_v4_seed_bangla.yml

# eval
python tools/eval.py -c configs/rec/... -o Global.checkpoints=./output/...

# export to on-device inference model
python tools/export_model.py -c configs/rec/... -o Global.pretrained_model=./output/...
```

Configs for detection / recognition / classication across mobile + server
variants live under `configs/`. dicts for bn/en/ru live in
`pkocr/utils/dict/`.

## Synthetic data factory (bn/en/ru)

`piku_data/` contains the original Piku data tooling used to build
fine-tune datasets from CC-100 corpora (streaming fetch → script-filter →
NFC dedup → font-rendered line images with Noto Bengali/Sans).

```bash
python -m piku_data.corpus --help   # streaming corpus fetch + clean
```

## Layout

| path | what |
|---|---|
| `pikuocr/` | Python package: pipelines, CLI, model zoo glue |
| `pkocr/` | training core: data, modeling, losses, optimizer, metrics |
| `pkstructure/` | document parsing (layout · table · KIE · recovery) |
| `tools/` | train.py · eval.py · export_model.py + infer tools |
| `configs/` | all training configs (det / rec / cls, mobile & server) |
| `deploy/` | server, C++, Jetson, android, ios deployment |
| `piku_data/` | CC-100 fetch + synthetic renderer for bn/en/ru |
| `fonts/` | Noto Sans + Noto Sans Bengali for the renderer |

## Governance

- License: Apache-2.0 — rebrand permitted; upstream attribution retained in [NOTICE](NOTICE)
- Upstream project: https://github.com/PaddlePaddle/PaddleOCR

*We will win.* 🏁
