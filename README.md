---
title: Emoji Expression Predictor
emoji: 😂
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# 😂 Emoji Expression Predictor

A custom Transformer encoder trained on the [TweetEval](https://huggingface.co/datasets/tweet_eval) emoji dataset. Type a sentence — the model predicts which of 20 emojis best fits it, with a top-5 confidence breakdown.

---

## Pipeline

```
raw text → clean → tokenise → encode → Transformer → softmax → emoji
```

1. Text is lowercased; URLs → `@url`, mentions → `@user`, `#tags` stripped
2. Encoded against a 20k-word vocabulary (unknown tokens → `<UNK>`)
3. Passed through a 2-block Transformer encoder
4. Top predicted class mapped to one of 20 emoji labels

---

## Architecture

```
PositionalEmbedding      vocab=20 000 · dim=128 · max_len=50
EncoderBlock × 2
  MultiHeadAttention     4 heads × 32-dim
  LayerNorm + Residual
  FeedForwardNetwork     256-dim ReLU
  LayerNorm + Residual
GlobalAveragePooling1D
Dense(256, ReLU) + Dropout(0.3)
Dense(20, Softmax)
```

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 20 000 |
| Max sequence length | 50 |
| Embedding dim | 128 |
| Attention heads | 4 |
| FF dim | 256 |
| Encoder blocks | 2 |
| Dropout | 0.3 |
| Optimiser | Adam + Warmup LR |
| Batch size | 64 |
| Max epochs | 30 (early stopping) |

---

## Emoji Classes

❤️ 😍 😂 💕 🔥 😊 😎 ✨ 💙 😘 📷 🇺🇸 ☀️ 💜 😉 💯 😁 🎄 📸 😜

---

## Project Structure

```
emoji_transformer/
├── app.py                  # Streamlit dashboard
├── inference.py            # predict(text) → dict
├── config.py               # all hyperparameters
├── train.py                # training script
├── Dockerfile              # container definition
├── .dockerignore
├── .streamlit/
│   └── config.toml         # dark theme + headless server config
├── requirements.txt        # deployment dependencies
├── saved_model/
│   └── final_model.keras
├── saved_data/
│   └── word2idx.json
├── model/
│   ├── transformer.py
│   ├── encoder_block.py
│   ├── attention.py
│   └── positional_encoding.py
├── data/
│   ├── preprocessor.py
│   └── vocab.py
└── utils/
    ├── callbacks.py
    └── class_weights.py
```
---

## Inference Module (`inference.py`)

- Loads `saved_data/word2idx.json` and `saved_model/final_model.keras` once (lazy singleton)
- Re-declares `WarmupSchedule` locally to pass it via `custom_objects` without importing `train.py` side-effects
- Exposes a single `predict(text: str) -> dict` function returning:
  - `input_text`, `clean_text`, `emoji`, `label`, `confidence`, `top5`
- Forces CPU-only mode (`CUDA_VISIBLE_DEVICES=""`) before TensorFlow import to prevent XLA/ptxas crash on machines where CUDA toolkit binaries (`ptxas`, `nvlink`) are not on `PATH`

---

## Deployment

### Run Locally

```bash
# activate your venv first
venv/bin/streamlit run app.py
# → http://localhost:8501
```

### Run with Docker

```bash
docker build -t emoji-app .
docker run -p 7860:7860 emoji-app
# → http://localhost:7860
```

### Hugging Face Spaces

Push to a Space with **Docker** SDK selected. The Space will automatically:
1. Build the image using the `Dockerfile`
2. Serve on port `7860`

Required files must be present in the repo root:
- `saved_model/final_model.keras`
- `saved_data/word2idx.json`

---

