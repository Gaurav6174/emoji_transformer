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

## Bug Fixes

### `train.py`
- Fixed `tf.math.minium()` typo → `tf.math.minimum()` in `WarmupSchedule.__call__` (caused a crash on every training step)
- Fixed label arrays: `train_df["label"].values` (pandas ExtensionArray) → `np.asarray(..., dtype=np.int64)` to satisfy NumPy type constraints
- Aligned `Transformer.call()` signature with Keras `Model` base class (`inputs`, `training`, `mask`)
- Fixed `build_model()` return type: `Transformer` → `tf.keras.Model` to match compiled model type

### `model/attention.py`
- Fixed `mask: tf.Tensor = None` → `mask: Optional[tf.Tensor] = None` in both `SingleAttentionHead` and `MultiHeadAttention`
- Removed unused imports: `EMBED_DIM`, `NUM_HEADS`

### `model/encoder_block.py`
- Fixed `mask: tf.Tensor = None` → `mask: Optional[tf.Tensor] = None` in `EncoderBlock.call()`
- Added missing `return config` in `EncoderBlock.get_config()` (caused silent `None` return)
- Removed unused imports: `EMBED_DIM`, `NUM_HEADS`, `FF_DIM`

### `model/transformer.py`
- Fixed `InvalidArgumentError: required broadcastable shapes` crash in `GlobalAveragePooling1D`
  - Root cause: the 4D attention mask `(B,1,1,T)` was being passed directly to the pooling layer, which expects shape `(B,T)`
  - Fix: separated into two masks — `attn_mask` `(B,1,1,T)` for encoder blocks, `seq_mask` `(B,T)` for pooling
- Aligned `from_config()` signature with Keras base class

---

## Inference Module (`inference.py`)

- Loads `saved_data/word2idx.json` and `saved_model/final_model.keras` once (lazy singleton)
- Re-declares `WarmupSchedule` locally to pass it via `custom_objects` without importing `train.py` side-effects
- Exposes a single `predict(text: str) -> dict` function returning:
  - `input_text`, `clean_text`, `emoji`, `label`, `confidence`, `top5`
- Forces CPU-only mode (`CUDA_VISIBLE_DEVICES=""`) before TensorFlow import to prevent XLA/ptxas crash on machines where CUDA toolkit binaries (`ptxas`, `nvlink`) are not on `PATH`

---

## Streamlit App (`app.py`)

- Single-page dark-themed dashboard
- 5 clickable example sentence buttons that pre-fill the input
- Large emoji display (90px) with confidence percentage
- Top-5 predictions shown as labelled progress bars
- Cleaned text displayed so users can see preprocessing effects
- Raw prediction JSON in a collapsed expander
- Model loaded once via `@st.cache_resource`
- Graceful error handling for empty input and model load failures
- **Bug fixed:** text area used `value=` without `key=`, causing Streamlit to reset the widget on every rerun (user had to type input twice). Fixed by using `key="user_input"` so Streamlit manages widget state automatically

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

## Known Issues Resolved

| Issue | Cause | Fix |
|---|---|---|
| `tf.math.minium` crash | Typo in `WarmupSchedule` | Renamed to `tf.math.minimum` |
| Broadcastable shapes crash | Wrong mask shape passed to pooling | Separated `attn_mask` and `seq_mask` |
| XLA/ptxas core dump | GPU kernel compilation with missing CUDA binaries | `CUDA_VISIBLE_DEVICES=""` forces CPU |
| Text input reset on predict | `value=` re-evaluated on every Streamlit rerun | Replaced with `key=` |
| Model load failure | `WarmupSchedule` not in `custom_objects` | Re-declared in `inference.py` |

---

## License

MIT