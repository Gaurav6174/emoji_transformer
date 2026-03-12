"""
Emoji Transformer inference pipeline.

Loads the trained model once and exposes a single predict() function.
WarmupSchedule is re-declared here (not imported from train.py) to avoid
executing training side-effects while still satisfying the custom_objects
requirement for model loading.
"""

import json
import os

#  Force CPU-only mode before TensorFlow is imported 
# Hides all GPU devices so TF never attempts XLA/ptxas GPU kernel compilation.
# For single-sample inference on a small model, CPU is equally fast and avoids
# the ptxas/nvlink crash that occurs when CUDA toolkit binaries are missing.
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress C++ INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN numeric warnings

import numpy as np
import tensorflow as tf

from config import DATA_DIR, EMOJI_LABELS, MODEL_DIR
from data.preprocessor import _clean
from data.vocab import encode_sentence
from model.attention import MultiHeadAttention, SingleAttentionHead
from model.encoder_block import EncoderBlock, FeedForwardNetwork
from model.positional_encoding import PositionalEmbedding
from model.transformer import Transformer


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    #Transformer learning-rate warm-up schedule (Vaswani et al. 2017).

    def __init__(self, embed_dim: int, warmup_steps: int = 4000):
        super().__init__()
        self.embed_dim = tf.cast(embed_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self) -> dict:
        return {
            "embed_dim": int(self.embed_dim.numpy()),
            "warmup_steps": self.warmup_steps,
        }


# All custom Keras objects required to deserialise the saved model.

_CUSTOM_OBJECTS: dict = {
    "PositionalEmbedding": PositionalEmbedding,
    "SingleAttentionHead": SingleAttentionHead,
    "MultiHeadAttention": MultiHeadAttention,
    "FeedForwardNetwork": FeedForwardNetwork,
    "EncoderBlock": EncoderBlock,
    "Transformer": Transformer,
    "WarmupSchedule": WarmupSchedule,
}


# Lazy singletons — loaded once, reused on every call to predict().

_model: tf.keras.Model | None = None
_word2idx: dict | None = None


def _load_resources() -> tuple[tf.keras.Model, dict]:
    global _model, _word2idx

    if _model is not None and _word2idx is not None:
        return _model, _word2idx  # already loaded

    vocab_path = os.path.join(DATA_DIR, "word2idx.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocab file not found at '{vocab_path}'.\n"
            "Make sure saved_data/word2idx.json exists."
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        _word2idx = json.load(f)

    model_path = os.path.join(MODEL_DIR, "final_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'.\n"
            "Make sure saved_model/final_model.keras exists."
        )
    _model = tf.keras.models.load_model(
        model_path,
        custom_objects=_CUSTOM_OBJECTS,
    )

    assert _model is not None and _word2idx is not None
    return _model, _word2idx


# Public API


def predict(text: str) -> dict:
    
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    model, word2idx = _load_resources()

    clean = _clean(text)

    encoded = encode_sentence(clean, word2idx)
    x = np.array([encoded], dtype=np.int32)  # shape: (1, MAX_SEQ_LEN)

    # forward pass 
    probs = model.predict(x, verbose=0)[0]  # shape: (NUM_CLASSES,)

    # decode results 
    label = int(np.argmax(probs))
    confidence = float(probs[label])

    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"emoji": EMOJI_LABELS[i], "confidence": float(probs[i])} for i in top5_indices
    ]

    return {
        "input_text": text,
        "clean_text": clean,
        "emoji": EMOJI_LABELS[label],
        "label": label,
        "confidence": confidence,
        "top5": top5,
    }
