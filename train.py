import json
import os
import site

import numpy as np
import tensorflow as tf

from config import (
    BATCH_SIZE,
    DATA_DIR,
    EMBED_DIM,
    EMOJI_LABELS,
    EPOCHS,
    FF_DIM,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    MODEL_DIR,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_HEADS,
    VOCAB_SIZE,
)
from data.loader import load_tweet_eval
from data.preprocessor import preprocess_dataframe
from data.vocab import build_vocab, encode_dataset, save_vocab, vocab_coverage
from model.transformer import Transformer
from utils.callbacks import get_callbacks
from utils.class_weights import compute_and_save_class_weights, print_class_weights

# Point XLA to the CUDA toolkit (libdevice + ptxas) shipped by nvidia-cuda-nvcc
_sp = (
    site.getsitepackages()[0] if site.getsitepackages() else site.getusersitepackages()
)
_cuda_dir = os.path.join(_sp, "nvidia", "cuda_nvcc")
if os.path.isdir(_cuda_dir):
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_cuda_dir}"


# reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# GPU configuration
def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f" {gpu.name}")
    else:
        print("  No GPU found — running on CPU.")
        print("  Training will be slower but functionally identical.")


# data pipeline
def build_data_pipeline():
    print("Loading Data...")
    train_df, val_df, test_df = load_tweet_eval()

    print("Preprocessing Text...")
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)
    print("  Preprocessing complete")

    print("Building Vocab...")
    word2idx = build_vocab(train_df)
    save_vocab(word2idx)

    print("Encoding Datasets...")
    X_train = encode_dataset(train_df, word2idx)
    X_val = encode_dataset(val_df, word2idx)
    X_test = encode_dataset(test_df, word2idx)

    y_train = np.asarray(train_df["label"].to_numpy(), dtype=np.int64)
    y_val = np.asarray(val_df["label"].to_numpy(), dtype=np.int64)
    y_test = np.asarray(test_df["label"].to_numpy(), dtype=np.int64)

    print("\n  Vocabulary Coverage:")
    vocab_coverage(train_df, word2idx)
    vocab_coverage(val_df, word2idx)
    vocab_coverage(test_df, word2idx)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), np.asarray(y_train, dtype=np.int64))
    np.save(os.path.join(DATA_DIR, "y_val.npy"), np.asarray(y_val, dtype=np.int64))
    np.save(os.path.join(DATA_DIR, "y_test.npy"), np.asarray(y_test, dtype=np.int64))
    print(f"\n  Arrays saved → {DATA_DIR}/")

    return X_train, X_val, X_test, y_train, y_val, y_test, word2idx


def load_cached_data():
    print("\n  Loading cached data arrays...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.asarray(np.load(os.path.join(DATA_DIR, "y_train.npy")), dtype=np.int64)
    y_val = np.asarray(np.load(os.path.join(DATA_DIR, "y_val.npy")), dtype=np.int64)
    y_test = np.asarray(np.load(os.path.join(DATA_DIR, "y_test.npy")), dtype=np.int64)

    with open(os.path.join(DATA_DIR, "word2idx.json"), "r") as f:
        word2idx = json.load(f)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        word2idx,
    )


# model builder
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim: int, warmup_steps: int = 4000):
        super().__init__()
        self.embed_dim = tf.cast(embed_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "embed_dim": int(self.embed_dim.numpy()),
            "warmup_steps": self.warmup_steps,
        }


def build_model() -> tf.keras.Model:

    print("Building Model...")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        num_classes=NUM_CLASSES,
    )

    schedule = WarmupSchedule(embed_dim=EMBED_DIM, warmup_steps=500)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=schedule,
            beta_1=0.9,  # momentum term
            beta_2=0.98,  # squared gradient term
            epsilon=1e-9,  # numerical statbility
            weight_decay  = 1e-4
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    model.build_and_summarise()

    return model


# training loop
def train(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_dict: dict,
) -> tf.keras.callbacks.History:

    print("Training...")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Max epochs   : {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples  : {len(X_val):,}")
    print()

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=get_callbacks(),
        verbose=1,  # prints a progress bar per epoch
    )

    return history


# save final model
def save_model(model: tf.keras.Model, history: tf.keras.callbacks.History) -> None:

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(model_path)
    print(f"\n  Full model saved   → {model_path}")

    # training summary
    hist = history.history
    summary = {
        "final_train_loss": float(hist["loss"][-1]),
        "final_train_accuracy": float(hist["accuracy"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "final_val_accuracy": float(hist["val_accuracy"][-1]),
        "best_val_accuracy": float(max(hist["val_accuracy"])),
        "best_epoch": int(np.argmax(hist["val_accuracy"])) + 1,
        "total_epochs_run": len(hist["loss"]),
        "config": {
            "VOCAB_SIZE": VOCAB_SIZE,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "EMBED_DIM": EMBED_DIM,
            "NUM_HEADS": NUM_HEADS,
            "FF_DIM": FF_DIM,
            "NUM_BLOCKS": NUM_BLOCKS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
        },
    }

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Training summary   → {summary_path}")
    print(f"\n  Best val accuracy  : {summary['best_val_accuracy']:.4f}")


# MAIN

if __name__ == "__main__":
    print("  EMOJI TRANSFORMER — Training Pipeline")

    configure_gpu()

    USE_CACHE = os.path.exists(os.path.join(DATA_DIR, "X_train.npy"))

    if USE_CACHE:
        print("\n  Cached data found — loading from disk.")
        X_train, X_val, X_test, y_train, y_val, y_test, word2idx = load_cached_data()
    else:
        print("\n  No cache found — running full data pipeline.")
        X_train, X_val, X_test, y_train, y_val, y_test, word2idx = build_data_pipeline()

    print("Computing Class Weights...")
    class_weight_dict = compute_and_save_class_weights(y_train)
    print_class_weights(class_weight_dict, EMOJI_LABELS)

    model = build_model()

    history = train(model, X_train, y_train, X_val, y_val, class_weight_dict)

    save_model(model, history)

    print("  Training complete!")
    print(f"  Weights → {MODEL_DIR}/best_weights.weights.h5")
    print(f"  Model   → {MODEL_DIR}/final_model.keras")
    print(f"  Logs    → {MODEL_DIR}/logs/")
