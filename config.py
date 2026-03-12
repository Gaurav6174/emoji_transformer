#DATA
VOCAB_SIZE = 20_000
MAX_SEQ_LEN = 50
NUM_CLASSES = 20

#TRANSFORMER
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_BLOCKS = 2
DROPOUT_RATE = 0.3 

#TRAINIG
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

#PATHS
DATA_DIR = "saved_data"
MODEL_DIR = "saved_model"

EMOJI_LABELS = [
    "❤️",  "😍",  "😂",  "💕",  "🔥",
    "😊",  "😎",  "✨",  "💙",  "😘",
    "📷",  "🇺🇸", "☀️",  "💜",  "😉",
    "💯",  "😁",  "🎄",  "📸",  "😜"
]

