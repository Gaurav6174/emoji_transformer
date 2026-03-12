import json
import os
import collections
import numpy as np
import pandas as pd
from config import VOCAB_SIZE, MAX_SEQ_LEN, DATA_DIR

#build it
def build_vocab(train_df: pd.DataFrame) -> dict:
    counter = collections.Counter()
    for sentence in train_df["clean_text"]:
        counter.update(sentence.split())
    
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        word2idx[word] = len(word2idx)
        
    return word2idx

#save and load
def save_vocab(word2idx: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "word2idx.json")
    with open(path, "w") as f:
        json.dump(word2idx, f)
    print(f"  Vocabulary saved → {path}")
    
def load_vocab() -> dict:
    path = os.path.join(DATA_DIR, "word2idx.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vocab file not found at '{path}'.\n"
            f"Run build_vocab() first to generate it."
        )
    with open(path, "r") as f:
        return json.load(f)
    
#encode
def encode_sentence(text :str, word2idx: dict)-> list[int]:
    tokens = text.split()[:MAX_SEQ_LEN]
    ids    = [word2idx.get(t, 1) for t in tokens]
    ids    += [0] * (MAX_SEQ_LEN - len(ids))
    return ids

def encode_dataset(df: pd.DataFrame, word2idx: dict) -> np.ndarray:
    encoded = [encode_sentence(text, word2idx) for text in df["clean_text"]]
    return np.array(encoded, dtype = np.int32)

#vocab coverage report
def vocab_coverage(df: pd.DataFrame, word2idx: dict) -> None:
    #what %ge of tokens in a given split r known,
    #useful for checking id vocab size is enough(>95% is good)

    total, known = 0, 0
    for sentence in df["clean_text"]:
        words = sentence.split()
        total += len(words)
        known += sum(1 for w in words if w in word2idx)

    print(f"  Vocab coverage : {100 * known / total:.2f}%  "
          f"({known:,} / {total:,} tokens recognised)")
         