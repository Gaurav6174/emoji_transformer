import json
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from config import NUM_CLASSES, DATA_DIR

def compute_and_save_class_weights(y_train: np.ndarray) -> dict:
    #weight_i = total_samples / (num_classes × count_i)
    
    weights_array = compute_class_weight(
        class_weight = "balanced",
        classes      = np.arange(NUM_CLASSES),
        y            = y_train
    )
    
    #convert to dict: model.fit() undrstnds
    class_weight_dict = {
        int(i): float(w)
        for i, w in enumerate(weights_array)
    }
    
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "class_weights.json")
    with open(path, "w") as f:
        json.dump(class_weight_dict, f, indent=2)

    print(f"  Class weights saved → {path}")
    
    return class_weight_dict

def load_class_weights() -> dict:
    
    path = os.path.join(DATA_DIR, "class_weights.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Class weights not found at '{path}'.\n"
            f"Run compute_and_save_class_weights() first."
        )
        
    with open(path, "r") as f:
        raw = json.load(f)
        
    return {int(k): v for k, v in raw.items()}

def print_class_weights(class_weight_dict: dict, emoji_labels: list) -> None:
    
    #summary  of class weights
    print("Class Weights Summary:")
    print(f"  {'Index':>6}  {'Emoji':>5}  {'Weight':>8}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*25}")
    
    for idx, weight in sorted(class_weight_dict.items()):
        bar   = "█" * int(weight * 8)
        emoji = emoji_labels[idx]
        print(f"  {idx:6d}  {emoji:>5}  {weight:8.4f}  {bar}")
        
    max_w = max(class_weight_dict.values())
    min_w = min(class_weight_dict.values())
    print(f"\n  Imbalance ratio : {max_w / min_w:.1f}x  "
          f"(highest weight / lowest weight)")
    
    