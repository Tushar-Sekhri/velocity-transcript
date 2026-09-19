"""
Velocity Transcript - Step 11
Loads the model trained purely on INCLUDE data, and evaluates it on
YOUR OWN recorded clips - a genuinely new signer/setup it has never
seen. This is the real answer to "will this work when I sign in
front of the webcam."

Usage:
    python evaluate_on_own.py
"""

import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from train_baseline import SimpleLSTMClassifier  # reuse the exact model class

DATA_DIR = "data/processed"
MODEL_PATH = "models/baseline_lstm.pt"
LABEL_MAP_PATH = "models/label_map.json"


def main():
    # ---- Load model config + label mapping ----
    with open(LABEL_MAP_PATH) as f:
        config = json.load(f)

    idx_to_label = {int(k): v for k, v in config["idx_to_label"].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    model = SimpleLSTMClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # ---- Load your own recordings ----
    X_own = np.load(f"{DATA_DIR}/X_own.npy")
    y_own_raw = np.load(f"{DATA_DIR}/y_own.npy")

    # Only evaluate on words the model actually knows (safety check)
    known_mask = np.array([label in label_to_idx for label in y_own_raw])
    unknown_words = set(y_own_raw[~known_mask].tolist())
    if unknown_words:
        print(f"Warning: skipping samples with unseen labels: {unknown_words}")

    X_own = X_own[known_mask]
    y_own_raw = y_own_raw[known_mask]
    y_own = np.array([label_to_idx[label] for label in y_own_raw])

    print(f"Evaluating on {len(X_own)} of your own recorded clips\n")

    X_tensor = torch.tensor(X_own, dtype=torch.float32)
    y_tensor = torch.tensor(y_own, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1)

    accuracy = (preds == y_tensor).float().mean().item()

    print(f"---- Domain-Gap Test Results (INCLUDE-trained model on YOUR clips) ----")
    print(f"Accuracy: {accuracy:.3f}\n")

    label_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(label_names)
    print(confusion_matrix(y_tensor.tolist(), preds.tolist(), labels=list(range(len(label_names)))))

    print("\nClassification Report:")
    print(classification_report(
        y_tensor.tolist(), preds.tolist(),
        labels=list(range(len(label_names))),
        target_names=label_names, zero_division=0
    ))


if __name__ == "__main__":
    main()