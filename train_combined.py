"""
Velocity Transcript - Step 12
Trains on INCLUDE (train split) + a stratified portion of your own
recordings, then evaluates SEPARATELY on:
    (a) INCLUDE's held-out test cluster (unchanged from before)
    (b) a held-out portion of your own recordings (never used in training)

Keeping these two results separate (not blended) tells us clearly
whether adding your data helps generalization to you specifically,
without an average masking the answer.

Usage:
    python train_combined.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from train_baseline import SimpleLSTMClassifier, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

DATA_DIR = "data/processed"
OWN_TEST_FRACTION = 0.3


class LandmarkSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, X, y, label_names, dataset_name):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1)

    accuracy = (preds == y_tensor).float().mean().item()

    print(f"\n---- {dataset_name} Results ----")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(label_names)
    print(confusion_matrix(y_tensor.tolist(), preds.tolist(), labels=list(range(len(label_names)))))
    print("\nClassification Report:")
    print(classification_report(
        y_tensor.tolist(), preds.tolist(),
        labels=list(range(len(label_names))),
        target_names=label_names, zero_division=0
    ))
    return accuracy


def main():
    # ---- Load INCLUDE data ----
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))
    train_idx = np.load(os.path.join(DATA_DIR, "train_indices.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "test_indices.npy"))

    classes = sorted(set(y_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    y = np.array([label_to_idx[label] for label in y_raw])

    X_include_train, y_include_train = X[train_idx], y[train_idx]
    X_include_test, y_include_test = X[test_idx], y[test_idx]

    # ---- Load own recordings and split (stratified) into own-train / own-test ----
    X_own = np.load(os.path.join(DATA_DIR, "X_own.npy"))
    y_own_raw = np.load(os.path.join(DATA_DIR, "y_own.npy"))
    y_own = np.array([label_to_idx[label] for label in y_own_raw])

    X_own_train, X_own_test, y_own_train, y_own_test = train_test_split(
        X_own, y_own, test_size=OWN_TEST_FRACTION, stratify=y_own, random_state=42
    )

    # ---- Combine training sets ----
    X_train_combined = np.concatenate([X_include_train, X_own_train], axis=0)
    y_train_combined = np.concatenate([y_include_train, y_own_train], axis=0)

    print(f"Combined train: {len(X_train_combined)} "
          f"(INCLUDE: {len(X_include_train)} + own: {len(X_own_train)})")
    print(f"INCLUDE test: {len(X_include_test)} | Own held-out test: {len(X_own_test)}")

    train_dataset = LandmarkSequenceDataset(X_train_combined, y_train_combined)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = X.shape[2]
    num_classes = len(classes)

    model = SimpleLSTMClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {total_loss/total:.4f} | Train Acc: {correct/total:.3f}")

    model.eval()
    label_names = [idx_to_label[i] for i in range(num_classes)]

    include_acc = evaluate(model, X_include_test, y_include_test, label_names, "INCLUDE Held-Out Test")
    own_acc = evaluate(model, X_own_test, y_own_test, label_names, "Own Held-Out Test")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/combined_lstm.pt")
    print(f"\nModel saved to models/combined_lstm.pt")
    print(f"\nSummary: INCLUDE test acc = {include_acc:.3f} | Own test acc = {own_acc:.3f}")


if __name__ == "__main__":
    main()