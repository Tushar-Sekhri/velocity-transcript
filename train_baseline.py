"""
Velocity Transcript - Step 8 (Phase 6: Baseline model)
Trains a simple single-layer LSTM classifier on our 3-word landmark
sequences, using the session-aware train/test split from session_split.py.

Usage:
    python train_baseline.py
"""

import os
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "data/processed"
EXPERIMENT_LOG = "experiments/experiment_log.csv"

# ---- Hyperparameters (explicit, so they're easy to change and log) ----
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 60


class LandmarkSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (num_layers, batch, hidden_size) - take the last layer's hidden state
        last_hidden = hidden[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        return self.fc(out)  # (batch, num_classes)


def log_experiment(row):
    os.makedirs("experiments", exist_ok=True)
    file_exists = os.path.isfile(EXPERIMENT_LOG)
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    # ---- Load data ----
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))
    train_idx = np.load(os.path.join(DATA_DIR, "train_indices.npy"))
    test_idx = np.load(os.path.join(DATA_DIR, "test_indices.npy"))

    classes = sorted(set(y_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    y = np.array([label_to_idx[label] for label in y_raw])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Classes: {classes}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    train_dataset = LandmarkSequenceDataset(X_train, y_train)
    test_dataset = LandmarkSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X.shape[2]  # 126
    num_classes = len(classes)

    model = SimpleLSTMClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Training loop ----
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

        train_acc = correct / total
        avg_loss = total_loss / total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f}")

    # ---- Evaluation on held-out test set ----
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"\n---- Final Test Results ----")
    print(f"Test Accuracy: {test_acc:.3f}")

    label_names = [idx_to_label[i] for i in range(num_classes)]
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(label_names)
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    # ---- Log this experiment ----
    log_experiment({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "INCLUDE-Greetings-3word",
        "num_classes": num_classes,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "sequence_length": X.shape[1],
        "features_per_frame": X.shape[2],
        "model": f"LSTM(hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})",
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_acc": round(train_acc, 3),
        "test_acc": round(test_acc, 3),
    })
    print(f"\nExperiment logged to {EXPERIMENT_LOG}")


if __name__ == "__main__":
    main()