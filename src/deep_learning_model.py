"""
deep_learning_model.py — feedforward neural network for binary intrusion detection.

Architecture: Input → BatchNorm → Dense(256, ReLU) → Dropout(0.3)
                               → Dense(128, ReLU) → Dropout(0.3)
                               → Dense(64,  ReLU)
                               → Dense(1,   Sigmoid)
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "results", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE    = 256
MAX_EPOCHS    = 50
PATIENCE      = 5
LEARNING_RATE = 1e-3
DROPOUT_RATE  = 0.3


class IntrusionNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch).squeeze(1), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        history.append((train_loss, val_loss))
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)

    return history, time.time() - t0


def evaluate_model(model, X_test, y_test, device):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    t0 = time.time()
    with torch.no_grad():
        y_score = model(X_tensor).squeeze(1).cpu().numpy()
    inference_time = time.time() - t0
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "model_name":       "Neural Network",
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":               f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc":          roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "inference_time":   inference_time,
        "train_time":       None,
        "y_pred":           y_pred,
        "y_score":          y_score,
    }


def train_deep_learning_model(data: dict):
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_test       = data["X_test"]
    y_test       = data["y_test"]
    input_dim    = X_train_full.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[deep learning] Device: {device}")

    # 90/10 train/val split
    n_val = max(1, int(0.10 * len(X_train_full)))
    idx   = np.random.permutation(len(X_train_full))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X_train_full[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train_full[train_idx], dtype=torch.float32)
    X_vl = torch.tensor(X_train_full[val_idx],   dtype=torch.float32)
    y_vl = torch.tensor(y_train_full[val_idx],   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl), batch_size=BATCH_SIZE)

    print("=" * 60)
    print("DEEP LEARNING MODEL")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {X_test.shape[0]}, Features: {input_dim}")
    print("=" * 60)

    model = IntrusionNet(input_dim=input_dim).to(device)
    history, train_time = train_model(model, train_loader, val_loader, device)

    metrics = evaluate_model(model, X_test, y_test, device)
    metrics["train_time"] = train_time
    metrics["history"]    = history

    print(f"\n  [Neural Network]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1        : {metrics['f1']:.4f}")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"    Train time: {metrics['train_time']:.2f}s")
    print(f"    Infer time: {metrics['inference_time']:.4f}s")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neural_network.pt"))
    print(f"[deep learning] Model saved to {MODELS_DIR}/neural_network.pt")

    print("\n" + "=" * 60)
    print("Deep learning model complete.")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_processed, processed_data_exists, preprocess
    if not processed_data_exists():
        preprocess()
    train_deep_learning_model(load_processed())
