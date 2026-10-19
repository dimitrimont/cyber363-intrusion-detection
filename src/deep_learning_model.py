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

from metrics import both_thresholds

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "results", "models")

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


def predict_scores(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()


def train_deep_learning_model(data: dict):
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]
    input_dim        = X_train.shape[1]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[deep learning] Device: {device}")

    # The shared validation rows serve early stopping as well as the budget threshold.
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=BATCH_SIZE)

    print("=" * 60)
    print("DEEP LEARNING MODEL")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {input_dim}")
    print("=" * 60)

    model = IntrusionNet(input_dim=input_dim).to(device)
    history, fit_s = train_model(model, train_loader, val_loader, device)

    t0 = time.time()
    test_score = predict_scores(model, X_test, device)
    predict_s = time.time() - t0
    val_score = predict_scores(model, X_val, device)
    rows = both_thresholds("Neural Network", y_test, (test_score >= 0.5).astype(int), test_score, val_score, y_val, data["test_new"])
    for r in rows:
        r.update(fit_s=fit_s, predict_s=predict_s, y_score=test_score, history=history)
        print(f"  [Neural Network — {r['rule']}] attacks caught {r['attacks_caught']:.4f} "
              f"(seen {r['seen_caught']:.4f}, new {r['new_caught']:.4f}), false alarms {r['false_alarms']:.4f}, "
              f"accuracy {r['accuracy']:.4f}, AUC {r['roc_auc']:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neural_network.pt"))
    print("[deep learning] Model saved to results/models/neural_network.pt")

    print("\n" + "=" * 60)
    print("Deep learning model complete.")
    print("=" * 60)

    return model, rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_processed, processed_data_exists, preprocess
    data = load_processed() if processed_data_exists() else preprocess()
    train_deep_learning_model(data)
