"""
classical_models.py — trains 4 classical ML classifiers on NSL-KDD.
"""

import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "results", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1":               f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc":          roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "inference_time":   inference_time,
        "y_pred":           y_pred,
        "y_score":          y_score,
    }


def _train(name, model, X_train, y_train, X_test, y_test, filename):
    print(f"\n[classical] Training {name} ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["train_time"]  = time.time() - t0
    metrics["model_name"]  = name
    joblib.dump(model, os.path.join(MODELS_DIR, filename))
    _print_metrics(metrics)
    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    return _train(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        X_train, y_train, X_test, y_test, "random_forest.joblib"
    )

def train_svm(X_train, y_train, X_test, y_test):
    return _train(
        "SVM (RBF)",
        SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
        X_train, y_train, X_test, y_test, "svm_rbf.joblib"
    )

def train_logistic_regression(X_train, y_train, X_test, y_test):
    return _train(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        X_train, y_train, X_test, y_test, "logistic_regression.joblib"
    )

def train_knn(X_train, y_train, X_test, y_test):
    return _train(
        "k-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        X_train, y_train, X_test, y_test, "knn.joblib"
    )


def _print_metrics(m):
    print(f"  [{m['model_name']}]")
    print(f"    Accuracy  : {m['accuracy']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1        : {m['f1']:.4f}")
    print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"    Train time: {m['train_time']:.2f}s")
    print(f"    Infer time: {m['inference_time']:.4f}s")


def train_all(data: dict):
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    print("=" * 60)
    print("CLASSICAL MODELS")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test : {X_test.shape[0]} samples")
    print("=" * 60)

    results, models = [], []
    for fn in [train_random_forest, train_svm, train_logistic_regression, train_knn]:
        model, metrics = fn(X_train, y_train, X_test, y_test)
        results.append(metrics)
        models.append(model)

    print("\n" + "=" * 60)
    print("Classical models complete.")
    print("=" * 60)
    return models, results


if __name__ == "__main__":
    from data_preprocessing import load_processed, processed_data_exists, preprocess
    if not processed_data_exists():
        preprocess()
    train_all(load_processed())
