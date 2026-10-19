"""
classical_models.py — trains 4 classical ML classifiers on NSL-KDD.

Each model is fitted on the training rows, scored on the validation rows (to pick the budget
threshold) and on the test file. Fitting and testing are timed apart.
"""

import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from metrics import both_thresholds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "results", "models")


def scores(model, X):
    """A score that rises with how attack-like a row is: a probability, or the SVM's margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def make_models():
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        # No probability=True: it ran a hidden 5-fold calibration (most of the first version's
        # 1,056 s) and changes nothing here. The margin ranks connections the same way.
        "SVM (RBF)":           SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def train_one(name, model, data, save=True):
    print(f"\n[classical] Training {name} ...")
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"])
    fit_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(data["X_test"])
    test_score = scores(model, data["X_test"])
    predict_s = time.time() - t0

    val_score = scores(model, data["X_val"])
    rows = both_thresholds(name, data["y_test"], pred, test_score, val_score, data["y_val"], data["test_new"])
    for r in rows:
        r.update(fit_s=fit_s, predict_s=predict_s, y_score=test_score)
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"))
    for r in rows:
        _print_metrics(r)
    return model, rows


def _print_metrics(m):
    rule = m["rule"] if m["threshold"] is None else f"{m['rule']} (threshold {m['threshold']:.3g})"
    print(f"  [{m['model_name']} — {rule}]")
    print(f"    Attacks caught : {m['attacks_caught']:.4f}  (seen types {m['seen_caught']:.4f}, new types {m['new_caught']:.4f})")
    print(f"    False alarms   : {m['false_alarms']:.4f}")
    print(f"    Accuracy       : {m['accuracy']:.4f}   macro F1 {m['macro_f1']:.4f}   ROC-AUC {m['roc_auc']:.4f}")
    print(f"    Fit {m['fit_s']:.2f}s, test {m['predict_s']:.2f}s")


def train_all(data: dict, skip=()):
    print("=" * 60)
    print("CLASSICAL MODELS")
    print(f"  Train: {data['X_train'].shape[0]} rows, {data['X_train'].shape[1]} features; "
          f"validation {data['X_val'].shape[0]}; test {data['X_test'].shape[0]}")
    print("=" * 60)

    results, models = [], []
    for name, model in make_models().items():
        if name in skip:
            continue
        model, rows = train_one(name, model, data)
        results += rows
        models.append(model)

    print("\n" + "=" * 60)
    print("Classical models complete.")
    print("=" * 60)
    return models, results


if __name__ == "__main__":
    from data_preprocessing import load_processed, processed_data_exists, preprocess
    data = load_processed() if processed_data_exists() else preprocess()
    train_all(data)
