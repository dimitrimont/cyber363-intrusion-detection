"""
evaluation.py — aggregates results and generates plots + summary CSV.

Every model appears twice: at its default decision (probability 0.5, or the SVM's margin 0) and at
the threshold that keeps false alarms on the validation rows within 1%.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")

MODEL_COLORS = {
    "Random Forest":       "#2196F3",
    "SVM (RBF)":           "#4CAF50",
    "Logistic Regression": "#FF9800",
    "k-Nearest Neighbors": "#9C27B0",
    "Neural Network":      "#F44336",
}
DEFAULT_COLOR = "#607D8B"
COLUMNS = ["accuracy", "attacks_caught", "seen_caught", "new_caught", "false_alarms",
           "attack_precision", "macro_f1", "roc_auc", "fit_s", "predict_s"]


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def build_summary_table(rows):
    df = pd.DataFrame([{**{"model": r["model_name"], "rule": r["rule"],
                           "threshold": None if r["threshold"] is None else round(float(r["threshold"]), 4)},
                        **{c: round(float(r[c]), 6) for c in COLUMNS}} for r in rows]).set_index(["model", "rule"])
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "summary.csv")
    df.to_csv(path)
    print(f"[eval] Summary table saved → {rel(path)}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string())
    return df


def plot_detection(rows):
    """Attacks caught (seen and new types) and false alarms, both thresholds, every model."""
    labels = [f"{r['model_name']}\n{'default' if r['threshold'] is None else '1% budget'}" for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(rows) + 1.5), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    axes[0].barh(y + 0.2, [r["seen_caught"] for r in rows], height=0.38, color="#455A64", label="attack types seen in training")
    axes[0].barh(y - 0.2, [r["new_caught"] for r in rows], height=0.38, color="#E64A19", label="attack types only in the test file")
    axes[0].set_xlim(0, 1); axes[0].set_xlabel("Share of attacks caught"); axes[0].legend(loc="lower right", fontsize=8)
    axes[1].barh(y, [r["false_alarms"] for r in rows], height=0.5, color="#90A4AE")
    axes[1].set_xlabel("Share of normal connections flagged")
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Attacks caught and false alarms on KDDTest+")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "detection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Detection chart saved → {rel(path)}")


def plot_roc_curves(rows, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    for r in rows:
        color = MODEL_COLORS.get(r["model_name"], DEFAULT_COLOR)
        if r["threshold"] is None:
            fpr, tpr, _ = roc_curve(y_test, r["y_score"])
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{r['model_name']} (AUC={r['roc_auc']:.3f})")
            ax.plot(r["false_alarms"], r["attacks_caught"], "o", color=color, ms=7)
        else:
            ax.plot(r["false_alarms"], r["attacks_caught"], "s", color=color, ms=7, mfc="white", mew=2)
    ax.plot([], [], "ko", label="default threshold"); ax.plot([], [], "ks", mfc="white", mew=2, label="1% false-alarm budget")
    ax.set_xlabel("False alarms (normal connections flagged)", fontsize=12)
    ax.set_ylabel("Attacks caught", fontsize=12)
    ax.set_title("ROC curves on KDDTest+, with both operating points", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] ROC curves saved → {rel(path)}")


def plot_fit_time(rows):
    rows = [r for r in rows if r["threshold"] is None]
    names = [r["model_name"] for r in rows]
    times = [max(r["fit_s"], 1e-3) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, times, color=[MODEL_COLORS.get(n, DEFAULT_COLOR) for n in names], height=0.6)
    ax.set_xscale("log")
    for bar, t in zip(bars, times):
        ax.text(t * 1.05, bar.get_y() + bar.get_height() / 2, f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m", va="center", fontsize=9)
    ax.set_xlabel("Fitting time only (seconds, log scale)")
    ax.set_title("Training time")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_time_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Training time chart saved → {rel(path)}")


def evaluate_all(rows, y_test) -> pd.DataFrame:
    print("=" * 60)
    print(f"EVALUATION  ({len(rows) // 2} models, two thresholds each)")
    print("=" * 60)
    df = build_summary_table(rows)
    plot_detection(rows)
    plot_roc_curves(rows, y_test)
    plot_fit_time(rows)
    print("=" * 60)
    return df


if __name__ == "__main__":
    csv_path = os.path.join(METRICS_DIR, "summary.csv")
    if os.path.exists(csv_path):
        print(pd.read_csv(csv_path, index_col=["model", "rule"]).to_string())
    else:
        print("No summary.csv found. Run main.py first.")
