"""
evaluation.py — aggregates results and generates plots + summary CSV.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_COLORS = {
    "Random Forest":       "#2196F3",
    "SVM (RBF)":           "#4CAF50",
    "Logistic Regression": "#FF9800",
    "k-Nearest Neighbors": "#9C27B0",
    "Neural Network":      "#F44336",
}
DEFAULT_COLOR = "#607D8B"


def build_summary_table(all_results):
    rows = [{
        "model":        m["model_name"],
        "accuracy":     round(m["accuracy"],   4),
        "precision":    round(m["precision"],  4),
        "recall":       round(m["recall"],     4),
        "f1":           round(m["f1"],         4),
        "roc_auc":      round(m["roc_auc"],    4),
        "train_time_s": round(m["train_time"] or 0, 2),
        "infer_time_s": round(m["inference_time"], 4),
    } for m in all_results]

    df = pd.DataFrame(rows).set_index("model")
    path = os.path.join(METRICS_DIR, "summary.csv")
    df.to_csv(path)
    print(f"[eval] Summary table saved → {path}")
    print(df.to_string())
    return df


def plot_confusion_matrices(all_results):
    for m in all_results:
        name = m["model_name"].replace(" ", "_").replace("(", "").replace(")", "")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {m['model_name']}")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"cm_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[eval] Confusion matrix saved → {path}")


def plot_roc_curves(all_results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for m in all_results:
        y_score = m.get("y_score")
        if y_score is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            color = MODEL_COLORS.get(m["model_name"], DEFAULT_COLOR)
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{m['model_name']} (AUC={m['roc_auc']:.3f})")
        except Exception as e:
            print(f"[eval] ROC skipped for {m['model_name']}: {e}")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] ROC curve overlay saved → {path}")


def _hbar(all_results, key, xlabel, title, filename, xlim=None):
    names  = [m["model_name"] for m in all_results]
    values = [m[key] for m in all_results]
    colors = [MODEL_COLORS.get(n, DEFAULT_COLOR) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if xlim:
        ax.set_xlim(xlim)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Plot saved → {path}")


def plot_accuracy_bar(all_results):
    names     = [m["model_name"] for m in all_results]
    accuracies = [m["accuracy"] for m in all_results]
    colors    = [MODEL_COLORS.get(n, DEFAULT_COLOR) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, accuracies, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, accuracies):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison — All Models", fontsize=14)
    ax.set_xlim([0.5, 1.02])
    ax.axvline(0.5, color="grey", linestyle="--", lw=1, label="Random baseline")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Accuracy bar chart saved → {path}")


def plot_f1_bar(all_results):
    _hbar(all_results, "f1", "Macro F1-Score", "F1-Score Comparison — All Models",
          "f1_bar.png", xlim=[0, 1.08])


def plot_training_time(all_results):
    names  = [m["model_name"] for m in all_results]
    times  = [max(m["train_time"] or 1e-3, 1e-3) for m in all_results]
    colors = [MODEL_COLORS.get(n, DEFAULT_COLOR) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.6)
    ax.set_xscale("log")
    for bar, t in zip(bars, times):
        label = f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m"
        ax.text(t * 1.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9)
    ax.set_xlabel("Training Time (seconds, log scale)", fontsize=12)
    ax.set_title("Training Time Comparison — All Models", fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_time_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Training time chart saved → {path}")


def evaluate_all(all_results, y_test) -> pd.DataFrame:
    print("=" * 60)
    print(f"EVALUATION  ({len(all_results)} models)")
    print("=" * 60)

    df = build_summary_table(all_results)
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results, y_test)
    plot_accuracy_bar(all_results)
    plot_training_time(all_results)
    plot_f1_bar(all_results)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print(f"  Plots   → {PLOTS_DIR}/")
    print(f"  Metrics → {METRICS_DIR}/summary.csv")
    print("=" * 60)
    return df


if __name__ == "__main__":
    csv_path = os.path.join(METRICS_DIR, "summary.csv")
    if os.path.exists(csv_path):
        print(pd.read_csv(csv_path, index_col="model").to_string())
    else:
        print("No summary.csv found. Run main.py first.")
