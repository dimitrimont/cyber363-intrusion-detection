"""
metrics.py — how a detector is scored, in the terms an analyst would use.

The first version reported precision, recall and F1 averaged over the two classes ("macro"). For a
detector that reads as "recall 0.80" when 63% of attacks were caught: the average of 97% of normal
connections passed and 63% of attacks flagged. The headline numbers here are per class:

- attacks caught: the share of attack connections flagged (the attack class's recall);
- false alarms: the share of normal connections flagged;
- both for attack types seen in training and for the 17 types that appear only in the test file.

Macro F1 and ROC AUC are kept beside them.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

FALSE_ALARM_BUDGET = 0.01


def budget_threshold(val_scores, y_val, budget: float = FALSE_ALARM_BUDGET) -> float:
    """The lowest threshold whose false-alarm rate on the validation rows stays within the budget.

    A connection is flagged when its score is at or above the threshold. Only training-side data is
    used, so the test file has no say in the choice.
    """
    val_scores = np.asarray(val_scores, dtype=float)
    normal = np.sort(val_scores[np.asarray(y_val) == 0])
    for t in np.unique(val_scores):
        flagged = len(normal) - np.searchsorted(normal, t, side="left")
        if flagged / max(len(normal), 1) <= budget:
            return float(t)
    return float(np.inf)


def detection_metrics(y_true, y_pred, y_score, test_new) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    attack = y_true == 1
    new = np.asarray(test_new, dtype=bool)
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "attacks_caught":   y_pred[attack].mean(),
        "seen_caught":      y_pred[attack & ~new].mean(),
        "new_caught":       y_pred[attack & new].mean() if (attack & new).any() else float("nan"),
        "false_alarms":     y_pred[~attack].mean(),
        "attack_precision": precision_score(y_true, y_pred, zero_division=0),
        "macro_f1":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc":          roc_auc_score(y_true, y_score),
    }


def both_thresholds(name, y_test, default_pred, test_score, val_score, y_val, test_new, budget=FALSE_ALARM_BUDGET):
    """Two rows for one model: its own default decision, and the false-alarm-budget threshold."""
    t = budget_threshold(val_score, y_val, budget)
    budget_pred = (np.asarray(test_score) >= t).astype(int)
    val_fa = float((np.asarray(val_score)[np.asarray(y_val) == 0] >= t).mean())
    return [
        {"model_name": name, "rule": "default", "threshold": None, **detection_metrics(y_test, default_pred, test_score, test_new)},
        {"model_name": name, "rule": f"{budget:.0%} false-alarm budget", "threshold": t, "val_false_alarms": val_fa,
         **detection_metrics(y_test, budget_pred, test_score, test_new)},
    ]
