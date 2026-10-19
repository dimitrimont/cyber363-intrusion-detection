import numpy as np
import pytest
from sklearn.metrics import recall_score

from metrics import budget_threshold, detection_metrics


def test_attacks_caught_is_not_the_macro_recall():
    y    = [0, 0, 0, 0, 1, 1, 1, 1]
    pred = [0, 0, 0, 1, 1, 1, 0, 0]
    new  = [0, 0, 0, 0, 0, 0, 1, 1]  # the last two attacks are of a type not in training
    m = detection_metrics(y, pred, pred, new)
    assert m["attacks_caught"] == 0.5
    assert m["seen_caught"] == 1.0 and m["new_caught"] == 0.0
    assert m["false_alarms"] == 0.25
    # What the first version printed as "Recall": the average of 3/4 normals passed and 2/4 attacks caught.
    assert recall_score(y, pred, average="macro") == pytest.approx(0.625)


def test_budget_threshold_is_the_lowest_within_budget():
    normal = [0.0] * 98 + [0.3, 0.6]
    scores = np.array(normal + [0.9] * 20)
    y = np.array([0] * 100 + [1] * 20)
    assert budget_threshold(scores, y, budget=0.01) == 0.6   # flags 1 of 100 normals
    assert budget_threshold(scores, y, budget=0.02) == 0.3   # flags 2 of 100
    assert budget_threshold(scores, y, budget=0.0) == 0.9    # flags none


def test_budget_threshold_holds_on_the_rows_it_was_chosen_on():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.beta(1, 8, 2000), rng.beta(8, 1, 500)])
    y = np.r_[np.zeros(2000), np.ones(500)]
    t = budget_threshold(scores, y, budget=0.01)
    assert (scores[:2000] >= t).mean() <= 0.01
