# Machine Learning for Network Intrusion Detection

**Course:** CYBER 363  
**Date:** April 8, 2026; revised October 2026 (the original is the tag `first-version`)

---

## 1. Introduction

This project applies machine learning to binary network intrusion detection using the NSL-KDD dataset. Five models are trained and compared — four classical classifiers and one deep learning model — to determine which best distinguishes normal traffic from attacks.

**Task:** Classify each network connection as `normal` (0) or `attack` (1).

---

## 2. Dataset

**NSL-KDD** is an improved version of the KDD Cup 1999 benchmark. It removes duplicate records from training and provides a more challenging test set. Files were downloaded from Kaggle and placed in `data/nslkdd/`.

| Split | Normal | Attack | Total | Attack % |
|---|---|---|---|---|
| Training | 67,343 | 58,621 | 125,964 | 46.5% |
| Test | 9,711 | 12,833 | 22,544 | 56.9% |

The test file is not drawn from the same traffic as the training file. It holds 38 labels against
the training file's 23, and **3,750 of its 12,833 attacks (29%) are of 17 types that never appear in
training** (`mscan` 996, `apache2` 737, `processtable` 685, `snmpguess` 331, `saint` 319, `mailbomb`
293, and eleven smaller ones). Some types that do appear are rare in training and common in test:
`guess_passwd` has 53 training rows and 1,231 test rows.

Each sample has 41 features (38 numeric, 3 categorical: `protocol_type`, `service`, `flag`). Attacks span four categories: DoS, Probe, R2L, and U2R — all collapsed to label 1 for binary classification.

---

## 3. Data Preprocessing

1. **Drop `difficulty`** — metadata column, not a feature.
2. **Binary label** — `normal` → 0, any attack → 1.
3. **Deduplicate** — removed 9 identical training rows.
4. **Fill NaN/inf** — replaced with column medians (computed from training data only).
5. **One-hot encode** — `protocol_type`, `service`, `flag` encoded via `pd.get_dummies()`, fit on training data only; test set aligned to the same columns. Feature count: 41 → **122**.
6. **Hold out a validation slice** — 10% of the training rows, stratified, used only to choose a
   threshold (and for the network's early stopping). Models are fitted on the other 90%.
7. **Normalize** — `StandardScaler` fit on the training rows and applied to the rest.

All steps are fit on training data only to prevent data leakage; the test file is only scored.

---

## 4. Model Training

### Classical Models (`src/classical_models.py`)

| Model | Description | Key Config |
|---|---|---|
| Random Forest | Ensemble of 100 decision trees; majority vote | `n_estimators=100`, `n_jobs=-1` |
| SVM (RBF) | Max-margin classifier with RBF kernel | `C=1.0`, `gamma='scale'` (no probability calibration; its margin is the score) |
| Logistic Regression | Linear probability model; interpretability baseline | `max_iter=1000` |
| k-Nearest Neighbors | Labels by majority vote of 5 nearest training samples | `k=5`, `n_jobs=-1` |

### Deep Learning Model (`src/deep_learning_model.py`)

Feedforward neural network built in PyTorch with 72,949 parameters:

```
Input (122) → BatchNorm → Dense(256, ReLU) → Dropout(0.3)
            → Dense(128, ReLU) → Dropout(0.3)
            → Dense(64, ReLU) → Dense(1, Sigmoid)
```

Trained with Adam (lr=0.001), batch size 256, BCE loss, and early stopping (patience=5) on the
validation slice.

### Scoring and thresholds

The first version of this report used precision, recall and F1 averaged over the two classes
("macro"), and read the average recall as detection. For a detector the useful numbers are per
class: the share of attacks caught and the share of normal connections flagged (false alarms), and
here also how many attacks of new types are caught. Each model is reported at two thresholds: its
default (probability 0.5, or the SVM's margin 0), and the lowest threshold that flags no more than 1%
of the validation slice's normal connections.

---

## 5. Results and Discussion

*Full metrics in `results/metrics/summary.csv`; plots in `results/plots/`.*

| Model | Threshold | Attacks caught | seen types | new types | False alarms | Accuracy | ROC AUC |
|---|---|---|---|---|---|---|---|
| Random Forest | default | 60.5% | 76.8% | 21.1% | 2.7% | 76.4% | **0.962** |
|  | 1% budget (0.077) | **90.0%** | **91.0%** | **87.5%** | 9.9% | **90.0%** |  |
| SVM (RBF) | default | 68.0% | 77.2% | 45.6% | 7.2% | 78.7% | 0.903 |
|  | 1% budget | 68.5% | 77.4% | 47.1% | 7.3% | 78.9% |  |
| Logistic Regression | default | 62.6% | 71.3% | 41.5% | 7.4% | 75.5% | 0.781 |
|  | 1% budget (0.75) | 60.5% | 69.8% | 38.0% | 7.0% | 74.5% |  |
| k-Nearest Neighbors | default | 65.0% | 73.9% | 43.2% | 7.2% | 76.9% | 0.820 |
|  | 1% budget (0.2) | 71.9% | 79.3% | 53.9% | 7.6% | 80.7% |  |
| Neural Network | default | 69.3% | 76.4% | 52.2% | 6.7% | 79.7% | 0.900 |
|  | 1% budget (0.39) | 70.3% | 77.1% | 53.7% | 7.0% | 80.0% |  |

### Why every model lands near 77%

Inside the training file the problem is nearly solved: by 5-fold cross-validation the random forest
is 99.9% accurate and logistic regression 97.4%. The first version of this report took logistic
regression's lower test score as proof that the classes are not linearly separable; the
cross-validation says otherwise. What every model loses on the test file is the attacks it has never
seen: at their default thresholds they catch 71 to 77% of attack types from training and 21 to 52% of
the new ones. The random forest misses all 1,231 `guess_passwd` attacks, a type it saw 53 times.

### Thresholds

The random forest has the highest ROC AUC (0.962) but the fewest attacks caught at its default
threshold (60.5%). Both are true: it scores new attacks above normal traffic, but rarely above 0.5.
Chosen on the validation slice for a 1% false-alarm budget, its threshold falls to 0.077, and it then
catches 90.0% of attacks, 87.5% of the new types. The cost is false alarms: 9.9% of the test file's
normal connections are flagged, not 1%, because its normal traffic differs from the training file's
too. The other models move little at their own 1% point, which sits near their default. Whether 10%
false alarms is acceptable depends on how many alerts can be read; at 0.5 the forest raises the fewest
(2.7%) and misses the most.

### Training and testing time

| Model | Fit | Score the test file |
|---|---|---|
| Random Forest | 2.4 s | 0.07 s |
| SVM (RBF) | 217 s | 20.2 s |
| Logistic Regression | 1.1 s | 0.009 s |
| k-Nearest Neighbors | 0.008 s | 7.4 s |
| Neural Network | 9.2 s | 0.02 s |

The first version reported 1,056 s for the SVM. Most of that was `probability=True`, which fits the
SVM five more times to calibrate probabilities (808 s for the fit with it, 108 s without, measured on
their own), plus scoring the test file, which the timer included. The calibration changed no
prediction and no AUC, so it is gone. k-NN's cost is all at test time: it compares each connection
with every training row. Times vary with the machine's load: across three runs of this pipeline the
SVM's fit took 177 to 219 s.

### Discussion

**Random Forest** is still the recommended model, now for a measured reason: it ranks attacks best,
and its threshold can be set from the training side to trade false alarms for detection; at a 1%
validation budget it catches 90% of attacks, including 87.5% of types it has never seen.

**SVM** catches 68.0% of attacks at its default threshold, second to the network, but costs by far
the most time and gains little at the budgeted threshold.

**Logistic Regression** is the fastest and the weakest at the test file, but not because it is
linear: inside the training file it is 97.4% accurate.

**k-NN** improves to 71.9% at its budgeted threshold, but every prediction compares a connection with
every training row, so it is slow to score (7.4 s for the test file, against 0.07 s for the forest).

**Neural Network** catches the most attacks at its default threshold (69.3%, AUC 0.900) with no
tuning beyond early stopping.

## 6. Conclusion & Future Work

On attack types they were trained on, all five models do well; on new types, which are 29% of the
test file's attacks, the default thresholds catch about half at best. The random forest's scores carry
more than its default threshold uses: chosen from training data alone, a lower threshold catches 90%
of attacks at 10% false alarms.

**Future improvements:**
- **Multi-class classification** — distinguishing DoS vs. Probe vs. R2L/U2R enables more targeted response, and shows which families the new types belong to.
- **Under-represented types** — `guess_passwd` and `warezmaster` are rare in training and common in test; reweighting or more data for R2L attacks.
- **Modern datasets** — CICIDS2017 or UNSW-NB15 reflect contemporary attack types better than the 1999-derived NSL-KDD traffic.
- **Feature selection** — SHAP or mutual information could reduce the 122 features and improve generalization.

---

## References

1. Tavallaee, M., et al. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE CISDA.
2. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.
4. Srivastava, N., et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* JMLR, 15(1), 1929–1958.
5. Kingma, D., & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR.
