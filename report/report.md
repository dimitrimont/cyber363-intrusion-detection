# Machine Learning for Network Intrusion Detection

**Course:** CYBER 363  
**Date:** April 8, 2026

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

Each sample has 41 features (38 numeric, 3 categorical: `protocol_type`, `service`, `flag`). Attacks span four categories: DoS, Probe, R2L, and U2R — all collapsed to label 1 for binary classification.

---

## 3. Data Preprocessing

1. **Drop `difficulty`** — metadata column, not a feature.
2. **Binary label** — `normal` → 0, any attack → 1.
3. **Deduplicate** — removed 9 identical training rows.
4. **Fill NaN/inf** — replaced with column medians (computed from training data only).
5. **One-hot encode** — `protocol_type`, `service`, `flag` encoded via `pd.get_dummies()`, fit on training data only; test set aligned to the same columns. Feature count: 41 → **122**.
6. **Normalize** — `StandardScaler` fit on training data and applied to both splits.

All steps are fit on training data only to prevent data leakage.

---

## 4. Model Training

### Classical Models (`src/classical_models.py`)

| Model | Description | Key Config |
|---|---|---|
| Random Forest | Ensemble of 100 decision trees; majority vote | `n_estimators=100`, `n_jobs=-1` |
| SVM (RBF) | Max-margin classifier with RBF kernel | `C=1.0`, `gamma='scale'` |
| Logistic Regression | Linear probability model; interpretability baseline | `max_iter=1000` |
| k-Nearest Neighbors | Labels by majority vote of 5 nearest training samples | `k=5`, `n_jobs=-1` |

### Deep Learning Model (`src/deep_learning_model.py`)

Feedforward neural network built in PyTorch with 72,949 parameters:

```
Input (122) → BatchNorm → Dense(256, ReLU) → Dropout(0.3)
            → Dense(128, ReLU) → Dropout(0.3)
            → Dense(64, ReLU) → Dense(1, Sigmoid)
```

Trained with Adam (lr=0.001), batch size 256, BCE loss, and early stopping (patience=5). Stopped at **epoch 10** with best weights from epoch 5 (val loss 0.0236).

---

## 5. Results and Discussion

### Summary

*Full metrics in `results/metrics/summary.csv`*

| Model | Accuracy | F1 | ROC-AUC | Train Time | Infer Time |
|---|---|---|---|---|---|
| Random Forest | 0.7768 | 0.7760 | **0.9598** | 1.40s | 0.027s |
| SVM (RBF) | **0.7872** | **0.7872** | 0.9040 | 1055.89s | 10.69s |
| Logistic Regression | 0.7547 | 0.7542 | 0.7822 | **1.23s** | **0.004s** |
| k-Nearest Neighbors | 0.7705 | 0.7703 | 0.8197 | 6.29s | 3.025s |
| Neural Network | 0.7727 | 0.7725 | 0.8883 | 9.09s | 0.034s |

### Accuracy & F1

*See [accuracy_bar.png](../results/plots/accuracy_bar.png) and [f1_bar.png](../results/plots/f1_bar.png)*

All models scored within a narrow 75–79% accuracy range, with F1 rankings matching accuracy rankings almost exactly. SVM edged out Random Forest on both metrics, but the gap is small (~1%).

### ROC-AUC

*See [roc_curves.png](../results/plots/roc_curves.png)*

Random Forest achieved the highest AUC (0.9598), meaning it is the best model for separating classes across all possible thresholds — even though its fixed-threshold accuracy was second. Logistic Regression was the weakest at 0.7822, confirming the decision boundary is not linearly separable in this feature space.

### Confusion Matrices

*See [cm_Random_Forest.png](../results/plots/cm_Random_Forest.png), [cm_SVM_RBF.png](../results/plots/cm_SVM_RBF.png), [cm_Logistic_Regression.png](../results/plots/cm_Logistic_Regression.png), [cm_k-Nearest_Neighbors.png](../results/plots/cm_k-Nearest_Neighbors.png), [cm_Neural_Network.png](../results/plots/cm_Neural_Network.png)*

SVM had the fewest false negatives (highest recall 0.8043), which is the most critical metric in an IDS — missed attacks are more dangerous than false alarms. Logistic Regression had the most missed attacks (recall 0.7755). Random Forest had the fewest false alarms (highest precision 0.8167).

### Training Time

*See [training_time_bar.png](../results/plots/training_time_bar.png)*

SVM's training time (1,055s / ~17.6 min) dwarfs every other model due to its quadratic scaling with sample count. All other models trained in under 10 seconds. This makes SVM impractical in scenarios requiring frequent retraining on new attack data.

### Discussion

**Random Forest** is the best overall model. Its AUC of 0.9598 means threshold tuning could push accuracy above 78.72% (the SVM's best), and it trains in 1.4 seconds. The SVM's marginally better accuracy is not worth a 750× training time cost.

**Logistic Regression** is the weakest classifier but the most practical for latency-critical deployments — 1.23s training and 0.004s inference. Its low AUC (0.7822) rules it out as a standalone IDS model but makes it viable as a fast first-pass filter.

**k-NN** is impractical for real-time use; 3 seconds to classify 22k samples does not scale to high-throughput traffic monitoring.

**Neural Network** performed competitively (AUC 0.8883, 0.034s inference) with no hyperparameter tuning beyond early stopping. Tree-based ensembles tend to outperform neural networks on structured tabular data, which is consistent with Random Forest beating it here.

---

## 6. Conclusion & Future Work

Random Forest is the recommended model for this task — best ROC-AUC, second-best accuracy, and fast training and inference. SVM achieved slightly higher accuracy but is too slow to retrain. Logistic Regression confirmed the boundary is non-linear. The Neural Network showed that deep learning is competitive on this dataset with minimal tuning.

**Future improvements:**
- **Threshold tuning** — Random Forest's AUC (0.9598) suggests a non-default threshold would improve recall without large accuracy loss.
- **Multi-class classification** — distinguishing DoS vs. Probe vs. R2L/U2R enables more targeted response.
- **Modern datasets** — CICIDS2017 or UNSW-NB15 reflect contemporary attack types better than the 1999-derived NSL-KDD traffic.
- **Feature selection** — SHAP or mutual information could reduce the 122 features and improve generalization.

---

## References

1. Tavallaee, M., et al. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE CISDA.
2. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.
4. Srivastava, N., et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* JMLR, 15(1), 1929–1958.
5. Kingma, D., & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR.
