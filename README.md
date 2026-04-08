# ML for Intrusion Detection — NSL-KDD

CYBER 363 project comparing classical and deep learning models for network intrusion detection using the NSL-KDD dataset.

## Dataset

**NSL-KDD** — available on Kaggle:
https://www.kaggle.com/datasets/hassan06/nslkdd

Download `KDDTrain+.txt` and `KDDTest+.txt` and place them in `data/nslkdd/`.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Full run

```bash
OMP_NUM_THREADS=1 python main.py
```

Preprocesses the data, trains all 5 models, and saves results to `results/`.

### Fast test (10% of data)

```bash
python main.py --quick
```

Completes in under a minute. Useful for verifying code changes before a full run.

### Force repreprocess

```bash
python main.py --reprocess
```

Re-runs preprocessing even if processed `.npy` arrays already exist on disk.

### All flags

| Flag | What it does |
|------|-------------|
| `--quick` | Use 10% of data for fast end-to-end testing |
| `--reprocess` | Re-run preprocessing even if processed data exists |

## Project Structure

```
project/
├── data/
│   ├── nslkdd/             # Raw NSL-KDD files (download separately)
│   └── processed/          # Saved numpy arrays (auto-generated)
├── notebooks/
│   └── exploration.ipynb   # EDA notebook
├── src/
│   ├── data_preprocessing.py
│   ├── classical_models.py
│   ├── deep_learning_model.py
│   └── evaluation.py
├── results/
│   ├── plots/              # Generated figures
│   ├── metrics/            # summary.csv
│   └── models/             # Saved model checkpoints
├── report/
│   └── report.md
├── main.py
├── requirements.txt
└── README.md
```

## Models

| Model | Type |
|---|---|
| Random Forest | Classical |
| SVM (RBF kernel) | Classical |
| Logistic Regression | Classical |
| k-Nearest Neighbors | Classical |
| Feedforward Neural Network | Deep Learning |

## Notes

- All random seeds are fixed at 42 for reproducibility.
- Preprocessing: categorical features one-hot encoded, numeric features StandardScaler normalized.
- Binary task: `normal` = 0, any attack type = 1.
