"""
data_preprocessing.py — loads and prepares NSL-KDD for model training.
Pipeline: load → clean/label → one-hot encode → normalize → split off validation → save

The validation slice (10% of the training file) is held out of every model's training and used
for one thing: choosing a threshold that keeps false alarms within a budget. The test file is only
ever scored.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 42
VAL_FRACTION = 0.10
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data", "nslkdd")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "results", "models")

# NSL-KDD has no header row — column names assigned manually
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

CATEGORICAL = ['protocol_type', 'service', 'flag']
ARRAYS = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "test_type", "test_new"]


def processed_dir(quick: bool = False) -> str:
    """Quick and full runs keep their arrays apart, so a --quick run never feeds a full one."""
    return os.path.join(PROCESSED_DIR, "quick" if quick else "full")


def load_raw_data(quick: bool = False, data_dir: str = DATA_DIR):
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path  = os.path.join(data_dir, "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"NSL-KDD files not found in {data_dir}/\n"
            "They ship with this repository; if they are missing, download KDDTrain+.txt and\n"
            "KDDTest+.txt from https://www.unb.ca/cic/datasets/nsl.html and place them in data/nslkdd/"
        )

    print("[data] Loading NSL-KDD ...")
    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMNS)
    print(f"[data] Raw — train: {df_train.shape}, test: {df_test.shape}")

    if quick:
        df_train = df_train.sample(frac=0.10, random_state=SEED)
        df_test  = df_test.sample(frac=0.10,  random_state=SEED)
        print(f"[data] --quick: {df_train.shape[0]} train, {df_test.shape[0]} test")

    return df_train, df_test


def attack_types(df_train, df_test):
    """Each test row's attack type, and whether that type appears anywhere in the training file.

    KDDTest+ holds 17 attack types (3,750 rows) that KDDTrain+ does not, so a model is scored partly
    on attacks it has never seen. The two are reported apart.
    """
    seen = set(df_train['label'])
    test_type = df_test['label'].to_numpy(dtype=str)
    test_new = np.array([t != 'normal' and t not in seen for t in test_type])
    return test_type, test_new


def clean_and_label(df_train, df_test):
    # Drop metadata column, binary-encode label
    df_train = df_train.drop(columns=['difficulty'])
    df_test  = df_test.drop(columns=['difficulty'])

    y_train = (df_train['label'] != 'normal').astype(int)
    y_test  = (df_test['label']  != 'normal').astype(int)
    df_train = df_train.drop(columns=['label'])
    df_test  = df_test.drop(columns=['label'])

    # Remove duplicate training rows
    n_before = len(df_train)
    df_train['__y'] = y_train.values
    df_train = df_train.drop_duplicates()
    y_train  = df_train['__y']
    df_train = df_train.drop(columns=['__y'])
    print(f"[data] Removed {n_before - len(df_train)} duplicate training rows")

    # Fill inf/NaN with column medians (computed from train only)
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
        df_test[col]  = df_test[col].replace([np.inf, -np.inf], np.nan)
        median = df_train[col].median()
        df_train[col] = df_train[col].fillna(median)
        df_test[col]  = df_test[col].fillna(median)

    print(f"[data] Train — normal: {(y_train==0).sum()}, attack: {(y_train==1).sum()}")
    print(f"[data] Test  — normal: {(y_test==0).sum()},  attack: {(y_test==1).sum()}")
    return df_train, y_train, df_test, y_test


def encode_categoricals(df_train, df_test):
    # One-hot encode on train, align test to same columns (unseen categories → 0)
    df_train_enc = pd.get_dummies(df_train, columns=CATEGORICAL)
    df_test_enc  = pd.get_dummies(df_test,  columns=CATEGORICAL)
    df_train_enc, df_test_enc = df_train_enc.align(df_test_enc, join='left', axis=1, fill_value=0)
    print(f"[data] After one-hot encoding: {df_train_enc.shape[1]} features")
    return df_train_enc, df_test_enc


def normalize(X_train, X_val, X_test, models_dir: str = MODELS_DIR):
    """Fit on the training rows only; validation and test are only transformed."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    print("[data] StandardScaler fitted on the training rows and saved.")
    return X_train_sc, scaler.transform(X_val), scaler.transform(X_test)


def save_processed(data: dict, quick: bool = False):
    d = processed_dir(quick)
    os.makedirs(d, exist_ok=True)
    for name in ARRAYS:
        np.save(os.path.join(d, f"{name}.npy"), data[name])
    print(f"[data] Saved processed arrays to {os.path.relpath(d, PROJECT_ROOT)}/")


def load_processed(quick: bool = False):
    d = processed_dir(quick)
    return {n: np.load(os.path.join(d, f"{n}.npy")) for n in ARRAYS}


def processed_data_exists(quick: bool = False):
    d = processed_dir(quick)
    return all(os.path.exists(os.path.join(d, f"{n}.npy")) for n in ARRAYS)


def preprocess(quick: bool = False, data_dir: str = DATA_DIR, save: bool = True):
    print("=" * 60)
    print("DATA PREPROCESSING — NSL-KDD")
    print("=" * 60)

    df_train, df_test                    = load_raw_data(quick=quick, data_dir=data_dir)
    test_type, test_new                  = attack_types(df_train, df_test)
    df_train, y_train, df_test, y_test   = clean_and_label(df_train, df_test)
    df_train_enc, df_test_enc            = encode_categoricals(df_train, df_test)

    X = df_train_enc.values.astype(np.float32)
    y = y_train.values.astype(np.float32)
    X_test = df_test_enc.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # 10% of the training file, stratified, held out for choosing thresholds
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_FRACTION, stratify=y, random_state=SEED)
    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    data = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test, "test_type": test_type, "test_new": test_new}
    if save:
        save_processed(data, quick=quick)

    print("=" * 60)
    print("Preprocessing complete.")
    print(f"  Train : {X_train.shape}  |  Attack rate: {y_train.mean()*100:.1f}%")
    print(f"  Val   : {X_val.shape}   |  Attack rate: {y_val.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape}   |  Attack rate: {y_test.mean()*100:.1f}%  "
          f"|  attacks of types not in training: {int(test_new.sum())}")
    print("=" * 60)

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    preprocess(quick=args.quick)
