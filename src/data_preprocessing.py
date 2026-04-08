"""
data_preprocessing.py — loads and prepares NSL-KDD for model training.
Pipeline: load → clean/label → one-hot encode → normalize → save
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data", "nslkdd")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "results", "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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


def load_raw_data(quick: bool = False):
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path  = os.path.join(DATA_DIR, "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"NSL-KDD files not found in {DATA_DIR}/\n"
            "Download KDDTrain+.txt and KDDTest+.txt from:\n"
            "  https://www.kaggle.com/datasets/hassan06/nslkdd\n"
            "and place them in data/nslkdd/"
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


def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("[data] StandardScaler fitted and saved.")
    return X_train_sc, X_test_sc


def save_processed(X_train, y_train, X_test, y_test):
    for name, arr in {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}.items():
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), arr)
    print(f"[data] Saved processed arrays to {PROCESSED_DIR}/")


def load_processed():
    return {n: np.load(os.path.join(PROCESSED_DIR, f"{n}.npy")) for n in ["X_train", "y_train", "X_test", "y_test"]}


def processed_data_exists():
    return all(os.path.exists(os.path.join(PROCESSED_DIR, f"{n}.npy")) for n in ["X_train", "y_train", "X_test", "y_test"])


def preprocess(quick: bool = False):
    print("=" * 60)
    print("DATA PREPROCESSING — NSL-KDD")
    print("=" * 60)

    df_train, df_test                    = load_raw_data(quick=quick)
    df_train, y_train, df_test, y_test   = clean_and_label(df_train, df_test)
    df_train_enc, df_test_enc            = encode_categoricals(df_train, df_test)

    X_train = df_train_enc.values.astype(np.float32)
    X_test  = df_test_enc.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_test  = y_test.values.astype(np.float32)

    X_train, X_test = normalize(X_train, X_test)
    save_processed(X_train, y_train, X_test, y_test)

    print("=" * 60)
    print("Preprocessing complete.")
    print(f"  Train : {X_train.shape}  |  Attack rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  : {X_test.shape}   |  Attack rate: {y_test.mean()*100:.1f}%")
    print("=" * 60)

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    preprocess(quick=args.quick)
