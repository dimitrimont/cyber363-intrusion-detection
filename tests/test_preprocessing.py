import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

import data_preprocessing as dp


def write_kdd(path, rows):
    """rows: (protocol, service, flag, src_bytes, label)"""
    out = []
    for proto, service, flag, nbytes, label in rows:
        r = {c: 0 for c in dp.COLUMNS}
        r.update(protocol_type=proto, service=service, flag=flag, src_bytes=nbytes, label=label, difficulty=20)
        out.append(r)
    pd.DataFrame(out, columns=dp.COLUMNS).to_csv(path, header=False, index=False)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    train = [("tcp", "http", "SF", int(rng.integers(100, 100_000)), "normal") for _ in range(300)]
    train += [("tcp", "private", "S0", int(rng.integers(0, 100_000)), "neptune") for _ in range(200)]
    test = [("tcp", "http", "SF", 200, "normal"), ("udp", "snmp", "SF", 50, "snmpguess"), ("tcp", "private", "S0", 1, "neptune")]
    write_kdd(tmp_path / "KDDTrain+.txt", train)
    write_kdd(tmp_path / "KDDTest+.txt", test * 10)
    monkeypatch.setattr(dp, "PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(dp, "MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


def test_unseen_categories_become_zeros(tiny):
    d = dp.preprocess(data_dir=str(tiny), save=False)
    train_cols = pd.get_dummies(pd.DataFrame({"service": ["http", "private"]})).shape[1]
    assert d["X_test"].shape[1] == d["X_train"].shape[1]
    assert train_cols == 2  # udp and snmp never appear in training, so they add no columns


def test_new_attack_types_are_marked(tiny):
    d = dp.preprocess(data_dir=str(tiny), save=False)
    assert list(d["test_type"][:3]) == ["normal", "snmpguess", "neptune"]
    assert list(d["test_new"][:3]) == [False, True, False]


def test_scaler_is_fitted_on_training_rows_only(tiny):
    d = dp.preprocess(data_dir=str(tiny), save=False)
    assert np.allclose(d["X_train"].mean(axis=0), 0, atol=1e-5)
    n = len(d["X_train"]) + len(d["X_val"])
    assert len(d["X_val"]) == pytest.approx(0.1 * n, abs=1)  # 10% of the training rows, held out for thresholds


def test_quick_and_full_arrays_are_kept_apart(tiny):
    dp.preprocess(data_dir=str(tiny), quick=False)
    assert dp.processed_data_exists(quick=False)
    assert not dp.processed_data_exists(quick=True)  # the first version would have reused the full arrays here
    full_rows = len(dp.load_processed(quick=False)["X_train"])
    dp.preprocess(data_dir=str(tiny), quick=True)
    assert len(dp.load_processed(quick=False)["X_train"]) == full_rows  # and a quick run no longer overwrites them


def test_svm_calibration_does_not_change_its_decisions():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(0, 1, (80, 3)), rng.normal(2, 1, (80, 3))]
    y = np.r_[np.zeros(80), np.ones(80)]
    plain = SVC(random_state=42).fit(X, y).predict(X)
    calibrated = SVC(probability=True, random_state=42).fit(X, y).predict(X)
    assert (plain == calibrated).all()
