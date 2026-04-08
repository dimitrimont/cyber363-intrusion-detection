"""
main.py — entry point for the NSL-KDD intrusion detection pipeline.

Usage:
  python main.py              # full run
  python main.py --quick      # 10% data subset for fast testing
  python main.py --reprocess  # force re-run preprocessing
"""

import os
import sys
import argparse
import random
import time
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true", help="Use 10% of data")
    parser.add_argument("--reprocess", action="store_true", help="Re-run preprocessing")
    return parser.parse_args()


def _clear_results():
    root = os.path.dirname(os.path.abspath(__file__))
    total = 0
    for subdir in ["results/plots", "results/metrics"]:
        d = os.path.join(root, subdir)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                total += 1
    if total:
        print(f"[main] Cleared {total} stale file(s) from results/")


def main():
    args = parse_args()

    print("=" * 60)
    print("  ML FOR INTRUSION DETECTION — NSL-KDD")
    print(f"  quick={args.quick}  reprocess={args.reprocess}")
    print("=" * 60)

    wall_start = time.time()
    _clear_results()

    # Stage 1 — Preprocessing
    from data_preprocessing import preprocess, load_processed, processed_data_exists
    if processed_data_exists() and not args.reprocess:
        print("[main] Processed data found — loading (use --reprocess to redo).")
        data = load_processed()
    else:
        data = preprocess(quick=args.quick)
    print(f"[main] X_train: {data['X_train'].shape}  X_test: {data['X_test'].shape}")

    # Stage 2 — Classical models
    from classical_models import train_all
    _, classical_results = train_all(data)

    # Stage 3 — Deep learning
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from deep_learning_model import train_deep_learning_model
    try:
        _, dl_metrics = train_deep_learning_model(data)
        all_results = classical_results + [dl_metrics]
    except Exception as e:
        print(f"[main] Deep learning failed: {e}")
        all_results = classical_results

    # Stage 4 — Evaluation
    from evaluation import evaluate_all
    df = evaluate_all(all_results, y_test=data["y_test"])

    elapsed = time.time() - wall_start
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Models: {len(all_results)}  |  Results: results/")
    print("=" * 60)
    print("\nTop models by accuracy:")
    print(df["accuracy"].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
