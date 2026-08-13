"""Generate false-positive alarm sets for the four datasets used in the paper.

Usage:
    python generate_fp.py --data_root /path/to/raw/datasets \
        --dataset_name ids2017 --model_type svm --split_seed 16

Supported datasets: ids2017, unsw-nb15, kdd, Malicious_TLS.
Supported detectors: svm, ae, n3ic.

The raw dataset directories are expected to contain the original CSV files
(see the README for the expected layout). Alarm sets are written to
./fp_data/{dataset}-csv-{model}/split_seed{seed}/alert.json.
"""
import argparse
import time
from pathlib import Path
from collections import Counter

from utils import (
    load_full_dataset,
    make_train_test_split,
    compute_binary_metrics,
    collect_false_positives,
    build_model,
)
from utils_save import save_fp_samples, save_pvoxel_4files

SUPPORTED_DATASETS = {"ids2017", "unsw-nb15", "kdd", "Malicious_TLS"}
SUPPORTED_MODELS = {"svm", "ae", "n3ic"}

# default sub-directory inside --data_root for each dataset
DATASET_DIRS = {
    "ids2017": "IDS2017/MachineLearningCVE",
    "unsw-nb15": "unsw-nb15/CSV Files",
    "kdd": "kdd",
    "Malicious_TLS": "Malicious_TLS",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="root directory containing the raw datasets")
    parser.add_argument("--data_type", type=str, choices=["csv"], default="csv")
    parser.add_argument("--model_type", type=str, default="svm")
    parser.add_argument("--dataset_name", type=str, default="ids2017")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pos_label", type=int, default=1,
                        help="label treated as malicious/alert")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_type = args.model_type
    pos_label = args.pos_label

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name} "
                         f"(supported: {sorted(SUPPORTED_DATASETS)})")
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported detector: {model_type} "
                         f"(supported: {sorted(SUPPORTED_MODELS)})")

    # ==== 1. Configuration ====
    data_root = Path(args.data_root)
    sub_dir = DATASET_DIRS[dataset_name]
    data_dir = data_root / sub_dir if sub_dir else data_root
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # ==== 2. Load the full dataset ====
    X_all, y_all, meta_all, feature_names = load_full_dataset(
        data_dir, dataset_name, args.data_type)

    # ==== 3. Train/test split ====
    train_idx, test_idx = make_train_test_split(
        y_all, test_size=args.test_size, seed=args.split_seed, stratify=True)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    meta_test = [meta_all[i] for i in test_idx]

    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    print(f"train label distribution: {Counter(y_train)}")
    print(f"test label distribution: {Counter(y_test)}")

    # ==== 4. Build and train the model ====
    model = build_model(model_type)
    if hasattr(model, "set_feature_names"):
        model.set_feature_names(feature_names)
    if hasattr(model, "set_dataset_name"):
        model.set_dataset_name(dataset_name)

    print(f"[INFO] Training model {model_type} ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[INFO] Training done in {time.time() - t0:.1f}s")

    # ==== 5. Predict on the test set ====
    print("[INFO] Predicting on test set ...")
    y_pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)
        y_score = proba[:, pos_label]
    except Exception:
        y_score = None

    metrics = compute_binary_metrics(y_true=y_test, y_pred=y_pred,
                                     pos_label=pos_label)
    print(f"[INFO] Metrics (pos={pos_label}): "
          f"acc={metrics['accuracy']:.4f}, TPR={metrics['TPR']:.4f}, "
          f"FPR={metrics['FPR']:.4f}, F1={metrics['F1']:.4f}")

    # ==== 6. Collect false positives and save the alarm set ====
    fp_idx, fp_stats = collect_false_positives(
        y_true=y_test, y_pred=y_pred, y_score=y_score, pos_label=pos_label)
    print(f"[INFO] Found {fp_stats['num_fp']} FP out of "
          f"{fp_stats['num_samples']} samples "
          f"(rate={fp_stats['fp_rate']:.4f})")

    out_dir = Path(f"./fp_data/{dataset_name}-csv-{model_type}/"
                   f"split_seed{args.split_seed}/")
    save_fp_samples(
        out_dir=out_dir,
        X_test=X_test,
        y_true=y_test,
        y_pred=y_pred,
        pos_label=pos_label,
        model_name=model_type,
        classifier_name=model_type,
        metrics=metrics,
    )

    # Also export FP/TP/FN/TN JSON files for the pVoxel C++ tool
    dataset_tag = f"{dataset_name}-csv-{model_type}-seed{args.split_seed}"
    save_pvoxel_4files(
        out_dir=out_dir,
        X_test=X_test,
        y_true=y_test,
        y_pred=y_pred,
        pos_label=pos_label,
        dataset_tag=dataset_tag,
    )


if __name__ == "__main__":
    main()
