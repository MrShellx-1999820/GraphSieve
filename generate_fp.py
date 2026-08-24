import re
import argparse, os, shutil, time, json
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
from utils import load_full_dataset, make_train_test_split, compute_binary_metrics, collect_false_positives
from flowlens_features import load_flowlens_dataset
from utils import build_model
from utils_save import save_fp_samples, save_pvoxel_4files

"""
    1. Choose a dataset
    2. Choose the feature type (pcap / csv)
    3. Choose a model
    4. Generate false-positive alarms

    The ./fp_data directory stores the generated alarm data. Sub-directories are named {dataset_name}-{data_type}-{model_type}; each contains the test alarm set, and each record carries its ground-truth label.

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, choices=["pcap", "csv"], default="csv")
    parser.add_argument("--model_type", type=str, help="detector type",
                        choices=["rf", "svm", "tree", "FlowLens", "xgb", "ae", "n3ic"], default="svm")
    parser.add_argument("--dataset_name", type=str, default="dapt-2020")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pos_label", type=int, default=1, help="label treated as malicious/alert")
    args = parser.parse_args()

    # ==== 1. Configuration ====
    data_type = args.data_type
    dataset_name = args.dataset_name
    model_type = args.model_type
    pos_label = args.pos_label

    DATA_ROOT = f"/data1/lx/dataset/{dataset_name}"
    # === Load data according to model_type ===
    feature_names = None
    if model_type == "FlowLens":
        # FlowLens: use PCAP + FlowLens features
        if data_type != "pcap":
            raise ValueError("model_type == 'FlowLens' requires --data_type pcap")
        X_all, y_all, meta_all = load_flowlens_dataset(dataset_name)
    else:
        # Default: CSV-based features
        if data_type == 'csv' and dataset_name == 'ids2017':
            DATA_ROOT = Path("/data1/lx/multi-rag/dataset/IDS2017/MachineLearningCVE")
        elif data_type == 'csv' and dataset_name == 'unsw-nb15':
            DATA_ROOT = Path("/data1/lx/dataset/unsw-nb15/CSV Files")
        elif data_type == 'csv' and dataset_name == "dapt-2020":
            DATA_ROOT = Path("/data1/lx/dataset/dapt-2020-csv")
        elif data_type == 'csv' and dataset_name == "ddos-2019":
            DATA_ROOT = Path("/data1/lx/multi-rag/dataset/CICDDoS2019")
        elif data_type == 'csv' and dataset_name == "Malicious_TLS":
            DATA_ROOT = Path("/data1/lx/dataset/Malicious_TLS/")
        elif data_type == 'csv' and dataset_name == "kdd":
            DATA_ROOT = Path("/data1/lx/dataset/kdd/")
        elif data_type == 'csv' and dataset_name == "iot-2023":
            DATA_ROOT = Path("/data1/lx/dataset/CICIoT2023/")
        elif data_type == 'csv' and dataset_name == 'unsw-nb15-lowvar':
            DATA_ROOT = Path("/data1/lx/dataset/unsw-nb15-lowvar")
        elif data_type == 'csv' and dataset_name == "ids2017-lowvar":
            DATA_ROOT = Path("/data1/lx/dataset/ids2017-lowvar")
        else:
            raise ValueError(f"unsupported combination: data_type={data_type}, dataset_name={dataset_name}")

        X_all, y_all, meta_all, feature_names = load_full_dataset(DATA_ROOT, dataset_name, data_type)

    if model_type == "n3ic":
        if data_type != "csv":
            raise ValueError("model_type == 'n3ic' requires --data_type csv")
        supported_ds = {"unsw-nb15", "ids2017", "kdd", "malicious_tls"}
        if dataset_name.lower() not in supported_ds:
            raise ValueError(
                "n3ic currently supports dataset_name in {unsw-nb15, ids2017, kdd, Malicious_TLS}"
            )

    # === 2. Online train/test split ===
    train_idx, test_idx = make_train_test_split(
        y_all,
        test_size=args.test_size,
        seed=args.split_seed,
        stratify=True
    )

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    meta_test = [meta_all[i] for i in test_idx]

    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    print(f"meta_test length: {len(meta_all)}")
    print(f"train label distribution: {Counter(y_train)}")
    print(f"test label distribution: {Counter(y_test)}")

    # === 3. Build and train the model ===
    # FlowLens is a feature type; default to xgb, keep everything else unchanged
    effective_model_type = model_type
    if model_type == "FlowLens":
        effective_model_type = "xgb"

    print(f"[INFO] Building model {model_type} (classifier={effective_model_type}) ...")
    model = build_model(effective_model_type)
    if hasattr(model, "set_feature_names"):
        model.set_feature_names(feature_names)
    if hasattr(model, "set_dataset_name"):
        model.set_dataset_name(dataset_name)

    print("[INFO] Training model ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[INFO] Training done in {time.time() - t0:.1f}s")

    # === 4. Predict on the test set ===
    print("[INFO] Predicting on test set ...")
    y_pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)
        # assume the pos_label corresponds to column index pos_label
        y_score = proba[:, pos_label]
    except Exception:
        y_score = None

    # === 4.1 Overall metrics (accuracy / TPR / FPR ...) ===
    metrics = compute_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        pos_label=pos_label,
    )
    print(f"[INFO] Metrics (pos={pos_label}): "
          f"acc={metrics['accuracy']:.4f}, "
          f"TPR={metrics['TPR']:.4f}, "
          f"FPR={metrics['FPR']:.4f}, "
          f"F1={metrics['F1']:.4f}")

    # === 5. Collect false positives ===
    fp_idx, fp_stats = collect_false_positives(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        pos_label=pos_label
    )
    print(f"[INFO] Found {fp_stats['num_fp']} FP out of {fp_stats['num_samples']} samples "
          f"(rate={fp_stats['fp_rate']:.4f})")

    out_dir = Path(f"./fp_data/{dataset_name}-{data_type}-{model_type}/"
                   f"split_seed{args.split_seed}/")
    save_fp_samples(
        out_dir=Path(out_dir),
        X_test=X_test,
        y_true=y_test,
        y_pred=y_pred,
        pos_label=pos_label,
        model_name=model_type,  # e.g., "rf" / "FlowLens"
        classifier_name=effective_model_type,  # e.g., "rf" / "xgb"
        metrics=metrics,
    )

    # Also export FP/TP/FN/TN JSON files for the pVoxel C++ tool
    dataset_tag = f"{dataset_name}-{data_type}-{model_type}-seed{args.split_seed}"
    save_pvoxel_4files(
        out_dir=Path(out_dir),
        X_test=X_test,
        y_true=y_test,
        y_pred=y_pred,
        pos_label=pos_label,
        dataset_tag=dataset_tag,
    )
