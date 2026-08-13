from pathlib import Path
import numpy as np
import json


def save_fp_samples(
    out_dir: Path,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int,
    model_name: str,
    classifier_name: str,
    metrics: dict,
):
    """
    Instead of saving only FPs, it now:
      - Collect all test samples predicted as pos_label (alarms)
      - Write their features and ground-truth labels into alert.json
      - Additionally write alert_summary.md summarizing alarm/FP counts
      - Also save the model name and metrics (accuracy / TPR / FPR ...)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # indices of all samples predicted as alarms
    alert_idx = np.where(y_pred == pos_label)[0]

    alert_records = []
    for i in alert_idx:
        feat_vec = X_test[i]
        feat_list = [float(x) for x in feat_vec]

        alert_records.append({
            "index": int(i),
            "features": feat_list,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
        })

    # write alert.json (structure unchanged: a flat list)
    alert_path = out_dir / "alert.json"
    with alert_path.open("w", encoding="utf-8") as f:
        json.dump(alert_records, f, indent=2, ensure_ascii=False)

    # count TP / FP within the alarms
    num_alerts = len(alert_idx)
    num_tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    num_fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())

    # write a markdown summary with model info and metrics
    md_path = out_dir / "alert_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Alert Summary\n\n")
        f.write(f"- Model name (config): {model_name}\n")
        f.write(f"- Classifier backend : {classifier_name}\n")
        f.write(f"- Pos label          : {pos_label}\n\n")

        f.write(f"- Total test samples         : {len(y_true)}\n")
        f.write(f"- Total alerts (pred=={pos_label}): {num_alerts}\n")
        f.write(f"- True positives (TP)        : {num_tp}\n")
        f.write(f"- False positives (FP)       : {num_fp}\n")
        if num_alerts > 0:
            fp_rate_in_alerts = num_fp / num_alerts
            f.write(f"- FP ratio among alerts      : {fp_rate_in_alerts:.4f}\n\n")

        # binary metrics
        f.write("## Binary metrics (pos vs others)\n")
        f.write(f"- Accuracy : {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"- Recall/TPR: {metrics.get('TPR', 0):.4f}\n")
        f.write(f"- FPR      : {metrics.get('FPR', 0):.4f}\n")
        f.write(f"- F1-score : {metrics.get('F1', 0):.4f}\n")
        f.write(f"- TP={metrics.get('TP', 0)}, "
                f"FP={metrics.get('FP', 0)}, "
                f"TN={metrics.get('TN', 0)}, "
                f"FN={metrics.get('FN', 0)}\n")

    # also write a JSON for programmatic access
    metrics_path = out_dir / "metrics.json"
    to_dump = {
        "model_name": model_name,         # e.g., "FlowLens" / "rf"
        "classifier_name": classifier_name,  # e.g., "xgb" / "rf"
        "pos_label": int(pos_label),
        "num_test_samples": int(len(y_true)),
        "num_alerts": int(num_alerts),
        "num_tp": int(num_tp),
        "num_fp": int(num_fp),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(to_dump, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {num_alerts} alerts to {alert_path}")
    print(f"[INFO] Summary written to {md_path}")
    print(f"[INFO] Metrics written to {metrics_path}")


def save_pvoxel_4files(
    out_dir: Path,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int,
    dataset_tag: str = "AlertTest",
):
    """
    Generate the four JSON files required by the pVoxel C++ tool:
      {dataset_tag}_FP.json / {dataset_tag}_TP.json / {dataset_tag}_FN.json / {dataset_tag}_TN.json

    Format follows the original pVoxel datasets:
      {
        "result": [
          {
            "code": <0 or 1>,   # y_true is used as the code
            "saddr": "0.0.0.0",
            "daddr": "0.0.0.0",
            "start": 0.0,
            "end": 0.0,
            "len": 1,
            "loss": 0.0,
            "feature": [ ... ]  # only FP/TP carry this field; FN/TN may omit it
          },
          ...
        ]
      }
    The C++ reduce_fp_flow only calls read_matrix (which needs feature) on FP/TP,
    while FN/TN are only used for statistics and may omit feature.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp_list = []
    tp_list = []
    fn_list = []
    tn_list = []

    for i in range(len(y_true)):
        yt = int(y_true[i])
        yp = int(y_pred[i])

        # basic metadata: placeholders when real IP/time are unavailable
        base_rec = {
            "code": yt,              # 0=benign, 1=malicious (same as y_true)
            "saddr": "0.0.0.0",
            "daddr": "0.0.0.0",
            "start": 0.0,
            "end": 0.0,
            "len": 1,
            "loss": 0.0,
        }

        # FP / TP must carry feature (the C++ code converts them to a matrix)
        if yp == pos_label and yt == pos_label:
            rec = dict(base_rec)
            rec["feature"] = [float(x) for x in X_test[i]]
            tp_list.append(rec)
        elif yp == pos_label and yt != pos_label:
            rec = dict(base_rec)
            rec["feature"] = [float(x) for x in X_test[i]]
            fp_list.append(rec)
        elif yp != pos_label and yt == pos_label:
            # FN: truly malicious but not alerted; statistics only, no feature needed
            fn_list.append(dict(base_rec))
        else:
            # TN: truly benign and not alerted
            tn_list.append(dict(base_rec))

    def _dump_json(name: str, data_list: list):
        path = out_dir / f"{dataset_tag}_{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"result": data_list}, f, indent=2, ensure_ascii=False)
        print(f"[pVoxel] {name} file written: {path} (num={len(data_list)})")

    _dump_json("FP", fp_list)
    _dump_json("TP", tp_list)
    _dump_json("FN", fn_list)
    _dump_json("TN", tn_list)