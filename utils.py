import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from detector import TorchAEAnomalyDetector, TorchN3ICDetector


def build_model(model_type: str):
    """Return a scikit-learn-style model according to model_type."""
    if model_type == "svm":
        # Use SGDClassifier + hinge to approximate a linear SVM (multi-threaded)
        # Note: no predict_proba; the caller already handles this via try/except
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="hinge",  # linear SVM
                alpha=1e-4,
                max_iter=20,  # increase if needed
                tol=1e-3,
                n_jobs=-1,  # parallelize
                random_state=42
            )
        )

    elif model_type == "ae":
        # Autoencoder-based anomaly detector
        return TorchAEAnomalyDetector(
            hidden_dims=(128, 32),
            lr=1e-3,
            batch_size=256,
            epochs=20,
            weight_decay=1e-5,
            device="cuda",
            verbose=True,  # show a progress bar
            unsupervised_quantile=0.95,
        )

    elif model_type == "n3ic":
        # N3IC (PyTorch): implemented following the original BNN pipeline
        return TorchN3ICDetector(
            neurons=(64, 32, 2),
            lr=1e-4,
            batch_size=256,
            epochs=15,
            val_size=0.2,
            random_state=0,
            device=None,
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def collect_false_positives(y_true, y_pred, y_score=None, pos_label=1):
    """
    Return the indices of FP samples along with simple statistics.
    By default, pos_label=1 is malicious and 0 is benign.
    FP: predicted as pos_label but the ground truth is not pos_label.
    If only benign-flagged-as-malicious matters, benign can be assumed to be 0:
        fp_mask = (y_true == 0) & (y_pred == pos_label)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp_mask = (y_true != pos_label) & (y_pred == pos_label)
    fp_idx = np.where(fp_mask)[0]

    stats = {
        "num_samples": int(len(y_true)),
        "num_fp": int(len(fp_idx)),
        "fp_rate": float(len(fp_idx) / len(y_true)) if len(y_true) > 0 else 0.0,
        "label_counts": {int(k): int(v) for k, v in Counter(y_true).items()}
    }

    return fp_idx, stats


def load_full_dataset(DATA_ROOT, dataset_name: str, data_type: str):
    """
    Load the full data without any train/test split.
    Returns:
      X_all: (N, D) np.ndarray
      y_all: (N,)   np.ndarray
      meta_all: list[dict] per-row metadata (source file, row index, etc.)

    data_type:
      - "csv": mainly IDS2017 / CICIDS2017
      - "pcap": reuse _load_split but merge everything into a single all set
    """
    if data_type != "csv":
        raise ValueError(f"Unsupported data_type: {data_type} (only csv is supported)")

    name = dataset_name.lower()
    if name in ["ids2017", "malicious_tls", "kdd"]:
        return load_ids2017_merged(DATA_ROOT, dataset_name)
    if "unsw-nb15" in name:
        return load_unsw15_merged(DATA_ROOT, dataset_name)
    raise ValueError(f"Unsupported dataset: {dataset_name} "
                     "(supported: ids2017, Malicious_TLS, kdd, unsw-nb15)")


def load_ids2017_merged(DATA_ROOT, dataset_name: str):
    """
    Low-memory CSV loader for IDS/CIC-style data.

    Supported: ids2017, malicious_tls, kdd.

    Core optimizations:
      1. Do not read all CSVs into memory at once;
      2. Read file by file, chunk by chunk;
      3. Build labels, convert features, and clean NaN/Inf within each chunk;
      4. Keep only cleaned X/y/meta and concatenate numpy arrays at the end.

    Returns:
      X_all: float32 feature matrix
      y_all: int64 binary labels
      meta_all: list[dict]
      feature_names: list[str]
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from collections import Counter
    from typing import List

    ds_dir = Path(DATA_ROOT)
    csv_files = sorted(ds_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {ds_dir}")

    dataset_key_base = dataset_name.lower().replace("_", "-")

    chunksize = 100_000

    def _read_csv_chunks(csv_path: Path):
        """
        Read CSV with a chunksize.
        Try the default encoding first; switch to latin1 if decoding fails at runtime.
        Use low_memory=False to minimize dtype instability.
        """
        try:
            return pd.read_csv(csv_path, low_memory=False, chunksize=chunksize)
        except UnicodeDecodeError:
            return pd.read_csv(csv_path, low_memory=False, chunksize=chunksize, encoding="latin1")

    def _iter_chunks_robust(csv_path: Path):
        """
        pandas' UnicodeDecodeError may surface during iteration; guard against it.
        """
        try:
            for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize):
                yield chunk
        except UnicodeDecodeError:
            for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize, encoding="latin1"):
                yield chunk

    def _align_feature_columns(feature_df: pd.DataFrame, feature_names):
        """
        Keep the feature-column order consistent across chunks.
        """
        if feature_names is None:
            feature_names = list(feature_df.columns)
            return feature_df, feature_names

        missing = [c for c in feature_names if c not in feature_df.columns]
        for c in missing:
            feature_df[c] = np.nan

        # keep only the columns confirmed in the first chunk to avoid dimension drift
        feature_df = feature_df[feature_names]
        return feature_df, feature_names

    def _append_valid_rows(
        X_parts,
        y_parts,
        meta_all,
        feature_df: pd.DataFrame,
        y_chunk: np.ndarray,
        raw_labels,
        origin_file: str,
        label_col: str,
        keep_mask: np.ndarray,
        global_offset: int,
        extra_meta_getter=None,
    ):
        """
        Append the cleaned chunk samples to X_parts/y_parts/meta_all.
        """
        if int(keep_mask.sum()) == 0:
            return

        X_chunk = feature_df.loc[keep_mask].to_numpy(dtype=np.float32)
        y_valid = y_chunk[keep_mask].astype(np.int64)

        X_parts.append(X_chunk)
        y_parts.append(y_valid)

        valid_idx = np.where(keep_mask)[0]
        for local_i in valid_idx:
            rec = {
                "global_index": int(global_offset + local_i),
                "raw_label": str(raw_labels.iloc[local_i]) if hasattr(raw_labels, "iloc") else str(raw_labels[local_i]),
                "label": int(y_chunk[local_i]),
                "origin_file": origin_file,
            }
            if extra_meta_getter is not None:
                rec.update(extra_meta_getter(local_i))
            meta_all.append(rec)

    def _strict_chunk_to_features(df: pd.DataFrame, drop_cols: List[str], feature_names):
        """
        Strict mode:
          - Drop drop_cols;
          - Convert to numeric where possible;
          - Inf -> NaN；
          - Drop rows containing NaN.
        """
        feature_df = df.drop(columns=list(set(drop_cols)), errors="ignore").copy()

        # More robust than the original: some numeric columns are read as object due to         # mixed types; force numeric conversion here.
        for c in feature_df.columns:
            if feature_df[c].dtype == "O":
                feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")

        feature_df = feature_df.select_dtypes(include=["number"]).copy()
        feature_df, feature_names = _align_feature_columns(feature_df, feature_names)

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        keep_mask = (~feature_df.isna().any(axis=1)).to_numpy()

        return feature_df, keep_mask, feature_names

    def _flexible_chunk_to_features(df: pd.DataFrame, drop_cols: List[str], label_col: str, feature_names):
        """
        Relaxed mode:
          - Drop non-feature columns;
          - Try to convert everything to numeric;
          - Drop all-NaN columns in the first chunk;
          - Fill remaining NaN with 0.
        """
        feature_df = df.drop(columns=list(set(drop_cols)), errors="ignore").copy()
        if label_col in feature_df.columns:
            feature_df = feature_df.drop(columns=[label_col])

        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        if feature_names is None:
            non_all_nan_cols = feature_df.columns[feature_df.notna().any(axis=0)]
            feature_df = feature_df[non_all_nan_cols]
            feature_names = list(feature_df.columns)
        else:
            missing = [c for c in feature_names if c not in feature_df.columns]
            for c in missing:
                feature_df[c] = np.nan
            feature_df = feature_df[feature_names]

        feature_df = feature_df.fillna(0.0)
        keep_mask = np.ones(len(feature_df), dtype=bool)
        return feature_df, keep_mask, feature_names

    X_parts = []
    y_parts = []
    meta_all = []
    feature_names = None

    total_rows = 0
    total_kept = 0
    total_dropped = 0
    global_offset = 0

    print(f"[INFO] Loading {dataset_name} from {ds_dir} with chunk size {chunksize}")

    # ========= 1. IDS2017 / Malicious_TLS / KDD =========
    if dataset_key_base in ["ids2017", "malicious-tls", "malicious_tls", "kdd"]:
        if dataset_key_base == "ids2017":
            label_col = " Label"
            benign_str = "BENIGN"
            drop_cols = [" Destination Port", label_col, "__origin_file"]

        elif dataset_key_base in ["malicious-tls", "malicious_tls"]:
            label_col = "Label"
            benign_str = "benign"
            drop_cols = [
                "tls_version", "tls_chosen_cipher_suit",
                "tls_cipher_suites_0", "tls_cipher_suites_1",
                "tls_cipher_suites_2", "tls_cipher_suites_3",
                "tls_cipher_suites_4", label_col, "__origin_file"
            ]

        elif dataset_key_base == "kdd":
            label_col = "Class"
            benign_str = "0"
            drop_cols = [label_col, "__origin_file"]

        for file_id, csv_path in enumerate(csv_files):
            print(f"[INFO] reading {csv_path.name}")
            for chunk_id, df in enumerate(_iter_chunks_robust(csv_path)):
                df["__origin_file"] = csv_path.name
                total_rows += len(df)

                if label_col not in df.columns:
                    raise ValueError(
                        f"Expected column {label_col!r} in {csv_path}, "
                        f"columns={list(df.columns)}"
                    )

                labels_raw = df[label_col].astype(str)

                if dataset_key_base == "kdd":
                    y_chunk = (labels_raw != benign_str).astype(int).to_numpy()
                else:
                    y_chunk = (labels_raw.str.strip() != benign_str).astype(int).to_numpy()

                feature_df, keep_mask, feature_names = _strict_chunk_to_features(
                    df=df,
                    drop_cols=drop_cols,
                    feature_names=feature_names,
                )

                kept = int(keep_mask.sum())
                dropped = int(len(df) - kept)
                total_kept += kept
                total_dropped += dropped

                _append_valid_rows(
                    X_parts=X_parts,
                    y_parts=y_parts,
                    meta_all=meta_all,
                    feature_df=feature_df,
                    y_chunk=y_chunk,
                    raw_labels=df[label_col],
                    origin_file=csv_path.name,
                    label_col=label_col,
                    keep_mask=keep_mask,
                    global_offset=global_offset,
                )

                global_offset += len(df)

                if chunk_id % 10 == 0:
                    print(
                        f"[INFO] {csv_path.name} chunk={chunk_id}, "
                        f"rows={len(df)}, kept={kept}, dropped={dropped}, "
                        f"total_kept={total_kept}"
                    )

def load_unsw15_merged(DATA_ROOT: Path, dataset_name: str):
    """
    Low-memory loader for UNSW-NB15.

    Key changes:
    1. Read chunk by chunk instead of concatenating raw DataFrames first;
    2. Encode categories, numericize, and clean NaN/Inf within each chunk;
    3. Concatenate X/y only at the end.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    ds_dir = Path(DATA_ROOT)

    default_col_names = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb",
        "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
        "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
    ]

    col_names = default_col_names

    # Try the official features file; fall back to default columns if absent.
    feat_candidates = [
        "UNSW_NB15_features.csv",
        "UNSW-NB15_features.csv",
        "NUSW-NB15_features.csv",
        "unsw_nb15_features.csv",
        "unsw-nb15_features.csv",
    ]
    feat_search_dirs = [ds_dir]
    for search_dir in feat_search_dirs:
        found = False
        for name in feat_candidates:
            p = search_dir / name
            if p.exists():
                try:
                    feat_df = pd.read_csv(p, encoding="latin1")
                    if "Name" in feat_df.columns and len(feat_df["Name"]) == 49:
                        col_names = feat_df["Name"].astype(str).tolist()
                        print(f"[INFO] UNSW-NB15 feature description: {p}")
                        found = True
                        break
                except Exception as e:
                    print(f"[WARN] failed to read feature description {p}: {e}")
        if found:
            break

    # locate the data file
    data_files = sorted(ds_dir.glob("UNSW-NB15_*.csv"))
    data_files = [
        p for p in data_files
        if p.stem in {"UNSW-NB15_1", "UNSW-NB15_2", "UNSW-NB15_3", "UNSW-NB15_4"}
    ]
    if not data_files:
        raise FileNotFoundError(f"No UNSW-NB15_1~4.csv found under {ds_dir}")
    print(f"[INFO] Load UNSW-NB15 original files: {[p.name for p in data_files]}")

    label_col = "Label"
    attack_cat_col = "attack_cat"
    categorical_cols = ["proto", "service", "state"]
    drop_cols = {label_col, attack_cat_col, "srcip", "dstip"}

    cat_maps = {c: {} for c in categorical_cols}

    def encode_category_series(s, col):
        mp = cat_maps[col]
        out = []
        for v in s.astype(str).fillna("").tolist():
            if v not in mp:
                mp[v] = len(mp)
            out.append(mp[v])
        return np.asarray(out, dtype=np.int32)

    X_parts = []
    y_parts = []
    meta_all = []

    total_rows = 0
    total_kept = 0
    total_dropped = 0
    global_index = 0

    # reduce further if memory is tight, e.g., 20000.
    chunksize = 50000
    feature_names = None

    for csv_path in data_files:
        print(f"[INFO] reading {csv_path}")

        reader = pd.read_csv(
            csv_path,
            header=None,
            names=col_names,
            encoding="cp1252",
            encoding_errors="replace",
            low_memory=False,
            chunksize=chunksize,
        )

        for chunk_id, df in enumerate(reader):
            total_rows += len(df)

            # labels
            labels_raw = df[label_col]
            if labels_raw.dtype == "O":
                labels_str = labels_raw.astype(str).str.strip()
                uniq = set(labels_str.dropna().unique().tolist())
                if uniq.issubset({"0", "1", "0.0", "1.0"}):
                    y_chunk = labels_str.astype(float).astype(int).to_numpy()
                else:
                    y_chunk = (labels_str.str.lower() != "normal").astype(int).to_numpy()
            else:
                y_chunk = labels_raw.astype(int).to_numpy()

            attack_values = (
                df[attack_cat_col].astype(str).str.strip().tolist()
                if attack_cat_col in df.columns
                else ["unknown"] * len(df)
            )

            # encode categorical columns with a global mapping for cross-chunk consistency
            for c in categorical_cols:
                if c in df.columns:
                    df[c] = encode_category_series(df[c], c)

            # numeric features
            feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # numericize mixed-type columns such as sport/dsport/ct_ftp_cmd
            for c in feature_df.columns:
                if feature_df[c].dtype == "O":
                    feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")

            feature_df = feature_df.select_dtypes(include=["number"]).copy()

            if feature_names is None:
                feature_names = list(feature_df.columns)
                print(f"[INFO] UNSW-NB15 feature dim = {len(feature_names)}")
            else:
                # align feature columns with the first chunk
                missing = [c for c in feature_names if c not in feature_df.columns]
                if missing:
                    for c in missing:
                        feature_df[c] = np.nan
                feature_df = feature_df[feature_names]

            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            keep_mask = ~feature_df.isna().any(axis=1)

            kept = int(keep_mask.sum())
            dropped = int(len(feature_df) - kept)
            total_kept += kept
            total_dropped += dropped

            if kept > 0:
                keep_np = keep_mask.to_numpy()
                X_chunk = feature_df.loc[keep_mask].to_numpy(dtype=np.float32)
                y_valid = y_chunk[keep_np].astype(np.int64)

                X_parts.append(X_chunk)
                y_parts.append(y_valid)

                valid_idx = np.where(keep_np)[0]
                for local_i in valid_idx:
                    meta_all.append({
                        "global_index": int(global_index + local_i),
                        "label": int(y_chunk[local_i]),
                        "raw_label": labels_raw.iloc[local_i],
                        "attack_cat": attack_values[local_i],
                    })

            global_index += len(df)

            if chunk_id % 10 == 0:
                print(
                    f"[INFO] {csv_path.name} chunk={chunk_id}, "
                    f"rows={len(df)}, kept={kept}, dropped={dropped}, "
                    f"total_kept={total_kept}"
                )

    if not X_parts:
        raise RuntimeError("No valid UNSW-NB15 rows after cleaning.")

    X_all = np.vstack(X_parts).astype(np.float32, copy=False)
    y_all = np.concatenate(y_parts).astype(np.int64, copy=False)

    print(
        f"[INFO] UNSW-NB15: total_rows={total_rows}, "
        f"dropped={total_dropped}, remain={total_kept}"
    )
    print(
        f"[INFO] UNSW-NB15 loaded: X_all={X_all.shape}, "
        f"positives={int(y_all.sum())}, negatives={int((y_all == 0).sum())}"
    )
    print(f"[INFO] categorical maps: { {k: len(v) for k, v in cat_maps.items()} }")

    return X_all, y_all, meta_all, list(feature_names)



def make_train_test_split(y_all,
                          test_size: float = 0.3,
                          seed: int = 42,
                          stratify: bool = True):
    """
    Given full labels y_all, return train_idx and test_idx index arrays.
    It does not modify the data itself.
    """
    y_all = np.asarray(y_all)
    indices = np.arange(len(y_all))

    if stratify:
        strat = y_all
    else:
        strat = None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strat
    )
    return train_idx, test_idx


def compute_binary_metrics(y_true, y_pred, pos_label=1):
    """
    Binary metrics computed as pos_label vs. others:
      - accuracy
      - precision
      - recall / TPR
      - FPR
      - F1
    All other classes are treated as negative.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    total = len(y_true_bin)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec_den = (tp + fp)
    precision = tp / prec_den if prec_den > 0 else 0.0
    rec_den = (tp + fn)
    recall = tp / rec_den if rec_den > 0 else 0.0
    fpr_den = (fp + tn)
    fpr = fp / fpr_den if fpr_den > 0 else 0.0
    f1_den = (precision + recall)
    f1 = 2 * precision * recall / f1_den if f1_den > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,  # = TPR
        "TPR": recall,
        "FPR": fpr,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


