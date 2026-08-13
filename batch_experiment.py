"""
GraphSieve Batch Experiment Runner (optimized).

Optimizations:
- kNN graph built once per k value, reused across resolutions
- Large validation sets subsampled for grid search
- Results written incrementally to prevent data loss

Usage:
    python batch_experiment.py                          # run all
    python batch_experiment.py --datasets Malicious_TLS --models svm
"""

import json
import argparse
import time
import os
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist, euclidean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc as sk_auc
import igraph as ig
import leidenalg


# =============================================================================
# Data Loading & Transform
# =============================================================================

def load_alert_data(alert_path):
    with open(alert_path, "r") as f:
        alerts = json.load(f)
    X = np.array([a["features"] for a in alerts], dtype=np.float64)
    y_true = np.array([a["y_true"] for a in alerts], dtype=int)
    return X, y_true


def pvoxel_transform(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    X_norm = np.clip((X - X_min) / denom, 0.0, 1.0)
    return np.log2(1.0 + X_norm)


def variance_feature_select(X, top_k=20):
    """Unsupervised feature selection: keep top-k features by variance.

    After pVoxel transform all features are in [0,1]. Low-variance features
    are near-constant and uninformative for clustering.
    """
    variances = X.var(axis=0)
    return np.argsort(variances)[::-1][:top_k].tolist()


# =============================================================================
# Graph Construction & Clustering
# =============================================================================

def build_knn_graph(X, k):
    n = X.shape[0]
    nn_k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dist_all, idx_all = nn.kneighbors(X)
    kdist = dist_all[:, -1]

    real_k = nn_k - 1
    sources = np.repeat(np.arange(n), real_k)
    targets = idx_all[:, 1:].flatten()
    distances = dist_all[:, 1:].flatten()

    # Cap weights to avoid numerical issues in Leiden
    weights = 1.0 / (distances + 1e-4)
    weights = np.clip(weights, 0, 1e4)

    G = ig.Graph(n=n, edges=list(zip(sources, targets)),
                 edge_attrs={"weight": weights})
    # Simplify: remove duplicate edges and self-loops (keep max weight)
    G.simplify(multiple=True, loops=False, combine_edges="max")
    weights = np.array(G.es["weight"])

    return G, weights, kdist


def run_leiden(G, weights, resolution=1.0, seed=42, max_iter=10):
    """Run Leiden with iteration cap to prevent hangs."""
    partition = leidenalg.find_partition(
        G, leidenalg.RBConfigurationVertexPartition,
        weights=weights, resolution_parameter=resolution,
        n_iterations=max_iter, seed=seed,
    )
    return np.array(partition.membership)


def is_graph_too_dense(G, max_avg_degree=25):
    """Check if graph is too dense for high-resolution Leiden."""
    avg_degree = 2.0 * G.ecount() / max(G.vcount(), 1)
    return avg_degree > max_avg_degree


def run_leiden_safe(G, weights, resolution=1.0, seed=42, max_iter=10):
    """Leiden with fallback: if it fails/hangs, return each node as its own cluster."""
    try:
        return run_leiden(G, weights, resolution, seed, max_iter)
    except Exception:
        # Fallback: each node is its own cluster
        return np.arange(G.vcount(), dtype=np.int32)


def merge_micro_clusters(X, labels, min_size=10):
    unique_labels, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique_labels, counts))
    core_mask = np.array([size_map[l] >= min_size for l in labels])
    micro_mask = ~core_mask
    if micro_mask.sum() == 0:
        return labels
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X[core_mask], labels[core_mask])
    new_labels = labels.copy()
    new_labels[micro_mask] = knn.predict(X[micro_mask])
    return new_labels


# =============================================================================
# Cluster Profiling & Unsupervised Scoring
# =============================================================================

def compute_cluster_profiles(X, cluster_labels, kdist=None):
    unique_labels = np.unique(cluster_labels)
    global_center = X.mean(axis=0, keepdims=True)

    valid = cluster_labels >= 0
    if valid.sum() == 0:
        return pd.DataFrame()
    largest_id = np.argmax(np.bincount(cluster_labels[valid]))
    anchor_center = X[cluster_labels == largest_id].mean(axis=0, keepdims=True)

    rows = []
    for lab in unique_labels:
        idx = np.where(cluster_labels == lab)[0]
        Xc = X[idx]
        size = len(idx)
        centroid = Xc.mean(axis=0, keepdims=True)
        dists = cdist(Xc, centroid, metric="euclidean").flatten()
        avg_intra = dists.mean()
        max_rad = dists.max()
        density = size / (avg_intra + 1e-6)

        row = {
            "cluster_id": int(lab),
            "size": size,
            "log_size": np.log1p(size),
            "dist_to_global": euclidean(centroid[0], global_center[0]),
            "dist_to_anchor": euclidean(centroid[0], anchor_center[0]),
            "avg_intra_dist": avg_intra,
            "max_radius": max_rad,
            "density_score": density,
            "density_size": density / size,
            "log_density": np.log1p(density),
        }
        if kdist is not None:
            row["avg_kdist"] = kdist[idx].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def unsupervised_score(df_profiles, method="combined"):
    """
    Pure unsupervised cluster FP scoring. NO labels used at inference.

    Only uses intrinsic cluster properties (no anchor-dependent signals).
    The validation set selects the best method+direction; test set applies blindly.
    """
    df = df_profiles.copy()

    def _norm(arr, invert=False):
        if arr.max() > arr.min():
            n = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            n = np.full_like(arr, 0.5)
        return (1.0 - n) if invert else n

    parts = method.rsplit("_", 1)
    signal = parts[0]
    direction = parts[1] if len(parts) > 1 else "auto"

    if signal == "density":
        raw = df["density_size"].values
        df["fp_score"] = _norm(raw, invert=(direction == "low"))

    elif signal == "intra":
        raw = df["avg_intra_dist"].values
        df["fp_score"] = _norm(raw, invert=(direction == "low"))

    elif signal == "kdist":
        raw = df["avg_kdist"].values if "avg_kdist" in df.columns else np.zeros(len(df))
        df["fp_score"] = _norm(raw, invert=(direction == "low"))

    elif signal == "size":
        raw = df["size"].values.astype(float)
        df["fp_score"] = _norm(raw, invert=(direction == "small"))

    elif signal == "compact":
        # Compactness = size / (avg_intra_dist * log(size))
        # High compactness = tight cluster
        size = df["size"].values.astype(float)
        intra = df["avg_intra_dist"].values
        compact = size / (intra * np.log1p(size) + 1e-10)
        df["fp_score"] = _norm(compact, invert=(direction == "low"))

    elif signal == "isolation":
        # Isolation = avg_kdist / avg_intra_dist
        # High isolation = points far from neighbors but close within cluster
        # This captures "tight island" pattern
        kd = df["avg_kdist"].values if "avg_kdist" in df.columns else np.ones(len(df))
        intra = df["avg_intra_dist"].values
        iso = kd / (intra + 1e-10)
        df["fp_score"] = _norm(iso, invert=(direction == "low"))

    elif signal == "combined":
        # Multi-signal: average of intrinsic signals (no anchor)
        invert = (direction == "inv")
        signals = [
            _norm(df["density_size"].values, invert=True),
            _norm(df["avg_intra_dist"].values),
        ]
        if "avg_kdist" in df.columns:
            signals.append(_norm(df["avg_kdist"].values))
        score = np.mean(signals, axis=0)
        df["fp_score"] = (1.0 - score) if invert else score

    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return df


# All candidate scoring methods (no anchor-dependent ones)
SCORING_METHODS = [
    "density_low", "density_high",
    "intra_high", "intra_low",
    "kdist_high", "kdist_low",
    "size_small", "size_large",
    "compact_high", "compact_low",
    "isolation_high", "isolation_low",
    "combined_fwd", "combined_inv",
]


# =============================================================================
# Evaluation
# =============================================================================

def cluster_sample_scores(cluster_labels, df_scored, min_cluster_size=5,
                          kdist=None, sample_blend=0.0):
    """Per-sample FP score from cluster scores + optional sample-level blending."""
    cid_score = dict(zip(df_scored["cluster_id"], df_scored["fp_score"]))
    cid_size = dict(zip(df_scored["cluster_id"], df_scored["size"]))

    sample_score = np.array([
        cid_score.get(cid, 0.0) if cid_size.get(cid, 0) >= min_cluster_size else 0.0
        for cid in cluster_labels
    ])

    if sample_blend > 0 and kdist is not None:
        kd = kdist.copy()
        kd_min, kd_max = kd.min(), kd.max()
        sample_outlier = ((kd - kd_min) / (kd_max - kd_min)
                          if kd_max > kd_min else np.zeros_like(kd))
        sample_score = (1.0 - sample_blend) * sample_score + sample_blend * sample_outlier

    return sample_score


def eval_cluster_scoring(cluster_labels, y_true, df_scored,
                         min_cluster_size=5, benign_label=0, pos_label=1,
                         kdist=None, sample_blend=0.0, fixed_tau=None):
    """
    Evaluate cluster scoring with optional sample-level blending.

    sample_blend: 0.0 = pure cluster-level, 1.0 = pure sample-level.
    Sample-level signal: global kdist (k-th nearest neighbor distance).
    fixed_tau: if given, metrics are computed at this threshold (selected on
               validation set) instead of being re-optimized on y_true.
    """
    sample_score = cluster_sample_scores(cluster_labels, df_scored,
                                         min_cluster_size, kdist, sample_blend)
    return eval_sample_scores(sample_score, y_true, benign_label, pos_label,
                              fixed_tau=fixed_tau)


def eval_sample_scores(sample_scores, y_true, benign_label=0, pos_label=1,
                       fixed_tau=None, rtpr_floor=0.02, min_rfpr=0.3):
    """
    Evaluate per-sample FP scores.

    Operating point selection (only when fixed_tau is None — intended for the
    validation set): maximize CER = R.FPR / R.TPR subject to
        R.FPR >= min_rfpr  (must remove a meaningful fraction of FPs)
        R.TPR >= rtpr_floor (denominator floor — keeps CER bounded & stable)
    Fallback: max profit (R.FPR - R.TPR) if no point satisfies the constraints.

    When fixed_tau is given (test-set evaluation), metrics are computed at
    that threshold directly — no test labels are used for threshold selection.

    AUC is always computed from the full threshold curve on y_true.
    """
    fp_total = int((y_true == benign_label).sum())
    tp_total = int((y_true == pos_label).sum())
    if fp_total == 0 or tp_total == 0:
        return None, None

    def _metrics_at(tau):
        mask = sample_scores >= tau
        fp_r = int(((y_true == benign_label) & mask).sum())
        tp_r = int(((y_true == pos_label) & mask).sum())
        return {"tau": float(tau),
                "R.FPR": fp_r / fp_total, "R.TPR": tp_r / tp_total,
                "fp_removed": fp_r, "tp_removed": tp_r}

    thresholds = np.sort(np.unique(sample_scores))[::-1]
    curve = []
    best, best_cer = None, -1

    for tau in thresholds:
        m = _metrics_at(tau)
        curve.append({"tau": m["tau"], "R.FPR": m["R.FPR"], "R.TPR": m["R.TPR"]})
        if fixed_tau is None:
            cer = m["R.FPR"] / max(m["R.TPR"], rtpr_floor)
            if m["R.FPR"] >= min_rfpr and m["R.TPR"] >= rtpr_floor and cer > best_cer:
                best_cer = cer
                best = m

    if fixed_tau is not None:
        best = _metrics_at(fixed_tau)
    elif best is None:
        # Fallback: max profit
        best_profit = -1
        for c in curve:
            profit = c["R.FPR"] - c["R.TPR"]
            if profit > best_profit:
                best_profit = profit
                best = _metrics_at(c["tau"])

    xs = [0.0] + [c["R.TPR"] for c in curve] + [1.0]
    ys = [0.0] + [c["R.FPR"] for c in curve] + [1.0]
    curve_auc = sk_auc(xs, ys)

    if best:
        best["auc"] = float(curve_auc)
        # CER reported with denominator floored at rtpr_floor to avoid
        # degenerate ratios (R.TPR -> 0 makes CER explode and unstable)
        best["CER"] = float(best["R.FPR"] / max(best["R.TPR"], rtpr_floor))
    return best, curve


# =============================================================================
# Unsupervised Baselines (NO labels for training, light tuning on val AUC)
# =============================================================================

def run_baseline_if(X_ref, X_test, contamination="auto", n_estimators=200, seed=42):
    """Isolation Forest: fit on reference data (no label separation), score test."""
    if len(X_ref) < 10:
        return None
    model = IsolationForest(
        n_estimators=n_estimators, max_samples="auto",
        contamination=contamination, random_state=seed, n_jobs=-1,
    )
    model.fit(X_ref)
    return model.score_samples(X_test)  # higher = more normal = likely FP


def run_baseline_lof(X_ref, X_test, n_neighbors=20):
    """LOF (novelty mode): fit on reference data, score test."""
    if len(X_ref) < n_neighbors + 1:
        return None
    from sklearn.neighbors import LocalOutlierFactor
    model = LocalOutlierFactor(
        n_neighbors=min(n_neighbors, len(X_ref) - 1),
        novelty=True, contamination="auto", n_jobs=-1,
    )
    model.fit(X_ref)
    return model.score_samples(X_test)  # higher = more normal = likely FP


def run_baseline_knn_dist(X_ref, X_test, k=20):
    """kNN distance: fit kNN on reference, use -distance as normality score."""
    if len(X_ref) < k + 1:
        return None
    k = min(k, len(X_ref) - 1)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_ref)
    dist, _ = nn.kneighbors(X_test)
    return -dist[:, -1]  # negative k-th NN distance; higher = closer = likely FP




# =============================================================================
# Single Experiment (optimized)
# =============================================================================

MAX_VAL_FOR_GRIDSEARCH = 20000  # subsample validation set if larger


def run_single_experiment(alert_path, val_ratio=0.3, val_seed=42,
                          top_k_features=20, k_list=None, res_list=None,
                          ms_list=None, cluster_seed=42):
    if k_list is None:
        k_list = [10, 30]
    if res_list is None:
        res_list = [0.5, 1.0, 10.0]
    if ms_list is None:
        ms_list = [5, 10]

    X, y_true = load_alert_data(alert_path)
    fp_total = int((y_true == 0).sum())
    tp_total = int((y_true == 1).sum())
    if fp_total < 10 or tp_total < 10:
        return None

    # Split
    idx = np.arange(len(y_true))
    idx_val, idx_test = train_test_split(
        idx, test_size=1.0 - val_ratio,
        random_state=val_seed, stratify=y_true,
    )
    X_val, y_val = X[idx_val], y_true[idx_val]
    X_test, y_test = X[idx_test], y_true[idx_test]

    # Transform
    X_val_t = pvoxel_transform(X_val)
    X_test_t = pvoxel_transform(X_test)

    # Unsupervised feature selection (variance-based, NO labels)
    sel_dims = variance_feature_select(X_val_t, top_k=top_k_features)
    X_val_sel = X_val_t[:, sel_dims]
    X_test_sel = X_test_t[:, sel_dims]

    # Subsample validation set for grid search if too large
    if len(X_val_sel) > MAX_VAL_FOR_GRIDSEARCH:
        rng = np.random.default_rng(val_seed)
        sub_idx = rng.choice(len(X_val_sel), MAX_VAL_FOR_GRIDSEARCH, replace=False)
        X_val_gs = X_val_sel[sub_idx]
        y_val_gs = y_val[sub_idx]
    else:
        X_val_gs = X_val_sel
        y_val_gs = y_val

    result = {
        "n_val": len(idx_val), "n_test": len(idx_test),
        "fp_test": int((y_test == 0).sum()),
        "tp_test": int((y_test == 1).sum()),
        "n_features_selected": len(sel_dims),
    }

    # =========================================================================
    # GraphSieve: Grid search (kNN graph built once per k)
    # Phase 1: search clustering params + scoring method (pure cluster-level)
    # Phase 2: search sample_blend on the best config (hybrid cluster+sample)
    # =========================================================================
    best_val = None
    best_val_auc = -1
    _verbose = os.environ.get("GS_VERBOSE", "0") == "1"

    # Phase 1: clustering + scoring (pure cluster-level, blend=0)
    for k in k_list:
        try:
            _t = time.time()
            G, w, kdist = build_knn_graph(X_val_gs, k)
            if _verbose:
                print(f"    [kNN k={k}] {time.time()-_t:.1f}s", flush=True)
        except Exception:
            continue

        # Check graph density - skip high resolution on dense graphs (causes Leiden hang)
        skip_high_res = is_graph_too_dense(G, max_avg_degree=25)
        if _verbose and skip_high_res:
            avg_deg = 2.0 * G.ecount() / max(G.vcount(), 1)
            print(f"    [WARN] Graph too dense (avg_degree={avg_deg:.1f}), "
                  f"skipping res>=10", flush=True)

        for res in res_list:
            if skip_high_res and res >= 10.0:
                if _verbose:
                    print(f"    [SKIP] res={res} on dense graph", flush=True)
                continue
            try:
                _t = time.time()
                cl = run_leiden(G, w, resolution=res, seed=cluster_seed)
                if _verbose:
                    n_cl = len(np.unique(cl))
                    print(f"    [Leiden res={res}] {time.time()-_t:.1f}s clusters={n_cl}", flush=True)
            except Exception:
                continue

            for ms in ms_list:
                try:
                    cl_m = merge_micro_clusters(X_val_gs, cl, min_size=ms)
                    df = compute_cluster_profiles(X_val_gs, cl_m, kdist)

                    for sm in SCORING_METHODS:
                        try:
                            df_s = unsupervised_score(df, method=sm)
                            r, _ = eval_cluster_scoring(cl_m, y_val_gs, df_s,
                                                        kdist=kdist, sample_blend=0.0)
                            if r and r["auc"] > best_val_auc:
                                best_val_auc = r["auc"]
                                best_val = {"k": k, "res": res, "ms": ms,
                                            "scoring": sm, "blend": 0.0}
                        except Exception:
                            continue
                except Exception:
                    continue

    # Phase 2: try sample_blend on the best config
    if best_val is not None:
        blend_candidates = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        for blend in blend_candidates:
            try:
                G, w, kdist = build_knn_graph(X_val_gs, best_val["k"])
                cl = run_leiden(G, w, resolution=best_val["res"], seed=cluster_seed)
                cl_m = merge_micro_clusters(X_val_gs, cl, min_size=best_val["ms"])
                df = compute_cluster_profiles(X_val_gs, cl_m, kdist)
                df_s = unsupervised_score(df, method=best_val["scoring"])
                r, _ = eval_cluster_scoring(cl_m, y_val_gs, df_s,
                                            kdist=kdist, sample_blend=blend)
                if r and r["auc"] > best_val_auc:
                    if _verbose:
                        print(f"    [blend={blend}] AUC {best_val_auc:.4f} → {r['auc']:.4f}",
                              flush=True)
                    best_val_auc = r["auc"]
                    best_val["blend"] = blend
            except Exception:
                continue

    if _verbose and best_val:
        print(f"    [BEST] k={best_val['k']} res={best_val['res']} "
              f"ms={best_val['ms']} scoring={best_val['scoring']} "
              f"blend={best_val.get('blend', 0.0)} "
              f"val_AUC={best_val_auc:.4f}", flush=True)

    if best_val is None:
        result["graphsieve"] = None
    else:
        result["best_params"] = best_val
        result["val_auc"] = best_val_auc

        # Select operating point (tau) on the FULL validation set with the
        # best config — the test set is never used for threshold selection.
        tau = None
        try:
            _t = time.time()
            G_v, w_v, kdist_v = build_knn_graph(X_val_sel, best_val["k"])
            res_v = best_val["res"]
            if is_graph_too_dense(G_v, max_avg_degree=25) and res_v >= 10.0:
                res_v = min(r for r in res_list if r < 10.0)
            cl_v = run_leiden(G_v, w_v, resolution=res_v, seed=cluster_seed)
            cl_v = merge_micro_clusters(X_val_sel, cl_v, min_size=best_val["ms"])
            df_v = compute_cluster_profiles(X_val_sel, cl_v, kdist_v)
            df_v = unsupervised_score(df_v, method=best_val["scoring"])
            r_val, _ = eval_cluster_scoring(cl_v, y_val, df_v,
                                            kdist=kdist_v,
                                            sample_blend=best_val.get("blend", 0.0))
            if r_val:
                tau = r_val["tau"]
                best_val["tau"] = tau
                result["val_op"] = {"tau": tau, "R.FPR": r_val["R.FPR"],
                                    "R.TPR": r_val["R.TPR"], "CER": r_val["CER"]}
            if _verbose:
                print(f"    [val op-point] tau={tau:.4f} R.FPR={r_val['R.FPR']:.3f} "
                      f"R.TPR={r_val['R.TPR']:.4f} ({time.time()-_t:.1f}s)",
                      flush=True)
        except Exception as e:
            if _verbose:
                print(f"    [val op-point] FAILED: {e}", flush=True)

        # Apply to test set
        try:
            _t = time.time()
            G, w, kdist = build_knn_graph(X_test_sel, best_val["k"])
            if _verbose:
                print(f"    [test kNN k={best_val['k']}] {time.time()-_t:.1f}s", flush=True)

            # Density check for test set too
            if is_graph_too_dense(G, max_avg_degree=25) and best_val["res"] >= 10.0:
                fallback_res = min(r for r in res_list if r < 10.0)
                if _verbose:
                    print(f"    [WARN] Test graph too dense, "
                          f"fallback res={best_val['res']}→{fallback_res}", flush=True)
                best_val = dict(best_val)
                best_val["res"] = fallback_res

            _t = time.time()
            cl = run_leiden(G, w, resolution=best_val["res"], seed=cluster_seed)
            if _verbose:
                print(f"    [test Leiden] {time.time()-_t:.1f}s", flush=True)
            cl = merge_micro_clusters(X_test_sel, cl, min_size=best_val["ms"])
            df = compute_cluster_profiles(X_test_sel, cl, kdist)
            df = unsupervised_score(df, method=best_val["scoring"])
            r_test, _ = eval_cluster_scoring(cl, y_test, df,
                                             kdist=kdist,
                                             sample_blend=best_val.get("blend", 0.0),
                                             fixed_tau=tau)
            result["graphsieve"] = r_test
        except Exception as e:
            result["graphsieve"] = None
            result["graphsieve_error"] = str(e)

    # =========================================================================
    # Unsupervised Baselines with light parameter tuning on validation set
    # Each baseline tunes ONE parameter with a few values (light, not exhaustive)
    # NOTE: baselines use FULL features (no feature selection) — feature
    # selection is part of GraphSieve's contribution, not shared with baselines.
    # =========================================================================

    # Baselines use full features (pVoxel transformed but NOT feature-selected)
    X_val_full = X_val_t
    X_test_full = X_test_t

    # Subsample validation set for baseline fitting (LOF is O(n^2) or worse)
    MAX_VAL_FOR_BASELINE = 5000
    if len(X_val_full) > MAX_VAL_FOR_BASELINE:
        rng = np.random.default_rng(val_seed)
        bl_idx = rng.choice(len(X_val_full), MAX_VAL_FOR_BASELINE, replace=False)
        X_val_bl = X_val_full[bl_idx]
        y_val_bl = y_val[bl_idx]
    else:
        X_val_bl = X_val_full
        y_val_bl = y_val

    def _tune_and_eval(run_fn, param_name, param_values, X_ref, y_ref, X_eval, y_eval):
        """Light tuning: try a few values for one param, select best on val AUC.
        Operating point (tau) is selected on validation scores and transferred
        to the test set — no test labels used for threshold selection."""
        best_auc = -1
        best_pv = param_values[len(param_values) // 2]  # default to middle value

        for pv in param_values:
            try:
                val_scores = run_fn(X_ref, X_ref, **{param_name: pv})
                if val_scores is None:
                    continue
                r, _ = eval_sample_scores(val_scores, y_ref)
                if r and r["auc"] > best_auc:
                    best_auc = r["auc"]
                    best_pv = pv
            except Exception:
                continue

        # Select tau on validation scores at the best param
        try:
            val_scores = run_fn(X_ref, X_ref, **{param_name: best_pv})
            r_val, _ = eval_sample_scores(val_scores, y_ref)
            tau = r_val["tau"] if r_val else None
        except Exception:
            tau = None

        # Apply best param + transferred tau: fit on ref, score on eval
        try:
            test_scores = run_fn(X_ref, X_eval, **{param_name: best_pv})
            if test_scores is None:
                return None, None
            r_test, _ = eval_sample_scores(test_scores, y_eval, fixed_tau=tau)
            return r_test, best_pv
        except Exception:
            return None, None

    # IF: tune contamination (5 values, including conservative ones)
    r_if, if_param = _tune_and_eval(
        run_baseline_if, "contamination", [0.01, 0.05, "auto", 0.1, 0.3],
        X_val_bl, y_val_bl, X_test_full, y_test,
    )
    result["if"] = r_if
    if if_param is not None:
        result["if_param"] = if_param

    # LOF: tune n_neighbors (3 values)
    r_lof, lof_param = _tune_and_eval(
        run_baseline_lof, "n_neighbors", [10, 20, 30],
        X_val_bl, y_val_bl, X_test_full, y_test,
    )
    result["lof"] = r_lof
    if lof_param is not None:
        result["lof_param"] = lof_param

    # KNN-distance: tune k (3 values)
    r_knn, knn_param = _tune_and_eval(
        run_baseline_knn_dist, "k", [5, 10, 20],
        X_val_bl, y_val_bl, X_test_full, y_test,
    )
    result["knn"] = r_knn
    if knn_param is not None:
        result["knn_param"] = knn_param

    return result


# =============================================================================
# Batch Runner
# =============================================================================

def discover_experiments(fp_data_dir):
    experiments = []
    fp_data = Path(fp_data_dir)
    for ds_dir in sorted(fp_data.iterdir()):
        if not ds_dir.is_dir():
            continue
        parts = ds_dir.name.rsplit("-", 2)
        if len(parts) < 3:
            continue
        for seed_dir in sorted(ds_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.replace("split_seed", ""))
            except ValueError:
                continue
            alert_path = seed_dir / "alert.json"
            if alert_path.exists():
                experiments.append({
                    "dataset": parts[0], "data_type": parts[1],
                    "model": parts[2], "seed": seed,
                    "alert_path": str(alert_path),
                })
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_data_dir", type=str, default="./fp_data")
    parser.add_argument("--output_dir", type=str, default="./batch_output")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--top_k_features", type=int, default=20)
    parser.add_argument("--k_list", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--res_list", type=float, nargs="+",
                        default=[0.5, 1.0, 10.0])
    parser.add_argument("--ms_list", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(args.fp_data_dir)
    if args.datasets:
        experiments = [e for e in experiments if e["dataset"] in args.datasets]
    if args.models:
        experiments = [e for e in experiments if e["model"] in args.models]
    if args.seeds:
        experiments = [e for e in experiments if e["seed"] in args.seeds]

    total_configs = (len(args.k_list) * len(args.res_list) * len(args.ms_list)
                     * len(SCORING_METHODS))
    print(f"Experiments: {len(experiments)}, "
          f"Cluster configs: {total_configs}, "
          f"Scoring methods: {len(SCORING_METHODS)} (auto-selected)")
    print(f"Output: {output_dir}")
    sys.stdout.flush()

    # Incremental results file
    results_csv = output_dir / "batch_results.csv"
    # Load existing results to support resume
    done_tags = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        done_tags = set(existing["tag"].tolist())
        print(f"Resuming: {len(done_tags)} experiments already done")
        sys.stdout.flush()

    all_rows = []
    if results_csv.exists():
        all_rows = pd.read_csv(results_csv).to_dict("records")

    for i, exp in enumerate(experiments):
        tag = f"{exp['dataset']}-{exp['data_type']}-{exp['model']}-seed{exp['seed']}"
        if tag in done_tags:
            print(f"[{i+1}/{len(experiments)}] {tag} — SKIP (done)")
            sys.stdout.flush()
            continue

        print(f"[{i+1}/{len(experiments)}] {tag} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = run_single_experiment(
                exp["alert_path"], val_ratio=args.val_ratio,
                top_k_features=args.top_k_features,
                k_list=args.k_list, res_list=args.res_list,
                ms_list=args.ms_list,
            )
            elapsed = time.time() - t0

            if result is None:
                print("SKIP (too few samples)", flush=True)
                continue

            # Flatten to row
            row = {
                "tag": tag, "dataset": exp["dataset"], "model": exp["model"],
                "seed": exp["seed"], "n_test": result["n_test"],
                "fp_test": result["fp_test"], "tp_test": result["tp_test"],
                "n_features": result["n_features_selected"],
                "time_s": round(elapsed, 1),
            }
            gs = result.get("graphsieve")
            if gs:
                row.update({"gs_R.FPR": gs["R.FPR"], "gs_R.TPR": gs["R.TPR"],
                            "gs_CER": gs["CER"], "gs_auc": gs["auc"]})
            bp = result.get("best_params")
            if bp:
                row.update({"gs_k": bp["k"], "gs_res": bp["res"],
                            "gs_ms": bp["ms"], "gs_scoring": bp["scoring"],
                            "gs_blend": bp.get("blend", 0.0),
                            "gs_tau": bp.get("tau")})
            row["gs_val_auc"] = result.get("val_auc")
            vop = result.get("val_op")
            if vop:
                row.update({"gs_val_R.FPR": vop["R.FPR"],
                            "gs_val_R.TPR": vop["R.TPR"]})

            # All baselines
            for bl_name in ["if", "lof", "knn"]:
                bl = result.get(bl_name)
                if bl:
                    row.update({
                        f"{bl_name}_R.FPR": bl["R.FPR"],
                        f"{bl_name}_R.TPR": bl["R.TPR"],
                        f"{bl_name}_CER": bl["CER"],
                        f"{bl_name}_auc": bl["auc"],
                    })

            all_rows.append(row)

            # Print inline
            scoring_str = bp.get("scoring", "?") if bp else "?"
            gs_str = (f"GS[{scoring_str}]:AUC={gs['auc']:.3f}") if gs else "GS:FAIL"
            bl_strs = []
            for bl_name, bl_label in [("if", "IF"), ("lof", "LOF"), ("knn", "KNN")]:
                bl = result.get(bl_name)
                bl_strs.append(f"{bl_label}:AUC={bl['auc']:.3f}" if bl else f"{bl_label}:FAIL")
            print(f"{gs_str} | {' | '.join(bl_strs)} | {elapsed:.0f}s", flush=True)

            # Incremental save
            pd.DataFrame(all_rows).to_csv(results_csv, index=False)

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

    # Final summary
    df = pd.DataFrame(all_rows)
    df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    # Summary by dataset-model
    all_methods = [("gs", "GS"), ("if", "IF"), ("lof", "LOF"), ("knn", "KNN")]
    active_methods = [(p, n) for p, n in all_methods
                      if f"{p}_auc" in df.columns and df[f"{p}_auc"].notna().any()]

    header = f"{'Dataset':<16} {'Model':<6}"
    for _, name in active_methods:
        header += f" | {name+' AUC':>10}"
    header += " | Best"
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for (ds, model), grp in df.groupby(["dataset", "model"]):
        row = {"dataset": ds, "model": model, "n_seeds": len(grp)}
        for prefix, name in active_methods:
            for metric in ["R.FPR", "R.TPR", "CER", "auc"]:
                col = f"{prefix}_{metric}"
                if col in grp.columns:
                    vals = grp[col].dropna()
                    if len(vals) > 0:
                        row[f"{name}_{metric}_mean"] = vals.mean()
                        row[f"{name}_{metric}_std"] = vals.std()
        summary_rows.append(row)

        # Find best method by AUC
        best_name, best_auc = "", -1
        line = f"{ds:<16} {model:<6}"
        for prefix, name in active_methods:
            auc_val = row.get(f"{name}_auc_mean", float("nan"))
            line += f" | {auc_val:>10.4f}"
            if auc_val > best_auc:
                best_auc = auc_val
                best_name = name
        line += f" | {best_name}"
        print(line)

    print("-" * len(header))
    for prefix, name in active_methods:
        parts = []
        for metric in ["R.FPR", "R.TPR", "CER", "auc"]:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    parts.append(f"{metric}={vals.mean():.4f}±{vals.std():.4f}")
        print(f"Overall {name}: {', '.join(parts)}")

    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_by_dataset.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(f"Detail:  {results_csv}")


if __name__ == "__main__":
    main()
