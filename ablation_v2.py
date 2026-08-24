"""GraphSieve ablation study (V7-aligned).

5 variants:
    Full           - complete pipeline (reused from batch_output_v7 results)
    w/o FeatSel    - no variance feature selection (full features)
    w/o Graph->KM  - KMeans replaces kNN graph + Leiden
    w/o AdaptSc    - scoring fixed to density_low (no direction selection)
    w/o Blend      - blend=0 (pure cluster-level scoring)

Protocol matches batch_experiment.py: 30/70 stratified split, all selection
on validation set (AUC for params, max-CER with floors for tau), tau
transferred to test. 5 seeds averaged.

Usage:
    python ablation_v2.py --output_dir ./ablation_output
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from batch_experiment import (
    load_alert_data, pvoxel_transform, variance_feature_select,
    build_knn_graph, run_leiden, merge_micro_clusters,
    compute_cluster_profiles, unsupervised_score, eval_cluster_scoring,
    SCORING_METHODS, is_graph_too_dense, MAX_VAL_FOR_GRIDSEARCH,
    discover_experiments,
)

VAL_SEED = 42
CLUSTER_SEED = 42
K_LIST = [10, 30]
RES_LIST = [0.5, 1.0, 10.0]
MS_LIST = [5, 10]
KM_CLUSTERS_LIST = [10, 20, 50, 100]
DBSCAN_EPS_LIST = [0.1, 0.3, 0.5, 1.0]
BLEND_CANDIDATES = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
FIXED_SCORING = "density_low"  # for w/o AdaptSc (pVoxel-style default)


# =============================================================================
# Shared helpers
# =============================================================================

def prepare_data(alert_path, val_ratio=0.3, top_k=20, use_featsel=True):
    """Split + transform + (optional) feature selection.
    Returns dict with val/test arrays and grid-search subsample."""
    X, y = load_alert_data(alert_path)
    idx = np.arange(len(y))
    idx_val, idx_test = train_test_split(
        idx, test_size=1.0 - val_ratio, random_state=VAL_SEED, stratify=y)
    X_val, y_val = pvoxel_transform(X[idx_val]), y[idx_val]
    X_test, y_test = pvoxel_transform(X[idx_test]), y[idx_test]

    if use_featsel:
        sel = variance_feature_select(X_val, top_k=top_k)
    else:
        sel = list(range(X_val.shape[1]))
    X_val, X_test = X_val[:, sel], X_test[:, sel]

    if len(X_val) > MAX_VAL_FOR_GRIDSEARCH:
        rng = np.random.default_rng(VAL_SEED)
        sub = rng.choice(len(X_val), MAX_VAL_FOR_GRIDSEARCH, replace=False)
        X_gs, y_gs = X_val[sub], y_val[sub]
    else:
        X_gs, y_gs = X_val, y_val

    return {"X_val": X_val, "y_val": y_val, "X_test": X_test,
            "y_test": y_test, "X_gs": X_gs, "y_gs": y_gs}


def leiden_cluster(X, k, res, allow_fallback=False):
    """kNN graph + Leiden with density guard. Returns (labels, kdist).
    In grid search (allow_fallback=False): returns (None, None) for combos
    that would hang. In final application (allow_fallback=True): falls back
    to res<10 instead of failing, matching batch_experiment.py behavior."""
    G, w, kdist = build_knn_graph(X, k)
    if is_graph_too_dense(G, max_avg_degree=25) and res >= 10.0:
        if not allow_fallback:
            return None, None
        res = 1.0
    cl = run_leiden(G, w, resolution=res, seed=CLUSTER_SEED)
    return cl, kdist


def kmeans_cluster(X, n_clusters):
    """KMeans; kdist computed from a k=10 NN index (for profiles/blend)."""
    nc = min(n_clusters, len(X) - 1)
    km = KMeans(n_clusters=nc, random_state=CLUSTER_SEED, n_init=10)
    labels = km.fit_predict(X)
    nn = NearestNeighbors(n_neighbors=min(11, len(X)), n_jobs=-1)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    return labels, dist[:, -1]


def dbscan_cluster(X, eps, min_samples=5):
    """DBSCAN; noise points become singleton clusters so that
    merge_micro_clusters can reassign them to the nearest core cluster.
    kdist from a k=10 NN index (for profiles/blend)."""
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples,
                    n_jobs=-1).fit_predict(X)
    noise = labels == -1
    if noise.any():
        labels = labels.copy()
        next_id = labels.max() + 1
        labels[noise] = np.arange(next_id, next_id + int(noise.sum()))
    nn = NearestNeighbors(n_neighbors=min(11, len(X)), n_jobs=-1)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    return labels, dist[:, -1]


def finalize(data, cluster_fn, ms, scoring, blend):
    """Cluster full val -> pick tau (max CER with floors) -> apply to test."""
    # Validation: select operating point
    cl_v, kdist_v = cluster_fn(data["X_val"])
    if cl_v is None:
        return None
    cl_v = merge_micro_clusters(data["X_val"], cl_v, min_size=ms)
    df_v = compute_cluster_profiles(data["X_val"], cl_v, kdist_v)
    df_v = unsupervised_score(df_v, method=scoring)
    r_val, _ = eval_cluster_scoring(cl_v, data["y_val"], df_v,
                                    kdist=kdist_v, sample_blend=blend)
    if r_val is None:
        return None
    tau = r_val["tau"]

    # Test: apply transferred tau
    cl_t, kdist_t = cluster_fn(data["X_test"])
    if cl_t is None:
        return None
    cl_t = merge_micro_clusters(data["X_test"], cl_t, min_size=ms)
    df_t = compute_cluster_profiles(data["X_test"], cl_t, kdist_t)
    df_t = unsupervised_score(df_t, method=scoring)
    r_test, _ = eval_cluster_scoring(cl_t, data["y_test"], df_t,
                                     kdist=kdist_t, sample_blend=blend,
                                     fixed_tau=tau)
    if r_test is not None:
        r_test["val_R.FPR"] = r_val["R.FPR"]
        r_test["val_R.TPR"] = r_val["R.TPR"]
        r_test["tau"] = tau
    return r_test


# =============================================================================
# Variant runners
# =============================================================================

def _leiden_search(data, scoring_methods, search_blend=True):
    """Shared Leiden grid search. Graph built ONCE per k and reused.
    scoring_methods: list of methods to try, or a single fixed string.
    Returns best config dict or None."""
    fixed = isinstance(scoring_methods, str)
    methods = [scoring_methods] if fixed else scoring_methods

    best, best_auc = None, -1
    graphs = {}  # k -> (G, w, kdist, too_dense)
    for k in K_LIST:
        try:
            G, w, kdist = build_knn_graph(data["X_gs"], k)
            graphs[k] = (G, w, kdist, is_graph_too_dense(G, max_avg_degree=25))
        except Exception:
            continue

    for k, (G, w, kdist, dense) in graphs.items():
        for res in RES_LIST:
            if dense and res >= 10.0:
                continue
            try:
                cl = run_leiden(G, w, resolution=res, seed=CLUSTER_SEED)
            except Exception:
                continue
            for ms in MS_LIST:
                cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=ms)
                df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
                for sm in methods:
                    df_s = unsupervised_score(df, method=sm)
                    r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                                kdist=kdist, sample_blend=0.0)
                    if r and r["auc"] > best_auc:
                        best_auc = r["auc"]
                        best = {"k": k, "res": res, "ms": ms,
                                "scoring": sm, "blend": 0.0}
    if best is None:
        return None

    if search_blend:
        G, w, kdist, dense = graphs[best["k"]]
        res = best["res"]
        if dense and res >= 10.0:
            return best  # blend phase would need a different res; keep blend=0
        cl = run_leiden(G, w, resolution=res, seed=CLUSTER_SEED)
        cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=best["ms"])
        df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
        df_s = unsupervised_score(df, method=best["scoring"])
        for blend in BLEND_CANDIDATES:
            r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                        kdist=kdist, sample_blend=blend)
            if r and r["auc"] > best_auc:
                best_auc = r["auc"]
                best["blend"] = blend
    return best


def run_full_reimpl(data):
    """Full pipeline: Leiden grid x 14 scorings, then blend search.
    (Only used if v7 reference results are unavailable.)"""
    best = _leiden_search(data, SCORING_METHODS, search_blend=True)
    if best is None:
        return None
    return finalize(data,
                    lambda X: leiden_cluster(X, best["k"], best["res"],
                                             allow_fallback=True),
                    best["ms"], best["scoring"], best["blend"])


def run_wo_featsel(alert_path):
    """w/o FeatSel: identical pipeline on FULL features."""
    data = prepare_data(alert_path, use_featsel=False)
    return run_full_reimpl(data)


def run_wo_graph(data):
    """w/o Graph->KM: KMeans clustering; n_clusters searched on val AUC."""
    best, best_auc = None, -1
    for nc in KM_CLUSTERS_LIST:
        try:
            cl, kdist = kmeans_cluster(data["X_gs"], nc)
        except Exception:
            continue
        for ms in MS_LIST:
            cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=ms)
            df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
            for sm in SCORING_METHODS:
                df_s = unsupervised_score(df, method=sm)
                r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                            kdist=kdist, sample_blend=0.0)
                if r and r["auc"] > best_auc:
                    best_auc = r["auc"]
                    best = {"nc": nc, "ms": ms, "scoring": sm, "blend": 0.0}
    if best is None:
        return None
    cl, kdist = kmeans_cluster(data["X_gs"], best["nc"])
    cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=best["ms"])
    df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
    df_s = unsupervised_score(df, method=best["scoring"])
    for blend in BLEND_CANDIDATES:
        r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                    kdist=kdist, sample_blend=blend)
        if r and r["auc"] > best_auc:
            best_auc = r["auc"]
            best["blend"] = blend
    return finalize(data,
                    lambda X: kmeans_cluster(X, best["nc"]),
                    best["ms"], best["scoring"], best["blend"])


def run_wo_graph_db(data):
    """w/o Graph->DB: DBSCAN replaces kNN graph + Leiden.
    eps searched on validation AUC (light: 4 values)."""
    best, best_auc = None, -1
    for eps in DBSCAN_EPS_LIST:
        try:
            cl, kdist = dbscan_cluster(data["X_gs"], eps)
        except Exception:
            continue
        if len(np.unique(cl)) < 2:
            continue
        for ms in MS_LIST:
            cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=ms)
            if len(np.unique(cl_m)) < 2:
                continue
            df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
            for sm in SCORING_METHODS:
                df_s = unsupervised_score(df, method=sm)
                r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                            kdist=kdist, sample_blend=0.0)
                if r and r["auc"] > best_auc:
                    best_auc = r["auc"]
                    best = {"eps": eps, "ms": ms, "scoring": sm, "blend": 0.0}
    if best is None:
        return None
    cl, kdist = dbscan_cluster(data["X_gs"], best["eps"])
    cl_m = merge_micro_clusters(data["X_gs"], cl, min_size=best["ms"])
    df = compute_cluster_profiles(data["X_gs"], cl_m, kdist)
    df_s = unsupervised_score(df, method=best["scoring"])
    for blend in BLEND_CANDIDATES:
        r, _ = eval_cluster_scoring(cl_m, data["y_gs"], df_s,
                                    kdist=kdist, sample_blend=blend)
        if r and r["auc"] > best_auc:
            best_auc = r["auc"]
            best["blend"] = blend
    return finalize(data,
                    lambda X: dbscan_cluster(X, best["eps"]),
                    best["ms"], best["scoring"], best["blend"])


def run_wo_adaptsc(data):
    """w/o AdaptSc: Leiden grid as usual, but scoring FIXED to density_low.
    Blend search still allowed (it is a separate component)."""
    best = _leiden_search(data, FIXED_SCORING, search_blend=True)
    if best is None:
        return None
    return finalize(data,
                    lambda X: leiden_cluster(X, best["k"], best["res"],
                                             allow_fallback=True),
                    best["ms"], FIXED_SCORING, best["blend"])


def run_wo_blend(data):
    """w/o Blend: full Leiden grid + scoring search, but blend forced to 0."""
    best = _leiden_search(data, SCORING_METHODS, search_blend=False)
    if best is None:
        return None
    return finalize(data,
                    lambda X: leiden_cluster(X, best["k"], best["res"],
                                             allow_fallback=True),
                    best["ms"], best["scoring"], 0.0)


# =============================================================================
# Main
# =============================================================================

VARIANTS = ["Full", "w/o FeatSel", "w/o Graph->DB", "w/o AdaptSc", "w/o Blend"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_data_dir", type=str, default="./fp_data")
    parser.add_argument("--output_dir", type=str, default="./ablation_output")
    parser.add_argument("--v7_results", type=str,
                        default="./batch_output_v7/batch_results.csv",
                        help="v7 results CSV to reuse for the Full variant")
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "ablation_results.csv"

    experiments = discover_experiments(args.fp_data_dir)
    if args.datasets:
        experiments = [e for e in experiments if e["dataset"] in args.datasets]
    if args.models:
        experiments = [e for e in experiments if e["model"] in args.models]
    if args.seeds:
        experiments = [e for e in experiments if e["seed"] in args.seeds]

    # Load Full-variant reference from v7
    v7 = pd.read_csv(args.v7_results) if os.path.exists(args.v7_results) else None

    # Resume support
    all_rows = []
    done = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        all_rows = existing.to_dict("records")
        done = set(zip(existing["tag"], existing["variant"]))
        print(f"Resuming: {len(done)} rows done")
    total = len(experiments)
    for i, exp in enumerate(experiments):
        tag = f"{exp['dataset']}-{exp['data_type']}-{exp['model']}-seed{exp['seed']}"
        print(f"[{i+1}/{total}] {tag}", flush=True)

        for variant in VARIANTS:
            if (tag, variant) in done:
                continue
            t0 = time.time()
            try:
                if variant == "Full" and v7 is not None:
                    ref = v7[v7["tag"] == tag]
                    if len(ref):
                        r = {"auc": ref.iloc[0]["gs_auc"],
                             "R.FPR": ref.iloc[0]["gs_R.FPR"],
                             "R.TPR": ref.iloc[0]["gs_R.TPR"],
                             "CER": ref.iloc[0]["gs_CER"]}
                    else:
                        r = run_full_reimpl(prepare_data(exp["alert_path"]))
                elif variant == "Full":
                    r = run_full_reimpl(prepare_data(exp["alert_path"]))
                elif variant == "w/o FeatSel":
                    r = run_wo_featsel(exp["alert_path"])
                elif variant == "w/o Graph->DB":
                    r = run_wo_graph_db(prepare_data(exp["alert_path"]))
                elif variant == "w/o Graph->KM":  # legacy, superseded by DB
                    r = run_wo_graph(prepare_data(exp["alert_path"]))
                elif variant == "w/o AdaptSc":
                    r = run_wo_adaptsc(prepare_data(exp["alert_path"]))
                elif variant == "w/o Blend":
                    r = run_wo_blend(prepare_data(exp["alert_path"]))
                else:
                    r = None
            except Exception as e:
                print(f"    {variant} FAILED: {e}", flush=True)
                r = None

            elapsed = time.time() - t0
            if r is not None:
                all_rows.append({
                    "tag": tag, "dataset": exp["dataset"],
                    "model": exp["model"], "seed": exp["seed"],
                    "variant": variant,
                    "auc": r["auc"], "R.FPR": r["R.FPR"],
                    "R.TPR": r["R.TPR"], "CER": r["CER"],
                    "time_s": round(elapsed, 1),
                })
                print(f"    {variant}: AUC={r['auc']:.3f} "
                      f"R.FPR={r['R.FPR']:.3f} R.TPR={r['R.TPR']:.4f} "
                      f"CER={r['CER']:.1f} ({elapsed:.0f}s)", flush=True)
            else:
                all_rows.append({"tag": tag, "dataset": exp["dataset"],
                                 "model": exp["model"], "seed": exp["seed"],
                                 "variant": variant, "auc": np.nan,
                                 "R.FPR": np.nan, "R.TPR": np.nan,
                                 "CER": np.nan, "time_s": round(elapsed, 1)})
                print(f"    {variant}: FAILED ({elapsed:.0f}s)", flush=True)

        # Incremental save after each experiment
        pd.DataFrame(all_rows).to_csv(results_csv, index=False)

    print(f"\nSaved: {results_csv}")

    # Summary: mean +/- std across seeds, per (dataset, model, variant)
    df = pd.read_csv(results_csv)
    summary = (df.groupby(["dataset", "model", "variant"])
               [["auc", "R.FPR", "R.TPR", "CER"]]
               .agg(["mean", "std"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Overall per variant (mean over all experiments)
    print(f"\n{'Variant':<16} {'AUC':>16} {'R.FPR':>16} {'R.TPR':>16} {'CER':>16}")
    print("-" * 84)
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        print(f"{variant:<16} "
              f"{sub['auc'].mean():.3f}±{sub['auc'].std():.3f}   "
              f"{sub['R.FPR'].mean():.3f}±{sub['R.FPR'].std():.3f}   "
              f"{sub['R.TPR'].mean():.3f}±{sub['R.TPR'].std():.3f}   "
              f"{sub['CER'].mean():.1f}±{sub['CER'].std():.1f}")


if __name__ == "__main__":
    main()
