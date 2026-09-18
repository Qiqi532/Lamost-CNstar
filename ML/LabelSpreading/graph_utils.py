"""Shared graph construction and Label Spreading utilities for CN-star detection.

Used by LabelSpreading, SpectralClustering, GNN, and UMAP modules.
Builds KNN similarity graphs from multiple feature spaces and propagates
known CN labels to unlabeled stars via manual Label Spreading iteration
(Zhou et al. algorithm, sparse-optimized).
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 50,
    metric: str = "euclidean",
    standardize: bool = True,
):
    """Build a weighted KNN adjacency matrix from feature matrix X.

    Returns a sparse CSR matrix (memory-efficient for 33K+ nodes).
    Uses RBF kernel on KNN distances: w_ij = exp(-gamma * d^2).
    """
    if standardize:
        X = StandardScaler().fit_transform(X)

    # Work around sklearn/numpy compatibility issue: sklearn.metrics.pairwise
    # cosine_distances crashes with numpy 2.x. Manually L2-normalize and use
    # euclidean distance (equivalent neighbor ranking to cosine on normed data).
    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X = X / norms
        metric = 'euclidean'

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    N = X.shape[0]
    sigma = np.median(distances[:, 1:][distances[:, 1:] > 0])
    if sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma ** 2)

    rows = np.repeat(np.arange(N), n_neighbors)
    # Skip self (index 0)
    cols = indices[:, 1:].ravel()
    dists = distances[:, 1:].ravel()
    weights = np.exp(-gamma * dists ** 2)

    # Symmetrize: build sparse from both (i,j) and (j,i)
    W = sparse.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(N, N),
    ).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def build_graph_from_cache(
    stars_clean: pd.DataFrame,
    n_neighbors: int = 50,
):
    """Build KNN graph from existing neighbor_indices in stars_clean.

    Zero-distance-cost alternative to build_knn_graph — reuses the
    cluster-priority KNN (K=180) already computed during preprocessing.
    Truncates to the requested n_neighbors per node.

    Uses RBF kernel on Euclidean distances in (teff, logg, feh) space.
    """
    N = len(stars_clean)
    neighbor_indices = stars_clean["neighbor_indices"].values
    neighbor_distances = stars_clean["neighbor_distances"].values

    k_use = min(n_neighbors, len(neighbor_indices[0]))
    rows = np.repeat(np.arange(N), k_use)
    cols = np.array([idx[:k_use] for idx in neighbor_indices]).ravel()
    dists = np.array([d[:k_use] for d in neighbor_distances]).ravel()

    sigma = np.median(dists[dists > 0])
    if sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma ** 2)
    weights = np.exp(-gamma * dists ** 2)

    W = sparse.coo_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(N, N),
    ).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def run_label_spreading(
    W: np.ndarray,
    y_labeled: np.ndarray,
    alpha: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, float]:
    """Run LabelSpreading on a pre-built graph (manual iteration).

    Implements the Zhou et al. algorithm directly using numpy, which
    supports arbitrary precomputed affinity matrices and distance metrics
    (unlike sklearn's LabelSpreading which only supports 'knn'/'rbf' kernels).

    Parameters
    ----------
    W : (N, N) sparse/ndarray affinity matrix.
    y_labeled : (N,) label array. Known positives = 1, unknown = -1.
    alpha : clamping factor (0=hard reset, 1=no clamping, typical 0.2-0.8).
    max_iter, tol : convergence controls.

    Returns
    -------
    prob_cn : (N,) probability of being CN-enhanced [0, 1].
    elapsed : wall-clock time in seconds.
    """
    t0 = time.time()
    N = W.shape[0]

    # Ensure sparse CSR for efficiency
    if not hasattr(W, "dot"):
        W = sparse.csr_matrix(W)

    # Build initial label matrix Y: (N, 2)
    Y = np.zeros((N, 2), dtype=np.float64)
    pos_mask = y_labeled == 1
    Y[pos_mask, 0] = 1.0

    # Normalized graph Laplacian: S = D^{-1/2} W D^{-1/2}
    deg = np.array(W.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-12)
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    D_diag = sparse.diags(D_inv_sqrt, 0, shape=(N, N), dtype=np.float64)
    S = D_diag @ W @ D_diag  # sparse @ sparse → sparse

    # Iterate: F_{t+1} = alpha * S @ F_t + (1-alpha) * Y
    F = Y.copy()
    for iteration in range(max_iter):
        F_new = alpha * S.dot(F) + (1.0 - alpha) * Y  # sparse.dot(dense) → dense
        diff = np.abs(F_new - F).max()
        F = F_new
        if diff < tol:
            break

    prob_cn = np.clip(F[:, 0], 0.0, 1.0)
    elapsed = time.time() - t0
    return prob_cn, elapsed


def evaluate_strategy(
    name: str,
    prob_cn: np.ndarray,
    stars_clean: pd.DataFrame,
    topk_list: List[int] = [50, 100, 200, 500],
) -> Dict:
    """Compute evaluation metrics for a label-spreading run.

    Parameters
    ----------
    name : strategy name for reporting.
    prob_cn : (N,) CN probabilities.
    stars_clean : metadata DataFrame with 'label' column (1=known CN, -1=unlabeled).
    topk_list : top-K values for precision/recall computation.

    Returns
    -------
    dict of metrics.
    """
    y_true = (stars_clean["label"].values == 1).astype(int)
    n_pos = int(y_true.sum())

    # --- self-consistency: mean prob of known CN stars ---
    pos_mask = y_true == 1
    mean_prob_pos = float(np.mean(prob_cn[pos_mask]))
    median_prob_pos = float(np.median(prob_cn[pos_mask]))
    median_prob_all = float(np.median(prob_cn))

    # --- pseudo PR-AUC using known positives ---
    # (This is a proxy: treat unlabeled as negative, so it's optimistic)
    from sklearn.metrics import average_precision_score, roc_auc_score
    pseudo_pr = average_precision_score(y_true, prob_cn)
    pseudo_roc = roc_auc_score(y_true, prob_cn)

    # --- top-K metrics ---
    order = np.argsort(prob_cn)[::-1]
    metrics = {}
    for k in topk_list:
        top_k_labels = y_true[order[:k]]
        metrics[f"precision@{k}"] = float(top_k_labels.mean())
        metrics[f"recall@{k}"] = float(top_k_labels.sum() / max(n_pos, 1))

    return {
        "name": name,
        "n_pos": n_pos,
        "n_total": len(prob_cn),
        "mean_prob_known_cn": mean_prob_pos,
        "median_prob_known_cn": median_prob_pos,
        "median_prob_all": median_prob_all,
        "prob_ratio": mean_prob_pos / max(median_prob_all, 1e-12),
        "pseudo_pr_auc": pseudo_pr,
        "pseudo_roc_auc": pseudo_roc,
        **metrics,
    }


def run_all_strategies(
    X_clean: np.ndarray,
    feature_df: pd.DataFrame,
    stars_clean: pd.DataFrame,
    k_values: List[int] = [30, 50, 100, 180],
    alpha: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """Run Label Spreading across multiple graph construction strategies.

    Returns dict with keys: 'results' (list of metric dicts), 'probs' (dict
    of (strategy_name -> prob array)), 'W' (dict of strategy -> adjacency).
    """
    N = len(X_clean)

    # Prepare labels: +1 for known CN, -1 for unknown
    y_labeled = stars_clean["label"].map({1: 1, -1: -1}).values.astype(int)

    # Feature spaces
    # (a) 700-D spectra — cosine distance
    # (b) 14-D physics features (FEATURE_COLS_14) — Euclidean
    # (c) 3-D delta features — Euclidean

    # 本地定义 14-D 特征列（避免 import ML.utils → torch → fbgemm.dll）
    FEATURE_COLS_14 = [
        "teff", "logg", "feh",
        "CN3839", "CN4142", "CH4300",
        "delta_CN3839", "delta_CN4142", "delta_CH4300",
        "knn_center_euclid", "knn_center_dist_z",
        "pca_1", "pca_2", "pca_3",
    ]
    feat_cols_14 = [c for c in FEATURE_COLS_14 if c in feature_df.columns]
    X_physics = feature_df[feat_cols_14].values.astype(np.float64)

    delta_cols = ["delta_CN3839", "delta_CN4142", "delta_CH4300"]
    delta_cols = [c for c in delta_cols if c in feature_df.columns]
    X_delta = feature_df[delta_cols].values.astype(np.float64) if delta_cols else None

    results = []
    probs = {}
    graphs = {}

    def _try_strategy(name, X, metric, K, do_standardize):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {name}  (K={K}, metric={metric})")
            print(f"{'='*60}")

        t0 = time.time()
        W = build_knn_graph(X, n_neighbors=K, metric=metric, standardize=do_standardize)
        t_graph = time.time() - t0
        if verbose:
            print(f"  Graph built: {t_graph:.1f}s, sparsity: {(W>0).sum()/W.size:.4f}")

        prob, t_ls = run_label_spreading(W, y_labeled, alpha=alpha)
        if verbose:
            print(f"  LabelSpreading: {t_ls:.1f}s")

        metrics = evaluate_strategy(name, prob, stars_clean)
        metrics["graph_time_s"] = t_graph
        metrics["ls_time_s"] = t_ls
        metrics["K"] = K
        metrics["metric"] = metric
        results.append(metrics)
        probs[name] = prob
        graphs[name] = W

        if verbose:
            print(f"  Mean prob (known CN): {metrics['mean_prob_known_cn']:.4f}")
            print(f"  Median prob (all):    {metrics['median_prob_all']:.4f}")
            print(f"  Prob ratio:           {metrics['prob_ratio']:.2f}")
            print(f"  Pseudo PR-AUC:        {metrics['pseudo_pr_auc']:.4f}")
            print(f"  Precision@100:        {metrics['precision@100']:.4f}")

        return metrics

    # Strategy 1: Spectra cosine (various K)
    for K in k_values:
        _try_strategy(f"spectra_cosine_K{K}", X_clean, "cosine", K, do_standardize=False)

    # Strategy 2: Physics features Euclidean (various K)
    for K in k_values:
        _try_strategy(f"physics_euclidean_K{K}", X_physics, "euclidean", K, do_standardize=True)

    # Strategy 3: Delta-only Euclidean
    if X_delta is not None:
        for K in [30, 50]:
            _try_strategy(f"delta_euclidean_K{K}", X_delta, "euclidean", K, do_standardize=True)

    # Strategy 4: Precomputed KNN from cache (zero compute cost)
    if "neighbor_indices" in stars_clean.columns:
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  cache_knn_K50  (reuse precomputed KNN)")
                print(f"{'='*60}")
            t0 = time.time()
            W_cache = build_graph_from_cache(stars_clean, n_neighbors=50)
            t_graph = time.time() - t0
            if verbose:
                print(f"  Graph built: {t_graph:.1f}s (from cache)")

            prob_cache, t_ls = run_label_spreading(W_cache, y_labeled, alpha=alpha)
            metrics_cache = evaluate_strategy("cache_knn_K50", prob_cache, stars_clean)
            metrics_cache["graph_time_s"] = t_graph
            metrics_cache["ls_time_s"] = t_ls
            metrics_cache["K"] = 50
            metrics_cache["metric"] = "precomputed"
            results.append(metrics_cache)
            probs["cache_knn_K50"] = prob_cache
            if verbose:
                print(f"  Mean prob (known CN): {metrics_cache['mean_prob_known_cn']:.4f}")
                print(f"  Precision@100:        {metrics_cache['precision@100']:.4f}")
        except Exception as e:
            if verbose:
                print(f"  cache_knn skipped: {e}")

    return {"results": results, "probs": probs, "graphs": graphs, "y_labeled": y_labeled}


def compute_cluster_zscore(
    y_prob: np.ndarray,
    cluster_ids: np.ndarray,
    min_mad: float = 0.05,
) -> np.ndarray:
    """Normalize probabilities to per-cluster z-scores (debiasing).

    Uses a minimum MAD floor to prevent explosion when most cluster
    members have near-zero probability after label spreading
    (common with highly concentrated propagation results).
    """
    z_scores = np.full_like(y_prob, np.nan)
    for cid in np.unique(cluster_ids):
        cmask = cluster_ids == cid
        cprobs = y_prob[cmask]
        cmed = np.nanmedian(cprobs)
        cmad = np.nanmedian(np.abs(cprobs - cmed))
        cmad = max(cmad, min_mad)
        z_scores[cmask] = (cprobs - cmed) / (1.4826 * cmad)
    return z_scores


def export_candidates(
    prob_cn: np.ndarray,
    stars_clean: pd.DataFrame,
    feature_df: pd.DataFrame,
    cluster_ids: np.ndarray,
    top_n: int = 200,
) -> pd.DataFrame:
    """Export top-N candidates with z-score debiasing and metadata."""
    z_scores = compute_cluster_zscore(prob_cn, cluster_ids)

    out = stars_clean[["uid", "ra", "dec", "teff", "logg", "feh", "label", "snru",
                        "mag_ps_g", "masked_cluster_id"]].copy()
    out["ls_prob"] = prob_cn
    out["ls_zscore"] = z_scores

    # Add CN indices from feature_df if available
    for col in ["CN3839", "CN4142", "CH4300", "delta_CN3839", "delta_CN4142", "delta_CH4300"]:
        if col in feature_df.columns:
            out[col] = feature_df[col].values

    out = out.sort_values("ls_zscore", ascending=False).head(top_n)
    return out.reset_index(drop=True)
