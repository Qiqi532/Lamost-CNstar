"""Spectral clustering utilities for CN-star subgroup discovery.

Operates on the graph Laplacian eigenvectors to find clusters
enriched in CN-enhanced stars — complementing Label Spreading's
propagation approach with a spectral decomposition perspective.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_normalized_laplacian(W) -> sparse.csr_matrix:
    """Compute normalized graph Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}."""
    if not hasattr(W, "dot"):
        W = sparse.csr_matrix(W)
    deg = np.array(W.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1e-12)
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    N = W.shape[0]
    D_diag = sparse.diags(D_inv_sqrt, 0, shape=(N, N))
    L_sym = sparse.eye(N, format="csr") - D_diag @ W @ D_diag
    return L_sym


def spectral_decomposition(
    W,
    n_eigenvectors: int = 20,
    which: str = "SM",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smallest eigenvectors of the normalized Laplacian.

    Parameters
    ----------
    W : sparse affinity matrix.
    n_eigenvectors : number of eigenvectors to compute.
    which : 'SM' for smallest eigenvalues (cluster structure).

    Returns
    -------
    eigenvalues : (k,) array.
    eigenvectors : (N, k) array.
    """
    L_sym = build_normalized_laplacian(W)
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_eigenvectors, which=which, tol=1e-4)
    elapsed = time.time() - t0
    print(f"  Eigendecomposition: {elapsed:.1f}s ({n_eigenvectors} eigenvectors)")
    return eigenvalues, eigenvectors


def cluster_on_eigenvectors(
    eigenvectors: np.ndarray,
    n_clusters_list: List[int] = None,
    random_seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Run KMeans on spectral embeddings for multiple cluster counts."""
    if n_clusters_list is None:
        n_clusters_list = [5, 10, 15, 20, 30]
    results = {}
    for n_clusters in n_clusters_list:
        km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)
        labels = km.fit_predict(eigenvectors)
        results[n_clusters] = labels
    return results


def evaluate_cluster_enrichment(
    cluster_labels: np.ndarray,
    cn_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute CN-star enrichment per cluster.

    Parameters
    ----------
    cluster_labels : (N,) cluster assignments.
    cn_labels : (N,) binary CN labels (1 = known CN).

    Returns
    -------
    DataFrame with columns: cluster, n_total, n_cn, enrichment_ratio, pct_cn.
    """
    global_cn_rate = cn_labels.mean()
    clusters = np.unique(cluster_labels)
    rows = []
    for cid in sorted(clusters):
        cmask = cluster_labels == cid
        n_c = cmask.sum()
        n_cn = cn_labels[cmask].sum()
        pct = n_cn / max(n_c, 1)
        enrichment = pct / max(global_cn_rate, 1e-12)
        rows.append({
            "cluster": cid, "n_total": n_c, "n_cn": n_cn,
            "pct_cn": pct, "enrichment": enrichment,
        })
    return pd.DataFrame(rows).sort_values("enrichment", ascending=False)


def run_spectral_pipeline(
    W,
    stars_clean: pd.DataFrame,
    n_eigenvectors: int = 20,
    n_clusters_list: List[int] = None,
) -> Dict:
    """End-to-end spectral clustering for CN-star discovery.

    Returns dict with eigenvalues, eigenvectors, clusterings, enrichment tables.
    """
    print(f"Spectral Clustering Pipeline")
    print(f"  Graph nodes: {W.shape[0]:,}")
    print(f"  Eigenvectors: {n_eigenvectors}")

    eigenvalues, eigenvectors = spectral_decomposition(W, n_eigenvectors)
    clusterings = cluster_on_eigenvectors(eigenvectors, n_clusters_list)

    cn_labels = (stars_clean["label"].values == 1).astype(int)
    enrichment_tables = {}
    for n_clusters, labels in clusterings.items():
        enrichment_tables[n_clusters] = evaluate_cluster_enrichment(labels, cn_labels)

    best_n = max(enrichment_tables, key=lambda k: enrichment_tables[k]["enrichment"].max())
    best_enrich = enrichment_tables[best_n]

    print(f"\n  Best clustering: n_clusters={best_n}")
    print(f"  Global CN rate: {cn_labels.mean():.4f}")
    print(f"  Top cluster enrichment: {best_enrich['enrichment'].iloc[0]:.1f}x")
    print(f"  Top cluster CN count: {int(best_enrich['n_cn'].iloc[0])}/{int(best_enrich['n_total'].iloc[0])}")

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "clusterings": clusterings,
        "enrichment_tables": enrichment_tables,
        "best_n_clusters": best_n,
        "best_cluster_labels": clusterings[best_n],
        "best_enrichment": best_enrich,
    }
