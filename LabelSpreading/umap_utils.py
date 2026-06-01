"""UMAP embedding and density-based candidate selection for CN-star detection.

UMAP preserves more global structure than t-SNE, making it better suited
for identifying coherent subpopulations in the spectral manifold.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def run_umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    n_components: int = 2,
    random_seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Compute UMAP embedding for spectral data.

    Parameters
    ----------
    X : (N, D) feature matrix.
    n_neighbors : UMAP neighborhood size (balance local/global).
    min_dist : minimum distance in embedded space (tightness of clusters).
    metric : distance metric.
    n_components : embedding dimension.
    """
    import umap
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_seed,
        n_jobs=-1,
        verbose=verbose,
    )
    embedding = reducer.fit_transform(X)
    elapsed = time.time() - t0
    if verbose:
        print(f"UMAP embedding: {elapsed:.1f}s  shape={embedding.shape}")
    return embedding


def compute_cn_density(
    embedding: np.ndarray,
    known_mask: np.ndarray,
    bw_method: float = 0.05,
) -> np.ndarray:
    """Compute kernel density estimate using known CN star positions.

    Returns density evaluated at every point in the embedding.
    Higher density = more similar to known CN stars in UMAP space.
    """
    coords = embedding[known_mask].T
    if coords.shape[1] < 3:
        # Too few points for KDE — fall back to distance-based
        centroid = embedding[known_mask].mean(axis=0)
        return -np.linalg.norm(embedding - centroid, axis=1)

    kde = gaussian_kde(coords, bw_method=bw_method)
    density = kde(embedding.T)
    return density


def select_umap_candidates(
    embedding: np.ndarray,
    density: np.ndarray,
    unlabeled_mask: np.ndarray,
    known_mask: np.ndarray,
    top_n: int = 200,
    density_percentile: float = 90,
) -> np.ndarray:
    """Select unlabeled candidates from high-density UMAP regions.

    Returns indices of selected candidates, sorted by density descending.
    """
    unl_indices = np.where(unlabeled_mask)[0]
    unl_density = density[unlabeled_mask]

    threshold = np.percentile(density[known_mask], 50)  # median known CN density
    qualified = unl_density >= threshold

    if qualified.sum() == 0:
        qualified = np.ones_like(unl_density, dtype=bool)

    order = np.argsort(unl_density[qualified])[::-1]
    selected = unl_indices[qualified][order][:top_n]
    return selected


def run_umap_pipeline(
    X: np.ndarray,
    stars_clean: pd.DataFrame,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    top_n: int = 200,
    verbose: bool = True,
) -> Dict:
    """End-to-end UMAP candidate discovery."""
    known_mask = stars_clean["label"].values == 1
    unlabeled_mask = stars_clean["label"].values == -1

    if verbose:
        print(f"UMAP Pipeline: {X.shape[0]:,} stars, {known_mask.sum()} known CN, n_neighbors={n_neighbors}")

    embedding = run_umap_embedding(X, n_neighbors=n_neighbors, min_dist=min_dist, verbose=verbose)
    density = compute_cn_density(embedding, known_mask)
    candidates = select_umap_candidates(embedding, density, unlabeled_mask, known_mask, top_n=top_n)

    if verbose:
        print(f"Candidates selected: {len(candidates)}")
        print(f"  Density range (candidates): {density[candidates].min():.4f} - {density[candidates].max():.4f}")
        print(f"  Density range (known CN):   {density[known_mask].min():.4f} - {density[known_mask].max():.4f}")

    return {
        "embedding": embedding,
        "density": density,
        "candidate_indices": candidates,
        "known_mask": known_mask,
        "unlabeled_mask": unlabeled_mask,
    }
