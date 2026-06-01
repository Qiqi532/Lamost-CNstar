"""Shared utilities for ML experiments — XGBoost & MLP CN-star detection.

Key design choices (informed by Deep/ and BinaryClassifier/ experiments):
- Masked-band clustering (CN/CH bands masked before PCA+KMeans) — avoids
  molecular-band-driven clustering bias
- 14-D feature set: teff/logg/feh + CN3839/CN4142/CH4300 + delta_CN3839/
  delta_CN4142/delta_CH4300 + knn_center_euclid + knn_center_dist_z + pca_1-3
- Cluster-conditional z-score ranking for debiased candidate selection
- Focal loss for both XGBoost (custom objective) and PyTorch MLP
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# 0. Focal Loss implementations
# ═══════════════════════════════════════════════════════════════════════

def xgb_focal_loss_obj(y_pred, dtrain, gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss custom objective for XGBoost.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Uses the well-tested approximation: modulate standard CE gradient
    and hessian by the focal weight (1 - p_t)^γ.

    Parameters
    ----------
    y_pred : raw margin scores (before sigmoid)
    dtrain : xgb.DMatrix
    gamma : focal focusing parameter (higher = more focus on hard examples)
    alpha : positive class weight
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-15, 1 - 1e-15)

    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    focal_weight = (1 - p_t) ** gamma

    grad = alpha_t * focal_weight * (p - y_true)
    hess = alpha_t * focal_weight * p * (1 - p)
    return grad, hess


def xgb_focal_loss_eval(y_pred, dtrain, gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss eval metric for XGBoost (for monitoring)."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)
    return "focal_loss", float(np.mean(loss))


# PyTorch FocalLoss — importable for MLP notebook or used via torch if available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FocalLoss(nn.Module):
        """Focal Loss for binary classification.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy(inputs, targets, reduction="none")
            p_t = targets * inputs + (1 - targets) * (1 - inputs)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            modulating_factor = (1 - p_t) ** self.gamma
            loss = alpha_t * modulating_factor * bce
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════
# 1. Data loading & preprocessing
# ═══════════════════════════════════════════════════════════════════════

def load_and_preprocess(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[List[str]] = None,
    common_wave: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> Dict:
    """Run the full preprocessing pipeline and return cleaned data.

    Returns dict with: X_clean, stars_clean, common_wave, pipe_summary.
    """
    import importlib
    from Deep import spectra_io as _sio
    importlib.reload(_sio)

    if cn_catalogs is None:
        cn_catalogs = ["CNstar.csv", "FT_cands.csv"]
    if common_wave is None:
        common_wave = np.arange(3800.0, 4500.0, 1.0)

    pipe = _sio.run_screening_preprocess_from_files(
        stars_csv=stars_csv,
        common_wave=common_wave,
        folder=spectra_folder,
        cn_catalogs=cn_catalogs,
        cn_match_tolerance_arcsec=1.0,
        uid_col="uid",
        snr_col="snru",
        anomaly_low_pct=0.5,
        anomaly_high_pct=99.5,
        show_progress=show_progress,
    )

    return {
        "X_clean": pipe["datacube_clean"],
        "stars_clean": pipe["stars_clean"],
        "common_wave": common_wave,
        "pipe_summary": pipe["summary"],
        "label_report": pipe.get("label_report", {}),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Masked-band clustering + cluster-aware KNN
# ═══════════════════════════════════════════════════════════════════════

def compute_masked_clustering(
    X_clean: np.ndarray,
    stars_clean: pd.DataFrame,
    common_wave: np.ndarray,
    knn_param_cols: Optional[List[str]] = None,
    K: int = 180,
    random_seed: int = 42,
    band_ranges: Optional[List[Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Masked-band clustering + cluster-priority KNN.

    1. Mask CN/CH bands in spectra
    2. PCA on masked spectra → KMeans clustering
    3. KNN within cluster (teff/logg/feh space), fill from global if needed
    4. Compute KNN center features and distance metrics

    Returns stars_clean with added columns:
      masked_cluster_id, neighbor_indices, neighbor_distances,
      knn_center_{col}, knn_center_euclid, knn_euclid_prob, knn_center_dist_z
    """
    if knn_param_cols is None:
        knn_param_cols = ["teff", "logg", "feh"]
    if band_ranges is None:
        band_ranges = [(3830, 3883), (4120, 4216), (4285, 4315)]

    stars = stars_clean.copy()

    # Validate columns
    knn_param_cols = [c for c in knn_param_cols if c in stars.columns]
    if len(knn_param_cols) < 3:
        raise RuntimeError(f"Need >=3 KNN param columns, got: {knn_param_cols}")

    # Standardize parameters
    param_df = stars[knn_param_cols].apply(pd.to_numeric, errors="coerce")
    param_df = param_df.fillna(param_df.median(numeric_only=True))
    scaler = StandardScaler()
    X_params_scaled = scaler.fit_transform(param_df.values)

    # Step 1: Mask CN/CH bands
    masked_cube = np.asarray(X_clean, dtype=float).copy()
    band_masks = []
    for l1, l2 in band_ranges:
        m = (common_wave >= l1) & (common_wave <= l2)
        band_masks.append(m)
    all_band = np.any(np.array(band_masks), axis=0)
    fill_vals = np.nanmedian(masked_cube[:, ~all_band], axis=1)
    fill_vals = np.where(np.isfinite(fill_vals), fill_vals, 1.0)
    for m in band_masks:
        masked_cube[:, m] = fill_vals[:, None]

    # Step 2: PCA on masked spectra → KMeans
    n_samples = masked_cube.shape[0]
    n_components = int(min(25, masked_cube.shape[1], max(2, n_samples - 1)))
    X_masked_emb = PCA(n_components=n_components, random_state=random_seed).fit_transform(masked_cube)

    n_clusters = int(np.clip(np.sqrt(n_samples) / 2, 10, 45))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)
    cluster_labels = kmeans.fit_predict(X_masked_emb)
    stars["masked_cluster_id"] = cluster_labels

    # Step 3: Cluster-priority KNN
    global_knn = NearestNeighbors(n_neighbors=min(K + 1, len(stars)), metric="euclidean")
    global_knn.fit(X_params_scaled)
    global_distances, global_indices = global_knn.kneighbors(X_params_scaled)

    nn_idx_all = []
    nn_dist_all = []
    for i in range(len(stars)):
        cid = cluster_labels[i]
        local_idx = np.where(cluster_labels == cid)[0]

        if len(local_idx) >= 3:
            local_X = X_params_scaled[local_idx]
            local_k = min(K + 1, len(local_idx))
            local_knn = NearestNeighbors(n_neighbors=local_k, metric="euclidean")
            local_knn.fit(local_X)
            local_pos = int(np.where(local_idx == i)[0][0])
            d_loc, idx_loc = local_knn.kneighbors(local_X[local_pos].reshape(1, -1))
            nb_local = local_idx[idx_loc[0]]
            keep = nb_local != i
            nb = nb_local[keep].tolist()
            dd = d_loc[0][keep].tolist()
        else:
            nb, dd = [], []

        # Fill from global if needed
        if len(nb) < K:
            g_nb = global_indices[i, 1:].tolist()
            g_dd = global_distances[i, 1:].tolist()
            for j, d in zip(g_nb, g_dd):
                if j not in nb:
                    nb.append(int(j))
                    dd.append(float(d))
                if len(nb) >= K:
                    break

        nn_idx_all.append(np.asarray(nb[:K], dtype=int))
        nn_dist_all.append(np.asarray(dd[:K], dtype=float))

    stars["neighbor_indices"] = nn_idx_all
    stars["neighbor_distances"] = nn_dist_all

    # Step 4: KNN center features
    center_scaled = np.zeros_like(X_params_scaled, dtype=float)
    center_euclid = np.full(len(stars), np.nan, dtype=float)
    for i, nb in enumerate(stars["neighbor_indices"]):
        nb = np.asarray(nb, dtype=int)
        if nb.size == 0:
            continue
        local_center = X_params_scaled[nb].mean(axis=0)
        center_scaled[i] = local_center
        center_euclid[i] = np.linalg.norm(X_params_scaled[i] - local_center)

    center_orig = scaler.inverse_transform(center_scaled)
    for j, col in enumerate(knn_param_cols):
        stars[f"knn_center_{col}"] = center_orig[:, j]

    stars["knn_center_euclid"] = center_euclid
    dist_rank = pd.Series(center_euclid).rank(pct=True, method="average")
    stars["knn_euclid_prob"] = dist_rank.clip(0.0, 1.0).values

    dist_med = np.nanmedian(center_euclid)
    dist_mad = np.nanmedian(np.abs(center_euclid - dist_med)) + 1e-12
    stars["knn_center_dist_z"] = (center_euclid - dist_med) / (1.4826 * dist_mad)

    print(f"  Masked clustering: {n_clusters} clusters, K={K}, PCA-{n_components}")
    return stars


# ═══════════════════════════════════════════════════════════════════════
# 3. Feature engineering
# ═══════════════════════════════════════════════════════════════════════

BAND_DEFS = {
    "CN3839": {"band": (3830, 3883), "blue": (3894, 3910), "red": (4000, 4020)},
    "CN4142": {"band": (4120, 4216), "blue": (4055, 4080), "red": (4240, 4280)},
    "CH4300": {"band": (4285, 4315), "blue": (4240, 4280), "red": (4390, 4460)},
}


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) > 0 else np.nan


def compute_band_index(wave: np.ndarray, flux: np.ndarray,
                       band: tuple, blue: tuple, red: tuple) -> float:
    band_mask = (wave >= band[0]) & (wave <= band[1])
    blue_mask = (wave >= blue[0]) & (wave <= blue[1])
    red_mask = (wave >= red[0]) & (wave <= red[1])
    f_band = _safe_mean(flux[band_mask])
    f_blue = _safe_mean(flux[blue_mask])
    f_red = _safe_mean(flux[red_mask])
    if not all(np.isfinite(v) for v in [f_band, f_blue, f_red]):
        return np.nan
    f_cont = 0.5 * (f_blue + f_red)
    if f_band <= 0 or f_cont <= 0:
        return np.nan
    return float(-2.5 * np.log10(f_band / f_cont))


def _neighbor_median(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    vals = values[np.asarray(neighbor_idx, dtype=int)]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) > 0 else np.nan


def compute_features(
    X_clean: np.ndarray,
    stars_clean: pd.DataFrame,
    common_wave: np.ndarray,
    n_pca: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute the full 14-D feature set.

    Feature columns (14):
      teff, logg, feh,
      CN3839, CN4142, CH4300,
      delta_CN3839, delta_CN4142, delta_CH4300,
      knn_center_euclid, knn_center_dist_z,
      pca_1, pca_2, pca_3

    Returns feature_df indexed the same as stars_clean.
    """
    meta_cols = ["teff", "logg", "feh", "label", "filepath",
                 "knn_center_euclid", "knn_center_dist_z", "knn_euclid_prob"]
    meta_cols = [c for c in meta_cols if c in stars_clean.columns]
    feature_df = stars_clean[meta_cols].copy()

    # 1. CN/CH band indices
    n = len(X_clean)
    cn3839 = np.array([compute_band_index(common_wave, X_clean[i],
                        **{k: v for k, v in BAND_DEFS["CN3839"].items()}) for i in range(n)])
    cn4142 = np.array([compute_band_index(common_wave, X_clean[i],
                        **{k: v for k, v in BAND_DEFS["CN4142"].items()}) for i in range(n)])
    ch4300 = np.array([compute_band_index(common_wave, X_clean[i],
                        **{k: v for k, v in BAND_DEFS["CH4300"].items()}) for i in range(n)])

    for arr, name in [(cn3839, "CN3839"), (cn4142, "CN4142"), (ch4300, "CH4300")]:
        arr[np.isnan(arr)] = np.nanmedian(arr)
        feature_df[name] = arr

    # 2. Delta features (relative to cluster neighbors)
    if "neighbor_indices" in stars_clean.columns:
        med_cn3839 = np.array([_neighbor_median(cn3839, nb)
                               for nb in stars_clean["neighbor_indices"]])
        med_cn4142 = np.array([_neighbor_median(cn4142, nb)
                               for nb in stars_clean["neighbor_indices"]])
        med_ch4300 = np.array([_neighbor_median(ch4300, nb)
                               for nb in stars_clean["neighbor_indices"]])

        feature_df["nb_med_CN3839"] = med_cn3839
        feature_df["nb_med_CN4142"] = med_cn4142
        feature_df["nb_med_CH4300"] = med_ch4300
        feature_df["delta_CN3839"] = cn3839 - med_cn3839
        feature_df["delta_CN4142"] = cn4142 - med_cn4142
        feature_df["delta_CH4300"] = ch4300 - med_ch4300
    else:
        feature_df["delta_CN3839"] = 0.0
        feature_df["delta_CN4142"] = 0.0
        feature_df["delta_CH4300"] = 0.0

    # 3. PCA on spectra
    pca = PCA(n_components=n_pca, random_state=random_seed)
    X_pca = pca.fit_transform(X_clean)
    for i in range(n_pca):
        feature_df[f"pca_{i+1}"] = X_pca[:, i]

    print(f"  PCA explained variance (cumulative): {pca.explained_variance_ratio_.sum():.3f}")

    return feature_df


# ═══════════════════════════════════════════════════════════════════════
# 4. Feature column definitions
# ═══════════════════════════════════════════════════════════════════════

FEATURE_COLS_14 = [
    "teff", "logg", "feh",
    "CN3839", "CN4142", "CH4300",
    "delta_CN3839", "delta_CN4142", "delta_CH4300",
    "knn_center_euclid", "knn_center_dist_z",
    "pca_1", "pca_2", "pca_3",
]

FEATURE_COLS_CN9 = [
    "CN3839", "CN4142", "CH4300",
    "delta_CN3839", "delta_CN4142", "delta_CH4300",
    "pca_1", "pca_2", "pca_3",
]

FEATURE_COLS_DELTA6 = [
    "delta_CN3839", "delta_CN4142", "delta_CH4300",
    "pca_1", "pca_2", "pca_3",
]

FEATURE_COLS_DELTA_ONLY = FEATURE_COLS_DELTA6  # alias for compatibility

FEATURE_COLS_CN_RAW6 = [
    "CN3839", "CN4142", "CH4300",
    "pca_1", "pca_2", "pca_3",
]

FEATURE_COLS_DELTA_CN6 = [
    "CN3839", "CN4142", "CH4300",
    "delta_CN3839", "delta_CN4142", "delta_CH4300",
]

FEATURE_COLS_CN_ONLY = [
    "CN3839", "CN4142", "CH4300",
    "delta_CN3839", "delta_CN4142", "delta_CH4300",
]


# ═══════════════════════════════════════════════════════════════════════
# 5. Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════

def topk_metrics(y_true: np.ndarray, y_score: np.ndarray,
                 topk_list: List[int] = [50, 100, 200]) -> Dict:
    out = {}
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    total_pos = max(int((y_true == 1).sum()), 1)
    for k in topk_list:
        k_eff = min(k, len(y_sorted))
        y_top = y_sorted[:k_eff]
        out[f"precision@{k}"] = float(y_top.mean()) if k_eff > 0 else np.nan
        out[f"recall@{k}"] = float(y_top.sum() / total_pos)
    return out


def evaluate_binary(name: str, y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5, topk_list: List[int] = [50, 100, 200]) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "model": name,
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        **topk_metrics(y_true, y_prob, topk_list),
    }


def compute_parameter_bias(y_prob: np.ndarray, feature_df: pd.DataFrame,
                           param_cols: List[str] = ["teff", "logg", "feh"]) -> Dict:
    """Compute Spearman correlation between predicted prob and stellar params."""
    bias = {}
    for col in param_cols:
        if col in feature_df.columns:
            valid = np.isfinite(feature_df[col].values) & np.isfinite(y_prob)
            if valid.sum() > 10:
                from scipy.stats import spearmanr
                corr, pval = spearmanr(feature_df[col].values[valid], y_prob[valid])
                bias[f"spearmanr_{col}"] = float(corr)
                bias[f"spearmanr_{col}_pval"] = float(pval)
    return bias


def compute_cluster_zscore(y_prob: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    """Normalize probabilities to per-cluster z-scores (debiasing)."""
    z_scores = np.full_like(y_prob, np.nan)
    for cid in np.unique(cluster_ids):
        cmask = cluster_ids == cid
        cprobs = y_prob[cmask]
        cmed = np.nanmedian(cprobs)
        cmad = np.nanmedian(np.abs(cprobs - cmed)) + 1e-12
        z_scores[cmask] = (cprobs - cmed) / (1.4826 * cmad)
    return z_scores


# ═══════════════════════════════════════════════════════════════════════
# 6. Hard negative mining
# ═══════════════════════════════════════════════════════════════════════

def hard_negative_mining(
    model, X_train: np.ndarray, y_train: np.ndarray,
    X_unlabeled: np.ndarray, unlabeled_indices: np.ndarray,
    n_hard: int = 200, prob_range: Tuple[float, float] = (0.3, 0.7),
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select 'hard' negatives from unlabeled pool.

    Hard negatives are unlabeled samples with intermediate predicted
    probability — these are the most confusing cases for the model.

    Returns (augmented X_train, augmented y_train, selected indices).
    """
    rng = np.random.RandomState(random_seed)
    prob_unl = model.predict_proba(X_unlabeled)[:, 1]

    # Find samples in the confusing probability range
    hard_mask = (prob_unl >= prob_range[0]) & (prob_unl <= prob_range[1])
    hard_idx_local = np.where(hard_mask)[0]

    if len(hard_idx_local) == 0:
        # Fallback: take highest-probability unlabeled as hard negatives
        n_take = min(n_hard, len(prob_unl))
        hard_idx_local = np.argsort(prob_unl)[-n_take:]

    if len(hard_idx_local) > n_hard:
        hard_idx_local = rng.choice(hard_idx_local, size=n_hard, replace=False)

    X_hard = X_unlabeled[hard_idx_local]
    y_hard = np.zeros(len(hard_idx_local))  # Treat as negatives

    X_aug = np.vstack([X_train, X_hard])
    y_aug = np.concatenate([y_train, y_hard])

    selected_global_idx = unlabeled_indices[hard_idx_local]
    return X_aug, y_aug, selected_global_idx


# ═══════════════════════════════════════════════════════════════════════
# 7. Visualization helpers
# ═══════════════════════════════════════════════════════════════════════

def plot_metric_comparison(results_df: pd.DataFrame, metric_cols: List[str],
                           title: str = "Model Comparison", figsize: Tuple = (14, 6)):
    """Bar chart comparing multiple models on key metrics."""
    fig, axes = plt.subplots(1, len(metric_cols), figsize=figsize)
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        if metric not in results_df.columns:
            ax.text(0.5, 0.5, f"No {metric}", ha="center", va="center")
            continue
        vals = results_df[metric].values
        names = results_df["model"].values
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(names)))
        bars = ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.2, axis="x")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_prob_distribution(probs_dict: Dict[str, np.ndarray],
                           title: str = "Probability Distribution"):
    """Compare probability distributions across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, probs in probs_dict.items():
        ax.hist(probs, bins=60, alpha=0.4, label=name, density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_candidate_spectra(
    candidate_df: pd.DataFrame,
    X_clean: np.ndarray,
    stars_clean: pd.DataFrame,
    common_wave: np.ndarray,
    top_n: int = 12,
    prob_col: str = "prob",
    title: str = "Candidate Spectra vs Cluster Mean",
):
    """Plot top candidate spectra against their cluster means."""
    if "masked_cluster_id" not in stars_clean.columns:
        print("No cluster info — plotting spectra without cluster reference.")
        stars_clean = stars_clean.copy()
        stars_clean["masked_cluster_id"] = 0

    cluster_ids = stars_clean["masked_cluster_id"].values
    cluster_mean = {}
    for cid in np.unique(cluster_ids):
        m = cluster_ids == cid
        cluster_mean[cid] = np.nanmedian(X_clean[m], axis=0)

    top = candidate_df.head(top_n)
    n_cols = 4
    n_rows = int(np.ceil(len(top) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        orig_idx = row.name if row.name < len(X_clean) else None
        if orig_idx is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"#{i+1}")
            continue

        cand_flux = X_clean[orig_idx]
        cid = cluster_ids[orig_idx] if orig_idx < len(cluster_ids) else 0
        if cid in cluster_mean:
            ax.plot(common_wave, cluster_mean[cid], color="darkorange",
                    linewidth=1.0, linestyle="--", alpha=0.85, label="Cluster mean")
        ax.plot(common_wave, cand_flux, color="navy", linewidth=1.0, label="Candidate")

        # Molecular band shading
        for l1, l2, c in [(3830, 3883, "blue"), (4120, 4216, "green"), (4285, 4315, "red")]:
            ax.axvspan(l1, l2, alpha=0.12, color=c, zorder=0)

        prob = row.get(prob_col, np.nan)
        ax.set_title(f"#{i+1} | prob={prob:.3f}", fontsize=8)
        ax.set_xlim(3800, 4500)
        ax.tick_params(labelsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("Norm Flux", fontsize=8)
        if i >= len(top) - n_cols:
            ax.set_xlabel("Wavelength (A)", fontsize=8)
        ax.grid(alpha=0.2, linestyle="--")
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    for j in range(len(top), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig


def plot_teff_logg_distribution(
    candidates_df: pd.DataFrame,
    stars_clean: pd.DataFrame,
    prob_col: str = "prob",
    title: str = "Candidate Distribution in Teff-Logg Space",
):
    """Scatter plot: candidates vs known CN stars in Teff-Logg."""
    need = ["teff", "logg", "feh", "label"]
    need = [c for c in need if c in stars_clean.columns]
    plot_base = stars_clean[need].dropna(subset=["teff", "logg", "feh"]).copy()

    fig, ax = plt.subplots(figsize=(9, 7))

    bg = plot_base[plot_base["label"] == -1]
    if len(bg) > 5000:
        bg = bg.sample(5000, random_state=42)
    ax.scatter(bg["teff"], bg["logg"], s=5, c="#b0b0b0", alpha=0.18,
               edgecolors="none", label="Unlabeled (bg)")

    known = plot_base[plot_base["label"] == 1]
    sc = ax.scatter(known["teff"], known["logg"], c=known["feh"], cmap="viridis",
                    s=80, marker="*", edgecolors="black", linewidth=0.4,
                    label=f"Known CN (n={len(known)})", zorder=3)

    # Candidates: match by index
    cand_idx = candidates_df.index.intersection(plot_base.index)
    if len(cand_idx) > 0:
        cand_plot = plot_base.loc[cand_idx]
        ax.scatter(cand_plot["teff"], cand_plot["logg"], facecolors="none",
                   edgecolors="#e74c3c", s=35, linewidths=1.0,
                   label=f"Candidates (n={len(cand_plot)})", zorder=2)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("[Fe/H]")
    ax.set_xlabel("Teff")
    ax.set_ylabel("logg")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_training_curves(history_list: List[Dict], metric: str = "loss",
                         title: str = "Training Curves"):
    """Plot training/validation curves from multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, hist in enumerate(history_list):
        label = hist.get("label", f"Run {i+1}")
        epochs = range(1, len(hist["train"]) + 1)
        ax.plot(epochs, hist["train"], alpha=0.7, label=f"{label} (train)")
        if "val" in hist:
            ax.plot(epochs, hist["val"], alpha=0.7, linestyle="--",
                    label=f"{label} (val)")
    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
