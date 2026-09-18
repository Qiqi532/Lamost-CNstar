"""Evaluation utilities for DeepSVDD on LAMOST CN-star detection.

Computes anomaly scores, ranks candidates, and produces evaluation metrics
comparing known CN stars against the unlabeled background population.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import SpectraDataset
from .models import DeepSVDD


@torch.no_grad()
def compute_anomaly_scores(
    model: DeepSVDD,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Compute anomaly scores for all spectra.

    Parameters
    ----------
    model : DeepSVDD
        Trained model.
    X : np.ndarray
        Spectra matrix, shape (n_samples, n_features).
    batch_size : int
        Batch size for inference.
    device : str
        Device for computation.

    Returns
    -------
    np.ndarray
        Anomaly scores, shape (n_samples,). Higher = more anomalous.
    """
    model.eval()
    model.to(device)

    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores = []
    for batch in loader:
        x = batch[0].to(device)
        score = model.anomaly_score(x)
        scores.append(score.cpu().numpy())

    return np.concatenate(scores)


def evaluate_model(
    model: DeepSVDD,
    dataset: SpectraDataset,
    batch_size: int = 512,
    device: str = "cpu",
    topk_list: List[int] = [50, 100, 200],
) -> Dict:
    """Evaluate DeepSVDD performance against known CN stars.

    Treats known CN stars as the positive class and evaluates how well
    anomaly scores separate them from unlabeled background stars.

    Returns
    -------
    Dict with keys:
        'scores_all' : anomaly scores for all samples
        'scores_positive' : scores for known CN stars
        'scores_unlabeled' : scores for unlabeled stars
        'auroc' : Area under ROC curve (higher = better separation)
        'auprc' : Area under PR curve
        'precision@k' : Precision in top-k ranked candidates
        'recall@k' : Recall in top-k ranked candidates
        'median_rank_pct' : Median percentile rank of known CN stars
        'summary_df' : DataFrame of per-sample scores with metadata
    """
    X_all = dataset.X.astype(np.float32)
    y_all = dataset.y.copy()

    scores_all = compute_anomaly_scores(model, X_all, batch_size, device)

    pos_mask = y_all == 1
    scores_pos = scores_all[pos_mask]
    scores_unl = scores_all[~pos_mask]

    n_pos = int(pos_mask.sum())
    n_total = len(scores_all)

    # Rank-based metrics
    order = np.argsort(scores_all)[::-1]  # descending anomaly score
    ranks = np.zeros(n_total, dtype=int)
    ranks[order] = np.arange(n_total)
    pos_ranks = ranks[pos_mask]

    # AUROC / AUPRC using scikit-learn
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_binary = (y_all == 1).astype(int)
    auroc = roc_auc_score(y_binary, scores_all)
    auprc = average_precision_score(y_binary, scores_all)

    # Precision/Recall at top-k
    topk = {}
    for k in topk_list:
        top_indices = order[:k]
        tp = int(y_binary[top_indices].sum())
        topk[f"precision@{k}"] = tp / k
        topk[f"recall@{k}"] = tp / max(n_pos, 1)

    # Build per-sample summary
    summary_df = dataset.meta.copy()
    summary_df["anomaly_score"] = scores_all
    summary_df["anomaly_rank"] = ranks + 1
    summary_df["anomaly_pct"] = (ranks + 1) / n_total  # small = high score
    summary_df = summary_df.sort_values("anomaly_score", ascending=False)

    return {
        "scores_all": scores_all,
        "scores_positive": scores_pos,
        "scores_unlabeled": scores_unl,
        "auroc": auroc,
        "auprc": auprc,
        **topk,
        "n_positive": n_pos,
        "n_total": n_total,
        "median_rank_pct": float(np.median(pos_ranks + 1) / n_total),
        "summary_df": summary_df,
    }


def select_candidates(
    metrics: Dict,
    top_n: int = 200,
    min_anomaly_score: Optional[float] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Select top CN-star candidates based on anomaly scores.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model().
    top_n : int
        Number of top candidates to select.
    min_anomaly_score : float or None
        Optional minimum anomaly score threshold.
    output_csv : str or None
        If provided, save candidates to this CSV path.

    Returns
    -------
    pd.DataFrame
        Top candidates with metadata and anomaly scores.
    """
    df = metrics["summary_df"].copy()

    # Filter to only unlabeled samples
    candidates = df[df["label"] == -1].copy()

    if min_anomaly_score is not None:
        candidates = candidates[candidates["anomaly_score"] >= min_anomaly_score]

    candidates = candidates.head(top_n)

    if output_csv:
        cols = [
            "anomaly_score", "anomaly_pct",
            "ra", "dec", "teff", "logg", "feh",
            "snru", "mag_ps_g", "filepath",
        ]
        avail_cols = [c for c in cols if c in candidates.columns]
        candidates[avail_cols].to_csv(output_csv, index=False)

    return candidates


def summarize_results(metrics: Dict) -> str:
    """Generate a human-readable summary of evaluation results.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model().

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = [
        "=" * 60,
        "DeepSVDD Evaluation Results",
        "=" * 60,
        f"Total samples:       {metrics['n_total']:,}",
        f"Known CN stars:      {metrics['n_positive']}",
        "",
        "--- Separation Metrics ---",
        f"AUROC:               {metrics['auroc']:.4f}",
        f"AUPRC:               {metrics['auprc']:.4f}",
        f"Median rank (pos):   {metrics['median_rank_pct']:.4f} "
        f"(lower = better, random = 0.5)",
        "",
        "--- Top-K Precision ---",
    ]

    for k in [50, 100, 200]:
        pk = metrics.get(f"precision@{k}")
        rk = metrics.get(f"recall@{k}")
        if pk is not None:
            lines.append(
                f"  precision@{k:3d}:     {pk:.4f}  "
                f"(recall: {rk:.4f})"
            )

    lines.extend([
        "",
        "--- Anomaly Score Statistics ---",
        f"  Known CN — mean:   {np.mean(metrics['scores_positive']):.4f}, "
        f"std: {np.std(metrics['scores_positive']):.4f}",
        f"  Unlabeled — mean:  {np.mean(metrics['scores_unlabeled']):.4f}, "
        f"std: {np.std(metrics['scores_unlabeled']):.4f}",
        "=" * 60,
    ])

    return "\n".join(lines)
