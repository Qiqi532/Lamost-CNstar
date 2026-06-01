"""Evaluation utilities for binary CN-star classifier.

Computes classification probabilities and evaluates against independent
test sets: held-out GCS, CNstar, and FT_cands.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .binary_trainer import BinaryClassifier


def evaluate_classifier(
    model: BinaryClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_name: str = "test",
    batch_size: int = 512,
    device: str = "cpu",
    topk_list: List[int] = [50, 100, 200, 500],
) -> Dict:
    """Evaluate binary classifier on a labeled test set.

    Parameters
    ----------
    model : BinaryClassifier
        Trained model.
    X_test : np.ndarray (n_samples, n_features)
        Test spectra.
    y_test : np.ndarray (n_samples,)
        Binary labels (1 = CN star, 0 = normal).
    test_name : str
        Name for this test set (e.g., 'CNstar', 'GCS-heldout').
    batch_size : int
        Inference batch size.
    device : str
    topk_list : list of int

    Returns
    -------
    Dict with keys: probs, auroc, auprc, precision@k, recall@k,
    n_positive, n_total, median_rank_pct, test_name.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    probs = model.predict(X_test, batch_size=batch_size)

    n_pos = int(y_test.sum())
    n_total = len(y_test)

    # Ranking metrics
    order = np.argsort(probs)[::-1]
    ranks = np.zeros(n_total, dtype=int)
    ranks[order] = np.arange(n_total)
    pos_ranks = ranks[y_test == 1]

    # AUROC / AUPRC (handle edge cases with only one class)
    auroc = roc_auc_score(y_test, probs) if n_pos > 0 and n_pos < n_total else float("nan")
    auprc = average_precision_score(y_test, probs) if n_pos > 0 else float("nan")

    # Top-k metrics
    topk = {}
    for k in topk_list:
        k_eff = min(k, n_total)
        top_indices = order[:k_eff]
        tp = int(y_test[top_indices].sum())
        topk[f"precision@{k}"] = tp / k_eff if k_eff > 0 else 0.0
        topk[f"recall@{k}"] = tp / max(n_pos, 1)

    return {
        "test_name": test_name,
        "probs": probs,
        "auroc": auroc,
        "auprc": auprc,
        **topk,
        "n_positive": n_pos,
        "n_total": n_total,
        "median_rank_pct": float(np.median(pos_ranks + 1) / n_total) if n_pos > 0 else float("nan"),
        "prob_mean_pos": float(probs[y_test == 1].mean()) if n_pos > 0 else float("nan"),
        "prob_mean_neg": float(probs[y_test == 0].mean()) if n_pos < n_total else float("nan"),
    }


def evaluate_on_all_test_sets(
    model: BinaryClassifier,
    X_gcs_heldout: np.ndarray,
    X_cnstar: np.ndarray,
    X_ft_cands: Optional[np.ndarray] = None,
    X_unlabeled: Optional[np.ndarray] = None,
    meta_unlabeled: Optional[pd.DataFrame] = None,
    batch_size: int = 512,
    device: str = "cpu",
    topk_list: List[int] = [50, 100, 200, 500],
) -> Dict:
    """Evaluate classifier on all independent test sets.

    Parameters
    ----------
    model : BinaryClassifier
    X_gcs_heldout : np.ndarray
        Held-out GCS spectra (all positive).
    X_cnstar : np.ndarray
        CNstar spectra (all positive).
    X_ft_cands : np.ndarray or None
        FT_cands spectra (all positive).
    X_unlabeled : np.ndarray or None
        Full unlabeled dataset for candidate ranking.
    meta_unlabeled : pd.DataFrame or None
        Metadata for unlabeled stars.
    batch_size : int
    device : str
    topk_list : list of int

    Returns
    -------
    Dict with keys: results (per-test-set dicts), candidates_df,
    summary_str.
    """
    results = {}

    # 1. Held-out GCS
    if len(X_gcs_heldout) > 0:
        y_gcs = np.ones(len(X_gcs_heldout))
        results["GCS_heldout"] = evaluate_classifier(
            model, X_gcs_heldout, y_gcs, "GCS_heldout",
            batch_size, device, topk_list,
        )

    # 2. CNstar
    if len(X_cnstar) > 0:
        y_cnstar = np.ones(len(X_cnstar))
        results["CNstar"] = evaluate_classifier(
            model, X_cnstar, y_cnstar, "CNstar",
            batch_size, device, topk_list,
        )

    # 3. FT_cands
    if X_ft_cands is not None and len(X_ft_cands) > 0:
        y_ft = np.ones(len(X_ft_cands))
        results["FT_cands"] = evaluate_classifier(
            model, X_ft_cands, y_ft, "FT_cands",
            batch_size, device, topk_list,
        )

    # 4. Score unlabeled and select candidates
    candidates_df = None
    if X_unlabeled is not None and meta_unlabeled is not None and len(X_unlabeled) > 0:
        probs_unl = model.predict(X_unlabeled, batch_size=batch_size)
        meta = meta_unlabeled.copy()
        meta["cn_prob"] = probs_unl

        order = np.argsort(probs_unl)[::-1]
        meta["cn_rank"] = np.argsort(order) + 1

        candidates_df = meta.sort_values("cn_prob", ascending=False)
        results["unlabeled"] = {
            "n_total": len(X_unlabeled),
            "prob_mean": float(probs_unl.mean()),
            "prob_std": float(probs_unl.std()),
            "prob_p90": float(np.percentile(probs_unl, 90)),
            "prob_p95": float(np.percentile(probs_unl, 95)),
            "prob_p99": float(np.percentile(probs_unl, 99)),
        }

    # Build summary string
    summary = summarize_binary_results(results)

    return {
        "results": results,
        "candidates_df": candidates_df,
        "summary": summary,
    }


def summarize_binary_results(results: Dict) -> str:
    """Generate human-readable summary of binary classifier evaluation."""
    lines = [
        "=" * 60,
        "Binary Classifier — Multi-Test Evaluation",
        "=" * 60,
    ]

    for name, res in results.items():
        if name == "unlabeled":
            lines.extend([
                f"\n--- Unlabeled Set ({res['n_total']:,} stars) ---",
                f"  Mean prob:        {res['prob_mean']:.4f}",
                f"  Std prob:         {res['prob_std']:.4f}",
                f"  P90 prob:         {res['prob_p90']:.4f}",
                f"  P95 prob:         {res['prob_p95']:.4f}",
                f"  P99 prob:         {res['prob_p99']:.4f}",
            ])
            continue

        lines.extend([
            f"\n--- {name} ({res['n_positive']} known CN / {res['n_total']} total) ---",
            f"  AUROC:            {res['auroc']:.4f}" if not np.isnan(res['auroc']) else "  AUROC:            N/A",
            f"  AUPRC:            {res['auprc']:.4f}" if not np.isnan(res['auprc']) else "  AUPRC:            N/A",
            f"  Median rank pct:  {res['median_rank_pct']:.4f}  (lower = better, random = 0.5)",
            f"  Mean prob (pos):  {res['prob_mean_pos']:.4f}",
            f"  Mean prob (neg):  {res['prob_mean_neg']:.4f}" if not np.isnan(res.get('prob_mean_neg', float('nan'))) else "",
        ])

        for k in [50, 100, 200, 500]:
            pk = res.get(f"precision@{k}")
            rk = res.get(f"recall@{k}")
            if pk is not None:
                lines.append(
                    f"  precision@{k:3d}:     {pk:.4f}  (recall: {rk:.4f})"
                )

    lines.append("=" * 60)
    return "\n".join(lines)


def select_candidates_binary(
    candidates_df: pd.DataFrame,
    top_n: int = 200,
    min_prob: float = 0.5,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Select top CN-star candidates from scored unlabeled stars.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Output from evaluate_on_all_test_sets, sorted by cn_prob descending.
    top_n : int
        Maximum number of candidates.
    min_prob : float
        Minimum probability threshold.
    output_csv : str or None

    Returns
    -------
    pd.DataFrame of top candidates.
    """
    df = candidates_df.copy()

    # Filter
    df = df[df["cn_prob"] >= min_prob]
    df = df.head(top_n)

    if output_csv:
        cols = [
            "cn_prob", "cn_rank",
            "ra", "dec", "teff", "logg", "feh",
            "snru", "mag_ps_g", "filepath",
        ]
        avail_cols = [c for c in cols if c in df.columns]
        df[avail_cols].to_csv(output_csv, index=False)
        print(f"Candidates saved to: {output_csv}")

    return df
