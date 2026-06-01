"""Evaluation utilities for EndToEndPU classifier.

Computes ranking metrics, top-K precision/recall, and candidate export.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    topk_list: List[int] = [50, 100, 200, 500],
    name: str = "",
) -> Dict:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    probs : (N,) predicted probabilities (higher → more likely CN).
    y_true : (N,) binary labels (1 = CN star, 0 = not CN / unlabeled).
    topk_list : list of int
        K values for top-K precision/recall.
    name : str
        Label for this evaluation.

    Returns
    -------
    dict with metrics.
    """
    n_pos = int(y_true.sum())
    n_total = len(y_true)

    # Ranking order
    order = np.argsort(probs)[::-1]

    # AUROC / AUPRC
    auroc = roc_auc_score(y_true, probs) if n_pos > 0 and n_pos < n_total else float("nan")
    auprc = average_precision_score(y_true, probs) if n_pos > 0 else float("nan")

    # Top-K
    topk = {}
    for k in topk_list:
        k_eff = min(k, n_total)
        top_idx = order[:k_eff]
        tp = int(y_true[top_idx].sum())
        topk[f"precision@{k}"] = tp / k_eff if k_eff > 0 else 0.0
        topk[f"recall@{k}"] = tp / max(n_pos, 1)

    # Median rank
    ranks = np.zeros(n_total, dtype=int)
    ranks[order] = np.arange(n_total)
    if n_pos > 0:
        pos_ranks = ranks[y_true == 1]
        median_rank_pct = float(np.median(pos_ranks + 1) / n_total)
    else:
        median_rank_pct = float("nan")

    # ROC/PR curves
    if n_pos > 0 and n_pos < n_total:
        fpr, tpr, _ = roc_curve(y_true, probs)
        precision, recall, _ = precision_recall_curve(y_true, probs)
    else:
        fpr = tpr = precision = recall = None

    return {
        "name": name,
        "auroc": auroc,
        "auprc": auprc,
        "precision@50": topk.get("precision@50", float("nan")),
        "precision@100": topk.get("precision@100", float("nan")),
        "precision@200": topk.get("precision@200", float("nan")),
        "precision@500": topk.get("precision@500", float("nan")),
        "recall@50": topk.get("recall@50", float("nan")),
        "recall@100": topk.get("recall@100", float("nan")),
        "recall@200": topk.get("recall@200", float("nan")),
        "recall@500": topk.get("recall@500", float("nan")),
        "median_rank_pct": median_rank_pct,
        "prob_mean_pos": float(probs[y_true == 1].mean()) if n_pos > 0 else float("nan"),
        "prob_mean_neg": float(probs[y_true == 0].mean()) if n_pos < n_total else float("nan"),
        "n_pos": n_pos,
        "n_total": n_total,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision,
        "recall_curve": recall,
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_pos: Optional[np.ndarray] = None,
    X_train_unl: Optional[np.ndarray] = None,
    meta: Optional[pd.DataFrame] = None,
    test_indices: Optional[np.ndarray] = None,
    topk_list: List[int] = [50, 100, 200, 500],
    device: str = "cpu",
    batch_size: int = 512,
) -> Dict:
    """Full evaluation: test set metrics + prediction on whole dataset.

    Parameters
    ----------
    model : SpectraResNet_CN_Attention
        Trained model.
    X_test, y_test : test data (may be None if no test holdout).
    X_train_pos, X_train_unl : training data for whole-dataset scoring.
    meta : metadata DataFrame (for candidate export).
    test_indices : indices of test data in meta (for excluding from candidates).
    topk_list : K values for top-K metrics.
    device : torch device.
    batch_size : inference batch size.

    Returns
    -------
    dict with keys: test_metrics, all_probs, candidates_df, summary_str.
    """
    import torch

    dev = torch.device(device)

    @torch.no_grad()
    def _predict(X_arr, batch_size=512):
        was_training = model.training
        model.eval()
        model.to(dev)
        X_t = torch.from_numpy(X_arr).float().to(dev)
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            preds.append(model(xb).cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(preds)

    results = {}

    # ── 1. Test set evaluation ──
    if X_test is not None and len(X_test) > 0:
        test_probs = _predict(X_test, batch_size=batch_size)
        y_test_bin = (y_test == 1).astype(int)
        results["test"] = compute_metrics(test_probs, y_test_bin, topk_list, name="Test")
        results["test"]["probs"] = test_probs

    # ── 2. Score on TRAINING positives (sanity check) ──
    if X_train_pos is not None and len(X_train_pos) > 0:
        train_pos_probs = _predict(X_train_pos, batch_size=batch_size)
        results["train_pos"] = {
            "n": len(X_train_pos),
            "prob_mean": float(train_pos_probs.mean()),
            "prob_std": float(train_pos_probs.std()),
            "prob_min": float(train_pos_probs.min()),
            "prob_max": float(train_pos_probs.max()),
        }

    # ── 3. Score on all unlabeled for candidate ranking ──
    if X_train_unl is not None and meta is not None and len(X_train_unl) > 0:
        # Exclude test indices from candidate pool
        if test_indices is not None and len(test_indices) > 0:
            unl_mask = ~np.isin(np.arange(len(meta)), test_indices)
            X_pool = X_train_unl  # already excludes test
            meta_pool = meta.iloc[np.arange(len(meta))[unl_mask]]
        else:
            X_pool = X_train_unl
            meta_pool = meta

        all_probs = _predict(X_pool, batch_size=batch_size)
        order = np.argsort(all_probs)[::-1]

        candidates = meta_pool.iloc[order].copy()
        candidates["cn_prob"] = all_probs[order]
        candidates["cn_rank"] = np.arange(1, len(candidates) + 1)

        results["candidates_df"] = candidates
        results["all_probs"] = all_probs

    # ── 4. Summary ──
    results["summary"] = _build_summary(results)
    return results


def _build_summary(results: Dict) -> str:
    """Generate human-readable evaluation summary."""
    lines = ["=" * 64, "End-to-End PU Network — Evaluation", "=" * 64]

    if "test" in results:
        t = results["test"]
        lines.append(f"\n--- Test Set ({t['n_pos']} pos / {t['n_total']} total) ---")
        lines.append(f"  AUROC:            {t['auroc']:.4f}")
        lines.append(f"  AUPRC:            {t['auprc']:.4f}")
        lines.append(f"  Median rank %:    {t['median_rank_pct']:.4f}  (lower=better)")
        for k in [50, 100, 200, 500]:
            pk = t.get(f"precision@{k}", float("nan"))
            rk = t.get(f"recall@{k}", float("nan"))
            lines.append(f"  P@{k:3d}:            {pk:.4f}  (recall: {rk:.4f})")

    if "train_pos" in results:
        tp = results["train_pos"]
        lines.append(f"\n--- Training Positives ({tp['n']} samples) ---")
        lines.append(f"  Mean prob:        {tp['prob_mean']:.4f} ± {tp['prob_std']:.4f}")
        lines.append(f"  Range:            [{tp['prob_min']:.4f}, {tp['prob_max']:.4f}]")

    lines.append("=" * 64)
    return "\n".join(lines)


def export_candidates(
    candidates_df: pd.DataFrame,
    top_n: int = 200,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Export top CN-star candidates to CSV.

    Parameters
    ----------
    candidates_df : DataFrame sorted by cn_prob descending.
    top_n : max number of candidates.
    output_csv : path to save CSV (optional).

    Returns
    -------
    DataFrame of top candidates.
    """
    df = candidates_df.head(top_n).copy()

    if output_csv:
        cols = ["cn_prob", "cn_rank"]
        extra_cols = ["uid", "ra", "dec", "teff", "logg", "feh", "snru"]
        for c in extra_cols:
            if c in df.columns:
                cols.append(c)
        # Add band index columns if present
        for bc in ["CN3839", "CN4142", "CH4300", "delta_CN3839", "delta_CN4142"]:
            if bc in df.columns:
                cols.append(bc)
        df[cols].to_csv(output_csv, index=False)
        print(f"Candidates exported to: {output_csv}")

    return df
