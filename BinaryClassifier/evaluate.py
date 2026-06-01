"""Evaluation utilities for binary CN-star classifier."""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def _compute_metrics(
    probs: np.ndarray,
    y_test: np.ndarray,
    topk_list: List[int] = [50, 100, 200, 500],
) -> Dict:
    """Shared metric computation for single model or ensemble."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    n_pos = int(y_test.sum())
    n_total = len(y_test)
    order = np.argsort(probs)[::-1]
    ranks = np.zeros(n_total, dtype=int)
    ranks[order] = np.arange(n_total)
    pos_ranks = ranks[y_test == 1] if n_pos > 0 else np.array([])

    auroc = roc_auc_score(y_test, probs) if n_pos > 0 and n_pos < n_total else float("nan")
    auprc = average_precision_score(y_test, probs) if n_pos > 0 else float("nan")

    topk = {}
    for k in topk_list:
        k_eff = min(k, n_total)
        tp = int(y_test[order[:k_eff]].sum())
        topk[f"precision@{k}"] = tp / k_eff if k_eff > 0 else 0.0
        topk[f"recall@{k}"] = tp / max(n_pos, 1)

    return {
        "probs": probs, "auroc": auroc, "auprc": auprc,
        **topk, "n_positive": n_pos, "n_total": n_total,
        "median_rank_pct": float(np.median(pos_ranks + 1) / n_total) if len(pos_ranks) > 0 else float("nan"),
        "prob_mean_pos": float(probs[y_test == 1].mean()) if n_pos > 0 else float("nan"),
        "prob_mean_neg": float(probs[y_test == 0].mean()) if n_pos < n_total else float("nan"),
    }


def evaluate_model(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_name: str = "test",
    batch_size: int = 512,
    device: str = "cpu",
    topk_list: List[int] = [50, 100, 200, 500],
    _already_predicted: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate a model (or ensemble) on a labeled test set.

    `predict_fn` can be model.predict or ensemble.predict.
    Pass `_already_predicted` to skip re-prediction.
    """
    probs = _already_predicted if _already_predicted is not None else predict_fn(X_test)
    result = _compute_metrics(probs, y_test, topk_list)
    result["test_name"] = test_name
    return result


def evaluate_ensemble(
    ensemble,
    X_gcs: np.ndarray,
    X_cnstar: np.ndarray,
    X_unlabeled: Optional[np.ndarray] = None,
    meta_unlabeled: Optional[pd.DataFrame] = None,
    batch_size: int = 512,
    device: str = "cpu",
    topk_list: List[int] = [50, 100, 200, 500],
) -> Dict:
    """Evaluate ensemble on held-out GCS, CNstar, and optionally unlabeled pool."""
    results = {}

    for name, X_test in [("GCS", X_gcs), ("CNstar", X_cnstar)]:
        if len(X_test) > 0:
            y_test = np.ones(len(X_test))
            results[name] = evaluate_model(
                ensemble.predict, X_test, y_test, name, batch_size, device, topk_list,
            )

    candidates_df = None
    if X_unlabeled is not None and meta_unlabeled is not None and len(X_unlabeled) > 0:
        probs, stds = ensemble.predict_with_std(X_unlabeled, batch_size, device)
        meta = meta_unlabeled.copy()
        meta["cn_prob"] = probs
        meta["cn_std"] = stds
        meta["cn_rank"] = np.argsort(np.argsort(probs)[::-1]) + 1
        candidates_df = meta.sort_values("cn_prob", ascending=False)
        results["unlabeled"] = {
            "n_total": len(X_unlabeled),
            "prob_mean": float(probs.mean()),
            "prob_std": float(probs.std()),
            "prob_p90": float(np.percentile(probs, 90)),
            "prob_p95": float(np.percentile(probs, 95)),
            "prob_p99": float(np.percentile(probs, 99)),
        }

    return {"results": results, "candidates_df": candidates_df,
            "summary": _summarize(results)}


def _summarize(results: Dict) -> str:
    lines = ["=" * 60, "Binary Classifier -- Evaluation", "=" * 60]
    for name, res in results.items():
        if name == "unlabeled":
            lines.extend([
                f"\n--- Unlabeled ({res['n_total']:,} stars) ---",
                f"  Mean prob:    {res['prob_mean']:.4f}",
                f"  Std prob:     {res['prob_std']:.4f}",
                f"  P90/P95/P99:  {res['prob_p90']:.4f} / {res['prob_p95']:.4f} / {res['prob_p99']:.4f}",
            ])
            continue
        lines.extend([
            f"\n--- {name} ({res['n_positive']} pos / {res['n_total']} total) ---",
            f"  AUROC:            {res['auroc']:.4f}" if not np.isnan(res['auroc']) else "  AUROC:            N/A",
            f"  AUPRC:            {res['auprc']:.4f}" if not np.isnan(res['auprc']) else "  AUPRC:            N/A",
            f"  Median rank:      {res['median_rank_pct']:.4f}  (<0.5 = better)",
            f"  Mean prob (pos):  {res['prob_mean_pos']:.4f}",
        ])
        for k in [50, 100, 200, 500]:
            pk = res.get(f"precision@{k}")
            rk = res.get(f"recall@{k}")
            if pk is not None:
                lines.append(f"  P@{k:3d}: {pk:.4f}  (R: {rk:.4f})")
    lines.append("=" * 60)
    return "\n".join(lines)


def export_candidates(
    candidates_df: pd.DataFrame,
    top_n: int = 200,
    min_prob: float = 0.5,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = candidates_df[candidates_df["cn_prob"] >= min_prob].head(top_n)
    if output_csv:
        cols = ["cn_prob", "cn_std", "cn_rank", "ra", "dec", "teff", "logg", "feh",
                "snru", "mag_ps_g", "filepath"]
        avail = [c for c in cols if c in df.columns]
        df[avail].to_csv(output_csv, index=False)
        print(f"Candidates saved: {output_csv}")
    return df
