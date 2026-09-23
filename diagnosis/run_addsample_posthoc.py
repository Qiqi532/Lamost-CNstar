"""Post-hoc contrasts for the add-sample experiment.

Reads the saved outputs of ``run_addsample_experiment.py`` and answers the
questions the main run left open:

1. The random-label control degraded AUC **significantly** (-0.0275, CI excludes
   zero), yet none of the real additions reached significance on their own.
   That contrast is the informative one: do the real labels beat the fake ones?
2. Is adding the confirmed negatives (M2) better than adding only positives (M1)?
3. Does the recall@500 gain survive a paired per-fold test?
4. Per-object view: for the 46 new positives and the 53 new negatives, which way
   does each one's out-of-fold score move when the labels are added?

Everything here is a re-analysis of already exported files -- no model is
refitted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CONFIGS = ("M0_baseline", "M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only", "M0_control")
TOPK = (500, 1000)


def paired_bootstrap_auc(y: np.ndarray, sa: np.ndarray, sb: np.ndarray,
                         n_boot: int = 8000, seed: int = 23):
    rng = np.random.default_rng(seed)
    n = y.size
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    d = np.asarray(d)
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_lower_tail": float(np.mean(d <= 0.0)),
        "significant_gain": bool(np.percentile(d, 2.5) > 0.0),
        "significant_loss": bool(np.percentile(d, 97.5) < 0.0),
    }


def paired_score_delta(sa: np.ndarray, sb: np.ndarray, mask: np.ndarray,
                       n_boot: int = 8000, seed: int = 31):
    """Bootstrap the mean paired difference of out-of-fold scores."""
    rng = np.random.default_rng(seed)
    d = sa[mask] - sb[mask]
    boot = np.array([rng.choice(d, size=d.size, replace=True).mean()
                     for _ in range(n_boot)])
    return {
        "n": int(d.size),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "share_increased": float(np.mean(d > 0)),
        "share_decreased": float(np.mean(d < 0)),
    }


def main() -> None:
    started = time.perf_counter()
    oof = pd.read_csv(RESULTS_DIR / "addsample_oof_scores.csv")
    folds = pd.read_csv(RESULTS_DIR / "addsample_fold_metrics.csv")
    y = oof["label"].to_numpy()
    scores = {c: oof[f"oof_score_{c}"].to_numpy(dtype=float) for c in CONFIGS}
    print(f"[in] {len(oof)} labelled objects, {y.sum()} positives", flush=True)

    base = scores["M0_baseline"]
    contrasts = {}
    for name in ("M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only", "M0_control"):
        contrasts[f"{name}_vs_M0_baseline"] = paired_bootstrap_auc(y, scores[name], base)
    # the key one: do real labels beat the random-label control?
    for name in ("M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only"):
        contrasts[f"{name}_vs_M0_control"] = paired_bootstrap_auc(
            y, scores[name], scores["M0_control"]
        )
    contrasts["M2_pos_plus_neg_vs_M1_add_pos"] = paired_bootstrap_auc(
        y, scores["M2_pos_plus_neg"], scores["M1_add_pos"]
    )
    for k, v in contrasts.items():
        print(f"[contrast] {k:<32} dAUC={v['mean_delta']:+.4f} "
              f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
              f"{'GAIN' if v['significant_gain'] else ('LOSS' if v['significant_loss'] else 'n.s.')}",
              flush=True)

    # per-object movement of the out-of-fold scores
    pos_mask = y == 1
    neg_mask = y == 0
    movement = {}
    for name in ("M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only", "M0_control"):
        movement[name] = {
            "new_positives": paired_score_delta(scores[name], base, pos_mask),
            "new_negatives": paired_score_delta(scores[name], base, neg_mask),
        }
        p, ng = movement[name]["new_positives"], movement[name]["new_negatives"]
        print(f"[move] {name:<16} positives {p['mean_delta']:+.4f} "
              f"({p['share_increased'] * 100:.0f}% up)  |  negatives "
              f"{ng['mean_delta']:+.4f} ({ng['share_increased'] * 100:.0f}% up)",
              flush=True)

    # paired per-fold test on recall@K
    rec = {}
    for K in TOPK:
        col = f"recall_at_{K}"
        piv = folds.pivot(index="fold", columns="config", values=col)
        entry = {}
        for name in ("M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only", "M0_control"):
            d = (piv[name] - piv["M0_baseline"]).to_numpy()
            n_improved = int((d > 1e-9).sum())
            n_worse = int((d < -1e-9).sum())
            entry[name] = {
                "mean_delta": float(d.mean()),
                "per_fold_delta": d.tolist(),
                "n_folds_improved": n_improved,
                "n_folds_tied": int(len(d) - n_improved - n_worse),
                "n_folds_worse": n_worse,
                "t_statistic": (float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
                                if d.std(ddof=1) > 0 else None),
            }
            t_stat = entry[name]["t_statistic"]
            t_txt = f"{t_stat:+.2f}" if t_stat is not None else "n/a"
            print(f"[recall@{K}] {name:<16} delta={d.mean():+.4f}  "
                  f"improved {n_improved}/{len(d)} folds  t={t_txt}", flush=True)
        rec[f"recall_at_{K}"] = entry

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "analysis": "post-hoc contrasts for the add-sample experiment",
        "n_labelled": int(len(oof)),
        "n_positives": int(y.sum()),
        "n_negatives": int((y == 0).sum()),
        "auc_contrasts": contrasts,
        "score_movement": movement,
        "recall_paired_tests": rec,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "addsample_posthoc.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"[out] {RESULTS_DIR / 'addsample_posthoc.json'}", flush=True)


if __name__ == "__main__":
    main()
