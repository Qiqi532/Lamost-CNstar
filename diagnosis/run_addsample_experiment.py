"""Does adding the newly confirmed positives actually improve the classifier?

Design
------
The 99 high-resolution labelled candidates (46 confirmed positives, 53 confirmed
non-enhanced) are the only independent labels in the project.  They are used in
a 5-fold cross-fit so that **every** labelled object is scored out of fold:

    for each fold:
        HO  = 1/5 of the 99      -> never used for training, kept in U,
                                    used only for scoring
        HI  = the other 4/5      -> available to the training set

Five ways of using the new information are compared, all under identical folds,
bag counts and hyper-parameters:

    M0_baseline       P = 91 frozen positives only
    M1_add_pos        P = 91 + HI positives
    M2_pos_plus_neg   P = 91 + HI positives, and HI negatives are treated as
                      *reliable negatives*: removed from U and forced into every
                      bag's negative sample
    M4_new_pos_only   P = HI positives only (can the new labels stand alone?)
    M0_control        P = 91 + as many random U objects as HI has positives
                      (negative control: is any gain just from having more rows?)

Metrics are computed on HO only:

    AUC(HO)          separates the held-out confirmed positives from the
                     held-out confirmed negatives -- the quantity that actually
                     matters, and independent of the size of the U pool
    recall@K         how many HO positives appear in the top-K of the config's
                     own unlabeled ranking (K = 500 / 1000 / 2500)
    precision_HO@K   among the HO objects inside that top-K, the share that are
                     real positives

Interpretation rule fixed in advance
------------------------------------
    improvement is accepted only if
        AUC(M1) > AUC(M0) beyond the across-fold spread, AND
        AUC(M0_control) shows no such gain
    If M0_control also improves, the gain comes from having more training rows
    rather than from the new labels, and the experiment is inconclusive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "crossfit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import crossfit_engine as ce  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LABEL_CSV = HERE / "cands_label.csv"
FROZEN_ARRAYS = (PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
                 / "crossfit_arrays_recall90.npz")

U_RATIO = 10
MAX_DEPTH = 3
MIN_CHILD_WEIGHT = 1
REG_LAMBDA = 1.0
NUM_BOOST_ROUND = 50
N_FOLDS = 5
N_BAGS = 60
SEED = 42
TOPK = (500, 1000, 2500)

CONFIGS = ("M0_baseline", "M1_add_pos", "M2_pos_plus_neg", "M4_new_pos_only", "M0_control")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOURS = {
    "M0_baseline": "#888780",
    "M1_add_pos": "#1D9E75",
    "M2_pos_plus_neg": "#185FA5",
    "M4_new_pos_only": "#534AB7",
    "M0_control": "#BA7517",
}


def xgb_params(seed: int) -> dict:
    return {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": MAX_DEPTH, "eta": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": MIN_CHILD_WEIGHT,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": REG_LAMBDA,
        "tree_method": "hist", "base_score": 0.5, "seed": int(seed),
        "verbosity": 0, "nthread": 1,
    }


def pu_bagging(X: np.ndarray, dall: xgb.DMatrix, pos_idx: np.ndarray,
               u_idx: np.ndarray, known_neg_idx: np.ndarray,
               n_bags: int, seed: int, verbose: bool = False) -> np.ndarray:
    """PU bagging over the whole object set; returns the mean score of every row."""

    rng = np.random.default_rng(seed)
    n_pos = pos_idx.size
    need = n_pos * U_RATIO
    total = np.zeros(X.shape[0], dtype=np.float64)

    known_neg_idx = np.asarray(known_neg_idx, dtype=int)
    if known_neg_idx.size:
        take = min(known_neg_idx.size, need)
        forced = rng.choice(known_neg_idx, size=take, replace=False)
        rest = need - take
    else:
        forced = np.empty(0, dtype=int)
        rest = need

    if rest > u_idx.size:
        raise ValueError("not enough unlabeled rows for the requested U:P ratio")

    for bag in range(n_bags):
        sampled = rng.choice(u_idx, size=rest, replace=False) if rest else np.empty(0, int)
        neg_idx = np.concatenate([forced, sampled])
        train_idx = np.concatenate([pos_idx, neg_idx])
        labels = np.concatenate([np.ones(n_pos, dtype=np.float32),
                                 np.zeros(neg_idx.size, dtype=np.float32)])
        dtrain = xgb.DMatrix(X[train_idx], label=labels)
        model = xgb.train(xgb_params(seed + bag), dtrain,
                          num_boost_round=NUM_BOOST_ROUND, verbose_eval=False)
        total += model.predict(dall)
    return total / n_bags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags", type=int, default=N_BAGS)
    parser.add_argument("--fold-limit", type=int, default=0,
                        help="debug: run only the first N folds")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = ce.load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = ce.label_array(stars)
    X = ce.prepare_spectral_features(data["X_clean"], standardize=True)
    n_total = X.shape[0]
    all_rows = np.arange(n_total)
    old_pos = np.where(y == 1)[0]
    u_all = np.setdiff1d(all_rows, old_pos)
    print(f"[data] n={n_total:,}  frozen positives={old_pos.size}  "
          f"unlabeled pool={u_all.size:,}", flush=True)

    lab = pd.read_csv(LABEL_CSV)
    lab["uid"] = lab["uid"].astype(str)
    uid_to_row = {u: i for i, u in enumerate(stars["uid"].astype(str))}
    lab["row"] = lab["uid"].map(uid_to_row).astype(int)
    lab = lab.loc[~lab["row"].isin(set(old_pos.tolist()))].reset_index(drop=True)
    new_pos = lab.loc[lab["label"] == 1, "row"].to_numpy()
    new_neg = lab.loc[lab["label"] == 0, "row"].to_numpy()
    print(f"[labels] new positives={new_pos.size}  new negatives={new_neg.size}",
          flush=True)

    labeled = np.concatenate([new_pos, new_neg])
    lab_y = np.concatenate([np.ones(new_pos.size), np.zeros(new_neg.size)])

    # rows available as *plain* unlabeled: never a labelled candidate
    u_plain = np.setdiff1d(u_all, labeled)
    print(f"[pool] unlabeled excluding all 99 labelled candidates: {u_plain.size:,}",
          flush=True)

    dall = xgb.DMatrix(X)
    print("[setup] built the global DMatrix", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(labeled, lab_y))
    if args.fold_limit:
        folds = folds[: args.fold_limit]

    rows: list[dict] = []
    pos_in_labeled = {int(r): i for i, r in enumerate(labeled)}
    # every labelled object gets exactly one out-of-fold score per config, so the
    # pooled AUC over all 99 carries more information than the per-fold ones
    oof = {name: np.full(labeled.size, np.nan) for name in CONFIGS}
    for fi, (_, ho_pos_in_fold) in enumerate(folds):
        ho = labeled[ho_pos_in_fold]
        hi = np.setdiff1d(labeled, ho)
        hi_pos = np.intersect1d(hi, new_pos)
        hi_neg = np.intersect1d(hi, new_neg)
        ho_pos = np.intersect1d(ho, new_pos)
        ho_neg = np.intersect1d(ho, new_neg)
        ho_lab = np.concatenate([np.ones(ho_pos.size), np.zeros(ho_neg.size)])
        ho_rows = np.concatenate([ho_pos, ho_neg])
        print(f"[fold {fi + 1}/{len(folds)}] HO={ho.size} "
              f"(pos={ho_pos.size}, neg={ho_neg.size})  HI={hi.size} "
              f"(pos={hi_pos.size}, neg={hi_neg.size})", flush=True)

        rng = np.random.default_rng(SEED + 100 + fi)
        control_rows = rng.choice(u_plain, size=hi_pos.size, replace=False)

        variants = {
            "M0_baseline": (old_pos, u_all, np.empty(0, int)),
            "M1_add_pos": (np.concatenate([old_pos, hi_pos]),
                           np.setdiff1d(u_all, hi_pos), np.empty(0, int)),
            "M2_pos_plus_neg": (np.concatenate([old_pos, hi_pos]),
                                np.setdiff1d(u_all, np.concatenate([hi_pos, hi_neg])),
                                hi_neg),
            "M4_new_pos_only": (hi_pos, np.setdiff1d(u_all, hi_pos), np.empty(0, int)),
            "M0_control": (np.concatenate([old_pos, control_rows]),
                           np.setdiff1d(u_all, control_rows), np.empty(0, int)),
        }

        for name in CONFIGS:
            pos_idx, u_idx, kn = variants[name]
            t0 = time.perf_counter()
            score = pu_bagging(X, dall, pos_idx, u_idx, kn,
                               n_bags=args.bags, seed=SEED + fi * 1000)
            for r in ho:
                oof[name][pos_in_labeled[int(r)]] = score[r]
            auc = float(roc_auc_score(ho_lab, score[ho_rows]))
            row = {
                "fold": fi + 1, "config": name,
                "n_train_pos": int(pos_idx.size),
                "n_unlabeled_pool": int(u_idx.size),
                "n_known_neg": int(kn.size),
                "auc_ho": auc,
                "ho_pos_score_median": float(np.median(score[ho_pos])),
                "ho_neg_score_median": float(np.median(score[ho_neg])),
                "seconds": float(time.perf_counter() - t0),
            }
            # ranking metrics against the config's own unlabeled pool
            ranked = u_idx[np.argsort(-score[u_idx])]
            for K in TOPK:
                top = ranked[:K]
                hit_pos = int(np.isin(top, ho_pos).sum())
                hit_neg = int(np.isin(top, ho_neg).sum())
                row[f"recall_at_{K}"] = hit_pos / max(ho_pos.size, 1)
                row[f"n_ho_pos_in_top_{K}"] = hit_pos
                row[f"n_ho_neg_in_top_{K}"] = hit_neg
                row[f"precision_HO_at_{K}"] = (
                    hit_pos / (hit_pos + hit_neg) if (hit_pos + hit_neg) else float("nan")
                )
            rows.append(row)
            print(f"    {name:<16} AUC={auc:.4f}  "
                  f"recall@500={row['recall_at_500']:.2f} "
                  f"recall@2500={row['recall_at_2500']:.2f}  "
                  f"({row['seconds']:.0f}s)", flush=True)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(RESULTS_DIR / "addsample_fold_metrics.csv", index=False)

    # ------------------------------------------------------------- summary
    def bootstrap_ci(y_true: np.ndarray, s: np.ndarray, n_boot: int = 4000,
                     seed: int = 7) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        n = y_true.size
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y_true[idx].min() == y_true[idx].max():
                continue
            vals.append(roc_auc_score(y_true[idx], s[idx]))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    # identical mask for every config because they share the fold assignment
    valid = ~np.isnan(oof[CONFIGS[0]])
    if not valid.any():
        raise RuntimeError("no out-of-fold scores were produced")
    if not valid.all() and not args.fold_limit:
        raise RuntimeError("some labelled objects have no out-of-fold score")
    yv = lab_y[valid]
    print(f"[oof] {int(valid.sum())}/{lab_y.size} labelled objects have "
          f"out-of-fold scores", flush=True)

    summary: dict[str, dict] = {}
    for name in CONFIGS:
        sub = fold_df.loc[fold_df["config"] == name]
        s = oof[name][valid]
        pooled = float(roc_auc_score(yv, s))
        lo, hi = bootstrap_ci(yv, s)
        summary[name] = {
            "n_train_pos_median": float(sub["n_train_pos"].median()),
            "auc_pooled_oof": pooled,
            "auc_pooled_ci95": [lo, hi],
            "auc_per_fold_mean": float(sub["auc_ho"].mean()),
            "auc_per_fold_std": float(sub["auc_ho"].std(ddof=1)),
            "auc_ho_per_fold": sub["auc_ho"].tolist(),
        }
        for K in TOPK:
            summary[name][f"recall_at_{K}_mean"] = float(sub[f"recall_at_{K}"].mean())
            summary[name][f"recall_at_{K}_std"] = float(sub[f"recall_at_{K}"].std(ddof=1))
            summary[name][f"precision_HO_at_{K}_mean"] = float(
                sub[f"precision_HO_at_{K}"].mean()
            )
        print(f"[summary] {name:<16} pooled AUC={pooled:.4f} "
              f"[{lo:.3f}, {hi:.3f}]  per-fold {sub['auc_ho'].mean():.4f} "
              f"± {sub['auc_ho'].std(ddof=1):.4f}", flush=True)

    base = summary["M0_baseline"]

    def bootstrap_diff(y_true: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                       n_boot: int = 4000, seed: int = 11):
        """Paired bootstrap of the AUC difference (same resampled objects)."""
        rng = np.random.default_rng(seed)
        n = y_true.size
        d = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y_true[idx].min() == y_true[idx].max():
                continue
            d.append(roc_auc_score(y_true[idx], s_a[idx])
                     - roc_auc_score(y_true[idx], s_b[idx]))
        return (float(np.mean(d)), float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)))

    deltas = {}
    for name in CONFIGS:
        if name == "M0_baseline":
            continue
        m, lo, hi = bootstrap_diff(yv, oof[name][valid], oof["M0_baseline"][valid])
        deltas[name] = {
            "mean_delta_auc_pooled": m,
            "delta_ci95": [lo, hi],
            "significant_gain_vs_M0": bool(lo > 0),
            "significant_loss_vs_M0": bool(hi < 0),
            "delta_recall_at_500": (summary[name]["recall_at_500_mean"]
                                    - base["recall_at_500_mean"]),
            "delta_precision_HO_at_500": (summary[name]["precision_HO_at_500_mean"]
                                          - base["precision_HO_at_500_mean"]),
        }
        print(f"[delta] {name:<16} AUC vs M0 = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if lo > 0 else ('loss' if hi < 0 else 'not significant')}",
              flush=True)

    # pre-registered decision rule
    verdict = {
        "M1_gain_significant": deltas["M1_add_pos"]["significant_gain_vs_M0"],
        "M2_gain_significant": deltas["M2_pos_plus_neg"]["significant_gain_vs_M0"],
        "M4_gain_significant": deltas["M4_new_pos_only"]["significant_gain_vs_M0"],
        "control_gain_significant": deltas["M0_control"]["significant_gain_vs_M0"],
        "rule": ("a gain counts only if the paired-bootstrap 95% CI of the AUC "
                 "difference excludes zero AND the random control shows no gain"),
    }
    if verdict["M1_gain_significant"] and not verdict["control_gain_significant"]:
        verdict["conclusion"] = "new positives improve the model"
    elif verdict["control_gain_significant"]:
        verdict["conclusion"] = ("inconclusive: the random-label control also "
                                 "improves, so the gain is a sample-size effect")
    else:
        verdict["conclusion"] = "no detectable improvement from adding positives"
    print(f"[verdict] {verdict['conclusion']}", flush=True)

    # -------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    order = list(CONFIGS)
    ax = axes[0]
    xs = np.arange(len(order))
    pooled = np.array([summary[c]["auc_pooled_oof"] for c in order])
    lo = np.array([summary[c]["auc_pooled_ci95"][0] for c in order])
    hi = np.array([summary[c]["auc_pooled_ci95"][1] for c in order])
    ax.bar(xs, pooled, yerr=[pooled - lo, hi - pooled], capsize=5,
           color=[COLOURS[c] for c in order], alpha=0.88)
    ax.axhline(base["auc_pooled_oof"], color="#888780", ls="--", lw=1.2,
               label=f"M0 baseline = {base['auc_pooled_oof']:.3f}")
    for x, v in zip(xs, pooled):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color="#5F5E5A")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("pooled out-of-fold AUC on the 99 labelled objects")
    ax.set_title("does adding the new labels rank real CN stars higher?", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    for c in order:
        sub = fold_df.loc[fold_df["config"] == c]
        ax.plot(np.array(TOPK), [sub[f"recall_at_{K}"].mean() for K in TOPK],
                marker="o", color=COLOURS[c], lw=1.8, label=c)
    ax.set_xlabel("candidate budget K")
    ax.set_ylabel("recall of held-out confirmed positives")
    ax.set_title("how many held-out positives reach the candidate list", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig7_addsample_experiment.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    oof_df = pd.DataFrame({
        "uid": stars.loc[labeled, "uid"].astype(str).to_numpy(),
        "label": lab_y.astype(int),
        **{f"oof_score_{name}": oof[name] for name in CONFIGS},
    })
    oof_df.to_csv(RESULTS_DIR / "addsample_oof_scores.csv", index=False)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "experiment": "adding newly confirmed high-resolution labels to XGBoost-PU",
        "protocol": {
            "cross_fit": f"{N_FOLDS}-fold over the labelled set, every labelled "
                         "object scored out of fold",
            "n_bags": args.bags, "u_to_p_ratio": U_RATIO,
            "max_depth": MAX_DEPTH, "num_boost_round": NUM_BOOST_ROUND,
            "n_new_positives": int(new_pos.size),
            "n_new_negatives": int(new_neg.size),
            "n_frozen_positives": int(old_pos.size),
            "primary_metric": "AUC among held-out confirmed positives vs negatives",
        },
        "per_config": summary,
        "delta_vs_baseline": deltas,
        "verdict": verdict,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "addsample_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[out] {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
