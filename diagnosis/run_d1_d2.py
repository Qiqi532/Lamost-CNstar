"""D1 + D2 diagnosis for the LAMOST CN-star sample.

Purpose
-------
Answer one question with numbers: **would adding more confirmed positives
actually improve the classifier?**

D1 - OOF score spectrum
    Full out-of-fold score distribution of the 41k unlabeled objects plus the
    position of every one of the 91 confirmed positives inside it.  The known
    positives are the ground truth; their position tells us whether the model
    sees a weak signal, no signal, or an actively *anti*-correlated signal.

D2 - Neighbourhood support in the model feature space
    Distances and neighbour counts in the *same* standardized 700-d space the
    model uses.

    Global clustering:
        ratio_pp_pu = mean(d to nearest other positive) / mean(d to nearest U)
        Compared against a permutation null, because in a 700-d space every
        object has a close neighbour and the raw ratio sits near 1 regardless
        of class structure.  Only the comparison carries information.

    Per-star neighbourhood support (the metric that actually answers the
    question):
        For each known positive x, take its k nearest neighbours among all
        41,243 objects and count how many of them are provisional new samples.
        The permutation control draws the same number of objects at random from
        the *remaining* U (excluding the provisional set) and repeats the count,
        giving a per-star p-value.

        Note on a trap that was fixed here: because the provisional new samples
        are themselves a subset of U, "distance from a positive to the nearest
        new sample" is **always** >= "distance to the nearest U object".  Any
        support metric built on that ratio can never fall below 1, so
        "no positive is closer to a new sample than to U" would be a
        mathematical artefact, not a finding.  The kNN enrichment count has no
        such constraint and is used instead.

New-sample placeholder
----------------------
No high-resolution confirmation is available yet, so the provisional new
sample set is the intersection of the two second-stage spectral methods
(indices AND differential area) at the 1.0-sigma tier, i.e. ``group_1p0 ==
"both"`` from ``feature/results_crosscheck/crosscheck_groups.csv``.  These
objects are model candidates, never used as training labels -- only as a
placeholder to estimate where future real confirmations are likely to land.

Exports are deliberately minimal: one diagnosis table, one metrics JSON, two
figures.  The 41k-row all-score table stays in memory and is never written.
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
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "crossfit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import crossfit_engine as ce  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CROSSCHECK_CSV = PROJECT_ROOT / "feature" / "results_crosscheck" / "crosscheck_groups.csv"
FROZEN_ARRAYS = (PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
                 / "crossfit_arrays_recall90.npz")
NEW_SAMPLE_TIER = "1p0"
NEW_SAMPLE_GROUP = "both"
TARGET_RECALL = 0.90
K_LIST = (50, 200)
N_PERM = 200
P_THRESHOLD = 0.05

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_U = "#B4B2A9"
COLOR_POS = "#E24B4A"
COLOR_NEW = "#1D9E75"
COLOR_HARD = "#BA7517"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = ce.load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = ce.label_array(stars)
    X = ce.prepare_spectral_features(data["X_clean"], standardize=True)

    required = {"uid", "label", "teff", "logg", "feh", "snru"}
    missing = required - set(stars.columns)
    if missing:
        raise RuntimeError(f"stars_clean is missing columns: {sorted(missing)}")
    if X.shape[0] != len(stars):
        raise RuntimeError("feature matrix and star table disagree on row count")
    print(f"[data] stars={len(stars):,}  features={X.shape[1]}  "
          f"known={int(y.sum())}  unlabeled={int((y == 0).sum()):,}", flush=True)
    return stars, X, y


def load_new_samples(stars: pd.DataFrame) -> np.ndarray:
    if not CROSSCHECK_CSV.exists():
        raise FileNotFoundError(CROSSCHECK_CSV)
    groups = pd.read_csv(CROSSCHECK_CSV)
    col = f"group_{NEW_SAMPLE_TIER}"
    if col not in groups.columns:
        raise RuntimeError(f"{CROSSCHECK_CSV.name} has no column {col}")
    picked = groups.loc[groups[col] == NEW_SAMPLE_GROUP]
    uid_set = set(picked["uid"].astype(str))
    idx = np.where(stars["uid"].astype(str).isin(uid_set).to_numpy())[0]
    print(f"[new] tier={NEW_SAMPLE_TIER} group={NEW_SAMPLE_GROUP}  "
          f"uids_in_csv={len(uid_set)}  matched_in_cache={idx.size}", flush=True)
    if idx.size != len(uid_set):
        raise RuntimeError("some provisional new-sample UIDs are not in the cache")
    return idx


# --------------------------------------------------------------------------- #
# D1 - score spectrum
# --------------------------------------------------------------------------- #
def run_d1(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pos = y == 1
    u_scores = np.sort(score[~pos])
    pos_scores = score[pos]
    pct = np.searchsorted(u_scores, pos_scores, side="right") / u_scores.size * 100.0
    above = pos_scores >= threshold
    edges = [0, 25, 50, 75, 90, 95, 100]
    bands = {
        f"{edges[i]}-{edges[i + 1]}": int(np.sum((pct > edges[i]) & (pct <= edges[i + 1])))
        for i in range(len(edges) - 1)
    }
    metrics = {
        "u_score_quantiles": {
            str(q): float(np.quantile(u_scores, q / 100.0))
            for q in (50, 90, 95, 99, 99.9)
        },
        "u_score_max": float(u_scores.max()),
        "known_score_min": float(pos_scores.min()),
        "known_score_median": float(np.median(pos_scores)),
        "known_score_max": float(pos_scores.max()),
        "threshold_recall90": float(threshold),
        "known_above_threshold": int(above.sum()),
        "known_below_threshold": int((~above).sum()),
        "pctile_in_u_percentile_bands": bands,
        "known_at_or_below_u_median": int(np.sum(pct <= 50.0)),
    }
    print(f"[D1] U median={metrics['u_score_quantiles']['50']:.4f}  "
          f"threshold={threshold:.4f}  known above={above.sum()}/91  "
          f"at-or-below U median={metrics['known_at_or_below_u_median']}/91",
          flush=True)
    return {"percentile_in_u": pct, "metrics": metrics}


# --------------------------------------------------------------------------- #
# D2 - neighbourhood structure
# --------------------------------------------------------------------------- #
def run_d2(X: np.ndarray, y: np.ndarray, new_idx: np.ndarray) -> dict:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_total = X.shape[0]
    Xp, Xn, Xnew = X[pos_idx], X[neg_idx], X[new_idx]

    d_pp_all = pairwise_distances(Xp, Xp)
    np.fill_diagonal(d_pp_all, np.inf)
    d_pp_nn1 = d_pp_all.min(axis=1)
    d_pu_nn1 = pairwise_distances(Xp, Xn).min(axis=1)
    d_pn_nn1 = pairwise_distances(Xp, Xnew).min(axis=1)
    ratio_pp_pu = d_pp_nn1 / d_pu_nn1

    # nearest neighbours of every positive among all objects (self excluded)
    D_all = pairwise_distances(Xp, X)
    D_all[np.arange(pos_idx.size), pos_idx] = np.inf
    kmax = max(K_LIST)
    knn_idx = np.argsort(D_all, axis=1)[:, :kmax]
    del D_all

    is_new = np.zeros(n_total, dtype=bool)
    is_new[new_idx] = True
    enrichment = {
        k: {
            "counts": is_new[knn_idx[:, :k]].sum(axis=1).astype(int),
            "expected": float(k * new_idx.size / n_total),
        }
        for k in K_LIST
    }

    r = float(ratio_pp_pu.mean())
    verdict = ("manifold_like" if r < 0.7 else
               "partially_clustered" if r <= 1.0 else
               "weak_or_scattered")
    metrics = {
        "n_known": int(pos_idx.size),
        "n_new_provisional": int(new_idx.size),
        "n_unlabeled": int(neg_idx.size),
        "d_pp_nn1_mean": float(d_pp_nn1.mean()),
        "d_pp_nn1_median": float(np.median(d_pp_nn1)),
        "d_pu_nn1_mean": float(d_pu_nn1.mean()),
        "d_pn_nn1_mean": float(d_pn_nn1.mean()),
        "ratio_pp_pu_mean": r,
        "ratio_pp_pu_median": float(np.median(ratio_pp_pu)),
        "ratio_pp_pu_verdict": verdict,
        "knn_enrichment": {
            f"k{k}": {
                "expected_random_count": enrichment[k]["expected"],
                "mean_count": float(enrichment[k]["counts"].mean()),
                "n_positives_with_any_new_neighbour": int(
                    np.sum(enrichment[k]["counts"] > 0)
                ),
                "max_count": int(enrichment[k]["counts"].max()),
            }
            for k in K_LIST
        },
    }
    print(f"[D2] d_pp_nn1={metrics['d_pp_nn1_mean']:.3f}  "
          f"d_pu_nn1={metrics['d_pu_nn1_mean']:.3f}  ratio={r:.3f} -> {verdict}",
          flush=True)
    for k in K_LIST:
        e = metrics["knn_enrichment"][f"k{k}"]
        print(f"[D2] k={k}: expected {e['expected_random_count']:.3f} new "
              f"neighbours, observed mean {e['mean_count']:.3f}, "
              f"positives with >=1: "
              f"{e['n_positives_with_any_new_neighbour']}/91", flush=True)

    return {
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "d_pp_nn1": d_pp_nn1,
        "d_pu_nn1": d_pu_nn1,
        "d_pn_nn1": d_pn_nn1,
        "ratio_pp_pu": ratio_pp_pu,
        "knn_idx": knn_idx,
        "enrichment": enrichment,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# permutation controls
# --------------------------------------------------------------------------- #
def null_controls(X: np.ndarray, y: np.ndarray, new_idx: np.ndarray,
                  d2: dict, n_perm: int = N_PERM, seed: int = 42) -> dict:
    """Permutation controls for both the global ratio and the kNN enrichment.

    The control draws the same number of objects as the provisional set, but
    from the *remaining* unlabeled pool so that the real and null sets never
    overlap.
    """

    rng = np.random.default_rng(seed)
    pos_idx = d2["pos_idx"]
    neg_idx = d2["neg_idx"]
    n_total = X.shape[0]
    n_pos, n_new = pos_idx.size, new_idx.size
    knn_idx = d2["knn_idx"]

    pool = np.setdiff1d(neg_idx, new_idx, assume_unique=False)
    Xn = np.ascontiguousarray(X[neg_idx], dtype=np.float32)
    Xp = np.ascontiguousarray(X[pos_idx], dtype=np.float32)
    Xn_sq = (Xn.astype(np.float64) ** 2).sum(axis=1)
    Xn_T = Xn.T

    def nn_to_u(Xq: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(Xq, dtype=np.float32)
        q_sq = (q.astype(np.float64) ** 2).sum(axis=1)[:, None]
        d2m = q_sq + Xn_sq[None, :] - 2.0 * (q @ Xn_T).astype(np.float64)
        np.maximum(d2m, 0.0, out=d2m)
        return np.sqrt(d2m).min(axis=1)

    # --- ratio null ---
    null_ratio = np.empty(n_perm)
    for t in range(n_perm):
        sel = rng.choice(neg_idx, size=n_pos, replace=False)
        sel_pos = np.searchsorted(neg_idx, sel)  # global index -> U column
        Xs = np.ascontiguousarray(X[sel], dtype=np.float32)
        Ds = pairwise_distances(Xs, Xs)
        np.fill_diagonal(Ds, np.inf)
        d2s = (Xs.astype(np.float64) ** 2).sum(axis=1)[:, None] \
            + Xn_sq[None, :] - 2.0 * (Xs @ Xn_T).astype(np.float64)
        np.maximum(d2s, 0.0, out=d2s)
        d2s[:, sel_pos] = np.inf
        null_ratio[t] = float((Ds.min(axis=1) / np.sqrt(d2s).min(axis=1)).mean())

    # --- enrichment null, reusing the precomputed neighbour index ---
    null_counts = {k: np.empty((n_pos, n_perm), dtype=np.int32) for k in K_LIST}
    for t in range(n_perm):
        sel = rng.choice(pool, size=n_new, replace=False)
        is_sel = np.zeros(n_total, dtype=bool)
        is_sel[sel] = True
        for k in K_LIST:
            null_counts[k][:, t] = is_sel[knn_idx[:, :k]].sum(axis=1)

    enrichment_stats = {}
    for k in K_LIST:
        real = d2["enrichment"][k]["counts"]
        nc = null_counts[k]
        pvals = (nc >= real[:, None]).mean(axis=1)
        expected = k * n_new / n_total
        enrichment_stats[f"k{k}"] = {
            "null_mean_count": float(nc.mean()),
            "null_p95_count": float(np.quantile(nc, 0.95)),
            "real_mean_count": float(real.mean()),
            "mean_enrichment_factor": float(real.mean() / expected) if expected else None,
            "n_positives_significantly_enriched": int(np.sum(pvals < P_THRESHOLD)),
            "share_positives_significantly_enriched": float(np.mean(pvals < P_THRESHOLD)),
            "pvals": pvals,
        }
        print(f"[null] k={k}: null mean count {nc.mean():.3f}, real "
              f"{real.mean():.3f}, enrichment x{real.mean() / expected:.2f}, "
              f"significant positives "
              f"{int(np.sum(pvals < P_THRESHOLD))}/91", flush=True)

    return {
        "n_permutations": int(n_perm),
        "n_null_pool": int(pool.size),
        "null_ratio_mean": float(null_ratio.mean()),
        "null_ratio_std": float(null_ratio.std(ddof=1)),
        "null_ratio_p05": float(np.quantile(null_ratio, 0.05)),
        "enrichment": enrichment_stats,
        "_null_ratio": null_ratio,
    }


# --------------------------------------------------------------------------- #
# D3 - per-star diagnosis table
# --------------------------------------------------------------------------- #
def build_diagnosis_table(stars, y, score, score_pct, d2, threshold, nulls) -> pd.DataFrame:
    pos_idx = d2["pos_idx"]
    pval_col = {}
    for k in K_LIST:
        pval_col[k] = nulls["enrichment"][f"k{k}"]["pvals"]

    table = pd.DataFrame(
        {
            "uid": stars.loc[pos_idx, "uid"].astype(str).to_numpy(),
            "teff": stars.loc[pos_idx, "teff"].to_numpy(),
            "logg": stars.loc[pos_idx, "logg"].to_numpy(),
            "feh": stars.loc[pos_idx, "feh"].to_numpy(),
            "snru": stars.loc[pos_idx, "snru"].to_numpy(),
            "xgb_pu_score": score[pos_idx],
            "score_pctile_in_u": score_pct,
            "d_pp_nn1": d2["d_pp_nn1"],
            "d_pu_nn1": d2["d_pu_nn1"],
            "d_pn_nn1": d2["d_pn_nn1"],
            "ratio_pp_pu": d2["ratio_pp_pu"],
            "above_threshold": score[pos_idx] >= threshold,
        }
    )
    for k in K_LIST:
        table[f"knn{k}_new_neighbours"] = d2["enrichment"][k]["counts"]
        table[f"knn{k}_expected_random"] = d2["enrichment"][k]["expected"]
        table[f"knn{k}_pvalue"] = pval_col[k]

    def tier(row: pd.Series) -> str:
        if row["above_threshold"]:
            return "easy"
        return "medium" if row["score_pctile_in_u"] >= 50.0 else "hard"

    def diagnosis(row: pd.Series) -> str:
        missed = not row["above_threshold"]
        targeted = row["knn50_pvalue"] < P_THRESHOLD
        if missed and targeted:
            return "missed_and_targeted"
        if missed:
            return "missed_not_targeted"
        return "kept_and_targeted" if targeted else "kept"

    def action(row: pd.Series) -> str:
        return {
            "missed_and_targeted": "priority_neighbourhood_observation",
            "missed_not_targeted": "feature_engineering_not_observation",
            "kept_and_targeted": "low_priority",
            "kept": "low_priority",
        }[row["diagnosis"]]

    table["difficulty_tier"] = table.apply(tier, axis=1)
    table["diagnosis"] = table.apply(diagnosis, axis=1)
    table["recommended_action"] = table.apply(action, axis=1)
    return table.sort_values("xgb_pu_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_score_spectrum(y, score, threshold) -> Path:
    pos = y == 1
    u_scores, pos_scores = score[~pos], score[pos]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[1.35, 1],
                             sharex=True, gridspec_kw={"hspace": 0.12})

    ax = axes[0]
    ax.hist(u_scores, bins=140, color=COLOR_U, alpha=0.85,
            label=f"unlabeled U (n={u_scores.size:,})")
    ax.set_yscale("log")
    ax.axvline(threshold, color="#185FA5", ls="--", lw=1.4,
               label=f"90% recall threshold = {threshold:.4f}")
    ax.hist(pos_scores, bins=140, color=COLOR_POS, alpha=0.9,
            label=f"known CN positives (n={pos_scores.size})")
    ax.set_ylabel("count (log)")
    ax.set_title("D1  OOF score spectrum: where the 91 confirmed positives sit "
                 "inside the unlabeled distribution", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[1]
    order = np.argsort(pos_scores)
    s = pos_scores[order]
    ax.scatter(s, np.arange(s.size), s=22, c=COLOR_POS, zorder=3, label="known CN")
    below = s < threshold
    ax.scatter(s[below], np.arange(s.size)[below], s=36, facecolors="none",
               edgecolors=COLOR_HARD, linewidths=1.2, zorder=4,
               label=f"below threshold (n={int(below.sum())})")
    ax.axvline(threshold, color="#185FA5", ls="--", lw=1.4)
    med = float(np.median(u_scores))
    ax.axvline(med, color="#5F5E5A", ls=":", lw=1.2, label=f"U median = {med:.3f}")
    ax.set_xlabel("xgb_pu_score (out-of-fold)")
    ax.set_ylabel("known CN, ranked")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    path = RESULTS_DIR / "fig1_score_spectrum.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_density_landscape(X, y, new_idx, d2, table) -> Path:
    pos_idx = d2["pos_idx"]
    neg_idx = d2["neg_idx"]
    rng = np.random.default_rng(42)
    sub = rng.choice(neg_idx, size=min(6000, neg_idx.size), replace=False)
    pca = PCA(n_components=2, random_state=42).fit(X[sub])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

    ax = axes[0]
    U2 = pca.transform(X[sub])
    ax.scatter(U2[:, 0], U2[:, 1], s=5, c=COLOR_U, alpha=0.35,
               label=f"unlabeled U (sample of {sub.size:,})")
    ax.scatter(*pca.transform(X[new_idx]).T, s=26, c=COLOR_NEW, alpha=0.9,
               marker="^", label=f"provisional new samples (n={new_idx.size})")
    ax.scatter(*pca.transform(X[pos_idx]).T, s=32, c=COLOR_POS, alpha=0.95,
               edgecolors="white", linewidths=0.4, label=f"known CN (n={pos_idx.size})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("D2  feature-space landscape (PCA on standardized 700-d)", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1]
    tier_colour = {"easy": COLOR_POS, "medium": "#888780", "hard": COLOR_HARD}
    cols = [tier_colour[t] for t in table["difficulty_tier"]]
    ax.scatter(table["score_pctile_in_u"], table["knn50_new_neighbours"],
               s=34, c=cols, alpha=0.9, edgecolors="white", linewidths=0.4)
    exp = float(table["knn50_expected_random"].iloc[0])
    ax.axhline(exp, color="#185FA5", ls="-.", lw=1.3,
               label=f"expected under random sampling = {exp:.3f}")
    ax.axvline(float(table.loc[table["above_threshold"], "score_pctile_in_u"].min()),
               color="#5F5E5A", ls=":", lw=1.2, label="recall-90 threshold")
    for tier, c in tier_colour.items():
        ax.scatter([], [], s=30, c=c, label=f"{tier} positive")
    ax.set_xlabel("score percentile inside U  (%)")
    ax.set_ylabel("new samples among its 50 nearest neighbours")
    ax.set_title("per-star neighbourhood support vs. model score", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    path = RESULTS_DIR / "fig2_density_landscape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-bags", type=int, default=100)
    parser.add_argument("--from-arrays", type=Path,
                        default=FROZEN_ARRAYS if FROZEN_ARRAYS.exists() else None,
                        help="load the frozen OOF scores from an existing "
                             "crossfit_arrays_*.npz instead of re-fitting")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    stars, X, y = load_inputs()
    new_idx = load_new_samples(stars)
    pos_idx = np.where(y == 1)[0]

    if args.from_arrays is not None:
        blob = np.load(args.from_arrays, allow_pickle=True)
        arr_uid = blob["uid"].astype(str)
        if not np.array_equal(arr_uid, stars["uid"].astype(str).to_numpy()):
            raise RuntimeError("npz uid order does not match the loaded cache")
        arr_label = blob["label"].astype(np.int8)
        if not np.array_equal(arr_label, stars["label"].to_numpy()):
            raise RuntimeError("npz label does not match the loaded cache")
        score = np.asarray(blob["score_mean"], dtype=np.float64)
        threshold, required, achieved = ce.exact_recall_threshold(
            score[pos_idx], TARGET_RECALL
        )
        print(f"[model] loaded frozen scores from {args.from_arrays.name}  "
              f"threshold={threshold:.6f}  required={required}  "
              f"recall={achieved:.4f}", flush=True)
    else:
        config = ce.CrossFitConfig(
            n_repeats=1 if args.smoke else args.n_repeats,
            n_bags=2 if args.smoke else args.n_bags,
            target_recall=TARGET_RECALL,
        )
        print(f"[model] {config.key}", flush=True)
        result = ce.run_crossfit_pu(X, y, config, progress=True)
        score = np.asarray(result["score_mean"])
        threshold = float(result["threshold"])
        print(f"[model] threshold={threshold:.6f}  recall="
              f"{result['achieved_known_recall']:.4f}  "
              f"elapsed={result['elapsed_seconds']:.0f}s", flush=True)

    if not args.smoke:
        reference = pd.read_csv(
            PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
            / "crossfit_known_cn_recall90.csv"
        )
        cur = pd.DataFrame({"uid": stars.loc[pos_idx, "uid"].astype(str),
                            "s": score[pos_idx]}).sort_values("uid")
        if np.allclose(reference.sort_values("uid")["xgb_pu_score"].to_numpy(),
                       cur["s"].to_numpy(), atol=1e-9):
            print("[check] scores match the frozen recall90 known-CN export "
                  "exactly", flush=True)
        else:
            print("[check] WARNING: scores differ from the frozen export",
                  flush=True)

    d1 = run_d1(y, score, threshold)
    d2 = run_d2(X, y, new_idx)
    nulls = null_controls(X, y, new_idx, d2, n_perm=30 if args.smoke else N_PERM)
    table = build_diagnosis_table(stars, y, score, d1["percentile_in_u"], d2,
                                  threshold, nulls)
    table.to_csv(RESULTS_DIR / "known_cn_diagnosis.csv", index=False)

    # significance of the global clustering ratio
    ratio_real = d2["metrics"]["ratio_pp_pu_mean"]
    z_ratio = (ratio_real - nulls["null_ratio_mean"]) / max(nulls["null_ratio_std"], 1e-12)
    p_ratio = float(np.mean(nulls["_null_ratio"] <= ratio_real))
    print(f"[sig] ratio real={ratio_real:.3f} vs null "
          f"{nulls['null_ratio_mean']:.3f}  z={z_ratio:+.2f}  p={p_ratio:.3f}",
          flush=True)

    # how many of the missed positives are targeted at all?
    missed = table.loc[~table["above_threshold"]]
    missed_stats = {
        "n_missed": int(len(missed)),
        "n_missed_targeted": int((missed["knn50_pvalue"] < P_THRESHOLD).sum()),
        "mean_knn50_count_missed": float(missed["knn50_new_neighbours"].mean()),
        "median_knn50_count_kept": float(
            table.loc[table["above_threshold"], "knn50_new_neighbours"].median()
        ),
    }
    print(f"[key] missed positives: {missed_stats['n_missed']}, "
          f"of which targeted by new samples: "
          f"{missed_stats['n_missed_targeted']}", flush=True)

    tier_counts = table["difficulty_tier"].value_counts().to_dict()
    diag_counts = table["diagnosis"].value_counts().to_dict()
    tier_enrichment = (
        table.groupby("difficulty_tier")["knn50_new_neighbours"]
        .agg(["mean", "median", "count"]).round(4).to_dict(orient="index")
    )

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "diagnosis_scope": "D1 score spectrum + D2 neighbourhood support",
        "threshold": threshold,
        "roc_auc": float(ce.roc_auc_score(y, score)),
        "pr_auc": float(ce.average_precision_score(y, score)),
        "feature_space": {
            "n_features": int(X.shape[1]),
            "standardized": True,
            "distance": "euclidean in standardized 700-d space",
        },
        "new_sample_proxy": {
            "source": str(CROSSCHECK_CSV.relative_to(PROJECT_ROOT)),
            "tier": NEW_SAMPLE_TIER,
            "group": NEW_SAMPLE_GROUP,
            "n": int(new_idx.size),
            "share_already_above_threshold": float(np.mean(score[new_idx] >= threshold)),
            "caveat": (
                "provisional placeholder: model candidates, not high-resolution "
                "confirmations; used only to estimate neighbourhood support, "
                "never as training labels"
            ),
        },
        "d1": d1["metrics"],
        "d2": d2["metrics"],
        "null_controls": {
            "n_permutations": nulls["n_permutations"],
            "n_null_pool": nulls["n_null_pool"],
            "null_ratio_mean": nulls["null_ratio_mean"],
            "null_ratio_std": nulls["null_ratio_std"],
            "null_ratio_p05": nulls["null_ratio_p05"],
            "enrichment": {
                k: {kk: vv for kk, vv in v.items() if kk != "pvals"}
                for k, v in nulls["enrichment"].items()
            },
        },
        "significance": {
            "ratio_real": float(ratio_real),
            "ratio_z_vs_null": float(z_ratio),
            "ratio_p_value_lower_tail": p_ratio,
            "positive_class_more_clustered_than_random": bool(p_ratio < 0.05),
        },
        "missed_positives": missed_stats,
        "difficulty_tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "diagnosis_counts": {k: int(v) for k, v in diag_counts.items()},
        "tier_enrichment_k50": tier_enrichment,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "density_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    figure_score_spectrum(y, score, threshold)
    figure_density_landscape(X, y, new_idx, d2, table)

    print(f"[D3] tiers={tier_counts}", flush=True)
    print(f"[D3] diagnosis={diag_counts}", flush=True)
    print(f"[out] {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
