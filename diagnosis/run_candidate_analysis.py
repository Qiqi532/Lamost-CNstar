"""Distribution and reliability analysis of the second-stage screening candidates.

Calibration idea
----------------
The 91 confirmed CN stars are ground truth.  Instead of asking "how many
candidates are real?" (unanswerable), we ask a calibrated question:

    Where do the candidates sit **relative to the known positives**
    in the same standardized 700-d feature space?

Concretely, for every object we count how many *known positives* appear among
its k nearest neighbours:

    known-positive enrichment  =  observed count / random expectation
    random expectation         =  k * 91 / 41243

This statistic is directly comparable between the two groups, because

  * a known positive's neighbour list contains *other* known positives
    (its own slot is masked out), and
  * a candidate's neighbour list contains known positives
    (candidates are not positives, so there is no self-contamination).

If the candidates' enrichment distribution matches the positives', they occupy
the same place in feature space.  If it matches the random expectation, they
are indistinguishable from ordinary unlabeled stars.

Under the simplifying assumption that a false positive behaves like a random
object, the mixing fraction gives a point estimate of the true-positive share:

    pi_hat = (mean_cand - mu0) / (mean_pos - mu0)

This is an **upper bound** on the real share, because a plausible false positive
(an unusual but non-CN star) is likely to be slightly closer to the positives
than a purely random object, which inflates the numerator.

Groups analysed (tier = 1.0 sigma from feature/results_crosscheck)
    both        index AND area agree       (the provisional "new samples")
    index_only  only the index method agrees
    area_only   only the area method agrees
    tier1       union of the above
    all         the full frozen 654-object threshold candidate set

Exports stay minimal: one table, one metrics JSON, two figures.
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
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "crossfit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import crossfit_engine as ce  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CROSSCHECK_CSV = PROJECT_ROOT / "feature" / "results_crosscheck" / "crosscheck_groups.csv"
FROZEN_ARRAYS = (PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
                 / "crossfit_arrays_recall90.npz")
TIER = "1p0"
K_LIST = (50, 200)
TARGET_RECALL = 0.90

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_U = "#B4B2A9"
C_ALL = "#888780"
C_TIER1 = "#BA7517"
C_BOTH = "#1D9E75"
C_POS = "#E24B4A"


def load_all():
    data = ce.load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = ce.label_array(stars)
    X = ce.prepare_spectral_features(data["X_clean"], standardize=True)

    blob = np.load(FROZEN_ARRAYS, allow_pickle=True)
    if not np.array_equal(blob["uid"].astype(str),
                          stars["uid"].astype(str).to_numpy()):
        raise RuntimeError("frozen score archive is not aligned with the cache")
    score = np.asarray(blob["score_mean"], dtype=np.float64)
    threshold, required, achieved = ce.exact_recall_threshold(
        score[y == 1], TARGET_RECALL
    )
    print(f"[data] stars={len(stars):,}  known={int(y.sum())}  "
          f"threshold={threshold:.6f}  recall={achieved:.4f}", flush=True)
    return stars, X, y, score, threshold


def load_groups(stars: pd.DataFrame) -> dict[str, np.ndarray]:
    groups = pd.read_csv(CROSSCHECK_CSV)
    col = f"group_{TIER}"
    if col not in groups.columns:
        raise RuntimeError(f"missing column {col}")
    pos_map = {u: i for i, u in enumerate(stars["uid"].astype(str))}
    out: dict[str, np.ndarray] = {}
    # NOTE: the frozen export labels these groups with hyphens, not underscores
    for name, label in {
        "both": "both",
        "index_only": "index-only",
        "area_only": "area-only",
        "neither": "neither",
    }.items():
        idx = np.array([pos_map[u] for u in groups.loc[groups[col] == label, "uid"].astype(str)],
                       dtype=int)
        out[name] = np.sort(idx)
        print(f"[group] {name} ({label}): {idx.size}", flush=True)
    out["tier1"] = np.sort(np.concatenate(
        [out["both"], out["index_only"], out["area_only"]]
    ))
    out["all"] = np.array([pos_map[u] for u in groups["uid"].astype(str)], dtype=int)
    out["all"] = np.sort(out["all"])
    print(f"[group] tier1 union: {out['tier1'].size}  "
          f"all candidates: {out['all'].size}", flush=True)
    return out

def positive_enrichment(X, y, query_idx, kmax):
    """Count known positives among the k nearest neighbours of each query."""
    pos_idx = np.where(y == 1)[0]
    is_pos = np.zeros(X.shape[0], dtype=bool)
    is_pos[pos_idx] = True

    D = pairwise_distances(X[query_idx], X)
    # mask self / same-row duplicates so an object can never be its own neighbour
    row = np.arange(query_idx.size)
    D[row, query_idx] = np.inf
    knn = np.argsort(D, axis=1)[:, :kmax]
    del D

    return {k: is_pos[knn[:, :k]].sum(axis=1).astype(int) for k in K_LIST}


def nearest_positive_distance(X, y, query_idx):
    pos_idx = np.where(y == 1)[0]
    return pairwise_distances(X[query_idx], X[pos_idx]).min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=60,
                        help="number of top-ranked candidates to print")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    stars, X, y, score, threshold = load_all()
    groups = load_groups(stars)
    n_total = X.shape[0]
    pos_idx = np.where(y == 1)[0]
    n_pos = pos_idx.size
    kmax = max(K_LIST)

    # ---------------- calibration: enrichment of the known positives ----------
    pos_counts = positive_enrichment(X, y, pos_idx, kmax)
    d_pos_np_masked = pairwise_distances(X[pos_idx], X[pos_idx])
    np.fill_diagonal(d_pos_np_masked, np.inf)
    d_pos_np = d_pos_np_masked.min(axis=1)

    # ---------------- enrichment of each candidate group ---------------------
    cand_metrics: dict[str, dict] = {}
    enrichment_by_group: dict[str, dict[int, np.ndarray]] = {}
    dnp_by_group: dict[str, np.ndarray] = {}
    for name in ("both", "index_only", "area_only", "neither", "tier1", "all"):
        idx = groups[name]
        counts = positive_enrichment(X, y, idx, kmax)
        enrichment_by_group[name] = counts
        dnp_by_group[name] = nearest_positive_distance(X, y, idx)
        cand_metrics[name] = {"n": int(idx.size)}

    # ---------------- random expectation and per-k summary -------------------
    mu0 = {k: float(k * n_pos / n_total) for k in K_LIST}
    mean_pos = {k: float(pos_counts[k].mean()) for k in K_LIST}

    summary: dict[str, dict] = {}
    for k in K_LIST:
        entry = {
            "random_expectation": mu0[k],
            "known_positives_mean": mean_pos[k],
            "known_positives_median": float(np.median(pos_counts[k])),
        }
        # false-positive proxy: candidates the model liked but both spectral
        # methods rejected.  Using them instead of a purely random object gives
        # a lower bound on the true-positive share.
        e_fp = float(enrichment_by_group["neither"][k].mean())
        for name in ("neither", "both", "index_only", "area_only", "tier1", "all"):
            c = enrichment_by_group[name][k]
            m = float(c.mean())
            entry[name] = {
                "n": int(c.size),
                "mean": m,
                "median": float(np.median(c)),
                "share_zero": float(np.mean(c == 0)),
                "share_ge_1": float(np.mean(c >= 1)),
                "enrichment_factor": float(m / mu0[k]) if mu0[k] else None,
                "pi_hat_upper": (
                    float((m - mu0[k]) / (mean_pos[k] - mu0[k]))
                    if mean_pos[k] > mu0[k] else None
                ),
                "pi_hat_lower": (
                    float((m - e_fp) / (mean_pos[k] - e_fp))
                    if mean_pos[k] > e_fp else None
                ),
            }
        summary[f"k{k}"] = entry
        b = entry["both"]
        pi_u, pi_l = b["pi_hat_upper"], b["pi_hat_lower"]
        print(f"[k={k}] random={mu0[k]:.3f}  positives={mean_pos[k]:.3f}  "
              f"neither={entry['neither']['mean']:.3f}  "
              f"both={b['mean']:.3f} (x{b['enrichment_factor']:.2f})  "
              f"pi_hat in [{pi_l:.3f}, {pi_u:.3f}]", flush=True)

    # Does two-method agreement actually buy higher reliability?
    for k in K_LIST:
        parts = []
        for name in ("both", "index_only", "area_only", "neither"):
            e = summary[f"k{k}"][name]
            parts.append(f"{name}(n={e['n']}): x{e['enrichment_factor']:.2f}")
        print(f"[agreement check k={k}] " + "  ".join(parts), flush=True)

    # ---------------- per-candidate reliability table ------------------------
    idx = groups["tier1"]
    group_of = {}
    for name in ("both", "index_only", "area_only"):
        for i in groups[name]:
            group_of[int(i)] = name

    table = pd.DataFrame({
        "uid": stars.loc[idx, "uid"].astype(str).to_numpy(),
        "group": [group_of[int(i)] for i in idx],
        "teff": stars.loc[idx, "teff"].to_numpy(),
        "logg": stars.loc[idx, "logg"].to_numpy(),
        "feh": stars.loc[idx, "feh"].to_numpy(),
        "snru": stars.loc[idx, "snru"].to_numpy(),
        "xgb_pu_score": score[idx],
        "above_threshold": score[idx] >= threshold,
        "knn50_known_positives": enrichment_by_group["tier1"][50],
        "knn200_known_positives": enrichment_by_group["tier1"][200],
        "d_nearest_positive": dnp_by_group["tier1"],
    })
    table["knn50_expected_random"] = mu0[50]
    table["knn50_enrichment"] = table["knn50_known_positives"] / mu0[50]
    # relative standing against the known-positive calibration distribution
    ref = np.sort(pos_counts[50])
    table["knn50_pctile_vs_positives"] = (
        np.searchsorted(ref, table["knn50_known_positives"].to_numpy(), "right")
        / ref.size
    )
    table = table.sort_values(
        ["knn50_known_positives", "xgb_pu_score"], ascending=False
    ).reset_index(drop=True)
    table.to_csv(RESULTS_DIR / "candidate_reliability.csv", index=False)

    # how many candidates reach the median known-positive enrichment?
    med_pos = float(np.median(pos_counts[50]))
    n_like_pos = int((table["knn50_known_positives"] >= max(med_pos, 1)).sum())
    print(f"[result] candidates with k=50 enrichment >= known-positive median "
          f"({med_pos:.0f}): {n_like_pos}/{len(table)}", flush=True)

    # ---------------- parameter-space comparison ----------------------------
    missed_idx = np.array([
        i for i in pos_idx if score[i] < threshold
    ], dtype=int)
    kept_idx = np.setdiff1d(pos_idx, missed_idx)
    param_sets = {
        "unlabeled": (np.where(y == 0)[0], C_U, 0.55, 1.0, "unlabeled U"),
        "all_candidates": (groups["all"], C_ALL, 0.9, 1.0, "all 654 candidates"),
        "tier1": (groups["tier1"], C_TIER1, 1.2, 1.0, "tier-1 union (171)"),
        "both": (groups["both"], C_BOTH, 1.8, 1.0, "both methods (135)"),
        "positives": (pos_idx, C_POS, 1.8, 1.0, "known CN (91)"),
    }

    param_stats = {}
    for name, (ids, _, _, _, _) in param_sets.items():
        sub = stars.loc[ids]
        param_stats[name] = {
            "n": int(len(ids)),
            "teff_median": float(sub["teff"].median()),
            "logg_median": float(sub["logg"].median()),
            "feh_median": float(sub["feh"].median()),
            "snru_median": float(sub["snru"].median()),
        }
    missed_stats = {
        "n": int(missed_idx.size),
        "teff_median": float(stars.loc[missed_idx, "teff"].median()),
        "feh_median": float(stars.loc[missed_idx, "feh"].median()),
        "snru_median": float(stars.loc[missed_idx, "snru"].median()),
        "share_teff_ge_5000": float((stars.loc[missed_idx, "teff"] >= 5000).mean()),
    }
    kept_stats = {
        "n": int(kept_idx.size),
        "teff_median": float(stars.loc[kept_idx, "teff"].median()),
        "share_teff_ge_5000": float((stars.loc[kept_idx, "teff"] >= 5000).mean()),
    }
    # do the candidates cover the missed-positive temperature regime?
    hot = stars.loc[groups["both"], "teff"] >= 5000
    coverage = {
        "missed_teff_median": missed_stats["teff_median"],
        "kept_teff_median": kept_stats["teff_median"],
        "share_both_candidates_teff_ge_5000": float(hot.mean()),
        "n_both_candidates_teff_ge_5000": int(hot.sum()),
    }
    print(f"[param] missed positives n={missed_idx.size}, "
          f"Teff median={missed_stats['teff_median']:.0f}, "
          f"share Teff>=5000={missed_stats['share_teff_ge_5000']:.2f}  |  "
          f"kept Teff median={kept_stats['teff_median']:.0f}  |  "
          f"both candidates Teff>=5000: {hot.mean() * 100:.1f}%", flush=True)

    # ---------------- figures ------------------------------------------------
    param_panels = [
        ("teff", "Teff (K)"), ("logg", "log g"),
        ("feh", "[Fe/H]"), ("snru", "SNR_u"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for ax, (col, xlabel) in zip(axes.ravel(), param_panels):
        u = stars.loc[np.where(y == 0)[0], col].to_numpy()
        lo, hi = np.percentile(u, [0.2, 99.8])
        bins = np.linspace(lo, hi, 45)
        ax.hist(u, bins=bins, density=True, color=C_U, alpha=0.55,
                label="unlabeled U", zorder=1)
        for name in ("all_candidates", "tier1", "both", "positives"):
            ids, colour, lw, alpha, label = param_sets[name]
            ax.hist(stars.loc[ids, col].to_numpy(), bins=bins, density=True,
                    histtype="step", color=colour, lw=lw, alpha=alpha,
                    label=label, zorder=3)
        m = stars.loc[missed_idx, col].to_numpy()
        ax.plot(m, np.full(m.size, ax.get_ylim()[1] * 0.02), "|", color="#993C1D",
                ms=11, mew=1.6, label=f"missed positives ({missed_idx.size})",
                zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Parameter distributions: candidates vs. the 91 confirmed positives",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_candidate_parameters.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    maxv = int(max(np.percentile(pos_counts[50], 99), table["knn50_known_positives"].max()))
    edges = np.arange(-0.5, maxv + 1.5, 1.0)
    ax.hist(pos_counts[50], bins=edges, density=True, color=C_POS, alpha=0.5,
            label=f"known positives (n={n_pos})")
    ax.hist(enrichment_by_group["both"][50], bins=edges, density=True,
            color=C_BOTH, alpha=0.55, label=f"both-method candidates "
                                            f"({groups['both'].size})")
    ax.axvline(mu0[50], color="#185FA5", ls="--", lw=1.4,
               label=f"random expectation = {mu0[50]:.3f}")
    ax.set_xlabel("known positives among the 50 nearest neighbours")
    ax.set_ylabel("density")
    ax.set_title("reliability calibration: candidates vs. ground truth", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.hist(d_pos_np, bins=40, density=True, color=C_POS, alpha=0.5,
            label="distance between known positives")
    ax.hist(dnp_by_group["both"], bins=40, density=True, color=C_BOTH, alpha=0.55,
            label="candidate -> nearest known positive")
    ax.set_xlabel("euclidean distance in standardized 700-d space")
    ax.set_ylabel("density")
    ax.set_title("distance structure", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_candidate_reliability.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- top list ----------------------------------------------
    top = table.head(args.top)
    print("\n[top candidates by known-positive enrichment]")
    for _, r in top.iterrows():
        print(f"  {r['uid']:>16}  {r['group']:<10}  k50pos={r['knn50_known_positives']:>3}  "
              f"score={r['xgb_pu_score']:.4f}  Teff={r['teff']:.0f}  "
              f"logg={r['logg']:.2f}  FeH={r['feh']:.2f}", flush=True)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "analysis": "second-stage candidate distribution vs. the 91 ground-truth positives",
        "threshold": threshold,
        "n_total": int(n_total),
        "n_known": int(n_pos),
        "group_sizes": {k: int(v.size) for k, v in groups.items()},
        "feature_space": {"n_features": int(X.shape[1]), "standardized": True},
        "calibration": {
            "statistic": "count of known positives among the k nearest neighbours",
            "random_expectation": mu0,
            "known_positive_mean": mean_pos,
            "known_positive_median": {k: float(np.median(pos_counts[k])) for k in K_LIST},
            "assumption": (
                "pi_hat assumes a false positive behaves like a random object; "
                "this makes pi_hat an upper bound on the true positive share"
            ),
        },
        "enrichment_summary": summary,
        "candidates_reaching_known_median": {
            "n": int(n_like_pos),
            "of": int(len(table)),
            "share": float(n_like_pos / len(table)),
            "k": 50,
            "threshold_count": med_pos,
        },
        "parameter_stats": param_stats,
        "missed_positives": missed_stats,
        "kept_positives": kept_stats,
        "temperature_coverage": coverage,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "candidate_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[out] {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
