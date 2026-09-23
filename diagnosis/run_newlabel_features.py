"""Feature characterisation of the high-resolution labelled candidate set.

Input
-----
``diagnosis/cands_label.csv`` -- ra, dec, uid, label

    label = 1   high-resolution spectroscopy confirms CN enhancement
    label = 0   checked with high-resolution spectroscopy, NOT enhanced

Both classes matter.  The positives are the new ground truth we want to add to
the training set.  The negatives are **reliable negatives drawn from the model's
high-score region**, which is exactly where the classifier currently makes its
mistakes.

What this script reports
------------------------
1. Sanity checks: every uid present in the cache; overlap with the frozen 91
   positives; overlap with the 654 threshold candidates and the tier-1 groups.
2. Parameter comparison: new positives / new negatives / the 91 / unlabeled.
3. Feature-space position using the same calibrated statistic as before --
   the number of *frozen 91* positives among the k nearest neighbours.
4. **Validation of that statistic**: does it actually separate the new
   negatives from the new positives?  Reported as a ROC-AUC with a
   permutation-free analytic standard error, computed leave-nothing-out because
   the new labels were never used to build the 91.

Point 4 is the important one: if the enrichment statistic cannot order the new
positives above the new negatives, then it is not a usable reliability measure
and the earlier candidate ranking must not be trusted.
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
from sklearn.metrics import pairwise_distances, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "crossfit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import crossfit_engine as ce  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LABEL_CSV = HERE / "cands_label.csv"
CROSSCHECK_CSV = PROJECT_ROOT / "feature" / "results_crosscheck" / "crosscheck_groups.csv"
FROZEN_ARRAYS = (PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
                 / "crossfit_arrays_recall90.npz")
K_LIST = (20, 50, 200)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_U = "#B4B2A9"
C_POS91 = "#E24B4A"
C_NEWPOS = "#1D9E75"
C_NEWNEG = "#BA7517"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = ce.load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = ce.label_array(stars)
    X = ce.prepare_spectral_features(data["X_clean"], standardize=True)
    blob = np.load(FROZEN_ARRAYS, allow_pickle=True)
    score = np.asarray(blob["score_mean"], dtype=np.float64)
    threshold, _, _ = ce.exact_recall_threshold(score[y == 1], 0.90)

    n_total = X.shape[0]
    pos91 = np.where(y == 1)[0]
    uid_to_row = {u: i for i, u in enumerate(stars["uid"].astype(str))}

    # ------------------------------------------------------------------ input
    lab = pd.read_csv(LABEL_CSV)
    lab["uid"] = lab["uid"].astype(str)
    print(f"[input] rows={len(lab)}  label=1: {int((lab['label'] == 1).sum())}  "
          f"label=0: {int((lab['label'] == 0).sum())}", flush=True)

    missing = [u for u in lab["uid"] if u not in uid_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} labelled uids are not in the cache")

    lab["row"] = lab["uid"].map(uid_to_row)
    lab["is_known_positive"] = lab["row"].isin(set(pos91.tolist()))
    lab["xgb_pu_score"] = score[lab["row"].to_numpy()]
    lab["above_threshold"] = lab["xgb_pu_score"] >= threshold

    for c in ("teff", "logg", "feh", "snru", "masked_cluster_id"):
        if c in stars.columns:
            lab[c] = stars.loc[lab["row"].to_numpy(), c].to_numpy()

    n_overlap = int(lab["is_known_positive"].sum())
    print(f"[check] labelled uids already among the frozen 91 positives: {n_overlap}",
          flush=True)
    if n_overlap:
        print("        (these will be excluded from the 'new' set)", flush=True)

    cand_uid = set(pd.read_csv(CROSSCHECK_CSV)["uid"].astype(str))
    tier1 = pd.read_csv(CROSSCHECK_CSV)
    tier1_uid = set(tier1.loc[tier1["group_1p0"] != "neither", "uid"].astype(str))
    lab["in_654_candidates"] = lab["uid"].isin(cand_uid)
    lab["in_tier1"] = lab["uid"].isin(tier1_uid)
    print(f"[check] drawn from the frozen 654 threshold candidates: "
          f"{int(lab['in_654_candidates'].sum())}/{len(lab)}", flush=True)
    print(f"[check] of which tier-1 (either spectral method): "
          f"{int(lab['in_tier1'].sum())}", flush=True)

    # new = labelled objects that are not already in the frozen 91
    newpos = lab.loc[(lab["label"] == 1) & (~lab["is_known_positive"])].copy()
    newneg = lab.loc[(lab["label"] == 0) & (~lab["is_known_positive"])].copy()
    print(f"[new] positives={len(newpos)}  negatives={len(newneg)}", flush=True)

    newpos_rows = newpos["row"].to_numpy()
    newneg_rows = newneg["row"].to_numpy()

    # ------------------------------------------------- calibrated enrichment
    def enrich(query_rows: np.ndarray, reference_rows: np.ndarray) -> dict:
        """Count reference objects among the k nearest neighbours of each query."""
        ref_mask = np.zeros(n_total, dtype=bool)
        ref_mask[reference_rows] = True
        D = pairwise_distances(X[query_rows], X)
        D[np.arange(query_rows.size), query_rows] = np.inf  # never self
        knn = np.argsort(D, axis=1)[:, : max(K_LIST)]
        del D
        return {k: ref_mask[knn[:, :k]].sum(axis=1).astype(int) for k in K_LIST}

    pos_enr = enrich(pos91, pos91)          # leave-one-out against the 91
    npos_enr = enrich(newpos_rows, pos91)   # new positives vs the frozen 91
    nneg_enr = enrich(newneg_rows, pos91)   # new negatives vs the frozen 91
    mu0 = {k: float(k * pos91.size / n_total) for k in K_LIST}

    enrich_stats = {}
    for k in K_LIST:
        enrich_stats[f"k{k}"] = {
            "random_expectation": mu0[k],
            "known91_mean": float(pos_enr[k].mean()),
            "known91_median": float(np.median(pos_enr[k])),
            "newpos_mean": float(npos_enr[k].mean()),
            "newpos_median": float(np.median(npos_enr[k])),
            "newnneg_mean": float(nneg_enr[k].mean()),
            "newneg_median": float(np.median(nneg_enr[k])),
            "newpos_enrichment_factor": float(npos_enr[k].mean() / mu0[k]),
            "newneg_enrichment_factor": float(nneg_enr[k].mean() / mu0[k]),
        }
        print(f"[enrich k={k}] random={mu0[k]:.3f}  known91={pos_enr[k].mean():.2f}  "
              f"newpos={npos_enr[k].mean():.2f}  newneg={nneg_enr[k].mean():.2f}",
              flush=True)

    # --------------------------------------------- validation of the statistic
    validation = {}
    for k in K_LIST:
        s = np.concatenate([npos_enr[k], nneg_enr[k]]).astype(float)
        lab_y = np.concatenate([np.ones(len(newpos)), np.zeros(len(newneg))])
        auc = float(roc_auc_score(lab_y, s))
        # Hanley-McNeil standard error for a two-class AUC
        n1, n0 = int(lab_y.sum()), int((1 - lab_y).sum())
        se = float(np.sqrt(
            (auc * (1 - auc) + (n1 - 1) * (auc / (2 - auc) - auc ** 2)
             + (n0 - 1) * (2 * auc ** 2 / (1 + auc) - auc ** 2)) / (n1 * n0)
        )) if n1 and n0 else float("nan")
        validation[f"k{k}"] = {
            "auc_newpos_vs_newneg": auc,
            "auc_std_error": se,
            "n_pos": n1, "n_neg": n0,
        }
        print(f"[validate k={k}] AUC(new positives > new negatives) = {auc:.3f} "
              f"± {se:.3f}", flush=True)

    # also score-based, as a reference point
    lab_y = np.concatenate([np.ones(len(newpos)), np.zeros(len(newneg))])
    auc_score = float(roc_auc_score(
        lab_y,
        np.concatenate([newpos["xgb_pu_score"].to_numpy(),
                        newneg["xgb_pu_score"].to_numpy()]),
    ))
    print(f"[validate] AUC using xgb_pu_score only = {auc_score:.3f}", flush=True)

    # single-parameter and combined references
    auc_feh = float(roc_auc_score(
        lab_y, -np.concatenate([newpos["feh"].to_numpy(), newneg["feh"].to_numpy()])
    ))
    auc_logg = float(roc_auc_score(
        lab_y, -np.concatenate([newpos["logg"].to_numpy(), newneg["logg"].to_numpy()])
    ))
    auc_teff = float(roc_auc_score(
        lab_y, -np.concatenate([newpos["teff"].to_numpy(), newneg["teff"].to_numpy()])
    ))
    from scipy.stats import rankdata
    r_score = rankdata(np.concatenate([newpos["xgb_pu_score"].to_numpy(),
                                       newneg["xgb_pu_score"].to_numpy()]))
    r_enr = rankdata(np.concatenate([npos_enr[50], nneg_enr[50]]).astype(float))
    auc_combo = float(roc_auc_score(lab_y, r_score + r_enr))
    auc_combo_feh = float(roc_auc_score(
        lab_y, rankdata(-np.concatenate([newpos["feh"].to_numpy(),
                                         newneg["feh"].to_numpy()])) + r_score
    ))
    print(f"[validate] AUC using -[Fe/H] alone = {auc_feh:.3f}  "
          f"(-logg) = {auc_logg:.3f}  (-Teff) = {auc_teff:.3f}", flush=True)
    print(f"[validate] AUC score+knn50 = {auc_combo:.3f}  "
          f"score-[Fe/H] = {auc_combo_feh:.3f}", flush=True)
    references = {
        "auc_xgb_pu_score": auc_score,
        "auc_neg_feh": auc_feh,
        "auc_neg_logg": auc_logg,
        "auc_neg_teff": auc_teff,
        "auc_rank_sum_score_and_knn50": auc_combo,
        "auc_rank_sum_score_and_neg_feh": auc_combo_feh,
    }

    # ------------------------------------------------------ parameter summary
    def pstats(df_or_rows) -> dict:
        sub = df_or_rows if isinstance(df_or_rows, pd.DataFrame) else stars.loc[df_or_rows]
        return {
            "n": int(len(sub)),
            "teff_median": float(sub["teff"].median()),
            "logg_median": float(sub["logg"].median()),
            "feh_median": float(sub["feh"].median()),
            "snru_median": float(sub["snru"].median()),
        }

    params = {
        "unlabeled": pstats(np.where(y == 0)[0]),
        "known91": pstats(pos91),
        "new_positives": pstats(newpos),
        "new_negatives": pstats(newneg),
    }
    print(f"[param] new pos: Teff={params['new_positives']['teff_median']:.0f} "
          f"logg={params['new_positives']['logg_median']:.2f} "
          f"FeH={params['new_positives']['feh_median']:.2f} "
          f"SNRu={params['new_positives']['snru_median']:.1f}", flush=True)
    print(f"[param] new neg: Teff={params['new_negatives']['teff_median']:.0f} "
          f"logg={params['new_negatives']['logg_median']:.2f} "
          f"FeH={params['new_negatives']['feh_median']:.2f} "
          f"SNRu={params['new_negatives']['snru_median']:.1f}", flush=True)

    # -------------------------------------------------------------- exports
    lab_out = lab.drop(columns=["row"]).sort_values(["label", "xgb_pu_score"],
                                                  ascending=[False, False])
    for k in K_LIST:
        lab_out[f"knn{k}_known91"] = np.nan
    lab_out.loc[lab_out["uid"].isin(newpos["uid"]), [f"knn{k}_known91" for k in K_LIST]] = \
        np.column_stack([npos_enr[k] for k in K_LIST])
    lab_out.loc[lab_out["uid"].isin(newneg["uid"]), [f"knn{k}_known91" for k in K_LIST]] = \
        np.column_stack([nneg_enr[k] for k in K_LIST])
    lab_out["is_new_positive"] = lab_out["uid"].isin(newpos["uid"])
    lab_out["is_new_negative"] = lab_out["uid"].isin(newneg["uid"])
    lab_out.to_csv(RESULTS_DIR / "newlabel_analysis.csv", index=False)

    # figure 1: parameter distributions
    panels = [("teff", "Teff (K)"), ("logg", "log g"),
              ("feh", "[Fe/H]"), ("snru", "SNR_u")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    u = stars.loc[np.where(y == 0)[0]]
    for ax, (col, xlabel) in zip(axes.ravel(), panels):
        lo, hi = np.percentile(u[col], [0.2, 99.8])
        bins = np.linspace(lo, hi, 40)
        ax.hist(u[col], bins=bins, density=True, color=C_U, alpha=0.5,
                label="unlabeled U", zorder=1)
        for sub, colour, name in ((stars.loc[pos91], C_POS91, f"frozen 91"),
                                  (newpos, C_NEWPOS, f"new positives ({len(newpos)})"),
                                  (newneg, C_NEWNEG, f"new negatives ({len(newneg)})")):
            ax.hist(sub[col], bins=bins, density=True, histtype="step",
                    color=colour, lw=1.8, label=name, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("High-resolution labelled candidates vs. the frozen 91 and the "
                 "unlabeled pool", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig5_newlabel_parameters.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # figure 2: enrichment validation
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    ax = axes[0]
    k = 50
    bins = np.arange(-0.5, max(np.percentile(pos_enr[k], 99),
                               npos_enr[k].max(), nneg_enr[k].max()) + 1.5, 1.0)
    ax.hist(pos_enr[k], bins=bins, density=True, color=C_POS91, alpha=0.45,
            label=f"frozen 91 (n={len(pos91)})")
    ax.hist(npos_enr[k], bins=bins, density=True, color=C_NEWPOS, alpha=0.55,
            label=f"new positives (n={len(newpos)})")
    ax.hist(nneg_enr[k], bins=bins, density=True, color=C_NEWNEG, alpha=0.55,
            label=f"new negatives (n={len(newneg)})")
    ax.axvline(mu0[k], color="#185FA5", ls="--", lw=1.4,
               label=f"random = {mu0[k]:.3f}")
    ax.set_xlabel(f"frozen-91 positives among the {k} nearest neighbours")
    ax.set_ylabel("density")
    ax.set_title("does the enrichment statistic predict the new labels?", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    entries = [
        ("-[Fe/H] alone", auc_feh, float("nan"), C_NEWNEG),
        ("knn50 (vs frozen 91)", validation["k50"]["auc_newpos_vs_newneg"],
         validation["k50"]["auc_std_error"], C_NEWPOS),
        ("knn200 (vs frozen 91)", validation["k200"]["auc_newpos_vs_newneg"],
         validation["k200"]["auc_std_error"], C_NEWPOS),
        ("xgb_pu_score alone", auc_score, float("nan"), "#888780"),
        ("score + knn50", auc_combo, float("nan"), "#534AB7"),
        ("score + -[Fe/H]", auc_combo_feh, float("nan"), "#534AB7"),
    ]
    ys = np.arange(len(entries))
    ax.barh(ys, [e[1] for e in entries], color=[e[3] for e in entries], alpha=0.88,
            xerr=[0 if np.isnan(e[2]) else e[2] for e in entries], capsize=4)
    ax.axvline(0.5, color="#185FA5", ls="--", lw=1.3, label="chance")
    for y, e in zip(ys, entries):
        ax.text(min(e[1] + 0.02, 0.97), y, f"{e[1]:.3f}", va="center", fontsize=11,
                color="#5F5E5A")
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries], fontsize=11)
    ax.set_xlabel("AUC separating the new positives from the new negatives")
    ax.set_xlim(0, 1.12)
    ax.set_title("which ranking key best predicts the new labels?", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig6_newlabel_validation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "analysis": "feature characterisation of the high-resolution labelled set",
        "input": {
            "file": LABEL_CSV.name,
            "rows": int(len(lab)),
            "label1": int((lab["label"] == 1).sum()),
            "label0": int((lab["label"] == 0).sum()),
            "already_in_frozen_91": n_overlap,
            "from_frozen_654_candidates": int(lab["in_654_candidates"].sum()),
            "from_tier1": int(lab["in_tier1"].sum()),
        },
        "new_positives": int(len(newpos)),
        "new_negatives": int(len(newneg)),
        "enrichment": enrich_stats,
        "validation": validation,
        "ranking_key_references": references,
        "parameters": params,
        "threshold": threshold,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "newlabel_features.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[out] {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
