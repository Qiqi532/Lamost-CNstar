"""Independent audit of the frozen ``xgb_threshold`` baseline.

Why this exists
---------------
The headline numbers of the threshold baseline (ROC-AUC 0.9883, PR-AUC 0.6030)
are computed on a 6,187-object held-out split, but the *only* positives anywhere
in that split come from the same 91-star literature sample the model was trained
on.  So the number measures "can the model re-find the stars it was taught",
not "can it find CN stars".

``diagnosis/cands_label.csv`` contains 99 high-resolution labels (46 confirmed
enhanced, 53 checked and not enhanced) that were **completely unknown when the
baseline was trained**.  Scoring those objects with the baseline gives the first
non-self-referential accuracy estimate for it.

What this script does
---------------------
1. Faithfully reproduces the baseline run (same splits, same seed, same 500
   bags) and **verifies** it against ``XGB/comparison_results.json``.
2. Scores all 41,243 objects, then audits the 99 independent labels:
   ROC-AUC, PR-AUC, score distributions, and how many confirmed positives /
   confirmed negatives fall above the model's own threshold.
3. Repeats the same audit with the frozen ``crossfit recall90`` scores, so the
   two models are compared on one common, independent yardstick.

Only the baseline model is refitted; the crossfit scores are read from the
frozen archive.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_dr13_all_cache import load_dr13_all_cache  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LABEL_CSV = HERE / "cands_label.csv"
COMPARISON_JSON = PROJECT_ROOT / "XGB" / "comparison_results.json"
FROZEN_ARRAYS = (PROJECT_ROOT / "crossfit" / "results" / "recall90" / "formal"
                 / "crossfit_arrays_recall90.npz")

RANDOM_SEED = 42
N_BAGS = 500

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_THR = "#185FA5"
C_XFIT = "#1D9E75"
C_POS = "#E24B4A"
C_NEG = "#BA7517"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    data = load_dr13_all_cache()
    X_clean = data["X_clean"]
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y_all = stars["label"].map({1: 1, -1: 0}).to_numpy().astype(int)
    n_total = len(y_all)
    print(f"[data] n={n_total:,}  positives={int(y_all.sum())}", flush=True)

    # ------------------------------------------------------ exact same splits
    all_idx = np.arange(n_total)
    tv_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
    tr_idx, val_idx = train_test_split(
        tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)
    y_tr, y_te = y_all[tr_idx], y_all[test_idx]
    n_pos_tr = int((y_tr == 1).sum())
    unl_tr_idx = np.where(y_tr == 0)[0]
    print(f"[split] train={len(tr_idx):,} (P={n_pos_tr})  test={len(test_idx):,}",
          flush=True)

    # ------------------------------------------------- reproduce the baseline
    ss = StandardScaler()
    X_all_b = ss.fit_transform(X_clean.astype(np.float32)).astype(np.float32)
    X_pos = X_all_b[tr_idx][np.where(y_tr == 1)[0]]
    X_te_b = X_all_b[test_idx]
    d_all = xgb.DMatrix(X_all_b)          # built once, values never change
    d_te = xgb.DMatrix(X_te_b)

    params = {
        "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "seed": RANDOM_SEED,
        "verbosity": 0, "n_jobs": 1,
    }
    rng = random.Random(RANDOM_SEED)
    te_sum = np.zeros(len(y_te))
    all_sum = np.zeros(n_total)
    t0 = time.perf_counter()
    for t in range(1, N_BAGS + 1):
        neg = rng.sample(list(unl_tr_idx), n_pos_tr)
        X_bal = np.vstack([X_pos, X_all_b[tr_idx][neg]])
        y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])
        model = xgb.train(params, xgb.DMatrix(X_bal, label=y_bal),
                          num_boost_round=50, verbose_eval=False)
        te_sum += model.predict(d_te)
        all_sum += model.predict(d_all)
        if t % 100 == 0:
            pm = te_sum / t
            print(f"  [{t:4d}/{N_BAGS}] ROC={roc_auc_score(y_te, pm):.4f} "
                  f"PR={average_precision_score(y_te, pm):.4f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    te_mean = te_sum / N_BAGS
    all_mean = all_sum / N_BAGS

    roc_te = float(roc_auc_score(y_te, te_mean))
    pr_te = float(average_precision_score(y_te, te_mean))
    reference = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))["baseline"]
    reproduced = (abs(roc_te - reference["roc"]) < 1e-6
                  and abs(pr_te - reference["pr"]) < 1e-6)
    print(f"[verify] reproduced ROC={roc_te:.6f} (ref {reference['roc']:.6f})  "
          f"PR={pr_te:.6f} (ref {reference['pr']:.6f})  "
          f"-> {'MATCH' if reproduced else 'MISMATCH'}", flush=True)

    # model's own threshold from the frozen export
    thr = float(reference["threshold"])

    # -------------------------------------------- independent label audit
    lab = pd.read_csv(LABEL_CSV)
    lab["uid"] = lab["uid"].astype(str)
    uid_to_row = {u: i for i, u in enumerate(stars["uid"].astype(str))}
    lab["row"] = lab["uid"].map(uid_to_row)
    lab = lab.loc[lab["row"].notna()].copy()
    lab["row"] = lab["row"].astype(int)
    lab = lab.loc[lab["label"].isin([0, 1]) & (lab["row"].map(lambda r: y_all[r]) == 0)]
    lab_y = lab["label"].to_numpy()
    lab_rows = lab["row"].to_numpy()
    thr_scores = all_mean[lab_rows]
    print(f"[labels] {len(lab)} independent objects "
          f"({int(lab_y.sum())} positive / {int((1 - lab_y).sum())} negative)",
          flush=True)

    blob = np.load(FROZEN_ARRAYS, allow_pickle=True)
    xfit_all = np.asarray(blob["score_mean"], dtype=np.float64)
    xfit_scores = xfit_all[lab_rows]

    def audit(scores: np.ndarray, name: str) -> dict:
        roc = float(roc_auc_score(lab_y, scores))
        pr = float(average_precision_score(lab_y, scores))
        p = scores[lab_y == 1]
        n = scores[lab_y == 0]
        # Hanley-McNeil standard error for the AUC
        n1, n0 = int(lab_y.sum()), int((1 - lab_y).sum())
        se = float(np.sqrt(
            (roc * (1 - roc) + (n1 - 1) * (roc / (2 - roc) - roc ** 2)
             + (n0 - 1) * (2 * roc ** 2 / (1 + roc) - roc ** 2)) / (n1 * n0)
        ))
        out = {
            "model": name,
            "auc_on_99": roc, "auc_std_error": se,
            "pr_auc_on_99": pr,
            "median_score_positive": float(np.median(p)),
            "median_score_negative": float(np.median(n)),
            "share_positive_above_threshold": float(np.mean(p >= thr)),
            "share_negative_above_threshold": float(np.mean(n >= thr)),
        }
        print(f"[audit] {name:<16} AUC={roc:.3f}±{se:.3f}  PR-AUC={pr:.3f}  "
              f"median pos={np.median(p):.3f} neg={np.median(n):.3f}  "
              f"above model threshold: pos {out['share_positive_above_threshold'] * 100:.0f}% "
              f"/ neg {out['share_negative_above_threshold'] * 100:.0f}%", flush=True)
        return out

    audit_thr = audit(thr_scores, "xgb_threshold")
    audit_xfit = audit(xfit_scores, "crossfit90")

    # how does the model behave on the frozen 91 (self-referential reference)?
    pos91 = np.where(y_all == 1)[0]
    self_ref = {
        "auc_on_91_selfref": float(roc_auc_score(y_all[pos91], all_mean[pos91]))
        if len(np.unique(all_mean[pos91])) > 1 else None,
        "note": "all 91 are training-adjacent; this is the self-referential number",
    }
    print(f"[selfref] xgb_threshold median score on the 91 = "
          f"{np.median(all_mean[pos91]):.3f}", flush=True)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(thr_scores[lab_y == 0], bins=bins, color=C_NEG, alpha=0.6,
            label=f"confirmed NON-enhanced (n={int((1 - lab_y).sum())})")
    ax.hist(thr_scores[lab_y == 1], bins=bins, color=C_POS, alpha=0.6,
            label=f"confirmed enhanced (n={int(lab_y.sum())})")
    ax.axvline(thr, color=C_THR, ls="--", lw=1.5,
               label=f"model threshold = {thr:.3f}")
    ax.set_xlabel("xgb_threshold score (500-bag PU, 700-d)")
    ax.set_ylabel("count")
    ax.set_title("xgb_threshold on the 99 independent labels", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    for scores, colour, name in ((thr_scores, C_THR, "xgb_threshold"),
                                 (xfit_scores, C_XFIT, "crossfit recall90")):
        fpr, tpr, _ = roc_curve(lab_y, scores)
        auc = roc_auc_score(lab_y, scores)
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#888780", ls=":", lw=1.2, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on the 99 independent labels", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig8_threshold_audit.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "analysis": "independent audit of the xgb_threshold baseline",
        "protocol": {
            "reproduced_baseline": bool(reproduced),
            "reference_roc": reference["roc"], "reference_pr": reference["pr"],
            "reproduced_roc": roc_te, "reproduced_pr": pr_te,
            "evaluation_split_size": int(len(y_te)),
            "n_bags": N_BAGS, "threshold": thr,
            "independent_labels": int(len(lab)),
            "independent_positives": int(lab_y.sum()),
        },
        "self_referential_headline": {
            "roc_on_15pct_holdout": roc_te,
            "pr_on_15pct_holdout": pr_te,
            "median_score_on_frozen_91": float(np.median(all_mean[pos91])),
            "caveat": ("the holdout shares the same 91-star literature sample; "
                       "positives are as few as 28 in the test split"),
        },
        "independent_audit": {"xgb_threshold": audit_thr, "crossfit90": audit_xfit},
        "feh_shortcut_documented": {
            "source": "XGB/XGB_PU_实验结果分析.md (2026-08-28)",
            "headline_pr": 0.6030,
            "pr_after_feh_matched_negative_sampling": 0.4495,
            "pr_feh_only_kde": 0.4435,
            "interpretation": ("the headline PR contains a large [Fe/H] prior; "
                               "pure CN discriminative power is ~0.45-0.51"),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (RESULTS_DIR / "threshold_audit.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"[out] {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
