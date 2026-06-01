"""Focal Loss γ tuning on CN 9-D feature set (XGBoost, single train/val/test).

Exposes:
  load_tuning_data() -> dict   — load cached features + train/val/test split
  run_gamma_tuning(...) -> DataFrame — run γ sensitivity experiments
"""
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings("ignore")

from ML.utils import (
    FEATURE_COLS_CN9,
    compute_cluster_zscore, compute_parameter_bias,
    xgb_focal_loss_obj, xgb_focal_loss_eval,
)

RANDOM_SEED = 42


# ═══════════════════════════════════════════════
# Custom asymmetric focal loss
# ═══════════════════════════════════════════════
def xgb_asym_focal_loss_obj(y_pred, dtrain, gamma_pos=3.0, gamma_neg=1.0, alpha=0.5):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    p_t = p * y_true + (1 - p) * (1 - y_true)
    gamma_t = gamma_pos * y_true + gamma_neg * (1 - y_true)
    focal_weight = (1 - p_t) ** gamma_t
    grad = alpha_t * focal_weight * (p - y_true)
    hess = alpha_t * focal_weight * p * (1 - p)
    return grad, hess


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def load_tuning_data():
    """Load cached feature data and prepare train/val/test splits.

    Returns dict with keys:
      df_model, y_all, cluster_ids, n_pos, use_cols,
      X_tr, y_tr, X_val, y_val, X_te, y_te,
      X_all_unl, df_unl, cluster_ids_unl
    """
    CACHE_DIR = Path(__file__).resolve().parent / "_cache"
    feature_df = pd.read_pickle(CACHE_DIR / "feature_df.pkl")
    stars_clean = pd.read_pickle(CACHE_DIR / "stars_clustered.pkl")

    use_cols = [c for c in FEATURE_COLS_CN9 if c in feature_df.columns]

    df_model = feature_df.dropna(subset=["label"]).copy()
    y_all = df_model["label"].map({1: 1, -1: 0}).values.astype(int)
    cluster_ids = stars_clean["masked_cluster_id"].values

    X_all = df_model[use_cols].values.astype(np.float32)
    ss = StandardScaler()
    X_all = ss.fit_transform(X_all).astype(np.float32)

    all_idx = np.arange(len(y_all))
    tv_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
    tr_idx, val_idx = train_test_split(
        tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)

    unl_idx = np.where(y_all == 0)[0]

    return {
        "df_model": df_model,
        "y_all": y_all,
        "cluster_ids": cluster_ids,
        "X_tr": X_all[tr_idx], "y_tr": y_all[tr_idx],
        "X_val": X_all[val_idx], "y_val": y_all[val_idx],
        "X_te": X_all[test_idx], "y_te": y_all[test_idx],
        "X_all_unl": X_all[unl_idx],
        "df_unl": df_model.iloc[unl_idx],
        "cluster_ids_unl": cluster_ids[unl_idx],
        "n_pos": int(y_all.sum()),
        "use_cols": use_cols,
    }


def run_gamma_tuning(X_tr, y_tr, X_val, y_val, X_te, y_te,
                     X_all_unl, df_unl, cluster_ids_unl,
                     standard_gammas=None, asymmetric_pairs=None,
                     include_bce=True, verbose=True):
    """Run Focal Loss γ sensitivity experiments.

    Parameters
    ----------
    standard_gammas : list of float
        γ values to test for standard focal loss.
    asymmetric_pairs : list of (float, float)
        (γ_pos, γ_neg) pairs for asymmetric focal loss.
    include_bce : bool
        Whether to include weighted and unweighted BCE baselines.

    Returns
    -------
    pd.DataFrame sorted by PR-AUC, columns: Loss, ROC, PR, P@50, P@100,
    |r_teff| raw, |r_teff| z, Mean bias raw, Mean bias z, Within-r, Time
    """
    if standard_gammas is None:
        standard_gammas = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
    if asymmetric_pairs is None:
        asymmetric_pairs = [(3, 1), (5, 1), (5, 2), (3, 0.5)]

    experiments = []
    for gamma in standard_gammas:
        experiments.append({"name": f"γ={gamma}", "type": "standard", "gamma": gamma})
    for gp, gn in asymmetric_pairs:
        experiments.append({"name": f"γ_pos={gp},γ_neg={gn}", "type": "asymmetric",
                            "gp": gp, "gn": gn})
    if include_bce:
        experiments.append({"name": "BCE+weight", "type": "bce_weighted"})
        experiments.append({"name": "BCE (no weight)", "type": "bce_plain"})

    results = []

    for exp in experiments:
        name = exp["name"]
        if verbose:
            print(f"  [{name}]", end=" ", flush=True)

        n_pos = int(y_tr.sum())
        n_neg = int((1 - y_tr).sum())
        alpha = n_neg / max(n_pos, 1)
        alpha = min(alpha / (1 + alpha), 0.95)

        xgb_params = {
            "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 2, "gamma": 0.1,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "seed": RANDOM_SEED,
            "verbosity": 0,
        }

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_te, label=y_te)

        t0 = time.time()

        if exp["type"] == "standard":
            _gamma = exp["gamma"]
            def obj(y_pred, dtrain):
                return xgb_focal_loss_obj(y_pred, dtrain, gamma=_gamma, alpha=alpha)
            model = xgb.train(xgb_params, dtrain, num_boost_round=300,
                              evals=[(dtrain, "train"), (dval, "val")],
                              obj=obj, early_stopping_rounds=30, verbose_eval=False)

        elif exp["type"] == "asymmetric":
            _gp, _gn = exp["gp"], exp["gn"]
            def obj(y_pred, dtrain):
                return xgb_asym_focal_loss_obj(y_pred, dtrain,
                                               gamma_pos=_gp, gamma_neg=_gn, alpha=alpha)
            model = xgb.train(xgb_params, dtrain, num_boost_round=300,
                              evals=[(dtrain, "train"), (dval, "val")],
                              obj=obj, early_stopping_rounds=30, verbose_eval=False)

        elif exp["type"] == "bce_weighted":
            xgb_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            model = xgb.train(xgb_params, dtrain, num_boost_round=300,
                              evals=[(dtrain, "train"), (dval, "val")],
                              early_stopping_rounds=30, verbose_eval=False)

        elif exp["type"] == "bce_plain":
            model = xgb.train(xgb_params, dtrain, num_boost_round=300,
                              evals=[(dtrain, "train"), (dval, "val")],
                              early_stopping_rounds=30, verbose_eval=False)

        train_time = time.time() - t0

        # Evaluate
        y_prob = model.predict(dtest)
        roc = roc_auc_score(y_te, y_prob)
        pr = average_precision_score(y_te, y_prob)

        order = np.argsort(y_prob)[::-1]
        p50 = y_te[order[:min(50, len(order))]].mean()
        p100 = y_te[order[:min(100, len(order))]].mean()

        # Bias on unlabeled
        prob_unl = model.predict(xgb.DMatrix(X_all_unl))
        bias_raw = compute_parameter_bias(prob_unl, df_unl)
        z_unl = compute_cluster_zscore(prob_unl, cluster_ids_unl)
        bias_z = compute_parameter_bias(z_unl, df_unl)

        bias_teff_raw = abs(bias_raw.get("spearmanr_teff", 0))
        bias_teff_z = abs(bias_z.get("spearmanr_teff", 0))
        mean_bias_raw = np.mean([abs(bias_raw.get(f"spearmanr_{p}", 0))
                                 for p in ["teff", "logg", "feh"]])
        mean_bias_z = np.mean([abs(bias_z.get(f"spearmanr_{p}", 0))
                               for p in ["teff", "logg", "feh"]])

        # Within-cluster r
        within_rs = []
        for cid in np.unique(cluster_ids_unl):
            cm = cluster_ids_unl == cid
            if cm.sum() >= 10 and "delta_CN3839" in df_unl.columns:
                dcn = df_unl.iloc[cm]["delta_CN3839"].values
                cp = prob_unl[cm]
                valid = np.isfinite(dcn) & np.isfinite(cp)
                if valid.sum() >= 10:
                    within_rs.append(spearmanr(dcn[valid], cp[valid])[0])
        within_r = float(np.median(within_rs)) if within_rs else np.nan

        results.append({
            "Loss": name, "ROC": roc, "PR": pr,
            "P@50": p50, "P@100": p100,
            "|r_teff| raw": bias_teff_raw, "|r_teff| z": bias_teff_z,
            "Mean bias raw": mean_bias_raw, "Mean bias z": mean_bias_z,
            "Within-r": within_r, "Time": train_time,
        })

        if verbose:
            print(f"PR={pr:.4f}  P@100={p100:.4f}  "
                  f"|r_teff|_z={bias_teff_z:.4f}  within-r={within_r:.4f}  "
                  f"({train_time:.0f}s)")

    results_df = pd.DataFrame(results).sort_values("PR", ascending=False)
    return results_df


# ═══════════════════════════════════════════════
# Standalone execution (for backward compatibility)
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 80)
    print("FOCAL LOSS γ TUNING — CN 9-D features, XGBoost")
    print("=" * 80)

    data = load_tuning_data()
    print(f"Loaded: {len(data['df_model']):,} rows")
    print(f"Features ({len(data['use_cols'])}): {data['use_cols']}")
    print(f"\nTrain: {len(data['X_tr'])} (pos={int(data['y_tr'].sum())})")
    print(f"Val:   {len(data['X_val'])} (pos={int(data['y_val'].sum())})")
    print(f"Test:  {len(data['X_te'])} (pos={int(data['y_te'].sum())})")

    results_df = run_gamma_tuning(
        data["X_tr"], data["y_tr"],
        data["X_val"], data["y_val"],
        data["X_te"], data["y_te"],
        data["X_all_unl"], data["df_unl"], data["cluster_ids_unl"],
    )

    print("\n" + "=" * 80)
    print("γ TUNING RESULTS — sorted by PR-AUC")
    print("=" * 80)

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest: {best['Loss']} — PR={best['PR']:.4f}, "
          f"within-r={best['Within-r']:.4f}, |r_teff|_z={best['|r_teff| z']:.4f}")

    results_df.to_csv("gamma_tuning_results.csv", index=False)
    print("\nSaved to gamma_tuning_results.csv")
    print("Done.")
