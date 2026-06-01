"""PU-Bagging on raw spectra (700 pixels) vs CN 9-D features.

Exposes:
  load_pu_data() -> dict        — load cached spectra + features + train/test split
  run_pu_bagging(...) -> dict   — run PU-Bagging on given feature matrix
  build_comparison_df(...) -> DataFrame — build comparison table from results
"""
import sys, time, warnings
from pathlib import Path
import random

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
)

RANDOM_SEED = 42
DEFAULT_T = 500


# ═══════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════

def load_pu_data():
    """Load cached spectra and feature data, prepare both CN 9-D and raw spectra matrices.

    Returns dict with keys:
      X_cn9, X_spec, y_all, df_model, cluster_ids,
      tr_idx, test_idx, use_cols_cn9
    """
    CACHE_DIR = Path(__file__).resolve().parent / "_cache"
    feature_df = pd.read_pickle(CACHE_DIR / "feature_df.pkl")
    stars_clean = pd.read_pickle(CACHE_DIR / "stars_clustered.pkl")
    X_spectra = np.load(CACHE_DIR / "X_clean.npy")

    df_model = feature_df.dropna(subset=["label"]).copy()
    y_all = df_model["label"].map({1: 1, -1: 0}).values.astype(int)
    cluster_ids = stars_clean["masked_cluster_id"].values

    # CN 9-D (baseline)
    use_cols_cn9 = [c for c in FEATURE_COLS_CN9 if c in df_model.columns]
    X_cn9 = df_model[use_cols_cn9].values.astype(np.float32)
    ss_cn9 = StandardScaler()
    X_cn9 = ss_cn9.fit_transform(X_cn9).astype(np.float32)

    # Raw spectra (700-D)
    X_spec = X_spectra.astype(np.float32)
    ss_spec = StandardScaler()
    X_spec = ss_spec.fit_transform(X_spec).astype(np.float32)

    # Train/test split
    all_idx = np.arange(len(y_all))
    tv_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
    tr_idx, val_idx = train_test_split(
        tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)

    return {
        "X_cn9": X_cn9,
        "X_spec": X_spec,
        "y_all": y_all,
        "df_model": df_model,
        "cluster_ids": cluster_ids,
        "tr_idx": tr_idx,
        "test_idx": test_idx,
        "use_cols_cn9": use_cols_cn9,
    }


def run_pu_bagging(X_all, y_all, tr_idx, test_idx, cluster_ids, df_model,
                   T=DEFAULT_T, name="model"):
    """Run PU-Bagging on given feature matrix.

    Parameters
    ----------
    X_all : np.ndarray (n_samples, n_features)
    y_all : np.ndarray (n_samples,)  — 1=known positive, 0=unlabeled
    tr_idx, test_idx : np.ndarray — train / test split indices
    cluster_ids : np.ndarray (n_samples,)
    df_model : pd.DataFrame — metadata for bias evaluation
    T : int — number of bagging iterations
    name : str — label for logging

    Returns
    -------
    dict with keys: name, dim, ROC, PR, P@50, P@100, |r_teff| raw,
    |r_teff| z, Mean bias raw, Mean bias z, Within-r, Stability, Time, T,
    probs_all, probs_all_std
    """
    X_tr = X_all[tr_idx]
    X_te = X_all[test_idx]
    y_tr_local = y_all[tr_idx]
    y_te = y_all[test_idx]

    pos_tr_idx = np.where(y_tr_local == 1)[0]
    n_pos = len(pos_tr_idx)
    unl_tr_idx = np.where(y_tr_local == 0)[0]
    X_pos = X_tr[pos_tr_idx]

    te_prob_sum = np.zeros(len(test_idx), dtype=np.float64)
    te_prob_sq = np.zeros(len(test_idx), dtype=np.float64)
    all_prob_sum = np.zeros(len(y_all), dtype=np.float64)
    all_prob_sq = np.zeros(len(y_all), dtype=np.float64)

    xgb_params = {
        "max_depth": 3, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
        "seed": RANDOM_SEED,
        "verbosity": 0, "n_jobs": 1,
    }

    rng = random.Random(RANDOM_SEED)
    t0 = time.time()
    report_every = max(T // 5, 1)

    for t in range(1, T + 1):
        neg_sample = rng.sample(list(unl_tr_idx), n_pos)
        X_neg = X_tr[neg_sample]
        X_bal = np.vstack([X_pos, X_neg])
        y_bal = np.hstack([np.ones(n_pos), np.zeros(n_pos)])

        dtrain = xgb.DMatrix(X_bal, label=y_bal)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, verbose_eval=False)

        p_te = model.predict(xgb.DMatrix(X_te))
        p_all = model.predict(xgb.DMatrix(X_all))

        te_prob_sum += p_te
        te_prob_sq += p_te ** 2
        all_prob_sum += p_all
        all_prob_sq += p_all ** 2

        if t % report_every == 0:
            p_m = te_prob_sum / t
            pr = average_precision_score(y_te, p_m)
            print(f"  [{name}] {t:4d}/{T}  PR={pr:.4f}  ({time.time()-t0:.0f}s)")

    p_te_mean = te_prob_sum / T
    p_te_std = np.sqrt(np.maximum(te_prob_sq / T - p_te_mean ** 2, 0))
    p_all_mean = all_prob_sum / T
    p_all_std = np.sqrt(np.maximum(all_prob_sq / T - p_all_mean ** 2, 0))

    elapsed = time.time() - t0

    # Metrics
    roc = roc_auc_score(y_te, p_te_mean)
    pr = average_precision_score(y_te, p_te_mean)
    order = np.argsort(p_te_mean)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()

    unl_idx_all = np.where(y_all == 0)[0]
    bias_raw = compute_parameter_bias(p_all_mean[unl_idx_all], df_model.iloc[unl_idx_all])
    z_all = compute_cluster_zscore(p_all_mean[unl_idx_all], cluster_ids[unl_idx_all])
    bias_z = compute_parameter_bias(z_all, df_model.iloc[unl_idx_all])

    within_rs = []
    for cid in np.unique(cluster_ids[unl_idx_all]):
        cm = cluster_ids[unl_idx_all] == cid
        if cm.sum() >= 10 and "delta_CN3839" in df_model.columns:
            dcn = df_model.iloc[unl_idx_all[cm]]["delta_CN3839"].values
            cp = p_all_mean[unl_idx_all][cm]
            valid = np.isfinite(dcn) & np.isfinite(cp)
            if valid.sum() >= 10:
                within_rs.append(spearmanr(dcn[valid], cp[valid])[0])
    within_r = float(np.median(within_rs)) if within_rs else np.nan

    return {
        "name": name,
        "dim": X_all.shape[1],
        "ROC": roc, "PR": pr,
        "P@50": p50, "P@100": p100,
        "|r_teff| raw": abs(bias_raw.get("spearmanr_teff", 0)),
        "|r_teff| z": abs(bias_z.get("spearmanr_teff", 0)),
        "Mean bias raw": np.mean([abs(bias_raw.get(f"spearmanr_{p}", 0))
                                  for p in ["teff", "logg", "feh"]]),
        "Mean bias z": np.mean([abs(bias_z.get(f"spearmanr_{p}", 0))
                                for p in ["teff", "logg", "feh"]]),
        "Within-r": within_r,
        "Stability": float(np.median(p_te_std)),
        "Time": elapsed,
        "T": T,
        "probs_all": p_all_mean,
        "probs_all_std": p_all_std,
    }


def build_comparison_df(res_cn9, res_spec):
    """Build a comparison DataFrame from two PU-Bagging result dicts."""
    results = []
    for r in [res_cn9, res_spec]:
        results.append({
            "Feature Set": r["name"],
            "Dim": r["dim"],
            "T": r["T"],
            "ROC": r["ROC"], "PR": r["PR"],
            "P@50": r["P@50"], "P@100": r["P@100"],
            "|r_teff| raw": r["|r_teff| raw"],
            "|r_teff| z": r["|r_teff| z"],
            "Mean bias raw": r["Mean bias raw"],
            "Mean bias z": r["Mean bias z"],
            "Within-r": r["Within-r"],
            "Stability": r["Stability"],
            "Time": r["Time"],
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════
# Standalone execution (for backward compatibility)
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    data = load_pu_data()
    print(f"Loaded: spectra ({len(data['X_spec'])}, {data['X_spec'].shape[1]}), "
          f"CN 9-D ({len(data['use_cols_cn9'])} cols)")
    print(f"Train: {len(data['tr_idx']):,} (pos={int(data['y_all'][data['tr_idx']].sum())})")
    print(f"Test:  {len(data['test_idx']):,} (pos={int(data['y_all'][data['test_idx']].sum())})")

    T = DEFAULT_T

    print(f"\n{'='*60}")
    print(f"PU-Bagging: CN 9-D (T={T})")
    print(f"{'='*60}")
    res_cn9 = run_pu_bagging(
        data["X_cn9"], data["y_all"], data["tr_idx"], data["test_idx"],
        data["cluster_ids"], data["df_model"], T=T, name="CN 9-D")

    print(f"\n{'='*60}")
    print(f"PU-Bagging: Raw Spectra 700-D (T={T})")
    print(f"{'='*60}")
    res_spec = run_pu_bagging(
        data["X_spec"], data["y_all"], data["tr_idx"], data["test_idx"],
        data["cluster_ids"], data["df_model"], T=T, name="Raw Spectra")

    comp_df = build_comparison_df(res_cn9, res_spec)
    print(f"\n{'='*60}")
    print("PU-Bagging COMPARISON")
    print(f"{'='*60}")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(comp_df.to_string(index=False))

    # Save probabilities
    out = data["df_model"][["teff", "logg", "feh", "label"]].copy()
    out["pu_spec_prob"] = res_spec["probs_all"]
    out["pu_spec_std"] = res_spec["probs_all_std"]
    out["pu_cn9_prob"] = res_cn9["probs_all"]
    out["pu_cn9_std"] = res_cn9["probs_all_std"]
    out.to_csv("pu_bagging_spec_probs.csv", index=False)
    print("\nSaved per-star probabilities to pu_bagging_spec_probs.csv")
    print("Done.")
