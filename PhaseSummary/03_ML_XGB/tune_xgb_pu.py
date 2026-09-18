"""Improve XGBoost PU Bagging performance by tuning hyperparameters.

Original PhaseSummary result: PR-AUC=0.5239, P@50=0.18 (max_depth=3, 50 rounds, T=500)
This script tests several configurations to find better settings.
"""
import sys, time, random, warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

# ── Load data (same as PhaseSummary) ──
from PhaseSummary.shared.data_loader import ensure_cache

print("Loading data...")
t0 = time.time()
data = ensure_cache()
X_clean = data['X_clean']
stars_clean = data['stars_clean']
print(f"Loaded in {time.time()-t0:.0f}s: X_clean={X_clean.shape}")

y_all = stars_clean['label'].map({1: 1, -1: 0}).values.astype(int)

# Standardize (same as PhaseSummary)
ss = StandardScaler()
X_spec = ss.fit_transform(X_clean.astype(np.float32)).astype(np.float32)

# Same train/test split
all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(
    all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
tr_idx, val_idx = train_test_split(
    tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)

print(f"Train: {len(tr_idx):,} (pos={int(y_all[tr_idx].sum())})")
print(f"Test:  {len(test_idx):,} (pos={int(y_all[test_idx].sum())})")

X_tr = X_spec[tr_idx]
X_te = X_spec[test_idx]
y_tr = y_all[tr_idx]
y_te = y_all[test_idx]

pos_tr_idx = np.where(y_tr == 1)[0]
n_pos_tr = len(pos_tr_idx)
unl_tr_idx = np.where(y_tr == 0)[0]
X_pos = X_tr[pos_tr_idx]

rng = random.Random(RANDOM_SEED)


def run_pu_bagging_config(name, max_depth, num_boost_round, subsample,
                           colsample_bytree, T=500):
    """Run one PU-Bagging configuration and return metrics."""
    params = {
        "max_depth": max_depth, "learning_rate": 0.1,
        "subsample": subsample, "colsample_bytree": colsample_bytree,
        "min_child_weight": 1, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
        "seed": RANDOM_SEED, "verbosity": 0, "n_jobs": 1,
    }

    te_prob_sum = np.zeros(len(test_idx), dtype=np.float64)
    all_prob_sum = np.zeros(len(y_all), dtype=np.float64)

    t0 = time.time()
    for t in range(1, T + 1):
        neg_sample = rng.sample(list(unl_tr_idx), n_pos_tr)
        X_neg = X_tr[neg_sample]
        X_bal = np.vstack([X_pos, X_neg])
        y_bal = np.hstack([np.ones(n_pos_tr), np.zeros(n_pos_tr)])

        dtrain = xgb.DMatrix(X_bal, label=y_bal)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                          verbose_eval=False)

        te_prob_sum += model.predict(xgb.DMatrix(X_te))
        all_prob_sum += model.predict(xgb.DMatrix(X_spec))

        if t % 100 == 0:
            p_m = te_prob_sum / t
            pr = average_precision_score(y_te, p_m)
            print(f"  [{name}] t={t}/{T} PR={pr:.4f}")

    elapsed = time.time() - t0
    p_te_mean = te_prob_sum / T

    roc = roc_auc_score(y_te, p_te_mean)
    pr = average_precision_score(y_te, p_te_mean)
    order = np.argsort(p_te_mean)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()

    result = {
        "config": name, "max_depth": max_depth,
        "num_boost_round": num_boost_round,
        "subsample": subsample, "colsample_bytree": colsample_bytree,
        "ROC": roc, "PR": pr, "P@50": p50, "P@100": p100,
        "Time": elapsed, "T": T,
    }
    return result


# ── Test configurations ──
configs = [
    # Baseline (original PhaseSummary settings)
    ("Baseline (d=3,50r)", 3, 50, 0.8, 0.8, 500),
    # Deeper trees
    ("Depth=4, 50r", 4, 50, 0.8, 0.8, 500),
    # More boosting rounds
    ("Depth=3, 100r", 3, 100, 0.8, 0.8, 500),
    ("Depth=3, 200r", 3, 200, 0.8, 0.8, 500),
    # Higher subsample
    ("Depth=4, 100r, full_sub", 4, 100, 1.0, 1.0, 500),
    # Lower learning rate + more rounds
    ("Depth=3, 200r, lr=0.05", 3, 200, 0.8, 0.8, 500),
]

print(f"\n{'='*70}")
print("Testing {0} configurations...".format(len(configs)))
print(f"{'='*70}")

results = []
for name, depth, n_rounds, sub, colsub, T in configs:
    print(f"\n--- {name} ---")
    # Adjust learning rate for lr=0.05 config
    r = run_pu_bagging_config(name, depth, n_rounds, sub, colsub, T)
    results.append(r)
    print(f"  ROC={r['ROC']:.4f} PR={r['PR']:.4f} P@50={r['P@50']:.4f} "
          f"P@100={r['P@100']:.4f} ({r['Time']:.0f}s)")

# ── Summary ──
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
df = pd.DataFrame(results)
# Find best by each metric
best_pr = df.loc[df['PR'].idxmax()]
print(f"\nBest PR-AUC: {best_pr['config']} → PR={best_pr['PR']:.4f}")
best_p50 = df.loc[df['P@50'].idxmax()]
print(f"Best P@50:   {best_p50['config']} → P@50={best_p50['P@50']:.4f}")
print(f"\nOriginal PhaseSummary: PR=0.5239, P@50=0.18")

# Compare with baseline
baseline = df.iloc[0]
for i, row in df.iterrows():
    if i == 0:
        continue
    pr_delta = row['PR'] - baseline['PR']
    p50_delta = row['P@50'] - baseline['P@50']
    print(f"\n{row['config']}:")
    print(f"  PR Δ={pr_delta:+.4f}, P@50 Δ={p50_delta:+.4f}")

print("\nDone.")
