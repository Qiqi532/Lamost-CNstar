"""Quick narrow-only (700px) Deep PU experiment — validates pipeline before full run."""
import sys, time, warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from Deep.models.resnet1d import SpectraResNet, SimpleConvNet
from Deep.pu_trainer import PUResampleTrainer

sys.path.insert(0, str(_PROJECT_ROOT / "ML"))
from utils import compute_cluster_zscore, compute_parameter_bias

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Load
CACHE_ML = Path("ML/_cache")
feature_df = pd.read_pickle(CACHE_ML / "feature_df.pkl")
stars_clean = pd.read_pickle(CACHE_ML / "stars_clustered.pkl")
X_all = np.load(CACHE_ML / "X_clean.npy").astype(np.float32)
print(f"Loaded: {X_all.shape}")

df_model = feature_df.dropna(subset=["label"]).copy()
y_all = df_model["label"].map({1: 1, -1: 0}).values.astype(np.float32)
cluster_ids = stars_clean["masked_cluster_id"].values

# Split
all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
tr_idx, val_idx = train_test_split(tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)
unl_idx = np.where(y_all == 0)[0]

tr_pos_mask = y_all[tr_idx] == 1
X_tr_pos = X_all[tr_idx][tr_pos_mask]
X_tr_unl = X_all[tr_idx][~tr_pos_mask]
X_val = X_all[val_idx]; y_val = y_all[val_idx]
X_te = X_all[test_idx]; y_te = y_all[test_idx]

ss = StandardScaler().fit(X_all[tr_idx])
X_tr_pos = ss.transform(X_tr_pos).astype(np.float32)
X_tr_unl = ss.transform(X_tr_unl).astype(np.float32)
X_val = ss.transform(X_val).astype(np.float32)
X_te = ss.transform(X_te).astype(np.float32)
X_all_s = ss.transform(X_all).astype(np.float32)

print(f"Train pos={len(X_tr_pos)}, unl={len(X_tr_unl)}, val={len(X_val)}, test={len(X_te)}")

results = {}

# --- Experiment A: SimpleConvNet ---
print("\n=== A: SimpleConvNet (700px) ===")
model_a = SimpleConvNet(input_dim=700)
trainer_a = PUResampleTrainer(model_a, device=DEVICE, mixup_alpha=0.2)
t0 = time.time()
model_a, ep_a, pr_a, hist_a = trainer_a.train(
    X_tr_pos, X_tr_unl, X_val, y_val, epochs=200, patience=40, verbose=True
)
p_te_a = trainer_a.predict(X_te)
p_all_a = trainer_a.predict(X_all_s)
results["SimpleConvNet"] = {"model": model_a, "p_te": p_te_a, "p_all": p_all_a,
                             "time": time.time()-t0, "best_ep": ep_a}

# --- Experiment B: ResNet+SE ---
print("\n=== B: ResNet+SE (700px) ===")
model_b = SpectraResNet(input_dim=700, base_ch=32)
trainer_b = PUResampleTrainer(model_b, device=DEVICE, mixup_alpha=0.2)
t0 = time.time()
model_b, ep_b, pr_b, hist_b = trainer_b.train(
    X_tr_pos, X_tr_unl, X_val, y_val, epochs=200, patience=40, verbose=True
)
p_te_b = trainer_b.predict(X_te)
p_all_b = trainer_b.predict(X_all_s)
results["ResNet+SE"] = {"model": model_b, "p_te": p_te_b, "p_all": p_all_b,
                         "time": time.time()-t0, "best_ep": ep_b}

# --- Evaluate ---
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

for name, r in results.items():
    p_te = r["p_te"]; p_all = r["p_all"]
    pr = average_precision_score(y_te, p_te)
    roc = roc_auc_score(y_te, p_te)
    order = np.argsort(p_te)[::-1]
    p50 = y_te[order[:min(50, len(order))]].mean()
    p100 = y_te[order[:min(100, len(order))]].mean()

    bias_raw = compute_parameter_bias(p_all[unl_idx], df_model.iloc[unl_idx])
    z_all = compute_cluster_zscore(p_all[unl_idx], cluster_ids[unl_idx])
    z_all = np.nan_to_num(z_all, nan=0.0)
    bias_z = compute_parameter_bias(z_all, df_model.iloc[unl_idx])

    within_rs = []
    for cid in np.unique(cluster_ids[unl_idx]):
        cm = cluster_ids[unl_idx] == cid
        if cm.sum() >= 10 and "delta_CN3839" in df_model.columns:
            dcn = df_model.iloc[unl_idx[cm]]["delta_CN3839"].values
            cp = p_all[unl_idx][cm]
            valid = np.isfinite(dcn) & np.isfinite(cp)
            if valid.sum() >= 10:
                within_rs.append(spearmanr(dcn[valid], cp[valid])[0])
    within_r = float(np.median(within_rs)) if within_rs else np.nan

    print(f"\n  {name} ({r['time']:.0f}s, best_ep={r['best_ep']}):")
    print(f"    PR={pr:.4f}  ROC={roc:.4f}  P@50={p50:.4f}  P@100={p100:.4f}")
    print(f"    |r_teff| raw={abs(bias_raw.get('spearmanr_teff',0)):.4f}  "
          f"|r_teff| z={abs(bias_z.get('spearmanr_teff',0)):.4f}")
    print(f"    Mean bias z={np.mean([abs(bias_z.get(f'spearmanr_{p}',0)) for p in ['teff','logg','feh']]):.4f}")
    print(f"    Within-cluster r={within_r:.4f}")

print("\n  Baseline: XGBoost PU-Bagging PR=0.848")

# Save
np.savez("Deep/_cache/pu_narrow_results.npz",
         simpleconv_te=p_te_a, simpleconv_all=p_all_a,
         resnet_te=p_te_b, resnet_all=p_all_b,
         y_te=y_te, y_all=y_all)
print("\nSaved to Deep/_cache/pu_narrow_results.npz")
print("Done.")
