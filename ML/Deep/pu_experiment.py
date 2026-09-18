"""Deep PU Learning experiment: narrow (700px) vs full (5000px) spectra.

Compares: Simple Conv1D vs ResNet+SE, with PU-Resample training.
"""
import sys, time, warnings, os
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
from Deep.pu_trainer import PUResampleTrainer, train_ensemble, ensemble_predict

# ML utilities for evaluation
sys.path.insert(0, str(_PROJECT_ROOT / "ML"))
from utils import compute_cluster_zscore, compute_parameter_bias

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ═══════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════
CACHE_ML = Path("ML/_cache")
CACHE_DEEP = Path("Deep/_cache")
CACHE_DEEP.mkdir(parents=True, exist_ok=True)

# Shared labels and cluster info
feature_df = pd.read_pickle(CACHE_ML / "feature_df.pkl")
stars_clean = pd.read_pickle(CACHE_ML / "stars_clustered.pkl")
df_model = feature_df.dropna(subset=["label"]).copy()
y_all = df_model["label"].map({1: 1, -1: 0}).values.astype(np.float32)
cluster_ids = stars_clean["masked_cluster_id"].values

# --- Narrow spectra (3800-4500A, 700px) ---
print("\n[1] Loading NARROW spectra (3800-4500A, 700px)...")
X_narrow = np.load(CACHE_ML / "X_clean.npy").astype(np.float32)
print(f"  Shape: {X_narrow.shape}")

# --- Full spectra (3800-8800A, 5000px) ---
FULL_CACHE = CACHE_DEEP / "X_full_5000.npy"
if FULL_CACHE.exists():
    print(f"\n[2] Loading FULL spectra (3800-8800A, 5000px) from cache...")
    X_full = np.load(FULL_CACHE).astype(np.float32)
    print(f"  Shape: {X_full.shape}")
else:
    print(f"\n[2] Generating FULL spectra (3800-8800A, 5000px) — one-time ~10 min...")
    from ML.utils import load_and_preprocess
    common_wave_full = np.arange(3800.0, 8800.0, 1.0)
    data_full = load_and_preprocess(
        stars_csv="stars.csv", spectra_folder="dr13_new",
        cn_catalogs=["CNstar.csv", "FT_cands.csv"],
        common_wave=common_wave_full, show_progress=False,
    )
    X_full = data_full["X_clean"].astype(np.float32)
    np.save(FULL_CACHE, X_full)
    print(f"  Shape: {X_full.shape} — cached to Deep/_cache/")

# ═══════════════════════════════════════════════
# 2. Train/val/test split (same as ML experiments)
# ═══════════════════════════════════════════════
all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(
    all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED
)
tr_idx, val_idx = train_test_split(
    tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED
)
unl_idx = np.where(y_all == 0)[0]

# Separate train positives and unlabeled
tr_pos_mask = y_all[tr_idx] == 1
X_tr_pos_narrow = X_narrow[tr_idx][tr_pos_mask]
X_tr_unl_narrow = X_narrow[tr_idx][~tr_pos_mask]
X_tr_pos_full = X_full[tr_idx][tr_pos_mask]
X_tr_unl_full = X_full[tr_idx][~tr_pos_mask]

X_val_narrow = X_narrow[val_idx]; y_val = y_all[val_idx]
X_te_narrow = X_narrow[test_idx]; y_te = y_all[test_idx]
X_val_full = X_full[val_idx]
X_te_full = X_full[test_idx]

print(f"\nTrain: {len(tr_idx)} (pos={int(tr_pos_mask.sum())}, unl={int((~tr_pos_mask).sum())})")
print(f"Val:   {len(val_idx)} (pos={int(y_val.sum())})")
print(f"Test:  {len(test_idx)} (pos={int(y_te.sum())})")

# Standardize
ss_narrow = StandardScaler().fit(X_narrow[tr_idx])
ss_full = StandardScaler().fit(X_full[tr_idx])

for arr in [X_tr_pos_narrow, X_tr_unl_narrow, X_val_narrow, X_te_narrow,
            X_tr_pos_full, X_tr_unl_full, X_val_full, X_te_full]:
    pass  # Will standardize per-experiment to avoid mutating

X_tr_pos_n = ss_narrow.transform(X_tr_pos_narrow).astype(np.float32)
X_tr_unl_n = ss_narrow.transform(X_tr_unl_narrow).astype(np.float32)
X_val_n = ss_narrow.transform(X_val_narrow).astype(np.float32)
X_te_n = ss_narrow.transform(X_te_narrow).astype(np.float32)
X_all_n = ss_narrow.transform(X_narrow).astype(np.float32)

X_tr_pos_f = ss_full.transform(X_tr_pos_full).astype(np.float32)
X_tr_unl_f = ss_full.transform(X_tr_unl_full).astype(np.float32)
X_val_f = ss_full.transform(X_val_full).astype(np.float32)
X_te_f = ss_full.transform(X_te_full).astype(np.float32)
X_all_f = ss_full.transform(X_full).astype(np.float32)


# ═══════════════════════════════════════════════
# 3. Experiment runner
# ═══════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, X_all, device):
    """Compute all evaluation metrics."""
    trainer = PUResampleTrainer(model, device=device)
    p_te = trainer.predict(X_test)
    p_all = trainer.predict(X_all)

    roc = roc_auc_score(y_test, p_te)
    pr = average_precision_score(y_test, p_te)
    order = np.argsort(p_te)[::-1]
    p50 = y_test[order[:min(50, len(order))]].mean()
    p100 = y_test[order[:min(100, len(order))]].mean()

    # Bias
    bias_raw = compute_parameter_bias(p_all[unl_idx], df_model.iloc[unl_idx])
    z_all = compute_cluster_zscore(p_all[unl_idx], cluster_ids[unl_idx])
    z_all = np.nan_to_num(z_all, nan=0.0)
    bias_z = compute_parameter_bias(z_all, df_model.iloc[unl_idx])

    bias_teff_raw = abs(bias_raw.get("spearmanr_teff", 0))
    bias_teff_z = abs(bias_z.get("spearmanr_teff", 0))
    mean_bias_raw = np.mean([abs(bias_raw.get(f"spearmanr_{p}", 0))
                             for p in ["teff", "logg", "feh"]])
    mean_bias_z = np.mean([abs(bias_z.get(f"spearmanr_{p}", 0))
                           for p in ["teff", "logg", "feh"]])

    # Within-cluster r
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

    return {
        "ROC": roc, "PR": pr, "P@50": p50, "P@100": p100,
        "|r_teff| raw": bias_teff_raw, "|r_teff| z": bias_teff_z,
        "Mean bias raw": mean_bias_raw, "Mean bias z": mean_bias_z,
        "Within-r": within_r,
    }


def run_experiment(name, model_factory, X_tr_pos, X_tr_unl, X_val, y_val, X_te, y_te, X_all,
                   epochs=200, patience=40):
    """Train one model with PU-Resample and evaluate."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model, ep, pr, hist = PUResampleTrainer(
        model_factory(), device=DEVICE, mixup_alpha=0.2
    ).train(
        X_tr_pos, X_tr_unl, X_val, y_val,
        epochs=epochs, patience=patience, verbose=True,
    )

    metrics = evaluate_model(model, X_te, y_te, X_all, DEVICE)
    metrics["Name"] = name
    metrics["BestEpoch"] = ep
    metrics["BestValPR"] = pr
    print(f"  Result: PR={metrics['PR']:.4f}, P@100={metrics['P@100']:.4f}, "
          f"|r_teff|_z={metrics['|r_teff| z']:.4f}, within-r={metrics['Within-r']:.4f}")
    return metrics


# ═══════════════════════════════════════════════
# 4. Run experiments
# ═══════════════════════════════════════════════
print(f"\n{'#'*60}")
print("# EXPERIMENT 1: Narrow (700px) — Conv1D baseline")
print(f"{'#'*60}")

all_metrics = []

# E1a: SimpleConvNet on narrow
m = run_experiment(
    "Narrow-SimpleConv (700px)",
    lambda: SimpleConvNet(input_dim=700),
    X_tr_pos_n, X_tr_unl_n, X_val_n, y_val, X_te_n, y_te, X_all_n,
    epochs=200, patience=40,
)
all_metrics.append(m)

# E1b: ResNet+SE on narrow
m = run_experiment(
    "Narrow-ResNet+SE (700px)",
    lambda: SpectraResNet(input_dim=700, base_ch=32),
    X_tr_pos_n, X_tr_unl_n, X_val_n, y_val, X_te_n, y_te, X_all_n,
    epochs=200, patience=40,
)
all_metrics.append(m)

print(f"\n{'#'*60}")
print("# EXPERIMENT 2: Full (5000px) — ResNet+SE")
print(f"{'#'*60}")

# E2a: SimpleConvNet on full
m = run_experiment(
    "Full-SimpleConv (5000px)",
    lambda: SimpleConvNet(input_dim=5000),
    X_tr_pos_f, X_tr_unl_f, X_val_f, y_val, X_te_f, y_te, X_all_f,
    epochs=200, patience=40,
)
all_metrics.append(m)

# E2b: ResNet+SE on full
m = run_experiment(
    "Full-ResNet+SE (5000px)",
    lambda: SpectraResNet(input_dim=5000, base_ch=64),
    X_tr_pos_f, X_tr_unl_f, X_val_f, y_val, X_te_f, y_te, X_all_f,
    epochs=200, patience=40,
)
all_metrics.append(m)

# ═══════════════════════════════════════════════
# 5. Ensemble — best configuration
# ═══════════════════════════════════════════════
print(f"\n{'#'*60}")
print("# EXPERIMENT 3: Ensemble (K=10) on Full ResNet+SE")
print(f"{'#'*60}")

models_k10, histories = train_ensemble(
    lambda: SpectraResNet(input_dim=5000, base_ch=64),
    X_tr_pos_f, X_tr_unl_f, X_val_f, y_val,
    n_models=10, device=DEVICE, epochs=200, patience=30, verbose=True,
)

p_te_mean, p_te_std = ensemble_predict(models_k10, X_te_f, DEVICE)
p_all_mean, p_all_std = ensemble_predict(models_k10, X_all_f, DEVICE)

# Evaluate ensemble
roc = roc_auc_score(y_te, p_te_mean)
pr = average_precision_score(y_te, p_te_mean)
order = np.argsort(p_te_mean)[::-1]
p50 = y_te[order[:min(50, len(order))]].mean()
p100 = y_te[order[:min(100, len(order))]].mean()

bias_raw = compute_parameter_bias(p_all_mean[unl_idx], df_model.iloc[unl_idx])
z_all = compute_cluster_zscore(p_all_mean[unl_idx], cluster_ids[unl_idx])
z_all = np.nan_to_num(z_all, nan=0.0)
bias_z = compute_parameter_bias(z_all, df_model.iloc[unl_idx])

within_rs = []
for cid in np.unique(cluster_ids[unl_idx]):
    cm = cluster_ids[unl_idx] == cid
    if cm.sum() >= 10 and "delta_CN3839" in df_model.columns:
        dcn = df_model.iloc[unl_idx[cm]]["delta_CN3839"].values
        cp = p_all_mean[unl_idx][cm]
        valid = np.isfinite(dcn) & np.isfinite(cp)
        if valid.sum() >= 10:
            within_rs.append(spearmanr(dcn[valid], cp[valid])[0])
within_r = float(np.median(within_rs)) if within_rs else np.nan

ens_metrics = {
    "Name": "Full-ResNet+SE Ensemble (K=10)",
    "ROC": roc, "PR": pr, "P@50": p50, "P@100": p100,
    "|r_teff| raw": abs(bias_raw.get("spearmanr_teff", 0)),
    "|r_teff| z": abs(bias_z.get("spearmanr_teff", 0)),
    "Mean bias raw": np.mean([abs(bias_raw.get(f"spearmanr_{p}", 0))
                               for p in ["teff", "logg", "feh"]]),
    "Mean bias z": np.mean([abs(bias_z.get(f"spearmanr_{p}", 0))
                             for p in ["teff", "logg", "feh"]]),
    "Within-r": within_r,
    "BestEpoch": -1,
    "BestValPR": -1,
    "Stability": float(np.median(p_te_std)),
}
all_metrics.append(ens_metrics)
print(f"  Ensemble PR={pr:.4f}, P@100={p100:.4f}, stability={ens_metrics['Stability']:.4f}")

# ═══════════════════════════════════════════════
# 6. Summary & comparison
# ═══════════════════════════════════════════════
# Add baselines from previous experiments
baselines = [
    {"Name": "XGBoost PU-Bagging (700px) [baseline]", "PR": 0.848, "P@100": 0.100,
     "|r_teff| raw": 0.121, "|r_teff| z": 0.043, "Mean bias z": 0.071,
     "Within-r": 0.244, "Stability": 0.137},
    {"Name": "CN 9-D + Focal γ=1.5 [baseline]", "PR": 0.259, "P@100": 0.050,
     "|r_teff| raw": 0.149, "|r_teff| z": 0.024, "Mean bias z": 0.045,
     "Within-r": 0.456, "Stability": np.nan},
]

print(f"\n{'='*80}")
print("DEEP PU LEARNING — RESULTS SUMMARY")
print(f"{'='*80}")

results_df = pd.DataFrame(all_metrics + baselines)
results_df = results_df.sort_values("PR", ascending=False)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(results_df.to_string(index=False))

results_df.to_csv("Deep/pu_experiment_results.csv", index=False)
print("\nSaved to Deep/pu_experiment_results.csv")

# Save ensemble probabilities for candidate inspection
out = df_model[["teff", "logg", "feh", "label"]].copy()
out["deep_prob"] = p_all_mean
out["deep_prob_std"] = p_all_std
out.to_csv("Deep/pu_deep_probs.csv", index=False)
print("Saved probabilities to Deep/pu_deep_probs.csv")
print("Done.")
