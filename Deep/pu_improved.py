"""Improved Deep PU: larger negative pool + augmentation + full 5000px spectra."""
import sys, time, warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from Deep.models.resnet1d import SpectraResNet, SimpleConvNet

sys.path.insert(0, str(_PROJECT_ROOT / "ML"))
from utils import compute_cluster_zscore, compute_parameter_bias

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════
CACHE_ML = Path("ML/_cache")
CACHE_DEEP = Path("Deep/_cache")

feature_df = pd.read_pickle(CACHE_ML / "feature_df.pkl")
stars_clean = pd.read_pickle(CACHE_ML / "stars_clustered.pkl")
df_model = feature_df.dropna(subset=["label"]).copy()

X_all = np.load(CACHE_DEEP / "X_full_5000.npy").astype(np.float32)
n_full = len(X_all)
# Align: full pipeline drops ~8 stars due to coverage, truncate labels to match
y_all = df_model["label"].map({1: 1, -1: 0}).values.astype(np.float32)[:n_full]
cluster_ids = stars_clean["masked_cluster_id"].values.astype(np.int64)[:n_full]
df_model = df_model.iloc[:n_full]
print(f"Full spectra: {X_all.shape}, positives={int(y_all.sum())}")

# Split
all_idx = np.arange(len(y_all))
tv_idx, test_idx = train_test_split(all_idx, test_size=0.15, stratify=y_all, random_state=RANDOM_SEED)
tr_idx, val_idx = train_test_split(tv_idx, test_size=0.18, stratify=y_all[tv_idx], random_state=RANDOM_SEED)
unl_idx = np.where(y_all == 0)[0]

tr_pos_mask = y_all[tr_idx] == 1
X_tr_pos = X_all[tr_idx][tr_pos_mask]
X_tr_unl = X_all[tr_idx][~tr_pos_mask]
n_pos = len(X_tr_pos)

X_val = X_all[val_idx]; y_val = y_all[val_idx]
X_te = X_all[test_idx]; y_te = y_all[test_idx]

# Standardize
ss = StandardScaler().fit(X_all[tr_idx])
X_tr_pos = ss.transform(X_tr_pos).astype(np.float32)
X_tr_unl = ss.transform(X_tr_unl).astype(np.float32)
X_val = ss.transform(X_val).astype(np.float32)
X_te = ss.transform(X_te).astype(np.float32)
X_all_s = ss.transform(X_all).astype(np.float32)

print(f"Train pos={n_pos}, unl={len(X_tr_unl)}, val={len(X_val)} (pos={int(y_val.sum())}), test={len(X_te)} (pos={int(y_te.sum())})")

# ═══════════════════════════════════════════════
# 2. Improved Trainer
# ═══════════════════════════════════════════════
class ImprovedPUTrainer:
    def __init__(self, model, device, n_neg=500, noise_std=0.02, mixup_alpha=0.3):
        self.model = model
        self.device = device
        self.n_neg = n_neg
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha

    def _augment(self, X_neg):
        if self.noise_std > 0:
            X_neg = X_neg + torch.randn_like(X_neg) * self.noise_std
        if self.mixup_alpha > 0 and len(X_neg) >= 2:
            perm = torch.randperm(len(X_neg))
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((len(X_neg) // 2,))
            for i in range(len(lam)):
                j = i * 2
                if j + 1 < len(X_neg):
                    lam_i = lam[i].to(X_neg.device)
                    X_neg[perm[j]] = lam_i * X_neg[perm[j]] + (1 - lam_i) * X_neg[perm[j + 1]]
        return X_neg

    @torch.no_grad()
    def predict(self, X, batch_size=512):
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            preds.append(self.model(xb).cpu().numpy())  # model returns sigmoid probs
        return np.concatenate(preds)

    def train(self, X_pos, X_unl, X_val_np, y_val_np, epochs=300, lr=1e-3, patience=60):
        model = self.model.to(self.device)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        pos_weight = torch.tensor([self.n_neg / len(X_pos)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_pos_t = torch.from_numpy(X_pos).float()
        rng = np.random.RandomState(RANDOM_SEED)
        best_val_pr = -np.inf
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            neg_idx = rng.choice(len(X_unl), size=min(self.n_neg, len(X_unl)), replace=False)
            X_neg_t = torch.from_numpy(X_unl[neg_idx]).float().to(self.device)
            X_neg_t = self._augment(X_neg_t)

            X_batch = torch.cat([X_pos_t.to(self.device), X_neg_t], dim=0)
            y_batch = torch.cat([torch.ones(len(X_pos_t)), torch.zeros(len(X_neg_t))]).to(self.device)
            perm = torch.randperm(len(y_batch))
            X_batch, y_batch = X_batch[perm], y_batch[perm]

            model.train()
            opt.zero_grad()
            logits = model(X_batch, return_logits=True)
            loss = criterion(logits, y_batch)
            loss.backward()
            opt.step()
            sched.step()

            # Validate on val PR (early stop on PR, not loss)
            val_preds = self.predict(X_val_np)
            val_pr = average_precision_score(y_val_np, val_preds)

            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 50 == 0:
                print(f"    Ep {epoch:3d} | loss={loss.item():.4f} | "
                      f"val_pr={val_pr:.4f} | best={best_val_pr:.4f} | lr={sched.get_last_lr()[0]:.2e}")

            if no_improve >= patience:
                print(f"    Early stop @ {epoch} (best_val_pr={best_val_pr:.4f})")
                break

        model.load_state_dict(best_state)
        model.eval()
        return model, best_val_pr


# ═══════════════════════════════════════════════
# 3. Evaluate
# ═══════════════════════════════════════════════
def evaluate(name, model, X_te_np, y_te_np, X_all_np):
    t = ImprovedPUTrainer(model, DEVICE)
    p_te = t.predict(X_te_np)
    p_all = t.predict(X_all_np)

    pr = average_precision_score(y_te_np, p_te)
    roc = roc_auc_score(y_te_np, p_te)
    order = np.argsort(p_te)[::-1]
    p50 = y_te_np[order[:min(50, len(order))]].mean()
    p100 = y_te_np[order[:min(100, len(order))]].mean()

    bias_raw = compute_parameter_bias(p_all[unl_idx], df_model.iloc[unl_idx])
    z_all = compute_cluster_zscore(p_all[unl_idx], cluster_ids[unl_idx])
    z_all = np.nan_to_num(z_all, nan=0.0)
    bias_z = compute_parameter_bias(z_all, df_model.iloc[unl_idx])

    within_rs = []
    for cid in np.unique(cluster_ids[unl_idx]):
        cm = cluster_ids[unl_idx] == cid
        if cm.sum() >= 10 and "delta_CN3839" in df_model.columns:
            dcn = df_model.iloc[unl_idx][cm]["delta_CN3839"].values
            cp = p_all[unl_idx][cm]
            valid = np.isfinite(dcn) & np.isfinite(cp)
            if valid.sum() >= 10:
                within_rs.append(spearmanr(dcn[valid], cp[valid])[0])
    within_r = float(np.median(within_rs)) if within_rs else np.nan

    print(f"\n  {name}:")
    print(f"    PR={pr:.4f}  ROC={roc:.4f}  P@50={p50:.4f}  P@100={p100:.4f}")
    print(f"    |r_teff| raw={abs(bias_raw.get('spearmanr_teff',0)):.4f}  "
          f"|r_teff| z={abs(bias_z.get('spearmanr_teff',0)):.4f}")
    print(f"    Mean bias z={np.mean([abs(bias_z.get(f'spearmanr_{p}',0)) for p in ['teff','logg','feh']]):.4f}")
    print(f"    Within-r={within_r:.4f}")

    return {"Name": name, "PR": pr, "ROC": roc, "P@50": p50, "P@100": p100,
            "|r_teff| raw": abs(bias_raw.get("spearmanr_teff", 0)),
            "|r_teff| z": abs(bias_z.get("spearmanr_teff", 0)),
            "Mean bias z": np.mean([abs(bias_z.get(f"spearmanr_{p}", 0))
                                     for p in ["teff", "logg", "feh"]]),
            "Within-r": within_r}


# ═══════════════════════════════════════════════
# 4. Experiments
# ═══════════════════════════════════════════════
all_results = []

for n_neg in [500, 1000, 2000]:
    print(f"\n{'='*60}")
    print(f"  NEGATIVE POOL = {n_neg}")
    print(f"{'='*60}")

    # ResNet+SE
    print("  [ResNet+SE]")
    t0 = time.time()
    model_r = SpectraResNet(input_dim=5000, base_ch=64)
    trainer_r = ImprovedPUTrainer(model_r, DEVICE, n_neg=n_neg, noise_std=0.02, mixup_alpha=0.3)
    model_r, val_pr_r = trainer_r.train(X_tr_pos, X_tr_unl, X_val, y_val, epochs=300, patience=60)
    r = evaluate(f"ResNet+SE n={n_neg}", model_r, X_te, y_te, X_all_s)
    r["Time"] = time.time() - t0
    all_results.append(r)

    # SimpleConvNet
    print("  [SimpleConvNet]")
    t0 = time.time()
    model_c = SimpleConvNet(input_dim=5000)
    trainer_c = ImprovedPUTrainer(model_c, DEVICE, n_neg=n_neg, noise_std=0.02, mixup_alpha=0.3)
    model_c, val_pr_c = trainer_c.train(X_tr_pos, X_tr_unl, X_val, y_val, epochs=300, patience=60)
    r = evaluate(f"SimpleConv n={n_neg}", model_c, X_te, y_te, X_all_s)
    r["Time"] = time.time() - t0
    all_results.append(r)

# ═══════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════
baselines = [
    {"Name": "XGBoost PU-Bagging 700px [best]", "PR": 0.848, "P@100": 0.100,
     "|r_teff| raw": 0.121, "|r_teff| z": 0.043, "Mean bias z": 0.071, "Within-r": 0.244},
    {"Name": "Deep ResNet 700px n=73 [old]", "PR": 0.413, "P@100": 0.080,
     "|r_teff| raw": 0.031, "|r_teff| z": 0.007, "Mean bias z": 0.030, "Within-r": 0.093},
]

print(f"\n{'='*80}")
print("IMPROVED DEEP PU — FINAL RESULTS (Full 5000px Spectra)")
print(f"{'='*80}")

results_df = pd.DataFrame(all_results + baselines).sort_values("PR", ascending=False)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(results_df.to_string(index=False))

results_df.to_csv("Deep/pu_improved_results.csv", index=False)
print("\nSaved to Deep/pu_improved_results.csv")
print("Done.")
