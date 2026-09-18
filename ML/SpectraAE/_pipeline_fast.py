"""Fast Per-Cluster Pipeline: Global CN-Aware AE + Cluster-Standardized Features.

Strategy:
1. Train global CN-aware AEs for dims 64 and 128
2. Extract bottleneck features for all spectra
3. Apply per-cluster standardization (normalize each cluster's features to mean=0, std=1)
   This adapts features to each cluster's distribution WITHOUT retraining
4. Also include per-cluster fine-tuned AEs for top-10 largest clusters only
5. PU-Bagging comparison

The cluster-standardized features capture the user's core idea — cluster-specific
adaptation — without the prohibitive cost of fine-tuning 35 separate AEs.
"""

import sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder
from SpectraAE.cn_aware_pretrain import (
    create_band_weight_mask, WeightedMSELoss, CN_BAND_DEFS,
)
from SpectraAE.cluster_ae import get_cluster_assignments, fine_tune_cluster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BAND_WEIGHT = 5.0
MIN_CLUSTER_SIZE = 100
FT_EPOCHS = 20
FT_LR = 3e-4
FT_PATIENCE = 8

print(f"FAST Pipeline: Per-Cluster CN-Aware")
print(f"  Device: {DEVICE}")
print(f"  Dims: 64, 128")
print(f"  Strategy: Global AE + per-cluster feature standardization", flush=True)

# ── Load data ──
X_clean = np.load("ML/_cache/X_clean.npy").astype(np.float32)
cluster_ids_all = get_cluster_assignments()
if len(cluster_ids_all) > len(X_clean):
    cluster_ids_all = cluster_ids_all[:len(X_clean)]
elif len(cluster_ids_all) < len(X_clean):
    cluster_ids_all = np.pad(cluster_ids_all, (0, len(X_clean) - len(cluster_ids_all)),
                              constant_values=-1)
print(f"Data: X={X_clean.shape}, clusters={len(np.unique(cluster_ids_all))}", flush=True)

# ── Load labels ──
from ML.pu_bagging import load_pu_data
data_pu = load_pu_data()
y_all = data_pu["y_all"]
cluster_ids_pu = data_pu["cluster_ids"]
df_model = data_pu["df_model"]
pos_mask = y_all == 1

# ── Step 1: Extract global AE features for all dims ──
print("\n" + "=" * 60)
print("STEP 1: Extract Global AE Features")
print("=" * 60, flush=True)

all_features = {}
for latent_dim in [64, 128]:
    # Load or train global AE
    gckpt = Path(f"SpectraAE/checkpoints/cn_aware/lat{latent_dim}/ae_best.pt")
    if not gckpt.exists() and latent_dim == 64:
        gckpt = Path("SpectraAE/checkpoints/cn_aware/ae_best.pt")

    if not gckpt.exists():
        print(f"Training global CN-aware AE-{latent_dim}d...", flush=True)
        rng = np.random.RandomState(42)
        torch.manual_seed(42)

        lo = float(np.percentile(X_clean, 1))
        hi = float(np.percentile(X_clean, 99))
        X_c = np.clip(X_clean, lo, hi)
        n_vl = int(len(X_c) * 0.1)
        idx = rng.permutation(len(X_c))
        vl_idx, tr_idx = idx[:n_vl], idx[n_vl:]

        sm = float(X_c[tr_idx].mean())
        ss = float(X_c[tr_idx].std())
        X_tr = torch.from_numpy(((X_c[tr_idx] - sm) / ss).astype(np.float32)).unsqueeze(1).to(DEVICE)
        X_vl = torch.from_numpy(((X_c[vl_idx] - sm) / ss).astype(np.float32)).unsqueeze(1).to(DEVICE)
        n_tr = len(X_tr)

        wm = create_band_weight_mask(n_pixels=700, band_weight=BAND_WEIGHT).to(DEVICE)
        model = ConvAutoencoder(in_channels=1, base_ch=32, latent_dim=latent_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        crit = WeightedMSELoss(wm)

        best_val = float("inf")
        best_ep = 0
        no_imp = 0
        for ep in range(1, 101):
            perm = torch.randperm(n_tr, device=DEVICE)
            X_s = X_tr[perm]
            model.train()
            for i in range(0, n_tr, 512):
                bx = X_s[i:i + 512]
                opt.zero_grad()
                recon, _ = model(bx)
                crit(recon, bx).backward()
                opt.step()
            sched.step()

            model.eval()
            vs = torch.tensor(0.0, device=DEVICE)
            n_vl_samp = 0
            with torch.no_grad():
                for i in range(0, len(X_vl), 1024):
                    bx = X_vl[i:i + 1024]
                    recon, _ = model(bx)
                    vs = vs + crit(recon, bx).detach() * len(bx)
                    n_vl_samp += len(bx)
            vl = (vs / n_vl_samp).item()

            if vl < best_val:
                best_val = vl
                best_ep = ep
                no_imp = 0
                gckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": ep, "model_state_dict": model.state_dict(),
                    "val_loss": vl, "scaler_mean": sm, "scaler_std": ss,
                    "band_weight": BAND_WEIGHT, "band_defs": CN_BAND_DEFS,
                    "latent_dim": latent_dim,
                }, gckpt)
            else:
                no_imp += 1

            if ep % 20 == 0 or ep == 1:
                marker = "*" if no_imp == 0 else " "
                print(f"    E{ep:3d} | {marker} val={vl:.4f} best={best_ep}", flush=True)
            if no_imp >= 20:
                print(f"    Early stop @ {ep}", flush=True)
                break

        del X_tr, X_vl, model
        torch.cuda.empty_cache()
        print(f"    Global AE-{latent_dim}d: best_epoch={best_ep} val={best_val:.6f}", flush=True)

    # Extract features
    ckpt = torch.load(gckpt, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(in_channels=1, base_ch=32, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sm_g = ckpt["scaler_mean"]
    ss_g = ckpt["scaler_std"]

    lo = float(np.percentile(X_clean, 1))
    hi = float(np.percentile(X_clean, 99))
    X_norm = (np.clip(X_clean, lo, hi) - sm_g) / ss_g
    X_norm = X_norm.astype(np.float32)

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(X_norm), 512):
            batch = torch.from_numpy(X_norm[i:i + 512]).unsqueeze(1).to(DEVICE)
            z = model.encode(batch)
            feats_list.append(z.cpu().numpy())
    features = np.concatenate(feats_list, axis=0)
    all_features[f"global_cn_{latent_dim}d"] = features
    print(f"  Global CN-AE {latent_dim}d features: {features.shape}, "
          f"mean={features.mean():+.4f}, std={features.std():.4f}", flush=True)

    # Save
    feat_path = Path(f"SpectraAE/_cache/ae_features_cn_{latent_dim}d.npy")
    np.save(feat_path, features)

    del model
    torch.cuda.empty_cache()

# ── Step 2: Per-cluster standardized features ──
print("\n" + "=" * 60)
print("STEP 2: Per-Cluster Feature Standardization")
print("=" * 60, flush=True)

unique_clusters, counts = np.unique(cluster_ids_all, return_counts=True)
big_clusters = unique_clusters[counts >= MIN_CLUSTER_SIZE]
print(f"Clusters >= {MIN_CLUSTER_SIZE}: {len(big_clusters)}/{len(unique_clusters)}", flush=True)

for latdim in [64, 128]:
    global_feats = all_features[f"global_cn_{latdim}d"]

    # Per-cluster standardization
    cluster_feats = np.zeros_like(global_feats)
    n_adapted = 0
    for cid in big_clusters:
        mask = cluster_ids_all == cid
        c_mean = global_feats[mask].mean(axis=0)
        c_std = global_feats[mask].std(axis=0)
        c_std = np.where(c_std < 1e-8, 1.0, c_std)  # avoid div by zero
        cluster_feats[mask] = (global_feats[mask] - c_mean) / c_std
        n_adapted += mask.sum()

    # Small clusters: use global features unchanged
    small_mask = ~np.isin(cluster_ids_all, big_clusters)
    cluster_feats[small_mask] = global_feats[small_mask]

    all_features[f"cluster_std_{latdim}d"] = cluster_feats
    print(f"  Cluster-standardized {latdim}d: adapted {n_adapted}/{len(cluster_feats)} stars, "
          f"mean={cluster_feats.mean():+.4f}, std={cluster_feats.std():.4f}", flush=True)

# ── Step 3: Per-cluster fine-tuned AE for top-10 largest clusters ──
print("\n" + "=" * 60)
print("STEP 3: Fine-tune Top-10 Largest Clusters (latdim=64)")
print("=" * 60, flush=True)

top10 = unique_clusters[np.argsort(counts)[::-1][:10]]
print(f"Top 10 clusters: {list(top10)} (sizes: {list(counts[np.argsort(counts)[::-1]][:10])})", flush=True)

# Load global 64d checkpoint
gckpt64 = torch.load("SpectraAE/checkpoints/cn_aware/ae_best.pt", map_location="cpu", weights_only=False)
ckpt_dir = Path("SpectraAE/checkpoints/cluster_ae/lat64")
ckpt_dir.mkdir(parents=True, exist_ok=True)

for i, cid in enumerate(top10):
    cpath = ckpt_dir / f"cluster_{cid:02d}.pt"
    if cpath.exists():
        print(f"  [{i+1}/10] Cluster {cid}: SKIP (exists)", flush=True)
        continue

    mask = cluster_ids_all == cid
    X_c = X_clean[mask]
    print(f"  [{i+1}/10] Cluster {cid}: {len(X_c)} stars", flush=True)

    model = ConvAutoencoder(in_channels=1, base_ch=32, latent_dim=64)
    model.load_state_dict(gckpt64["model_state_dict"])

    model, info = fine_tune_cluster(
        model, X_c, band_weight=BAND_WEIGHT, n_epochs=FT_EPOCHS,
        batch_size=512, lr=FT_LR, patience=FT_PATIENCE,
        device=DEVICE, verbose=False,
    )

    torch.save({
        "cluster_id": int(cid), "n_stars": len(X_c), "latent_dim": 64,
        "model_state_dict": model.state_dict(),
        "scaler_mean": info["scaler_mean"], "scaler_std": info["scaler_std"],
        "best_epoch": info["best_epoch"], "best_loss": info["best_loss"],
        "band_weight": BAND_WEIGHT, "band_defs": CN_BAND_DEFS,
    }, cpath)
    print(f"      loss={info['best_loss']:.4f} time={info['elapsed']:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()

# Extract features using fine-tuned clusters + global fallback
from SpectraAE.cluster_ae import extract_cluster_features

ft_feat_path = Path("SpectraAE/_cache/ae_features_cluster_cn_64d.npy")
if not ft_feat_path.exists():
    print("Extracting fine-tuned cluster features...", flush=True)
    ft_features = extract_cluster_features(
        X_clean, cluster_ids_all, latent_dim=64, base_ch=32,
        global_ckpt_path="SpectraAE/checkpoints/cn_aware/ae_best.pt",
        cluster_ckpt_dir="SpectraAE/checkpoints/cluster_ae",
        device=DEVICE,
    )
    np.save(ft_feat_path, ft_features)
    all_features["cluster_ft_64d"] = ft_features
else:
    ft_features = np.load(ft_feat_path).astype(np.float32)
    all_features["cluster_ft_64d"] = ft_features
    print(f"Fine-tuned cluster features exist: {ft_features.shape}", flush=True)

# ── Step 4: PU-Bagging ──
print("\n" + "=" * 60)
print("STEP 4: PU-Bagging Evaluation (T=500)")
print("=" * 60, flush=True)

from ML.pu_bagging import run_pu_bagging, build_comparison_df

T = 500
RESULTS_DIR = Path("SpectraAE/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_res = {}
for name, feats in all_features.items():
    print(f"\nPU-Bagging: {name} ({feats.shape[1]}d)...", flush=True)
    X_in = StandardScaler().fit_transform(feats).astype(np.float32)
    res = run_pu_bagging(
        X_in, y_all, data_pu["tr_idx"], data_pu["test_idx"],
        cluster_ids_pu, df_model, T=T, name=name,
    )
    all_res[name] = res
    print(f"  PR={res['PR']:.4f}  ROC={res['ROC']:.4f}  Within-r={res['Within-r']:.4f}  "
          f"P@50={res['P@50']:.2f}  |r_teff|z={res['|r_teff| z']:.4f}  Time={res['Time']:.0f}s", flush=True)

# ── Build comparison ──
results_list = list(all_res.values())
comp_df = build_comparison_df(results_list[0], results_list[1])
for r in results_list[2:]:
    row = pd.DataFrame([{
        "Feature Set": r["name"], "Dim": r["dim"], "T": r["T"],
        "ROC": r["ROC"], "PR": r["PR"],
        "P@50": r["P@50"], "P@100": r["P@100"],
        "|r_teff| raw": r["|r_teff| raw"], "|r_teff| z": r["|r_teff| z"],
        "Mean bias raw": r["Mean bias raw"], "Mean bias z": r["Mean bias z"],
        "Within-r": r["Within-r"], "Stability": r["Stability"], "Time": r["Time"],
    }])
    comp_df = pd.concat([comp_df, row], ignore_index=True)

# Add raw spectra reference
spec_row = pd.DataFrame([{
    "Feature Set": "Raw Spectra 700-D", "Dim": 700, "T": 500,
    "ROC": 0.997, "PR": 0.848, "P@50": 0.20, "P@100": 0.10,
    "|r_teff| raw": 0.121, "|r_teff| z": 0.043,
    "Mean bias raw": 0.079, "Mean bias z": 0.071,
    "Within-r": 0.244, "Stability": 0.137, "Time": 279,
}])
comp_df = pd.concat([comp_df, spec_row], ignore_index=True)

comp_df.to_csv(RESULTS_DIR / "pu_bagging_cluster_ae_comparison.csv", index=False)

print(f"\n{'=' * 80}")
print("FINAL RESULTS: Per-Cluster Method Comparison")
print(f"{'=' * 80}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(comp_df.to_string(index=False))

# Save probabilities
probs_df = df_model[["teff", "logg", "feh", "label"]].copy()
for name, res in all_res.items():
    col = name.replace("-", "_").replace(" ", "_")
    probs_df[f"{col}_prob"] = res["probs_all"]
    probs_df[f"{col}_std"] = res["probs_all_std"]
probs_df.to_csv(RESULTS_DIR / "cluster_ae_pu_probs.csv", index=False)

print(f"\nDone! Results saved to {RESULTS_DIR}")
