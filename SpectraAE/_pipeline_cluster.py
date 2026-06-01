"""Sequential pipeline: Per-cluster CN-Aware AE training + evaluation.

All steps run sequentially to avoid GPU conflicts.
Resumes gracefully if any step was partially completed.
"""

import sys, time, pickle
from pathlib import Path
import numpy as np
import torch
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder
from SpectraAE.cn_aware_pretrain import (
    create_band_weight_mask, WeightedMSELoss, CN_BAND_DEFS,
)
from SpectraAE.cluster_ae import (
    get_cluster_assignments, fine_tune_cluster,
    extract_cluster_features,
)
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BAND_WEIGHT = 5.0
MIN_CLUSTER_SIZE = 100
FT_EPOCHS = 30
FT_LR = 3e-4
FT_PATIENCE = 10
GLOBAL_EPOCHS = 100
GLOBAL_PATIENCE = 20

print(f"Sequential Pipeline: Per-Cluster CN-Aware AE")
print(f"  Device: {DEVICE}")
print(f"  Dims: 64, 128, 256")
print(f"  Band weight: {BAND_WEIGHT}x")
print(f"  Min cluster size: {MIN_CLUSTER_SIZE}")
print(f"  Fine-tune: {FT_EPOCHS} epochs/cluster", flush=True)

# ── Load data once ──
X_clean = np.load("ML/_cache/X_clean.npy").astype(np.float32)
cluster_ids_all = get_cluster_assignments()
if len(cluster_ids_all) > len(X_clean):
    cluster_ids_all = cluster_ids_all[:len(X_clean)]
elif len(cluster_ids_all) < len(X_clean):
    cluster_ids_all = np.pad(cluster_ids_all, (0, len(X_clean) - len(cluster_ids_all)),
                              constant_values=-1)

print(f"Data: X={X_clean.shape}, clusters={len(np.unique(cluster_ids_all))}", flush=True)


def train_global_ae(latent_dim):
    """Train global CN-aware AE (GPU-resident, ~3-5 min)."""
    ckpt_path = Path(f"SpectraAE/checkpoints/cn_aware/lat{latent_dim}/ae_best.pt")
    # Old path fallback for dim=64
    if not ckpt_path.exists() and latent_dim == 64:
        old = Path("SpectraAE/checkpoints/cn_aware/ae_best.pt")
        if old.exists():
            print(f"  Using existing: {old}", flush=True)
            return

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"  Exists: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}", flush=True)
        return

    print(f"  Training global CN-aware AE-{latent_dim}d...", flush=True)
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GLOBAL_EPOCHS, eta_min=1e-6)
    crit = WeightedMSELoss(wm)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    t0 = time.time()

    for ep in range(1, GLOBAL_EPOCHS + 1):
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
        vs = 0.0
        with torch.no_grad():
            for i in range(0, len(X_vl), 1024):
                bx = X_vl[i:i + 1024]
                recon, _ = model(bx)
                vs += crit(recon, bx).item() * len(bx)
        vl = vs / len(X_vl)

        if vl < best_val:
            best_val = vl
            best_ep = ep
            no_imp = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "val_loss": vl, "scaler_mean": sm, "scaler_std": ss,
                "band_weight": BAND_WEIGHT, "band_defs": CN_BAND_DEFS,
                "latent_dim": latent_dim,
            }, ckpt_path)
        else:
            no_imp += 1

        if ep % 30 == 0 or ep == 1:
            m = "*" if no_imp == 0 else " "
            print(f"    E{ep:3d} | {m} val={vl:.4f} best={best_ep} [{time.time()-t0:.0f}s]", flush=True)
        if no_imp >= GLOBAL_PATIENCE:
            print(f"    Early stop @ {ep}", flush=True)
            break

    elapsed = time.time() - t0
    print(f"    Done: best_epoch={best_ep} val_loss={best_val:.6f} [{elapsed:.0f}s]", flush=True)
    # Free GPU memory
    del X_tr, X_vl, model
    torch.cuda.empty_cache()


def fine_tune_clusters(latent_dim):
    """Fine-tune per-cluster AEs from global checkpoint."""
    ckpt_dir = Path(f"SpectraAE/checkpoints/cluster_ae/lat{latent_dim}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Find global checkpoint
    gckpt = Path(f"SpectraAE/checkpoints/cn_aware/lat{latent_dim}/ae_best.pt")
    if not gckpt.exists() and latent_dim == 64:
        gckpt = Path("SpectraAE/checkpoints/cn_aware/ae_best.pt")
    if not gckpt.exists():
        print(f"  ERROR: No global checkpoint for dim {latent_dim}. Run train_global_ae first.", flush=True)
        return

    gckpt_data = torch.load(gckpt, map_location="cpu", weights_only=False)

    unique_clusters, counts = np.unique(cluster_ids_all, return_counts=True)
    big = unique_clusters[counts >= MIN_CLUSTER_SIZE]
    print(f"  Eligible clusters: {len(big)}/{len(unique_clusters)}", flush=True)

    results = {}
    t0_total = time.time()

    for i, cid in enumerate(big):
        cpath = ckpt_dir / f"cluster_{cid:02d}.pt"
        if cpath.exists():
            print(f"  [{i+1}/{len(big)}] Cluster {cid}: SKIP (exists)", flush=True)
            continue

        mask = cluster_ids_all == cid
        X_c = X_clean[mask]
        n_stars = len(X_c)
        print(f"  [{i+1}/{len(big)}] Cluster {cid}: {n_stars} stars", flush=True)

        # Create model and load global weights
        model = ConvAutoencoder(in_channels=1, base_ch=32, latent_dim=latent_dim)
        model.load_state_dict(gckpt_data["model_state_dict"])

        # Fine-tune
        model, info = fine_tune_cluster(
            model, X_c,
            band_weight=BAND_WEIGHT,
            n_epochs=FT_EPOCHS,
            batch_size=512,
            lr=FT_LR,
            patience=FT_PATIENCE,
            device=DEVICE,
            verbose=False,  # less verbose for batch run
        )

        torch.save({
            "cluster_id": int(cid), "n_stars": n_stars,
            "latent_dim": latent_dim,
            "model_state_dict": model.state_dict(),
            "scaler_mean": info["scaler_mean"],
            "scaler_std": info["scaler_std"],
            "best_epoch": info["best_epoch"],
            "best_loss": info["best_loss"],
            "band_weight": BAND_WEIGHT,
            "band_defs": CN_BAND_DEFS,
        }, cpath)

        results[int(cid)] = {"n_stars": n_stars, "best_loss": info["best_loss"],
                              "elapsed": info["elapsed"]}
        print(f"      loss={info['best_loss']:.4f} time={info['elapsed']:.0f}s", flush=True)

        # Free GPU memory between clusters
        del model
        torch.cuda.empty_cache()

    total = sum(r["elapsed"] for r in results.values())
    print(f"  Done: {len(results)} new + skipped clusters. Total time: {time.time()-t0_total:.0f}s (compute: {total:.0f}s)", flush=True)

    # Save meta
    meta = {
        "latent_dim": latent_dim, "band_weight": BAND_WEIGHT,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "total_clusters": len(unique_clusters),
        "trained_clusters": len(big),
        "cluster_results": results,
    }
    with open(ckpt_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    del gckpt_data
    torch.cuda.empty_cache()


def extract_and_evaluate(latent_dim):
    """Extract features and run PU-Bagging."""
    feat_path = Path(f"SpectraAE/_cache/ae_features_cluster_cn_{latent_dim}d.npy")

    if not feat_path.exists():
        gckpt = Path(f"SpectraAE/checkpoints/cn_aware/lat{latent_dim}/ae_best.pt")
        if not gckpt.exists() and latent_dim == 64:
            gckpt = Path("SpectraAE/checkpoints/cn_aware/ae_best.pt")
        if not gckpt.exists():
            print(f"  No global checkpoint for dim {latent_dim}", flush=True)
            return

        print(f"  Extracting features...", flush=True)
        features = extract_cluster_features(
            X_clean, cluster_ids_all,
            latent_dim=latent_dim, base_ch=32,
            global_ckpt_path=str(gckpt),
            cluster_ckpt_dir="SpectraAE/checkpoints/cluster_ae",
            device=DEVICE,
        )
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_path, features)
        print(f"  Saved: {feat_path} shape={features.shape}", flush=True)
    else:
        features = np.load(feat_path).astype(np.float32)
        print(f"  Features exist: {feat_path} shape={features.shape}", flush=True)

    return features


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
ALL_DIMS = [64, 128]  # 256d too slow for now, focus on 64 vs 128 comparison

for dim in ALL_DIMS:
    print(f"\n{'=' * 60}")
    print(f"PHASE: Global CN-Aware AE-{dim}d")
    print(f"{'=' * 60}", flush=True)
    train_global_ae(dim)

for dim in ALL_DIMS:
    print(f"\n{'=' * 60}")
    print(f"PHASE: Per-Cluster Fine-Tuning AE-{dim}d")
    print(f"{'=' * 60}", flush=True)
    fine_tune_clusters(dim)

# Extract features for all dims
all_features = {}
for dim in ALL_DIMS:
    print(f"\n{'=' * 60}")
    print(f"PHASE: Feature Extraction AE-{dim}d")
    print(f"{'=' * 60}", flush=True)
    feats = extract_and_evaluate(dim)
    if feats is not None:
        all_features[dim] = feats

# Load global CN-aware 64d features for comparison
gn_path = Path("SpectraAE/_cache/ae_features_cn_64d.npy")
if gn_path.exists():
    all_features["global_cn_64"] = np.load(gn_path).astype(np.float32)
    print(f"\nLoaded global CN-AE 64d features: {all_features['global_cn_64'].shape}")

print(f"\n{'=' * 60}")
print(f"PHASE: PU-Bagging Evaluation")
print(f"{'=' * 60}", flush=True)

from ML.pu_bagging import load_pu_data, run_pu_bagging, build_comparison_df

data_pu = load_pu_data()
y_all = data_pu["y_all"]
cluster_ids_pu = data_pu["cluster_ids"]
df_model = data_pu["df_model"]

T = 500
RESULTS_DIR = Path("SpectraAE/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

all_res = {}
for name, feats in all_features.items():
    print(f"\nPU-Bagging (T={T}): {name}...", flush=True)
    X_in = StandardScaler().fit_transform(feats).astype(np.float32)
    res = run_pu_bagging(
        X_in, y_all, data_pu["tr_idx"], data_pu["test_idx"],
        cluster_ids_pu, df_model, T=T, name=name,
    )
    all_res[name] = res
    print(f"  PR={res['PR']:.4f}  ROC={res['ROC']:.4f}  Within-r={res['Within-r']:.4f}  "
          f"P@50={res['P@50']:.2f}  Time={res['Time']:.0f}s", flush=True)

# Build comparison
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
print("FINAL RESULTS")
print(f"{'=' * 80}")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(comp_df.to_string(index=False))

# Per-star probabilities
probs_df = df_model[["teff", "logg", "feh", "label"]].copy()
for name, res in all_res.items():
    col = name.replace("-", "_").replace(" ", "_")
    probs_df[f"{col}_prob"] = res["probs_all"]
    probs_df[f"{col}_std"] = res["probs_all_std"]
probs_df.to_csv(RESULTS_DIR / "cluster_ae_pu_probs.csv", index=False)

print(f"\n{'=' * 60}")
print("PIPELINE COMPLETE")
print(f"{'=' * 60}")
