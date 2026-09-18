"""Train per-cluster CN-aware AEs for multiple latent dimensions.

Phase 1: Train global CN-aware AEs for dims 64/128/256 (skip 64 if exists)
Phase 2: Fine-tune each cluster from global checkpoint
Phase 3: Extract cluster-conditioned features
Phase 4: Quick validation

Usage:
    python SpectraAE/_train_cluster_ae.py --dims 64,128,256
    python SpectraAE/_train_cluster_ae.py --dims 64  # quick test
"""

import sys, time, pickle
from pathlib import Path
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder
from SpectraAE.cn_aware_pretrain import (
    cn_aware_pretrain_autoencoder, create_band_weight_mask,
    WeightedMSELoss, CN_BAND_DEFS,
)
from SpectraAE.cluster_ae import (
    get_cluster_assignments, fine_tune_cluster,
    extract_cluster_features, run_per_cluster_training,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BAND_WEIGHT = 5.0
MIN_CLUSTER_SIZE = 100
FT_EPOCHS = 50
FT_PATIENCE = 15
FT_LR = 3e-4
BATCH_SIZE = 512


def _get_global_ckpt_path(latent_dim: int) -> Path:
    """Resolve global CN-aware AE checkpoint path for a given latent dim."""
    # New path format: SpectraAE/checkpoints/cn_aware/lat64/ae_best.pt
    new_path = Path(f"SpectraAE/checkpoints/cn_aware/lat{latent_dim}/ae_best.pt")
    if new_path.exists():
        return new_path
    # Old path format (only for dim=64): SpectraAE/checkpoints/cn_aware/ae_best.pt
    old_path = Path("SpectraAE/checkpoints/cn_aware/ae_best.pt")
    if latent_dim == 64 and old_path.exists():
        return old_path
    return new_path  # return new path even if non-existent (for training)


def train_global_cn_ae(latent_dim: int):
    """Train a global CN-aware AE for a given latent dimension.

    Uses GPU-resident training for speed (~3-5 min per model).
    """
    ckpt_path = _get_global_ckpt_path(latent_dim)
    if ckpt_path.exists():
        print(f"  Global checkpoint exists: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"    epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', float('nan')):.6f}")
        return

    print(f"  Training global CN-aware AE-{latent_dim}d...", flush=True)
    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    print(f"  Loaded X_clean: {X.shape}")

    # GPU-resident training for speed
    rng = np.random.RandomState(42)
    torch.manual_seed(42)

    lo = float(np.percentile(X, 1))
    hi = float(np.percentile(X, 99))
    X_clipped = np.clip(X, lo, hi)
    n_total = len(X_clipped)
    n_val = int(n_total * 0.1)
    indices = rng.permutation(n_total)
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]
    scaler_mean = float(X_clipped[tr_idx].mean())
    scaler_std = float(X_clipped[tr_idx].std())

    X_train = torch.from_numpy(
        ((X_clipped[tr_idx] - scaler_mean) / scaler_std).astype(np.float32)
    ).unsqueeze(1).to(DEVICE)
    X_val = torch.from_numpy(
        ((X_clipped[val_idx] - scaler_mean) / scaler_std).astype(np.float32)
    ).unsqueeze(1).to(DEVICE)
    n_tr, n_vl = len(X_train), len(X_val)

    weight_mask = create_band_weight_mask(n_pixels=700, band_weight=BAND_WEIGHT).to(DEVICE)
    band_mask_binary = (weight_mask > 1.1).float()
    n_band_px = int(band_mask_binary.sum().item())

    model = ConvAutoencoder(in_channels=1, base_ch=32, latent_dim=latent_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params  latent={latent_dim}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6,
    )
    criterion = WeightedMSELoss(weight_mask)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, 201):
        perm = torch.randperm(n_tr, device=DEVICE)
        X_tr_s = X_train[perm]

        model.train()
        tr_sum = 0.0
        for i in range(0, n_tr, BATCH_SIZE):
            batch_x = X_tr_s[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * len(batch_x)

        scheduler.step()

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for i in range(0, n_vl, BATCH_SIZE * 2):
                batch_x = X_val[i:i + BATCH_SIZE * 2]
                recon, _ = model(batch_x)
                val_sum += criterion(recon, batch_x).item() * len(batch_x)
        val_loss = val_sum / n_vl

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "scaler_mean": scaler_mean,
                "scaler_std": scaler_std, "band_weight": BAND_WEIGHT,
                "band_defs": CN_BAND_DEFS, "latent_dim": latent_dim,
            }, ckpt_path)
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            marker = "*" if no_improve == 0 else " "
            print(f"    E{epoch:3d} | {marker} val={val_loss:.4f}  "
                  f"best={best_epoch}  [{elapsed:.0f}s]", flush=True)

        if no_improve >= 30:
            print(f"    Early stop @ {epoch}", flush=True)
            break

    elapsed = time.time() - t0
    print(f"    Global AE-{latent_dim}d: best_epoch={best_epoch}  "
          f"val_loss={best_val_loss:.6f}  time={elapsed:.0f}s", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=str, default="64,128,256",
                        help="Comma-separated latent dimensions")
    parser.add_argument("--skip-global", action="store_true",
                        help="Skip global AE training")
    parser.add_argument("--skip-cluster", action="store_true",
                        help="Skip cluster fine-tuning")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features (requires existing checkpoints)")
    args = parser.parse_args()

    dims = [int(d.strip()) for d in args.dims.split(",")]
    print(f"Per-Cluster CN-Aware AE Pipeline")
    print(f"  Dimensions: {dims}")
    print(f"  Band weight: {BAND_WEIGHT}x")
    print(f"  Min cluster size: {MIN_CLUSTER_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Fine-tune: {FT_EPOCHS} epochs/cluster, lr={FT_LR}, patience={FT_PATIENCE}")

    # ── Phase 1: Global CN-aware AEs ──────────────────────────
    if not args.skip_global and not args.extract_only:
        print(f"\n{'=' * 60}")
        print(f"PHASE 1: Global CN-aware AE Training")
        print(f"{'=' * 60}", flush=True)
        for d in dims:
            print(f"\n--- Global AE-{d}d ---", flush=True)
            train_global_cn_ae(d)

    # ── Phase 2: Per-cluster fine-tuning ──────────────────────
    if not args.skip_cluster and not args.extract_only:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: Per-Cluster Fine-Tuning")
        print(f"{'=' * 60}", flush=True)
        for d in dims:
            global_ckpt = str(_get_global_ckpt_path(d))
            if not Path(global_ckpt).exists():
                print(f"ERROR: Global checkpoint not found for lat{d} at {global_ckpt}. "
                       f"Run without --skip-global first.")
                continue
            print(f"\n--- Cluster fine-tuning: AE-{d}d (global: {global_ckpt}) ---", flush=True)
            run_per_cluster_training(
                latent_dim=d,
                band_weight=BAND_WEIGHT,
                min_cluster_size=MIN_CLUSTER_SIZE,
                n_epochs=FT_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=FT_LR,
                patience=FT_PATIENCE,
                device=DEVICE,
                global_ckpt_path=global_ckpt,
            )

    # ── Phase 3: Feature extraction ───────────────────────────
    print(f"\n{'=' * 60}")
    print(f"PHASE 3: Feature Extraction")
    print(f"{'=' * 60}", flush=True)

    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    cluster_ids = get_cluster_assignments()
    if len(cluster_ids) > len(X):
        cluster_ids = cluster_ids[:len(X)]
    elif len(cluster_ids) < len(X):
        cluster_ids = np.pad(cluster_ids, (0, len(X) - len(cluster_ids)),
                              constant_values=-1)

    for d in dims:
        global_ckpt = str(_get_global_ckpt_path(d))
        if not Path(global_ckpt).exists():
            print(f"WARNING: No global ckpt for lat{d}, skipping feature extraction.")
            continue

        feat_path = Path(f"SpectraAE/_cache/ae_features_cluster_cn_{d}d.npy")
        if feat_path.exists():
            feats = np.load(feat_path)
            print(f"  Features already exist: {feat_path}  shape={feats.shape}")
            continue

        print(f"\n  Extracting cluster features for AE-{d}d...", flush=True)
        features = extract_cluster_features(
            X, cluster_ids,
            latent_dim=d,
            base_ch=32,
            global_ckpt_path=global_ckpt,
            cluster_ckpt_dir="SpectraAE/checkpoints/cluster_ae",
            device=DEVICE,
        )
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_path, features)
        print(f"  Saved to {feat_path}")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
