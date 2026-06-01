"""Per-Cluster CN-Aware Autoencoder Fine-Tuning.

Key insight: Different stellar clusters have different spectral characteristics
(Teff/logg/Feh distributions, CN abundance baselines). Training one global AE
forces a "compromise" representation that may miss cluster-specific CN signatures.

This module fine-tunes a global CN-aware AE on each cluster separately, producing
cluster-conditioned encoders that better capture within-cluster CN variations.

Workflow:
    1. Start from global CN-aware AE checkpoint (transfer learning)
    2. Fine-tune on each cluster with >= min_cluster_size stars
    3. Extract features using per-cluster encoders (fallback to global for small clusters)
    4. PU-Bagging on cluster-conditioned features

Usage:
    python SpectraAE/cluster_ae.py --latent-dim 64 --min-size 100
    python SpectraAE/cluster_ae.py --latent-dim 128 --min-size 100
"""

import sys, time, pickle, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder
from SpectraAE.cn_aware_pretrain import (
    create_band_weight_mask, WeightedMSELoss, CN_BAND_DEFS,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_cluster_assignments() -> Tuple[np.ndarray, np.ndarray]:
    """Load cluster IDs for all stars.

    Returns
    -------
    cluster_ids : np.ndarray, shape (N,)
        Integer cluster ID per star (-1 for unassigned).
    cluster_sizes : np.ndarray, shape (n_clusters,)
        Number of stars per cluster ID (indexed by cluster_id).
    """
    import pandas as pd
    stars = pd.read_pickle("ML/_cache/stars_clustered.pkl")
    cluster_ids = stars["masked_cluster_id"].values.astype(np.int32)

    # Verify alignment with X_clean
    X = np.load("ML/_cache/X_clean.npy")
    if len(cluster_ids) != len(X):
        print(f"WARNING: cluster_ids ({len(cluster_ids)}) != X_clean ({len(X)}). "
              f"Padding cluster_ids with -1.")

    return cluster_ids


def fine_tune_cluster(
    model: ConvAutoencoder,
    X_cluster: np.ndarray,
    band_weight: float = 5.0,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    patience: int = 15,
    device: str = DEFAULT_DEVICE,
    verbose: bool = True,
) -> Tuple[ConvAutoencoder, Dict]:
    """Fine-tune a pre-trained CN-aware AE on a single cluster's spectra.

    Uses GPU-resident training for speed.

    Parameters
    ----------
    model : ConvAutoencoder
        Pre-trained global CN-aware AE (weights will be updated in-place).
    X_cluster : np.ndarray, shape (n_cluster, 700)
        Spectra belonging to this cluster.
    band_weight : float
        CN band pixel weight multiplier.
    n_epochs : int
        Maximum fine-tuning epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for fine-tuning (lower than pretraining).
    patience : int
        Early stopping patience.
    device : str
        PyTorch device.
    verbose : bool
        Print progress.

    Returns
    -------
    model : ConvAutoencoder
        Fine-tuned model (best checkpoint restored, in eval mode).
    info : dict
        Training summary.
    """
    n_stars = len(X_cluster)
    if n_stars < batch_size:
        batch_size = max(n_stars, 16)

    torch.manual_seed(42)

    # Normalize
    lo = float(np.percentile(X_cluster, 1))
    hi = float(np.percentile(X_cluster, 99))
    X_clip = np.clip(X_cluster, lo, hi)
    scaler_mean = float(X_clip.mean())
    scaler_std = float(X_clip.std())
    X_norm = ((X_clip - scaler_mean) / scaler_std).astype(np.float32)

    # GPU-resident data
    X_t = torch.from_numpy(X_norm).unsqueeze(1).to(device)
    n_train = len(X_t)

    # Weight mask
    weight_mask = create_band_weight_mask(n_pixels=700, band_weight=band_weight).to(device)
    band_mask_binary = (weight_mask > 1.1).float()
    n_band_px = int(band_mask_binary.sum().item())

    model = model.to(device)
    model.train()

    criterion = WeightedMSELoss(weight_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    no_improve = 0
    losses = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Shuffle on GPU (fast)
        perm = torch.randperm(n_train, device=device)
        X_shuffled = X_t[perm]

        model.train()
        tr_sum = torch.tensor(0.0, device=device)
        n_samples = 0
        for i in range(0, n_train, batch_size):
            batch_x = X_shuffled[i:i + batch_size]
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            # Accumulate on GPU, only sync once per epoch
            tr_sum = tr_sum + loss.detach() * len(batch_x)
            n_samples += len(batch_x)

        # Single CPU sync per epoch
        avg_loss = (tr_sum / n_samples).item()
        losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improve = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if no_improve >= patience:
            if verbose:
                print(f"    Early stop @ {epoch} (best={best_loss:.6f} @ {best_epoch})", flush=True)
            break

    # Restore best
    model.load_state_dict(best_state_dict)
    model.eval()

    elapsed = time.time() - t0
    info = {
        "n_stars": n_stars, "n_epochs_run": epoch,
        "best_epoch": best_epoch, "best_loss": best_loss,
        "scaler_mean": scaler_mean, "scaler_std": scaler_std,
        "elapsed": elapsed, "band_weight": band_weight,
        "losses": losses,
    }
    return model, info


def run_per_cluster_training(
    latent_dim: int = 64,
    base_ch: int = 32,
    band_weight: float = 5.0,
    min_cluster_size: int = 100,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    patience: int = 15,
    device: str = DEFAULT_DEVICE,
    global_ckpt_path: str = "SpectraAE/checkpoints/cn_aware/ae_best.pt",
    output_dir: str = "SpectraAE/checkpoints/cluster_ae",
) -> Dict[int, Dict]:
    """Fine-tune CN-aware AE on each large cluster.

    Parameters
    ----------
    latent_dim : int
        Bottleneck dimension (must match global checkpoint).
    base_ch : int
        Base channel count.
    band_weight : float
        CN band pixel weight.
    min_cluster_size : int
        Minimum stars in a cluster to warrant its own AE.
    n_epochs : int
        Max fine-tuning epochs per cluster.
    batch_size, lr, patience : standard training params.
    device : str
    global_ckpt_path : str
        Path to global CN-aware AE checkpoint.
    output_dir : str
        Directory to save per-cluster checkpoints.

    Returns
    -------
    cluster_results : dict
        cluster_id -> {"info": dict, "ckpt_path": str, "n_stars": int}
    """
    # Load data
    cluster_ids = get_cluster_assignments()
    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    if len(cluster_ids) > len(X):
        cluster_ids = cluster_ids[:len(X)]
    elif len(cluster_ids) < len(X):
        cluster_ids = np.pad(cluster_ids, (0, len(X) - len(cluster_ids)),
                              constant_values=-1)

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    big_clusters = unique_clusters[counts >= min_cluster_size]

    print(f"=" * 60)
    print(f"Per-Cluster CN-Aware AE Fine-Tuning")
    print(f"  Latent dim: {latent_dim}  Band weight: {band_weight}x")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Total clusters: {len(unique_clusters)}  "
          f"Eligible (>= {min_cluster_size}): {len(big_clusters)}")
    print(f"  Small clusters (using global AE): "
          f"{(counts < min_cluster_size).sum()}")
    print(f"=" * 60, flush=True)

    # Load global model once
    global_ckpt = torch.load(global_ckpt_path, map_location="cpu", weights_only=False)
    global_scaler_mean = global_ckpt.get("scaler_mean", 1.0)
    global_scaler_std = global_ckpt.get("scaler_std", 0.2)

    out_dir = Path(output_dir) / f"lat{latent_dim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_results = {}

    for i, cid in enumerate(big_clusters):
        ckpt_path = out_dir / f"cluster_{cid:02d}.pt"
        if ckpt_path.exists():
            print(f"\n[{i+1}/{len(big_clusters)}] Cluster {cid}: SKIP (checkpoint exists)", flush=True)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cluster_results[int(cid)] = {
                "n_stars": ckpt.get("n_stars", 0),
                "best_epoch": ckpt.get("best_epoch", 0),
                "best_loss": ckpt.get("best_loss", float("nan")),
                "elapsed": 0,
                "ckpt_path": str(ckpt_path),
            }
            continue

        mask = cluster_ids == cid
        X_c = X[mask]
        n_stars = len(X_c)
        print(f"\n[{i+1}/{len(big_clusters)}] Cluster {cid}: {n_stars} stars", flush=True)

        # Create fresh model and load global weights
        model = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
        model.load_state_dict(global_ckpt["model_state_dict"])

        # Fine-tune
        model, info = fine_tune_cluster(
            model, X_c,
            band_weight=band_weight,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            verbose=True,
        )

        # Save
        torch.save({
            "cluster_id": int(cid),
            "n_stars": n_stars,
            "latent_dim": latent_dim,
            "model_state_dict": model.state_dict(),
            "scaler_mean": info["scaler_mean"],
            "scaler_std": info["scaler_std"],
            "best_epoch": info["best_epoch"],
            "best_loss": info["best_loss"],
            "band_weight": band_weight,
            "band_defs": CN_BAND_DEFS,
        }, ckpt_path)

        cluster_results[int(cid)] = {
            "n_stars": n_stars,
            "best_epoch": info["best_epoch"],
            "best_loss": info["best_loss"],
            "elapsed": info["elapsed"],
            "ckpt_path": str(ckpt_path),
        }

        print(f"    Saved to {ckpt_path.name}  "
              f"best_loss={info['best_loss']:.6f}  "
              f"time={info['elapsed']:.0f}s", flush=True)

    # Save metadata
    meta = {
        "latent_dim": latent_dim,
        "band_weight": band_weight,
        "min_cluster_size": min_cluster_size,
        "global_ckpt": global_ckpt_path,
        "global_scaler_mean": global_scaler_mean,
        "global_scaler_std": global_scaler_std,
        "total_clusters": len(unique_clusters),
        "trained_clusters": len(big_clusters),
        "small_clusters": int((counts < min_cluster_size).sum()),
        "cluster_results": cluster_results,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\n{'=' * 60}")
    print(f"Complete: {len(cluster_results)} clusters fine-tuned")
    total_time = sum(r["elapsed"] for r in cluster_results.values())
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Checkpoints: {out_dir}")
    print(f"{'=' * 60}", flush=True)

    return cluster_results


def extract_cluster_features(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    latent_dim: int = 64,
    base_ch: int = 32,
    global_ckpt_path: str = "SpectraAE/checkpoints/cn_aware/ae_best.pt",
    cluster_ckpt_dir: str = "SpectraAE/checkpoints/cluster_ae",
    device: str = DEFAULT_DEVICE,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract features using per-cluster encoders, falling back to global.

    For each star:
    - If its cluster has a fine-tuned AE checkpoint, use that encoder.
    - Otherwise, use the global CN-aware encoder.

    Parameters
    ----------
    X : np.ndarray, shape (N, 700)
        Continuum-normalized spectra.
    cluster_ids : np.ndarray, shape (N,)
        Integer cluster ID per star.
    latent_dim : int
        Bottleneck dimension.
    base_ch : int
        Base channel count.
    global_ckpt_path : str
        Path to global CN-aware AE checkpoint.
    cluster_ckpt_dir : str
        Directory containing per-cluster checkpoints.
    device : str
    batch_size : int
        Inference batch size.

    Returns
    -------
    features : np.ndarray, shape (N, latent_dim)
    """
    N = len(X)
    ckpt_dir = Path(cluster_ckpt_dir) / f"lat{latent_dim}"
    features = np.zeros((N, latent_dim), dtype=np.float32)

    # Load global model once
    global_ckpt = torch.load(global_ckpt_path, map_location="cpu", weights_only=False)
    global_scaler_mean = global_ckpt.get("scaler_mean", 1.0)
    global_scaler_std = global_ckpt.get("scaler_std", 0.2)

    global_model = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
    global_model.load_state_dict(global_ckpt["model_state_dict"])
    global_model = global_model.to(device)
    global_model.eval()

    # Find unique clusters
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    n_global_used = 0
    n_cluster_used = 0

    print(f"Extracting features for {N:,} stars across {n_clusters} clusters...", flush=True)

    for i, cid in enumerate(unique_clusters):
        mask = cluster_ids == cid
        n_stars = mask.sum()
        ckpt_path = ckpt_dir / f"cluster_{int(cid):02d}.pt"

        if ckpt_path.exists():
            # Use cluster-specific encoder
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()
            sm = ckpt["scaler_mean"]
            ss = ckpt["scaler_std"]
            n_cluster_used += n_stars
        else:
            # Fall back to global encoder
            model = global_model
            sm = global_scaler_mean
            ss = global_scaler_std
            n_global_used += n_stars

        # Normalize and extract
        X_c = X[mask]
        lo = float(np.percentile(X_c, 1))
        hi = float(np.percentile(X_c, 99))
        X_norm = (np.clip(X_c, lo, hi) - sm) / ss
        X_norm = X_norm.astype(np.float32)

        feats_list = []
        with torch.no_grad():
            for j in range(0, n_stars, batch_size):
                batch = torch.from_numpy(X_norm[j:j + batch_size]).unsqueeze(1).to(device)
                z = model.encode(batch)
                feats_list.append(z.cpu().numpy())
        features[mask] = np.concatenate(feats_list, axis=0)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_clusters}] Cluster {int(cid)}: {n_stars} stars", flush=True)

    print(f"\nFeature extraction complete:")
    print(f"  Cluster-specific AE: {n_cluster_used:,} stars")
    print(f"  Global AE (fallback): {n_global_used:,} stars")
    print(f"  Shape: {features.shape}")
    print(f"  mean={features.mean():.4f}, std={features.std():.4f}")

    # Dead dim check
    fvar = features.var(axis=0)
    n_dead = int((fvar < 1e-8).sum())
    print(f"  Dead dims: {n_dead}/{latent_dim}")

    return features


# ══════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-cluster CN-aware AE fine-tuning & feature extraction")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--band-weight", type=float, default=5.0)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only extract features")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip training and use existing checkpoints")
    args = parser.parse_args()

    out_dir = Path("SpectraAE/checkpoints/cluster_ae") / f"lat{args.latent_dim}"
    meta_path = out_dir / "meta.pkl"

    if not args.extract_only and not args.skip_train:
        results = run_per_cluster_training(
            latent_dim=args.latent_dim,
            base_ch=args.base_ch,
            band_weight=args.band_weight,
            min_cluster_size=args.min_size,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )

    # Feature extraction
    if meta_path.exists() or args.skip_train or args.extract_only:
        if not meta_path.exists() and args.extract_only:
            print(f"WARNING: No meta file at {meta_path}. Using global only.")
        elif not meta_path.exists():
            print(f"No checkpoints found. Run training first (omit --skip-train).")
        else:
            print("\nExtracting cluster-conditioned features...")
            cluster_ids = get_cluster_assignments()
            X = np.load("ML/_cache/X_clean.npy").astype(np.float32)

            if len(cluster_ids) > len(X):
                cluster_ids = cluster_ids[:len(X)]
            elif len(cluster_ids) < len(X):
                cluster_ids = np.pad(cluster_ids, (0, len(X) - len(cluster_ids)),
                                      constant_values=-1)

            features = extract_cluster_features(
                X, cluster_ids,
                latent_dim=args.latent_dim,
                base_ch=args.base_ch,
                device=args.device,
            )

            feat_path = Path("SpectraAE/_cache") / f"ae_features_cluster_cn_{args.latent_dim}d.npy"
            np.save(feat_path, features)
            print(f"Saved features to {feat_path}")
    print("Done.")
