"""CN-aware 1D Conv Autoencoder pretraining with band-weighted MSE loss.

Focuses on three molecular bands (CN3839, CN4142, CH4300) by applying higher
weight to pixels in those regions, forcing the AE to preserve CN-discriminative
features that standard MSE training ignores.

Usage:
    python SpectraAE/cn_aware_pretrain.py                      # train with defaults
    python SpectraAE/cn_aware_pretrain.py --band-weight 8.0    # stronger CN focus
    python SpectraAE/cn_aware_pretrain.py --latent-dim 128     # larger bottleneck

Or import:
    from SpectraAE.cn_aware_pretrain import cn_aware_pretrain_autoencoder
    model, history = cn_aware_pretrain_autoencoder(X)
"""

import sys, time, argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CN molecular band definitions (same as ML/utils.py BAND_DEFS)
CN_BAND_DEFS = {
    "CN3839": (3830, 3883),
    "CN4142": (4120, 4216),
    "CH4300": (4285, 4315),
}


def create_band_weight_mask(
    n_pixels: int = 700,
    wave_start: float = 3800.0,
    wave_step: float = 1.0,
    band_weight: float = 5.0,
    band_defs: Optional[Dict[str, tuple]] = None,
    edge_smooth: int = 0,
) -> torch.Tensor:
    """Create per-pixel weight mask with higher weights on CN/CH bands.

    Parameters
    ----------
    n_pixels : int
        Number of wavelength pixels (default 700 for 3800-4500A at 1A step).
    wave_start : float
        Starting wavelength in Angstroms.
    wave_step : float
        Wavelength step in Angstroms.
    band_weight : float
        Weight multiplier for band-region pixels (relative to continuum=1.0).
    band_defs : dict, optional
        Dict of band_name -> (wave_start, wave_end). Uses CN_BAND_DEFS if None.
    edge_smooth : int
        Number of pixels to linearly ramp weight at band edges (0 = sharp).

    Returns
    -------
    weight_mask : torch.Tensor, shape (1, 1, n_pixels)
        Float tensor with weight values (1.0 continuum, band_weight in bands).
    """
    if band_defs is None:
        band_defs = CN_BAND_DEFS

    wave = wave_start + np.arange(n_pixels) * wave_step
    weights = np.ones(n_pixels, dtype=np.float32)

    for (w1, w2) in band_defs.values():
        band_mask = (wave >= w1) & (wave <= w2)
        if edge_smooth > 0:
            # Create ramped transition at edges
            band_idx = np.where(band_mask)[0]
            for i in band_idx:
                # Distance from nearest band edge (in pixels)
                dist_left = i - band_idx[0]
                dist_right = band_idx[-1] - i
                dist = min(dist_left, dist_right)
                if dist < edge_smooth:
                    ramp = (dist + 1) / (edge_smooth + 1)
                    w = 1.0 + (band_weight - 1.0) * ramp
                else:
                    w = band_weight
                weights[i] = max(weights[i], w)
        else:
            weights[band_mask] = np.maximum(weights[band_mask], band_weight)

    return torch.from_numpy(weights).reshape(1, 1, n_pixels)


class WeightedMSELoss(nn.Module):
    """Per-pixel weighted MSE loss for CN-aware AE training."""

    def __init__(self, weight_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight_mask)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (recon - target) ** 2
        return (sq_err * self.weight).mean()


def compute_band_continuum_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    band_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-pixel band and continuum MSE for monitoring.

    Returns per-pixel average MSE, correctly handling batch dimension.
    """
    B = recon.shape[0]
    n_band = int(band_mask.sum().item())
    n_cont = band_mask.numel() - n_band

    with torch.no_grad():
        sq_err = (recon - target) ** 2
        total = sq_err.mean().item()

        band_loss = (sq_err * band_mask).sum().item() / (B * n_band) if n_band > 0 else 0.0

        continuum_mask = (band_mask == 0).float()
        cont_loss = (sq_err * continuum_mask).sum().item() / (B * n_cont) if n_cont > 0 else 0.0

    return {"total": total, "band": band_loss, "continuum": cont_loss}


def cn_aware_pretrain_autoencoder(
    X: np.ndarray,
    latent_dim: int = 64,
    base_ch: int = 32,
    band_weight: float = 5.0,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 200,
    val_split: float = 0.1,
    patience: int = 30,
    device: str = DEFAULT_DEVICE,
    checkpoint_dir: str = "SpectraAE/checkpoints/cn_aware",
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[ConvAutoencoder, Dict]:
    """Train a CN-aware 1D Conv Autoencoder with band-weighted MSE loss.

    Parameters
    ----------
    X : np.ndarray, shape (N, 700)
        Continuum-normalized spectra (3800-4500A, 1A step).
    latent_dim : int
        Bottleneck dimension (default 64).
    base_ch : int
        Base channel count for encoder.
    band_weight : float
        Weight multiplier for CN/CH band pixels (default 5.0).
        Continuum pixels get weight 1.0.
    batch_size, lr, weight_decay, n_epochs, val_split, patience, device :
        Standard training hyperparameters (see pretrain_autoencoder).
    checkpoint_dir : str
        Directory to save best model checkpoint.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Print training progress.

    Returns
    -------
    model : ConvAutoencoder
        Trained model (best checkpoint restored, in eval mode).
    history : dict
        Keys: train_losses, val_losses, best_epoch, best_val_loss,
              scaler_mean, scaler_std, elapsed_seconds,
              band_losses (train), val_band_losses,
              cont_losses (train), val_cont_losses,
              band_weight, band_defs.
    """
    rng = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    n_pixels = X.shape[1]

    # ── Create weight mask ───────────────────────────────────────
    weight_mask = create_band_weight_mask(
        n_pixels=n_pixels,
        wave_start=3800.0,
        wave_step=1.0,
        band_weight=band_weight,
    ).to(device)

    # Binary band mask for reporting (1.0 in bands, 0.0 elsewhere)
    band_mask_binary = (weight_mask > 1.1).float()

    if verbose:
        n_band_px = int(band_mask_binary.sum().item())
        n_cont_px = n_pixels - n_band_px
        print(f"CN band weight: {band_weight}x  "
              f"Band pixels: {n_band_px}  Continuum pixels: {n_cont_px}")
        print(f"Bands: {list(CN_BAND_DEFS.keys())}")

    # ── Normalize (same as pretrain.py) ──────────────────────────
    lo_global = float(np.percentile(X, 1))
    hi_global = float(np.percentile(X, 99))
    X_clipped = np.clip(X, lo_global, hi_global)

    n_total = len(X_clipped)
    n_val = int(n_total * val_split)
    indices = rng.permutation(n_total)
    val_indices = indices[:n_val]
    tr_indices = indices[n_val:]

    scaler_mean = float(X_clipped[tr_indices].mean())
    scaler_std = float(X_clipped[tr_indices].std())

    X_train = ((X_clipped[tr_indices] - scaler_mean) / scaler_std).astype(np.float32)
    X_val = ((X_clipped[val_indices] - scaler_mean) / scaler_std).astype(np.float32)

    # ── DataLoaders ──────────────────────────────────────────────
    X_tr_t = torch.from_numpy(X_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, X_tr_t),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, X_val_t),
        batch_size=batch_size * 2, shuffle=False,
    )

    if verbose:
        print(f"Train: {len(X_tr_t):,}  Val: {len(X_val_t):,}")
        print(f"Scaler: mean={scaler_mean:.4f}, std={scaler_std:.4f}  "
              f"(clip [{lo_global:.4f}, {hi_global:.4f}])")

    # ── Model ────────────────────────────────────────────────────
    model = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
    model = model.to(device)
    if verbose:
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} params  Device: {device}")

    # ── Optimizer & Scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )
    criterion = WeightedMSELoss(weight_mask)

    # ── Training ─────────────────────────────────────────────────
    checkpoint_path = Path(checkpoint_dir) / "ae_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []
    train_band_losses, train_cont_losses = [], []
    val_band_losses, val_cont_losses = [], []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        tr_loss_sum = 0.0
        tr_band_sum = 0.0
        tr_cont_sum = 0.0
        n_tr_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * len(batch_x)
            comp = compute_band_continuum_loss(recon, batch_x, band_mask_binary)
            tr_band_sum += comp["band"] * len(batch_x)
            tr_cont_sum += comp["continuum"] * len(batch_x)
            n_tr_batches += 1
        tr_loss = tr_loss_sum / len(X_tr_t)
        train_losses.append(tr_loss)
        train_band_losses.append(tr_band_sum / len(X_tr_t))
        train_cont_losses.append(tr_cont_sum / len(X_tr_t))

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_band_sum = 0.0
        val_cont_sum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                recon, _ = model(batch_x)
                val_loss_sum += criterion(recon, batch_x).item() * len(batch_x)
                comp = compute_band_continuum_loss(recon, batch_x, band_mask_binary)
                val_band_sum += comp["band"] * len(batch_x)
                val_cont_sum += comp["continuum"] * len(batch_x)
        val_loss = val_loss_sum / len(X_val_t)
        val_losses.append(val_loss)
        val_band_losses.append(val_band_sum / len(X_val_t))
        val_cont_losses.append(val_cont_sum / len(X_val_t))

        # Early stopping (on weighted val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_band_loss": val_band_losses[-1],
                "val_cont_loss": val_cont_losses[-1],
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
                "band_weight": band_weight,
                "band_defs": CN_BAND_DEFS,
            }, checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = scheduler.get_last_lr()[0]
            marker = "*" if no_improve == 0 else " "
            print(f"  Epoch {epoch:3d}/{n_epochs} | {marker} "
                  f"tr={tr_loss:.6f} (b={train_band_losses[-1]:.6f} c={train_cont_losses[-1]:.6f})  "
                  f"val={val_loss:.6f} (b={val_band_losses[-1]:.6f} c={val_cont_losses[-1]:.6f})  "
                  f"lr={lr_now:.2e}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop @ {epoch} (best_val={best_val_loss:.6f} @ epoch {best_epoch})")
            break

    elapsed = time.time() - t0

    # ── Restore best checkpoint ──────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if verbose:
        print(f"Best epoch: {best_epoch}  val_loss={best_val_loss:.6f}  "
              f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_band_losses": train_band_losses,
        "train_cont_losses": train_cont_losses,
        "val_band_losses": val_band_losses,
        "val_cont_losses": val_cont_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "elapsed_seconds": elapsed,
        "band_weight": band_weight,
        "band_defs": CN_BAND_DEFS,
    }
    return model, history


# ══════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CN-aware Conv AE on LAMOST spectra")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--band-weight", type=float, default=5.0,
                        help="Weight multiplier for CN/CH band pixels")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    CACHE_DIR = Path("ML/_cache")
    X = np.load(CACHE_DIR / "X_clean.npy").astype(np.float32)
    print(f"Loaded X_clean: {X.shape}")

    model, history = cn_aware_pretrain_autoencoder(
        X,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        band_weight=args.band_weight,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )

    # Quick reconstruction check
    x_sample = torch.from_numpy(
        (np.clip(X[:8], float(np.percentile(X, 1)), float(np.percentile(X, 99)))
         - history["scaler_mean"]) / history["scaler_std"]
    ).float().unsqueeze(1).to(args.device)
    with torch.no_grad():
        recon, z = model(x_sample)
    print(f"Sample recon MSE: {nn.MSELoss()(recon, x_sample).item():.6f}")
    print("Done.")
