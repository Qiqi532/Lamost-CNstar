"""Pretrain 1D Conv Autoencoder on LAMOST spectra (unsupervised).

Usage:
    python SpectraAE/pretrain.py                    # train with defaults
    python SpectraAE/pretrain.py --epochs 300       # custom epochs

Or import:
    from SpectraAE.pretrain import pretrain_autoencoder
    model, history = pretrain_autoencoder(X)
"""

import sys, time, argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain_autoencoder(
    X: np.ndarray,
    latent_dim: int = 64,
    base_ch: int = 32,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 200,
    val_split: float = 0.1,
    patience: int = 30,
    device: str = DEFAULT_DEVICE,
    checkpoint_dir: str = "SpectraAE/checkpoints",
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[ConvAutoencoder, Dict]:
    """Train a 1D Conv Autoencoder on LAMOST spectra.

    Parameters
    ----------
    X : np.ndarray, shape (N, 700)
        Continuum-normalized spectra.
    latent_dim : int
        Bottleneck dimension (default 64).
    base_ch : int
        Base channel count for encoder.
    lr, weight_decay : float
        AdamW optimizer parameters.
    n_epochs : int
        Maximum training epochs.
    val_split : float
        Fraction of data for validation (default 0.1).
    patience : int
        Early stopping patience on validation loss.
    device : str
        'cuda' or 'cpu'.
    checkpoint_dir : str
        Directory to save best model checkpoint.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    model : ConvAutoencoder
        Trained model (best checkpoint restored, in eval mode).
    history : dict
        Keys: train_losses, val_losses, best_epoch,
              scaler_mean, scaler_std, elapsed_seconds.
    """
    rng = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)

    # ── Normalize ──────────────────────────────────────────────
    # Global clip: remove extreme outliers from all pixels uniformly
    lo_global = float(np.percentile(X, 1))
    hi_global = float(np.percentile(X, 99))
    X_clipped = np.clip(X, lo_global, hi_global)

    n_total = len(X_clipped)
    n_val = int(n_total * val_split)
    indices = rng.permutation(n_total)
    val_indices = indices[:n_val]
    tr_indices = indices[n_val:]

    # Global standardization (scalars, not per-pixel)
    # Continuum-normalized spectra are centered around 1.0, global std ~0.24
    scaler_mean = float(X_clipped[tr_indices].mean())
    scaler_std = float(X_clipped[tr_indices].std())

    X_train = ((X_clipped[tr_indices] - scaler_mean) / scaler_std).astype(np.float32)
    X_val = ((X_clipped[val_indices] - scaler_mean) / scaler_std).astype(np.float32)

    # ── DataLoaders ────────────────────────────────────────────
    X_tr_t = torch.from_numpy(X_train).unsqueeze(1)  # (N, 1, 700)
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

    # ── Model ──────────────────────────────────────────────────
    model = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
    model = model.to(device)
    if verbose:
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} params  Device: {device}")

    # ── Optimizer & Scheduler ──────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    # ── Training ───────────────────────────────────────────────
    checkpoint_path = Path(checkpoint_dir) / "ae_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        tr_loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * len(batch_x)
        tr_loss = tr_loss_sum / len(X_tr_t)
        train_losses.append(tr_loss)

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                recon, _ = model(batch_x)
                val_loss_sum += criterion(recon, batch_x).item() * len(batch_x)
        val_loss = val_loss_sum / len(X_val_t)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
            }, checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = scheduler.get_last_lr()[0]
            marker = "*" if no_improve == 0 else " "
            print(f"  Epoch {epoch:3d}/{n_epochs} | {marker} "
                  f"tr={tr_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stop @ {epoch} (best_val={best_val_loss:.6f} @ epoch {best_epoch})")
            break

    elapsed = time.time() - t0

    # ── Restore best checkpoint ────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if verbose:
        print(f"Best epoch: {best_epoch}  val_loss={best_val_loss:.6f}  "
              f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "elapsed_seconds": elapsed,
    }
    return model, history


# ══════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Conv AE on LAMOST spectra")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--latent-dim", type=int, default=64, help="Bottleneck dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    CACHE_DIR = Path("ML/_cache")
    X = np.load(CACHE_DIR / "X_clean.npy").astype(np.float32)
    print(f"Loaded X_clean: {X.shape}")

    model, history = pretrain_autoencoder(
        X,
        latent_dim=args.latent_dim,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
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
