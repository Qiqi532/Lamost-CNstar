"""Train 256-dim Conv Autoencoder on LAMOST spectra.

Usage:
    python SpectraAE/train_ae256.py                    # train with defaults
    python SpectraAE/train_ae256.py --epochs 300        # custom epochs
    python SpectraAE/train_ae256.py --lr 5e-4 --batch-size 128
"""

import sys, time, argparse, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.pretrain import pretrain_autoencoder
from SpectraAE.models.autoencoder import ConvAutoencoder

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train 256-dim Conv AE on LAMOST")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────
    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    print(f"Loaded X_clean: {X.shape} | device: {args.device}")

    # ── Train ──────────────────────────────────────────────────
    model, history = pretrain_autoencoder(
        X,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        checkpoint_dir="SpectraAE/checkpoints/ae256",
    )

    # ── Save training history for notebook ─────────────────────
    history_path = Path("SpectraAE/_cache/ae256_history.pkl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

    # ── Quick reconstruction check ─────────────────────────────
    model.eval()
    X_t = torch.from_numpy(
        (np.clip(X[:8], float(np.percentile(X, 1)), float(np.percentile(X, 99)))
         - history["scaler_mean"]) / history["scaler_std"]
    ).float().unsqueeze(1).to(args.device)

    with torch.no_grad():
        recon, z = model(X_t)
    mse = nn.MSELoss()(recon, X_t).item()
    print(f"Sample recon MSE: {mse:.6f}")
    print(f"Latent shape: {z.shape}")

    # Compare with 64-dim checkpoint if exists
    ckpt64_path = Path("SpectraAE/checkpoints/ae_best.pt")
    if ckpt64_path.exists():
        ckpt64 = torch.load(ckpt64_path, map_location=args.device, weights_only=False)
        model64 = ConvAutoencoder(in_channels=1, latent_dim=64).to(args.device)
        model64.load_state_dict(ckpt64["model_state_dict"])
        model64.eval()
        with torch.no_grad():
            recon64, _ = model64(X_t)
        mse64 = nn.MSELoss()(recon64, X_t).item()
        print(f"AE-64d sample recon MSE: {mse64:.6f} (vs AE-256d: {mse:.6f})")

    print("Done.")


if __name__ == "__main__":
    main()
