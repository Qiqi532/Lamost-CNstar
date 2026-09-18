"""Extract bottleneck features from a trained Conv Autoencoder.

Usage:
    python SpectraAE/extract_features.py                    # load ae_best.pt, extract all
    python SpectraAE/extract_features.py --checkpoint path  # custom checkpoint

Or import:
    from SpectraAE.extract_features import extract_features
    features = extract_features(model, X, scaler_mean, scaler_std)
"""

import sys, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SpectraAE.models.autoencoder import ConvAutoencoder

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features(
    model: ConvAutoencoder,
    X: np.ndarray,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    batch_size: int = 512,
    device: str = DEFAULT_DEVICE,
    verbose: bool = True,
) -> np.ndarray:
    """Extract 64-d bottleneck features from the encoder.

    Parameters
    ----------
    model : ConvAutoencoder
        Trained autoencoder (only encoder is used).
    X : np.ndarray, shape (N, 700)
        Continuum-normalized spectra.
    scaler_mean, scaler_std : np.ndarray or None
        Per-pixel normalization stats (same as used during pretraining).
        If None, no normalization is applied.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    features : np.ndarray, shape (N, latent_dim)
        Bottleneck representations.
    """
    # Normalize (global clip + standardization, matching pretrain.py)
    if scaler_mean is not None and scaler_std is not None:
        lo = float(np.percentile(X, 1))
        hi = float(np.percentile(X, 99))
        X = np.clip(X, lo, hi)
        X = (X - scaler_mean) / scaler_std
    X = X.astype(np.float32)

    model.eval()
    model = model.to(device)

    features_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).unsqueeze(1).to(device)
            z = model.encode(batch)
            features_list.append(z.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    if verbose:
        print(f"Extracted features: {features.shape}, "
              f"mean={features.mean():.4f}, std={features.std():.4f}")
    return features


# ══════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AE bottleneck features")
    parser.add_argument("--checkpoint", type=str, default="SpectraAE/checkpoints/ae_best.pt")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--output", type=str, default="SpectraAE/_cache/ae_features_64d.npy")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    # Load spectra
    X = np.load("ML/_cache/X_clean.npy").astype(np.float32)
    print(f"Loaded X_clean: {X.shape}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run pretrain.py first.")

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    scaler_mean = ckpt.get("scaler_mean")
    scaler_std = ckpt.get("scaler_std")
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")

    # Build model and load weights
    model = ConvAutoencoder(in_channels=1, latent_dim=args.latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])

    # Extract
    features = extract_features(
        model, X, scaler_mean, scaler_std,
        device=args.device,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    print(f"Saved to {output_path}")
