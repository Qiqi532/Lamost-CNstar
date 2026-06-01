from __future__ import annotations

"""AE-based classifier: wraps pretrained AE encoder with a classifier head.

Provides:
  - AEClassifier: nn.Module wrapping Encoder + classifier MLP head
  - build_ae_classifier(): factory that loads checkpoint, constructs model
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import ConvAutoencoder directly to avoid SpectraAE __init__ chain (which needs xgboost)
import importlib.util
_ae_spec = importlib.util.spec_from_file_location(
    "autoencoder",
    str(_PROJECT_ROOT / "SpectraAE" / "models" / "autoencoder.py"),
)
_ae_module = importlib.util.module_from_spec(_ae_spec)
_ae_spec.loader.exec_module(_ae_module)
ConvAutoencoder = _ae_module.ConvAutoencoder
Encoder = _ae_module.Encoder


class AEClassifier(nn.Module):
    """Pretrained AE encoder + classification head for PU learning.

    Pipeline:
      Input (B, 1, 700)
        → Encoder (pretrained Conv1d)
        → latent vector (B, latent_dim)
        → FC(latent_dim → hidden) → LayerNorm → ReLU → Dropout
        → FC(hidden → 1) → Sigmoid

    Parameters
    ----------
    encoder : nn.Module
        Pretrained Encoder from ConvAutoencoder.
    latent_dim : int
        Dimension of encoder bottleneck.
    hidden : int
        Hidden dimension of classifier MLP head.
    dropout : float
        Dropout rate in classifier head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Weight init for head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 700) or (B, 1, 700) input spectra.
            return_logits: if True, return logits before sigmoid.

        Returns:
            (B,) probabilities or logits.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, 700)

        z = self.encoder(x)  # (B, latent_dim)
        logits = self.head(z).squeeze(-1)

        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def freeze_encoder(self):
        """Freeze encoder parameters (for frozen feature training)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = True


def build_ae_classifier(
    checkpoint_path: str,
    latent_dim: int,
    base_ch: int = 32,
    freeze_encoder: bool = False,
    hidden: int = 128,
    dropout: float = 0.3,
    device: str = "cuda",
    verbose: bool = True,
) -> AEClassifier:
    """Build an AEClassifier from a pretrained AE checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint from pretrain.py or cn_aware_pretrain.py.
    latent_dim : int
        Bottleneck dimension (must match the checkpoint's architecture).
    base_ch : int
        Base channel count (32 for most AEs, 64 for ae256).
    freeze_encoder : bool
        If True, freeze encoder weights.
    hidden : int
        Hidden dim for classifier head.
    dropout : float
        Dropout rate for classifier head.
    device : str
        Device to load model on.
    verbose : bool
        Print checkpoint info.

    Returns
    -------
    AEClassifier model in eval mode.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        # Try relative to project root
        ckpt_path = _PROJECT_ROOT / checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if verbose:
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  AE epoch: {ckpt.get('epoch', '?')}  val_loss: {ckpt.get('val_loss', '?'):.6f}"
              if isinstance(ckpt.get('val_loss'), (int, float)) else "")
        print(f"  latent_dim={latent_dim}, base_ch={base_ch}, freeze_encoder={freeze_encoder}")

    # Build ConvAutoencoder and load weights
    ae = ConvAutoencoder(in_channels=1, base_ch=base_ch, latent_dim=latent_dim)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()

    # Extract encoder
    encoder = ae.encoder

    # Wrap in classifier
    model = AEClassifier(
        encoder=encoder,
        latent_dim=latent_dim,
        hidden=hidden,
        dropout=dropout,
    )

    if freeze_encoder:
        model.freeze_encoder()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  AEClassifier: {n_params:,} params  |  {n_trainable:,} trainable")

    return model


def extract_features_from_ae(
    model: AEClassifier,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Extract latent features from pretrained AE encoder.

    Parameters
    ----------
    model : AEClassifier
        Model with pretrained encoder.
    X : (N, 700) array
        Input spectra.
    batch_size : int
        Batch size for inference.
    device : str

    Returns
    -------
    (N, latent_dim) feature array.
    """
    import numpy as np
    model.encoder.eval()
    model.encoder.to(device)

    features = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            if xb.dim() == 2:
                xb = xb.unsqueeze(1)
            z = model.encoder(xb)
            features.append(z.cpu().numpy())

    return np.concatenate(features, axis=0)
