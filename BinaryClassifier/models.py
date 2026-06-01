"""Model architectures for binary CN-star classification.

Conv1D encoder adapted for wider spectral range (3800-5000Å, ~1200 pixels).
Also supports 12-D physics features via a simple MLP.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# Conv1D encoder for full spectral range (~1200 pixels)
# ═══════════════════════════════════════════════════════════════════

class SpectralConvEncoder(nn.Module):
    """1D Conv encoder for wide spectral range (~1200 pix, 3800-5000Å).

    Three conv blocks with moderate pooling, preserving local spectral
    features including CN3839, CN4142, CH4300, Ca H&K, CH G-band.
    """

    def __init__(self, input_dim: int = 1201, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1: ~1200 -> ~600 -> ~150
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),

            # Block 2: ~150 -> ~75 -> ~18
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),

            # Block 3: ~18 -> ~18 -> preserve
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(8)
        conv_out = 128 * 8  # 1024

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 256, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim, bias=False),
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        x = self.conv(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# MLP encoder for physics features (12-D)
# ═══════════════════════════════════════════════════════════════════

class PhysicsMLPEncoder(nn.Module):
    """MLP encoder for 12-D physics feature vectors."""

    def __init__(self, input_dim: int = 12, latent_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim, bias=False),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════
# Binary classifier
# ═══════════════════════════════════════════════════════════════════

class BinaryClassifier(nn.Module):
    """Binary classifier with configurable encoder backbone.

    Supports 'conv1d' (spectral pixels) and 'mlp' (physics features).
    """

    def __init__(
        self,
        input_dim: int = 1201,
        latent_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "conv1d",
    ):
        super().__init__()

        if encoder_type == "conv1d":
            self.encoder = SpectralConvEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
            )
        elif encoder_type == "mlp":
            self.encoder = PhysicsMLPEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 16, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.encoder_type = encoder_type

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        prob = self.classifier(z)
        return prob.squeeze(-1), z

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
        self.eval()
        target_device = torch.device(device)
        self.to(target_device)

        probs = []
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(target_device)
            prob, _ = self.forward(x)
            probs.append(prob.cpu().numpy())
        return np.concatenate(probs)
