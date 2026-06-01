"""Deep SVDD (Support Vector Data Description) for anomaly detection.

Implements the Deep One-Class Classification approach from Ruff et al.
(ICML 2018) adapted for stellar spectral anomaly detection.

Reference: Ruff, L., et al. "Deep One-Class Classification." ICML 2018.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class CNStarEncoder(nn.Module):
    """MLP encoder mapping spectra to a compact latent representation.

    Architecture: Linear(bias=False) -> LeakyReLU -> Dropout (x N layers)
    No bias terms -- prevents hypersphere collapse (Ruff et al. 2018).
    """

    def __init__(
        self,
        input_dim: int = 700,
        hidden_dims: tuple = (256, 128, 64),
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim, bias=False),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, latent_dim, bias=False))

        self.encoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNStarConvEncoder(nn.Module):
    """1D Convolutional encoder for learning local spectral features.

    Architecture: Conv1d -> BN -> LeakyReLU -> MaxPool (x 3 blocks)
                 -> AdaptiveAvgPool -> Flatten -> FC -> latent

    Convolutions naturally detect local absorption bands (CN, CH, Mg b,
    Na D, Ca II) across the full spectrum, making better use of the
    5180-pixel input than a pure MLP.
    """

    def __init__(
        self,
        input_dim: int = 5180,
        latent_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv blocks: learn local spectral features with dimension reduction
        self.conv = nn.Sequential(
            # Block 1: 5180 -> 2590 -> 647
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),

            # Block 2: 647 -> 324 -> 81
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),

            # Block 3: 81 -> 81 -> 20
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
        )

        # Adaptive pooling -> fixed size regardless of input length
        self.pool = nn.AdaptiveAvgPool1d(10)  # -> (128, 10)

        # FC head: map conv features to latent space
        conv_out_dim = 128 * 10  # 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 256, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim, bias=False),
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class DeepSVDD(nn.Module):
    """Deep Support Vector Data Description.

    Learns a compact hypersphere in latent space that encloses normal data.
    Anomalies fall outside the sphere (larger distance from center).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : tuple of int
        Hidden layer sizes for the MLP encoder (ignored for Conv1D).
    latent_dim : int
        Latent representation dimensionality.
    dropout : float
        Dropout rate used in the encoder.
    encoder_type : str
        'mlp' or 'conv1d'.
    objective : str
        'one-class' or 'soft-boundary'.
    nu : float
        Nu parameter for soft-boundary objective (0 < nu <= 1).
    """

    def __init__(
        self,
        input_dim: int = 700,
        hidden_dims: tuple = (256, 128, 64),
        latent_dim: int = 32,
        dropout: float = 0.2,
        encoder_type: str = "mlp",
        objective: str = "one-class",
        nu: float = 0.1,
    ):
        super().__init__()

        if encoder_type == "conv1d":
            self.encoder = CNStarConvEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                dropout=dropout,
            )
        else:
            self.encoder = CNStarEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                dropout=dropout,
            )

        self.latent_dim = latent_dim
        self.objective = objective
        self.nu = nu
        self.encoder_type = encoder_type

        if objective == "soft-boundary":
            self.R = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.register_buffer("R", torch.tensor(0.0))

        self.register_buffer("c", torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def init_center(self, dataloader: torch.utils.data.DataLoader, eps: float = 0.1):
        device = next(self.parameters()).device
        self.eval()

        latents = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                z = self.encoder(x)
                latents.append(z.cpu())

        z_all = torch.cat(latents, dim=0)
        c_init = z_all.mean(dim=0)

        c_init[(c_init.abs() < eps) & (c_init >= 0)] = eps
        c_init[(c_init.abs() < eps) & (c_init < 0)] = -eps

        self.c = c_init.to(device)

        if self.objective == "soft-boundary":
            dists = torch.sum((z_all.to(device) - self.c) ** 2, dim=1)
            sorted_dists = torch.sort(dists)[0]
            idx = int((1.0 - self.nu) * len(dists))
            idx = min(idx, len(dists) - 1)
            self.R.data = torch.sqrt(sorted_dists[idx]).clone().detach()

        self.train()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            scores = torch.sum((z - self.c) ** 2, dim=1)
        return scores

    def compute_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        z = self.encoder(x)
        dists = torch.sum((z - self.c) ** 2, dim=1)

        if self.objective == "soft-boundary":
            scores = dists - self.R**2
            loss = self.R**2 + (1.0 / self.nu) * torch.mean(
                torch.clamp(scores, min=0)
            )
        else:
            loss = torch.mean(dists)

        stats = {
            "loss_total": loss.item(),
            "dist_mean": dists.mean().item(),
            "dist_std": dists.std().item(),
        }

        if self.objective == "soft-boundary":
            stats["R"] = self.R.item()
            stats["frac_outside"] = (dists > self.R**2).float().mean().item()

        return loss, stats
