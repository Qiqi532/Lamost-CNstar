"""1D Convolutional Autoencoder for LAMOST stellar spectra.

Encoder: Conv1d(1→32→64→128) with stride-2 downsampling → AdaptiveAvgPool1d → Linear → 64-d bottleneck.
Decoder: Linear → Upsample + Conv1d blocks → reconstruct 700-pixel spectrum.

Input: (B, 1, 700) continuum-normalized flux.
Output: (B, 1, 700) reconstructed flux, (B, 64) bottleneck features.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """1D Conv encoder: 700-pixel spectrum → 64-d latent vector."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32, latent_dim: int = 64):
        super().__init__()
        self.pad = nn.ReflectionPad1d(2)  # 700 → 704 (divisible by 8)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_ch * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 700)
        x = self.pad(x)           # (B, 1, 704)
        x = self.conv1(x)         # (B, 32, 352)
        x = self.conv2(x)         # (B, 64, 176)
        x = self.conv3(x)         # (B, 128, 88)
        x = self.pool(x)          # (B, 128, 1)
        x = x.flatten(1)          # (B, 128)
        x = self.fc(x)            # (B, latent_dim)
        return x


class Decoder(nn.Module):
    """1D decoder: 64-d latent → 700-pixel spectrum via Upsample + Conv1d blocks."""

    def __init__(self, latent_dim: int = 64, base_ch: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch * 4)  # 64 → 128

        self.deconv1 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv1d(base_ch, 1, kernel_size=7, padding=3, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        x = self.fc(z)                          # (B, 128)
        x = x.unsqueeze(-1)                     # (B, 128, 1)

        x = nn.functional.interpolate(x, size=88, mode='linear', align_corners=False)
        x = self.deconv1(x)                     # (B, 64, 88)

        x = nn.functional.interpolate(x, size=176, mode='linear', align_corners=False)
        x = self.deconv2(x)                     # (B, 32, 176)

        x = nn.functional.interpolate(x, size=352, mode='linear', align_corners=False)
        x = self.deconv3(x)                     # (B, 1, 352)

        x = nn.functional.interpolate(x, size=700, mode='linear', align_corners=False)
        return x                                # (B, 1, 700)


class ConvAutoencoder(nn.Module):
    """1D Conv Autoencoder for LAMOST 700-pixel spectra.

    forward(x) → (x_recon, z)
    encode(x)  → z (64-d bottleneck)
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, base_ch, latent_dim)
        self.decoder = Decoder(latent_dim, base_ch)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Quick shape test ──────────────────────────────────────────────
if __name__ == "__main__":
    model = ConvAutoencoder()
    x = torch.randn(4, 1, 700)
    x_recon, z = model(x)
    print(f"Input:  {x.shape}")
    print(f"Recon:  {x_recon.shape}")
    print(f"Latent: {z.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
