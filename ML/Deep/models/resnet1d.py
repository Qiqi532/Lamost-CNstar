"""1D ResNet with Squeeze-and-Excitation for stellar spectral classification.

Designed for LAMOST spectra: learns multi-scale spectral features from CN bands
to broad continuum shape, with SE blocks automatically weighting key wavelength regions.
"""

import torch
import torch.nn as nn


class SE1d(nn.Module):
    """Squeeze-and-Excitation block for 1D signals (channel attention)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.shape
        w = x.mean(dim=-1)  # (B, C) — global average pooling over length
        w = self.fc(w).view(b, c, 1)  # (B, C, 1)
        return x * w


class ResBlock1d(nn.Module):
    """1D residual block with optional SE and stride."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SE1d(out_ch, se_reduction) if se_reduction > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class SpectraResNet(nn.Module):
    """1D ResNet + SE for spectral binary classification.

    Args:
        input_dim: number of wavelength pixels (e.g., 700 or 5000)
        base_ch: base channel count
        dropout: dropout rate in classifier head
    """

    def __init__(self, input_dim: int = 5000, base_ch: int = 64, dropout: float = 0.3):
        super().__init__()

        # Stem: initial downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # input_dim/4
        )

        # ResBlocks with progressive channel expansion and spatial reduction
        self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7, stride=2, se_reduction=8)
        self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5, stride=2, se_reduction=8)
        self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, se_reduction=8)
        self.layer4 = ResBlock1d(base_ch * 8, base_ch * 8, kernel_size=3, stride=2, se_reduction=8)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        final_ch = base_ch * 8  # 512

        # Classifier head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        logits = self.fc(x).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class SimpleConvNet(nn.Module):
    """Lightweight Conv1D baseline (similar to Deep/'s CNStarConvEncoder)."""

    def __init__(self, input_dim: int = 5000, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        logits = self.fc(x).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
