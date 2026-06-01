"""1D ResNet + CN-band attention for CN-enhanced star classification.

Architecture:
  - CNBandInputAttention: learnable wavelength attention initialized to emphasize
    CN/CH molecular bands, directly integrating the "focus on CN bands" insight
    from SpectraAE into the classification network.
  - SE1d: Squeeze-and-Excitation channel attention (same as Deep/models/resnet1d.py).
  - ResBlock1d: 1D residual block with optional SE.
  - SpectraResNet_CN_Attention: full model combining all components.

The CN attention is applied at the input level as a learnable per-wavelength
multiplicative gate, initialized with Gaussian peaks at CN band centers.
This gives the model an inductive bias to attend to CN-relevant regions while
still learning to adjust weights from data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# CN molecular band definitions (wavelength in Å, same as project-wide)
# ═══════════════════════════════════════════════════════════════════

MOLECULAR_BAND_RANGES = [
    (3830, 3883),   # CN3839  (~53 pix at 1Å step)
    (4120, 4216),   # CN4142  (~96 pix)
    (4285, 4315),   # CH4300  (~30 pix)
]

# Total: ~179 pixels out of 700 ≈ 25.6% of spectrum


# ═══════════════════════════════════════════════════════════════════
# CN-band input attention
# ═══════════════════════════════════════════════════════════════════

class CNBandInputAttention(nn.Module):
    """Learnable per-wavelength attention mask initialized to CN bands.

    The attention weight w(λ) is a learnable parameter of shape (1, 1, n_pixels).
    At initialization, w(λ) ≈ 1.0 for continuum and ≈ band_weight_init for
    CN-band pixels. During training, the model learns to adjust these weights.

    Output: x_attended = x * softplus(w) / mean(softplus(w))
    Normalisation keeps the overall flux scale stable.

    Parameters
    ----------
    n_pixels : int
        Number of wavelength pixels (700 for 3800–4500Å at 1Å).
    wave_start : float
        Starting wavelength (default 3800Å).
    wave_step : float
        Wavelength step (default 1Å).
    band_weight_init : float
        Initial weight for CN band pixels (continuum init = 1.0).
        Higher values → stronger initial CN focus.
    edge_smooth : int
        Pixels for Gaussian taper at band edges (0 = sharp boundary).
    """

    def __init__(
        self,
        n_pixels: int = 700,
        wave_start: float = 3800.0,
        wave_step: float = 1.0,
        band_weight_init: float = 3.0,
        edge_smooth: int = 3,
    ):
        super().__init__()

        # Build initial weight vector
        wave = wave_start + np.arange(n_pixels, dtype=np.float64) * wave_step
        init_weights = np.ones(n_pixels, dtype=np.float32)

        for lo, hi in MOLECULAR_BAND_RANGES:
            band_mask = (wave >= lo) & (wave <= hi)
            if edge_smooth > 0:
                band_idx = np.where(band_mask)[0]
                if len(band_idx) == 0:
                    continue
                for i in band_idx:
                    dist_left = i - band_idx[0]
                    dist_right = band_idx[-1] - i
                    dist = min(dist_left, dist_right)
                    if dist < edge_smooth:
                        ramp = 0.5 + 0.5 * np.cos(np.pi * dist / edge_smooth)
                        w = 1.0 + (band_weight_init - 1.0) * (1.0 - ramp)
                    else:
                        w = band_weight_init
                    init_weights[i] = max(init_weights[i], w)
            else:
                init_weights[band_mask] = np.maximum(
                    init_weights[band_mask], band_weight_init
                )

        # Store as log-weights for unconstrained optimization
        # softplus(log_w) ≈ init_weights after initialisation
        init_log = np.log(np.exp(init_weights) - 1.0 + 1e-3)
        self.log_weight = nn.Parameter(
            torch.from_numpy(init_log).float().view(1, 1, n_pixels)
        )

        # Record band mask for reporting
        band_binary = np.zeros(n_pixels, dtype=np.float32)
        for lo, hi in MOLECULAR_BAND_RANGES:
            band_binary[(wave >= lo) & (wave <= hi)] = 1.0
        self.register_buffer(
            "band_mask", torch.from_numpy(band_binary).view(1, 1, n_pixels)
        )

        self.n_pixels = n_pixels

    def get_weights(self) -> torch.Tensor:
        """Return current attention weights (softplus of log-weights)."""
        w = F.softplus(self.log_weight)  # (1, 1, n_pixels)
        return w / w.mean()  # normalise to mean=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CN-band attention.

        Args:
            x: (B, 1, n_pixels) or (B, n_pixels) — input spectra.
        Returns:
            x_attended: same shape as input, with wavelength attention applied.
        """
        squeeze_last = x.dim() == 2
        if squeeze_last:
            x = x.unsqueeze(1)

        w = self.get_weights()
        out = x * w

        if squeeze_last:
            out = out.squeeze(1)
        return out

    def get_band_weight_ratio(self) -> float:
        """Mean attention weight in CN bands / mean weight in continuum."""
        w = self.get_weights()  # (1, 1, n_pixels)
        w_flat = w.view(-1)
        band_mask_flat = self.band_mask.view(-1)
        w_band = w_flat[band_mask_flat > 0.5].mean().item()
        w_cont = w_flat[band_mask_flat < 0.5].mean().item()
        return w_band / max(w_cont, 1e-8)


# ═══════════════════════════════════════════════════════════════════
# SE channel attention (same as Deep/models/resnet1d.py)
# ═══════════════════════════════════════════════════════════════════

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
        b, c, _ = x.shape
        w = x.mean(dim=-1)  # global average pooling over length
        w = self.fc(w).view(b, c, 1)
        return x * w


# ═══════════════════════════════════════════════════════════════════
# 1D Residual Block
# ═══════════════════════════════════════════════════════════════════

class ResBlock1d(nn.Module):
    """1D residual block with optional SE and stride."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
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


# ═══════════════════════════════════════════════════════════════════
# Full model: 1D ResNet + CN attention
# ═══════════════════════════════════════════════════════════════════

class SpectraResNet_CN_Attention(nn.Module):
    """1D ResNet with CN-band attention for end-to-end PU classification.

    Pipeline:
      Input (B, 1, 700)
        → CNBandInputAttention (learnable wavelength gate)
        → Stem: Conv1d(k=7,s=2) → BN → ReLU → MaxPool(k=2,s=2)
        → ResBlock layers with SE channel attention
        → AdaptiveAvgPool1d(1)
        → Classifier: 512 → 128 → BN → ReLU → Dropout → 1

    Parameters
    ----------
    input_dim : int
        Number of spectral pixels (default 700).
    base_ch : int
        Base channel count.
    dropout : float
        Dropout rate in classifier head.
    se_reduction : int
        SE reduction ratio (0 = disable SE).
    use_cn_attention : bool
        Enable CN-band input attention.
    cn_band_weight_init : float
        Initial CN band weight in attention mask.
    freeze_cn_attention : bool
        Freeze CN attention weights.
    res_blocks : int
        Number of ResBlocks (3 or 4).
    """

    def __init__(
        self,
        input_dim: int = 700,
        base_ch: int = 64,
        dropout: float = 0.3,
        se_reduction: int = 8,
        use_cn_attention: bool = True,
        cn_band_weight_init: float = 3.0,
        freeze_cn_attention: bool = False,
        res_blocks: int = 4,
    ):
        super().__init__()

        # ── CN-band attention (applied to raw input) ──
        self.use_cn_attention = use_cn_attention
        if use_cn_attention:
            self.cn_attention = CNBandInputAttention(
                n_pixels=input_dim,
                band_weight_init=cn_band_weight_init,
            )
            if freeze_cn_attention:
                for p in self.cn_attention.parameters():
                    p.requires_grad = False

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # input_dim/4
        )

        # ── ResBlocks ──
        if res_blocks == 4:
            self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7,
                                     stride=2, se_reduction=se_reduction)
            self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5,
                                     stride=2, se_reduction=se_reduction)
            self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3,
                                     stride=2, se_reduction=se_reduction)
            self.layer4 = ResBlock1d(base_ch * 8, base_ch * 8, kernel_size=3,
                                     stride=1, se_reduction=se_reduction)  # maintain resolution
            final_ch = base_ch * 8  # 512
        else:  # res_blocks == 3
            self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7,
                                     stride=2, se_reduction=se_reduction)
            self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5,
                                     stride=2, se_reduction=se_reduction)
            self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3,
                                     stride=1, se_reduction=se_reduction)  # maintain resolution
            self.layer4 = None
            final_ch = base_ch * 8  # 512

        # ── Global pooling ──
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Classifier head (LayerNorm avoids batch-size=1 issues) ──
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # ── Weight initialisation ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, n_pixels) or (B, 1, n_pixels).
            return_logits: if True, return raw logits (before sigmoid).

        Returns:
            probabilities (B,) if not return_logits, else logits (B,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, n_pixels)

        # CN-band attention
        if self.use_cn_attention:
            x = self.cn_attention(x)

        # Stem
        x = self.stem(x)

        # ResBlocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        # Pool + classify
        x = self.pool(x)
        logits = self.fc(x).squeeze(-1)

        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def get_cn_attention_profile(self) -> np.ndarray:
        """Return current CN attention weights as numpy array (for visualisation)."""
        if not self.use_cn_attention:
            return np.ones(700)
        with torch.no_grad():
            w = self.cn_attention.get_weights()
            return w.cpu().numpy().flatten()


# ═══════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════

def create_model(config) -> SpectraResNet_CN_Attention:
    """Create a SpectraResNet_CN_Attention from config."""
    return SpectraResNet_CN_Attention(
        input_dim=config.input_dim,
        base_ch=config.base_ch,
        dropout=config.dropout,
        se_reduction=config.se_reduction,
        use_cn_attention=config.use_cn_attention,
        cn_band_weight_init=config.cn_band_weight_init,
        freeze_cn_attention=config.freeze_cn_attention,
        res_blocks=config.res_blocks,
    )
