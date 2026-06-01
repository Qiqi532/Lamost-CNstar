"""Extended model variants for Experiment 2.

Provides:
  - create_model_extended(): factory supporting res_blocks=5
  - LabelSmoothBCEWithLogitsLoss: BCE with label smoothing for PU
  - build_model_variants(): returns all grid variants
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from EndToEndPU.models.resnet_cn_attention import (
    SpectraResNet_CN_Attention, CNBandInputAttention, SE1d, ResBlock1d,
)


class SpectraResNetExtended(SpectraResNet_CN_Attention):
    """Extended 1D ResNet supporting res_blocks=5."""

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
        # Skip parent __init__, build manually
        nn.Module.__init__(self)

        self.use_cn_attention = use_cn_attention
        if use_cn_attention:
            self.cn_attention = CNBandInputAttention(
                n_pixels=input_dim, band_weight_init=cn_band_weight_init,
            )
            if freeze_cn_attention:
                for p in self.cn_attention.parameters():
                    p.requires_grad = False

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Build ResBlocks dynamically
        if res_blocks == 5:
            self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7,
                                     stride=2, se_reduction=se_reduction)
            self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5,
                                     stride=2, se_reduction=se_reduction)
            self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3,
                                     stride=2, se_reduction=se_reduction)
            self.layer4 = ResBlock1d(base_ch * 8, base_ch * 16, kernel_size=3,
                                     stride=2, se_reduction=se_reduction)
            self.layer5 = ResBlock1d(base_ch * 16, base_ch * 16, kernel_size=3,
                                     stride=1, se_reduction=se_reduction)
            self.layer4_extra = None
            final_ch = base_ch * 16
        elif res_blocks == 4:
            self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7,
                                     stride=2, se_reduction=se_reduction)
            self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5,
                                     stride=2, se_reduction=se_reduction)
            self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3,
                                     stride=2, se_reduction=se_reduction)
            self.layer4 = ResBlock1d(base_ch * 8, base_ch * 8, kernel_size=3,
                                     stride=1, se_reduction=se_reduction)
            self.layer5 = None
            final_ch = base_ch * 8
        else:  # res_blocks == 3
            self.layer1 = ResBlock1d(base_ch, base_ch * 2, kernel_size=7,
                                     stride=2, se_reduction=se_reduction)
            self.layer2 = ResBlock1d(base_ch * 2, base_ch * 4, kernel_size=5,
                                     stride=2, se_reduction=se_reduction)
            self.layer3 = ResBlock1d(base_ch * 4, base_ch * 8, kernel_size=3,
                                     stride=1, se_reduction=se_reduction)
            self.layer4 = None
            self.layer5 = None
            final_ch = base_ch * 8

        # Pool + Classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.use_cn_attention:
            x = self.cn_attention(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        x = self.pool(x)
        logits = self.fc(x).squeeze(-1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class LabelSmoothBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing for PU classification.

    In PU learning with label smoothing:
      - Positive labels (1.0) → 1.0 - smoothing (e.g., 0.9)
      - Unlabeled-as-negative (0.0) → smoothing (e.g., 0.1)

    This prevents the model from becoming overconfident on the tiny positive set.
    """

    def __init__(self, smoothing: float = 0.1, pos_weight: float = 10.0):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.pos_target = 1.0 - smoothing
        self.neg_target = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed BCE loss.

        Args:
            logits: (B,) raw logits.
            targets: (B,) binary labels (1=pos, 0=unl).

        Returns:
            scalar loss.
        """
        pos_mask = targets == 1
        neg_mask = targets == 0

        # Smoothed targets
        smooth_targets = torch.full_like(logits, self.neg_target)
        smooth_targets[pos_mask] = self.pos_target

        # BCE loss with logits
        loss = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none')

        # Apply positive weight
        if pos_mask.sum() > 0:
            loss = loss.clone()
            loss[pos_mask] *= self.pos_weight

        return loss.mean()


def create_model_extended(
    input_dim: int = 700,
    base_ch: int = 64,
    dropout: float = 0.3,
    se_reduction: int = 8,
    use_cn_attention: bool = True,
    cn_band_weight_init: float = 3.0,
    res_blocks: int = 4,
) -> SpectraResNetExtended:
    """Factory for extended ResNet with configurable depth."""
    return SpectraResNetExtended(
        input_dim=input_dim,
        base_ch=base_ch,
        dropout=dropout,
        se_reduction=se_reduction,
        use_cn_attention=use_cn_attention,
        cn_band_weight_init=cn_band_weight_init,
        res_blocks=res_blocks,
    )


def build_model_variants() -> Dict[str, Dict]:
    """Define all model variants for the Experiment 2 grid.

    Returns dict mapping variant name → dict of constructor kwargs.
    """
    return {
        "baseline_64d4": {
            "base_ch": 64, "res_blocks": 4, "dropout": 0.3,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "wider_128d4": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.3,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "wider_256d4": {
            "base_ch": 256, "res_blocks": 4, "dropout": 0.3,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "deeper_64d5": {
            "base_ch": 64, "res_blocks": 5, "dropout": 0.3,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "high_dropout_128d4_d05": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.5,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "very_high_dropout_128d4_d07": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.7,
            "use_swa": False, "label_smoothing": 0.0,
        },
        "swa_128d4_d05": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.5,
            "use_swa": True, "label_smoothing": 0.0,
        },
        "label_smooth_128d4_d05": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.5,
            "use_swa": False, "label_smoothing": 0.1,
        },
        "combined_128d4_d05": {
            "base_ch": 128, "res_blocks": 4, "dropout": 0.5,
            "use_swa": True, "label_smoothing": 0.1,
        },
    }
