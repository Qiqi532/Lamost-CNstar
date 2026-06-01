"""Configuration for end-to-end PU deep network training."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EndToEndPUConfig:
    """Hyperparameters for 1D ResNet + CN attention + nnPU loss.

    Attributes
    ----------
    input_dim : int
        Number of spectral pixels (3800–4500Å at 1Å step = 700).
    base_ch : int
        Base channel count for ResNet.
    dropout : float
        Dropout rate in classifier head.
    se_reduction : int
        Reduction ratio for SE blocks (0 = no SE).
    use_cn_attention : bool
        Enable CN-band input attention.
    cn_band_weight_init : float
        Initial weight multiplier for CN band positions in the attention mask.
    freeze_cn_attention : bool
        If True, CN attention weights are fixed (not learned).
    res_blocks : int
        Number of residual blocks (3 or 4).
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization strength.
    batch_size : int
        Training batch size.
    n_epochs : int
        Maximum training epochs.
    pi_p : float or None
        Class prior π_p = p(y=1). If None, computed from data as n_pos / n_total.
        Typical value: 73/33565 ≈ 0.00217.
    nnpu_clamp : bool
        Use non-negative clamping (True → nnPU, False → uPU).
    mixup_alpha : float
        Beta distribution alpha for mixup in unlabeled data (0 = no mixup).
    pos_augment : bool
        Enable spectral augmentation for positive samples.
    pos_augment_factor : int
        Number of augmented copies per positive sample.
    early_stopping_patience : int
        Patience for early stopping on validation PR-AUC.
    lr_scheduler_patience : int
        Patience for ReduceLROnPlateau.
    lr_scheduler_factor : float
        Factor for ReduceLROnPlateau.
    val_split : float
        Fraction of data used for validation.
    random_seed : int
        Random seed for reproducibility.
    device : str
        Device to use.
    """

    # ── Network architecture ──
    input_dim: int = 700
    base_ch: int = 64
    dropout: float = 0.3
    se_reduction: int = 8
    use_cn_attention: bool = True
    cn_band_weight_init: float = 3.0
    freeze_cn_attention: bool = False
    res_blocks: int = 4

    # ── Training ──
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 300

    # ── PU loss ──
    pi_p: Optional[float] = None  # None → auto from data
    nnpu_clamp: bool = True

    # ── Regularization ──
    mixup_alpha: float = 0.2
    pos_augment: bool = True
    pos_augment_factor: int = 20
    grad_clip: float = 1.0
    hard_neg_frac: float = 0.3

    # ── Loss ──
    loss_mode: str = "weighted_bce"  # "weighted_bce", "nnpu", "upu"
    positive_weight: float = 10.0    # pos weight multiplier for weighted_bce
    neg_ratio: float = 1.0           # ratio of pseudo-negatives to positives per epoch

    # ── Optimization ──
    early_stopping_patience: int = 50
    lr_scheduler_patience: int = 15
    lr_scheduler_factor: float = 0.5

    # ── Data ──
    val_split: float = 0.15

    # ── Reproducibility ──
    random_seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"
