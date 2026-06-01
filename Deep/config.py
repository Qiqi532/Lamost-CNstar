"""Configuration for DeepSVDD training and evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DeepSVDDConfig:
    """Hyperparameters and settings for DeepSVDD anomaly detection.

    Attributes
    ----------
    input_dim : int
        Number of input features (spectral pixels).
    hidden_dims : List[int]
        Hidden layer sizes for the encoder MLP.
    latent_dim : int
        Dimension of the latent representation (hypersphere space).
    dropout : float
        Dropout rate in encoder layers.
    nu : float
        DeepSVDD nu parameter (soft-boundary trade-off, 0 < nu <= 1).
        Smaller values tighten the boundary.
    weight_decay : float
        L2 regularization strength.
    learning_rate : float
        Initial learning rate for Adam optimizer.
    batch_size : int
        Training batch size.
    n_epochs : int
        Maximum number of training epochs.
    center_eps : float
        Epsilon for center initialization (prevent center from being too
        close to zero).
    objective : str
        DeepSVDD objective: 'soft-boundary' (with nu) or 'one-class' (center
        distance only).
    warm_up_n_epochs : int
        Number of warm-up epochs before enforcing DeepSVDD objective.
    early_stopping_patience : int
        Patience for early stopping on validation loss.
    lr_scheduler_patience : int
        Patience for ReduceLROnPlateau.
    lr_scheduler_factor : float
        Factor for ReduceLROnPlateau.
    random_seed : int
        Random seed for reproducibility.
    device : str
        Device to use ('cuda' or 'cpu').
    """

    # Network architecture
    input_dim: int = 5180  # 3710-8890Å at 1Å step (full LRS blue+red arms)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    latent_dim: int = 32
    dropout: float = 0.3
    encoder_type: str = "conv1d"  # 'mlp' or 'conv1d'

    # DeepSVDD parameters
    nu: float = 0.05  # Smaller → tighter boundary (0 < nu <= 1)
    center_eps: float = 0.1  # Epsilon for center init (avoid trivial zero)
    weight_decay: float = 1e-6
    objective: str = "soft-boundary"  # 'soft-boundary' or 'one-class'

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 128
    n_epochs: int = 200

    # Data pre-cleaning (exclude high-CN-index unlabeled stars from training)
    clean_cn_percentile: float = 95.0  # Exclude top (100-N)% CN-index stars
    clean_min_positive: int = 5  # 正样本少于这个数则不做清洗
    warm_up_n_epochs: int = 0
    early_stopping_patience: int = 30
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    # Reproducibility
    random_seed: int = 42
    device: str = "cpu"
