"""Extended configuration dataclasses for experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AEPretrainConfig:
    """Config for AE pretraining + fine-tuning experiment.

    Attributes
    ----------
    ae_checkpoint_path : str
        Path to pretrained AE .pt file.
    ae_latent_dim : int
        Bottleneck dimension (must match checkpoint).
    ae_base_ch : int
        Base channel count (must match checkpoint).
    freeze_encoder : bool
        If True, freeze encoder weights (feature extraction mode).
        If False, fine-tune encoder + classifier end-to-end.
    classifier_hidden : int
        Hidden dim in classifier MLP head.
    classifier_dropout : float
        Dropout rate in classifier head.
    encoder_lr_mult : float
        LR multiplier for encoder vs classifier (only for fine-tuning).
    """

    # AE checkpoint
    ae_checkpoint_path: str = "SpectraAE/checkpoints/ae256/ae_best.pt"
    ae_latent_dim: int = 256
    ae_base_ch: int = 64
    freeze_encoder: bool = False  # False = fine-tune, True = frozen features

    # Classifier head
    classifier_hidden: int = 128
    classifier_dropout: float = 0.3

    # Training (mirrors EndToEndPUConfig defaults)
    learning_rate: float = 1e-3
    encoder_lr_mult: float = 0.1  # encoder LR = learning_rate * encoder_lr_mult
    weight_decay: float = 1e-2
    batch_size: int = 64
    n_epochs: int = 300
    early_stopping_patience: int = 50

    # PU loss
    loss_mode: str = "weighted_bce"
    positive_weight: float = 10.0
    neg_ratio: float = 1.0
    grad_clip: float = 1.0
    hard_neg_frac: float = 0.3

    # Regularization
    mixup_alpha: float = 0.2
    pos_augment: bool = True
    pos_augment_factor: int = 20

    # Reproducibility
    random_seed: int = 42
    device: str = "cuda"


@dataclass
class LargeModelConfig:
    """Config for larger model + stronger regularization experiment.

    Attributes
    ----------
    base_ch : int
        Base channel count (64, 128, 256).
    res_blocks : int
        Number of ResBlocks (3, 4, 5).
    dropout : float
        Dropout rate.
    use_swa : bool
        Enable Stochastic Weight Averaging.
    swa_start_epoch : int
        Epoch to start SWA (fraction of n_epochs, e.g. 0.75 * n_epochs).
    label_smoothing : float
        Label smoothing factor (0 = no smoothing).
    """

    # Architecture
    base_ch: int = 64
    res_blocks: int = 4
    dropout: float = 0.3
    se_reduction: int = 8

    # Regularization
    use_swa: bool = False
    swa_start_frac: float = 0.75  # start SWA at 75% of epochs
    label_smoothing: float = 0.0

    # Training (mirrors EndToEndPUConfig defaults)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    n_epochs: int = 300
    early_stopping_patience: int = 50

    # PU loss
    loss_mode: str = "weighted_bce"
    positive_weight: float = 10.0
    neg_ratio: float = 1.0
    grad_clip: float = 1.0
    hard_neg_frac: float = 0.3

    # Regularization
    mixup_alpha: float = 0.2
    pos_augment: bool = True

    # Reproducibility
    random_seed: int = 42
    device: str = "cuda"

    @property
    def swa_start_epoch(self) -> int:
        return max(1, int(self.n_epochs * self.swa_start_frac))


@dataclass
class ActiveLearningConfig:
    """Config for active learning experiment.

    Attributes
    ----------
    n_initial_pos : int
        Number of initial training positives.
    n_confirm_pool : int
        Number of held-out positives in confirmation pool.
    k_per_round : int
        Number of candidates to "confirm" each round.
    n_rounds : int
        Number of active learning rounds.
    strategy : str
        Selection strategy: 'confidence', 'uncertainty', 'committee', 'random'.
    n_committee : int
        Number of models for committee strategy.
    """

    # Active learning setup
    n_initial_pos: int = 34
    n_confirm_pool: int = 20
    k_per_round: int = 5
    n_rounds: int = 4
    strategy: str = "confidence"  # confidence, uncertainty, committee, random
    n_committee: int = 3

    # Training (per-round, mirrors EndToEndPUConfig)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    n_epochs: int = 200  # fewer epochs per round (total train = n_epochs * n_rounds)
    early_stopping_patience: int = 40

    # Model
    base_ch: int = 64
    dropout: float = 0.3
    se_reduction: int = 8
    use_cn_attention: bool = True
    cn_band_weight_init: float = 3.0
    res_blocks: int = 4

    # PU loss
    loss_mode: str = "weighted_bce"
    positive_weight: float = 10.0
    neg_ratio: float = 1.0
    grad_clip: float = 1.0
    hard_neg_frac: float = 0.3

    # Regularization
    mixup_alpha: float = 0.2
    pos_augment: bool = True

    # Reproducibility
    random_seed: int = 42
    device: str = "cuda"
