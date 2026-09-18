"""Training loop for DeepSVDD on LAMOST spectral data."""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import DeepSVDDConfig
from .data import SpectraDataset, create_dataloaders, get_clean_training_mask
from .models import DeepSVDD


class DeepSVDDTrainer:
    """Trainer for DeepSVDD anomaly detection model.

    Handles the full training lifecycle: center initialization, warm-up,
    main training loop with early stopping, and model checkpointing.

    Parameters
    ----------
    config : DeepSVDDConfig
        Training configuration.
    save_dir : str or Path, optional
        Directory for saving model checkpoints.
    """

    def __init__(
        self,
        config: DeepSVDDConfig,
        save_dir: Optional[str] = None,
    ):
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("Deep/checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[DeepSVDD] = None
        self.optimizer: Optional[Adam] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.history: List[Dict] = []

        self._set_seed(config.random_seed)

    @staticmethod
    def _set_seed(seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_model(self) -> DeepSVDD:
        """Initialize the DeepSVDD model."""
        model = DeepSVDD(
            input_dim=self.config.input_dim,
            hidden_dims=tuple(self.config.hidden_dims),
            latent_dim=self.config.latent_dim,
            dropout=self.config.dropout,
            encoder_type=self.config.encoder_type,
            objective=self.config.objective,
            nu=self.config.nu,
        )
        model.to(self.config.device)
        return model

    def _init_center(self, model: DeepSVDD, train_loader: DataLoader):
        """Initialize hypersphere center from forward pass on training data."""
        model.init_center(train_loader, eps=self.config.center_eps)

    def train(
        self,
        dataset: SpectraDataset,
        verbose: bool = True,
    ) -> DeepSVDD:
        """Train DeepSVDD on the given dataset.

        The model is trained only on unlabeled data (normal-background class).
        Known CN stars are used only for monitoring during training.

        Parameters
        ----------
        dataset : SpectraDataset
            Loaded spectral data.
        verbose : bool
            Print training progress.

        Returns
        -------
        DeepSVDD
            Trained model (best weights loaded, in eval mode).
        """
        config = self.config

        # CN-index pre-cleaning: exclude high-CN unlabeled stars from training
        clean_mask = get_clean_training_mask(
            dataset,
            percentile=config.clean_cn_percentile,
            min_positive=config.clean_min_positive,
            verbose=verbose,
        )

        # Create data loaders with cleaned training set
        loaders = create_dataloaders(
            dataset=dataset,
            batch_size=config.batch_size,
            val_split=0.1,
            random_seed=config.random_seed,
            clean_mask=clean_mask,
        )
        train_loader = loaders["train"]
        if verbose:
            print(f"  Train samples (clean): {loaders['n_train_clean']:,} "
                  f"/ {loaders['n_train_total']:,} unlabeled")

        # Build model
        model = self._build_model()
        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model: {n_params:,} parameters")
            print(f"  Input: {config.input_dim}")
            print(f"  Hidden: {config.hidden_dims}")
            print(f"  Latent: {config.latent_dim}")
            print(f"  Encoder: {config.encoder_type}")
            print(f"  Objective: {config.objective}")

        # Initialize center (before optimizer to avoid affecting gradients)
        if verbose:
            print("Initializing hypersphere center...")
        self._init_center(model, train_loader)

        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            verbose=verbose,
        )

        # Training state
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        self.history = []

        if verbose:
            print(f"\nTraining for up to {config.n_epochs} epochs...")
            print(f"  Train samples: {len(train_loader.dataset)}")
            print("-" * 60)

        t_start = time.time()

        for epoch in range(1, config.n_epochs + 1):
            # Train one epoch
            train_stats = self._train_epoch(model, train_loader, epoch)

            # Validation
            val_loss = self._validate(model, loaders["val_normal"])

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record history
            lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "train_loss": train_stats["loss_total"],
                "train_dist_mean": train_stats["dist_mean"],
                "val_loss": val_loss,
                "lr": lr,
            }
            if "R" in train_stats:
                record["R"] = train_stats["R"]
            self.history.append(record)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                flag = " *" if val_loss == best_val_loss else ""
                print(
                    f"Epoch {epoch:3d}/{config.n_epochs} | "
                    f"loss: {train_stats['loss_total']:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"lr: {lr:.2e}{flag}"
                )

            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

        # Restore best weights
        model.load_state_dict(best_model_state)
        model.eval()

        train_time = time.time() - t_start
        if verbose:
            print("-" * 60)
            print(f"Training completed in {train_time:.1f}s")
            print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = self.save_dir / "deep_svdd_best.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "config": self.config,
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "center": model.c.cpu(),
        }, ckpt_path)
        if verbose:
            print(f"Checkpoint saved to: {ckpt_path}")

        self.model = model
        return model

    def _train_epoch(
        self,
        model: DeepSVDD,
        loader: DataLoader,
        epoch: int,
    ) -> dict:
        """Run one training epoch."""
        model.train()
        device = self.config.device

        total_loss = 0.0
        total_dist_mean = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            self.optimizer.zero_grad()
            loss, stats = model.compute_loss(x)
            loss.backward()
            self.optimizer.step()

            total_loss += stats["loss_total"]
            total_dist_mean += stats["dist_mean"]
            n_batches += 1

        return {
            "loss_total": total_loss / n_batches,
            "dist_mean": total_dist_mean / n_batches,
            **({"R": model.R.item()} if model.objective == "soft-boundary" else {}),
        }

    @torch.no_grad()
    def _validate(self, model: DeepSVDD, loader: DataLoader) -> float:
        """Compute validation loss on normal-data samples."""
        model.eval()
        device = self.config.device

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            loss, _ = model.compute_loss(x)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def load_checkpoint(self, checkpoint_path: str) -> DeepSVDD:
        """Load a saved model checkpoint."""
        model = self._build_model()
        ckpt = torch.load(checkpoint_path, map_location=self.config.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        self.history = ckpt.get("history", [])
        return model

    def get_history_df(self):
        """Return training history as a DataFrame for plotting."""
        import pandas as pd
        return pd.DataFrame(self.history)
