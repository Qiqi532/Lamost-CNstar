"""Binary classifier training for CN-star detection.

Replaces DeepSVDD one-class training with supervised binary classification
using GCS-augmented positives and masked-cluster negatives.
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════
# Conv1D encoder adapted for narrow-band spectra (~180 pixels)
# ═══════════════════════════════════════════════════════════════════

class NarrowBandConvEncoder(nn.Module):
    """Lightweight 1D Conv encoder for narrow-band spectral regions.

    Designed for ~180-pixel inputs (concatenated CN3839/CN4142/CH4300 band cores).
    Uses less aggressive pooling to preserve resolution.
    """

    def __init__(self, input_dim: int = 180, latent_dim: int = 32, dropout: float = 0.3):
        super().__init__()

        self.conv = nn.Sequential(
            # Block 1: input -> input/2 -> input/4
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Block 3 — no stride reduction
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )

        self.pool = nn.AdaptiveAvgPool1d(8)
        conv_out_dim = 64 * 8

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 128, bias=False),
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


class BinaryClassifier(nn.Module):
    """Binary classifier with Conv1D encoder backbone.

    Encoder -> latent representation -> classification head -> sigmoid.
    """

    def __init__(
        self,
        input_dim: int = 180,
        latent_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = NarrowBandConvEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16, bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (probability, latent_vector)."""
        z = self.encoder(x)
        prob = self.classifier(z)
        return prob.squeeze(-1), z

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
        """Predict probabilities for input data.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
        batch_size : int
        device : str

        Returns
        -------
        np.ndarray of probabilities in [0, 1].
        """
        self.eval()
        # Move to correct device if needed
        original_device = next(self.parameters()).device
        target_device = original_device if original_device.type != 'cpu' else torch.device(device)
        self.to(target_device)

        probs = []
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(target_device)
            prob, _ = self.forward(x)
            probs.append(prob.cpu().numpy())

        return np.concatenate(probs)


# ═══════════════════════════════════════════════════════════════════
# Mixup
# ═══════════════════════════════════════════════════════════════════

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation to a batch.

    Returns mixed_x, y_a, y_b, lam where the loss should be:
        lam * BCE(pred, y_a) + (1-lam) * BCE(pred, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════
# Balanced batch sampler
# ═══════════════════════════════════════════════════════════════════

def create_balanced_dataloader(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    batch_size: int = 64,
    random_seed: int = 42,
) -> DataLoader:
    """Create a DataLoader where each batch has equal pos/neg samples.

    Samples with replacement from the smaller class to achieve balance.
    """
    rng = np.random.RandomState(random_seed)
    n_pos = len(X_pos)
    n_neg = len(X_neg)
    half = batch_size // 2

    # Replicate smaller class to match the larger one for balanced epoch sizing
    n_per_class = max(n_pos, n_neg)

    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)

    if n_pos < n_per_class:
        pos_idx = np.concatenate([pos_idx, rng.choice(n_pos, n_per_class - n_pos)])
    if n_neg < n_per_class:
        neg_idx = np.concatenate([neg_idx, rng.choice(n_neg, n_per_class - n_neg)])

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Build balanced batches
    X_batches = []
    y_batches = []

    for i in range(0, n_per_class, half):
        p_idx = pos_idx[i:i + half]
        n_idx = neg_idx[i:i + half]

        actual_half = min(len(p_idx), len(n_idx))
        if actual_half == 0:
            break

        X_batch = np.vstack([
            X_pos[p_idx[:actual_half]],
            X_neg[n_idx[:actual_half]],
        ])
        y_batch = np.concatenate([
            np.ones(actual_half, dtype=np.float32),
            np.zeros(actual_half, dtype=np.float32),
        ])

        # Shuffle within batch so pos/neg are interleaved
        perm = rng.permutation(len(X_batch))
        X_batches.append(X_batch[perm])
        y_batches.append(y_batch[perm])

    X_all = np.concatenate(X_batches, axis=0).astype(np.float32)
    y_all = np.concatenate(y_batches).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class BinaryTrainer:
    """Trainer for binary CN-star classifier."""

    def __init__(
        self,
        input_dim: int = 180,
        latent_dim: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        mixup_alpha: float = 0.2,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        lr_scheduler_patience: int = 8,
        lr_scheduler_factor: float = 0.5,
        random_seed: int = 42,
        device: str = "cpu",
        save_dir: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.mixup_alpha = mixup_alpha
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.random_seed = random_seed
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("Deep/checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[BinaryClassifier] = None
        self.history: List[Dict] = []

        self._set_seed(random_seed)

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self) -> BinaryClassifier:
        model = BinaryClassifier(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )
        model.to(self.device)
        return model

    def train(
        self,
        X_pos: np.ndarray,
        X_neg: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> BinaryClassifier:
        """Train the binary classifier.

        Parameters
        ----------
        X_pos : np.ndarray (n_pos, input_dim)
            Augmented GCS positive spectra.
        X_neg : np.ndarray (n_neg, input_dim)
            Masked-cluster negative spectra.
        X_val, y_val : np.ndarray or None
            Optional validation set.
        verbose : bool

        Returns
        -------
        BinaryClassifier (best weights loaded, in eval mode).
        """
        # Build model
        model = self._build_model()
        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"Binary classifier: {n_params:,} parameters")
            print(f"  Input dim: {self.input_dim}")
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  Pos: {len(X_pos)}, Neg: {len(X_neg)}")

        # Balanced data loader
        train_loader = create_balanced_dataloader(
            X_pos, X_neg, batch_size=64, random_seed=self.random_seed,
        )

        # Loss with class weighting
        pos_weight_tensor = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCELoss()  # We'll handle weighting manually

        # Optimizer
        optimizer = Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience, verbose=verbose,
        )

        # Training state
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        self.history = []

        if verbose:
            print(f"\nTraining for up to {self.n_epochs} epochs...")
            print("-" * 55)

        t_start = time.time()

        for epoch in range(1, self.n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                # Apply mixup
                if self.mixup_alpha > 0:
                    x, y_a, y_b, lam = mixup_batch(x, y, self.mixup_alpha)
                    y_a = y_a.to(self.device)
                    y_b = y_b.to(self.device)

                    optimizer.zero_grad()
                    pred, _ = model(x)
                    loss = mixup_loss(criterion, pred, y_a, y_b, lam)
                else:
                    optimizer.zero_grad()
                    pred, _ = model(x)
                    # Apply class weight
                    weight = torch.where(y > 0.5, self.pos_weight, 1.0)
                    loss = (weight * criterion(pred, y)).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += ((pred > 0.5) == y).float().mean().item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_acc = epoch_acc / max(n_batches, 1)

            # Validation
            val_loss = float("inf")
            val_acc = 0.0
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loss, val_acc = self._validate(
                    model, X_val, y_val, criterion,
                )

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
            }
            self.history.append(record)

            # Early stopping
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
                    f"Epoch {epoch:3d}/{self.n_epochs} | "
                    f"loss: {train_loss:.4f} | acc: {train_acc:.3f} | "
                    f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f} | "
                    f"lr: {lr:.2e}{flag}"
                )

            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

        model.load_state_dict(best_model_state)
        model.eval()

        train_time = time.time() - t_start
        if verbose:
            print("-" * 55)
            print(f"Training completed in {train_time:.1f}s")
            print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = self.save_dir / "binary_classifier_best.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, ckpt_path)
        if verbose:
            print(f"Checkpoint saved to: {ckpt_path}")

        self.model = model
        return model

    @torch.no_grad()
    def _validate(
        self,
        model: BinaryClassifier,
        X_val: np.ndarray,
        y_val: np.ndarray,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        model.eval()
        x = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        pred, _ = model(x)
        loss = criterion(pred, y).mean().item()
        acc = ((pred > 0.5) == y).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict probabilities for input spectra."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load_checkpoint().")
        self.model.eval()
        self.model.to(self.device)

        probs = []
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(self.device)
            prob, _ = self.model(x)
            probs.append(prob.cpu().numpy())

        return np.concatenate(probs)

    def load_checkpoint(self, checkpoint_path: str) -> BinaryClassifier:
        model = self._build_model()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        self.history = ckpt.get("history", [])
        return model
