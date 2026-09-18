"""Binary classifier training with ensemble support.

Supports single-model or ensemble training with different random seeds
and negative samples for robust CN-star probability estimation.
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .models import BinaryClassifier


# ═══════════════════════════════════════════════════════════════════
# Mixup
# ═══════════════════════════════════════════════════════════════════

def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def _mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════
# Balanced batch loader
# ═══════════════════════════════════════════════════════════════════

def create_balanced_loader(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    batch_size: int = 64,
    random_seed: int = 42,
) -> DataLoader:
    rng = np.random.RandomState(random_seed)
    n_pos, n_neg = len(X_pos), len(X_neg)
    n_per_class = max(n_pos, n_neg)
    half = batch_size // 2

    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)
    if n_pos < n_per_class:
        pos_idx = np.concatenate([pos_idx, rng.choice(n_pos, n_per_class - n_pos)])
    if n_neg < n_per_class:
        neg_idx = np.concatenate([neg_idx, rng.choice(n_neg, n_per_class - n_neg)])

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    X_batches, y_batches = [], []
    for i in range(0, n_per_class, half):
        p_idx = pos_idx[i:i + half]
        n_idx = neg_idx[i:i + half]
        actual = min(len(p_idx), len(n_idx))
        if actual == 0:
            break
        X_batch = np.vstack([X_pos[p_idx[:actual]], X_neg[n_idx[:actual]]])
        y_batch = np.concatenate([np.ones(actual), np.zeros(actual)])
        perm = rng.permutation(2 * actual)
        X_batches.append(X_batch[perm])
        y_batches.append(y_batch[perm])

    X_all = np.concatenate(X_batches).astype(np.float32)
    y_all = np.concatenate(y_batches).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


# ═══════════════════════════════════════════════════════════════════
# Single model trainer
# ═══════════════════════════════════════════════════════════════════

class BinaryTrainer:
    """Train a single binary classifier."""

    def __init__(
        self,
        input_dim: int = 1201,
        latent_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "conv1d",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        mixup_alpha: float = 0.2,
        n_epochs: int = 80,
        early_stopping_patience: int = 15,
        random_seed: int = 42,
        device: str = "cpu",
        save_dir: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.mixup_alpha = mixup_alpha
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("BinaryClassifier/checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[BinaryClassifier] = None
        self.history: List[Dict] = []

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _build_model(self) -> BinaryClassifier:
        return BinaryClassifier(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            encoder_type=self.encoder_type,
        ).to(self.device)

    def train(
        self,
        X_pos: np.ndarray,
        X_neg: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> BinaryClassifier:
        model = self._build_model()
        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"  Params: {n_params:,} | Pos: {len(X_pos)} Neg: {len(X_neg)} "
                  f"| Dim: {self.input_dim} | Type: {self.encoder_type}")

        train_loader = create_balanced_loader(X_pos, X_neg, batch_size=64, random_seed=self.random_seed)
        pos_weight_t = torch.tensor([self.pos_weight], device=self.device)
        criterion = nn.BCELoss(reduction="none")
        optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience = 0
        self.history = []

        t0 = time.time()
        for epoch in range(1, self.n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.mixup_alpha > 0:
                    x_m, y_a, y_b, lam = _mixup_batch(x, y, self.mixup_alpha)
                    optimizer.zero_grad()
                    pred, _ = model(x_m)
                    loss = _mixup_loss(criterion, pred, y_a, y_b, lam).mean()
                else:
                    optimizer.zero_grad()
                    pred, _ = model(x)
                    weight = torch.where(y > 0.5, pos_weight_t, torch.tensor(1.0, device=self.device))
                    loss = (weight * criterion(pred, y)).mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += ((pred > 0.5) == y).float().mean().item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_acc = epoch_acc / max(n_batches, 1)

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            val_loss = float("inf")
            val_acc = 0.0
            if X_val is not None and y_val is not None and len(X_val) > 0:
                val_loss, val_acc = self._validate(model, X_val, y_val, criterion, pos_weight_t)

            record = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                      "val_loss": val_loss, "val_acc": val_acc, "lr": lr}
            self.history.append(record)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

            if verbose and epoch % 20 == 0:
                flag = " *" if val_loss == best_val_loss else ""
                print(f"  Epoch {epoch:3d}/{self.n_epochs} | loss: {train_loss:.4f} | "
                      f"acc: {train_acc:.3f} | val_loss: {val_loss:.4f} | lr: {lr:.2e}{flag}")

            if patience >= self.early_stopping_patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch} (best: {best_epoch})")
                break

        model.load_state_dict(best_state)
        model.eval()
        if verbose:
            print(f"  Done in {time.time()-t0:.1f}s | best epoch {best_epoch} | val_loss {best_val_loss:.4f}")

        self.model = model

        # Save
        ckpt_path = self.save_dir / f"binary_cls_seed{self.random_seed}.pt"
        torch.save({"model_state_dict": best_state, "input_dim": self.input_dim,
                     "latent_dim": self.latent_dim, "encoder_type": self.encoder_type,
                     "history": self.history, "best_epoch": best_epoch}, ckpt_path)

        return model

    @torch.no_grad()
    def _validate(self, model, X_val, y_val, criterion, pos_weight_t):
        model.eval()
        x = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        pred, _ = model(x)
        weight = torch.where(y > 0.5, pos_weight_t, torch.tensor(1.0, device=self.device))
        loss = (weight * criterion(pred, y)).mean().item()
        acc = ((pred > 0.5) == y).float().mean().item()
        return loss, acc

    def load_checkpoint(self, path: str) -> BinaryClassifier:
        model = self._build_model()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        return model


# ═══════════════════════════════════════════════════════════════════
# Ensemble trainer
# ═══════════════════════════════════════════════════════════════════

class EnsembleTrainer:
    """Train an ensemble of binary classifiers with different seeds.

    Each model sees different negatives and uses a different random seed,
    producing more robust probability estimates via averaging.
    """

    def __init__(
        self,
        n_models: int = 5,
        base_seed: int = 42,
        **trainer_kwargs,
    ):
        self.n_models = n_models
        self.base_seed = base_seed
        self.trainer_kwargs = trainer_kwargs
        self.models: List[BinaryClassifier] = []
        self.trainers: List[BinaryTrainer] = []

    def train(
        self,
        X_pos: np.ndarray,
        X_neg_pool: np.ndarray,
        n_neg_per_model: int,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[BinaryClassifier]:
        """Train ensemble with different negatives per model."""
        rng = np.random.RandomState(self.base_seed)

        for i in range(self.n_models):
            seed = self.base_seed + i * 100
            if verbose:
                print(f"\n--- Model {i+1}/{self.n_models} (seed={seed}) ---")

            # Sample different negatives
            neg_idx = rng.choice(len(X_neg_pool), size=n_neg_per_model, replace=False)
            X_neg = X_neg_pool[neg_idx]

            trainer = BinaryTrainer(random_seed=seed, **self.trainer_kwargs)
            model = trainer.train(X_pos, X_neg, X_val, y_val, verbose=verbose)
            self.models.append(model)
            self.trainers.append(trainer)

        if verbose:
            print(f"\nEnsemble of {self.n_models} models trained.")
        return self.models

    def predict(self, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
        """Average predictions across ensemble."""
        all_probs = []
        for model in self.models:
            probs = model.predict(X, batch_size=batch_size, device=device)
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def predict_with_std(self, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std of ensemble predictions."""
        all_probs = []
        for model in self.models:
            probs = model.predict(X, batch_size=batch_size, device=device)
            all_probs.append(probs)
        all_probs = np.array(all_probs)
        return all_probs.mean(axis=0), all_probs.std(axis=0)

    def load_checkpoints(self, paths: List[str]) -> List[BinaryClassifier]:
        """Load ensemble from saved checkpoints."""
        self.models = []
        for i, path in enumerate(paths):
            trainer = BinaryTrainer(random_seed=0, **self.trainer_kwargs)
            model = trainer.load_checkpoint(path)
            self.models.append(model)
            self.trainers.append(trainer)
        return self.models
