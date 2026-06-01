"""PU-Resample Trainer for deep learning on LAMOST spectra.

Each epoch: randomly sample pseudo-negatives from unlabeled pool,
train on balanced (pos + pseudo-neg) set. Simulates bagging through
training dynamics without needing 1000 separate models.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, average_precision_score


class PUResampleTrainer:
    """Trains a single model with per-epoch PU pseudo-negative resampling.

    Args:
        model: PyTorch nn.Module (binary classifier with sigmoid output)
        device: torch device
        val_split: fraction of unlabeled held out for validation
        lr: learning rate
        weight_decay: L2 regularization
        mixup_alpha: Beta distribution alpha for mixup (0 = no mixup)
        pos_upsample: repeat factor for positive samples in training batch
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        mixup_alpha: float = 0.2,
        pos_upsample: int = 1,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.mixup_alpha = mixup_alpha
        self.pos_upsample = pos_upsample

    def _mixup(self, x1: torch.Tensor, x2: torch.Tensor, lam: float):
        """Mix two spectra and their labels."""
        x_mixed = lam * x1 + (1 - lam) * x2
        return x_mixed

    def _create_balanced_loader(self, X_pos, y_pos, X_neg, y_neg, batch_size=64):
        """Build DataLoader with balanced positive/pseudo-negative batches."""
        if self.pos_upsample > 1 and len(X_pos) > 0:
            X_pos = X_pos.repeat(self.pos_upsample, 1)
            y_pos = y_pos.repeat(self.pos_upsample)

        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        perm = torch.randperm(len(y))
        ds = TensorDataset(X[perm], y[perm])
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Predict probabilities on numpy array."""
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i : i + batch_size]
            preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds)

    def train(
        self,
        X_train_pos: np.ndarray,       # (n_pos, n_features) — known CN stars
        X_train_unlabeled: np.ndarray, # (n_unl, n_features) — all unlabeled in train
        X_val: np.ndarray,             # validation spectra
        y_val: np.ndarray,             # validation labels (0/1)
        epochs: int = 200,
        patience: int = 30,
        batch_size: int = 64,
        verbose: bool = True,
    ):
        """PU-Resample training loop.

        Each epoch: sample |pos| pseudo-negatives, train balanced, validate.
        """
        model = self.model.to(self.device)
        opt = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.BCELoss()

        n_pos = len(X_train_pos)
        n_unl = len(X_train_unlabeled)
        X_pos_t = torch.from_numpy(X_train_pos).float()
        y_pos_t = torch.ones(n_pos).float()

        best_val_pr = -np.inf
        best_state = None
        best_epoch = 0
        no_improve = 0
        history = {"train_loss": [], "val_pr": []}

        rng = np.random.RandomState(42)

        for epoch in range(1, epochs + 1):
            # --- PU Resample: draw pseudo-negatives ---
            neg_idx = rng.choice(n_unl, size=n_pos, replace=False)
            X_neg_t = torch.from_numpy(X_train_unlabeled[neg_idx]).float()
            y_neg_t = torch.zeros(n_pos).float()

            # Apply mixup between some pseudo-negatives
            if self.mixup_alpha > 0 and n_pos >= 2:
                perm = torch.randperm(n_pos)
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=n_pos // 2)
                lam = torch.from_numpy(lam).float()
                for i in range(len(lam)):
                    j = i * 2
                    if j + 1 < n_pos:
                        x1, x2 = X_neg_t[perm[j]], X_neg_t[perm[j + 1]]
                        X_neg_t[perm[j]] = lam[i] * x1 + (1 - lam[i]) * x2

            loader = self._create_balanced_loader(
                X_pos_t, y_pos_t, X_neg_t, y_neg_t, batch_size=batch_size
            )

            # --- Train one epoch ---
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= n_pos * 2

            # --- Validate ---
            val_preds = self.predict(X_val)
            val_pr = average_precision_score(y_val, val_preds)
            val_bce = float(nn.BCELoss()(
                torch.from_numpy(val_preds), torch.from_numpy(y_val).float()
            ))

            sched.step(val_bce)
            history["train_loss"].append(epoch_loss)
            history["val_pr"].append(val_pr)

            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if verbose and epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | loss={epoch_loss:.4f} | "
                      f"val_pr={val_pr:.4f} | best={best_val_pr:.4f} @ {best_epoch}")

            if no_improve >= patience:
                if verbose:
                    print(f"    Early stop @ epoch {epoch}")
                break

        model.load_state_dict(best_state)
        model.eval()
        return model, best_epoch, best_val_pr, history


def train_ensemble(
    model_factory,
    X_train_pos: np.ndarray,
    X_train_unlabeled: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_models: int = 10,
    device: torch.device = torch.device("cpu"),
    epochs: int = 200,
    patience: int = 30,
    verbose: bool = True,
):
    """Train an ensemble of PU-Resample models with different seeds."""
    models = []
    histories = []

    for k in range(n_models):
        if verbose:
            print(f"\n  [{k+1}/{n_models}] Training ensemble member...")

        model = model_factory()
        trainer = PUResampleTrainer(model, device=device, mixup_alpha=0.2)

        t0 = time.time()
        model, best_ep, best_pr, hist = trainer.train(
            X_train_pos, X_train_unlabeled, X_val, y_val,
            epochs=epochs, patience=patience, verbose=verbose,
        )
        elapsed = time.time() - t0

        models.append(model)
        histories.append(hist)

        if verbose:
            print(f"    Done: best_pr={best_pr:.4f} @ ep={best_ep} ({elapsed:.0f}s)")

    return models, histories


def ensemble_predict(models, X: np.ndarray, device: torch.device, batch_size=256) -> np.ndarray:
    """Average predictions from ensemble members."""
    all_preds = []
    for m in models:
        # Create temporary trainer just for predict
        t = PUResampleTrainer(m, device=device)
        all_preds.append(t.predict(X, batch_size=batch_size))
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)
