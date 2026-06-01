"""Extended PU trainer with SWA and label smoothing.

Extends EndToEndPUTrainer to support:
  - Stochastic Weight Averaging (SWA) via torch.optim.swa_utils
  - Label smoothing in BCE loss
  - Configurable dropout, base_ch, res_blocks
"""

import copy
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score

from EndToEndPU.augmentation import augment_positive_spectra
from EndToEndPU.experiments.exp2_large_models.extended_models import (
    LabelSmoothBCEWithLogitsLoss,
)


class ExtendedPUTrainer:
    """PU trainer with SWA + label smoothing support.

    Parameters
    ----------
    model : nn.Module
        Model to train (should accept (x, return_logits=True)).
    config : LargeModelConfig
        Training configuration.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(getattr(config, "device", "cpu"))

        self.loss_mode = getattr(config, "loss_mode", "weighted_bce")
        self.positive_weight = getattr(config, "positive_weight", 10.0)
        self.grad_clip = getattr(config, "grad_clip", 1.0)
        self.hard_neg_frac = getattr(config, "hard_neg_frac", 0.3)
        self.label_smoothing = getattr(config, "label_smoothing", 0.0)
        self.use_swa = getattr(config, "use_swa", False)
        self.swa_start_epoch = getattr(config, "swa_start_epoch", 225)

    def _build_loader(self, X_pos, X_neg, y_pos, y_neg, batch_size, shuffle=True):
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        perm = torch.randperm(len(y))
        ds = TensorDataset(X[perm], y[perm])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def _mixup_batch(self, x, rng):
        n = x.shape[0]
        if n < 2:
            return x
        half = n // 2
        perm = torch.randperm(n)
        for i in range(half):
            j = i * 2
            if j + 1 < n:
                lam = rng.beta(self.config.mixup_alpha, self.config.mixup_alpha)
                x[perm[j]] = lam * x[perm[j]] + (1.0 - lam) * x[perm[j + 1]]
        return x

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        X_t = torch.from_numpy(X).float()
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(self.device)
            preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds)

    def train(
        self,
        X_train_pos: np.ndarray,
        X_train_unl: np.ndarray,
        X_val: np.ndarray,
        y_val_bin: np.ndarray,
        wave: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        config = self.config
        model = self.model.to(self.device)

        n_pos_raw = len(X_train_pos)
        n_total = n_pos_raw + len(X_train_unl)
        pi_p = n_pos_raw / n_total

        opt = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=config.n_epochs, eta_min=1e-6)

        # SWA
        swa_model = None
        if self.use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            swa_model = AveragedModel(model)
            swa_scheduler = None  # will create at SWA start

        # Loss
        if self.label_smoothing > 0:
            criterion = LabelSmoothBCEWithLogitsLoss(
                smoothing=self.label_smoothing,
                pos_weight=self.positive_weight,
            )
        else:
            criterion = None  # use manual weighted BCE

        X_val_t = torch.from_numpy(X_val).float().to(self.device)

        n_unl = len(X_train_unl)
        train_rng = np.random.RandomState(getattr(config, "random_seed", 42))

        best_val_pr = -np.inf
        best_val_roc = 0.0
        best_state = None
        best_epoch = 0
        no_improve = 0

        history = {
            "train_loss": [], "val_pr": [], "val_roc": [],
            "train_pos_prob_mean": [], "train_unl_prob_mean": [],
            "lr": [],
        }

        unl_scores_cache = np.full(n_unl, 0.5, dtype=np.float32)
        last_rescore = -999
        t0 = time.time()
        swa_started = False

        for epoch in range(1, config.n_epochs + 1):
            # Augment
            if config.pos_augment and wave is not None:
                X_pos_aug = augment_positive_spectra(
                    X_train_pos, wave,
                    n_noise=5, n_rv=3, n_tilt=2, n_depth=3, n_mixup=4,
                    random_seed=getattr(config, "random_seed", 42) + epoch,
                )
            else:
                X_pos_aug = X_train_pos

            n_aug_pos = len(X_pos_aug)
            neg_ratio = getattr(config, "neg_ratio", 1.0)
            neg_n = max(int(n_aug_pos * neg_ratio), config.batch_size * 2)

            # Hard negative mining
            if self.hard_neg_frac > 0 and (epoch - last_rescore >= 10 or epoch == 1):
                unl_scores_cache = self.predict(X_train_unl, batch_size=1024)
                last_rescore = epoch

            if self.hard_neg_frac > 0:
                n_hard = int(neg_n * self.hard_neg_frac)
                n_random = neg_n - n_hard
                hard_idx = np.argsort(unl_scores_cache)[-n_hard:]
                rand_idx = train_rng.choice(n_unl, size=n_random, replace=False)
                neg_idx = np.concatenate([hard_idx, rand_idx])
                neg_idx = np.unique(neg_idx)[:neg_n]
            else:
                neg_idx = train_rng.choice(n_unl, size=neg_n, replace=False)

            X_neg_epoch = X_train_unl[neg_idx]
            n_neg_used = len(X_neg_epoch)

            # Build tensors
            X_pos_t = torch.from_numpy(X_pos_aug).float()
            y_pos_t = torch.ones(n_aug_pos).float()
            X_neg_t = torch.from_numpy(X_neg_epoch).float()
            y_neg_t = torch.zeros(n_neg_used).float()

            # Mixup
            if config.mixup_alpha > 0 and n_neg_used >= 2:
                X_neg_t = self._mixup_batch(X_neg_t, train_rng)

            loader = self._build_loader(X_pos_t, X_neg_t, y_pos_t, y_neg_t, config.batch_size)

            # Train epoch
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = model(xb, return_logits=True)

                if criterion is not None:
                    loss = criterion(logits, yb)
                else:
                    bce = nn.BCEWithLogitsLoss(reduction='none')
                    losses = bce(logits, yb)
                    pos_mask = yb == 1
                    if pos_mask.sum() > 0:
                        losses = losses.clone()
                        losses[pos_mask] *= self.positive_weight
                    loss = losses.mean()

                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(n_batches, 1))
            history["lr"].append(opt.param_groups[0]["lr"])
            sched.step()

            # SWA update
            if self.use_swa and epoch >= self.swa_start_epoch:
                if not swa_started:
                    swa_scheduler = SWALR(opt, swa_lr=config.learning_rate * 0.05)
                    swa_started = True
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # Diagnose
            with torch.no_grad():
                pos_p = self.predict(X_train_pos, batch_size=512).mean()
                unl_sample = X_train_unl[train_rng.choice(n_unl, size=min(2000, n_unl), replace=False)]
                unl_p = self.predict(unl_sample, batch_size=512).mean()
            history["train_pos_prob_mean"].append(float(pos_p))
            history["train_unl_prob_mean"].append(float(unl_p))

            # Validate (use SWA model if active)
            eval_model = swa_model if (self.use_swa and swa_started) else model
            eval_model.eval()

            with torch.no_grad():
                val_logits = []
                for i in range(0, len(X_val_t), 512):
                    xb = X_val_t[i:i + 512]
                    val_logits.append(eval_model(xb, return_logits=True).cpu().numpy())
                val_preds = 1.0 / (1.0 + np.exp(-np.concatenate(val_logits)))

            val_pr = average_precision_score(y_val_bin, val_preds)
            val_roc = roc_auc_score(y_val_bin, val_preds)
            history["val_pr"].append(val_pr)
            history["val_roc"].append(val_roc)

            # Early stopping (always on raw model for consistency)
            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_val_roc = val_roc
                best_epoch = epoch
                if self.use_swa and swa_started:
                    best_state = copy.deepcopy(swa_model.state_dict())
                else:
                    best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % 30 == 0 or epoch == 1 or no_improve == 0):
                marker = "*" if no_improve == 0 else " "
                print(
                    f"  {marker}Ep {epoch:3d}/{config.n_epochs} | "
                    f"loss={history['train_loss'][-1]:.4f} | "
                    f"pos_p={pos_p:.3f} unl_p={unl_p:.4f} | "
                    f"val_pr={val_pr:.4f} val_roc={val_roc:.4f} | "
                    f"best={best_val_pr:.4f}@{best_epoch}"
                    + (" [SWA]" if (self.use_swa and swa_started) else "")
                )

            if no_improve >= config.early_stopping_patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}")
                break

        elapsed = time.time() - t0

        # Restore best
        model.load_state_dict(best_state)
        model.eval()

        if verbose:
            print(
                f"\nTraining complete: best val PR={best_val_pr:.4f} "
                f"ROC={best_val_roc:.4f} @ epoch {best_epoch}  "
                f"({elapsed:.0f}s/{elapsed/60:.1f}min)"
            )

        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_val_pr": best_val_pr,
            "best_val_roc": best_val_roc,
            "history": history,
            "elapsed_seconds": elapsed,
            "pi_p": pi_p,
            "loss_mode": self.loss_mode,
            "positive_weight": self.positive_weight,
            "use_swa": self.use_swa,
            "label_smoothing": self.label_smoothing,
        }
