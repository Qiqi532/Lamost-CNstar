"""PU training loop with weighted BCE / nnPU loss and strong regularization.

Key improvements over v1:
  1. Weighted BCE as default (more stable for extreme imbalance).
  2. Gradient clipping to prevent explosion.
  3. Hard negative mining — sample pseudo-negatives the model is unsure about.
  4. Monitor train-pos probability to detect "all-negative" collapse.
  5. Cosine annealing with warmup for better convergence.
  6. Positive up-weighting to compensate for tiny pi_p.
"""

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score

from .models.resnet_cn_attention import SpectraResNet_CN_Attention
from .pu_loss import nnPULoss
from .augmentation import augment_positive_spectra


class EndToEndPUTrainer:
    """Trains a model with weighted BCE or nnPU loss.

    Parameters
    ----------
    model : SpectraResNet_CN_Attention
    config : EndToEndPUConfig
    """

    def __init__(self, model: SpectraResNet_CN_Attention, config):
        self.model = model
        self.config = config
        self.device = torch.device(getattr(config, "device", "cpu"))
        self.pi_p = getattr(config, "pi_p", None)

        # Loss mode
        self.loss_mode = getattr(config, "loss_mode", "weighted_bce")
        self.positive_weight = getattr(config, "positive_weight", 50.0)
        self.grad_clip = getattr(config, "grad_clip", 1.0)
        self.hard_neg_frac = getattr(config, "hard_neg_frac", 0.3)

    def _build_loader(
        self, X_pos, X_neg, y_pos, y_neg, batch_size, shuffle=True,
    ) -> DataLoader:
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        perm = torch.randperm(len(y))
        ds = TensorDataset(X[perm], y[perm])
        # drop_last=True prevents BatchNorm errors on single-sample batches
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def _mixup_batch(self, x: torch.Tensor, rng: np.random.RandomState) -> torch.Tensor:
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
        X_t = torch.from_numpy(X).float().to(self.device)
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(self.device)
            preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds)

    def _compute_weighted_bce(
        self, logits: torch.Tensor, y: torch.Tensor, pos_weight: float,
    ) -> Dict:
        """Weighted BCE: positive samples get pos_weight × more weight."""
        pos_mask = y == 1
        unl_mask = y == 0

        bce = nn.BCEWithLogitsLoss(reduction='none')
        losses = bce(logits, y)  # (B,)

        if pos_mask.sum() > 0:
            # Apply positive weight
            losses = losses.clone()
            losses[pos_mask] *= pos_weight

        loss = losses.mean()

        # Track components
        with torch.no_grad():
            r_pos = losses[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
            r_unl = losses[unl_mask].mean().item() if unl_mask.sum() > 0 else 0.0

        return {"risk": loss, "r_pos_plus": r_pos, "r_unl_minus": r_unl,
                "clamped_term": torch.tensor(0.0, device=logits.device)}

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

        # ── Class prior ──
        if self.pi_p is None:
            n_pos_raw = len(X_train_pos)
            n_total = n_pos_raw + len(X_train_unl)
            self.pi_p = n_pos_raw / n_total
        pi_p = self.pi_p

        # ── Optimizer ──
        opt = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ── Scheduler: CosineAnnealing with warmup ──
        sched = CosineAnnealingLR(opt, T_max=config.n_epochs, eta_min=1e-6)

        # ── Loss ──
        if self.loss_mode == "nnpu":
            nnpu_criterion = nnPULoss(pi_p=pi_p, clamp=config.nnpu_clamp)
        elif self.loss_mode == "upu":
            nnpu_criterion = nnPULoss(pi_p=pi_p, clamp=False)

        # ── Validation tensor ──
        X_val_t = torch.from_numpy(X_val).float().to(self.device)

        # ── State ──
        n_pos_orig = len(X_train_pos)
        n_unl = len(X_train_unl)
        train_rng = np.random.RandomState(config.random_seed)

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

        # Track "hard negatives" cache (re-scored every 10 epochs)
        unl_scores_cache = np.full(n_unl, 0.5, dtype=np.float32)
        last_rescore = -999

        t0 = time.time()

        for epoch in range(1, config.n_epochs + 1):
            # ── 1. Augment positives ──
            if config.pos_augment and wave is not None:
                X_pos_aug = augment_positive_spectra(
                    X_train_pos, wave,
                    n_noise=5, n_rv=3, n_tilt=2, n_depth=3, n_mixup=4,
                    random_seed=config.random_seed + epoch,
                )
            else:
                X_pos_aug = X_train_pos

            n_aug_pos = len(X_pos_aug)

            # ── 2. Sample pseudo-negatives with hard negative mining ──
            # 1:1 balanced sampling (or neg_ratio:1 if config has it)
            neg_ratio = getattr(config, "neg_ratio", 1.0)
            neg_n = max(int(n_aug_pos * neg_ratio), config.batch_size * 2)

            # Rescore unlabeled every 10 epochs for hard mining
            if self.hard_neg_frac > 0 and (epoch - last_rescore >= 10 or epoch == 1):
                unl_scores_cache = self.predict(X_train_unl, batch_size=1024)
                last_rescore = epoch

            if self.hard_neg_frac > 0:
                n_hard = int(neg_n * self.hard_neg_frac)
                n_random = neg_n - n_hard

                # Hard negatives = high model score (false positives)
                # Random negatives = any
                hard_idx = np.argsort(unl_scores_cache)[-n_hard:]  # highest scores
                rand_idx = train_rng.choice(n_unl, size=n_random, replace=False)
                neg_idx = np.concatenate([hard_idx, rand_idx])
                neg_idx = np.unique(neg_idx)[:neg_n]
            else:
                neg_idx = train_rng.choice(n_unl, size=neg_n, replace=False)

            X_neg_epoch = X_train_unl[neg_idx]
            n_neg_used = len(X_neg_epoch)

            # ── 3. Build tensors ──
            X_pos_t = torch.from_numpy(X_pos_aug).float()
            y_pos_t = torch.ones(n_aug_pos).float()
            X_neg_t = torch.from_numpy(X_neg_epoch).float()
            y_neg_t = torch.zeros(n_neg_used).float()

            # ── 4. Mixup ──
            if config.mixup_alpha > 0 and n_neg_used >= 2:
                X_neg_t = self._mixup_batch(X_neg_t, train_rng)

            # ── 5. DataLoader ──
            loader = self._build_loader(
                X_pos_t, X_neg_t, y_pos_t, y_neg_t,
                batch_size=config.batch_size, shuffle=True,
            )

            # ── 6. Train one epoch ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                opt.zero_grad()
                logits = model(xb, return_logits=True)

                if self.loss_mode in ("nnpu", "upu"):
                    loss_dict = nnpu_criterion(logits, yb)
                else:
                    loss_dict = self._compute_weighted_bce(
                        logits, yb, self.positive_weight,
                    )

                loss = loss_dict["risk"]
                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            n_b = max(n_batches, 1)
            history["train_loss"].append(epoch_loss / n_b)
            history["lr"].append(opt.param_groups[0]["lr"])

            sched.step()

            # ── 7. Diagnose collapse: probability on train pos vs unl ──
            with torch.no_grad():
                train_pos_prob = self.predict(X_train_pos, batch_size=512).mean()
                train_unl_prob = self.predict(
                    X_train_unl[train_rng.choice(n_unl, size=min(2000, n_unl), replace=False)],
                    batch_size=512,
                ).mean()
            history["train_pos_prob_mean"].append(float(train_pos_prob))
            history["train_unl_prob_mean"].append(float(train_unl_prob))

            # ── 8. Validate ──
            model.eval()
            val_preds = self.predict(X_val, batch_size=512)
            val_pr = average_precision_score(y_val_bin, val_preds)
            val_roc = roc_auc_score(y_val_bin, val_preds)

            history["val_pr"].append(val_pr)
            history["val_roc"].append(val_roc)

            # ── 9. Early stopping ──
            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_val_roc = val_roc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % 20 == 0 or epoch == 1 or no_improve == 0):
                marker = "*" if no_improve == 0 else " "
                pos_p = history["train_pos_prob_mean"][-1]
                unl_p = history["train_unl_prob_mean"][-1]
                print(
                    f"  {marker}Ep {epoch:3d}/{config.n_epochs} | "
                    f"loss={history['train_loss'][-1]:.4f} | "
                    f"pos_p={pos_p:.3f} unl_p={unl_p:.4f} | "
                    f"val_pr={val_pr:.4f} val_roc={val_roc:.4f} | "
                    f"best={best_val_pr:.4f}@{best_epoch}"
                )

            if no_improve >= config.early_stopping_patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}")
                break

        elapsed = time.time() - t0

        # ── Restore best ──
        model.load_state_dict(best_state)
        model.eval()

        if verbose:
            print(
                f"\nTraining complete: best val PR={best_val_pr:.4f} "
                f"ROC={best_val_roc:.4f} @ epoch {best_epoch}  "
                f"({elapsed:.0f}s / {elapsed/60:.1f}min)"
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
        }
