"""Training loop for AE-based classifier fine-tuning.

Replicates the EndToEndPUTrainer recipe (weighted BCE, augmentation,
hard negative mining, cosine annealing, gradient clipping) adapted
for the AEClassifier model architecture.
"""

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score


def _build_loader(X_pos, X_neg, y_pos, y_neg, batch_size, shuffle=True):
    """Build a balanced DataLoader from pos+neg tensors."""
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
    perm = torch.randperm(len(y))
    ds = TensorDataset(X[perm], y[perm])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def _mixup_batch(x: torch.Tensor, alpha: float, rng: np.random.RandomState) -> torch.Tensor:
    """Apply mixup within a batch."""
    n = x.shape[0]
    if n < 2 or alpha <= 0:
        return x
    half = n // 2
    perm = torch.randperm(n)
    for i in range(half):
        j = i * 2
        if j + 1 < n:
            lam = rng.beta(alpha, alpha)
            x[perm[j]] = lam * x[perm[j]] + (1.0 - lam) * x[perm[j + 1]]
    return x


@torch.no_grad()
def _predict(model, X: np.ndarray, batch_size: int = 512, device: str = "cuda") -> np.ndarray:
    """Predict probabilities for numpy array."""
    model.eval()
    model.to(device)
    X_t = torch.from_numpy(X).float()
    preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def train_ae_finetune(
    model,
    config,
    X_train_pos: np.ndarray,
    X_train_unl: np.ndarray,
    X_val: np.ndarray,
    y_val_bin: np.ndarray,
    wave: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """Fine-tune AEClassifier on PU classification task.

    Uses the same training recipe as EndToEndPUTrainer:
      - Weighted BCE loss (positive_weight from config)
      - Physics-motivated positive augmentation (5 transforms)
      - Hard negative mining (rescore every 10 epochs)
      - CosineAnnealingLR scheduler
      - Gradient clipping
      - Early stopping on validation PR-AUC

    Supports discriminative learning rates: encoder gets encoder_lr_mult * base_lr.

    Parameters
    ----------
    model : AEClassifier
        The AE-based classifier (encoder weights loaded).
    config : AEPretrainConfig or similar
        Must have: learning_rate, encoder_lr_mult, weight_decay, batch_size,
        n_epochs, early_stopping_patience, positive_weight, neg_ratio,
        grad_clip, hard_neg_frac, mixup_alpha, pos_augment, device.
    X_train_pos : (n_pos, 700) training positives.
    X_train_unl : (n_unl, 700) unlabeled pool.
    X_val : (n_val, 700) validation spectra.
    y_val_bin : (n_val,) binary validation labels (1=pos, 0=unl).
    wave : (700,) wavelength grid for augmentation.
    verbose : bool

    Returns
    -------
    dict with keys: model, best_epoch, best_val_pr, best_val_roc,
                    history, elapsed_seconds, pi_p, loss_mode, positive_weight.
    """
    device = torch.device(getattr(config, "device", "cuda"))
    model = model.to(device)

    # Class prior
    n_pos_raw = len(X_train_pos)
    n_total = n_pos_raw + len(X_train_unl)
    pi_p = n_pos_raw / n_total

    # ── Optimizer with discriminative LRs ──
    encoder_lr_mult = getattr(config, "encoder_lr_mult", 0.1)
    has_encoder_params = any(p.requires_grad for p in model.encoder.parameters())
    if has_encoder_params and encoder_lr_mult != 1.0:
        opt = AdamW([
            {"params": model.encoder.parameters(), "lr": config.learning_rate * encoder_lr_mult},
            {"params": model.head.parameters(), "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
    else:
        opt = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    sched = CosineAnnealingLR(opt, T_max=config.n_epochs, eta_min=1e-6)

    # ── Validation tensor ──
    X_val_t = torch.from_numpy(X_val).float().to(device)

    # ── State ──
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

    for epoch in range(1, config.n_epochs + 1):
        # ── 1. Augment positives ──
        if config.pos_augment and wave is not None:
            from EndToEndPU.augmentation import augment_positive_spectra
            X_pos_aug = augment_positive_spectra(
                X_train_pos, wave,
                n_noise=5, n_rv=3, n_tilt=2, n_depth=3, n_mixup=4,
                random_seed=getattr(config, "random_seed", 42) + epoch,
            )
        else:
            X_pos_aug = X_train_pos

        n_aug_pos = len(X_pos_aug)

        # ── 2. Sample pseudo-negatives ──
        neg_ratio = getattr(config, "neg_ratio", 1.0)
        neg_n = max(int(n_aug_pos * neg_ratio), config.batch_size * 2)
        hard_neg_frac = getattr(config, "hard_neg_frac", 0.3)

        if hard_neg_frac > 0 and (epoch - last_rescore >= 10 or epoch == 1):
            unl_scores_cache = _predict(model, X_train_unl, batch_size=1024, device=str(device))
            last_rescore = epoch

        if hard_neg_frac > 0:
            n_hard = int(neg_n * hard_neg_frac)
            n_random = neg_n - n_hard
            hard_idx = np.argsort(unl_scores_cache)[-n_hard:]
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
        mixup_alpha = getattr(config, "mixup_alpha", 0.2)
        if mixup_alpha > 0 and n_neg_used >= 2:
            X_neg_t = _mixup_batch(X_neg_t, mixup_alpha, train_rng)

        # ── 5. DataLoader ──
        loader = _build_loader(X_pos_t, X_neg_t, y_pos_t, y_neg_t, config.batch_size)

        # ── 6. Train one epoch ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pos_weight = getattr(config, "positive_weight", 10.0)
        bce = nn.BCEWithLogitsLoss(reduction='none')

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb, return_logits=True)

            # Weighted BCE
            losses = bce(logits, yb)
            pos_mask = yb == 1
            if pos_mask.sum() > 0:
                losses = losses.clone()
                losses[pos_mask] *= pos_weight
            loss = losses.mean()

            loss.backward()

            grad_clip = getattr(config, "grad_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()

        # ── 7. Diagnose collapse ──
        with torch.no_grad():
            pos_p = _predict(model, X_train_pos, device=str(device)).mean()
            unl_sample = X_train_unl[train_rng.choice(n_unl, size=min(2000, n_unl), replace=False)]
            unl_p = _predict(model, unl_sample, device=str(device)).mean()
        history["train_pos_prob_mean"].append(float(pos_p))
        history["train_unl_prob_mean"].append(float(unl_p))

        # ── 8. Validate ──
        model.eval()
        val_preds = _predict(model, X_val, device=str(device))
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

        if verbose and (epoch % 30 == 0 or epoch == 1 or no_improve == 0):
            marker = "*" if no_improve == 0 else " "
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
            f"\nAE Fine-tune complete: best val PR={best_val_pr:.4f} "
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
        "loss_mode": "weighted_bce",
        "positive_weight": getattr(config, "positive_weight", 10.0),
    }
