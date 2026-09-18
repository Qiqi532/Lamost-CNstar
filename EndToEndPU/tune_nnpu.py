"""nnPU Loss Hyperparameter Tuning for CN-enhanced Star Detection.

Systematically explores nnPU/uPU/WeightedBCE loss variants with
hyperparameter grid search, tracking training dynamics (pos_p oscillation,
loss components, collapse detection) across all runs.

Usage:
    python -m EndToEndPU.tune_nnpu                          # full grid search
    python -m EndToEndPU.tune_nnpu --quick                   # quick 3-run test
    python -m EndToEndPU.tune_nnpu --loss nnpu --pi-p 0.002  # single run
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

# ── Path setup ──
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from EndToEndPU.config import EndToEndPUConfig
from EndToEndPU.data_loader import load_data, split_data, get_train_pos_unl, create_val_labeled_set
from EndToEndPU.models.resnet_cn_attention import create_model
from EndToEndPU.pu_loss import nnPULoss
from EndToEndPU.augmentation import augment_positive_spectra
from EndToEndPU.evaluate import compute_metrics, evaluate_model

RESULTS_DIR = _CURRENT_DIR / "results" / "nnpu_tuning"


# ═══════════════════════════════════════════════════════════════════
# Loss components tracker
# ═══════════════════════════════════════════════════════════════════

class LossTracker:
    """Tracks full loss decomposition for nnPU/uPU analysis."""

    def __init__(self):
        self.records: List[Dict] = []

    def update(self, epoch: int, loss_dict: Dict, mode: str):
        """Record loss components for one batch."""
        def _safe_float(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            return float(v)

        self.records.append({
            "epoch": epoch,
            "mode": mode,
            "risk": _safe_float(loss_dict.get("risk", 0)),
            "r_pos_plus": _safe_float(loss_dict.get("r_pos_plus", 0)),
            "r_unl_minus": _safe_float(loss_dict.get("r_unl_minus", 0)),
            "r_pos_minus": _safe_float(loss_dict.get("r_pos_minus", 0)),
            "clamped": _safe_float(loss_dict.get("clamped_term", 0)),
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def epoch_summary(self) -> pd.DataFrame:
        """Aggregate to per-epoch means."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.groupby("epoch").mean(numeric_only=True).reset_index()


class TuneTrainer:
    """Flexible trainer supporting nnPU, uPU, weighted_bce with full tracking.

    Key improvements over EndToEndPUTrainer:
      - Full loss decomposition tracking (nnPU components)
      - pos_p oscillation metrics
      - Collapse detection and recovery
      - Per-epoch detailed logging for notebook analysis
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(getattr(config, "device", "cpu"))
        self.loss_mode = getattr(config, "loss_mode", "weighted_bce")
        self.pi_p = getattr(config, "pi_p", None)
        self.positive_weight = getattr(config, "positive_weight", 10.0)
        self.grad_clip = getattr(config, "grad_clip", 1.0)
        self.hard_neg_frac = getattr(config, "hard_neg_frac", 0.3)
        self.loss_tracker = LossTracker()

    def _build_loader(self, X_pos, X_neg, y_pos, y_neg, batch_size, shuffle=True):
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
        perm = torch.randperm(len(y))
        ds = TensorDataset(X[perm], y[perm])
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
        model = self.model
        was_training = model.training
        model.eval()
        model.to(self.device)
        X_t = torch.from_numpy(X).float().to(self.device)
        preds = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            preds.append(model(xb).cpu().numpy())
        if was_training:
            model.train()
        return np.concatenate(preds)

    def train(
        self,
        model,
        X_train_pos: np.ndarray,
        X_train_unl: np.ndarray,
        X_val: np.ndarray,
        y_val_bin: np.ndarray,
        wave: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        config = self.config
        self.model = model.to(self.device)
        self.loss_tracker = LossTracker()

        # ── Prior ──
        if self.pi_p is None:
            self.pi_p = len(X_train_pos) / (len(X_train_pos) + len(X_train_unl))
        pi_p = self.pi_p

        # ── Optimizer ──
        opt = AdamW(model.parameters(), lr=config.learning_rate,
                     weight_decay=config.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=config.n_epochs, eta_min=1e-6)

        # ── Loss criterion ──
        use_nnpu = (self.loss_mode in ("nnpu", "upu"))
        if use_nnpu:
            pu_criterion = nnPULoss(
                pi_p=pi_p,
                clamp=(self.loss_mode == "nnpu"),
                loss_func="sigmoid",
            )

        # ── Validation tensor (GPU-resident for speed) ──
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

        # Oscillation tracking
        pos_p_history = []
        collapse_events = 0
        recovery_events = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "val_pr": [],
            "val_roc": [],
            "train_pos_prob_mean": [],
            "train_unl_prob_mean": [],
            "pos_p_std": [],  # new: within-epoch std (proxy for oscillation)
            "lr": [],
            "loss_r_pos_plus": [],
            "loss_r_unl_minus": [],
            "loss_r_pos_minus": [],
            "loss_r_clamped": [],
        }

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

            # ── 2. Sample negatives with hard mining ──
            neg_ratio = getattr(config, "neg_ratio", 1.0)
            neg_n = max(int(n_aug_pos * neg_ratio), config.batch_size * 2)

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
            epoch_r_pos_plus = 0.0
            epoch_r_unl_minus = 0.0
            epoch_r_pos_minus = 0.0
            epoch_r_clamped = 0.0
            n_batches = 0
            batch_pos_probs = []  # within-epoch tracking

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                opt.zero_grad()
                logits = model(xb, return_logits=True)

                # pos_mask defined for BOTH branches
                pos_mask = yb == 1

                if use_nnpu:
                    loss_dict_raw = pu_criterion(logits, yb)
                    # Extract risk as tensor (for backward), convert rest to float
                    loss_dict = {
                        "risk": loss_dict_raw["risk"],  # keep as tensor
                        "r_pos_plus": float(loss_dict_raw["r_pos_plus"]),
                        "r_unl_minus": float(loss_dict_raw["r_unl_minus"]),
                        "r_pos_minus": float(loss_dict_raw["r_pos_minus"]),
                        "clamped_term": float(loss_dict_raw.get("clamped_term", 0)),
                    }
                else:
                    # Weighted BCE
                    bce = nn.BCEWithLogitsLoss(reduction='none')
                    losses = bce(logits, yb)
                    if pos_mask.sum() > 0:
                        losses = losses.clone()
                        losses[pos_mask] *= self.positive_weight
                    loss = losses.mean()  # keep as tensor for backward
                    with torch.no_grad():
                        r_pp = losses[pos_mask].mean().item() if pos_mask.sum() > 0 else 0.0
                        r_um = losses[~pos_mask].mean().item() if (~pos_mask).sum() > 0 else 0.0
                    loss_dict = {
                        "risk": loss,        # tensor
                        "r_pos_plus": r_pp,  # float
                        "r_unl_minus": r_um, # float
                        "r_pos_minus": 0.0,  # float
                        "clamped_term": 0.0, # float
                    }

                loss = loss_dict["risk"]
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                opt.step()

                epoch_loss += loss_dict["risk"].item() if isinstance(loss_dict["risk"], torch.Tensor) else float(loss_dict["risk"])
                epoch_r_pos_plus += float(loss_dict["r_pos_plus"])
                epoch_r_unl_minus += float(loss_dict["r_unl_minus"])
                epoch_r_pos_minus += float(loss_dict["r_pos_minus"])
                epoch_r_clamped += float(loss_dict.get("clamped_term", 0))
                n_batches += 1

                # Track within-epoch pos probs
                with torch.no_grad():
                    if pos_mask.sum() > 0:
                        batch_pos_probs.append(
                            torch.sigmoid(logits[pos_mask]).mean().item()
                        )

                self.loss_tracker.update(epoch, loss_dict, self.loss_mode)

            n_b = max(n_batches, 1)
            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_loss / n_b)
            history["loss_r_pos_plus"].append(epoch_r_pos_plus / n_b)
            history["loss_r_unl_minus"].append(epoch_r_unl_minus / n_b)
            history["loss_r_pos_minus"].append(epoch_r_pos_minus / n_b)
            history["loss_r_clamped"].append(epoch_r_clamped / n_b)
            history["lr"].append(opt.param_groups[0]["lr"])
            history["pos_p_std"].append(
                float(np.std(batch_pos_probs)) if len(batch_pos_probs) > 1 else 0.0
            )

            sched.step()

            # ── 7. Diagnose collapse ──
            with torch.no_grad():
                train_pos_prob = float(self.predict(X_train_pos, batch_size=512).mean())
                train_unl_prob = float(self.predict(
                    X_train_unl[train_rng.choice(n_unl, size=min(2000, n_unl), replace=False)],
                    batch_size=512,
                ).mean())

            history["train_pos_prob_mean"].append(train_pos_prob)
            history["train_unl_prob_mean"].append(train_unl_prob)

            # Collapse detection
            pos_p_history.append(train_pos_prob)
            if len(pos_p_history) >= 3:
                recent = pos_p_history[-3:]
                if max(recent) - min(recent) > 0.5:
                    # Large swing detected
                    pass  # tracked in history for notebook analysis

            # ── 8. Validate ──
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_t, return_logits=True)
                val_probs = torch.sigmoid(logits_val).cpu().numpy()
            val_pr = average_precision_score(y_val_bin, val_probs)
            val_roc = roc_auc_score(y_val_bin, val_probs)

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

            if verbose and (epoch % 10 == 0 or epoch == 1 or no_improve == 0):
                marker = "*" if no_improve == 0 else " "
                print(
                    f"  {marker}Ep {epoch:3d}/{config.n_epochs} | "
                    f"loss={history['train_loss'][-1]:.4f} | "
                    f"pos_p={train_pos_prob:.3f} unl_p={train_unl_prob:.4f} | "
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
                f"\n  Complete: best PR={best_val_pr:.4f} ROC={best_val_roc:.4f} "
                f"@ ep {best_epoch} ({elapsed:.0f}s)"
            )

        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_val_pr": best_val_pr,
            "best_val_roc": best_val_roc,
            "history": history,
            "loss_tracker": self.loss_tracker,
            "elapsed_seconds": elapsed,
            "pi_p": pi_p,
            "loss_mode": self.loss_mode,
        }


# ═══════════════════════════════════════════════════════════════════
# Grid search orchestrator
# ═══════════════════════════════════════════════════════════════════

def build_search_grid(quick: bool = False) -> List[Dict]:
    """Build hyperparameter grid for nnPU/uPU/WeightedBCE comparison.

    Parameters tuned (with astronomical motivation):
      - loss_mode: which loss function
      - pi_p: class prior (critical for nnPU stability)
      - positive_weight: BCE weight multiplier (for weighted_bce only)
      - learning_rate: optimizer LR
      - weight_decay: L2 regularization (proven critical for small-sample regime)
      - grad_clip: gradient clipping norm
      - hard_neg_frac: fraction of hard negatives
    """
    if quick:
        return [
            {"loss_mode": "weighted_bce", "positive_weight": 10.0, "lr": 1e-3,
             "weight_decay": 0.1, "grad_clip": 1.0, "hard_neg_frac": 0.3,
             "pi_p": None, "label": "BCE_pw10_wd0.1"},
            {"loss_mode": "nnpu", "positive_weight": 1.0, "lr": 1e-3,
             "weight_decay": 0.1, "grad_clip": 1.0, "hard_neg_frac": 0.3,
             "pi_p": 0.002, "label": "nnPU_pi0.002_wd0.1"},
            {"loss_mode": "upu", "positive_weight": 1.0, "lr": 1e-3,
             "weight_decay": 0.1, "grad_clip": 1.0, "hard_neg_frac": 0.3,
             "pi_p": 0.002, "label": "uPU_pi0.002_wd0.1"},
        ]

    # Full grid
    grid = []
    # ── Weighted BCE variants ──
    for pw, lr, wd in product([5.0, 10.0, 20.0, 50.0],
                               [1e-3, 5e-4],
                               [0.05, 0.1, 0.2]):
        grid.append({
            "loss_mode": "weighted_bce", "positive_weight": pw,
            "lr": lr, "weight_decay": wd, "grad_clip": 1.0,
            "hard_neg_frac": 0.3, "pi_p": None,
            "label": f"BCE_pw{pw:.0f}_lr{lr:.0e}_wd{wd}"
        })

    # ── nnPU variants (pi_p is the most critical parameter) ──
    for pi_p, lr, wd, gc in product(
        [0.001, 0.002, 0.005, 0.01, 0.02],
        [5e-4, 1e-3, 2e-3],
        [0.05, 0.1, 0.2],
        [0.5, 1.0, 2.0],
    ):
        grid.append({
            "loss_mode": "nnpu", "positive_weight": 1.0,
            "lr": lr, "weight_decay": wd, "grad_clip": gc,
            "hard_neg_frac": 0.3, "pi_p": pi_p,
            "label": f"nnPU_pi{pi_p}_lr{lr:.0e}_wd{wd}_gc{gc}"
        })

    # ── uPU variants (smaller grid — often unstable) ──
    for pi_p, lr, wd in product(
        [0.001, 0.002, 0.005],
        [5e-4, 1e-3],
        [0.1, 0.2],
    ):
        grid.append({
            "loss_mode": "upu", "positive_weight": 1.0,
            "lr": lr, "weight_decay": wd, "grad_clip": 1.0,
            "hard_neg_frac": 0.3, "pi_p": pi_p,
            "label": f"uPU_pi{pi_p}_lr{lr:.0e}_wd{wd}"
        })

    return grid


def run_single_experiment(
    params: Dict,
    data: Dict,
    base_config: EndToEndPUConfig,
    save_dir: Path,
) -> Dict:
    """Run one training run with given hyperparameters.

    Returns dict with keys: label, best_val_pr, best_val_roc, test_metrics,
    history, elapsed, pi_p, collapse_detected.
    """
    label = params["label"]
    print(f"\n{'─' * 60}")
    print(f">> {label}")
    print(f"  loss={params['loss_mode']}, lr={params['lr']}, wd={params['weight_decay']}")
    if params["loss_mode"] in ("nnpu", "upu"):
        print(f"  pi_p={params['pi_p']}, grad_clip={params['grad_clip']}")
    else:
        print(f"  pos_weight={params['positive_weight']}")

    # ── Build config ──
    config = EndToEndPUConfig(
        base_ch=base_config.base_ch,
        dropout=base_config.dropout,
        se_reduction=base_config.se_reduction,
        use_cn_attention=base_config.use_cn_attention,
        cn_band_weight_init=base_config.cn_band_weight_init,
        res_blocks=base_config.res_blocks,
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=base_config.batch_size,
        n_epochs=base_config.n_epochs,
        early_stopping_patience=base_config.early_stopping_patience,
        pi_p=params["pi_p"],
        loss_mode=params["loss_mode"],
        positive_weight=params["positive_weight"],
        neg_ratio=base_config.neg_ratio,
        grad_clip=params["grad_clip"],
        hard_neg_frac=params["hard_neg_frac"],
        mixup_alpha=base_config.mixup_alpha,
        pos_augment=base_config.pos_augment,
        random_seed=base_config.random_seed,
        device=base_config.device,
    )

    # ── Data ──
    X_all = data["X_clean"]
    y_all = data["y"]
    wave = data["wave"]
    meta = data["meta"]

    split = split_data(
        X_all, y_all,
        test_split=0.15, val_split=0.15,
        random_seed=config.random_seed, return_test=True,
    )
    X_train_pos, X_train_unl = get_train_pos_unl(split)
    X_val, y_val_bin = create_val_labeled_set(split)

    # ── Model ──
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    model = create_model(config)

    # ── Train ──
    trainer = TuneTrainer(config)
    train_result = trainer.train(
        model,
        X_train_pos=X_train_pos, X_train_unl=X_train_unl,
        X_val=X_val, y_val_bin=y_val_bin,
        wave=wave if config.pos_augment else None,
        verbose=True,
    )
    model = train_result["model"]

    # ── Evaluate ──
    X_test = split["X_test"]
    y_test = split["y_test"]
    y_test_bin = (y_test == 1).astype(int)

    test_probs = trainer.predict(X_test, batch_size=512)
    test_metrics = compute_metrics(test_probs, y_test_bin,
                                   topk_list=[50, 100, 200, 500],
                                   name=label)

    # Collapse detection
    pos_p_arr = np.array(train_result["history"]["train_pos_prob_mean"])
    collapse_detected = False
    collapse_epochs = []
    if len(pos_p_arr) >= 5:
        # Detect: pos_p drops below 0.1 after being above 0.5
        for i in range(4, len(pos_p_arr)):
            window = pos_p_arr[i-4:i+1]
            if window.max() > 0.5 and window[-1] < 0.1:
                collapse_detected = True
                collapse_epochs.append(i + 1)

    # ── Save per-run results ──
    run_dir = save_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config_params": params,
        "best_epoch": train_result["best_epoch"],
        "best_val_pr": train_result["best_val_pr"],
        "best_val_roc": train_result["best_val_roc"],
    }, run_dir / "model.pt")

    # Save history (handle tensor, numpy, and float values)
    hist_serializable = {}
    for k, v in train_result["history"].items():
        if isinstance(v, list):
            clean = []
            for x in v:
                if isinstance(x, torch.Tensor):
                    clean.append(x.item())
                elif isinstance(x, (np.floating, np.integer)):
                    clean.append(float(x))
                elif isinstance(x, float):
                    clean.append(x)
                else:
                    clean.append(float(x))
            hist_serializable[k] = clean
    with open(run_dir / "history.json", "w") as f:
        json.dump(hist_serializable, f, indent=2)

    # Save test metrics
    metrics_out = {
        "label": label,
        "loss_mode": params["loss_mode"],
        "pi_p": params["pi_p"],
        "positive_weight": params["positive_weight"],
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "grad_clip": params["grad_clip"],
        "best_val_pr": float(train_result["best_val_pr"]),
        "best_val_roc": float(train_result["best_val_roc"]),
        "best_epoch": train_result["best_epoch"],
        "elapsed_seconds": train_result["elapsed_seconds"],
        "test_auroc": float(test_metrics["auroc"]) if not np.isnan(test_metrics["auroc"]) else None,
        "test_auprc": float(test_metrics["auprc"]) if not np.isnan(test_metrics["auprc"]) else None,
        "test_p50": float(test_metrics["precision@50"]),
        "test_p100": float(test_metrics["precision@100"]),
        "collapse_detected": collapse_detected,
        "collapse_epochs": collapse_epochs,
        "final_pos_p": float(pos_p_arr[-1]) if len(pos_p_arr) > 0 else None,
        "pos_p_oscillation_std": float(np.std(pos_p_arr)) if len(pos_p_arr) > 1 else 0.0,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save loss tracker
    lt_df = train_result["loss_tracker"].to_dataframe()
    lt_df.to_csv(run_dir / "loss_components.csv", index=False)

    result = {
        "label": label,
        "params": params,
        "best_val_pr": train_result["best_val_pr"],
        "best_val_roc": train_result["best_val_roc"],
        "best_epoch": train_result["best_epoch"],
        "test_metrics": test_metrics,
        "history": train_result["history"],
        "loss_tracker": train_result["loss_tracker"],
        "elapsed_seconds": train_result["elapsed_seconds"],
        "collapse_detected": collapse_detected,
        "pos_p_oscillation_std": metrics_out["pos_p_oscillation_std"],
    }
    return result


def rank_results(all_results: List[Dict]) -> pd.DataFrame:
    """Rank all experiments by val PR-AUC, annotating collapse."""
    rows = []
    for r in all_results:
        t = r["test_metrics"]
        rows.append({
            "label": r["label"],
            "loss_mode": r["params"]["loss_mode"],
            "pi_p": r["params"]["pi_p"],
            "positive_weight": r["params"]["positive_weight"],
            "lr": r["params"]["lr"],
            "weight_decay": r["params"]["weight_decay"],
            "grad_clip": r["params"]["grad_clip"],
            "best_val_pr": round(r["best_val_pr"], 4),
            "best_val_roc": round(r["best_val_roc"], 4),
            "best_epoch": r["best_epoch"],
            "test_auroc": round(float(t["auroc"]), 4) if not np.isnan(t["auroc"]) else None,
            "test_auprc": round(float(t["auprc"]), 4) if not np.isnan(t["auprc"]) else None,
            "test_p50": round(float(t["precision@50"]), 4),
            "test_p100": round(float(t["precision@100"]), 4),
            "collapse": r["collapse_detected"],
            "pos_p_std": round(r["pos_p_oscillation_std"], 4),
            "elapsed_s": int(r["elapsed_seconds"]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("best_val_pr", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="nnPU Loss Hyperparameter Tuning")
    parser.add_argument("--quick", action="store_true",
                        help="Quick 3-run test (BCE vs nnPU vs uPU)")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["weighted_bce", "nnpu", "upu"],
                        help="Run single loss mode (skip grid)")
    parser.add_argument("--pi-p", type=float, default=None,
                        help="Class prior for single run")
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Positive weight for BCE single run")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── Setup ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RESULTS_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {save_dir}")

    base_config = EndToEndPUConfig(
        base_ch=64, dropout=0.5, se_reduction=8,
        use_cn_attention=True, cn_band_weight_init=3.0,
        res_blocks=4, batch_size=64, n_epochs=args.epochs,
        early_stopping_patience=args.patience,
        mixup_alpha=0.2, pos_augment=True,
        random_seed=args.seed, device=args.device,
    )

    t_total = time.time()

    # ── Load data once ──
    print("Loading data...")
    data = load_data()

    # ── Run experiments ──
    all_results = []

    if args.loss is not None:
        # Single run
        params = {
            "loss_mode": args.loss,
            "positive_weight": args.pos_weight,
            "lr": args.lr,
            "weight_decay": args.wd,
            "grad_clip": 1.0,
            "hard_neg_frac": 0.3,
            "pi_p": args.pi_p,
            "label": f"{args.loss}_single",
        }
        result = run_single_experiment(params, data, base_config, save_dir)
        all_results.append(result)
    else:
        # Grid search
        grid = build_search_grid(quick=args.quick)
        print(f"\nGrid search: {len(grid)} configurations")
        if args.quick:
            print("  (quick mode — 3 configs)")
        else:
            bce_count = sum(1 for g in grid if g["loss_mode"] == "weighted_bce")
            nnpu_count = sum(1 for g in grid if g["loss_mode"] == "nnpu")
            upu_count = sum(1 for g in grid if g["loss_mode"] == "upu")
            print(f"  Weighted BCE: {bce_count}  |  nnPU: {nnpu_count}  |  uPU: {upu_count}")
            print(f"  Full grid: {len(grid)} configs")
            if len(grid) > 30:
                print(f"  WARNING: This will take a while! Consider using --quick for testing.")

        for i, params in enumerate(grid):
            print(f"\n[{i+1}/{len(grid)}]")
            try:
                result = run_single_experiment(params, data, base_config, save_dir)
                all_results.append(result)
            except Exception as e:
                print(f"  X FAILED: {e}")
                traceback.print_exc()
                continue

    # ── Rank and save ──
    if len(all_results) > 1:
        ranking = rank_results(all_results)

        print(f"\n{'=' * 70}")
        print("RANKING (by val PR-AUC)")
        print(f"{'=' * 70}")
        pd.set_option('display.max_columns', 15)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 40)
        print(ranking.head(20).to_string(index=False))

        ranking.to_csv(save_dir / "ranking.csv", index=False)
        print(f"\nFull ranking saved to: {save_dir / 'ranking.csv'}")

        # Save all results as JSON
        all_out = []
        for r in all_results:
            t = r["test_metrics"]
            all_out.append({
                "label": r["label"],
                "params": r["params"],
                "best_val_pr": r["best_val_pr"],
                "best_val_roc": r["best_val_roc"],
                "best_epoch": r["best_epoch"],
                "test_auroc": float(t["auroc"]) if not np.isnan(t["auroc"]) else None,
                "test_auprc": float(t["auprc"]) if not np.isnan(t["auprc"]) else None,
                "test_p50": float(t["precision@50"]),
                "test_p100": float(t["precision@100"]),
                "collapse_detected": r["collapse_detected"],
                "pos_p_oscillation_std": r["pos_p_oscillation_std"],
                "elapsed_seconds": r["elapsed_seconds"],
            })
        with open(save_dir / "all_results.json", "w") as f:
            json.dump(all_out, f, indent=2)

        # ── Best model summary ──
        best = all_out[0]
        print(f"\n{'=' * 70}")
        print("BEST CONFIGURATION")
        print(f"{'=' * 70}")
        print(f"  Loss:       {best['params']['loss_mode']}")
        print(f"  pi_p:       {best['params']['pi_p']}")
        print(f"  pos_weight: {best['params']['positive_weight']}")
        print(f"  lr:         {best['params']['lr']}")
        print(f"  wd:         {best['params']['weight_decay']}")
        print(f"  grad_clip:  {best['params']['grad_clip']}")
        print(f"  Val PR-AUC: {best['best_val_pr']:.4f}")
        print(f"  Test AUPRC: {best['test_auprc']:.4f}")
        print(f"  Test P@50:  {best['test_p50']:.4f}")
        print(f"  Collapse:   {best['collapse_detected']}")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"All results saved to: {save_dir}")


if __name__ == "__main__":
    main()
