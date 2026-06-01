"""Active learning loop for PU classification.

Simulates iterative astronomer confirmation:
  1. Start with initial positive set + held-out confirmed pool
  2. Train model → score pool → select K candidates → confirm → retrain
  3. Track metrics at each round

The "confirmation" step simulates what would happen if an astronomer
labeled the top model predictions and confirmed them as true CN stars.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sklearn.metrics import average_precision_score, roc_auc_score

from EndToEndPU.evaluate import compute_metrics
from EndToEndPU.experiments.exp3_active_learning.active_selection import select_candidates


def active_learning_loop(
    config,
    X_all: np.ndarray,
    y_all: np.ndarray,
    meta,
    wave: np.ndarray,
    model_factory,
    verbose: bool = True,
) -> Dict:
    """Run PU active learning with iterative candidate confirmation.

    Parameters
    ----------
    config : ActiveLearningConfig
        Active learning configuration.
    X_all : (N, 700) array
        All spectra.
    y_all : (N,) array
        Labels (1=CN, -1=unlabeled).
    meta : pd.DataFrame
        Metadata for candidate export.
    wave : (700,) array
        Wavelength grid.
    model_factory : callable
        Function that takes (random_seed) and returns a new model.
    verbose : bool

    Returns
    -------
    dict with:
        round_metrics : list of dicts (one per round + initial)
        final_model : trained model after all rounds
        selection_history : list of selected indices per round
        strategy : str
        n_rounds : int
    """
    strategy = config.strategy
    device = config.device
    K = config.k_per_round
    N_rounds = config.n_rounds
    n_initial = config.n_initial_pos
    n_confirm = config.n_confirm_pool
    seed = getattr(config, "random_seed", 42)

    rng = np.random.RandomState(seed)

    # ── 1. Prepare splits ──
    pos_idx = np.where(y_all == 1)[0]
    unl_idx = np.where(y_all == -1)[0]
    rng.shuffle(pos_idx)

    # Split positives: test (hold out) + train initial + confirmed pool
    n_test_pos = max(1, int(len(pos_idx) * 0.15))
    test_pos = pos_idx[:n_test_pos]
    remaining_pos = pos_idx[n_test_pos:]

    # Split remaining into initial and confirmed pool
    rng.shuffle(remaining_pos)
    initial_pos = remaining_pos[:n_initial]
    confirmed_pool = remaining_pos[n_initial:n_initial + n_confirm]

    # Unlabeled split
    rng.shuffle(unl_idx)
    unl_test_n = max(2000, int(len(unl_idx) * 0.15))
    test_unl = unl_idx[:unl_test_n]
    train_unl_all = unl_idx[unl_test_n:]

    # Test set (fixed throughout)
    test_idx = np.concatenate([test_pos, test_unl])
    X_test = X_all[test_idx]
    y_test = (y_all[test_idx] == 1).astype(int)
    n_test_pos_actual = (y_test == 1).sum()

    if verbose:
        print(f"\nActive Learning Setup:")
        print(f"  Test positives: {n_test_pos_actual}")
        print(f"  Initial training positives: {len(initial_pos)}")
        print(f"  Confirmed pool: {len(confirmed_pool)}")
        print(f"  Unlabeled training: {len(train_unl_all):,}")
        print(f"  Rounds: {N_rounds}  K per round: {K}")
        print(f"  Strategy: {strategy}")

    # ── 2. Active learning loop ──
    current_pos = initial_pos.copy()
    confirmed_so_far = []
    selection_history = []
    round_metrics = []

    # For committee strategy, train multiple models
    committee_models = []

    X_pos_pool = X_all[confirmed_pool]

    for round_num in range(N_rounds + 1):
        n_current_pos = len(current_pos)
        if verbose:
            print(f"\n{'─' * 48}")
            print(f"Round {round_num}/{N_rounds} ({n_current_pos} training positives)")
            print(f"{'─' * 48}")

        # Train model on current positives
        X_train_pos = X_all[current_pos]
        X_train_unl = X_all[train_unl_all]

        model = model_factory(seed + round_num * 1000)
        trainer = _build_trainer(model, config)

        train_result = trainer.train(
            X_train_pos=X_train_pos,
            X_train_unl=X_train_unl,
            X_val=X_test,  # use test set as validation (small positives)
            y_val_bin=y_test.astype(np.float32),
            wave=wave,
            verbose=False,
        )
        model = train_result["model"]

        # Evaluate on test set
        test_metrics = evaluate_on_test(model, X_test, y_test, device)
        test_metrics["round"] = round_num
        test_metrics["n_pos_train"] = n_current_pos
        round_metrics.append(test_metrics)

        if verbose:
            print(f"  Test AUROC={test_metrics['auroc']:.4f}  "
                  f"AUPRC={test_metrics['auprc']:.4f}  "
                  f"P@50={test_metrics.get('precision@50', 0):.3f}")

        if round_num == N_rounds:
            break  # No more selection after final round

        # Select K candidates from confirmed pool
        remaining_confirmed = np.array([i for i in range(len(confirmed_pool))
                                        if i not in confirmed_so_far])
        if len(remaining_confirmed) < K:
            if verbose:
                print(f"  Only {len(remaining_confirmed)} remaining in confirmed pool, stopping early")
            break

        X_remaining = X_pos_pool[remaining_confirmed]

        if strategy == "committee":
            # Train additional committee members
            committee = [model]
            for ci in range(1, config.n_committee):
                m2 = model_factory(seed + round_num * 1000 + ci * 100)
                t2 = _build_trainer(m2, config)
                t2.train(
                    X_train_pos=X_train_pos, X_train_unl=X_train_unl,
                    X_val=X_test, y_val_bin=y_test.astype(np.float32),
                    wave=wave, verbose=False,
                )
                committee.append(m2)
            selected_in_pool = select_candidates(
                "committee", committee, X_remaining, K, device,
                rng=rng,
            )
        else:
            selected_in_pool = select_candidates(
                strategy, model, X_remaining, K, device,
                rng=rng,
            )

        # Map back to confirmed_pool indices
        selected_confirmed_indices = remaining_confirmed[selected_in_pool]

        # Add to training positives
        current_pos = np.concatenate([current_pos, confirmed_pool[selected_confirmed_indices]])
        confirmed_so_far.extend(selected_confirmed_indices.tolist())
        selection_history.append({
            "round": round_num,
            "selected_indices": selected_confirmed_indices.tolist(),
            "n_added": len(selected_confirmed_indices),
        })

        if verbose:
            selected_labels = y_all[confirmed_pool[selected_confirmed_indices]]
            n_true_pos = (selected_labels == 1).sum()
            print(f"  Selected: {n_true_pos}/{len(selected_confirmed_indices)} true positives "
                  f"(purity={n_true_pos/len(selected_confirmed_indices):.2f})")

    return {
        "round_metrics": round_metrics,
        "final_model": model,
        "selection_history": selection_history,
        "strategy": strategy,
        "n_rounds": N_rounds,
        "n_initial": n_initial,
        "n_confirm": n_confirm,
        "k_per_round": K,
        "test_positives": n_test_pos_actual,
    }


@torch.no_grad()
def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray, device: str) -> Dict:
    """Compute test metrics for a model.

    Returns dict with auroc, auprc, precision@K, recall@K.
    """
    model.eval()
    model.to(device)
    X_t = torch.from_numpy(X_test).float()
    preds = []
    for i in range(0, len(X_t), 512):
        xb = X_t[i:i + 512].to(device)
        preds.append(model(xb).cpu().numpy())
    probs = np.concatenate(preds)
    return compute_metrics(probs, y_test)


def _build_trainer(model, config):
    """Build a trainer for the active learning round."""
    from EndToEndPU.trainer import EndToEndPUTrainer
    from EndToEndPU.config import EndToEndPUConfig

    train_config = EndToEndPUConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        early_stopping_patience=config.early_stopping_patience,
        loss_mode=config.loss_mode,
        positive_weight=config.positive_weight,
        neg_ratio=config.neg_ratio,
        grad_clip=config.grad_clip,
        hard_neg_frac=config.hard_neg_frac,
        mixup_alpha=config.mixup_alpha,
        pos_augment=config.pos_augment,
        random_seed=getattr(config, "random_seed", 42),
        device=config.device,
        use_cn_attention=config.use_cn_attention,
    )
    return EndToEndPUTrainer(model, train_config)
