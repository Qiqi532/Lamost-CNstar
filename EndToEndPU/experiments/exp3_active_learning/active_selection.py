"""Active learning selection strategies for PU classification.

Strategies:
  1. confidence: Select top-K highest predicted probability
  2. uncertainty: Select K with probability closest to 0.5
  3. committee: Select K with highest prediction variance across committee
  4. random: Random selection (baseline)
"""

from typing import List, Optional
import numpy as np


def _predict_probs(model, X: np.ndarray, batch_size: int = 512, device: str = "cuda") -> np.ndarray:
    """Predict probabilities for numpy array."""
    import torch
    model.eval()
    model.to(device)
    X_t = torch.from_numpy(X).float()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def confidence_selection(
    model,
    X_pool: np.ndarray,
    K: int,
    device: str = "cuda",
    excluded_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select top-K by predicted probability (most confident positives).

    Parameters
    ----------
    model : nn.Module
        Trained model.
    X_pool : (N, 700) array
        Pool of candidates to select from.
    K : int
        Number of candidates to select.
    device : str
    excluded_indices : array or None
        Indices to exclude from selection (already selected in prior rounds).

    Returns
    -------
    (K,) array of indices into X_pool, sorted by confidence descending.
    """
    probs = _predict_probs(model, X_pool, device=device)

    # Mask excluded
    if excluded_indices is not None:
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        valid_indices = np.where(mask)[0]
        valid_probs = probs[valid_indices]
        top_k_valid = np.argsort(valid_probs)[-K:][::-1]
        return valid_indices[top_k_valid]
    else:
        return np.argsort(probs)[-K:][::-1]


def uncertainty_selection(
    model,
    X_pool: np.ndarray,
    K: int,
    device: str = "cuda",
    excluded_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select K with probability closest to 0.5 (most uncertain).

    In PU learning, these are the samples the model is least sure about —
    potentially ambiguous CN stars that would benefit from confirmation.
    """
    probs = _predict_probs(model, X_pool, device=device)
    uncertainty = np.abs(probs - 0.5)

    if excluded_indices is not None:
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        valid_indices = np.where(mask)[0]
        valid_unc = uncertainty[valid_indices]
        # Select K with smallest uncertainty distance (closest to 0.5)
        top_k_valid = np.argsort(valid_unc)[:K]
        return valid_indices[top_k_valid]
    else:
        return np.argsort(uncertainty)[:K]


def committee_selection(
    models: list,
    X_pool: np.ndarray,
    K: int,
    device: str = "cuda",
    excluded_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select K by highest prediction variance across committee.

    Query-by-committee: where the committee disagrees most,
    the sample is most informative for labeling.

    Parameters
    ----------
    models : list of nn.Module
        Committee of independently trained models.
    X_pool : (N, 700) array
    K : int
    device : str
    excluded_indices : array or None

    Returns
    -------
    (K,) array of indices with highest prediction variance.
    """
    all_probs = np.array([_predict_probs(m, X_pool, device=device) for m in models])
    variance = all_probs.var(axis=0)

    if excluded_indices is not None:
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        valid_indices = np.where(mask)[0]
        valid_var = variance[valid_indices]
        top_k_valid = np.argsort(valid_var)[-K:][::-1]
        return valid_indices[top_k_valid]
    else:
        return np.argsort(variance)[-K:][::-1]


def random_selection(
    X_pool: np.ndarray,
    K: int,
    rng: np.random.RandomState,
    excluded_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Random selection baseline.

    Parameters
    ----------
    X_pool : (N, 700) array
    K : int
    rng : np.random.RandomState
    excluded_indices : array or None

    Returns
    -------
    (K,) array of random indices.
    """
    if excluded_indices is not None:
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        valid_indices = np.where(mask)[0]
        return rng.choice(valid_indices, size=min(K, len(valid_indices)), replace=False)
    else:
        return rng.choice(len(X_pool), size=K, replace=False)


def select_candidates(
    strategy: str,
    model_or_models,
    X_pool: np.ndarray,
    K: int,
    device: str = "cuda",
    excluded_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Dispatch to the appropriate selection strategy.

    Parameters
    ----------
    strategy : str
        One of: 'confidence', 'uncertainty', 'committee', 'random'.
    model_or_models : nn.Module or list of nn.Module
        Single model (confidence/uncertainty/random) or list (committee).
    X_pool : (N, 700) array
    K : int
    device : str
    excluded_indices : array or None
    rng : np.random.RandomState, required for 'random'

    Returns
    -------
    (K,) array of selected indices into X_pool.
    """
    if strategy == "confidence":
        return confidence_selection(model_or_models, X_pool, K, device, excluded_indices)
    elif strategy == "uncertainty":
        return uncertainty_selection(model_or_models, X_pool, K, device, excluded_indices)
    elif strategy == "committee":
        if not isinstance(model_or_models, list):
            raise ValueError("Committee strategy requires a list of models")
        return committee_selection(model_or_models, X_pool, K, device, excluded_indices)
    elif strategy == "random":
        if rng is None:
            rng = np.random.RandomState(42)
        return random_selection(X_pool, K, rng, excluded_indices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
