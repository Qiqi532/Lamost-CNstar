"""Local data loading with caching for EndToEndPU.

Copies preprocessed data from ML/_cache/ on first use, caches to
EndToEndPU/_cache/ for subsequent fast loads.  Follows the same
cache-isolation pattern as LabelSpreading/ and PhaseSummary/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ──
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CACHE_DIR = _CURRENT_DIR / "_cache"
_ML_CACHE_DIR = _PROJECT_ROOT / "ML" / "_cache"


def _try_copy_from_ml_cache() -> bool:
    """Copy preprocessed data from ML/_cache/ to local _cache/. Returns True on success."""
    required = ["X_clean.npy", "stars_clustered.pkl", "feature_df.pkl"]
    if not _ML_CACHE_DIR.exists():
        return False
    if not all((_ML_CACHE_DIR / f).exists() for f in required):
        return False

    print("Copying cached data from ML/_cache → EndToEndPU/_cache/ ...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in required:
        shutil.copy2(_ML_CACHE_DIR / f, _CACHE_DIR / f)
    return True


def _load_stars_compat(path):
    """Load stars DataFrame with compatibility for older pandas pickle format.

    Tries multiple strategies: direct pickle, CSV fallback.
    """
    import pandas as pd
    from pathlib import Path
    path = Path(path)

    # Strategy 1: Try direct read_pickle first
    try:
        stars = pd.read_pickle(str(path))
        return stars
    except Exception:
        pass

    # Strategy 2: Try CSV fallback
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        print(f"  (loading from CSV fallback: {csv_path})")
        stars = pd.read_csv(str(csv_path))
        return stars

    # Strategy 3: Try patched pickle load
    try:
        import pandas.core.arrays.string_ as _smod
        _orig_init = _smod.StringDtype.__init__
        _smod.StringDtype.__init__ = lambda self, *a, **kw: (
            object.__setattr__(self, 'storage', a[0] if a and isinstance(a[0], str)
                               else kw.get('storage', None))
        )
        try:
            stars = pd.read_pickle(str(path))
            print(f"  (loaded with compat patch)")
            return stars
        finally:
            _smod.StringDtype.__init__ = _orig_init
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot load stars from {path}. Tried pickle, CSV, and compat modes. "
        f"Current pandas: {pd.__version__}"
    )


def load_data(
    force_reload: bool = False,
) -> Dict:
    """Load or build cached data for EndToEndPU.

    Returns
    -------
    dict with keys:
        X_clean  — (N, 700) float32 continuum-normalised spectra.
        y        — (N,) int labels: 1=CN star, -1=unlabeled.
        meta     — pd.DataFrame with stellar params and metadata.
        wave     — (700,) wavelength grid.
        feature_df — pd.DataFrame with 14-D features.
    """
    spectra_cache = _CACHE_DIR / "X_clean.npy"
    stars_cache = _CACHE_DIR / "stars_clustered.pkl"

    if not force_reload and spectra_cache.exists() and stars_cache.exists():
        print("Loading from EndToEndPU/_cache/ ...")
        X_clean = np.load(spectra_cache).astype(np.float32)
        stars = _load_stars_compat(stars_cache)
    else:
        if not force_reload and _try_copy_from_ml_cache():
            return load_data(force_reload=False)
        stars = _load_stars_compat(_ML_CACHE_DIR / "stars_clustered.pkl")

    y = stars["label"].values.astype(int)
    wave = np.arange(3800.0, 4500.0, 1.0, dtype=np.float64)

    # Load feature_df if available
    feature_cache = _CACHE_DIR / "feature_df.pkl"
    if feature_cache.exists():
        try:
            feature_df = pd.read_pickle(feature_cache)
        except Exception:
            csv_fallback = _CACHE_DIR / "feature_df.csv"
            if csv_fallback.exists():
                feature_df = pd.read_csv(str(csv_fallback))
            else:
                feature_df = None
    else:
        feature_df = None

    n_pos = int((y == 1).sum())
    n_total = len(y)
    pi_p = n_pos / n_total

    print(f"  X_clean: {X_clean.shape}")
    print(f"  Stars:   {n_total:,} total  |  {n_pos} known CN  |  "
          f"π_p = {pi_p:.5f}")
    print(f"  Wave:    {wave[0]:.0f}–{wave[-1]:.0f} A  ({len(wave)} pixels)")

    return {
        "X_clean": X_clean,
        "y": y,
        "meta": stars,
        "wave": wave,
        "feature_df": feature_df,
        "pi_p": pi_p,
    }


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_split: float = 0.15,
    val_split: float = 0.15,
    random_seed: int = 42,
    return_test: bool = True,
) -> Dict:
    """Stratified train/val(/test) split preserving positive samples.

    Since positives are extremely rare (73), splitting is done by:
    1. Split positives into test, val, train.
    2. Split unlabeled into corresponding portions.

    Args:
        X: (N, 700) spectra.
        y: (N,) labels (1=CN, -1=unlabeled).
        test_split: fraction of positives held out for testing.
        val_split: fraction of remaining positives used for validation.
        random_seed: random seed.
        return_test: if False, all data goes to train+val (no test holdout).

    Returns:
        dict with train/val/test splits (as arrays and masks).
    """
    rng = np.random.RandomState(random_seed)

    pos_idx = np.where(y == 1)[0]
    unl_idx = np.where(y == -1)[0]

    rng.shuffle(pos_idx)

    if return_test and test_split > 0:
        n_test = max(1, int(len(pos_idx) * test_split))
        test_pos_idx = pos_idx[:n_test]
        remaining_pos_idx = pos_idx[n_test:]
    else:
        n_test = 0
        test_pos_idx = np.array([], dtype=int)
        remaining_pos_idx = pos_idx

    n_val = max(1, int(len(remaining_pos_idx) * val_split))
    val_pos_idx = remaining_pos_idx[:n_val]
    train_pos_idx = remaining_pos_idx[n_val:]

    # ── Check for label leakage via UID ──
    # (Not implemented here — if uid column exists, we'd check)

    # Split unlabeled
    # Use a fraction roughly matching the positive split for test unlabeled
    unl_n = len(unl_idx)
    if return_test and n_test > 0:
        unl_test_n = max(2000, int(unl_n * test_split))
    else:
        unl_test_n = 0
    unl_val_n = max(1500, int(unl_n * val_split))
    unl_train_n = unl_n - unl_test_n - unl_val_n

    rng.shuffle(unl_idx)
    test_unl_idx = unl_idx[:unl_test_n] if unl_test_n > 0 else np.array([], dtype=int)
    val_unl_idx = unl_idx[unl_test_n:unl_test_n + unl_val_n]
    train_unl_idx = unl_idx[unl_test_n + unl_val_n:]

    # Assemble
    train_idx = np.concatenate([train_pos_idx, train_unl_idx])
    val_idx = np.concatenate([val_pos_idx, val_unl_idx])
    if return_test and n_test > 0:
        test_idx = np.concatenate([test_pos_idx, test_unl_idx])
    else:
        test_idx = np.array([], dtype=int)

    # Shuffle train and val
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    result = {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "train_pos_idx": np.where(y[train_idx] == 1)[0],  # indices within X_train
        "train_unl_idx": np.where(y[train_idx] == -1)[0],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "val_pos_mask": y[val_idx] == 1,
        "val_unl_mask": y[val_idx] == -1,
    }

    if return_test and n_test > 0:
        result.update({
            "X_test": X[test_idx],
            "y_test": y[test_idx],
            "test_pos_mask": y[test_idx] == 1,
            "test_unl_mask": y[test_idx] == -1,
        })
    else:
        result.update({
            "X_test": None,
            "y_test": None,
            "test_pos_mask": None,
            "test_unl_mask": None,
        })

    # Print summary
    print(f"\nData split (seed={random_seed}):")
    print(f"  Train: {len(train_idx):,} total  |  "
          f"{len(train_pos_idx)} pos  |  {len(train_unl_idx):,} unl")
    print(f"  Val:   {len(val_idx):,} total  |  "
          f"{len(val_pos_idx)} pos  |  {len(val_unl_idx):,} unl")
    if return_test and n_test > 0:
        print(f"  Test:  {len(test_idx):,} total  |  "
              f"{len(test_pos_idx)} pos  |  {len(test_unl_idx):,} unl")

    return result


def get_train_pos_unl(
    split: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training positives and unlabeled arrays from split dict."""
    train_pos = split["X_train"][split["train_pos_idx"]]
    train_unl = split["X_train"][split["train_unl_idx"]]
    return train_pos, train_unl


def create_val_labeled_set(split: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Create validation set with binary labels (1=pos, 0=unl) suitable for PR-AUC.

    Returns (X_val, y_val_binary) where y_val_binary ∈ {0, 1}.
    """
    X_val = split["X_val"]
    y_val = split["y_val"]
    # Convert -1 → 0 (unlabeled treated as negative for evaluation)
    y_val_bin = (y_val == 1).astype(np.float32)
    return X_val, y_val_bin
