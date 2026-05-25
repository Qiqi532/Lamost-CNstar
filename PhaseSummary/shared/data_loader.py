"""Shared data loading module for Phase Summary notebooks.

Provides unified data loading with caching, shared across all 4 sub-project notebooks.
Uses ML/_cache/ for persistence - only recomputes on first run.

Key outputs:
  - X_clean: (N, 700) continuum-normalized spectra matrix
  - stars_clean: metadata DataFrame with labels, cluster IDs, KNN neighbors
  - feature_df: 14-D feature DataFrame
  - common_wave: (700,) wavelength grid (3800-4500Å)
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CACHE_DIR = _PROJECT_ROOT / "ML" / "_cache"
_SPECTRA_CACHE = _CACHE_DIR / "X_clean.npy"
_STARS_CACHE = _CACHE_DIR / "stars_clustered.pkl"
_FEATURE_CACHE = _CACHE_DIR / "feature_df.pkl"


def ensure_cache(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    force_recompute: bool = False,
) -> Dict:
    """Load or build cached preprocessed data.

    On first call, runs the full pipeline: load → normalize → label →
    dedup → anomaly filter → masked clustering → feature engineering.
    Results are cached in ML/_cache/ for subsequent calls.

    Returns dict with: X_clean, stars_clean, feature_df, common_wave.
    """
    if not force_recompute and _SPECTRA_CACHE.exists() and _STARS_CACHE.exists() and _FEATURE_CACHE.exists():
        print("Loading from cache...")
        X_clean = np.load(_SPECTRA_CACHE).astype(np.float32)
        stars_clean = pd.read_pickle(_STARS_CACHE)
        feature_df = pd.read_pickle(_FEATURE_CACHE)
        common_wave = np.arange(3800.0, 4500.0, 1.0, dtype=np.float64)
        print(f"  X_clean: {X_clean.shape}")
        print(f"  stars_clean: {len(stars_clean)} rows")
        print(f"  feature_df: {feature_df.shape}")
        print(f"  Known CN stars: {(stars_clean['label']==1).sum()}")
        return {"X_clean": X_clean, "stars_clean": stars_clean,
                "feature_df": feature_df, "common_wave": common_wave}

    print("Building cache from scratch (this may take a few minutes)...")
    t0 = time.time()

    from ML.utils import (
        load_and_preprocess, compute_masked_clustering, compute_features,
    )

    data = load_and_preprocess(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=["CNstar.csv", "FT_cands.csv"],
        show_progress=True,
    )
    X_clean = data["X_clean"]
    stars_clean = data["stars_clean"]
    common_wave = data["common_wave"]

    print(f"\nPreprocessing summary: {data['pipe_summary']}")
    print(f"Label report: {data['label_report']}")

    # Masked-band clustering + KNN
    stars_clean = compute_masked_clustering(
        X_clean=X_clean,
        stars_clean=stars_clean,
        common_wave=common_wave,
        K=180,
    )

    # Feature engineering (14-D features)
    feature_df = compute_features(
        X_clean=X_clean,
        stars_clean=stars_clean,
        common_wave=common_wave,
        n_pca=3,
    )

    # Cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_SPECTRA_CACHE, X_clean)
    stars_clean.to_pickle(_STARS_CACHE)
    feature_df.to_pickle(_FEATURE_CACHE)

    elapsed = time.time() - t0
    print(f"\nCache built in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  X_clean: {X_clean.shape}")
    print(f"  stars_clean: {len(stars_clean)} rows")
    print(f"  Known CN stars: {(stars_clean['label']==1).sum()}")

    return {"X_clean": X_clean, "stars_clean": stars_clean,
            "feature_df": feature_df, "common_wave": common_wave}


# ═══════════════════════════════════════════════════════════════════
# CN band definitions (shared across all notebooks)
# ═══════════════════════════════════════════════════════════════════

BAND_DEFS = {
    "CN3839": {"band": (3830, 3883), "blue": (3894, 3910), "red": (4000, 4020)},
    "CN4142": {"band": (4120, 4216), "blue": (4055, 4080), "red": (4240, 4280)},
    "CH4300": {"band": (4285, 4315), "blue": (4240, 4280), "red": (4390, 4460)},
}

MOLECULAR_BAND_RANGES = [(3830, 3883), (4120, 4216), (4285, 4315)]
CN_BAND_NAMES = ["CN3839", "CN4142", "CH4300"]


def cross_validate_candidates(
    candidates_a: pd.DataFrame,
    candidates_b: pd.DataFrame,
    match_col: str = "uid",
    common_only: bool = True,
) -> Dict:
    """Cross-validate candidate lists from two methods.

    Parameters
    ----------
    candidates_a, candidates_b : DataFrames with candidate lists.
        Must have 'uid' column and a probability/score column.
    match_col : column name to match on (default: 'uid').
    common_only : if True, only return intersection.

    Returns dict with overlap statistics and matched candidates.
    """
    if match_col not in candidates_a.columns or match_col not in candidates_b.columns:
        available_a = list(candidates_a.columns)
        available_b = list(candidates_b.columns)
        return {
            "error": f"match_col '{match_col}' not found",
            "cols_a": available_a,
            "cols_b": available_b,
        }

    uids_a = set(candidates_a[match_col].values)
    uids_b = set(candidates_b[match_col].values)
    common = uids_a & uids_b
    only_a = uids_a - uids_b
    only_b = uids_b - uids_a

    matched = candidates_a[candidates_a[match_col].isin(common)].copy()
    matched_b = candidates_b[candidates_b[match_col].isin(common)].copy()

    return {
        "n_a": len(uids_a),
        "n_b": len(uids_b),
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "overlap_rate": len(common) / max(len(uids_a), 1),
        "matched_a": matched,
        "matched_b": matched_b,
        "only_a_uids": only_a,
        "only_b_uids": only_b,
    }


def compute_cn_indices(
    X: np.ndarray,
    common_wave: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CN3839, CN4142, CH4300 band indices for all spectra.

    Returns (cn3839, cn4142, ch4300) arrays of shape (N,).
    """
    from ML.utils import compute_band_index

    n = len(X)
    cn3839 = np.array([compute_band_index(common_wave, X[i],
                        **{k: v for k, v in BAND_DEFS["CN3839"].items()})
                       for i in range(n)])
    cn4142 = np.array([compute_band_index(common_wave, X[i],
                        **{k: v for k, v in BAND_DEFS["CN4142"].items()})
                       for i in range(n)])
    ch4300 = np.array([compute_band_index(common_wave, X[i],
                        **{k: v for k, v in BAND_DEFS["CH4300"].items()})
                       for i in range(n)])
    for arr in [cn3839, cn4142, ch4300]:
        arr[np.isnan(arr)] = np.nanmedian(arr)
    return cn3839, cn4142, ch4300
