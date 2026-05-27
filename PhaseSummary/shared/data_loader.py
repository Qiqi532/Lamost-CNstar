"""PhaseSummary 本地数据预加载模块。

将预处理后的数据缓存到 PhaseSummary/_cache/，避免跨文件夹依赖 ML/_cache/。
首次运行自动从 ML/_cache 复制（如存在），否则从头构建。

使用方式:
    from PhaseSummary.shared.data_loader import ensure_cache, BAND_DEFS
    data = ensure_cache()
    X_clean = data['X_clean']
    stars_clean = data['stars_clean']
    feature_df = data['feature_df']
    common_wave = data['common_wave']
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits as afits

warnings.filterwarnings("ignore")

# ── 路径设置 ──────────────────────────────────────────────────────────
_CURRENT_DIR = Path(__file__).resolve().parent  # PhaseSummary/shared/
_PHASE_SUMMARY_DIR = _CURRENT_DIR.parent        # PhaseSummary/
_PROJECT_ROOT = _PHASE_SUMMARY_DIR.parent       # Lamost/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CACHE_DIR = _PHASE_SUMMARY_DIR / "_cache"
_SPECTRA_CACHE = _CACHE_DIR / "X_clean.npy"
_RAW_SPECTRA_CACHE = _CACHE_DIR / "X_raw.npy"
_STARS_CACHE = _CACHE_DIR / "stars_clustered.pkl"
_FEATURE_CACHE = _CACHE_DIR / "feature_df.pkl"

# ML 缓存（作为备选来源，避免从头计算）
_ML_CACHE_DIR = _PROJECT_ROOT / "ML" / "_cache"


def _try_load_from_ml_cache() -> bool:
    """尝试从 ML/_cache 复制数据到本地 _cache。成功返回 True。"""
    if not _ML_CACHE_DIR.exists():
        return False
    ml_spectra = _ML_CACHE_DIR / "X_clean.npy"
    ml_stars = _ML_CACHE_DIR / "stars_clustered.pkl"
    ml_feature = _ML_CACHE_DIR / "feature_df.pkl"
    if not (ml_spectra.exists() and ml_stars.exists() and ml_feature.exists()):
        return False

    print("从 ML/_cache 复制缓存数据到 PhaseSummary/_cache/ ...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(ml_spectra, _SPECTRA_CACHE)
    shutil.copy2(ml_stars, _STARS_CACHE)
    shutil.copy2(ml_feature, _FEATURE_CACHE)
    return True


def _load_raw_spectrum(filepath: str, rv: float, common_wave: np.ndarray,
                        c_kms: float = 300000.0) -> Optional[np.ndarray]:
    """读取单条原始光谱（未归一化），RV校正后插值到common_wave。"""
    try:
        with afits.open(filepath, memmap=False) as hdul:
            if len(hdul) < 2 or hdul[1].data is None or len(hdul[1].data) == 0:
                return None
            row_data = hdul[1].data[0]
            wave = np.asarray(row_data["WAVELENGTH"], dtype=float)
            flux = np.asarray(row_data["FLUX"], dtype=float)
    except Exception:
        return None

    if not np.isfinite(rv):
        rv = 0.0
    wave_rest = wave / (1.0 + float(rv) / c_kms)

    finite_mask = np.isfinite(wave_rest) & np.isfinite(flux)
    wave_rest = wave_rest[finite_mask]
    flux = flux[finite_mask]

    if len(wave_rest) < 20:
        return None

    order = np.argsort(wave_rest)
    wave_rest = wave_rest[order]
    flux = flux[order]

    wave_rest, unique_idx = np.unique(wave_rest, return_index=True)
    flux = flux[unique_idx]

    if len(wave_rest) < 20:
        return None

    flux_interp = np.interp(common_wave, wave_rest, flux)
    return flux_interp.astype(np.float32)


def _build_raw_cache(stars_clean: pd.DataFrame, common_wave: np.ndarray,
                     show_progress: bool = True) -> np.ndarray:
    """从FITS文件读取所有恒星的原始（未归一化）光谱。"""
    n = len(stars_clean)
    X_raw = np.empty((n, len(common_wave)), dtype=np.float32)

    filepaths = stars_clean["filepath"].values
    rvs = stars_clean["rv"].values

    iterator = range(n)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=n, desc="Loading raw spectra")
        except Exception:
            pass

    fail_count = 0
    for i in iterator:
        # filepath 可能是相对路径（如 Data/dr13_new/...），拼到 _PROJECT_ROOT
        fp = str(filepaths[i])
        if not os.path.isabs(fp):
            fp = str(_PROJECT_ROOT / fp)
        raw = _load_raw_spectrum(fp, float(rvs[i]), common_wave)
        if raw is not None:
            X_raw[i] = raw
        else:
            X_raw[i] = np.nan
            fail_count += 1

    if fail_count > 0:
        print(f"  [WARNING] {fail_count}/{n} raw spectra failed to load (filled with NaN)")

    return X_raw


def _auto_detect_path(path: str) -> str:
    """若 path 不存在，尝试加 Data/ 前缀（适配文件集中到 Data/ 的布局）。"""
    if os.path.exists(path):
        return path
    alt = os.path.join("Data", path)
    if os.path.exists(alt):
        return alt
    return path  # 保持原路径，让下游报清晰错误


def _build_cache_from_scratch(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[List[str]] = None,
):
    """从头构建缓存（仅在无现成缓存时调用）。"""
    from ML.utils import (
        load_and_preprocess, compute_masked_clustering, compute_features,
    )

    stars_csv = _auto_detect_path(stars_csv)
    spectra_folder = _auto_detect_path(spectra_folder)

    if cn_catalogs is None:
        _default_catalogs = ["CNstar.csv", "FT_cands.csv"]
        cn_catalogs = [_auto_detect_path(cat) for cat in _default_catalogs]
    else:
        cn_catalogs = [_auto_detect_path(cat) for cat in cn_catalogs]

    data = load_and_preprocess(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=cn_catalogs,
        show_progress=True,
    )
    X_clean = data["X_clean"]
    stars_clean = data["stars_clean"]
    common_wave = data["common_wave"]

    print(f"\nPreprocessing summary: {data['pipe_summary']}")
    print(f"Label report: {data['label_report']}")

    stars_clean = compute_masked_clustering(
        X_clean=X_clean,
        stars_clean=stars_clean,
        common_wave=common_wave,
        K=180,
    )

    feature_df = compute_features(
        X_clean=X_clean,
        stars_clean=stars_clean,
        common_wave=common_wave,
        n_pca=3,
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_SPECTRA_CACHE, X_clean)
    stars_clean.to_pickle(_STARS_CACHE)
    feature_df.to_pickle(_FEATURE_CACHE)

    return X_clean, stars_clean, feature_df, common_wave


def ensure_cache(
    force_recompute: bool = False,
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[List[str]] = None,
) -> Dict:
    """加载或构建预处理数据（缓存到 PhaseSummary/_cache/）。

    首次调用时自动从 ML/_cache 复制（如存在），避免重新处理 FITS。
    后续调用直接从本地缓存加载，仅需数秒。

    Parameters
    ----------
    force_recompute : bool
        强制从头重建缓存（忽略已有数据）。
    stars_csv, spectra_folder, cn_catalogs :
        仅在首次构建缓存时生效。

    Returns
    -------
    dict with keys: X_clean, X_raw, stars_clean, feature_df, common_wave
    """
    if not force_recompute and _SPECTRA_CACHE.exists() and _STARS_CACHE.exists() and _FEATURE_CACHE.exists():
        print("从 PhaseSummary 本地缓存加载...")
        X_clean = np.load(_SPECTRA_CACHE).astype(np.float32)
        stars_clean = pd.read_pickle(_STARS_CACHE)
        feature_df = pd.read_pickle(_FEATURE_CACHE)
        common_wave = np.arange(3800.0, 4500.0, 1.0, dtype=np.float64)
        print(f"  X_clean: {X_clean.shape}")
        print(f"  stars_clean: {len(stars_clean)} rows")
        print(f"  feature_df: {feature_df.shape}")
        print(f"  Known CN stars: {(stars_clean['label'] == 1).sum()}")

        # 原始光谱缓存
        if _RAW_SPECTRA_CACHE.exists():
            X_raw = np.load(_RAW_SPECTRA_CACHE).astype(np.float32)
            print(f"  X_raw: {X_raw.shape}")
        else:
            print("  构建原始光谱缓存 (X_raw)...")
            X_raw = _build_raw_cache(stars_clean, common_wave, show_progress=True)
            np.save(_RAW_SPECTRA_CACHE, X_raw)

        return {
            "X_clean": X_clean,
            "X_raw": X_raw,
            "stars_clean": stars_clean,
            "feature_df": feature_df,
            "common_wave": common_wave,
        }

    if not force_recompute and _try_load_from_ml_cache():
        return ensure_cache(force_recompute=False)

    print("Building cache from scratch (this may take a few minutes)...")
    t0 = time.time()

    X_clean, stars_clean, feature_df, common_wave = _build_cache_from_scratch(
        stars_csv=stars_csv,
        spectra_folder=spectra_folder,
        cn_catalogs=cn_catalogs,
    )

    elapsed = time.time() - t0
    print(f"\nCache built in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  X_clean: {X_clean.shape}")
    print(f"  stars_clean: {len(stars_clean)} rows")
    print(f"  Known CN stars: {(stars_clean['label'] == 1).sum()}")

    # 构建原始光谱缓存
    print("  构建原始光谱缓存 (X_raw)...")
    X_raw = _build_raw_cache(stars_clean, common_wave, show_progress=True)
    np.save(_RAW_SPECTRA_CACHE, X_raw)

    return {
        "X_clean": X_clean,
        "X_raw": X_raw,
        "stars_clean": stars_clean,
        "feature_df": feature_df,
        "common_wave": common_wave,
    }


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
