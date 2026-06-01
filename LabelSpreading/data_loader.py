"""LabelSpreading 本地数据预加载模块。

将预处理后的数据缓存到 LabelSpreading/_cache/，避免每次运行 notebook
都重新加载 FITS 文件（耗时 5-7 分钟）。缓存一次后，后续加载仅需数秒。

使用方式:
    from data_loader import ensure_cache
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径设置 ──────────────────────────────────────────────────────────
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CACHE_DIR = _CURRENT_DIR / "_cache"
_SPECTRA_CACHE = _CACHE_DIR / "X_clean.npy"
_STARS_CACHE = _CACHE_DIR / "stars_clustered.pkl"
_FEATURE_CACHE = _CACHE_DIR / "feature_df.pkl"

# ML 缓存（作为备选来源，避免重复计算）
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

    print("从 ML/_cache 复制缓存数据到本地...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(ml_spectra, _SPECTRA_CACHE)
    shutil.copy2(ml_stars, _STARS_CACHE)
    shutil.copy2(ml_feature, _FEATURE_CACHE)
    return True


def _build_cache_from_scratch(
    stars_csv: str = "stars.csv",
    spectra_folder: str = "dr13_new",
    cn_catalogs: Optional[List[str]] = None,
):
    """从头构建缓存（仅在无现成缓存时调用）。"""
    from ML.utils import (
        load_and_preprocess, compute_masked_clustering, compute_features,
    )

    # 自动检测 Data/ 子目录
    if not os.path.exists(stars_csv) and os.path.exists(os.path.join("Data", stars_csv)):
        stars_csv = os.path.join("Data", stars_csv)
    if not os.path.exists(spectra_folder) and os.path.exists(os.path.join("Data", spectra_folder)):
        spectra_folder = os.path.join("Data", spectra_folder)

    if cn_catalogs is None:
        _default_catalogs = ["CNstar.csv", "FT_cands.csv"]
        cn_catalogs = []
        for cat in _default_catalogs:
            if os.path.exists(cat):
                cn_catalogs.append(cat)
            elif os.path.exists(os.path.join("Data", cat)):
                cn_catalogs.append(os.path.join("Data", cat))
            else:
                cn_catalogs.append(cat)

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
    """加载或构建预处理数据（缓存到 LabelSpreading/_cache/）。

    首次调用时自动从 ML/_cache 复制（如存在），避免重新处理 FITS。
    后续调用直接从本地缓存加载，仅需数秒。

    Parameters
    ----------
    force_recompute : bool
        强制从头重建缓存（忽略已有数据）。
    stars_csv, spectra_folder, cn_catalogs :
        仅在 force_recompute=True 且本地无缓存时生效。

    Returns
    -------
    dict with keys: X_clean, stars_clean, feature_df, common_wave
    """
    if not force_recompute and _SPECTRA_CACHE.exists() and _STARS_CACHE.exists() and _FEATURE_CACHE.exists():
        print("从 LabelSpreading 本地缓存加载...")
        X_clean = np.load(_SPECTRA_CACHE).astype(np.float32)
        stars_clean = pd.read_pickle(_STARS_CACHE)
        feature_df = pd.read_pickle(_FEATURE_CACHE)
        common_wave = np.arange(3800.0, 4500.0, 1.0, dtype=np.float64)
        print(f"  X_clean: {X_clean.shape}")
        print(f"  stars_clean: {len(stars_clean)} rows")
        print(f"  feature_df: {feature_df.shape}")
        print(f"  Known CN stars: {(stars_clean['label'] == 1).sum()}")
        return {
            "X_clean": X_clean,
            "stars_clean": stars_clean,
            "feature_df": feature_df,
            "common_wave": common_wave,
        }

    if not force_recompute and _try_load_from_ml_cache():
        # 复制成功后走本地加载逻辑
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

    return {
        "X_clean": X_clean,
        "stars_clean": stars_clean,
        "feature_df": feature_df,
        "common_wave": common_wave,
    }


# ═══════════════════════════════════════════════════════════════════════
# CN 分子带定义（供 notebooks 直接引用）
# ═══════════════════════════════════════════════════════════════════════

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
    """交叉验证两个方法的候选体列表。

    返回重叠统计和匹配的候选体。
    """
    if match_col not in candidates_a.columns or match_col not in candidates_b.columns:
        return {
            "error": f"match_col '{match_col}' not found",
            "cols_a": list(candidates_a.columns),
            "cols_b": list(candidates_b.columns),
        }

    uids_a = set(candidates_a[match_col].values)
    uids_b = set(candidates_b[match_col].values)
    common = uids_a & uids_b
    only_a = uids_a - uids_b
    only_b = uids_b - uids_a

    matched = candidates_a[candidates_a[match_col].isin(common)].copy()

    return {
        "n_a": len(uids_a),
        "n_b": len(uids_b),
        "n_common": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "overlap_rate": len(common) / max(len(uids_a), 1),
        "matched_a": matched,
        "only_a_uids": only_a,
        "only_b_uids": only_b,
    }
