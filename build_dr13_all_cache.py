"""为 dr13_all 数据集（feh<-0.7 扩展至 -2.5）构建缓存 —— v2 合并版。

相比 v1 的改进（解决"CN 星减少"的问题）：
1. 光谱多目录合并：Data/dr13_all → Data/dr13_new → Data/CNstars + Data/FT_cands
   主样本缺失的 CN 星光谱可从 dr13_new / CN FITS 兜底找回。
2. 联合 CN 标签匹配（三路取并集）：
   a. CNstar.csv + FT_cands.csv 天球匹配 (1 arcsec)
   b. CNstars/ + FT_cands/ FITS 头文件的 obsid 直接匹配（精确，不受重观测天球偏移影响）
   c. CNstars/ + FT_cands/ FITS 头文件的 RA/DEC 天球匹配 (1 arcsec)

数据来源
--------
- 星表:  Data/stars1.csv   (59,137 行, feh ∈ [-2.5, -0.7])
- 光谱:  Data/dr13_all/ (主) + Data/dr13_new/ (补) + Data/CNstars/ + Data/FT_cands/ (CN 兜底)

生成的缓存 (dr13_all_cache/)
---------------------------
- X_clean.npy           (N, 700) float32  连续谱归一化光谱 (3800–4500 Å)
- stars_clustered.pkl   元数据 + label + masked_cluster_id + neighbor_indices
- feature_df.pkl        14-D 特征

用法
----
    D:/Anaconda/envs/myenv/python.exe build_dr13_all_cache.py            # 完整构建
    D:/Anaconda/envs/myenv/python.exe build_dr13_all_cache.py --subset 5000
    D:/Anaconda/envs/myenv/python.exe build_dr13_all_cache.py --force

后续直接读取:
    from build_dr13_all_cache import load_dr13_all_cache
    data = load_dr13_all_cache()
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

# ── 路径设置 ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent  # Lamost/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CACHE_DIR = _PROJECT_ROOT / "Data" / "dr13_all_cache"
_STARS_CSV = "stars1.csv"
_SPECTRA_FOLDERS = ["dr13_all", "dr13_new"]
_CN_CSV_CATALOGS = ["CNstar.csv", "FT_cands.csv"]
_CN_FITS_DIRS = ["CNstars", "FT_cands"]

COMMON_WAVE = np.arange(3800.0, 4500.0, 1.0, dtype=np.float64)


def _read_cn_fits_headers(dirs) -> pd.DataFrame:
    """读取 CN FITS 头文件，返回 [obsid, ra, dec, lmjd, planid, spid, fiberid]."""
    from astropy.io import fits as afits

    rows = []
    for d in dirs:
        d = _PROJECT_ROOT / "Data" / d
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.fits*")):
            try:
                with afits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                ra = float(hdr.get("RA", np.nan))
                dec = float(hdr.get("DEC", np.nan))
                obsid = str(hdr.get("OBSID", "")).strip()
                lmjd = str(hdr.get("LMJD", "")).strip()
                planid = str(hdr.get("PLANID", "")).strip()
                spid = str(hdr.get("SPID", "")).strip()
                fiberid = str(hdr.get("FIBERID", "")).strip()
                rows.append([obsid, ra, dec, lmjd, planid, spid, fiberid])
            except Exception:
                continue
    return pd.DataFrame(
        rows, columns=["obsid", "ra", "dec", "lmjd", "planid", "spid", "fiberid"]
    )


def label_cn_stars(stars_df: pd.DataFrame, tolerance_arcsec: float = 1.0) -> pd.DataFrame:
    """联合 CN 标签匹配：CSV 天球 + CN FITS obsid + CN FITS 天球，取并集。

    返回带 label 列 (1=CN, -1=unlabeled) 的 stars_df 副本。
    """
    from ML.utils import _auto_detect_path

    stars = stars_df.copy()
    label = pd.Series(-1, index=stars.index)

    if "ra" in stars.columns and "dec" in stars.columns:
        coords_stars = SkyCoord(
            ra=stars["ra"].values * u.deg, dec=stars["dec"].values * u.deg
        )

        # (a) CSV 天球匹配
        for cat in _CN_CSV_CATALOGS:
            p = _auto_detect_path(cat)
            if not os.path.exists(p):
                continue
            try:
                c = pd.read_csv(p)
            except Exception:
                continue
            if "ra" not in c.columns or "dec" not in c.columns:
                continue
            cc = SkyCoord(ra=c["ra"].values * u.deg, dec=c["dec"].values * u.deg)
            idx, sep2d, _ = coords_stars.match_to_catalog_sky(cc)
            hit = sep2d.arcsec <= tolerance_arcsec
            label.iloc[np.where(hit)[0]] = 1

    # (b) + (c) CN FITS 头文件匹配（obsid 精确 + RA/DEC 天球）
    cn_fits = _read_cn_fits_headers(_CN_FITS_DIRS)
    if len(cn_fits) > 0 and "obsid" in stars.columns:
        stars_obsid = stars["obsid"].astype(str).str.strip()
        cn_obsid = cn_fits["obsid"].astype(str).str.strip()
        hit_obsid = stars_obsid.isin(set(cn_obsid) - {""})
        label[hit_obsid.values] = 1

        if "ra" in stars.columns and "dec" in stars.columns:
            coords_stars = SkyCoord(
                ra=stars["ra"].values * u.deg, dec=stars["dec"].values * u.deg
            )
            valid = cn_fits["ra"].notna() & cn_fits["dec"].notna()
            if valid.sum() > 0:
                cc = SkyCoord(
                    ra=cn_fits.loc[valid, "ra"].values * u.deg,
                    dec=cn_fits.loc[valid, "dec"].values * u.deg,
                )
                idx, sep2d, _ = coords_stars.match_to_catalog_sky(cc)
                hit = sep2d.arcsec <= tolerance_arcsec
                label.iloc[np.where(hit)[0]] = 1

    stars["label"] = label.values
    return stars


def build_cache(subset: Optional[int] = None, force: bool = False, show_progress: bool = True):
    """运行完整管线（联合标签 + 多目录光谱）并缓存到 dr13_all_cache/。

    注意：本函数为原始脚本（已丢失）的重建版本。由于多目录光谱合并 +
    连续谱归一化的完整管线较复杂，且当前项目缓存 Data/dr13_all_cache/ 已存在、
    可直接通过 load_dr13_all_cache() 读取，故这里仅保留入口与提示，不做从头重建。
    """
    if _CACHE_DIR.exists() and (_CACHE_DIR / "X_clean.npy").exists() and not force:
        print(f"缓存已存在: {_CACHE_DIR}")
        print("如需重建，请加 --force。")
        return

    raise NotImplementedError(
        "dr13_all 缓存的从头重建路径未完整实现（原始脚本已丢失）。\n"
        "现有缓存 Data/dr13_all_cache/ 可直接通过 load_dr13_all_cache() 读取。\n"
        "如确需从头重建，可参照 PhaseSummary/shared/data_loader.py 的 "
        "_build_cache_from_scratch 与 ML/utils.load_and_preprocess 组合实现。"
    )


def load_dr13_all_cache() -> Dict:
    """加载 dr13_all 缓存 (与 PhaseSummary.ensure_cache 返回结构对齐)。"""
    X_clean_path = _CACHE_DIR / "X_clean.npy"
    stars_path = _CACHE_DIR / "stars_clustered.pkl"
    feat_path = _CACHE_DIR / "feature_df.pkl"

    missing = [p.name for p in (X_clean_path, stars_path, feat_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缓存缺失 {missing}，请先运行 build_dr13_all_cache.py 重建缓存。"
        )

    X_clean = np.load(X_clean_path).astype(np.float32)
    stars_clean = pd.read_pickle(stars_path)
    feature_df = pd.read_pickle(feat_path)
    common_wave = COMMON_WAVE

    n_pos = int((stars_clean["label"] == 1).sum())
    print("从 dr13_all_cache 加载:")
    print(f"  X_clean:      {X_clean.shape}")
    print(f"  stars_clean:  {len(stars_clean):,} 行")
    print(f"  feature_df:   {feature_df.shape}")
    print(f"  已知 CN 星:   {n_pos}")

    return {
        "X_clean": X_clean,
        "stars_clean": stars_clean,
        "feature_df": feature_df,
        "common_wave": common_wave,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 dr13_all 数据缓存 (v2 合并版)")
    parser.add_argument("--subset", type=int, default=None, help="只构建前 N 颗星")
    parser.add_argument("--force", action="store_true", help="强制重建缓存")
    args = parser.parse_args()

    build_cache(subset=args.subset, force=args.force, show_progress=True)
