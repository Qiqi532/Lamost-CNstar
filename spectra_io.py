"""Reusable spectral I/O and preprocessing utilities for LAMOST-like FITS data.

This module supports:
1) schema audit and numeric coercion
2) CN catalog labeling by coordinate match
3) FITS loading + RV correction + interpolation + normalization
4) uid-based deduplication and anomaly filtering
5) optional wide CSV export for cross-notebook reuse
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
from scipy import signal


DEFAULT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "lmjd",
    "planid",
    "spid",
    "fiberid",
    "rv",
)


def audit_star_table(
    stars_df: pd.DataFrame,
    required_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Check schema and missing rates before loading spectra.

    Returns a dictionary with row count, missing rates, and columns check result.
    """
    if required_cols is None:
        required_cols = DEFAULT_REQUIRED_COLUMNS

    missing_columns = [c for c in required_cols if c not in stars_df.columns]
    missing_rate = {}
    for c in required_cols:
        if c in stars_df.columns:
            missing_rate[c] = float(stars_df[c].isna().mean())

    report = {
        "n_rows": int(len(stars_df)),
        "required_cols": list(required_cols),
        "missing_columns": missing_columns,
        "missing_rate": missing_rate,
    }
    return report


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, stripped column names."""
    out = df.copy()
    out.columns = out.columns.str.lower().str.strip()
    return out


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Coerce selected columns to numeric with NaN for invalid entries."""
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_catalog_input(catalog: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load one catalog from DataFrame or CSV path, with normalized columns."""
    if isinstance(catalog, pd.DataFrame):
        return normalize_column_names(catalog)

    catalog_path = str(catalog)
    out = pd.read_csv(catalog_path)
    return normalize_column_names(out)


def label_stars_by_cn_catalog(
    stars_df: pd.DataFrame,
    cn_catalogs: Sequence[Union[str, Path, pd.DataFrame]],
    tolerance_arcsec: float = 1.0,
    positive_label: int = 1,
    unlabeled_value: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Label stars via sky-coordinate matching to one or more CN catalogs."""
    out = normalize_column_names(stars_df)
    if "label" not in out.columns:
        out["label"] = unlabeled_value

    for c in ("ra", "dec"):
        if c not in out.columns:
            raise ValueError(f"stars_df missing required coordinate column: {c}")

    if len(cn_catalogs) == 0:
        rep = {
            "n_catalog_rows": 0,
            "n_positive": int((out["label"] == positive_label).sum()),
            "n_unlabeled": int((out["label"] == unlabeled_value).sum()),
            "tolerance_arcsec": float(tolerance_arcsec),
        }
        return out, rep

    cn_frames = []
    for one in cn_catalogs:
        cn_one = load_catalog_input(one)
        if {"ra", "dec"}.issubset(cn_one.columns):
            cn_frames.append(cn_one[["ra", "dec"]].copy())

    if len(cn_frames) == 0:
        rep = {
            "n_catalog_rows": 0,
            "n_positive": int((out["label"] == positive_label).sum()),
            "n_unlabeled": int((out["label"] == unlabeled_value).sum()),
            "tolerance_arcsec": float(tolerance_arcsec),
            "warning": "no_valid_cn_catalog_with_ra_dec",
        }
        return out, rep

    cn_all = pd.concat(cn_frames, ignore_index=True).drop_duplicates(subset=["ra", "dec"])
    cn_all = coerce_numeric_columns(cn_all, ["ra", "dec"])
    cn_all = cn_all.dropna(subset=["ra", "dec"]).reset_index(drop=True)

    out = coerce_numeric_columns(out, ["ra", "dec"])
    work = out.dropna(subset=["ra", "dec"]).copy()
    if len(work) == 0 or len(cn_all) == 0:
        rep = {
            "n_catalog_rows": int(len(cn_all)),
            "n_positive": int((out["label"] == positive_label).sum()),
            "n_unlabeled": int((out["label"] == unlabeled_value).sum()),
            "tolerance_arcsec": float(tolerance_arcsec),
            "warning": "empty_valid_coordinates",
        }
        return out, rep

    work = work.reset_index().rename(columns={"index": "__orig_idx__"})

    catalog_coords = SkyCoord(ra=work["ra"].values * u.deg, dec=work["dec"].values * u.deg)
    cn_coords = SkyCoord(ra=cn_all["ra"].values * u.deg, dec=cn_all["dec"].values * u.deg)

    idx, d2d, _ = cn_coords.match_to_catalog_sky(catalog_coords)
    valid_matches = d2d <= (float(tolerance_arcsec) * u.arcsec)

    matched_catalog_idx = idx[valid_matches]
    matched_sep_arcsec = d2d[valid_matches].arcsec

    best_sep: Dict[int, float] = {}
    for i_cat, sep in zip(matched_catalog_idx, matched_sep_arcsec):
        i_cat_int = int(i_cat)
        sep_float = float(sep)
        if (i_cat_int not in best_sep) or (sep_float < best_sep[i_cat_int]):
            best_sep[i_cat_int] = sep_float

    if len(best_sep) > 0:
        matched_idx_work = np.array(sorted(best_sep.keys()), dtype=int)
        matched_idx_orig = work.loc[matched_idx_work, "__orig_idx__"].values
        out.loc[matched_idx_orig, "label"] = positive_label

    rep = {
        "n_catalog_rows": int(len(cn_all)),
        "n_match_unique": int(len(best_sep)),
        "n_positive": int((out["label"] == positive_label).sum()),
        "n_unlabeled": int((out["label"] == unlabeled_value).sum()),
        "tolerance_arcsec": float(tolerance_arcsec),
    }
    return out, rep


def build_spectrum_filepath(row: pd.Series, folder: str) -> str:
    """Build a LAMOST-like spectrum file path from one metadata row."""
    lmjd = int(row["lmjd"])
    planid = str(row["planid"]).strip()
    spid = int(row["spid"])
    fiberid = int(row["fiberid"])
    return os.path.join(folder, f"spec-{lmjd}-{planid}_sp{spid:02d}-{fiberid:03d}.fits.gz")


def _safe_read_wave_flux(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with afits.open(filepath, memmap=False) as hdul:
            if len(hdul) < 2 or hdul[1].data is None or len(hdul[1].data) == 0:
                return None
            row_data = hdul[1].data[0]
            wave = np.asarray(row_data["WAVELENGTH"], dtype=float)
            flux = np.asarray(row_data["FLUX"], dtype=float)
            return wave, flux
    except Exception:
        return None


def read_single_spectrum(
    row: pd.Series,
    common_wave: np.ndarray,
    folder: str,
    c_kms: float = 300000.0,
    min_points: int = 20,
    require_full_coverage: bool = True,
) -> Optional[Tuple[str, np.ndarray]]:
    """Read one FITS spectrum, RV-correct to rest frame, and interpolate.

    Returns
    -------
    (filepath, flux_interp) when successful, else None.
    """
    filepath = build_spectrum_filepath(row, folder)
    if not os.path.exists(filepath):
        return None

    payload = _safe_read_wave_flux(filepath)
    if payload is None:
        return None

    wave, flux = payload

    rv = row.get("rv", 0.0)
    if not np.isfinite(rv):
        rv = 0.0

    wave_rest = wave / (1.0 + float(rv) / c_kms)

    finite_mask = np.isfinite(wave_rest) & np.isfinite(flux)
    wave_rest = wave_rest[finite_mask]
    flux = flux[finite_mask]

    if len(wave_rest) < min_points:
        return None

    order = np.argsort(wave_rest)
    wave_rest = wave_rest[order]
    flux = flux[order]

    wave_rest, unique_idx = np.unique(wave_rest, return_index=True)
    flux = flux[unique_idx]

    if len(wave_rest) < min_points:
        return None

    if require_full_coverage:
        if wave_rest.min() > common_wave[0] or wave_rest.max() < common_wave[-1]:
            return None

    flux_interp = np.interp(common_wave, wave_rest, flux)
    if not np.all(np.isfinite(flux_interp)):
        return None

    return filepath, flux_interp.astype(np.float32)


def load_spectra(
    stars_df: pd.DataFrame,
    common_wave: np.ndarray,
    folder: str,
    show_progress: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    """Batch load spectra from metadata table.

    Returns
    -------
    datacube: np.ndarray
        Shape (n_valid, n_wave).
    filepaths: list[str]
        Loaded file path per row in datacube.
    valid_indices: np.ndarray
        Row indices in the original stars_df used in datacube.
    drop_log: pd.DataFrame
        Failed rows with reason.
    """
    iterator: Iterable[Tuple[int, pd.Series]] = stars_df.iterrows()
    if show_progress:
        try:
            from tqdm.notebook import tqdm  # type: ignore

            iterator = tqdm(iterator, total=len(stars_df))
        except Exception:
            pass

    datacube: List[np.ndarray] = []
    filepaths: List[str] = []
    valid_indices: List[int] = []
    drop_rows: List[Dict[str, object]] = []

    for idx, row in iterator:
        try:
            fp = build_spectrum_filepath(row, folder)
        except Exception:
            drop_rows.append({"index": int(idx), "reason": "invalid_path_fields"})
            continue

        if not os.path.exists(fp):
            drop_rows.append({"index": int(idx), "filepath": fp, "reason": "file_not_found"})
            continue

        out = read_single_spectrum(row, common_wave=common_wave, folder=folder)
        if out is None:
            drop_rows.append({"index": int(idx), "filepath": fp, "reason": "read_or_qc_failed"})
            continue

        filepath, flux_interp = out
        datacube.append(flux_interp)
        filepaths.append(filepath)
        valid_indices.append(int(idx))

    datacube_arr = np.asarray(datacube, dtype=np.float32)
    valid_indices_arr = np.asarray(valid_indices, dtype=int)
    drop_log = pd.DataFrame(drop_rows)
    return datacube_arr, filepaths, valid_indices_arr, drop_log


def smooth_rescale_array(
    flux: np.ndarray,
    n_smooth: int = 0,
    n_rescale: int = 80,
    deg: int = 3,
    gauss_kernel: bool = True,
) -> np.ndarray:
    """Continuum-like smoothing/rescaling used in notebook pipeline."""
    arr = np.asarray(flux, dtype=float)

    if n_smooth > 0:
        window_length = 2 * n_smooth + 1
        if window_length < len(arr):
            arr_smooth = signal.savgol_filter(arr, window_length, deg)
        else:
            arr_smooth = arr.copy()
    else:
        arr_smooth = arr.copy()

    if gauss_kernel:
        x = np.linspace(-3, 3, 2 * n_rescale + 1)
        kernel = np.exp(-0.5 * x**2)
    else:
        kernel = np.ones(2 * n_rescale + 1, dtype=float)
    kernel = kernel / np.sum(kernel)

    if len(arr_smooth) > len(kernel):
        continuum = np.convolve(arr_smooth, kernel, mode="same")
    else:
        continuum = np.full_like(arr_smooth, np.nanmedian(arr_smooth))

    safe_cont = np.where(np.abs(continuum) < 1e-8, np.nanmedian(continuum), continuum)
    return (arr_smooth / safe_cont).astype(np.float32)


def normalize_datacube(
    datacube: np.ndarray,
    n_smooth: int = 0,
    n_rescale: int = 80,
    deg: int = 3,
    gauss_kernel: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize all spectra and return valid-mask after finite-value QC."""
    if datacube.ndim != 2:
        raise ValueError("datacube must be 2D: (n_spectra, n_wave)")

    out = np.asarray(
        [
            smooth_rescale_array(
                flux,
                n_smooth=n_smooth,
                n_rescale=n_rescale,
                deg=deg,
                gauss_kernel=gauss_kernel,
            )
            for flux in datacube
        ],
        dtype=np.float32,
    )

    good_mask = np.all(np.isfinite(out), axis=1)
    return out[good_mask], good_mask


def build_valid_metadata(
    stars_df: pd.DataFrame,
    valid_indices: np.ndarray,
    filepaths: Sequence[str],
    good_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build metadata table aligned with loaded/normalized datacube."""
    stars_valid = stars_df.iloc[valid_indices].copy().reset_index(drop=True)
    stars_valid["filepath"] = list(filepaths)

    if good_mask is not None:
        stars_valid = stars_valid.loc[np.asarray(good_mask)].reset_index(drop=True)

    return stars_valid


def deduplicate_by_uid_max_snr(
    stars_df: pd.DataFrame,
    datacube: Optional[np.ndarray] = None,
    uid_col: str = "uid",
    snr_col: str = "snru",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray, Dict[str, object]]:
    """Deduplicate by uid, keeping the row with maximum SNR."""
    out = stars_df.copy().reset_index(drop=True)
    if uid_col not in out.columns or snr_col not in out.columns:
        keep_idx = np.arange(len(out), dtype=int)
        rep = {
            "n_before": int(len(out)),
            "n_after": int(len(out)),
            "n_removed": 0,
            "warning": f"missing_uid_or_snr_column ({uid_col}, {snr_col})",
        }
        return out, datacube, keep_idx, rep

    out["__idx__"] = np.arange(len(out), dtype=int)
    snr_num = pd.to_numeric(out[snr_col], errors="coerce").fillna(-np.inf)
    out["__snr__"] = snr_num.values

    keep_idx = out.loc[out.groupby(uid_col)["__snr__"].idxmax(), "__idx__"].values.astype(int)
    keep_idx = np.sort(keep_idx)

    out_dedup = out.iloc[keep_idx].drop(columns=["__idx__", "__snr__"]).reset_index(drop=True)
    cube_dedup = datacube[keep_idx] if datacube is not None else None

    rep = {
        "n_before": int(len(out)),
        "n_after": int(len(out_dedup)),
        "n_removed": int(len(out) - len(out_dedup)),
        "uid_col": uid_col,
        "snr_col": snr_col,
    }
    return out_dedup, cube_dedup, keep_idx, rep


def anomaly_filter_median_mad(
    datacube: np.ndarray,
    stars_df: Optional[pd.DataFrame] = None,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> Tuple[np.ndarray, Optional[pd.DataFrame], np.ndarray, Dict[str, object]]:
    """Filter spectral outliers using per-spectrum median and MAD percentiles."""
    if datacube.ndim != 2:
        raise ValueError("datacube must be 2D")

    medians = np.nanmedian(datacube, axis=1)
    abs_dev = np.abs(datacube - medians[:, None])
    mads = np.nanmedian(abs_dev, axis=1)

    med_low, med_high = np.nanpercentile(medians, [percentile_low, percentile_high])
    mad_low, mad_high = np.nanpercentile(mads, [percentile_low, percentile_high])

    good_mask = (
        np.isfinite(medians)
        & np.isfinite(mads)
        & (medians >= med_low)
        & (medians <= med_high)
        & (mads >= mad_low)
        & (mads <= mad_high)
    )

    cube_clean = datacube[good_mask]
    stars_clean = None
    if stars_df is not None:
        stars_clean = stars_df.iloc[good_mask].reset_index(drop=True)

    rep = {
        "n_before": int(len(datacube)),
        "n_after": int(len(cube_clean)),
        "n_removed": int(np.sum(~good_mask)),
        "median_low": float(med_low),
        "median_high": float(med_high),
        "mad_low": float(mad_low),
        "mad_high": float(mad_high),
    }
    return cube_clean, stars_clean, good_mask, rep


def spectra_to_wide_dataframe(
    datacube: np.ndarray,
    common_wave: np.ndarray,
    metadata_df: Optional[pd.DataFrame] = None,
    flux_prefix: str = "flux_",
    wave_digits: int = 1,
) -> pd.DataFrame:
    """Convert spectra matrix to a wide DataFrame.

    Output columns are flux_3800.0, flux_3801.0, ... plus optional metadata.
    """
    if datacube.ndim != 2:
        raise ValueError("datacube must be 2D")
    if datacube.shape[1] != len(common_wave):
        raise ValueError("datacube.shape[1] must equal len(common_wave)")

    flux_cols = [f"{flux_prefix}{w:.{wave_digits}f}" for w in common_wave]
    flux_df = pd.DataFrame(datacube, columns=flux_cols)

    if metadata_df is None:
        return flux_df

    if len(metadata_df) != len(flux_df):
        raise ValueError("metadata_df row count must match datacube row count")

    return pd.concat([metadata_df.reset_index(drop=True), flux_df], axis=1)


def export_spectra_csv(
    datacube: np.ndarray,
    common_wave: np.ndarray,
    out_csv: str,
    metadata_df: Optional[pd.DataFrame] = None,
    flux_prefix: str = "flux_",
    wave_digits: int = 1,
) -> str:
    """Export spectra matrix to CSV (wide format) and return output path."""
    out_df = spectra_to_wide_dataframe(
        datacube=datacube,
        common_wave=common_wave,
        metadata_df=metadata_df,
        flux_prefix=flux_prefix,
        wave_digits=wave_digits,
    )
    out_df.to_csv(out_csv, index=False)
    return out_csv


def run_load_normalize_pipeline(
    stars_df: pd.DataFrame,
    common_wave: np.ndarray,
    folder: str,
    show_progress: bool = False,
) -> Dict[str, object]:
    """High-level helper for loading + normalization + aligned metadata."""
    datacube_raw, filepaths, valid_indices, drop_log = load_spectra(
        stars_df=stars_df,
        common_wave=common_wave,
        folder=folder,
        show_progress=show_progress,
    )

    datacube_norm, good_mask = normalize_datacube(datacube_raw)
    stars_valid = build_valid_metadata(
        stars_df=stars_df,
        valid_indices=valid_indices,
        filepaths=filepaths,
        good_mask=good_mask,
    )

    summary = {
        "n_input": int(len(stars_df)),
        "n_loaded": int(len(datacube_raw)),
        "n_normalized_ok": int(len(datacube_norm)),
        "n_dropped_load_stage": int(len(drop_log)),
        "n_dropped_norm_stage": int(np.sum(~good_mask)),
    }

    return {
        "datacube_raw": datacube_raw,
        "datacube_norm": datacube_norm,
        "filepaths": filepaths,
        "valid_indices": valid_indices,
        "good_mask": good_mask,
        "stars_valid": stars_valid,
        "drop_log": drop_log,
        "summary": summary,
    }


def run_screening_preprocess_pipeline(
    stars_df: pd.DataFrame,
    common_wave: np.ndarray,
    folder: str,
    cn_catalogs: Optional[Sequence[Union[str, Path, pd.DataFrame]]] = None,
    cn_match_tolerance_arcsec: float = 1.0,
    numeric_cols: Optional[Sequence[str]] = None,
    dropna_cols: Optional[Sequence[str]] = None,
    uid_col: str = "uid",
    snr_col: str = "snru",
    anomaly_low_pct: float = 0.5,
    anomaly_high_pct: float = 99.5,
    show_progress: bool = False,
) -> Dict[str, object]:
    """End-to-end preprocessing pipeline for CN screening workflows.

    Stages:
    1) schema/column normalization and numeric coercion
    2) optional CN labeling by sky match
    3) FITS loading + RV correction + interpolation + normalization
    4) uid deduplication by max snru
    5) anomaly filtering by median/MAD percentile clipping
    """
    if numeric_cols is None:
        numeric_cols = ["ra", "dec", "teff", "logg", "feh", "rv", "lmjd", "spid", "fiberid", "snru", "mag_ps_g"]
    if dropna_cols is None:
        dropna_cols = ["ra", "dec", "teff", "logg", "feh", "lmjd", "uid", "spid", "fiberid", "snru", "mag_ps_g"]

    stars_0 = normalize_column_names(stars_df)
    stars_0 = coerce_numeric_columns(stars_0, numeric_cols)
    n_input = len(stars_0)

    keep_dropna_cols = [c for c in dropna_cols if c in stars_0.columns]
    stars_1 = stars_0.dropna(subset=keep_dropna_cols).copy().reset_index(drop=True)

    if "label" not in stars_1.columns:
        stars_1["label"] = -1

    label_report: Dict[str, object] = {
        "n_catalog_rows": 0,
        "n_positive": int((stars_1["label"] == 1).sum()),
        "n_unlabeled": int((stars_1["label"] == -1).sum()),
        "tolerance_arcsec": float(cn_match_tolerance_arcsec),
    }
    if cn_catalogs is not None and len(cn_catalogs) > 0:
        stars_1, label_report = label_stars_by_cn_catalog(
            stars_df=stars_1,
            cn_catalogs=cn_catalogs,
            tolerance_arcsec=cn_match_tolerance_arcsec,
        )

    load_pack = run_load_normalize_pipeline(
        stars_df=stars_1,
        common_wave=common_wave,
        folder=folder,
        show_progress=show_progress,
    )

    stars_loaded = load_pack["stars_valid"]
    X_loaded = load_pack["datacube_norm"]

    stars_dedup, X_dedup, dedup_keep_idx, dedup_report = deduplicate_by_uid_max_snr(
        stars_df=stars_loaded,
        datacube=X_loaded,
        uid_col=uid_col,
        snr_col=snr_col,
    )

    X_clean, stars_clean, anomaly_mask, anomaly_report = anomaly_filter_median_mad(
        datacube=X_dedup,
        stars_df=stars_dedup,
        percentile_low=anomaly_low_pct,
        percentile_high=anomaly_high_pct,
    )

    summary = {
        "n_input": int(n_input),
        "n_after_dropna": int(len(stars_1)),
        "n_positive_labeled": int((stars_1["label"] == 1).sum()),
        "n_loaded": int(load_pack["summary"]["n_loaded"]),
        "n_normalized_ok": int(load_pack["summary"]["n_normalized_ok"]),
        "n_after_uid_dedup": int(len(X_dedup)),
        "n_after_anomaly_filter": int(len(X_clean)),
    }

    return {
        "stars_stage_dropna_labeled": stars_1,
        "stars_loaded": stars_loaded,
        "stars_dedup": stars_dedup,
        "stars_clean": stars_clean,
        "datacube_loaded_norm": X_loaded,
        "datacube_dedup": X_dedup,
        "datacube_clean": X_clean,
        "load_drop_log": load_pack["drop_log"],
        "load_summary": load_pack["summary"],
        "label_report": label_report,
        "dedup_report": dedup_report,
        "anomaly_report": anomaly_report,
        "dedup_keep_idx": dedup_keep_idx,
        "anomaly_keep_mask": anomaly_mask,
        "summary": summary,
    }


def run_screening_preprocess_from_files(
    stars_csv: Union[str, Path],
    common_wave: np.ndarray,
    folder: str,
    cn_catalogs: Optional[Sequence[Union[str, Path, pd.DataFrame]]] = None,
    cn_match_tolerance_arcsec: float = 1.0,
    numeric_cols: Optional[Sequence[str]] = None,
    dropna_cols: Optional[Sequence[str]] = None,
    uid_col: str = "uid",
    snr_col: str = "snru",
    anomaly_low_pct: float = 0.5,
    anomaly_high_pct: float = 99.5,
    show_progress: bool = False,
) -> Dict[str, object]:
    """File-based wrapper for end-to-end preprocessing.

    This helper is intended for notebook usage where users want one-step execution
    from input CSV files without creating ``stars_df`` beforehand.
    """
    stars_df = load_catalog_input(stars_csv)

    out = run_screening_preprocess_pipeline(
        stars_df=stars_df,
        common_wave=common_wave,
        folder=folder,
        cn_catalogs=cn_catalogs,
        cn_match_tolerance_arcsec=cn_match_tolerance_arcsec,
        numeric_cols=numeric_cols,
        dropna_cols=dropna_cols,
        uid_col=uid_col,
        snr_col=snr_col,
        anomaly_low_pct=anomaly_low_pct,
        anomaly_high_pct=anomaly_high_pct,
        show_progress=show_progress,
    )

    out["stars_source"] = str(stars_csv)
    if cn_catalogs is None:
        out["cn_catalog_sources"] = []
    else:
        srcs = []
        for one in cn_catalogs:
            if isinstance(one, pd.DataFrame):
                srcs.append("<DataFrame>")
            else:
                srcs.append(str(one))
        out["cn_catalog_sources"] = srcs
    return out
