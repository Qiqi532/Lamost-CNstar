"""CN/CH spectral-index and differential-area analysis for XGB candidates.

This module is deliberately self-contained so that the frozen threshold CSV is
the only source of candidate membership.  The PhaseSummary CSV may contribute
only ``xgb_pu_score`` after an exact UID-set validation.

The cached spectra are continuum-normalised 1-Angstrom samples on 3800--4499
Angstrom.  Per-pixel flux uncertainties are not present in the cache; the
index uncertainty implementation is nevertheless provided and returns NaN
when no uncertainty array is supplied rather than manufacturing one from S/N.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_CANDIDATE_PATH = PROJECT_ROOT / "XGB" / "XGB_PU_candidates_threshold.csv"
SCORE_SOURCE_PATH = PROJECT_ROOT / "PhaseSummary" / "03_ML_XGB" / "XGB_PU_candidates_threshold.csv"
RESULTS_DIR = PROJECT_ROOT / "feature" / "results"
EXPECTED_CANDIDATE_ROWS = 654
MAX_REFERENCE_MEMBERS = 50
REFERENCE_MIN_HIGH_CONFIDENCE = 10


# The first mapping is the exact standard formula requested in the plan.  The
# project mapping is retained only as a sensitivity comparison and never feeds
# the standard-index thresholds.
STANDARD_BAND_DEFS: Dict[str, Dict[str, Any]] = {
    "CN3839": {
        "band": (3861.0, 3884.0),
        "continuum": (3894.0, 3910.0),
        "scale_regions": ((3894.0, 3910.0), (4000.0, 4020.0)),
        "formula": "-2.5log10(F3861-3884/F3894-3910)",
    },
    "CN4142": {
        "band": (4120.0, 4216.0),
        "blue": (4055.0, 4080.0),
        "red": (4240.0, 4280.0),
        "scale_regions": ((4055.0, 4080.0), (4240.0, 4280.0)),
        "formula": "-2.5log10(F4120-4216/(0.5F4055-4080+0.5F4240-4280))",
    },
    "CH4300": {
        "band": (4285.0, 4315.0),
        "blue": (4240.0, 4280.0),
        "red": (4390.0, 4460.0),
        "scale_regions": ((4240.0, 4280.0), (4390.0, 4460.0)),
        "formula": "-2.5log10(F4285-4315/(0.5F4240-4280+0.5F4390-4460))",
    },
}

PROJECT_BAND_DEFS: Dict[str, Dict[str, Any]] = {
    "CN3839": {"band": (3830.0, 3883.0), "blue": (3894.0, 3910.0), "red": (4000.0, 4020.0)},
    "CN4142": {"band": (4120.0, 4216.0), "blue": (4055.0, 4080.0), "red": (4240.0, 4280.0)},
    "CH4300": {"band": (4285.0, 4315.0), "blue": (4240.0, 4280.0), "red": (4390.0, 4460.0)},
}

BANDS = tuple(STANDARD_BAND_DEFS)
CN_BANDS = ("CN3839", "CN4142")


class InputConsistencyError(ValueError):
    """Raised when the frozen candidate inputs do not meet hard boundaries."""


@dataclass(frozen=True)
class InputBundle:
    candidates: pd.DataFrame
    authority_sha256: str
    score_source_sha256: str
    manifest: Dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uid_set(frame: pd.DataFrame, label: str) -> set[str]:
    if "uid" not in frame.columns:
        raise InputConsistencyError(f"{label} lacks required uid column")
    if frame["uid"].isna().any():
        raise InputConsistencyError(f"{label} contains null UID")
    frame_uids = frame["uid"].astype(str)
    if frame_uids.duplicated().any():
        duplicates = frame_uids[frame_uids.duplicated()].tolist()[:5]
        raise InputConsistencyError(f"{label} contains duplicate UID(s): {duplicates}")
    return set(frame_uids)


def load_and_validate_inputs(
    authority_path: Path = AUTHORITY_CANDIDATE_PATH,
    score_source_path: Path = SCORE_SOURCE_PATH,
    expected_rows: Optional[int] = EXPECTED_CANDIDATE_ROWS,
) -> InputBundle:
    """Read the two threshold tables and enforce exact UID-set equality."""

    authority_path = Path(authority_path)
    score_source_path = Path(score_source_path)
    for path in (authority_path, score_source_path):
        if not path.exists():
            raise FileNotFoundError(path)

    authority = pd.read_csv(authority_path)
    score_source = pd.read_csv(score_source_path)
    required_authority = {
        "uid", "ra", "dec", "teff", "logg", "feh", "label", "snru", "xgb_pu_std", "masked_cluster_id"
    }
    missing = sorted(required_authority - set(authority.columns))
    if missing:
        raise InputConsistencyError(f"authority candidate table missing columns: {missing}")
    if "xgb_pu_prob" not in score_source.columns:
        raise InputConsistencyError("score source lacks xgb_pu_prob")
    if expected_rows is not None and len(authority) != expected_rows:
        raise InputConsistencyError(f"authority candidate row count {len(authority)} != expected {expected_rows}")
    authority_uids = _uid_set(authority, "authority candidate table")
    score_uids = _uid_set(score_source, "score source table")
    if authority_uids != score_uids:
        only_authority = sorted(authority_uids - score_uids)[:10]
        only_score = sorted(score_uids - authority_uids)[:10]
        raise InputConsistencyError(
            "candidate UID sets are not exactly equal; "
            f"only_authority={only_authority}, only_score_source={only_score}"
        )

    score = score_source[["uid", "xgb_pu_prob"]].copy()
    score["uid"] = score["uid"].astype(str)
    merged = authority.copy()
    merged["uid"] = merged["uid"].astype(str)
    merged = merged.merge(score, on="uid", how="left", validate="one_to_one", sort=False)
    if merged["xgb_pu_prob"].isna().any():
        raise InputConsistencyError("xgb_pu_prob left join produced missing scores")
    merged = merged.rename(columns={"xgb_pu_prob": "xgb_pu_score"})
    if len(merged) != len(authority) or set(merged["uid"]) != authority_uids:
        raise InputConsistencyError("score join changed candidate membership or row count")

    manifest = {
        "authority_path": str(authority_path.relative_to(PROJECT_ROOT)),
        "score_source_path": str(score_source_path.relative_to(PROJECT_ROOT)),
        "authority_sha256": _sha256(authority_path),
        "score_source_sha256": _sha256(score_source_path),
        "authority_rows": int(len(authority)),
        "authority_unique_uid": int(authority["uid"].nunique()),
        "score_source_rows": int(len(score_source)),
        "score_source_unique_uid": int(score_source["uid"].nunique()),
        "uid_sets_equal": True,
        "candidate_membership_source": "XGB/XGB_PU_candidates_threshold.csv",
        "score_source_role": "supplementary xgb_pu_prob only",
        "joined_score_column": "xgb_pu_score",
    }
    return InputBundle(
        candidates=merged,
        authority_sha256=manifest["authority_sha256"],
        score_source_sha256=manifest["score_source_sha256"],
        manifest=manifest,
    )


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _mean_and_error(values: np.ndarray, errors: Optional[np.ndarray]) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if errors is None:
        return mean, float("nan")
    errors = np.asarray(errors, dtype=float)[finite]
    errors = errors[np.isfinite(errors)]
    if errors.size != values.size:
        return mean, float("nan")
    return mean, float(np.sqrt(np.sum(errors * errors)) / values.size)


def _mask(wave: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    return (wave >= float(bounds[0])) & (wave <= float(bounds[1]))


def _validate_band_coverage(wave: np.ndarray, defs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    wave = np.asarray(wave, dtype=float)
    checks: Dict[str, Any] = {}
    for name, definition in defs.items():
        ranges = [definition["band"]]
        if "continuum" in definition:
            ranges.append(definition["continuum"])
        else:
            ranges.extend([definition["blue"], definition["red"]])
        checks[name] = {
            "ranges": [[float(a), float(b)] for a, b in ranges],
            "all_covered": all(bool(np.any(_mask(wave, bounds))) for bounds in ranges),
            "pixels": [int(_mask(wave, bounds).sum()) for bounds in ranges],
        }
    return checks


def compute_index_with_error(
    flux: np.ndarray,
    wave: np.ndarray,
    definition: Mapping[str, Any],
    flux_error: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute a magnitude index and analytic 1-sigma error when supplied."""

    band_values = flux[_mask(wave, definition["band"])]
    band_errors = None if flux_error is None else flux_error[_mask(wave, definition["band"])]
    f_band, e_band = _mean_and_error(band_values, band_errors)
    if "continuum" in definition:
        cont_values = flux[_mask(wave, definition["continuum"])]
        cont_errors = None if flux_error is None else flux_error[_mask(wave, definition["continuum"])]
        f_cont, e_cont = _mean_and_error(cont_values, cont_errors)
    else:
        blue = _mask(wave, definition["blue"])
        red = _mask(wave, definition["red"])
        f_blue, e_blue = _mean_and_error(flux[blue], None if flux_error is None else flux_error[blue])
        f_red, e_red = _mean_and_error(flux[red], None if flux_error is None else flux_error[red])
        f_cont = 0.5 * (f_blue + f_red)
        e_cont = float(0.5 * np.sqrt(e_blue * e_blue + e_red * e_red))
    if not (np.isfinite(f_band) and np.isfinite(f_cont) and f_band > 0 and f_cont > 0):
        return float("nan"), float("nan")
    coefficient = -2.5 / math.log(10.0)
    value = coefficient * math.log(f_band / f_cont)
    if flux_error is None or not (np.isfinite(e_band) and np.isfinite(e_cont)):
        return float(value), float("nan")
    error = abs(coefficient) * math.sqrt((e_band / f_band) ** 2 + (e_cont / f_cont) ** 2)
    return float(value), float(error)


def compute_index(flux: np.ndarray, wave: np.ndarray, definition: Mapping[str, Any]) -> float:
    return compute_index_with_error(flux, wave, definition, flux_error=None)[0]


def compute_index_matrix(X: np.ndarray, wave: np.ndarray, defs: Mapping[str, Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.full((len(X), len(defs)), np.nan, dtype=float)
    errors = np.full_like(values, np.nan)
    for row, flux in enumerate(np.asarray(X)):
        for col, definition in enumerate(defs.values()):
            values[row, col], errors[row, col] = compute_index_with_error(flux, wave, definition)
    return values, errors


def _coverage_for_flux(flux: np.ndarray, wave: np.ndarray, bounds: Sequence[float]) -> float:
    mask = _mask(wave, bounds)
    return float(np.isfinite(flux[mask]).mean()) if mask.any() else 0.0


def compute_coverage(flux: np.ndarray, wave: np.ndarray) -> Dict[str, float]:
    result = {"valid_fraction_3800_4500": float(np.isfinite(flux).mean())}
    for name, definition in STANDARD_BAND_DEFS.items():
        result[f"valid_fraction_{name}"] = _coverage_for_flux(flux, wave, definition["band"])
    return result


def local_scale_and_residual(
    reference_flux: np.ndarray,
    target_flux: np.ndarray,
    wave: np.ndarray,
    scale_regions: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, float]:
    region_mask = np.zeros(len(wave), dtype=bool)
    for bounds in scale_regions:
        region_mask |= _mask(wave, bounds)
    ref_level = _finite_mean(reference_flux[region_mask])
    target_level = _finite_mean(target_flux[region_mask])
    if not (np.isfinite(ref_level) and np.isfinite(target_level) and target_level > 0):
        return np.full_like(target_flux, np.nan, dtype=float), float("nan")
    scale = ref_level / target_level
    scaled_target = np.asarray(target_flux, dtype=float) * scale
    return np.asarray(reference_flux, dtype=float) - scaled_target, float(scale)


def compute_area_features(
    reference_flux: np.ndarray,
    target_flux: np.ndarray,
    wave: np.ndarray,
    definition: Mapping[str, Any],
) -> Dict[str, float]:
    """Return residual-area diagnostics, normalized by the science-band width."""

    residual, scale = local_scale_and_residual(reference_flux, target_flux, wave, definition["scale_regions"])
    mask = _mask(wave, definition["band"])
    valid = mask & np.isfinite(residual) & np.isfinite(reference_flux) & (reference_flux > 0)
    if valid.sum() < 2:
        return {
            "signed": np.nan, "positive": np.nan, "relative": np.nan,
            "positive_fraction": np.nan, "max_depth": np.nan, "rms": np.nan,
            "smooth_signed": np.nan, "scale": scale, "valid_fraction": float(valid.mean()),
        }
    x = wave[valid]
    r = residual[valid]
    width = float(definition["band"][1] - definition["band"][0])
    signed = float(np.trapezoid(r, x) / width)
    positive = float(np.trapezoid(np.maximum(r, 0.0), x) / width)
    # Use the same locally scaled target as the residual definition for A_rel.
    region_mask = np.zeros(len(wave), dtype=bool)
    for bounds in definition["scale_regions"]:
        region_mask |= _mask(wave, bounds)
    ref_level = _finite_mean(reference_flux[region_mask])
    target_level = _finite_mean(target_flux[region_mask])
    scale_value = ref_level / target_level if np.isfinite(ref_level) and np.isfinite(target_level) and target_level > 0 else np.nan
    scaled_target = np.asarray(target_flux, dtype=float) * scale_value if np.isfinite(scale_value) else np.full_like(target_flux, np.nan, dtype=float)
    ratio_valid = valid & np.isfinite(scaled_target) & (reference_flux > 0)
    relative = float(np.trapezoid(1.0 - scaled_target[ratio_valid] / reference_flux[ratio_valid], wave[ratio_valid]) / width) if ratio_valid.sum() >= 2 else np.nan
    smooth = np.convolve(r, np.array([0.25, 0.5, 0.25]), mode="same")
    smooth_signed = float(np.trapezoid(smooth, x) / width)
    return {
        "signed": signed,
        "positive": positive,
        "relative": relative,
        "positive_fraction": float(np.mean(r > 0)),
        "max_depth": float(np.max(r)),
        "rms": float(np.sqrt(np.mean(r * r))),
        "smooth_signed": smooth_signed,
        "scale": float(scale_value),
        "valid_fraction": float(valid.mean()),
    }


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def distribution_stats(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "mad": np.nan}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size >= 2 else np.nan,
        "median": float(np.median(values)),
        "mad": float(np.median(np.abs(values - np.median(values)))),
    }


def significance(value: float, stats: Mapping[str, float]) -> Tuple[float, float, float]:
    if not np.isfinite(value):
        return np.nan, np.nan, np.nan
    z_mean = (value - stats["mean"]) / stats["std"] if np.isfinite(stats["std"]) and stats["std"] > 0 else np.nan
    robust_denominator = 1.4826 * stats["mad"] if np.isfinite(stats["mad"]) and stats["mad"] > 0 else np.nan
    z_robust = (value - stats["median"]) / robust_denominator if np.isfinite(robust_denominator) else np.nan
    delta_median = value - stats["median"] if np.isfinite(stats["median"]) else np.nan
    return float(delta_median), float(z_mean), float(z_robust)


def _reference_quality(pool_size: int) -> str:
    if pool_size < 5:
        return "insufficient"
    if pool_size < 10:
        return "descriptive_only"
    if pool_size < 20:
        return "limited"
    return "good"


def _standardization(stars: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    columns = ["teff", "logg", "feh"]
    center = np.array([float(np.nanmedian(stars[col].astype(float))) for col in columns])
    scale = np.array([float(np.nanstd(stars[col].astype(float))) for col in columns])
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return center, scale


def build_reference_map(
    stars: pd.DataFrame,
    target_indices: Sequence[int],
    candidate_uids: set[str],
    max_members: int = MAX_REFERENCE_MEMBERS,
) -> Dict[int, Dict[str, Any]]:
    """Build per-target nearest normal reference members within each cluster."""

    labels = stars["label"].to_numpy()
    uids = stars["uid"].astype(str).to_numpy()
    cluster_ids = stars["masked_cluster_id"].to_numpy()
    normal_mask = (labels != 1) & ~np.isin(uids, list(candidate_uids))
    values = stars[["teff", "logg", "feh"]].astype(float).to_numpy()
    center, scale = _standardization(stars)
    standardized = (values - center) / scale
    pools: Dict[Any, np.ndarray] = {}
    for cluster_id in pd.unique(cluster_ids[np.flatnonzero(normal_mask)]):
        pools[cluster_id] = np.flatnonzero(normal_mask & (cluster_ids == cluster_id))

    result: Dict[int, Dict[str, Any]] = {}
    for raw_target in target_indices:
        target = int(raw_target)
        cluster_id = cluster_ids[target]
        pool = pools.get(cluster_id, np.array([], dtype=int))
        pool = pool[pool != target]
        pool_size = int(len(pool))
        quality = _reference_quality(pool_size)
        if pool_size == 0 or quality == "insufficient":
            refs = np.array([], dtype=int)
        else:
            distances = np.linalg.norm(standardized[pool] - standardized[target], axis=1)
            order = np.argsort(np.where(np.isfinite(distances), distances, np.inf), kind="stable")
            refs = pool[order[:max_members]]
        result[target] = {
            "reference_indices": refs,
            "reference_pool_size": pool_size,
            "reference_quality": quality,
            "standardization_center": center,
            "standardization_scale": scale,
        }
    return result


def _stats_for_indices(values: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    return distribution_stats(values[indices]) if len(indices) else distribution_stats([])


def _safe_bool(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value >= threshold)


def _area_stats_for_refs(area_values: Mapping[str, np.ndarray], refs: np.ndarray, band: str, metric: str) -> Dict[str, float]:
    if len(refs) == 0:
        return distribution_stats([])
    return distribution_stats(area_values[f"{band}_{metric}"][refs])


def screening_tier(
    reference_quality: str,
    index_dual_2sigma: bool,
    area_dual_2sigma: bool,
    index_dual_1p5sigma: bool,
    area_dual_1p5sigma: bool,
) -> str:
    """Assign the explanatory screening tier without changing strict classes."""

    high_quality = reference_quality in {"good", "limited"}
    if reference_quality == "insufficient":
        return "insufficient_reference"
    if index_dual_2sigma and area_dual_2sigma and high_quality:
        return "strict_consensus_2sigma"
    if index_dual_1p5sigma and area_dual_1p5sigma and high_quality:
        return "moderate_consensus_1p5sigma"
    if (index_dual_1p5sigma or area_dual_1p5sigma) and high_quality:
        return "single_method_1p5sigma"
    return "no_dual_band_support"


def _add_significance_columns(row: MutableMapping[str, Any], prefix: str, value: float, stats: Mapping[str, float]) -> None:
    delta, z_mean, z_robust = significance(value, stats)
    row[f"{prefix}_ref_n"] = stats["n"]
    row[f"{prefix}_ref_mean"] = stats["mean"]
    row[f"{prefix}_ref_std"] = stats["std"]
    row[f"{prefix}_ref_median"] = stats["median"]
    row[f"{prefix}_ref_mad"] = stats["mad"]
    row[f"{prefix}_delta_median"] = delta
    row[f"{prefix}_z_mean_std"] = z_mean
    row[f"{prefix}_z_robust"] = z_robust


def _target_record(
    stars: pd.DataFrame,
    X: np.ndarray,
    wave: np.ndarray,
    target_idx: int,
    set_name: str,
    reference_info: Mapping[str, Any],
    std_values: np.ndarray,
    std_errors: np.ndarray,
    project_values: np.ndarray,
    normal_area_values: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    target_flux = np.asarray(X[target_idx], dtype=float)
    refs = np.asarray(reference_info["reference_indices"], dtype=int)
    reference_flux = np.nanmedian(X[refs], axis=0) if len(refs) else np.full(X.shape[1], np.nan)
    row: Dict[str, Any] = {
        "uid": str(stars.iloc[target_idx]["uid"]),
        "cache_index": int(target_idx),
        "evaluation_set": set_name,
        "ra": float(stars.iloc[target_idx]["ra"]),
        "dec": float(stars.iloc[target_idx]["dec"]),
        "teff": float(stars.iloc[target_idx]["teff"]),
        "logg": float(stars.iloc[target_idx]["logg"]),
        "feh": float(stars.iloc[target_idx]["feh"]),
        "label": int(stars.iloc[target_idx]["label"]),
        "snru": float(stars.iloc[target_idx]["snru"]),
        "masked_cluster_id": stars.iloc[target_idx]["masked_cluster_id"],
        "reference_pool_size": int(reference_info["reference_pool_size"]),
        "n_reference_members": int(len(refs)),
        "reference_quality": reference_info["reference_quality"],
        "flux_error_available": False,
    }
    row.update(compute_coverage(target_flux, wave))
    for col, value in (("xgb_pu_score", np.nan), ("xgb_pu_std", np.nan)):
        if col in stars.columns:
            row[col] = stars.iloc[target_idx][col]
        else:
            row[col] = value

    for col, band in enumerate(BANDS):
        lower = band.lower()
        row[f"std_{lower}"] = std_values[target_idx, col]
        row[f"std_{lower}_sigma"] = std_errors[target_idx, col]
        row[f"project_{lower}"] = project_values[target_idx, col]
        stats = _stats_for_indices(std_values[:, col], refs)
        _add_significance_columns(row, f"std_{lower}", std_values[target_idx, col], stats)
        row[f"project_{lower}_valid"] = bool(np.isfinite(project_values[target_idx, col]))

        area = compute_area_features(reference_flux, target_flux, wave, STANDARD_BAND_DEFS[band]) if len(refs) else {
            "signed": np.nan, "positive": np.nan, "relative": np.nan,
            "positive_fraction": np.nan, "max_depth": np.nan, "rms": np.nan,
            "smooth_signed": np.nan, "scale": np.nan, "valid_fraction": 0.0,
        }
        for metric, value in area.items():
            row[f"area_{lower}_{metric}"] = value
        for metric in ("signed", "relative"):
            stats = _area_stats_for_refs(normal_area_values, refs, lower, metric)
            _add_significance_columns(row, f"area_{lower}_{metric}", area[metric], stats)

    for threshold_label, threshold in (("1p5", 1.5), ("2p0", 2.0), ("2p5", 2.5)):
        index_both = all(_safe_bool(row[f"std_{band.lower()}_z_mean_std"], threshold) for band in CN_BANDS)
        area_both = all(_safe_bool(row[f"area_{band.lower()}_relative_z_mean_std"], threshold) for band in CN_BANDS)
        index_robust_both = all(_safe_bool(row[f"std_{band.lower()}_z_robust"], threshold) for band in CN_BANDS)
        area_robust_both = all(_safe_bool(row[f"area_{band.lower()}_relative_z_robust"], threshold) for band in CN_BANDS)
        row[f"index_cn_dual_ge_{threshold_label}"] = index_both
        row[f"area_cn_dual_ge_{threshold_label}"] = area_both
        row[f"index_cn_dual_robust_ge_{threshold_label}"] = index_robust_both
        row[f"area_cn_dual_robust_ge_{threshold_label}"] = area_robust_both

    index_dual_2 = bool(row["index_cn_dual_ge_2p0"])
    area_dual_2 = bool(row["area_cn_dual_ge_2p0"])
    quality = str(row["reference_quality"])
    high_quality = quality in {"good", "limited"}
    if quality == "insufficient":
        classification = "insufficient_reference"
    elif quality == "descriptive_only":
        classification = "weak_or_mixed"
    elif index_dual_2 and area_dual_2 and high_quality:
        classification = "consensus_strong"
    elif index_dual_2 and high_quality:
        classification = "index_only"
    elif area_dual_2 and high_quality:
        classification = "area_only"
    elif any(row[f"index_cn_dual_ge_{label}"] or row[f"area_cn_dual_ge_{label}"] for label in ("1p5", "2p0", "2p5")):
        classification = "weak_or_mixed"
    else:
        classification = "not_supported"
    row["classification"] = classification
    # The strict classification above follows the approved plan exactly.  This
    # separate, non-exclusive screening tier makes the threshold sensitivity
    # visible instead of presenting 20 strict consensuses as the only stars
    # with absorption evidence.
    row["screening_tier"] = screening_tier(
        quality,
        index_dual_2,
        area_dual_2,
        bool(row["index_cn_dual_ge_1p5"]),
        bool(row["area_cn_dual_ge_1p5"]),
    )
    index_cn_dual = index_dual_2 and high_quality
    area_cn_dual = area_dual_2 and high_quality
    row["index_ch_class"] = (
        "CH-strong-like" if index_cn_dual and _safe_bool(row["std_ch4300_z_mean_std"], 2.0)
        else "CH-normal-like" if index_cn_dual else "ambiguous"
    )
    row["area_ch_class"] = (
        "CH-strong-like" if area_cn_dual and _safe_bool(row["area_ch4300_relative_z_mean_std"], 2.0)
        else "CH-normal-like" if area_cn_dual else "ambiguous"
    )
    if classification == "consensus_strong":
        if row["index_ch_class"] == row["area_ch_class"] and row["index_ch_class"] in {"CH-normal-like", "CH-strong-like"}:
            row["consensus_ch_class"] = row["index_ch_class"]
        else:
            row["consensus_ch_class"] = "ambiguous"
    else:
        row["consensus_ch_class"] = "ambiguous"
    return row


def _normal_area_distributions(
    stars: pd.DataFrame,
    X: np.ndarray,
    wave: np.ndarray,
    normal_indices: Sequence[int],
    reference_map: Mapping[int, Mapping[str, Any]],
    relevant_clusters: set[Any],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    n = len(stars)
    output = {f"{band.lower()}_{metric}": np.full(n, np.nan, dtype=float) for band in BANDS for metric in ("signed", "relative")}
    normal_indices = [int(i) for i in normal_indices if stars.iloc[int(i)]["masked_cluster_id"] in relevant_clusters]
    for count, target_idx in enumerate(normal_indices, start=1):
        info = reference_map.get(target_idx)
        if info is None or info["reference_quality"] == "insufficient" or len(info["reference_indices"]) < 1:
            continue
        refs = np.asarray(info["reference_indices"], dtype=int)
        reference_flux = np.nanmedian(X[refs], axis=0)
        target_flux = np.asarray(X[target_idx], dtype=float)
        for band in BANDS:
            area = compute_area_features(reference_flux, target_flux, wave, STANDARD_BAND_DEFS[band])
            output[f"{band.lower()}_signed"][target_idx] = area["signed"]
            output[f"{band.lower()}_relative"][target_idx] = area["relative"]
        if verbose and count % 2000 == 0:
            print(f"  leave-one-out normal area: {count}/{len(normal_indices)}")
    return output


def _find_target_indices(stars: pd.DataFrame, candidates: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    uid_to_index = {str(uid): i for i, uid in enumerate(stars["uid"].astype(str))}
    candidate_indices = np.array([uid_to_index[str(uid)] for uid in candidates["uid"]], dtype=int)
    known_indices = np.flatnonzero(stars["label"].to_numpy() == 1)
    missing = [uid for uid, idx in zip(candidates["uid"], candidate_indices) if str(stars.iloc[idx]["uid"]) != str(uid)]
    if missing:
        raise InputConsistencyError(f"candidate UID mapping failed: {missing[:5]}")
    return candidate_indices, known_indices


def analyze_cached_data(
    candidates: pd.DataFrame,
    cache: Mapping[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute all candidate and known-CN features from an already-loaded cache."""

    X = np.asarray(cache["X_clean"], dtype=float)
    wave = np.asarray(cache["common_wave"], dtype=float)
    stars = cache["stars_clean"].reset_index(drop=True).copy()
    if len(X) != len(stars) or X.shape[1] != len(wave):
        raise InputConsistencyError(f"cache shape mismatch: X={X.shape}, stars={stars.shape}, wave={wave.shape}")
    if not {"uid", "label", "masked_cluster_id", "teff", "logg", "feh"}.issubset(stars.columns):
        raise InputConsistencyError("cache metadata lacks required columns")
    coverage_checks = _validate_band_coverage(wave, STANDARD_BAND_DEFS)
    if not all(item["all_covered"] for item in coverage_checks.values()):
        raise InputConsistencyError(f"standard band coverage failed: {coverage_checks}")
    candidate_indices, known_indices = _find_target_indices(stars, candidates)
    candidate_uid_set = set(candidates["uid"].astype(str))
    target_indices = np.unique(np.concatenate([candidate_indices, known_indices]))
    target_clusters = set(stars.iloc[target_indices]["masked_cluster_id"].tolist())
    normal_mask = (stars["label"].to_numpy() != 1) & ~stars["uid"].astype(str).isin(candidate_uid_set).to_numpy()
    normal_indices = np.flatnonzero(normal_mask)
    area_target_indices = np.flatnonzero(normal_mask & stars["masked_cluster_id"].isin(target_clusters).to_numpy())
    if verbose:
        print(f"  target candidates={len(candidate_indices)}, known CN={len(known_indices)}, normal references={len(normal_indices)}")
        print(f"  standard-band coverage={coverage_checks}")

    candidate_refs = build_reference_map(stars, candidate_indices, candidate_uid_set)
    known_refs = build_reference_map(stars, known_indices, candidate_uid_set)
    normal_refs = build_reference_map(stars, area_target_indices, candidate_uid_set)
    std_values, std_errors = compute_index_matrix(X, wave, STANDARD_BAND_DEFS)
    project_values, _ = compute_index_matrix(X, wave, PROJECT_BAND_DEFS)
    normal_area_values = _normal_area_distributions(stars, X, wave, area_target_indices, normal_refs, target_clusters, verbose=verbose)

    # Add the supplementary XGB columns to target metadata without altering the cache.
    cache_score = candidates.set_index("uid")["xgb_pu_score"].to_dict()
    cache_std = candidates.set_index("uid")["xgb_pu_std"].to_dict() if "xgb_pu_std" in candidates else {}
    for idx in np.concatenate([candidate_indices, known_indices]):
        uid = str(stars.iloc[int(idx)]["uid"])
        if uid in cache_score:
            stars.loc[int(idx), "xgb_pu_score"] = cache_score[uid]
            stars.loc[int(idx), "xgb_pu_std"] = cache_std.get(uid, np.nan)

    candidate_rows = [_target_record(stars, X, wave, int(idx), "xgb_candidate", candidate_refs[int(idx)], std_values, std_errors, project_values, normal_area_values) for idx in candidate_indices]
    known_rows = [_target_record(stars, X, wave, int(idx), "known_cn", known_refs[int(idx)], std_values, std_errors, project_values, normal_area_values) for idx in known_indices]
    candidate_df = pd.DataFrame(candidate_rows)
    known_df = pd.DataFrame(known_rows)
    return {
        "candidates": candidate_df,
        "known_cn": known_df,
        "cache": cache,
        "stars": stars,
        "candidate_indices": candidate_indices,
        "known_indices": known_indices,
        "reference_maps": {"candidates": candidate_refs, "known_cn": known_refs, "normal": normal_refs},
        "normal_area_values": normal_area_values,
        "standard_index_values": std_values,
        "standard_index_errors": std_errors,
        "project_index_values": project_values,
        "coverage_checks": coverage_checks,
        "candidate_uid_set": candidate_uid_set,
    }


def _quantiles(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    if column not in frame:
        return {}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"q05": np.nan, "q50": np.nan, "q95": np.nan}
    q = values.quantile([0.05, 0.5, 0.95])
    return {"q05": float(q.loc[0.05]), "q50": float(q.loc[0.5]), "q95": float(q.loc[0.95])}


def build_summary(result: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = result["candidates"]
    known = result["known_cn"]
    summary: Dict[str, Any] = {
        "candidate_input": dict(input_manifest),
        "cache_rows": int(len(result["stars"])),
        "cache_spectrum_shape": list(np.asarray(result["cache"]["X_clean"]).shape),
        "known_cn_count": int(len(known)),
        "candidate_count": int(len(candidates)),
        "candidate_valid_spectrum_count": int((candidates["valid_fraction_3800_4500"] >= 0.95).sum()),
        "candidate_reference_quality": {str(k): int(v) for k, v in candidates["reference_quality"].value_counts(dropna=False).to_dict().items()},
        "known_reference_quality": {str(k): int(v) for k, v in known["reference_quality"].value_counts(dropna=False).to_dict().items()},
        "candidate_classification_counts": {str(k): int(v) for k, v in candidates["classification"].value_counts(dropna=False).to_dict().items()},
        "candidate_screening_tier_counts": {str(k): int(v) for k, v in candidates["screening_tier"].value_counts(dropna=False).to_dict().items()},
        "known_classification_counts": {str(k): int(v) for k, v in known["classification"].value_counts(dropna=False).to_dict().items()},
        "known_screening_tier_counts": {str(k): int(v) for k, v in known["screening_tier"].value_counts(dropna=False).to_dict().items()},
        "candidate_threshold_sensitivity": {},
        "known_threshold_sensitivity": {},
        "distributions": {},
        "limitations": {
            "per_pixel_flux_error": "unavailable in dr13_all_cache; analytic index error columns are NaN and no S/N proxy is substituted",
            "method_dependence": "index and area methods share the same CN/CH bands and are complementary, not independent chemical confirmations",
            "classification_scope": "spectral CN/CH support; chemical N abundance is not directly validated",
        },
    }
    for threshold_label, threshold in (("1p5", "1p5"), ("2p0", "2p0"), ("2p5", "2p5")):
        summary["candidate_threshold_sensitivity"][threshold_label] = {
            "index_supported": int(candidates[f"index_cn_dual_ge_{threshold}"].sum()),
            "area_supported": int(candidates[f"area_cn_dual_ge_{threshold}"].sum()),
            "robust_index_supported": int(candidates[f"index_cn_dual_robust_ge_{threshold}"].sum()),
            "robust_area_supported": int(candidates[f"area_cn_dual_robust_ge_{threshold}"].sum()),
        }
        summary["known_threshold_sensitivity"][threshold_label] = {
            "index_supported": int(known[f"index_cn_dual_ge_{threshold}"].sum()),
            "area_supported": int(known[f"area_cn_dual_ge_{threshold}"].sum()),
            "robust_index_supported": int(known[f"index_cn_dual_robust_ge_{threshold}"].sum()),
            "robust_area_supported": int(known[f"area_cn_dual_robust_ge_{threshold}"].sum()),
            "index_recall": float(known[f"index_cn_dual_ge_{threshold}"].mean()),
            "area_recall": float(known[f"area_cn_dual_ge_{threshold}"].mean()),
            "robust_index_recall": float(known[f"index_cn_dual_robust_ge_{threshold}"].mean()),
            "robust_area_recall": float(known[f"area_cn_dual_robust_ge_{threshold}"].mean()),
        }
    for frame_name, frame in (("candidate", candidates), ("known_cn", known)):
        summary["distributions"][frame_name] = {
            column: _quantiles(frame, column)
            for column in ("std_cn3839_z_mean_std", "std_cn4142_z_mean_std", "std_ch4300_z_mean_std", "area_cn3839_relative_z_mean_std", "area_cn4142_relative_z_mean_std", "area_ch4300_relative_z_mean_std", "teff", "logg", "feh")
        }
    consensus = candidates.loc[candidates["classification"] == "consensus_strong"]
    index_only = candidates.loc[candidates["classification"] == "index_only"]
    area_only = candidates.loc[candidates["classification"] == "area_only"]
    summary["examples"] = {
        "consensus_strong_uids": consensus.sort_values("xgb_pu_score", ascending=False).head(10)["uid"].tolist(),
        "index_only_uids": index_only.sort_values("xgb_pu_score", ascending=False).head(10)["uid"].tolist(),
        "area_only_uids": area_only.sort_values("xgb_pu_score", ascending=False).head(10)["uid"].tolist(),
        "threshold_near_uids": candidates.loc[(candidates["std_cn3839_z_mean_std"] - 2).abs().add((candidates["std_cn4142_z_mean_std"] - 2).abs()) < 0.3].head(10)["uid"].tolist(),
    }
    return summary


def _relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_outputs(result: Mapping[str, Any], input_manifest: Mapping[str, Any], output_dir: Path = RESULTS_DIR) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    known = result["known_cn"]
    summary = build_summary(result, input_manifest)
    outputs: Dict[str, Path] = {}
    manifest = dict(input_manifest)
    manifest.update({
        "cache_rows": int(len(result["stars"])),
        "cache_X_shape": list(np.asarray(result["cache"]["X_clean"]).shape),
        "common_wave_start": float(result["cache"]["common_wave"][0]),
        "common_wave_end": float(result["cache"]["common_wave"][-1]),
        "common_wave_step": float(result["cache"]["common_wave"][1] - result["cache"]["common_wave"][0]),
        "known_cn_count": int(len(known)),
        "candidate_count": int(len(candidates)),
        "standard_band_definitions": STANDARD_BAND_DEFS,
        "project_band_definitions": PROJECT_BAND_DEFS,
        "coverage_checks": result["coverage_checks"],
        "reference_policy": "same masked_cluster_id; exclude label==1 and all threshold candidates; nearest 50 by standardized teff/logg/feh; per-target leave-one-out for normal area distributions",
        "reference_quality_policy": {"good": ">=20", "limited": "10-19", "descriptive_only": "5-9", "insufficient": "<5"},
        "error_propagation": "analytic index propagation implemented; cache has no per-pixel flux errors, so output uncertainty columns are NaN",
    })
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    outputs["input_manifest"] = manifest_path
    for name, frame in (
        ("xgb_candidates_all_features.csv", candidates),
        ("xgb_candidates_index_supported.csv", candidates[candidates["index_cn_dual_ge_2p0"] & candidates["reference_quality"].isin(["good", "limited"])]),
        ("xgb_candidates_area_supported.csv", candidates[candidates["area_cn_dual_ge_2p0"] & candidates["reference_quality"].isin(["good", "limited"])]),
        ("xgb_candidates_consensus.csv", candidates[candidates["classification"] == "consensus_strong"]),
        ("xgb_candidates_moderate.csv", candidates[candidates["index_cn_dual_ge_1p5"] & candidates["area_cn_dual_ge_1p5"] & candidates["reference_quality"].isin(["good", "limited"])]),
        ("xgb_candidates_relaxed_union.csv", candidates[(candidates["index_cn_dual_ge_1p5"] | candidates["area_cn_dual_ge_1p5"]) & candidates["reference_quality"].isin(["good", "limited"])]),
        ("xgb_candidates_robust_consensus.csv", candidates[candidates["index_cn_dual_robust_ge_2p0"] & candidates["area_cn_dual_robust_ge_2p0"] & candidates["reference_quality"].isin(["good", "limited"])]),
        ("known_cn_feature_validation.csv", known),
    ):
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs[name] = path
    summary_path = output_dir / "feature_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    outputs["feature_analysis_summary"] = summary_path
    return outputs


def _plot_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def plot_band_definitions(wave: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    _plot_style()
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(3800, 4500)
    for name, definition in STANDARD_BAND_DEFS.items():
        low, high = definition["band"]
        ax.axvspan(low, high, alpha=0.25, label=f"{name} science")
        for region in definition["scale_regions"]:
            ax.axvspan(region[0], region[1], alpha=0.08)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Band regions")
    ax.set_title("Standard science bands and local-continuum scale regions")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax: Any, frame: pd.DataFrame, x: str, y: str, color: str = "classification", title: str = "") -> None:
    if frame.empty or x not in frame or y not in frame:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        return
    if color in frame:
        classes = frame[color].astype(str)
        for cls in sorted(classes.unique()):
            mask = classes == cls
            ax.scatter(frame.loc[mask, x], frame.loc[mask, y], s=12, alpha=0.7, label=cls)
        ax.legend(fontsize=6, loc="best")
    else:
        ax.scatter(frame[x], frame[y], s=12, alpha=0.7)
    ax.axvline(2, color="black", linestyle="--", linewidth=0.7)
    ax.axhline(2, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")


def make_plots(result: Mapping[str, Any], output_dir: Path = RESULTS_DIR) -> List[Path]:
    import matplotlib.pyplot as plt

    _plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    known = result["known_cn"]
    outputs: List[Path] = []
    band_path = output_dir / "band_definitions.png"
    plot_band_definitions(np.asarray(result["cache"]["common_wave"]), band_path)
    outputs.append(band_path)

    quality_counts = candidates["reference_quality"].value_counts().reindex(["good", "limited", "descriptive_only", "insufficient"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    quality_counts.plot.bar(ax=ax, color=["#2ca02c", "#ffbf00", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Candidate count")
    ax.set_title("Reference quality")
    fig.tight_layout()
    path = output_dir / "reference_quality.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _scatter(axes[0, 0], candidates, "std_cn3839_z_mean_std", "std_cn4142_z_mean_std", title="Standard-index CN evidence")
    _scatter(axes[0, 1], candidates, "area_cn3839_relative_z_mean_std", "area_cn4142_relative_z_mean_std", title="Differential-area CN evidence")
    _scatter(axes[1, 0], candidates, "std_cn3839_z_mean_std", "area_cn3839_relative_z_mean_std", title="CN3839 index vs area")
    _scatter(axes[1, 1], candidates, "std_cn4142_z_mean_std", "area_cn4142_relative_z_mean_std", title="CN4142 index vs area")
    fig.tight_layout()
    path = output_dir / "index_diagnostics.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, band in zip(axes, BANDS):
        lower = band.lower()
        ax.hist(candidates[f"area_{lower}_relative"].dropna(), bins=30, alpha=0.75, label="candidates")
        ax.hist(known[f"area_{lower}_relative"].dropna(), bins=30, alpha=0.5, label="known CN")
        ax.set_title(band); ax.set_xlabel("A_rel"); ax.set_ylabel("count")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "area_diagnostics.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, group in candidates.groupby("classification", sort=True):
        ax.scatter(group["std_cn4142_z_mean_std"], group["area_cn4142_relative_z_mean_std"], s=14, alpha=0.75, label=cls)
    ax.axvline(2, color="black", linestyle="--", linewidth=0.8); ax.axhline(2, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("CN4142 standard-index z"); ax.set_ylabel("CN4142 area z"); ax.set_title("Index/area consistency"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "index_area_consistency.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    quality_ok = candidates["reference_quality"].isin(["good", "limited"])
    index_2 = candidates["index_cn_dual_ge_2p0"] & quality_ok
    area_2 = candidates["area_cn_dual_ge_2p0"] & quality_ok
    both_15 = candidates["index_cn_dual_ge_1p5"] & candidates["area_cn_dual_ge_1p5"] & quality_ok
    either_15 = (candidates["index_cn_dual_ge_1p5"] | candidates["area_cn_dual_ge_1p5"]) & quality_ok
    strict = candidates["classification"] == "consensus_strong"
    labels = ["all", "index ≥2σ", "area ≥2σ", "either method ≥1.5σ", "both methods ≥1.5σ", "strict consensus ≥2σ", "CH-normal strict", "CH-strong strict"]
    values = [len(candidates), int(index_2.sum()), int(area_2.sum()), int(either_15.sum()), int(both_15.sum()), int(strict.sum()), int((strict & (candidates["consensus_ch_class"] == "CH-normal-like")).sum()), int((strict & (candidates["consensus_ch_class"] == "CH-strong-like")).sum())]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(labels, values, color="#4c78a8"); ax.set_ylabel("count"); ax.set_title("Selection funnel"); ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    path = output_dir / "selection_funnel.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, column in zip(axes, ["teff", "logg", "feh"]):
        ax.hist(candidates[column], bins=30, alpha=0.75, label="XGB candidates")
        ax.hist(known[column], bins=30, alpha=0.5, label="known CN")
        ax.set_xlabel(column); ax.set_ylabel("count")
    axes[0].legend(fontsize=7); fig.tight_layout()
    path = output_dir / "parameter_distribution.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), subplot_kw={"projection": "aitoff"})
    for ax, ra_col, dec_col, title in ((axes[0], "ra", "dec", "RA/Dec"), (axes[1], "ra", "dec", "RA/Dec: consensus")):
        frame = candidates if title == "RA/Dec" else candidates[candidates["classification"] == "consensus_strong"]
        if not frame.empty:
            ra = np.deg2rad((frame[ra_col].to_numpy() + 180.0) % 360.0 - 180.0)
            dec = np.deg2rad(frame[dec_col].to_numpy())
            ax.scatter(ra, dec, s=10, alpha=0.7)
        ax.set_title(title); ax.grid(True)
    fig.tight_layout()
    path = output_dir / "sky_distribution.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); outputs.append(path)

    _plot_examples(result, output_dir / "candidate_spectra_consensus.png", candidates[candidates["classification"] == "consensus_strong"].head(4))
    outputs.append(output_dir / "candidate_spectra_consensus.png")
    disagreements = candidates[candidates["classification"].isin(["index_only", "area_only"])].head(4)
    _plot_examples(result, output_dir / "candidate_spectra_disagreement.png", disagreements)
    outputs.append(output_dir / "candidate_spectra_disagreement.png")
    return outputs


def _plot_examples(result: Mapping[str, Any], output_path: Path, frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    X = np.asarray(result["cache"]["X_clean"])
    wave = np.asarray(result["cache"]["common_wave"])
    refs = result["reference_maps"]["candidates"]
    if frame.empty:
        fig, ax = plt.subplots(figsize=(10, 3)); ax.text(0.5, 0.5, "No examples in this category", ha="center", va="center"); ax.axis("off"); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig); return
    fig, axes = plt.subplots(len(frame), 1, figsize=(11, max(3, 2.7 * len(frame))), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], frame.iterrows()):
        idx = int(row["cache_index"]); ref_idx = np.asarray(refs[idx]["reference_indices"], dtype=int)
        ref_flux = np.nanmedian(X[ref_idx], axis=0) if len(ref_idx) else np.full(X.shape[1], np.nan)
        ax.plot(wave, X[idx], color="#1f77b4", linewidth=0.65, label="candidate")
        ax.plot(wave, ref_flux, color="#d62728", linewidth=0.75, label="cluster normal median")
        for band, definition in STANDARD_BAND_DEFS.items():
            ax.axvspan(*definition["band"], alpha=0.08)
        ax.set_xlim(3800, 4500); ax.set_ylabel("flux")
        ax.set_title(f"{row['uid']} | class={row['classification']} | XGB={row.get('xgb_pu_score', np.nan):.3f} | z(CN)=({row['std_cn3839_z_mean_std']:.2f},{row['std_cn4142_z_mean_std']:.2f}) | area z=({row['area_cn3839_relative_z_mean_std']:.2f},{row['area_cn4142_relative_z_mean_std']:.2f}) | nref={row['n_reference_members']}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("Wavelength (Å)")
    fig.tight_layout(); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)


def run_analysis(
    authority_path: Path = AUTHORITY_CANDIDATE_PATH,
    score_source_path: Path = SCORE_SOURCE_PATH,
    output_dir: Path = RESULTS_DIR,
    expected_rows: Optional[int] = EXPECTED_CANDIDATE_ROWS,
    write_result_files: bool = True,
    make_result_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete reproducible analysis and optionally write artifacts."""

    inputs = load_and_validate_inputs(authority_path, score_source_path, expected_rows=expected_rows)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from build_dr13_all_cache import load_dr13_all_cache

    cache = load_dr13_all_cache()
    result = analyze_cached_data(inputs.candidates, cache, verbose=verbose)
    result["input_manifest"] = inputs.manifest
    if write_result_files:
        result["outputs"] = write_outputs(result, inputs.manifest, output_dir=output_dir)
    if make_result_plots:
        result["plot_outputs"] = make_plots(result, output_dir=output_dir)
    return result


__all__ = [
    "AUTHORITY_CANDIDATE_PATH", "SCORE_SOURCE_PATH", "RESULTS_DIR", "STANDARD_BAND_DEFS", "PROJECT_BAND_DEFS",
    "InputConsistencyError", "load_and_validate_inputs", "compute_index", "compute_index_with_error",
    "compute_area_features", "build_reference_map", "analyze_cached_data", "build_summary", "write_outputs",
    "make_plots", "run_analysis", "screening_tier",
]
