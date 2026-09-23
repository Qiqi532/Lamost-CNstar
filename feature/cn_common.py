"""Shared building blocks for the decoupled CN index / differential-area screenings.

This module is the single common layer used by both method modules
(``cn_index_screening`` and ``cn_area_screening``).  It re-exports the validated
primitives from the original ``cn_feature_analysis`` library so the latter stays
untouched and its tests keep passing, and adds the higher-level, method-agnostic
helpers: cache/target preparation, per-method feature tables, the shared Top-30
spectrum plot, candidate-table export and summary assembly.

Design rules honoured here (per the approved task):
* Both methods share one reference-pool construction (same ``masked_cluster_id``,
  exclude label==1 and all 654 threshold candidates, nearest 50 by standardized
  teff/logg/[Fe/H], per-wavelength median reference).
* The area method builds its normal distribution with leave-one-out.
* Standard (paper) bands are the primary screening definition; the project's old
  wide bands are kept only as a ``project_`` sensitivity control, strictly separated.
* Primary screening is relaxed to 1 sigma (mean/std z) for both CN3839 & CN4142
  simultaneously; 1.5/2.0/2.5 sigma tiers are reported in parallel, never reverse
  tuned to hit a target count.
* CH4300 never enters the total score; it only marks CH-normal-like / CH-strong-like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# Re-export the validated primitives from the original base library so it is
# reused rather than duplicated or deleted.
from cn_feature_analysis import (  # noqa: F401
    AUTHORITY_CANDIDATE_PATH,
    EXPECTED_CANDIDATE_ROWS,
    InputConsistencyError,
    MAX_REFERENCE_MEMBERS,
    PROJECT_BAND_DEFS,
    SCORE_SOURCE_PATH,
    BANDS,
    CN_BANDS,
    STANDARD_BAND_DEFS,
    build_reference_map,
    compute_area_features,
    compute_index,
    compute_index_matrix,
    compute_index_with_error,
    distribution_stats,
    load_and_validate_inputs,
    local_scale_and_residual,
    significance,
    _find_target_indices,
    _normal_area_distributions,
    _reference_quality,
    _safe_bool,
    _standardization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_INDEX_DIR = PROJECT_ROOT / "feature" / "results_index"
RESULTS_AREA_DIR = PROJECT_ROOT / "feature" / "results_area"

# Four screening tiers, in sigma, with stable label suffixes.
TIERS: Tuple[Tuple[float, str], ...] = (
    (1.0, "1p0"),
    (1.5, "1p5"),
    (2.0, "2p0"),
    (2.5, "2p5"),
)

# Lower-case CN band keys used for the dual-CN screen (CH4300 is deliberately
# excluded -- it only tags CH-normal/CH-strong, never enters the total score).
_CN_LOWER = ("cn3839", "cn4142")


def load_cache() -> Dict[str, Any]:
    """Load the DR13-all cache via the project's own entry point."""

    if str(PROJECT_ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(PROJECT_ROOT))
    from build_dr13_all_cache import load_dr13_all_cache

    return load_dr13_all_cache()


def _coerce_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def prepare_targets(
    candidates: pd.DataFrame,
    cache: Mapping[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate cache, map candidate/known indices, and build the three reference maps.

    Returns a context dict shared by both method modules.  The reference pool
    always excludes all known CN stars (label == 1) and all 654 threshold
    candidates.  The normal population used for the area leave-one-out
    distribution is restricted to clusters that actually contain a target.
    """

    X = np.asarray(cache["X_clean"], dtype=float)
    wave = np.asarray(cache["common_wave"], dtype=float)
    stars = cache["stars_clean"].reset_index(drop=True).copy()
    if len(X) != len(stars) or X.shape[1] != len(wave):
        raise InputConsistencyError(f"cache shape mismatch: X={X.shape}, stars={stars.shape}, wave={wave.shape}")
    if not {"uid", "label", "masked_cluster_id", "teff", "logg", "feh"}.issubset(stars.columns):
        raise InputConsistencyError("cache metadata lacks required columns")

    candidate_indices, known_indices = _find_target_indices(stars, candidates)
    candidate_uid_set = set(candidates["uid"].astype(str))

    candidate_refs = build_reference_map(stars, candidate_indices, candidate_uid_set)
    known_refs = build_reference_map(stars, known_indices, candidate_uid_set)

    target_indices = np.unique(np.concatenate([candidate_indices, known_indices]))
    target_clusters = set(stars.iloc[target_indices]["masked_cluster_id"].tolist())
    normal_mask = (stars["label"].to_numpy() != 1) & ~stars["uid"].astype(str).isin(candidate_uid_set).to_numpy()
    area_target_indices = np.flatnonzero(normal_mask & stars["masked_cluster_id"].isin(target_clusters).to_numpy())
    normal_refs = build_reference_map(stars, area_target_indices, candidate_uid_set)

    # Attach the supplementary XGB columns to the target rows (no cache mutation).
    score_map = candidates.set_index("uid")["xgb_pu_score"].to_dict()
    std_map = candidates.set_index("uid")["xgb_pu_std"].to_dict() if "xgb_pu_std" in candidates else {}
    for idx in np.concatenate([candidate_indices, known_indices]):
        uid = str(stars.iloc[int(idx)]["uid"])
        if uid in score_map:
            stars.loc[int(idx), "xgb_pu_score"] = score_map[uid]
            stars.loc[int(idx), "xgb_pu_std"] = std_map.get(uid, np.nan)

    if verbose:
        print(f"  targets: candidates={len(candidate_indices)}, known CN={len(known_indices)}, normal refs={len(area_target_indices)}")
    return {
        "X": X,
        "wave": wave,
        "stars": stars,
        "candidate_indices": candidate_indices,
        "known_indices": known_indices,
        "candidate_uid_set": candidate_uid_set,
        "candidate_refs": candidate_refs,
        "known_refs": known_refs,
        "normal_refs": normal_refs,
        "area_target_indices": area_target_indices,
        "target_clusters": target_clusters,
    }


def _base_row(stars: pd.DataFrame, idx: int, refs: np.ndarray, info: Mapping[str, Any]) -> Dict[str, Any]:
    s = stars.iloc[idx]
    return {
        "uid": str(s["uid"]),
        "cache_index": int(idx),
        "ra": _coerce_float(s["ra"]),
        "dec": _coerce_float(s["dec"]),
        "teff": _coerce_float(s["teff"]),
        "logg": _coerce_float(s["logg"]),
        "feh": _coerce_float(s["feh"]),
        "label": int(s["label"]),
        "snru": _coerce_float(s["snru"]),
        "masked_cluster_id": s["masked_cluster_id"],
        "xgb_pu_score": _coerce_float(s["xgb_pu_score"]) if "xgb_pu_score" in stars.columns else float("nan"),
        "xgb_pu_std": _coerce_float(s["xgb_pu_std"]) if "xgb_pu_std" in stars.columns else float("nan"),
        "n_reference_members": int(len(refs)),
        "reference_pool_size": int(info["reference_pool_size"]),
        "reference_quality": info["reference_quality"],
    }


# --------------------------------------------------------------------------- #
# Index method
# --------------------------------------------------------------------------- #
def _index_record(ctx: Mapping[str, Any], idx: int, std_values: np.ndarray, project_values: np.ndarray) -> Dict[str, Any]:
    stars = ctx["stars"]
    refs = np.asarray(ctx["candidate_refs"][idx]["reference_indices"] if idx in ctx["candidate_refs"] else ctx["known_refs"][idx]["reference_indices"], dtype=int)
    info = ctx["candidate_refs"][idx] if idx in ctx["candidate_refs"] else ctx["known_refs"][idx]
    row = _base_row(stars, idx, refs, info)
    for col, band in enumerate(BANDS):
        lower = band.lower()
        sv = std_values[idx, col]
        pv = project_values[idx, col]
        row[f"std_{lower}"] = sv
        row[f"project_{lower}"] = pv
        sstats = distribution_stats(std_values[refs, col]) if len(refs) else distribution_stats([])
        pstats = distribution_stats(project_values[refs, col]) if len(refs) else distribution_stats([])
        d, z, zr = significance(sv, sstats)
        row[f"std_{lower}_z"] = z
        row[f"std_{lower}_z_robust"] = zr
        row[f"std_{lower}_delta"] = d
        pd_, pz, pzr = significance(pv, pstats)
        row[f"project_{lower}_z"] = pz
        row[f"project_{lower}_z_robust"] = pzr
        row[f"project_{lower}_delta"] = pd_
    for threshold, label in TIERS:
        row[f"grade_{label}"] = bool(all(_safe_bool(row[f"std_{b}_z"], threshold) for b in _CN_LOWER))
        row[f"grade_{label}_robust"] = bool(all(_safe_bool(row[f"std_{b}_z_robust"], threshold) for b in _CN_LOWER))
        row[f"grade_project_{label}"] = bool(all(_safe_bool(row[f"project_{b}_z"], threshold) for b in _CN_LOWER))
    cn_dual = bool(row["grade_1p0"])
    row["index_primary_pass"] = cn_dual
    row["ch_class"] = (
        "CH-strong-like" if (cn_dual and _safe_bool(row["std_ch4300_z"], 2.0))
        else "CH-normal-like" if cn_dual else "ambiguous"
    )
    return row


def build_index_table(ctx: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    std_values, _ = compute_index_matrix(ctx["X"], ctx["wave"], STANDARD_BAND_DEFS)
    project_values, _ = compute_index_matrix(ctx["X"], ctx["wave"], PROJECT_BAND_DEFS)
    cand_rows = [_index_record(ctx, int(i), std_values, project_values) for i in ctx["candidate_indices"]]
    known_rows = [_index_record(ctx, int(i), std_values, project_values) for i in ctx["known_indices"]]
    return pd.DataFrame(cand_rows), pd.DataFrame(known_rows)


# --------------------------------------------------------------------------- #
# Area method
# --------------------------------------------------------------------------- #
def _area_record(ctx: Mapping[str, Any], idx: int, normal_area: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    X = ctx["X"]
    wave = ctx["wave"]
    stars = ctx["stars"]
    refs = np.asarray(ctx["candidate_refs"][idx]["reference_indices"] if idx in ctx["candidate_refs"] else ctx["known_refs"][idx]["reference_indices"], dtype=int)
    info = ctx["candidate_refs"][idx] if idx in ctx["candidate_refs"] else ctx["known_refs"][idx]
    row = _base_row(stars, idx, refs, info)
    ref_flux = np.nanmedian(X[refs], axis=0) if len(refs) else np.full(X.shape[1], np.nan)
    cand_flux = X[idx]
    for band in BANDS:
        lower = band.lower()
        area = compute_area_features(ref_flux, cand_flux, wave, STANDARD_BAND_DEFS[band])
        row[f"area_{lower}_relative"] = area["relative"]
        row[f"area_{lower}_signed"] = area["signed"]
        rel_vals = normal_area[f"{lower}_relative"][refs] if len(refs) else np.array([])
        sig_vals = normal_area[f"{lower}_signed"][refs] if len(refs) else np.array([])
        rstats = distribution_stats(rel_vals) if len(refs) else distribution_stats([])
        sstats = distribution_stats(sig_vals) if len(refs) else distribution_stats([])
        d, z, zr = significance(area["relative"], rstats)
        row[f"area_{lower}_z"] = z
        row[f"area_{lower}_z_robust"] = zr
        row[f"area_{lower}_delta"] = d
        _, zs, zsr = significance(area["signed"], sstats)
        row[f"area_{lower}_signed_z"] = zs
    for threshold, label in TIERS:
        row[f"grade_{label}"] = bool(all(_safe_bool(row[f"area_{b}_z"], threshold) for b in _CN_LOWER))
        row[f"grade_{label}_robust"] = bool(all(_safe_bool(row[f"area_{b}_z_robust"], threshold) for b in _CN_LOWER))
    cn_dual = bool(row["grade_1p0"])
    row["area_primary_pass"] = cn_dual
    row["ch_class"] = (
        "CH-strong-like" if (cn_dual and _safe_bool(row["area_ch4300_z"], 2.0))
        else "CH-normal-like" if cn_dual else "ambiguous"
    )
    return row


def build_area_table(ctx: Mapping[str, Any], verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    normal_area = _normal_area_distributions(
        ctx["stars"], ctx["X"], ctx["wave"], ctx["area_target_indices"], ctx["normal_refs"], ctx["target_clusters"], verbose=verbose
    )
    cand_rows = [_area_record(ctx, int(i), normal_area) for i in ctx["candidate_indices"]]
    known_rows = [_area_record(ctx, int(i), normal_area) for i in ctx["known_indices"]]
    return pd.DataFrame(cand_rows), pd.DataFrame(known_rows), normal_area


# --------------------------------------------------------------------------- #
# Shared summary / export
# --------------------------------------------------------------------------- #
def build_method_summary(
    cand_df: pd.DataFrame,
    known_df: pd.DataFrame,
    method_name: str,
    input_manifest: Mapping[str, Any],
    ctx: Mapping[str, Any],
    include_project: bool = False,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "method": method_name,
        "primary_screening": "CN3839 and CN4142 mean/std z >= 1.0 simultaneously (relaxed from the paper 2 sigma to circle reliable candidates faster)",
        "primary_screening_threshold_sigma": 1.0,
        "candidate_input": dict(input_manifest),
        "cache_rows": int(len(ctx["stars"])),
        "cache_spectrum_shape": [int(ctx["X"].shape[0]), int(ctx["X"].shape[1])],
        "known_cn_count": int(len(known_df)),
        "candidate_count": int(len(cand_df)),
        "reference_quality": {str(k): int(v) for k, v in cand_df["reference_quality"].value_counts(dropna=False).to_dict().items()},
        "known_reference_quality": {str(k): int(v) for k, v in known_df["reference_quality"].value_counts(dropna=False).to_dict().items()},
        "tier_counts": {},
        "known_recall": {},
        "band_definitions": STANDARD_BAND_DEFS,
        "reference_policy": "same masked_cluster_id; exclude label==1 and all 654 threshold candidates; nearest 50 by standardized teff/logg/feh; per-wavelength median reference; area normal distribution via leave-one-out",
        "reference_quality_policy": {"good": ">=20", "limited": "10-19", "descriptive_only": "5-9", "insufficient": "<5"},
        "ch4300_role": "never merged into the total score; only marks CH-normal-like / CH-strong-like",
        "limitations": {
            "per_pixel_flux_error": "unavailable in dr13_all_cache; analytic index error columns are NaN and no S/N proxy is substituted",
            "method_dependence": "index and area methods share the same CN/CH bands; they are complementary, not independent chemical confirmations",
            "classification_scope": "spectral CN/CH support only; chemical N abundance is not directly validated",
            "no_target_trimming": "thresholds are reported as sensitivity, not reverse-tuned to a DR5/DR13 expected count",
        },
    }
    for threshold, label in TIERS:
        summary["tier_counts"][label] = {
            "candidate_cn_dual_pass": int(cand_df[f"grade_{label}"].sum()),
            "candidate_cn_dual_pass_robust": int(cand_df[f"grade_{label}_robust"].sum()),
        }
        summary["known_recall"][label] = {
            "recall": float(known_df[f"grade_{label}"].mean()),
            "n_pass": int(known_df[f"grade_{label}"].sum()),
            "recall_robust": float(known_df[f"grade_{label}_robust"].mean()),
            "n_pass_robust": int(known_df[f"grade_{label}_robust"].sum()),
        }
        if include_project:
            summary["tier_counts"][label]["candidate_cn_dual_pass_project"] = int(cand_df[f"grade_project_{label}"].sum())
            summary["known_recall"][label]["recall_project"] = float(known_df[f"grade_project_{label}"].mean())
            summary["known_recall"][label]["n_pass_project"] = int(known_df[f"grade_project_{label}"].sum())
    return summary


def write_candidate_csv(cand_df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    selected = cand_df[cand_df["grade_1p0"]].copy()
    if "xgb_pu_score" in selected.columns:
        selected = selected.sort_values("xgb_pu_score", ascending=False)
    selected.to_csv(out_path, index=False)
    return out_path


def plot_top30(
    ctx: Mapping[str, Any],
    cand_df: pd.DataFrame,
    out_path: Path,
    score_col: str = "xgb_pu_score",
    topn: int = 30,
    suptitle: str = "CN screening - top 30 candidates versus matched normal spectra",
) -> Path:
    """Shared Top-30 high-score candidate spectrum plot (exactly the required style)."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    X = ctx["X"]
    wave = ctx["wave"]
    refs = ctx["candidate_refs"]
    if score_col in cand_df.columns and cand_df[score_col].notna().any():
        frame = cand_df.sort_values(score_col, ascending=False).head(topn)
    else:
        frame = cand_df.head(topn)
    n = len(frame)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.2), squeeze=False)
    band_spans = [(3830, 3883, "royalblue"), (4120, 4216, "green"), (4285, 4315, "red")]
    for rank, (ax, (_, row)) in enumerate(zip(axes.flat, frame.iterrows()), start=1):
        idx = int(row["cache_index"])
        ref_idx = np.asarray(refs[idx]["reference_indices"], dtype=int)
        ref_flux = np.nanmedian(X[ref_idx], axis=0) if len(ref_idx) else np.full(X.shape[1], np.nan)
        cand_flux = X[idx]
        ax.plot(wave, ref_flux, color="seagreen", linestyle="--", linewidth=1, label="cluster normal median")
        ax.plot(wave, cand_flux, color="navy", linewidth=0.8, label="candidate")
        for lo, hi, color in band_spans:
            ax.axvspan(lo, hi, color=color, alpha=0.10)
        score = _coerce_float(row.get(score_col, np.nan))
        title = (
            f"#{rank} XGB={score:.3f}\n"
            f"Teff={row['teff']:.0f} logg={row['logg']:.2f} [Fe/H]={row['feh']:.2f}\n"
            f"SNRu={row['snru']:.0f}, cluster={row['masked_cluster_id']}"
        )
        ax.set_title(title, fontsize=8)
        ax.set_xlim(3800, 4500)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel("Normalized flux")
    fig.suptitle(suptitle, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "PROJECT_ROOT", "RESULTS_INDEX_DIR", "RESULTS_AREA_DIR", "TIERS",
    "STANDARD_BAND_DEFS", "PROJECT_BAND_DEFS", "BANDS", "CN_BANDS",
    "InputConsistencyError", "load_and_validate_inputs", "load_cache",
    "prepare_targets", "build_index_table", "build_area_table",
    "build_method_summary", "write_candidate_csv", "plot_top30",
]
