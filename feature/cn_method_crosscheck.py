"""Index vs Area method cross-check (reuses cn_common, no logic duplication).

Compares the two decoupled CN screening methods (cn_index_screening / cn_area_screening)
on the same 654 frozen candidates: how many candidates / known CN stars each method
flags at each tier, the overlap between methods, the consistency of their z-scores, and
what spectrally distinguishes the shared / index-only / area-only groups.

All heavy lifting (input validation, cache load, reference pools, index & area tables)
is delegated to ``cn_common``; this module only adds group bookkeeping, overlap stats,
and the cross-check figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cn_common as cc

RESULTS_CROSSCHECK_DIR = cc.PROJECT_ROOT / "feature" / "results_crosscheck"
TIERS = ("1p0", "1p5", "2p0", "2p5")
_GROUPS = ("both", "index-only", "area-only", "neither")
_BAND_SPANS = [(3830, 3883, "royalblue"), (4120, 4216, "green"), (4285, 4315, "red")]


try:  # prefer correct (tie-aware) ranks for Spearman
    from scipy.stats import rankdata as _rankdata

    def _rank(a: np.ndarray) -> np.ndarray:
        return _rankdata(a)
except Exception:  # fallback: ordinal ranks (no tie averaging)
    def _rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a)).astype(float)


# --------------------------------------------------------------------------- #
# Load + reproducibility self-check
# --------------------------------------------------------------------------- #
def load_method_tables(
    repro_check: bool = True,
    index_csv: Optional[str] = None,
    area_csv: Optional[str] = None,
    tier: str = "1p0",
) -> Dict[str, Any]:
    """Build both full method tables and verify they reproduce the frozen exports.

    Returns a dict with ``index_cand, index_known, area_cand, area_known, ctx,
    inputs, cache, normal_area``.  ``index_cand`` / ``area_cand`` are the full 654-row
    tables (with ``grade_*`` etc.); ``*_known`` are the 91 known-CN rows.
    """

    inputs = cc.load_and_validate_inputs()
    cache = cc.load_cache()
    ctx = cc.prepare_targets(inputs.candidates, cache, verbose=True)
    index_cand, index_known = cc.build_index_table(ctx)
    area_cand, area_known, normal_area = cc.build_area_table(ctx, verbose=True)

    if repro_check:
        _repro_check(index_cand, area_cand, index_csv, area_csv, tier)
    return {
        "index_cand": index_cand,
        "index_known": index_known,
        "area_cand": area_cand,
        "area_known": area_known,
        "ctx": ctx,
        "inputs": inputs,
        "cache": cache,
        "normal_area": normal_area,
    }


def _repro_check(index_cand, area_cand, index_csv, area_csv, tier):
    idx_csv = Path(index_csv) if index_csv else cc.RESULTS_INDEX_DIR / "cn_index_candidates.csv"
    are_csv = Path(area_csv) if area_csv else cc.RESULTS_AREA_DIR / "cn_area_candidates.csv"
    if not idx_csv.exists() or not are_csv.exists():
        raise cc.InputConsistencyError("frozen candidate CSVs missing; cannot run reproducibility self-check")
    frozen_idx = set(pd.read_csv(idx_csv)["uid"].astype(str))
    frozen_are = set(pd.read_csv(are_csv)["uid"].astype(str))
    rebuilt_idx = set(index_cand[index_cand[f"grade_{tier}"]]["uid"].astype(str))
    rebuilt_are = set(area_cand[area_cand[f"grade_{tier}"]]["uid"].astype(str))
    if frozen_idx != rebuilt_idx:
        raise cc.InputConsistencyError(
            f"index {tier} uid set mismatch: frozen={len(frozen_idx)} rebuilt={len(rebuilt_idx)}"
        )
    if frozen_are != rebuilt_are:
        raise cc.InputConsistencyError(
            f"area {tier} uid set mismatch: frozen={len(frozen_are)} rebuilt={len(rebuilt_are)}"
        )


# --------------------------------------------------------------------------- #
# Group / overlap bookkeeping
# --------------------------------------------------------------------------- #
def group_labels(index_cand, area_cand, tier: str = "1p0") -> pd.Series:
    """Return a Series (uid -> both/index-only/area-only/neither), length = full universe."""

    idx = index_cand[["uid"]].copy()
    idx["uid"] = idx["uid"].astype(str)
    idx["idx"] = index_cand[f"grade_{tier}"].values.astype(bool)
    are = area_cand[["uid"]].copy()
    are["uid"] = are["uid"].astype(str)
    are["are"] = area_cand[f"grade_{tier}"].values.astype(bool)
    m = idx.merge(are, on="uid")

    def _lab(r):
        if r["idx"] and r["are"]:
            return "both"
        if r["idx"]:
            return "index-only"
        if r["are"]:
            return "area-only"
        return "neither"

    m["group"] = m.apply(_lab, axis=1)
    return m.set_index("uid")["group"].rename(f"group_{tier}")


def candidate_overlap(index_cand, area_cand, tier: str = "1p0") -> Dict[str, Any]:
    total = len(index_cand)
    idx_u = set(index_cand[index_cand[f"grade_{tier}"]]["uid"].astype(str))
    are_u = set(area_cand[area_cand[f"grade_{tier}"]]["uid"].astype(str))
    both = idx_u & are_u
    index_only = idx_u - are_u
    area_only = are_u - idx_u
    union = idx_u | are_u
    jaccard = len(both) / len(union) if union else 0.0
    return {
        "n_index": len(idx_u), "n_area": len(are_u),
        "both": len(both), "index_only": len(index_only), "area_only": len(area_only),
        "neither": total - len(union), "jaccard": jaccard,
        "uid_both": sorted(both), "uid_index_only": sorted(index_only), "uid_area_only": sorted(area_only),
    }


def known_overlap(index_known, area_known, tier: str = "1p0") -> Dict[str, Any]:
    total = len(index_known)
    idx_u = set(index_known[index_known[f"grade_{tier}"]]["uid"].astype(str))
    are_u = set(area_known[area_known[f"grade_{tier}"]]["uid"].astype(str))
    both = idx_u & are_u
    index_only = idx_u - are_u
    area_only = are_u - idx_u
    union = idx_u | are_u
    jaccard = len(both) / len(union) if union else 0.0
    return {
        "n_index": len(idx_u), "n_area": len(are_u),
        "both": len(both), "index_only": len(index_only), "area_only": len(area_only),
        "neither": total - len(union), "jaccard": jaccard,
        "uid_both": sorted(both), "uid_index_only": sorted(index_only), "uid_area_only": sorted(area_only),
    }


def build_overlap_summary(index_cand, area_cand, index_known, area_known) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        co = candidate_overlap(index_cand, area_cand, tier)
        ko = known_overlap(index_known, area_known, tier)
        nk = len(index_known)
        rows.append({
            "tier": tier,
            "n_index": co["n_index"], "n_area": co["n_area"],
            "both": co["both"], "index_only": co["index_only"], "area_only": co["area_only"], "jaccard": co["jaccard"],
            "known_both": ko["both"], "known_index_only": ko["index_only"],
            "known_area_only": ko["area_only"], "known_neither": ko["neither"],
            "known_recall_index": (ko["both"] + ko["index_only"]) / nk,
            "known_recall_area": (ko["both"] + ko["area_only"]) / nk,
        })
    cols = ["tier", "n_index", "n_area", "both", "index_only", "area_only", "jaccard",
            "known_both", "known_index_only", "known_area_only", "known_neither",
            "known_recall_index", "known_recall_area"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (float("nan"), float("nan"))
    xr, yr = x[mask], y[mask]
    pearson = float(np.corrcoef(xr, yr)[0, 1])
    spearman = float(np.corrcoef(_rank(xr), _rank(yr))[0, 1])
    return (pearson, spearman)


def plot_overlap_by_tier(summary_df: pd.DataFrame, out_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    w = 0.25
    ax.bar(x - w, summary_df["both"], w, label="both")
    ax.bar(x, summary_df["index_only"], w, label="index-only")
    ax.bar(x + w, summary_df["area_only"], w, label="area-only")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["tier"])
    ax.set_xlabel("tier (sigma)")
    ax.set_ylabel("candidate count")
    ax.set_title("Method overlap by tier (candidates)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)
    plt.close(fig)
    return None


def plot_z_consistency(index_cand, area_cand, band: str, labels=None, out_path: Optional[str] = None):
    if labels is None:
        labels = group_labels(index_cand, area_cand, "1p0")
    uids = list(labels.index)
    idx_s = index_cand.set_index(index_cand["uid"].astype(str))
    are_s = area_cand.set_index(area_cand["uid"].astype(str))
    x = idx_s[f"std_{band}_z"].reindex(uids).values.astype(float)
    y = are_s[f"area_{band}_z"].reindex(uids).values.astype(float)
    g = labels.reindex(uids).values
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    cmap = {"both": "black", "index-only": "navy", "area-only": "darkred", "neither": "lightgray"}
    for grp in _GROUPS:
        m = g == grp
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], s=14, color=cmap[grp], label=f"{grp} (n={int(m.sum())})", alpha=0.7, edgecolors="none")
    for v in (1, 2):
        ax.axhline(v, color="gray", ls=":", lw=0.8)
        ax.axvline(v, color="gray", ls=":", lw=0.8)
        ax.axhline(-v, color="gray", ls=":", lw=0.8)
        ax.axvline(-v, color="gray", ls=":", lw=0.8)
    pearson, spearman = _corr(x, y)
    ax.set_xlabel(f"Index z ({band})")
    ax.set_ylabel(f"Area z ({band})")
    ax.set_title(f"CN{band[2:]} z consistency  Pearson={pearson:.2f}  Spearman={spearman:.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path), (pearson, spearman)
    plt.close(fig)
    return None, (pearson, spearman)


def _residual_curves(ctx, index_cand, area_cand, tier):
    labels = group_labels(index_cand, area_cand, tier)
    X = ctx["X"]
    wave = ctx["wave"]
    refs = ctx["candidate_refs"]
    idx_map = index_cand.set_index(index_cand["uid"].astype(str))["cache_index"].astype(int)
    curves = {}
    for grp in ("both", "index-only", "area-only"):
        uids = list(labels[labels == grp].index)
        arr = []
        for uid in uids:
            ci = int(idx_map[uid])
            ref_idx = np.asarray(refs[ci]["reference_indices"], dtype=int)
            ref_flux = np.nanmedian(X[ref_idx], axis=0) if len(ref_idx) else np.full(X.shape[1], np.nan)
            arr.append(X[ci] - ref_flux)
        curves[grp] = {
            "n": len(uids),
            "curve": np.nanmean(np.array(arr), axis=0) if arr else np.zeros(X.shape[1]),
        }
    return curves, wave


def group_residual_band_means(ctx, index_cand, area_cand, tier: str = "1p0") -> pd.DataFrame:
    curves, wave = _residual_curves(ctx, index_cand, area_cand, tier)
    rows = []
    for grp in ("both", "index-only", "area-only"):
        c = curves[grp]["curve"]
        rows.append({
            "group": grp, "n": curves[grp]["n"],
            "cn3839": float(c[(wave >= 3861) & (wave <= 3884)].mean()),
            "cn4142": float(c[(wave >= 4120) & (wave <= 4216)].mean()),
            "ch4300": float(c[(wave >= 4285) & (wave <= 4315)].mean()),
        })
    return pd.DataFrame(rows)


def plot_group_residual_profiles(
    ctx, index_cand, area_cand, tier: str = "1p0", bands: bool = True, out_path: Optional[str] = None
):
    curves, wave = _residual_curves(ctx, index_cand, area_cand, tier)
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = {"both": "black", "index-only": "navy", "area-only": "darkred"}
    for grp in ("both", "index-only", "area-only"):
        ax.plot(wave, curves[grp]["curve"], color=cmap[grp], label=f"{grp} (n={curves[grp]['n']})", lw=1.2)
    for lo, hi, col in _BAND_SPANS:
        ax.axvspan(lo, hi, color=col, alpha=0.10)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(3800, 4500)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Median (candidate - reference) flux")
    if bands:
        bm = group_residual_band_means(ctx, index_cand, area_cand, tier)
        txt = " | ".join(
            f"{r.group} CN3839={r.cn3839:.3f} CN4142={r.cn4142:.3f} CH4300={r.ch4300:.3f}"
            for r in bm.itertuples()
        )
        ax.set_title("Group residual profiles\n" + txt, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)
    plt.close(fig)
    return None


def plot_group_spectra_grid(ctx, cand_df, labels, group: str, out_path: str):
    sub = cand_df[cand_df["uid"].astype(str).isin(labels[labels == group].index)]
    n = len(sub)
    suptitle = f"{group} candidates (n={n}) - versus matched normal spectra"
    return cc.plot_top30(ctx, sub, out_path, suptitle=suptitle)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _build_groups_frame(index_cand, area_cand) -> pd.DataFrame:
    idx = index_cand[["uid", "cache_index", "teff", "logg", "feh", "snru",
                       "masked_cluster_id", "xgb_pu_score", "std_cn3839_z", "std_cn4142_z"]].copy()
    idx["uid"] = idx["uid"].astype(str)
    are = area_cand[["uid", "area_cn3839_z", "area_cn4142_z"]].copy()
    are["uid"] = are["uid"].astype(str)
    df = idx.merge(are, on="uid").rename(
        columns={"std_cn3839_z": "index_z_cn3839", "std_cn4142_z": "index_z_cn4142"}
    )
    for tier in TIERS:
        gl = group_labels(index_cand, area_cand, tier)
        df[f"group_{tier}"] = gl.reindex(df["uid"]).values
    return df


def export_outputs(t: Dict[str, Any], summary_df: pd.DataFrame, out_dir=None, tier: str = "1p0") -> Dict[str, Any]:
    out_dir = Path(out_dir) if out_dir else RESULTS_CROSSCHECK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    index_cand = t["index_cand"]
    area_cand = t["area_cand"]
    ctx = t["ctx"]

    groups = _build_groups_frame(index_cand, area_cand)
    groups.to_csv(out_dir / "crosscheck_groups.csv", index=False)

    labels = group_labels(index_cand, area_cand, tier)
    top30_map = {"both": "shared_top30.png", "index-only": "index_only_top30.png", "area-only": "area_only_top30.png"}
    for grp, fname in top30_map.items():
        plot_group_spectra_grid(ctx, index_cand, labels, grp, out_dir / fname)

    resid_path = plot_group_residual_profiles(ctx, index_cand, area_cand, tier=tier, out_path=out_dir / "group_residual_profiles.png")

    zc = {}
    for band in ("cn3839", "cn4142"):
        _, (pr, sr) = plot_z_consistency(index_cand, area_cand, band, labels=labels,
                                          out_path=out_dir / f"z_consistency_{band}.png")
        zc[band] = {"pearson": pr, "spearman": sr}

    summary = {
        "method": "index_vs_area_crosscheck",
        "candidate_universe": int(len(index_cand)),
        "known_universe": int(len(t["index_known"])),
        "tier_counts_and_overlap": json.loads(summary_df.to_json(orient="records")),
        "known_crosscheck_1p0": known_overlap(t["index_known"], t["area_known"], "1p0"),
        "z_consistency": zc,
        "group_residual_band_means_1p0": json.loads(
            group_residual_band_means(ctx, index_cand, area_cand, tier).to_json(orient="records")
        ),
        "reproducibility": "rebuilt 1sigma uid sets match frozen exports (checked in load_method_tables)",
    }
    (out_dir / "crosscheck_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    return {
        "groups_csv": out_dir / "crosscheck_groups.csv",
        "shared_top30": out_dir / top30_map["both"],
        "index_only_top30": out_dir / top30_map["index-only"],
        "area_only_top30": out_dir / top30_map["area-only"],
        "residual_profiles": resid_path,
        "summary_json": out_dir / "crosscheck_summary.json",
        "z_consistency_png": {b: out_dir / f"z_consistency_{b}.png" for b in ("cn3839", "cn4142")},
        "overlap_png": plot_overlap_by_tier(summary_df, out_dir / "overlap_by_tier.png"),
    }


def run_crosscheck(out_dir=None, repro_check: bool = True, tier: str = "1p0") -> Dict[str, Any]:
    t = load_method_tables(repro_check=repro_check, tier=tier)
    summary_df = build_overlap_summary(t["index_cand"], t["area_cand"], t["index_known"], t["area_known"])
    paths = export_outputs(t, summary_df, out_dir=out_dir, tier=tier)
    paths["tables"] = t
    paths["summary_df"] = summary_df
    return paths


if __name__ == "__main__":
    res = run_crosscheck()
    print("crosscheck done; files:", {k: str(v) for k, v in res.items() if k not in ("tables", "summary_df")})
