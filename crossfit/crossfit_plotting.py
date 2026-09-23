"""Visualisation helpers for the cross-fit XGBoost-PU candidate screening.

The candidate-spectrum style deliberately reproduces the original project
figure: a seagreen dashed cluster-median reference, the navy candidate spectrum
and shaded CN3839 / CN4142 / CH4300 molecular bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BAND_SPANS = (
    (3830.0, 3883.0, "royalblue", "CN3839"),
    (4120.0, 4216.0, "green", "CN4142"),
    (4285.0, 4315.0, "red", "CH4300"),
)
PARAMETER_COLUMNS = ("teff", "logg", "feh")


def apply_style() -> None:
    """Apply the project plotting defaults."""

    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})


def build_parameter_scaling(
    all_scores: pd.DataFrame,
    columns: tuple[str, ...] = PARAMETER_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, means, stds) for the stellar-parameter matrix."""

    raw = all_scores[list(columns)].to_numpy(dtype=float)
    means = np.nanmean(raw, axis=0)
    stds = np.nanstd(raw, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return raw, means, stds


def matched_cluster_median(
    source_index: int,
    all_scores: pd.DataFrame,
    X_clean: np.ndarray,
    parameter_scaled: np.ndarray,
    excluded_indices: set[int],
    n_neighbors: int = 50,
) -> np.ndarray:
    """Median spectrum of the nearest normal stars in the same masked cluster.

    Members must share ``masked_cluster_id``, carry ``label == -1`` and must not
    belong to the candidate set or be the target itself.
    """

    cluster_id = all_scores.iloc[source_index]["masked_cluster_id"]
    cluster_values = all_scores["masked_cluster_id"].to_numpy()
    label_values = all_scores["label"].to_numpy()
    members = np.flatnonzero((cluster_values == cluster_id) & (label_values == -1))
    if excluded_indices:
        members = np.array([i for i in members if i not in excluded_indices], dtype=int)
    members = members[members != source_index]
    if members.size == 0:
        return np.nanmedian(X_clean, axis=0)
    distance = np.linalg.norm(parameter_scaled[members] - parameter_scaled[source_index], axis=1)
    selected = members[np.argsort(distance)[: min(n_neighbors, members.size)]]
    return np.nanmedian(X_clean[selected], axis=0)


def plot_top_candidate_spectra(
    all_scores: pd.DataFrame,
    X_clean: np.ndarray,
    common_wave: np.ndarray,
    candidates: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
    title: str = "Highest-score candidates versus matched normal spectra",
    score_column: str = "xgb_pu_score",
) -> pd.DataFrame:
    """Draw the top-N candidates next to their cluster reference spectrum.

    Returns the frame of candidates that were actually drawn.
    """

    all_scores = all_scores.reset_index(drop=True)
    X_clean = np.asarray(X_clean)
    common_wave = np.asarray(common_wave)
    counted = candidates.copy()
    if "source_index" in counted.columns:
        ordered = counted.sort_values(score_column, ascending=False).head(top_n).reset_index(drop=True)
        excluded = set(counted["source_index"].astype(int))
    else:
        ordered = counted.sort_values(score_column, ascending=False).head(top_n).reset_index(drop=True)
        excluded = set()

    _, means, stds = build_parameter_scaling(all_scores)
    raw = all_scores[list(PARAMETER_COLUMNS)].to_numpy(dtype=float)
    parameter_scaled = (raw - means) / stds

    n_cols = 5
    n_rows = int(np.ceil(max(len(ordered), 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for position, (_, candidate) in enumerate(ordered.iterrows()):
        ax = axes[position]
        source_index = int(candidate["source_index"]) if "source_index" in candidate else None
        if source_index is not None:
            reference = matched_cluster_median(
                source_index, all_scores, X_clean, parameter_scaled, excluded
            )
            ax.plot(
                common_wave,
                reference,
                color="seagreen",
                linestyle="--",
                linewidth=1,
                label="cluster normal median",
            )
            ax.plot(common_wave, X_clean[source_index], color="navy", linewidth=0.8, label="candidate")
        for left, right, color, _name in BAND_SPANS:
            ax.axvspan(left, right, alpha=0.10, color=color)
        rank = int(candidate.get("candidate_rank", position + 1))
        score = float(candidate.get(score_column, np.nan))
        teff = float(candidate.get("teff", np.nan))
        logg = float(candidate.get("logg", np.nan))
        feh = float(candidate.get("feh", np.nan))
        snru = float(candidate.get("snru", np.nan))
        cluster = candidate.get("masked_cluster_id", "NA")
        ax.set_title(
            f"#{rank} score={score:.3f}\n"
            f"Teff={teff:.0f} logg={logg:.2f} [Fe/H]={feh:.2f}\n"
            f"SNRu={snru:.1f}, cluster={cluster}",
            fontsize=8,
        )
        ax.set_xlim(3800, 4500)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    for position in range(len(ordered), len(axes)):
        axes[position].axis("off")

    fig.supxlabel("Wavelength (Angstrom)")
    fig.supylabel("Normalized flux")
    fig.suptitle(title, y=1.005)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    return ordered


def plot_score_distribution(
    all_scores: pd.DataFrame,
    result: dict[str, object],
    score_column: str = "xgb_pu_score",
) -> None:
    """OOF score histogram plus the known-CN order statistic."""

    threshold = float(result["threshold"])
    known_scores = all_scores.loc[all_scores["label"] == 1, score_column].to_numpy()
    unlabeled_scores = all_scores.loc[all_scores["label"] == -1, score_column].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(unlabeled_scores, bins=80, alpha=0.75, label="unlabeled", color="gray")
    axes[0].hist(known_scores, bins=30, alpha=0.75, label="known CN", color="darkorange")
    axes[0].axvline(
        threshold, color="crimson", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}"
    )
    axes[0].set_xlabel("OOF xgb_pu_score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("OOF score distributions")
    axes[0].legend()

    ordered = np.sort(known_scores)[::-1]
    axes[1].plot(np.arange(1, len(ordered) + 1), ordered, marker=".", linewidth=1)
    axes[1].axvline(
        int(result["required_known"]),
        color="crimson",
        linestyle="--",
        label=f"{int(result['required_known'])}th highest",
    )
    axes[1].axhline(threshold, color="black", linestyle=":")
    axes[1].set_xlabel("Known CN rank by descending score")
    axes[1].set_ylabel("OOF xgb_pu_score")
    axes[1].set_title(f"Known CN order statistic: {int(result['required_known'])}/{len(ordered)} retained")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_repeat_stability(result: dict[str, object]) -> pd.DataFrame:
    """Fold metrics and repeat-to-repeat threshold / candidate count."""

    history = pd.DataFrame(result["history"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.index + 1, history["fold_roc_auc"], marker="o", label="fold ROC-AUC")
    axes[0].plot(history.index + 1, history["fold_pr_auc"], marker="o", label="fold PR-AUC")
    axes[0].set_xlabel("Training fold row")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Fold metrics")
    axes[0].legend()

    thresholds = np.asarray(result["repeat_thresholds"])
    counts = np.asarray(result["repeat_candidate_counts"])
    repeat_table = pd.DataFrame(
        {"repeat": np.arange(1, len(thresholds) + 1), "threshold": thresholds, "candidate_count": counts}
    )
    axes[1].plot(repeat_table["repeat"], repeat_table["threshold"], marker="o", label="threshold")
    twin = axes[1].twinx()
    twin.plot(
        repeat_table["repeat"], repeat_table["candidate_count"], marker="s", color="darkorange",
        label="candidate count",
    )
    axes[1].set_xlabel("Repeat")
    axes[1].set_ylabel("Threshold")
    twin.set_ylabel("Candidates")
    axes[1].set_title("Repeat-to-repeat stability")
    plt.tight_layout()
    plt.show()
    return repeat_table


def plot_parameter_distribution(
    all_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    score_column: str = "xgb_pu_score",
) -> None:
    """Teff-logg plane and one-dimensional parameter histograms."""

    u_table = all_scores[all_scores["label"] == -1]
    known_table = all_scores[all_scores["label"] == 1]
    limits = {
        column: (float(all_scores[column].min()), float(all_scores[column].max()))
        for column in PARAMETER_COLUMNS
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.scatter(u_table["teff"], u_table["logg"], s=2, alpha=0.12, color="gray", label="all U")
    ax.scatter(known_table["teff"], known_table["logg"], s=22, color="darkorange", label="known CN")
    ax.scatter(
        candidates["teff"], candidates["logg"], s=12, color="crimson", alpha=0.75,
        label=f"candidates ({len(candidates):,})",
    )
    ax.set_xlabel("Teff")
    ax.set_ylabel("logg")
    ax.set_title("Teff-logg")
    ax.legend(markerscale=2)

    for column, axis in zip(PARAMETER_COLUMNS, [axes[0, 1], axes[1, 0], axes[1, 1]]):
        axis.hist(u_table[column], bins=60, alpha=0.5, color="gray", label="all U")
        axis.hist(known_table[column], bins=30, alpha=0.6, color="darkorange", label="known CN")
        axis.hist(candidates[column], bins=30, alpha=0.7, color="crimson", label="candidates")
        axis.set_xlim(limits[column])
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
        axis.legend()

    fig.suptitle(f"Parameter distributions; final candidates={len(candidates):,}", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_sky_distribution(
    all_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    sample_size: int = 5000,
) -> None:
    """Equatorial scatter plus a Galactic Aitoff projection."""

    background = all_scores.loc[all_scores["label"] == -1].sample(
        n=min(sample_size, int((all_scores["label"] == -1).sum())), random_state=42
    )
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(background["ra"], background["dec"], s=2, alpha=0.18, color="gray", label="U background")
    ax1.scatter(
        candidates["ra"], candidates["dec"], s=10, alpha=0.75, color="crimson",
        label=f"candidates ({len(candidates):,})",
    )
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.set_title("Equatorial sky distribution")
    ax1.legend()

    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        background_gal = SkyCoord(
            ra=background["ra"].to_numpy() * u.deg, dec=background["dec"].to_numpy() * u.deg
        ).galactic
        candidate_gal = SkyCoord(
            ra=candidates["ra"].to_numpy() * u.deg, dec=candidates["dec"].to_numpy() * u.deg
        ).galactic
        ax2 = fig.add_subplot(1, 2, 2, projection="aitoff")
        ax2.scatter(
            np.deg2rad(background_gal.l.wrap_at(180 * u.deg).degree),
            np.deg2rad(background_gal.b.degree),
            s=2, alpha=0.12, color="gray", label="U background",
        )
        ax2.scatter(
            np.deg2rad(candidate_gal.l.wrap_at(180 * u.deg).degree),
            np.deg2rad(candidate_gal.b.degree),
            s=10, alpha=0.75, color="crimson", label="candidates",
        )
        ax2.set_title("Galactic l-b Aitoff")
        ax2.grid(True)
        ax2.legend(loc="lower left", fontsize=8)
    except Exception as error:  # pragma: no cover - astropy optional
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.text(0.5, 0.5, f"astropy unavailable:\n{error}", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


def plot_config_search(search_summary: pd.DataFrame) -> None:
    """U:P ratio curve and the regularization candidate-count heat map."""

    ratio = search_summary[search_summary["stage"] == "ratio_pre"].sort_values("u_to_p_ratio")
    regular = search_summary[search_summary["stage"] == "regularization_pre"].copy()
    formal = search_summary[search_summary["stage"] == "formal"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    axes[0].plot(ratio["u_to_p_ratio"], ratio["candidate_count"], marker="o", linewidth=2)
    axes[0].set_xlabel("Unlabeled:positive ratio")
    axes[0].set_ylabel("Candidates at exact 90% recall")
    axes[0].set_title("U:P ratio screening")

    if len(regular):
        heat = regular.pivot_table(
            index="max_depth",
            columns=["min_child_weight", "reg_lambda"],
            values="candidate_count",
            aggfunc="min",
        )
        image = axes[1].imshow(heat.to_numpy(), cmap="viridis")
        axes[1].set_yticks(range(len(heat.index)), heat.index)
        axes[1].set_xticks(
            range(len(heat.columns)),
            [f"w={w}, lam={lam:g}" for w, lam in heat.columns],
            rotation=45,
            ha="right",
        )
        axes[1].set_xlabel("Regularization combination")
        axes[1].set_ylabel("max_depth")
        axes[1].set_title("Regularization screening")
        fig.colorbar(image, ax=axes[1], label="candidate count")
    else:
        axes[1].axis("off")

    if len(formal):
        labels = formal["config_key"].str.replace("_s5_r3_b100", "", regex=False)
        axes[2].barh(labels, formal["candidate_count"], xerr=formal["candidate_count_std"], color="steelblue")
        axes[2].set_xlabel("Candidates at 90% recall")
        axes[2].set_title("Formal reruns (5 folds x 3 repeats)")
        axes[2].tick_params(axis="y", labelsize=8)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    return None


def plot_threshold_sensitivity(
    known_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    recalls: np.ndarray | None = None,
) -> pd.DataFrame:
    """Candidate count as a function of the targeted known-CN recall."""

    from .crossfit_engine import exact_recall_threshold

    if recalls is None:
        recalls = np.linspace(0.80, 1.00, 21)
    # Guard against floating-point spill above 1.0 (e.g. arange endpoints).
    recalls = np.clip(np.asarray(recalls, dtype=float), 1e-6, 1.0)
    rows = []
    for target in recalls:
        threshold, required, achieved = exact_recall_threshold(known_scores, float(target))
        rows.append(
            {
                "target_recall": float(target),
                "required_known": required,
                "achieved_known_recall": achieved,
                "threshold": threshold,
                "candidate_count": int(np.sum(unlabeled_scores >= threshold)),
            }
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["target_recall"], frame["candidate_count"], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Targeted known-CN recall")
    ax.set_ylabel("Candidate count (log scale)")
    ax.set_title("Candidate count versus recall target")
    plt.tight_layout()
    plt.show()
    return frame
