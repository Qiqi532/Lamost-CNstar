"""Repeated cross-fitted XGBoost-PU engine for the LAMOST CN-star sample.

Design notes
------------
* The model output is called a *score*, never a probability.  Each base learner
  sees a balanced P/U sample, so ``binary:logistic`` is bounded to [0, 1] but
  does not recover the survey prevalence.
* Every object receives exactly one out-of-fold (OOF) prediction per repeat, so
  both known-CN and unlabeled scores are out of fold.
* The main threshold uses an explicit order statistic (the k-th largest known-CN
  OOF score with ``k = ceil(target_recall * n_positive)``) instead of an
  interpolated quantile, so the main result retains at least the requested
  number of known stars by construction.
* Exports are deliberately minimal: candidate table, known-CN table, summary
  JSON and a compact array archive.  The 41k-row all-score table is kept in
  memory for the notebook, but is only written when ``export_full_scores=True``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Best configuration found by the config search (u10_d3_w1_l1p0).
BEST_FORMAL_CONFIG = {
    "u_to_p_ratio": 10,
    "max_depth": 3,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}


def recall_tag(target_recall: float) -> str:
    """Return a stable filename tag such as ``recall90`` or ``recall95``."""

    return f"recall{int(round(target_recall * 100))}"


@dataclass(frozen=True)
class CrossFitConfig:
    """One repeated cross-fit / P-U bagging configuration."""

    u_to_p_ratio: int = 10
    max_depth: int = 3
    min_child_weight: int = 1
    reg_lambda: float = 1.0
    n_splits: int = 5
    n_repeats: int = 3
    n_bags: int = 100
    num_boost_round: int = 50
    target_recall: float = 0.90
    random_seed: int = 42
    standardize: bool = True

    def validate(self, n_positive: int, n_unlabeled: int) -> None:
        if self.u_to_p_ratio < 1:
            raise ValueError("u_to_p_ratio must be at least 1")
        if self.max_depth < 1 or self.min_child_weight < 1:
            raise ValueError("tree regularization parameters must be positive")
        if self.n_splits < 2 or self.n_repeats < 1 or self.n_bags < 1:
            raise ValueError("split, repeat, and bag counts must be positive")
        if self.num_boost_round < 1:
            raise ValueError("num_boost_round must be positive")
        if not 0 < self.target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        if n_positive < self.n_splits:
            raise ValueError("n_positive must be at least n_splits")
        if n_unlabeled < n_positive * self.u_to_p_ratio:
            raise ValueError("not enough unlabeled objects for the requested U:P ratio")

    @property
    def key(self) -> str:
        return (
            f"u{self.u_to_p_ratio}_d{self.max_depth}_w{self.min_child_weight}"
            f"_l{str(self.reg_lambda).replace('.', 'p')}_s{self.n_splits}"
            f"_r{self.n_repeats}_b{self.n_bags}_t{int(round(self.target_recall * 100))}"
        )


def required_known_count(n_positive: int, target_recall: float = 0.90) -> int:
    """Return the explicit order-statistic count used for thresholding."""

    if n_positive < 1 or not 0 < target_recall <= 1:
        raise ValueError("positive count and target recall must be valid")
    return int(math.ceil(n_positive * target_recall))


def exact_recall_threshold(
    positive_scores: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, int, float]:
    """Return the highest observed threshold retaining the required positives."""

    scores = np.asarray(positive_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("positive_scores must be a non-empty 1-D array")
    if not np.isfinite(scores).all():
        raise ValueError("positive_scores contains a non-finite value")
    required = required_known_count(scores.size, target_recall)
    threshold = float(np.sort(scores)[::-1][required - 1])
    achieved = float(np.mean(scores >= threshold))
    return threshold, required, achieved


def prepare_spectral_features(X_clean: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Return finite contiguous float32 model features."""

    X = np.asarray(X_clean, dtype=np.float32)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X_clean must be a finite 2-D array")
    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)
    return np.ascontiguousarray(X)


def _xgb_params(config: CrossFitConfig, seed: int) -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(config.max_depth),
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(config.min_child_weight),
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": float(config.reg_lambda),
        "tree_method": "hist",
        "base_score": 0.5,
        "seed": int(seed),
        "verbosity": 0,
        "nthread": 1,
    }


def make_fixed_folds(
    y: np.ndarray,
    config: CrossFitConfig,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Create deterministic folds shared by every configuration."""

    y = np.asarray(y, dtype=np.int8)
    folds: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for repeat in range(config.n_repeats):
        splitter = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_seed + repeat * 10_000,
        )
        folds.append(
            [(train.copy(), holdout.copy()) for train, holdout in splitter.split(np.zeros_like(y), y)]
        )
    return folds


def run_crossfit_pu(
    X: np.ndarray,
    y: np.ndarray,
    config: CrossFitConfig,
    progress: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    fixed_folds: list[list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> dict[str, object]:
    """Run repeated OOF PU bagging and compute the exact recall threshold."""

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X and y have incompatible shapes")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    n_positive = int(y.sum())
    n_unlabeled = int(y.size - n_positive)
    config.validate(n_positive, n_unlabeled)

    folds = fixed_folds if fixed_folds is not None else make_fixed_folds(y, config)
    if len(folds) != config.n_repeats:
        raise ValueError("fixed_folds repeat count does not match config")

    scores_by_repeat = np.full((config.n_repeats, y.size), np.nan, dtype=np.float64)
    bag_std_by_repeat = np.full_like(scores_by_repeat, np.nan)
    fold_by_repeat = np.full((config.n_repeats, y.size), -1, dtype=np.int16)
    history: list[dict[str, object]] = []
    experiment_start = time.perf_counter()

    for repeat, repeat_folds in enumerate(folds):
        split_seed = config.random_seed + repeat * 10_000
        if progress:
            print(f"Repeat {repeat + 1}/{config.n_repeats} (U:P={config.u_to_p_ratio}:1)", flush=True)
        if len(repeat_folds) != config.n_splits:
            raise ValueError("fixed_folds fold count does not match config")

        for fold, (train_idx, holdout_idx) in enumerate(repeat_folds):
            fold_start = time.perf_counter()
            train_positive = train_idx[y[train_idx] == 1]
            train_unlabeled = train_idx[y[train_idx] == 0]
            sample_size = train_positive.size * config.u_to_p_ratio
            if train_unlabeled.size < sample_size:
                raise ValueError("not enough unlabeled training objects for requested PU bags")

            holdout_matrix = xgb.DMatrix(X[holdout_idx])
            prediction_sum = np.zeros(holdout_idx.size, dtype=np.float64)
            prediction_sq_sum = np.zeros(holdout_idx.size, dtype=np.float64)
            rng = np.random.default_rng(split_seed + fold * 1_000 + 17)

            for bag in range(config.n_bags):
                sampled_unlabeled = rng.choice(train_unlabeled, size=sample_size, replace=False)
                train_indices = np.concatenate([train_positive, sampled_unlabeled])
                train_labels = np.concatenate(
                    [
                        np.ones(train_positive.size, dtype=np.float32),
                        np.zeros(sample_size, dtype=np.float32),
                    ]
                )
                model = xgb.train(
                    _xgb_params(config, split_seed + fold * 1_000 + bag),
                    xgb.DMatrix(X[train_indices], label=train_labels),
                    num_boost_round=config.num_boost_round,
                    verbose_eval=False,
                )
                prediction = model.predict(holdout_matrix).astype(np.float64, copy=False)
                prediction_sum += prediction
                prediction_sq_sum += prediction**2

            fold_score = prediction_sum / config.n_bags
            fold_bag_std = np.sqrt(
                np.maximum(prediction_sq_sum / config.n_bags - fold_score**2, 0.0)
            )
            scores_by_repeat[repeat, holdout_idx] = fold_score
            bag_std_by_repeat[repeat, holdout_idx] = fold_bag_std
            fold_by_repeat[repeat, holdout_idx] = fold
            holdout_y = y[holdout_idx]

            row = {
                "config_key": config.key,
                "repeat": repeat + 1,
                "fold": fold + 1,
                "split_seed": split_seed,
                "u_to_p_ratio": config.u_to_p_ratio,
                "max_depth": config.max_depth,
                "min_child_weight": config.min_child_weight,
                "reg_lambda": config.reg_lambda,
                "n_bags": config.n_bags,
                "num_boost_round": config.num_boost_round,
                "target_recall": config.target_recall,
                "n_train": int(train_idx.size),
                "n_train_positive": int(train_positive.size),
                "n_sampled_unlabeled_per_bag": int(sample_size),
                "n_holdout": int(holdout_idx.size),
                "n_holdout_positive": int(holdout_y.sum()),
                "fold_roc_auc": float(roc_auc_score(holdout_y, fold_score)),
                "fold_pr_auc": float(average_precision_score(holdout_y, fold_score)),
                "fold_score_min": float(fold_score.min()),
                "fold_score_max": float(fold_score.max()),
                "fold_seconds": float(time.perf_counter() - fold_start),
                "elapsed_seconds": float(time.perf_counter() - experiment_start),
            }
            history.append(row)
            if progress:
                print(
                    f"  fold {fold + 1}/{config.n_splits}: "
                    f"P_train={train_positive.size}, U/bag={sample_size}, "
                    f"ROC={row['fold_roc_auc']:.4f}, PR={row['fold_pr_auc']:.4f}, "
                    f"{row['fold_seconds']:.1f}s",
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(row)

    if not np.isfinite(scores_by_repeat).all():
        raise RuntimeError("every object must receive exactly one OOF score per repeat")
    if np.any((scores_by_repeat < 0.0) | (scores_by_repeat > 1.0)):
        raise RuntimeError("binary:logistic produced a score outside [0, 1]")

    score_mean = scores_by_repeat.mean(axis=0)
    score_repeat_std = scores_by_repeat.std(axis=0, ddof=1 if config.n_repeats > 1 else 0)
    score_bag_std = bag_std_by_repeat.mean(axis=0)
    positive_mask = y == 1
    threshold, required, achieved = exact_recall_threshold(
        score_mean[positive_mask], config.target_recall
    )
    repeat_thresholds = np.array(
        [
            exact_recall_threshold(scores_by_repeat[r, positive_mask], config.target_recall)[0]
            for r in range(config.n_repeats)
        ]
    )
    repeat_candidate_counts = np.array(
        [
            int(np.sum(scores_by_repeat[r, ~positive_mask] >= repeat_thresholds[r]))
            for r in range(config.n_repeats)
        ],
        dtype=np.int64,
    )
    selection_frequency = np.mean(scores_by_repeat >= repeat_thresholds[:, None], axis=0)

    return {
        "config": config,
        "config_key": config.key,
        "scores_by_repeat": scores_by_repeat,
        "bag_std_by_repeat": bag_std_by_repeat,
        "fold_by_repeat": fold_by_repeat,
        "score_mean": score_mean,
        "score_repeat_std": score_repeat_std,
        "score_bag_std": score_bag_std,
        "repeat_thresholds": repeat_thresholds,
        "repeat_candidate_counts": repeat_candidate_counts,
        "selection_frequency": selection_frequency,
        "threshold": threshold,
        "required_known": required,
        "achieved_known_recall": achieved,
        "roc_auc": float(roc_auc_score(y, score_mean)),
        "pr_auc": float(average_precision_score(y, score_mean)),
        "history": pd.DataFrame(history),
        "elapsed_seconds": float(time.perf_counter() - experiment_start),
    }


def load_project_data() -> dict[str, object]:
    """Load the DR13-all cache through the established project API."""

    from build_dr13_all_cache import load_dr13_all_cache

    return load_dr13_all_cache()


def label_array(stars: pd.DataFrame) -> np.ndarray:
    """Map ``label`` in {-1, 1} to a binary PU target."""

    if not set(stars["label"].unique()).issubset({-1, 1}):
        raise ValueError("stars label must contain only -1 and 1")
    return stars["label"].map({1: 1, -1: 0}).to_numpy(dtype=np.int8)


def prepare_output_tables(
    stars: pd.DataFrame,
    result: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Build the in-memory all-score, known-star and candidate tables."""

    table = stars.copy().reset_index(drop=True)
    table.insert(0, "source_index", np.arange(len(table), dtype=np.int64))
    scores_by_repeat = np.asarray(result["scores_by_repeat"])
    folds = np.asarray(result["fold_by_repeat"])
    table["xgb_pu_score"] = np.asarray(result["score_mean"])
    table["xgb_pu_repeat_std"] = np.asarray(result["score_repeat_std"])
    table["xgb_pu_bag_std"] = np.asarray(result["score_bag_std"])
    table["xgb_pu_selection_frequency"] = np.asarray(result["selection_frequency"])
    for repeat in range(scores_by_repeat.shape[0]):
        table[f"xgb_pu_score_repeat_{repeat + 1}"] = scores_by_repeat[repeat]
        table[f"xgb_pu_fold_repeat_{repeat + 1}"] = folds[repeat]

    recall = float(result["config"].target_recall)
    flag = f"selected_at_{recall_tag(recall)}"
    threshold = float(result["threshold"])
    table[flag] = (table["label"] == -1) & (table["xgb_pu_score"] >= threshold)

    candidates = table.loc[table[flag]].copy()
    candidates = candidates.sort_values("xgb_pu_score", ascending=False).reset_index(drop=True)
    candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))

    known = table.loc[table["label"] == 1].copy()
    known = known.sort_values("xgb_pu_score", ascending=False).reset_index(drop=True)

    return {"all_scores": table, "known_cn": known, "candidates": candidates}


def result_summary(
    result: dict[str, object],
    tables: dict[str, pd.DataFrame],
    stage: str,
) -> dict[str, object]:
    """Assemble a JSON-safe summary for one run."""

    all_scores = tables["all_scores"]
    recall = float(result["config"].target_recall)
    threshold = float(result["threshold"])
    # NOTE: the ``selected_at_*`` column only marks *unlabeled* candidates, so
    # the known-star count must be recomputed from the score threshold itself.
    selected_known = int(
        (all_scores.loc[all_scores["label"] == 1, "xgb_pu_score"] >= threshold).sum()
    )
    return {
        "stage": stage,
        "config_key": result["config_key"],
        "config": asdict(result["config"]),
        "n_total": int(len(all_scores)),
        "n_known_cn": int((all_scores["label"] == 1).sum()),
        "n_unlabeled": int((all_scores["label"] == -1).sum()),
        "objective": "binary:logistic",
        "score_interpretation": (
            "out-of-fold PU ensemble score; not a calibrated survey probability"
        ),
        "score_min": float(all_scores["xgb_pu_score"].min()),
        "score_max": float(all_scores["xgb_pu_score"].max()),
        "target_recall": recall,
        "threshold": float(result["threshold"]),
        "required_known": int(result["required_known"]),
        "selected_known": int(selected_known),
        "achieved_known_recall": float(result["achieved_known_recall"]),
        "candidate_count": int(len(tables["candidates"])),
        "roc_auc_treating_u_as_zero": float(result["roc_auc"]),
        "pr_auc_treating_u_as_zero": float(result["pr_auc"]),
        "repeat_thresholds": np.asarray(result["repeat_thresholds"]).tolist(),
        "repeat_candidate_counts": np.asarray(result["repeat_candidate_counts"]).tolist(),
        "threshold_std": (
            float(np.std(result["repeat_thresholds"], ddof=1))
            if len(result["repeat_thresholds"]) > 1
            else 0.0
        ),
        "candidate_count_std": (
            float(np.std(result["repeat_candidate_counts"], ddof=1))
            if len(result["repeat_candidate_counts"]) > 1
            else 0.0
        ),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def save_minimal_outputs(
    output_dir: Path,
    result: dict[str, object],
    tables: dict[str, pd.DataFrame],
    stage: str,
    export_full_scores: bool = False,
) -> dict[str, Path]:
    """Write the reduced export set for one run.

    Files written: candidate table, known-CN table, summary JSON and a compact
    array archive.  The 41k-row all-score table is only written when
    ``export_full_scores`` is True.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = recall_tag(float(result["config"].target_recall))
    paths = {
        "candidates": output_dir / f"crossfit_candidates_{tag}.csv",
        "known_cn": output_dir / f"crossfit_known_cn_{tag}.csv",
        "summary": output_dir / f"crossfit_summary_{tag}.json",
        "arrays": output_dir / f"crossfit_arrays_{tag}.npz",
        "history": output_dir / f"crossfit_training_history_{tag}.csv",
    }
    tables["candidates"].to_csv(paths["candidates"], index=False)
    tables["known_cn"].to_csv(paths["known_cn"], index=False)
    pd.DataFrame(result["history"]).to_csv(paths["history"], index=False)
    np.savez_compressed(
        paths["arrays"],
        scores_by_repeat=np.asarray(result["scores_by_repeat"]),
        bag_std_by_repeat=np.asarray(result["bag_std_by_repeat"]),
        fold_by_repeat=np.asarray(result["fold_by_repeat"]),
        score_mean=np.asarray(result["score_mean"]),
        repeat_thresholds=np.asarray(result["repeat_thresholds"]),
        repeat_candidate_counts=np.asarray(result["repeat_candidate_counts"]),
        label=np.asarray(tables["all_scores"]["label"]),
        uid=np.asarray(tables["all_scores"]["uid"].astype(str)),
    )
    if export_full_scores:
        paths["all_scores"] = output_dir / f"crossfit_all_scores_{tag}.csv"
        tables["all_scores"].to_csv(paths["all_scores"], index=False)
    summary = result_summary(result, tables, stage)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def run_formal_line(
    target_recall: float,
    output_dir: Path,
    config_values: dict[str, object] | None = None,
    progress: bool = True,
    export_full_scores: bool = False,
    data: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one formal cross-fit line and save the reduced export set.

    Returns the data, model matrix, tables and result so notebooks can keep the
    variables alive for custom visualisation.  Pass ``data`` to reuse an
    already-loaded cache and avoid a second full read.
    """

    values = dict(BEST_FORMAL_CONFIG)
    if config_values:
        values.update(config_values)
    config = CrossFitConfig(target_recall=float(target_recall), **values)  # type: ignore[arg-type]

    if data is None:
        data = load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = label_array(stars)
    X = prepare_spectral_features(data["X_clean"], standardize=config.standardize)
    result = run_crossfit_pu(X, y, config, progress=progress)
    tables = prepare_output_tables(stars, result)
    paths = save_minimal_outputs(
        Path(output_dir), result, tables, f"formal_{recall_tag(target_recall)}", export_full_scores
    )
    return {
        "config": config,
        "data": data,
        "stars": stars,
        "X_model": X,
        "common_wave": np.asarray(data["common_wave"]),
        "y": y,
        "result": result,
        "tables": tables,
        "paths": paths,
        "output_dir": Path(output_dir),
    }


def run_config_search(
    output_dir: Path,
    progress: bool = True,
    smoke: bool = False,
    data: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run U:P ratio screening, regularization screening and formal reruns.

    Only the search table, the per-stage summary JSON and the retained
    variables are returned; no per-configuration score tables are exported.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = load_project_data()
    stars = pd.DataFrame(data["stars_clean"]).reset_index(drop=True)
    y = label_array(stars)
    X = prepare_spectral_features(data["X_clean"], standardize=True)

    base = CrossFitConfig(
        n_splits=3 if smoke else 5,
        n_repeats=1,
        n_bags=2 if smoke else 30,
        num_boost_round=3 if smoke else 50,
        target_recall=0.90,
        random_seed=42,
    )
    fixed_folds = make_fixed_folds(y, base)
    search_rows: list[dict[str, object]] = []
    stage_results: dict[str, dict[str, object]] = {}

    ratio_configs = [_replace(base, u_to_p_ratio=r) for r in (1, 3, 5, 10)]
    for config in ratio_configs:
        result = run_crossfit_pu(X, y, config, progress=progress, fixed_folds=fixed_folds)
        tables = prepare_output_tables(stars, result)
        stage_results[config.key] = {"config": config, "result": result, "tables": tables}
        search_rows.append(_search_row(result, tables, "ratio_pre"))

    ratio_frame = pd.DataFrame(search_rows)
    ratio_selected = _select_configs(ratio_frame, ratio_configs, 2)

    regularization_configs: list[CrossFitConfig] = []
    for ratio_config in ratio_selected:
        for depth in (2, 3):
            for weight in (1, 3):
                for lam in (1.0, 5.0):
                    regularization_configs.append(
                        _replace(
                            base,
                            u_to_p_ratio=ratio_config.u_to_p_ratio,
                            max_depth=depth,
                            min_child_weight=weight,
                            reg_lambda=lam,
                        )
                    )
    for config in regularization_configs:
        result = run_crossfit_pu(X, y, config, progress=progress, fixed_folds=fixed_folds)
        tables = prepare_output_tables(stars, result)
        stage_results[config.key] = {"config": config, "result": result, "tables": tables}
        search_rows.append(_search_row(result, tables, "regularization_pre"))

    all_pre_configs = ratio_configs + regularization_configs
    pre_frame = pd.DataFrame(search_rows)
    formal_candidates = _select_configs(pre_frame, all_pre_configs, 3)

    formal_base = _replace(
        base,
        n_repeats=2 if smoke else 3,
        n_bags=3 if smoke else 100,
        num_boost_round=3 if smoke else 50,
    )
    formal_configs = [
        _replace(
            formal_base,
            u_to_p_ratio=c.u_to_p_ratio,
            max_depth=c.max_depth,
            min_child_weight=c.min_child_weight,
            reg_lambda=c.reg_lambda,
        )
        for c in formal_candidates
    ]
    formal_folds = make_fixed_folds(y, formal_base)
    formal_rows: list[dict[str, object]] = []
    formal_results: dict[str, dict[str, object]] = {}
    for config in formal_configs:
        result = run_crossfit_pu(X, y, config, progress=progress, fixed_folds=formal_folds)
        tables = prepare_output_tables(stars, result)
        formal_results[config.key] = {"config": config, "result": result, "tables": tables}
        formal_rows.append(_search_row(result, tables, "formal"))

    search_frame = pd.DataFrame(search_rows + formal_rows)
    search_frame.to_csv(output_dir / "config_search_summary.csv", index=False)

    formal_frame = pd.DataFrame(formal_rows)
    best_key = str(
        formal_frame.sort_values(
            ["candidate_count", "candidate_count_std", "threshold_std", "config_key"]
        ).iloc[0]["config_key"]
    )
    best = formal_results[best_key]
    search_summary = {
        "objective": "minimum candidate count subject to exact 90% known-CN recall",
        "n_total": int(len(stars)),
        "n_known_cn": int((stars["label"] == 1).sum()),
        "stage_counts": {
            "ratio_pre": len(ratio_configs),
            "regularization_pre": len(regularization_configs),
            "formal": len(formal_configs),
        },
        "ratio_preselection_config_keys": [c.key for c in ratio_selected],
        "formal_config_keys": [c.key for c in formal_configs],
        "best_config_key": best_key,
        "best_config": asdict(best["config"]),
        "python_executable": sys.executable,
        "platform": os.name,
    }
    (output_dir / "config_search_summary.json").write_text(
        json.dumps(search_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_manifest(output_dir, extra={"config_search_summary": search_summary})

    return {
        "data": data,
        "stars": stars,
        "X_model": X,
        "common_wave": np.asarray(data["common_wave"]),
        "y": y,
        "search_summary": search_frame,
        "search_summary_meta": search_summary,
        "stage_results": stage_results,
        "formal_results": formal_results,
        "best_config": best["config"],
        "best_result": best["result"],
        "best_tables": best["tables"],
        "output_dir": output_dir,
    }


def _replace(config: CrossFitConfig, **changes: object) -> CrossFitConfig:
    values = asdict(config)
    values.update(changes)
    return CrossFitConfig(**values)  # type: ignore[arg-type]


def _search_row(
    result: dict[str, object],
    tables: dict[str, pd.DataFrame],
    stage: str,
) -> dict[str, object]:
    summary = result_summary(result, tables, stage)
    return {
        "stage": stage,
        "config_key": summary["config_key"],
        "u_to_p_ratio": summary["config"]["u_to_p_ratio"],
        "max_depth": summary["config"]["max_depth"],
        "min_child_weight": summary["config"]["min_child_weight"],
        "reg_lambda": summary["config"]["reg_lambda"],
        "n_repeats": summary["config"]["n_repeats"],
        "n_bags": summary["config"]["n_bags"],
        "num_boost_round": summary["config"]["num_boost_round"],
        "threshold": summary["threshold"],
        "threshold_std": summary["threshold_std"],
        "candidate_count": summary["candidate_count"],
        "candidate_count_std": summary["candidate_count_std"],
        "required_known": summary["required_known"],
        "achieved_known_recall": summary["achieved_known_recall"],
        "roc_auc": summary["roc_auc_treating_u_as_zero"],
        "pr_auc": summary["pr_auc_treating_u_as_zero"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def _select_configs(
    rows: pd.DataFrame,
    configs: Iterable[CrossFitConfig],
    n: int,
) -> list[CrossFitConfig]:
    """Pick the ``n`` best configurations by candidate count, then stability."""

    config_map = {config.key: config for config in configs}
    ordered = rows.sort_values(
        ["candidate_count", "candidate_count_std", "threshold_std", "config_key"]
    )
    selected: list[CrossFitConfig] = []
    for key in ordered["config_key"]:
        if key in config_map and config_map[key] not in selected:
            selected.append(config_map[key])
        if len(selected) >= n:
            break
    return selected


def write_manifest(output_dir: Path, extra: dict[str, object] | None = None) -> Path:
    """Write a small reproducibility manifest without embedding secrets."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_executable": sys.executable,
        "required_python_executable": r"D:\Anaconda\envs\myenv\python.exe",
        "output_dir": str(output_dir.resolve()),
        "files": sorted(
            path.name
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "crossfit_manifest.json"
        ),
    }
    if extra:
        manifest.update(extra)
    path = output_dir / "crossfit_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "search"), default="formal")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--export-full-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "search":
        bundle = run_config_search(args.output_dir, progress=not args.quiet, smoke=args.smoke)
        print("Config search complete")
        print(f"  best config: {bundle['best_config_key']}")
        print(f"  search rows: {len(bundle['search_summary'])}")
    else:
        bundle = run_formal_line(
            args.target_recall,
            args.output_dir / recall_tag(args.target_recall),
            progress=not args.quiet,
            export_full_scores=args.export_full_scores,
        )
        result = bundle["result"]
        tables = bundle["tables"]
        print("Formal line complete")
        print(f"  config: {bundle['config'].key}")
        print(f"  threshold: {result['threshold']:.6f}")
        print(
            f"  known recall: {result['achieved_known_recall']:.4f} "
            f"({result['required_known']} required)"
        )
        print(f"  candidates: {len(tables['candidates']):,}")
    print(f"  outputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
