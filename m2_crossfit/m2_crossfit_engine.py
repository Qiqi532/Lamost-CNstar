"""M2 cross-fitted XGBoost-PU training and export utilities.

The M2 recipe combines confirmed positives, confirmed hard negatives and the
remaining survey objects as an unlabeled pool.  All survey scores used for
thresholding are out of fold.  A separate full-label ensemble is trained only
to provide a secondary score after the OOF candidate set has been fixed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


STATUS_UNLABELED = 0
STATUS_POSITIVE = 1
STATUS_RELIABLE_NEGATIVE = -1


@dataclass(frozen=True)
class M2Config:
    """Configuration shared by M2 OOF and full-label ensemble training."""

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

    def validate(self, status: np.ndarray) -> None:
        status = np.asarray(status)
        valid = {STATUS_UNLABELED, STATUS_POSITIVE, STATUS_RELIABLE_NEGATIVE}
        if not set(np.unique(status)).issubset(valid):
            raise ValueError("status must contain only -1, 0 and 1")
        n_pos = int(np.sum(status == STATUS_POSITIVE))
        n_neg = int(np.sum(status == STATUS_RELIABLE_NEGATIVE))
        n_u = int(np.sum(status == STATUS_UNLABELED))
        if n_pos < self.n_splits or n_neg < self.n_splits:
            raise ValueError("positive and reliable-negative counts must support all folds")
        if self.u_to_p_ratio < 1 or self.n_splits < 2:
            raise ValueError("u_to_p_ratio and n_splits are invalid")
        if self.n_repeats < 1 or self.n_bags < 1 or self.num_boost_round < 1:
            raise ValueError("repeat, bag and boosting counts must be positive")
        if not 0 < self.target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        max_train_pos = n_pos - math.floor(n_pos / self.n_splits)
        needed_u = max(max_train_pos * self.u_to_p_ratio - n_neg, 0)
        if n_u < needed_u:
            raise ValueError("not enough unlabeled objects for the requested U:P ratio")

    @property
    def key(self) -> str:
        return (
            f"m2_u{self.u_to_p_ratio}_d{self.max_depth}_w{self.min_child_weight}"
            f"_l{str(self.reg_lambda).replace('.', 'p')}_s{self.n_splits}"
            f"_r{self.n_repeats}_b{self.n_bags}_t{int(round(self.target_recall * 100))}"
        )


def prepare_features(X_clean: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Return finite contiguous float32 features."""

    X = np.asarray(X_clean, dtype=np.float32)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X_clean must be a finite 2-D array")
    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)
    return np.ascontiguousarray(X)


def required_positive_count(n_positive: int, target_recall: float) -> int:
    if n_positive < 1 or not 0 < target_recall <= 1:
        raise ValueError("positive count and target recall must be valid")
    return int(math.ceil(n_positive * target_recall))


def exact_recall_threshold(
    positive_scores: np.ndarray,
    target_recall: float,
) -> tuple[float, int, float]:
    """Return the highest observed threshold retaining the requested positives."""

    scores = np.asarray(positive_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("positive_scores must be a finite non-empty vector")
    required = required_positive_count(scores.size, target_recall)
    threshold = float(np.sort(scores)[::-1][required - 1])
    return threshold, required, float(np.mean(scores >= threshold))


def map_new_labels(
    stars: pd.DataFrame,
    label_frame: pd.DataFrame,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Validate and map the 99 high-resolution labels to survey row indices."""

    required_star_columns = {"uid", "label"}
    required_label_columns = {"uid", "label"}
    if not required_star_columns.issubset(stars.columns):
        raise ValueError("stars must contain uid and label")
    if not required_label_columns.issubset(label_frame.columns):
        raise ValueError("label_frame must contain uid and label")

    stars = stars.reset_index(drop=True)
    labels = label_frame.copy()
    labels["uid"] = labels["uid"].astype(str)
    if labels["uid"].duplicated().any():
        raise ValueError("new labels contain duplicate UIDs")
    if not set(labels["label"].unique()).issubset({0, 1}):
        raise ValueError("new label values must be 0 or 1")

    star_uid = stars["uid"].astype(str)
    if star_uid.duplicated().any():
        raise ValueError("survey stars contain duplicate UIDs")
    uid_to_row = pd.Series(np.arange(len(stars), dtype=int), index=star_uid).to_dict()
    labels["source_index"] = labels["uid"].map(uid_to_row)
    if labels["source_index"].isna().any():
        missing = labels.loc[labels["source_index"].isna(), "uid"].tolist()
        raise ValueError(f"new-label UIDs missing from survey cache: {missing[:5]}")
    labels["source_index"] = labels["source_index"].astype(int)

    old_positive = np.flatnonzero(stars["label"].to_numpy() == 1)
    if labels["source_index"].isin(old_positive).any():
        raise ValueError("new labels overlap the frozen 91 positives")
    new_positive = labels.loc[labels["label"] == 1, "source_index"].to_numpy(dtype=int)
    new_negative = labels.loc[labels["label"] == 0, "source_index"].to_numpy(dtype=int)
    return {
        "labels": labels.sort_values("source_index").reset_index(drop=True),
        "old_positive": old_positive.astype(int),
        "new_positive": new_positive,
        "new_negative": new_negative,
    }


def build_status(
    n_total: int,
    old_positive: np.ndarray,
    new_positive: np.ndarray,
    new_negative: np.ndarray,
) -> np.ndarray:
    """Build the three-state target vector and reject row overlap."""

    groups = [
        np.asarray(old_positive, dtype=int),
        np.asarray(new_positive, dtype=int),
        np.asarray(new_negative, dtype=int),
    ]
    all_labelled = np.concatenate(groups)
    if all_labelled.size != np.unique(all_labelled).size:
        raise ValueError("positive and reliable-negative row sets overlap")
    if all_labelled.size and (all_labelled.min() < 0 or all_labelled.max() >= n_total):
        raise ValueError("label row lies outside the survey table")
    status = np.full(n_total, STATUS_UNLABELED, dtype=np.int8)
    status[np.concatenate(groups[:2])] = STATUS_POSITIVE
    status[groups[2]] = STATUS_RELIABLE_NEGATIVE
    return status


def _xgb_params(config: M2Config, seed: int) -> dict[str, object]:
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


def make_three_class_folds(
    status: np.ndarray,
    config: M2Config,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Create deterministic folds stratified over P, reliable N and U."""

    status = np.asarray(status, dtype=np.int8)
    config.validate(status)
    folds: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for repeat in range(config.n_repeats):
        splitter = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_seed + repeat * 10_000,
        )
        folds.append(
            [
                (train.copy(), holdout.copy())
                for train, holdout in splitter.split(np.zeros(status.size), status)
            ]
        )
    return folds


def _train_bagged_fold(
    X: np.ndarray,
    holdout_idx: np.ndarray,
    train_positive: np.ndarray,
    train_reliable_negative: np.ndarray,
    train_unlabeled: np.ndarray,
    config: M2Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one M2 bagging fold and return mean/std holdout scores."""

    n_positive = train_positive.size
    negative_target = n_positive * config.u_to_p_ratio
    n_random_u = max(negative_target - train_reliable_negative.size, 0)
    if n_random_u > train_unlabeled.size:
        raise ValueError("not enough train-fold U rows for M2 sampling")

    holdout_matrix = xgb.DMatrix(X[holdout_idx])
    prediction_sum = np.zeros(holdout_idx.size, dtype=np.float64)
    prediction_sq_sum = np.zeros(holdout_idx.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for bag in range(config.n_bags):
        sampled_u = (
            rng.choice(train_unlabeled, size=n_random_u, replace=False)
            if n_random_u
            else np.empty(0, dtype=int)
        )
        negative_idx = np.concatenate([train_reliable_negative, sampled_u])
        train_idx = np.concatenate([train_positive, negative_idx])
        train_y = np.concatenate(
            [
                np.ones(train_positive.size, dtype=np.float32),
                np.zeros(negative_idx.size, dtype=np.float32),
            ]
        )
        model = xgb.train(
            _xgb_params(config, seed + bag),
            xgb.DMatrix(X[train_idx], label=train_y),
            num_boost_round=config.num_boost_round,
            verbose_eval=False,
        )
        prediction = model.predict(holdout_matrix).astype(np.float64, copy=False)
        prediction_sum += prediction
        prediction_sq_sum += prediction**2
    mean = prediction_sum / config.n_bags
    std = np.sqrt(np.maximum(prediction_sq_sum / config.n_bags - mean**2, 0.0))
    return mean, std


def run_m2_crossfit(
    X: np.ndarray,
    status: np.ndarray,
    config: M2Config,
    progress: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run repeated survey-wide M2 cross-fitting."""

    X = np.asarray(X, dtype=np.float32)
    status = np.asarray(status, dtype=np.int8)
    if X.ndim != 2 or X.shape[0] != status.size:
        raise ValueError("X and status have incompatible shapes")
    config.validate(status)
    folds = make_three_class_folds(status, config)
    scores_by_repeat = np.full((config.n_repeats, status.size), np.nan, dtype=np.float64)
    bag_std_by_repeat = np.full_like(scores_by_repeat, np.nan)
    fold_by_repeat = np.full((config.n_repeats, status.size), -1, dtype=np.int16)
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for repeat, repeat_folds in enumerate(folds):
        if progress:
            print(f"M2 repeat {repeat + 1}/{config.n_repeats}", flush=True)
        for fold, (train_idx, holdout_idx) in enumerate(repeat_folds):
            fold_start = time.perf_counter()
            train_positive = train_idx[status[train_idx] == STATUS_POSITIVE]
            train_reliable_negative = train_idx[
                status[train_idx] == STATUS_RELIABLE_NEGATIVE
            ]
            train_unlabeled = train_idx[status[train_idx] == STATUS_UNLABELED]
            mean, bag_std = _train_bagged_fold(
                X,
                holdout_idx,
                train_positive,
                train_reliable_negative,
                train_unlabeled,
                config,
                seed=config.random_seed + repeat * 100_000 + fold * 1_000 + 17,
            )
            scores_by_repeat[repeat, holdout_idx] = mean
            bag_std_by_repeat[repeat, holdout_idx] = bag_std
            fold_by_repeat[repeat, holdout_idx] = fold
            labelled_holdout = holdout_idx[status[holdout_idx] != STATUS_UNLABELED]
            labelled_y = (status[labelled_holdout] == STATUS_POSITIVE).astype(int)
            labelled_score = mean[np.searchsorted(holdout_idx, labelled_holdout)]
            row: dict[str, object] = {
                "repeat": repeat + 1,
                "fold": fold + 1,
                "n_train_positive": int(train_positive.size),
                "n_train_reliable_negative": int(train_reliable_negative.size),
                "n_train_unlabeled": int(train_unlabeled.size),
                "n_holdout": int(holdout_idx.size),
                "n_holdout_positive": int(np.sum(status[holdout_idx] == STATUS_POSITIVE)),
                "n_holdout_reliable_negative": int(
                    np.sum(status[holdout_idx] == STATUS_RELIABLE_NEGATIVE)
                ),
                "seconds": float(time.perf_counter() - fold_start),
            }
            if np.unique(labelled_y).size == 2:
                row["labelled_roc_auc"] = float(roc_auc_score(labelled_y, labelled_score))
                row["labelled_pr_auc"] = float(
                    average_precision_score(labelled_y, labelled_score)
                )
            else:
                row["labelled_roc_auc"] = float("nan")
                row["labelled_pr_auc"] = float("nan")
            history.append(row)
            if progress:
                print(
                    f"  fold {fold + 1}/{config.n_splits}: "
                    f"P={train_positive.size}, RN={train_reliable_negative.size}, "
                    f"ROC={row['labelled_roc_auc']:.4f}, PR={row['labelled_pr_auc']:.4f}, "
                    f"{row['seconds']:.1f}s",
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(row)

    if not np.isfinite(scores_by_repeat).all() or (fold_by_repeat < 0).any():
        raise RuntimeError("every row must receive one OOF score per repeat")
    if np.any((scores_by_repeat < 0.0) | (scores_by_repeat > 1.0)):
        raise RuntimeError("M2 score outside [0, 1]")

    score_mean = scores_by_repeat.mean(axis=0)
    score_repeat_std = scores_by_repeat.std(axis=0, ddof=1 if config.n_repeats > 1 else 0)
    score_bag_std = bag_std_by_repeat.mean(axis=0)
    positive_mask = status == STATUS_POSITIVE
    unlabeled_mask = status == STATUS_UNLABELED
    labelled_mask = status != STATUS_UNLABELED
    threshold, required, achieved = exact_recall_threshold(
        score_mean[positive_mask], config.target_recall
    )
    repeat_thresholds = np.array(
        [
            exact_recall_threshold(
                scores_by_repeat[r, positive_mask], config.target_recall
            )[0]
            for r in range(config.n_repeats)
        ]
    )
    repeat_candidate_counts = np.array(
        [
            int(np.sum(scores_by_repeat[r, unlabeled_mask] >= repeat_thresholds[r]))
            for r in range(config.n_repeats)
        ],
        dtype=np.int64,
    )
    selection_frequency = np.mean(
        scores_by_repeat >= repeat_thresholds[:, None], axis=0
    )
    labelled_y = positive_mask[labelled_mask].astype(int)
    labelled_score = score_mean[labelled_mask]
    return {
        "config": config,
        "status": status,
        "scores_by_repeat": scores_by_repeat,
        "bag_std_by_repeat": bag_std_by_repeat,
        "fold_by_repeat": fold_by_repeat,
        "score_mean": score_mean,
        "score_repeat_std": score_repeat_std,
        "score_bag_std": score_bag_std,
        "selection_frequency": selection_frequency,
        "repeat_thresholds": repeat_thresholds,
        "repeat_candidate_counts": repeat_candidate_counts,
        "threshold": threshold,
        "required_positive": required,
        "achieved_positive_recall": achieved,
        "labelled_roc_auc": float(roc_auc_score(labelled_y, labelled_score)),
        "labelled_pr_auc": float(average_precision_score(labelled_y, labelled_score)),
        "history": pd.DataFrame(history),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def run_newlabel_oof_m2(
    X: np.ndarray,
    old_positive: np.ndarray,
    new_positive: np.ndarray,
    new_negative: np.ndarray,
    config: M2Config,
    progress: bool = True,
) -> dict[str, object]:
    """Strict 5-fold OOF M2 evaluation on the 99 new labels.

    Held-out new labels are excluded from both the positive/reliable-negative
    sets and the ordinary U pool, so they cannot be sampled as pseudo-negatives.
    """

    X = np.asarray(X, dtype=np.float32)
    old_positive = np.asarray(old_positive, dtype=int)
    new_positive = np.asarray(new_positive, dtype=int)
    new_negative = np.asarray(new_negative, dtype=int)
    labelled = np.concatenate([new_positive, new_negative])
    labelled_y = np.concatenate(
        [np.ones(new_positive.size, dtype=int), np.zeros(new_negative.size, dtype=int)]
    )
    all_rows = np.arange(X.shape[0], dtype=int)
    plain_u = np.setdiff1d(
        all_rows,
        np.concatenate([old_positive, new_positive, new_negative]),
        assume_unique=False,
    )
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    oof = np.full(labelled.size, np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for fold, (train_local, holdout_local) in enumerate(
        splitter.split(np.zeros(labelled.size), labelled_y)
    ):
        fold_start = time.perf_counter()
        train_labelled = labelled[train_local]
        train_positive = np.concatenate(
            [old_positive, np.intersect1d(train_labelled, new_positive)]
        )
        train_reliable_negative = np.intersect1d(train_labelled, new_negative)
        holdout = labelled[holdout_local]
        mean, _ = _train_bagged_fold(
            X,
            holdout,
            train_positive,
            train_reliable_negative,
            plain_u,
            config,
            seed=config.random_seed + fold * 1_000 + 313,
        )
        oof[holdout_local] = mean
        y_holdout = labelled_y[holdout_local]
        row = {
            "fold": fold + 1,
            "n_train_positive": int(train_positive.size),
            "n_train_reliable_negative": int(train_reliable_negative.size),
            "n_holdout": int(holdout.size),
            "roc_auc": float(roc_auc_score(y_holdout, mean)),
            "pr_auc": float(average_precision_score(y_holdout, mean)),
            "seconds": float(time.perf_counter() - fold_start),
        }
        fold_rows.append(row)
        if progress:
            print(
                f"new-label fold {fold + 1}/{config.n_splits}: "
                f"ROC={row['roc_auc']:.4f}, PR={row['pr_auc']:.4f}, "
                f"{row['seconds']:.1f}s",
                flush=True,
            )
    if not np.isfinite(oof).all():
        raise RuntimeError("every new label must receive one strict OOF score")
    return {
        "rows": labelled,
        "labels": labelled_y,
        "oof_score": oof,
        "roc_auc": float(roc_auc_score(labelled_y, oof)),
        "pr_auc": float(average_precision_score(labelled_y, oof)),
        "fold_metrics": pd.DataFrame(fold_rows),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def train_full_ensemble(
    X: np.ndarray,
    status: np.ndarray,
    config: M2Config,
    progress: bool = True,
) -> np.ndarray:
    """Train M2 on all confirmed labels and score the entire survey."""

    X = np.asarray(X, dtype=np.float32)
    status = np.asarray(status, dtype=np.int8)
    config.validate(status)
    positive = np.flatnonzero(status == STATUS_POSITIVE)
    reliable_negative = np.flatnonzero(status == STATUS_RELIABLE_NEGATIVE)
    unlabeled = np.flatnonzero(status == STATUS_UNLABELED)
    negative_target = positive.size * config.u_to_p_ratio
    n_random_u = max(negative_target - reliable_negative.size, 0)
    if n_random_u > unlabeled.size:
        raise ValueError("not enough U rows for the full-label ensemble")
    prediction_sum = np.zeros(X.shape[0], dtype=np.float64)
    matrix_all = xgb.DMatrix(X)
    rng = np.random.default_rng(config.random_seed + 9_000_000)
    for bag in range(config.n_bags):
        sampled_u = rng.choice(unlabeled, size=n_random_u, replace=False)
        negative = np.concatenate([reliable_negative, sampled_u])
        train_idx = np.concatenate([positive, negative])
        train_y = np.concatenate(
            [
                np.ones(positive.size, dtype=np.float32),
                np.zeros(negative.size, dtype=np.float32),
            ]
        )
        model = xgb.train(
            _xgb_params(config, config.random_seed + 9_000_000 + bag),
            xgb.DMatrix(X[train_idx], label=train_y),
            num_boost_round=config.num_boost_round,
            verbose_eval=False,
        )
        prediction_sum += model.predict(matrix_all)
        if progress and (bag + 1) % max(config.n_bags // 10, 1) == 0:
            print(f"full-label ensemble {bag + 1}/{config.n_bags}", flush=True)
    score = prediction_sum / config.n_bags
    if np.any((score < 0.0) | (score > 1.0)):
        raise RuntimeError("full-label score outside [0, 1]")
    return score


def hanley_mcneil_auc_se(y_true: np.ndarray, score: np.ndarray) -> float:
    """Hanley-McNeil standard error for a binary ROC-AUC."""

    y_true = np.asarray(y_true, dtype=int)
    auc = float(roc_auc_score(y_true, score))
    n1 = int(y_true.sum())
    n0 = int(y_true.size - n1)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    return float(
        np.sqrt(
            (auc * (1.0 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2))
            / (n1 * n0)
        )
    )


def evaluate_scores(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "roc_auc_se": hanley_mcneil_auc_se(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)),
        "positive_score_median": float(np.median(score[y_true == 1])),
        "negative_score_median": float(np.median(score[y_true == 0])),
    }


def build_output_tables(
    stars: pd.DataFrame,
    old_positive: np.ndarray,
    status: np.ndarray,
    result: dict[str, object],
    final_score: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Build all-score, candidate and 91-star recognition tables."""

    stars = stars.reset_index(drop=True).copy()
    status = np.asarray(status, dtype=np.int8)
    table = stars.copy()
    table.insert(0, "source_index", np.arange(len(table), dtype=int))
    table["m2_status"] = status
    table["m2_oof_score"] = np.asarray(result["score_mean"])
    table["m2_repeat_std"] = np.asarray(result["score_repeat_std"])
    table["m2_bag_std"] = np.asarray(result["score_bag_std"])
    table["m2_selection_frequency"] = np.asarray(result["selection_frequency"])
    table["final_model_score"] = np.asarray(final_score)
    scores_by_repeat = np.asarray(result["scores_by_repeat"])
    folds_by_repeat = np.asarray(result["fold_by_repeat"])
    for repeat in range(scores_by_repeat.shape[0]):
        table[f"m2_oof_score_repeat_{repeat + 1}"] = scores_by_repeat[repeat]
        table[f"m2_fold_repeat_{repeat + 1}"] = folds_by_repeat[repeat]
    threshold = float(result["threshold"])
    table["m2_selected_at_recall90"] = (
        (status == STATUS_UNLABELED) & (table["m2_oof_score"] >= threshold)
    )

    candidates = table.loc[table["m2_selected_at_recall90"]].copy()
    candidates = candidates.sort_values("m2_oof_score", ascending=False).reset_index(drop=True)
    candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))
    candidates["m2_recall90_threshold"] = threshold

    known91 = table.iloc[np.asarray(old_positive, dtype=int)].copy()
    known91["ground_truth_label"] = 1
    known91["label"] = (known91["m2_oof_score"] >= threshold).astype(int)
    known91["recognition_status"] = np.where(
        known91["label"] == 1, "recognized", "missed"
    )
    known91["positive_rank"] = known91["m2_oof_score"].rank(
        method="min", ascending=False
    ).astype(int)
    if "teff" in known91.columns:
        known91["teff_ge_5000"] = known91["teff"] >= 5000
    known91["m2_recall90_threshold"] = threshold
    known91 = known91.sort_values(
        ["label", "m2_oof_score"], ascending=[True, True]
    ).reset_index(drop=True)
    return {"all_scores": table, "candidates": candidates, "known91": known91}


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def archive_uid_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return a pickle-free one-dimensional Unicode array for NPZ storage."""

    uid = np.asarray(values, dtype=str)
    if uid.ndim != 1:
        raise ValueError("UID values must be one-dimensional")
    return uid


def save_outputs(
    output_dir: Path,
    stars: pd.DataFrame,
    mapping: dict[str, np.ndarray | pd.DataFrame],
    status: np.ndarray,
    result: dict[str, object],
    final_score: np.ndarray,
    baseline_audit: dict[str, float],
    newlabel_oof: dict[str, object],
) -> dict[str, Path]:
    """Write the M2 result tables, archive and JSON summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_output_tables(
        stars,
        np.asarray(mapping["old_positive"]),
        status,
        result,
        final_score,
    )
    paths = {
        "candidates": output_dir / "m2_candidates_recall90.csv",
        "known91": output_dir / "known91_recognition_labels.csv",
        "labelled_oof": output_dir / "m2_labelled_oof_scores.csv",
        "history": output_dir / "m2_training_history.csv",
        "arrays": output_dir / "m2_all_scores.npz",
        "summary": output_dir / "m2_summary.json",
    }
    tables["candidates"].to_csv(paths["candidates"], index=False)
    tables["known91"].to_csv(paths["known91"], index=False)
    pd.DataFrame(result["history"]).to_csv(paths["history"], index=False)

    label_rows = np.asarray(newlabel_oof["rows"], dtype=int)
    labelled_table = stars.iloc[label_rows].copy().reset_index(drop=True)
    labelled_table.insert(0, "source_index", label_rows)
    labelled_table["highres_label"] = np.asarray(newlabel_oof["labels"], dtype=int)
    labelled_table["m2_newlabel_oof_score"] = np.asarray(newlabel_oof["oof_score"])
    labelled_table.to_csv(paths["labelled_oof"], index=False)
    np.savez_compressed(
        paths["arrays"],
        status=np.asarray(status),
        score_mean=np.asarray(result["score_mean"]),
        scores_by_repeat=np.asarray(result["scores_by_repeat"]),
        bag_std_by_repeat=np.asarray(result["bag_std_by_repeat"]),
        fold_by_repeat=np.asarray(result["fold_by_repeat"]),
        selection_frequency=np.asarray(result["selection_frequency"]),
        final_model_score=np.asarray(final_score),
        uid=archive_uid_array(stars["uid"]),
    )

    old_positive = np.asarray(mapping["old_positive"])
    new_positive = np.asarray(mapping["new_positive"])
    new_negative = np.asarray(mapping["new_negative"])
    summary = {
        "config": asdict(result["config"]),
        "n_total": int(len(stars)),
        "n_old_positive": int(old_positive.size),
        "n_new_positive": int(new_positive.size),
        "n_reliable_negative": int(new_negative.size),
        "n_unlabeled": int(np.sum(status == STATUS_UNLABELED)),
        "baseline_external_audit_on_99": baseline_audit,
        "strict_m2_newlabel_oof": {
            "roc_auc": float(newlabel_oof["roc_auc"]),
            "pr_auc": float(newlabel_oof["pr_auc"]),
            "elapsed_seconds": float(newlabel_oof["elapsed_seconds"]),
        },
        "full_survey_m2_oof": {
            "labelled_roc_auc": float(result["labelled_roc_auc"]),
            "labelled_pr_auc": float(result["labelled_pr_auc"]),
            "threshold": float(result["threshold"]),
            "required_positive": int(result["required_positive"]),
            "achieved_positive_recall": float(result["achieved_positive_recall"]),
            "candidate_count": int(len(tables["candidates"])),
            "known91_recognized": int(tables["known91"]["label"].sum()),
            "known91_missed": int((tables["known91"]["label"] == 0).sum()),
            "repeat_thresholds": np.asarray(result["repeat_thresholds"]),
            "repeat_candidate_counts": np.asarray(result["repeat_candidate_counts"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
        },
        "score_interpretation": (
            "M2 scores are ranking scores under constructed PU sampling, not calibrated probabilities"
        ),
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    return paths
