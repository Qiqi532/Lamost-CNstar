"""Unit tests for the consolidated cross-fit XGBoost-PU engine.

These tests use a small synthetic dataset and the standard library test runner:

    D:\\Anaconda\\envs\\myenv\\python.exe -m unittest crossfit.test_crossfit_engine -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crossfit.crossfit_engine import (  # noqa: E402
    CrossFitConfig,
    exact_recall_threshold,
    label_array,
    make_fixed_folds,
    prepare_output_tables,
    prepare_spectral_features,
    recall_tag,
    required_known_count,
    result_summary,
    run_crossfit_pu,
    save_minimal_outputs,
)


def make_synthetic(n_positive: int = 8, n_unlabeled: int = 32, n_features: int = 20, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_total = n_positive + n_unlabeled
    X = rng.normal(size=(n_total, n_features)).astype(np.float32)
    X[:n_positive, 0] += 1.5
    y = np.concatenate([np.ones(n_positive, dtype=np.int8), np.zeros(n_unlabeled, dtype=np.int8)])
    return X, y


def make_stars(n_positive: int = 8, n_unlabeled: int = 32, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_positive + n_unlabeled
    return pd.DataFrame(
        {
            "uid": [f"G{i:014d}" for i in range(n_total)],
            "ra": rng.uniform(0, 360, n_total),
            "dec": rng.uniform(-10, 60, n_total),
            "teff": rng.uniform(4300, 5200, n_total),
            "logg": rng.uniform(0.8, 2.9, n_total),
            "feh": rng.uniform(-2.0, -0.7, n_total),
            "snru": rng.uniform(20, 80, n_total),
            "masked_cluster_id": rng.integers(0, 5, n_total),
            "label": [1] * n_positive + [-1] * n_unlabeled,
        }
    )


def small_config(**changes) -> CrossFitConfig:
    values = dict(
        u_to_p_ratio=2,
        max_depth=3,
        min_child_weight=1,
        reg_lambda=1.0,
        n_splits=4,
        n_repeats=2,
        n_bags=3,
        num_boost_round=3,
        target_recall=0.90,
        random_seed=42,
    )
    values.update(changes)
    return CrossFitConfig(**values)


class ThresholdTests(unittest.TestCase):
    def test_required_count_uses_ceil(self) -> None:
        self.assertEqual(required_known_count(91, 0.90), 82)
        self.assertEqual(required_known_count(91, 0.95), 87)
        self.assertEqual(required_known_count(73, 0.95), 70)

    def test_exact_threshold_is_order_statistic(self) -> None:
        scores = np.linspace(0.0, 1.0, 91)
        threshold, required, achieved = exact_recall_threshold(scores, 0.90)
        self.assertEqual(required, 82)
        self.assertAlmostEqual(threshold, np.sort(scores)[::-1][81])
        self.assertGreaterEqual(achieved, 0.90 - 1e-12)

    def test_threshold_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            exact_recall_threshold(np.array([]), 0.9)
        with self.assertRaises(ValueError):
            exact_recall_threshold(np.array([0.1, np.nan]), 0.9)

    def test_recall_tag(self) -> None:
        self.assertEqual(recall_tag(0.90), "recall90")
        self.assertEqual(recall_tag(0.95), "recall95")


class CrossFitTests(unittest.TestCase):
    def test_every_object_scored_once_per_repeat(self) -> None:
        X, y = make_synthetic()
        config = small_config()
        result = run_crossfit_pu(X, y, config, progress=False)
        self.assertTrue(np.isfinite(result["scores_by_repeat"]).all())
        self.assertEqual(result["scores_by_repeat"].shape, (config.n_repeats, y.size))
        self.assertTrue((result["fold_by_repeat"] >= 0).all())

    def test_scores_bounded(self) -> None:
        X, y = make_synthetic()
        result = run_crossfit_pu(X, y, small_config(), progress=False)
        scores = result["scores_by_repeat"]
        self.assertGreaterEqual(scores.min(), 0.0)
        self.assertLessEqual(scores.max(), 1.0)

    def test_repeatable_for_same_seed(self) -> None:
        X, y = make_synthetic()
        first = run_crossfit_pu(X, y, small_config(), progress=False)
        second = run_crossfit_pu(X, y, small_config(), progress=False)
        np.testing.assert_allclose(first["score_mean"], second["score_mean"])
        self.assertEqual(first["threshold"], second["threshold"])

    def test_u_to_p_ratio_sampling_uses_only_training_folds(self) -> None:
        X, y = make_synthetic()
        config = small_config(u_to_p_ratio=3)
        folds = make_fixed_folds(y, config)
        for repeat_folds in folds:
            for train_idx, holdout_idx in repeat_folds:
                train_positive = train_idx[y[train_idx] == 1]
                train_unlabeled = train_idx[y[train_idx] == 0]
                self.assertEqual(train_positive.size * config.u_to_p_ratio, 18)
                self.assertGreaterEqual(train_unlabeled.size, 18)
                self.assertTrue(set(train_idx).isdisjoint(set(holdout_idx)))
        result = run_crossfit_pu(X, y, config, progress=False, fixed_folds=folds)
        self.assertEqual(result["required_known"], required_known_count(int(y.sum()), 0.90))

    def test_held_out_positive_count_matches_folds(self) -> None:
        X, y = make_synthetic()
        config = small_config()
        result = run_crossfit_pu(X, y, config, progress=False)
        history = result["history"]
        self.assertEqual(int(history["n_holdout_positive"].sum()), int(y.sum()) * config.n_repeats)
        self.assertEqual(len(history), config.n_splits * config.n_repeats)

    def test_ratio_validation_rejects_insufficient_unlabeled(self) -> None:
        X, y = make_synthetic(n_positive=8, n_unlabeled=9)
        with self.assertRaises(ValueError):
            run_crossfit_pu(X, y, small_config(u_to_p_ratio=5), progress=False)


class TableAndExportTests(unittest.TestCase):
    def test_tables_and_summary_agree(self) -> None:
        X, y = make_synthetic()
        stars = make_stars()
        result = run_crossfit_pu(X, y, small_config(), progress=False)
        tables = prepare_output_tables(stars, result)
        candidates = tables["candidates"]
        self.assertTrue((candidates["label"] == -1).all())
        self.assertEqual(len(tables["known_cn"]), int(stars["label"].eq(1).sum()))
        self.assertEqual(len(tables["all_scores"]), len(stars))
        if len(candidates):
            self.assertGreaterEqual(
                float(candidates["xgb_pu_score"].min()), float(result["threshold"]) - 1e-12
            )
        self.assertTrue(
            (tables["all_scores"]["xgb_pu_score"] >= 0.0).all()
            and (tables["all_scores"]["xgb_pu_score"] <= 1.0).all()
        )

    def test_save_minimal_outputs_writes_only_requested_files(self) -> None:
        X, y = make_synthetic()
        stars = make_stars()
        result = run_crossfit_pu(X, y, small_config(), progress=False)
        tables = prepare_output_tables(stars, result)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_minimal_outputs(Path(tmp), result, tables, "test")
            names = sorted(p.name for p in Path(tmp).iterdir())
            self.assertIn("crossfit_candidates_recall90.csv", names)
            self.assertIn("crossfit_known_cn_recall90.csv", names)
            self.assertIn("crossfit_summary_recall90.json", names)
            self.assertIn("crossfit_arrays_recall90.npz", names)
            self.assertNotIn("crossfit_all_scores_recall90.csv", names)
            written = pd.read_csv(paths["candidates"])
            self.assertEqual(len(written), len(tables["candidates"]))
            with np.load(paths["arrays"]) as archive:
                self.assertIn("score_mean", archive.files)
                self.assertEqual(archive["label"].size, len(stars))

    def test_summary_selected_known_is_recomputed_from_threshold(self) -> None:
        """The candidate flag only marks unlabeled stars, so known counts must not use it."""

        X, y = make_synthetic()
        stars = make_stars()
        result = run_crossfit_pu(X, y, small_config(), progress=False)
        tables = prepare_output_tables(stars, result)
        summary = result_summary(result, tables, "test")

        known_scores = tables["all_scores"].loc[tables["all_scores"]["label"] == 1, "xgb_pu_score"]
        expected = int((known_scores >= float(result["threshold"])).sum())
        self.assertEqual(summary["selected_known"], expected)
        self.assertGreater(summary["selected_known"], 0)
        self.assertGreaterEqual(summary["selected_known"], int(result["required_known"]))

    def test_full_score_export_is_opt_in(self) -> None:
        X, y = make_synthetic()
        stars = make_stars()
        result = run_crossfit_pu(X, y, small_config(), progress=False)
        tables = prepare_output_tables(stars, result)
        with tempfile.TemporaryDirectory() as tmp:
            save_minimal_outputs(Path(tmp), result, tables, "test", export_full_scores=True)
            self.assertTrue((Path(tmp) / "crossfit_all_scores_recall90.csv").exists())


class FeaturePreparationTests(unittest.TestCase):
    def test_standardize_is_finite_and_contiguous(self) -> None:
        X, _ = make_synthetic()
        prepared = prepare_spectral_features(X, standardize=True)
        self.assertTrue(np.isfinite(prepared).all())
        self.assertTrue(prepared.flags["C_CONTIGUOUS"])
        self.assertAlmostEqual(float(prepared.mean()), 0.0, places=5)

    def test_label_array_rejects_unknown_labels(self) -> None:
        stars = make_stars()
        stars.loc[0, "label"] = 0
        with self.assertRaises(ValueError):
            label_array(stars)

    def test_label_array_maps_confirmed_positives(self) -> None:
        stars = make_stars()
        mapped = label_array(stars)
        self.assertEqual(int(mapped.sum()), int(stars["label"].eq(1).sum()))
        self.assertTrue(np.isin(mapped, [0, 1]).all())


if __name__ == "__main__":
    unittest.main()
