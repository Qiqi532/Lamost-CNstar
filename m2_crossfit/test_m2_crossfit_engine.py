from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import m2_crossfit.m2_crossfit_engine as engine

from m2_crossfit.m2_crossfit_engine import (
    M2Config,
    STATUS_POSITIVE,
    STATUS_RELIABLE_NEGATIVE,
    STATUS_UNLABELED,
    build_output_tables,
    build_status,
    exact_recall_threshold,
    make_three_class_folds,
    map_new_labels,
    prepare_features,
    required_positive_count,
    run_m2_crossfit,
    run_newlabel_oof_m2,
    train_full_ensemble,
)


def synthetic_data(seed: int = 11):
    rng = np.random.default_rng(seed)
    n_old, n_new_pos, n_new_neg, n_u, n_features = 8, 4, 8, 60, 12
    n_total = n_old + n_new_pos + n_new_neg + n_u
    X = rng.normal(size=(n_total, n_features)).astype(np.float32)
    positive = np.arange(n_old + n_new_pos)
    reliable_negative = np.arange(n_old + n_new_pos, n_old + n_new_pos + n_new_neg)
    X[positive, :2] += 1.5
    X[reliable_negative, :2] -= 0.5
    old = np.arange(n_old)
    new_pos = np.arange(n_old, n_old + n_new_pos)
    new_neg = reliable_negative
    status = build_status(n_total, old, new_pos, new_neg)
    stars = pd.DataFrame(
        {
            "uid": [f"s{i:03d}" for i in range(n_total)],
            "label": [1] * n_old + [-1] * (n_total - n_old),
            "teff": rng.uniform(4300, 5300, n_total),
            "logg": rng.uniform(1, 3, n_total),
            "feh": rng.uniform(-2, -0.5, n_total),
            "snru": rng.uniform(5, 50, n_total),
        }
    )
    return X, stars, old, new_pos, new_neg, status


def small_config(**changes):
    values = dict(
        u_to_p_ratio=2,
        n_splits=4,
        n_repeats=2,
        n_bags=2,
        num_boost_round=3,
        target_recall=0.90,
        random_seed=17,
    )
    values.update(changes)
    return M2Config(**values)


class LabelAndThresholdTests(unittest.TestCase):
    def test_required_count_for_137(self):
        self.assertEqual(required_positive_count(137, 0.90), 124)

    def test_exact_threshold_uses_order_statistic(self):
        score = np.linspace(0, 1, 137)
        threshold, required, achieved = exact_recall_threshold(score, 0.90)
        self.assertEqual(required, 124)
        self.assertAlmostEqual(threshold, np.sort(score)[::-1][123])
        self.assertGreaterEqual(achieved, 0.90)

    def test_map_new_labels_rejects_overlap(self):
        _, stars, _, _, _, _ = synthetic_data()
        frame = pd.DataFrame({"uid": ["s000"], "label": [1]})
        with self.assertRaises(ValueError):
            map_new_labels(stars, frame)

    def test_status_counts(self):
        _, _, _, _, _, status = synthetic_data()
        self.assertEqual(int(np.sum(status == STATUS_POSITIVE)), 12)
        self.assertEqual(int(np.sum(status == STATUS_RELIABLE_NEGATIVE)), 8)
        self.assertEqual(int(np.sum(status == STATUS_UNLABELED)), 60)


class CrossFitTests(unittest.TestCase):
    def test_folds_hold_out_every_row_once_per_repeat(self):
        _, _, _, _, _, status = synthetic_data()
        config = small_config()
        folds = make_three_class_folds(status, config)
        for repeat in folds:
            seen = np.zeros(status.size, dtype=int)
            for train, holdout in repeat:
                self.assertTrue(set(train).isdisjoint(set(holdout)))
                seen[holdout] += 1
            np.testing.assert_array_equal(seen, np.ones(status.size, dtype=int))

    def test_m2_crossfit_scores_are_complete_and_bounded(self):
        X, _, _, _, _, status = synthetic_data()
        result = run_m2_crossfit(prepare_features(X), status, small_config(), progress=False)
        self.assertEqual(result["scores_by_repeat"].shape, (2, len(status)))
        self.assertTrue(np.isfinite(result["scores_by_repeat"]).all())
        self.assertGreaterEqual(float(result["score_mean"].min()), 0.0)
        self.assertLessEqual(float(result["score_mean"].max()), 1.0)
        self.assertEqual(int(result["required_positive"]), 11)

    def test_same_seed_is_repeatable(self):
        X, _, _, _, _, status = synthetic_data()
        prepared = prepare_features(X)
        first = run_m2_crossfit(prepared, status, small_config(), progress=False)
        second = run_m2_crossfit(prepared, status, small_config(), progress=False)
        np.testing.assert_allclose(first["score_mean"], second["score_mean"])

    def test_strict_newlabel_oof_covers_every_new_label(self):
        X, _, old, new_pos, new_neg, _ = synthetic_data()
        result = run_newlabel_oof_m2(
            prepare_features(X), old, new_pos, new_neg, small_config(), progress=False
        )
        self.assertEqual(len(result["oof_score"]), len(new_pos) + len(new_neg))
        self.assertTrue(np.isfinite(result["oof_score"]).all())

    def test_full_ensemble_scores_all_rows(self):
        X, _, _, _, _, status = synthetic_data()
        score = train_full_ensemble(prepare_features(X), status, small_config(), progress=False)
        self.assertEqual(score.shape, (len(status),))
        self.assertTrue(np.isfinite(score).all())


class OutputTableTests(unittest.TestCase):
    def test_candidates_exclude_all_confirmed_labels_and_known_table_has_old_rows(self):
        X, stars, old, _, _, status = synthetic_data()
        prepared = prepare_features(X)
        result = run_m2_crossfit(prepared, status, small_config(), progress=False)
        final_score = train_full_ensemble(prepared, status, small_config(), progress=False)
        tables = build_output_tables(stars, old, status, result, final_score)
        self.assertTrue((tables["candidates"]["m2_status"] == STATUS_UNLABELED).all())
        self.assertEqual(len(tables["known91"]), len(old))
        self.assertTrue(set(tables["known91"]["label"].unique()).issubset({0, 1}))
        self.assertTrue((tables["known91"]["ground_truth_label"] == 1).all())

    def test_archive_uid_array_is_unicode_not_object(self):
        _, stars, _, _, _, _ = synthetic_data()
        uid = engine.archive_uid_array(stars["uid"])
        self.assertEqual(uid.dtype.kind, "U")
        self.assertEqual(uid.tolist(), stars["uid"].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
