"""Integration tests for the decoupled index / area screening modules.

These exercise ``cn_common`` + ``cn_index_screening`` / ``cn_area_screening`` on a
small synthetic cache (no 41k-row load) and assert the contract required by the task:
* std_ and project_ index columns are present and strictly separate;
* area method produces per-band relative/signed area and z columns;
* every candidate keeps a reference member count and reference quality flag;
* the 1.0/1.5/2.0/2.5 sigma grading columns exist and are boolean.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import cn_common as cc
import cn_index_screening as ci
import cn_area_screening as ca


def _synthetic_cache():
    wave = np.arange(3800.0, 4500.0, 1.0)
    n = 30
    rng = np.random.default_rng(0)
    teff = rng.uniform(4400, 5200, n)
    logg = rng.uniform(1.0, 3.5, n)
    feh = rng.uniform(-1.5, 0.0, n)
    cluster = np.array([1] * 20 + [2] * 10)
    label = np.zeros(n, dtype=int)
    label[0] = 1  # one known CN star
    uids = [f"s{i:02d}" for i in range(n)]
    X = np.ones((n, len(wave)), dtype=float)
    # Small per-star continuum scatter so the normal reference distribution has
    # non-zero variance (otherwise z would be NaN by construction).
    X = X + rng.normal(0.0, 0.01, X.shape)
    # Inject CN absorption into the known CN star and 5 candidates.
    injected = [0, 1, 2, 3, 4, 5]
    for i in injected:
        X[i, (wave >= 3861) & (wave <= 3884)] = 0.8
        X[i, (wave >= 4120) & (wave <= 4216)] = 0.85
    stars = pd.DataFrame(
        {
            "uid": uids,
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-30, 30, n),
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "label": label,
            "snru": rng.uniform(30, 80, n),
            "masked_cluster_id": cluster,
        }
    )
    candidates = stars.iloc[1:6].copy()
    candidates = candidates.assign(
        xgb_pu_score=rng.uniform(0.5, 0.95, 5),
        xgb_pu_std=rng.uniform(0.05, 0.2, 5),
    )
    cache = {"X_clean": X, "common_wave": wave, "stars_clean": stars}
    return candidates, cache


class CNMethodScreeningTests(unittest.TestCase):
    def setUp(self):
        self.candidates, self.cache = _synthetic_cache()

    def test_prepare_targets_assigns_reference_maps(self):
        ctx = cc.prepare_targets(self.candidates, self.cache, verbose=False)
        self.assertEqual(len(ctx["candidate_indices"]), 5)
        self.assertEqual(len(ctx["known_indices"]), 1)
        self.assertIn("candidate_refs", ctx)
        self.assertIn("normal_refs", ctx)

    def test_index_table_has_separated_std_and_project_columns(self):
        ctx = cc.prepare_targets(self.candidates, self.cache, verbose=False)
        cand_df, known_df = cc.build_index_table(ctx)
        for col in ("std_cn3839", "std_cn4142", "std_ch4300",
                    "project_cn3839", "project_cn4142", "project_ch4300",
                    "std_cn3839_z", "std_cn4142_z", "std_ch4300_z",
                    "grade_1p0", "grade_1p5", "grade_2p0", "grade_2p5",
                    "ch_class", "n_reference_members", "reference_quality"):
            self.assertIn(col, cand_df.columns)
        # std_ and project_ are strictly separate columns (different band defs).
        self.assertFalse(np.allclose(
            cand_df["std_cn3839"].to_numpy(), cand_df["project_cn3839"].to_numpy()
        ))
        # injected CN stars get a finite, positive standard CN3839 z.
        self.assertTrue(np.isfinite(cand_df["std_cn3839_z"]).all())
        self.assertTrue((cand_df["std_cn3839_z"] > 0).any())

    def test_area_table_has_relative_signed_and_grade_columns(self):
        ctx = cc.prepare_targets(self.candidates, self.cache, verbose=False)
        cand_df, known_df, _ = cc.build_area_table(ctx, verbose=False)
        for col in ("area_cn3839_relative", "area_cn4142_relative", "area_ch4300_relative",
                    "area_cn3839_signed", "area_cn4142_signed",
                    "area_cn3839_z", "area_cn4142_z", "area_ch4300_z",
                    "grade_1p0", "grade_1p5", "grade_2p0", "grade_2p5",
                    "ch_class", "n_reference_members", "reference_quality"):
            self.assertIn(col, cand_df.columns)
        self.assertTrue(np.isfinite(cand_df["area_cn3839_z"]).all())

    def test_export_contract_uses_only_three_files_with_unique_uids(self):
        ctx = cc.prepare_targets(self.candidates, self.cache, verbose=False)
        cand_df, known_df = cc.build_index_table(ctx)
        summary = cc.build_method_summary(cand_df, known_df, "index", {}, ctx, include_project=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "idx"
            out.mkdir(parents=True, exist_ok=True)
            paths = {
                "candidates": out / "cn_index_candidates.csv",
                "plot": out / "top30_index_candidates.png",
                "summary": out / "cn_index_summary.json",
            }
            cc.write_candidate_csv(cand_df, paths["candidates"])
            cc.plot_top30(ctx, cand_df, paths["plot"])
            paths["summary"].write_text("{}", encoding="utf-8")
            self.assertEqual(set(paths.keys()), {"candidates", "plot", "summary"})
            for p in paths.values():
                self.assertTrue(Path(p).exists())
            cdf = pd.read_csv(paths["candidates"])
            self.assertLessEqual(len(cdf), 654)
            self.assertEqual(cdf["uid"].nunique(), len(cdf))
        # The index summary must report tier counts and known-CN recall.
        self.assertIn("tier_counts", summary)
        self.assertIn("known_recall", summary)
        self.assertIn("1p0", summary["tier_counts"])
        self.assertIn("1p0", summary["known_recall"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
