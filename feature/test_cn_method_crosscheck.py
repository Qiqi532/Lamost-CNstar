"""Synthetic (no 41k-cache) unit tests for cn_method_crosscheck.

These tests build tiny fake method tables and exercise only the pure bookkeeping /
overlap functions. They never call load_method_tables / load_cache, so they run in
well under a second and stay fully deterministic.
"""

import os
import sys
import unittest

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cn_method_crosscheck as cx  # noqa: E402

# A 5-star fake universe with 4-tier grade flags for BOTH methods.
_UIDS = ["a", "b", "c", "d", "e"]
_INDEX = pd.DataFrame({
    "uid": _UIDS,
    "grade_1p0": [True, True, True, False, False],
    "grade_1p5": [True, True, False, False, False],
    "grade_2p0": [True, False, False, False, False],
    "grade_2p5": [False, False, False, False, False],
    "std_cn3839_z": [2.1, 1.6, 1.2, 0.3, -0.1],
    "std_cn4142_z": [1.9, 1.4, 1.1, 0.2, 0.0],
    "area_cn3839_z": [2.0, 1.5, 0.4, 1.3, -0.2],
    "area_cn4142_z": [1.8, 1.3, 0.3, 1.1, -0.1],
})
_AREA = pd.DataFrame({
    "uid": _UIDS,
    "grade_1p0": [True, True, False, True, False],
    "grade_1p5": [True, False, False, False, False],
    "grade_2p0": [False, False, False, False, False],
    "grade_2p5": [False, False, False, False, False],
    "std_cn3839_z": [2.0, 1.5, 0.4, 1.3, -0.2],
    "std_cn4142_z": [1.8, 1.3, 0.3, 1.1, -0.1],
    "area_cn3839_z": [2.0, 1.5, 0.4, 1.3, -0.2],
    "area_cn4142_z": [1.8, 1.3, 0.3, 1.1, -0.1],
})


class TestGroupLabels(unittest.TestCase):
    def test_categories_and_counts(self):
        gl = cx.group_labels(_INDEX, _AREA, tier="1p0")
        self.assertEqual(len(gl), 5)
        vc = gl.value_counts().to_dict()
        self.assertEqual(vc.get("both"), 2)        # a, b
        self.assertEqual(vc.get("index-only"), 1)  # c
        self.assertEqual(vc.get("area-only"), 1)   # d
        self.assertEqual(vc.get("neither"), 1)     # e
        self.assertTrue(set(gl.unique()) <= {"both", "index-only", "area-only", "neither"})

    def test_default_tier_is_1p0(self):
        # Default tier must equal explicit "1p0".
        gl_default = cx.group_labels(_INDEX, _AREA)          # default tier="1p0"
        gl_1p0 = cx.group_labels(_INDEX, _AREA, tier="1p0")
        pd.testing.assert_series_equal(gl_default, gl_1p0, check_names=False)
        # At 1p5 the partition changes (area drops 'b'), proving tier is effective:
        # both -> {a} only, 'b' becomes index-only.
        gl_1p5 = cx.group_labels(_INDEX, _AREA, tier="1p5")
        vc_1p5 = gl_1p5.value_counts().to_dict()
        self.assertEqual(vc_1p5.get("both"), 1)        # a
        self.assertEqual(vc_1p5.get("index-only"), 1)  # b
        self.assertEqual(vc_1p5.get("area-only", 0), 0)   # zero-count category absent from value_counts


class TestCandidateOverlap(unittest.TestCase):
    def test_jaccard_and_partition(self):
        co = cx.candidate_overlap(_INDEX, _AREA, tier="1p0")
        self.assertEqual(co["n_index"], 3)
        self.assertEqual(co["n_area"], 3)
        self.assertEqual(co["both"], 2)        # a, b
        self.assertEqual(co["index_only"], 1)  # c
        self.assertEqual(co["area_only"], 1)   # d
        self.assertEqual(co["neither"], 1)     # e
        self.assertAlmostEqual(co["jaccard"], 2.0 / 4.0)

    def test_jaccard_zero_when_disjoint(self):
        idx = _INDEX.copy(); are = _AREA.copy()
        # force disjoint: only 'a' passes index, only 'e' passes area
        idx["grade_1p0"] = [True, False, False, False, False]
        are["grade_1p0"] = [False, False, False, False, True]
        co = cx.candidate_overlap(idx, are, tier="1p0")
        self.assertEqual(co["both"], 0)
        self.assertEqual(co["jaccard"], 0.0)
        self.assertEqual(co["neither"], 3)


class TestKnownOverlap(unittest.TestCase):
    def test_known_partition(self):
        ko = cx.known_overlap(_INDEX, _AREA, tier="1p0")
        self.assertEqual(ko["n_index"], 3)
        self.assertEqual(ko["n_area"], 3)
        self.assertEqual(ko["both"], 2)
        self.assertEqual(ko["index_only"], 1)
        self.assertEqual(ko["area_only"], 1)


class TestBuildOverlapSummary(unittest.TestCase):
    def test_four_tier_rows(self):
        df = cx.build_overlap_summary(_INDEX, _AREA, _INDEX, _AREA)
        self.assertEqual(len(df), 4)
        self.assertListEqual(list(df["tier"]), ["1p0", "1p5", "2p0", "2p5"])
        expected_cols = ["tier", "n_index", "n_area", "both", "index_only",
                         "area_only", "jaccard", "known_both", "known_index_only",
                         "known_area_only", "known_neither", "known_recall_index",
                         "known_recall_area"]
        for c in expected_cols:
            self.assertIn(c, df.columns)

    def test_recall_fractions_in_unit_interval(self):
        df = cx.build_overlap_summary(_INDEX, _AREA, _INDEX, _AREA)
        for v in df["known_recall_index"].tolist() + df["known_recall_area"].tolist():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
