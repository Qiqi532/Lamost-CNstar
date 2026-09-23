import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cn_feature_analysis import (
    InputConsistencyError,
    PROJECT_BAND_DEFS,
    STANDARD_BAND_DEFS,
    AUTHORITY_CANDIDATE_PATH,
    SCORE_SOURCE_PATH,
    build_reference_map,
    compute_area_features,
    compute_index,
    compute_index_with_error,
    load_and_validate_inputs,
    screening_tier,
)


def _wave():
    return np.arange(3800.0, 4501.0, 1.0)


class CNFeatureAnalysisTests(unittest.TestCase):
    def test_flat_spectrum_standard_indices_are_zero(self):
        wave = _wave()
        flux = np.ones_like(wave)
        for definition in STANDARD_BAND_DEFS.values():
            self.assertAlmostEqual(compute_index(flux, wave, definition), 0.0, places=12)


    def test_injected_cn_absorption_increases_standard_cn_index(self):
        wave = _wave()
        flux = np.ones_like(wave)
        mask = (wave >= 3861) & (wave <= 3884)
        flux[mask] = 0.75
        self.assertGreater(compute_index(flux, wave, STANDARD_BAND_DEFS["CN3839"]), 0)


    def test_index_error_propagation_is_analytic_when_errors_are_supplied(self):
        wave = _wave()
        flux = np.ones_like(wave)
        errors = np.full_like(wave, 0.01)
        value, sigma = compute_index_with_error(flux, wave, STANDARD_BAND_DEFS["CN3839"], errors)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertTrue(np.isfinite(sigma) and sigma > 0)
        _, unavailable_sigma = compute_index_with_error(flux, wave, STANDARD_BAND_DEFS["CN3839"])
        self.assertTrue(np.isnan(unavailable_sigma))


    def test_equal_spectra_have_zero_differential_area(self):
        wave = _wave()
        flux = np.ones_like(wave)
        area = compute_area_features(flux, flux, wave, STANDARD_BAND_DEFS["CN3839"])
        self.assertAlmostEqual(area["signed"], 0.0, places=12)
        self.assertAlmostEqual(area["relative"], 0.0, places=12)


    def test_deeper_absorption_has_positive_signed_and_relative_area(self):
        wave = _wave()
        reference = np.ones_like(wave)
        target = reference.copy()
        target[(wave >= 3861) & (wave <= 3884)] = 0.8
        area = compute_area_features(reference, target, wave, STANDARD_BAND_DEFS["CN3839"])
        self.assertGreater(area["signed"], 0)
        self.assertGreater(area["relative"], 0)
        self.assertGreater(area["positive"], 0)


    def test_reference_map_excludes_candidates_and_target_itself(self):
        stars = pd.DataFrame(
            {
                "uid": ["a", "b", "c", "d", "e", "f", "g", "known"],
                "label": [-1, -1, -1, -1, -1, -1, -1, 1],
                "masked_cluster_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "teff": [4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507],
                "logg": [1, 1, 1, 1, 1, 1, 1, 1],
                "feh": [-1, -1, -1, -1, -1, -1, -1, -1],
            }
        )
        refs = build_reference_map(stars, [0, 7], {"a", "b"}, max_members=50)
        self.assertNotIn(0, refs[0]["reference_indices"])
        self.assertNotIn(1, refs[0]["reference_indices"])
        self.assertNotIn(7, refs[0]["reference_indices"])
        self.assertEqual(refs[0]["reference_quality"], "descriptive_only")
        self.assertEqual(refs[7]["reference_quality"], "descriptive_only")


    def test_small_cluster_quality_flags_follow_rules(self):
        rows = []
        for i in range(4):
            rows.append({"uid": f"tiny{i}", "label": -1, "masked_cluster_id": 10, "teff": 4500 + i, "logg": 1, "feh": -1})
        for i in range(7):
            rows.append({"uid": f"small{i}", "label": -1, "masked_cluster_id": 11, "teff": 4600 + i, "logg": 1, "feh": -1})
        stars = pd.DataFrame(rows)
        refs = build_reference_map(stars, [0, 4], set(), max_members=50)
        self.assertEqual(refs[0]["reference_quality"], "insufficient")
        self.assertEqual(refs[4]["reference_quality"], "descriptive_only")


    def test_standard_and_project_definitions_are_separate(self):
        self.assertNotEqual(STANDARD_BAND_DEFS["CN3839"]["band"], PROJECT_BAND_DEFS["CN3839"]["band"])
        self.assertIn("continuum", STANDARD_BAND_DEFS["CN3839"])
        self.assertIn("blue", PROJECT_BAND_DEFS["CN3839"])

    def test_screening_tiers_expose_relaxed_evidence_without_relabeling_strict_class(self):
        self.assertEqual(screening_tier("good", True, True, True, True), "strict_consensus_2sigma")
        self.assertEqual(screening_tier("good", False, False, True, True), "moderate_consensus_1p5sigma")
        self.assertEqual(screening_tier("good", True, False, True, False), "single_method_1p5sigma")
        self.assertEqual(screening_tier("good", False, False, False, False), "no_dual_band_support")
        self.assertEqual(screening_tier("descriptive_only", True, True, True, True), "no_dual_band_support")
        self.assertEqual(screening_tier("insufficient", True, True, True, True), "insufficient_reference")


    def test_input_join_requires_exact_uid_sets_and_preserves_rows(self):
        bundle = load_and_validate_inputs(AUTHORITY_CANDIDATE_PATH, SCORE_SOURCE_PATH, expected_rows=654)
        self.assertEqual(len(bundle.candidates), 654)
        self.assertEqual(bundle.candidates["uid"].nunique(), 654)
        self.assertTrue(bundle.candidates["xgb_pu_score"].notna().all())

        authority_frame = pd.DataFrame({"uid": ["u1", "u2"], "ra": [1, 2], "dec": [3, 4]})
        score_frame = pd.DataFrame({"uid": ["u1", "u3"], "xgb_pu_prob": [0.8, 0.9]})
        with patch("cn_feature_analysis.pd.read_csv", side_effect=[authority_frame, score_frame]):
            with self.assertRaises(InputConsistencyError):
                load_and_validate_inputs(AUTHORITY_CANDIDATE_PATH, SCORE_SOURCE_PATH, expected_rows=None)


    def test_authority_file_is_not_modified_by_input_validation(self):
        before = hashlib.sha256(AUTHORITY_CANDIDATE_PATH.read_bytes()).hexdigest()
        load_and_validate_inputs(AUTHORITY_CANDIDATE_PATH, SCORE_SOURCE_PATH, expected_rows=654)
        after = hashlib.sha256(AUTHORITY_CANDIDATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)
