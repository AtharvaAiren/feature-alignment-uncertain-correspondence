import inspect
import json
import unittest

import numpy as np

from marginal_alignment.campaign.maps import estimate_covariance
from marginal_alignment.iclr_extension.pseudo_orientation import (
    fit_bures_pseudo_orientation,
)
from marginal_alignment.linalg import symmetric_power


def _rotated_mixture(
    rng: np.random.Generator,
    rows: int,
    angle_degrees: float,
    *,
    dimension: int = 8,
    classes: int = 6,
    noise: float = 0.35,
):
    if rows % classes:
        raise ValueError("rows must be divisible by classes")
    labels = np.repeat(np.arange(classes), rows // classes)
    rng.shuffle(labels)
    angles = 2.0 * np.pi * np.arange(classes) / classes
    prototypes = np.zeros((classes, dimension))
    prototypes[:, :2] = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    clean = prototypes[labels] + rng.normal(scale=noise, size=(rows, dimension))
    angle = np.deg2rad(angle_degrees)
    rotation = np.eye(dimension)
    rotation[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    return clean @ rotation, labels, prototypes


def _synthetic_fit(
    *, minimum_relative_improvement: float = 0.001, source_angle: float = 20.0
):
    rng = np.random.default_rng(123)
    reference, _, prototypes = _rotated_mixture(rng, 6000, 0.0)
    source, _, _ = _rotated_mixture(rng, 6000, source_angle)
    test, test_labels, _ = _rotated_mixture(rng, 3000, source_angle)
    # Duplicate each semantic direction to exercise the hard >=8-anchor rule
    # while retaining the analytically simple six-direction construction.
    readout_weight = 4.0 * np.repeat(prototypes, 2, axis=0)
    fitted = fit_bures_pseudo_orientation(
        source,
        reference,
        readout_weight=readout_weight,
        candidate_ranks=(2,),
        confidence_threshold=0.3,
        minimum_class_mass=10.0,
        minimum_validation_class_mass=2.0,
        minimum_explained_discrepancy_energy=0.5,
        minimum_subspace_stability=0.05,
        minimum_relative_improvement=minimum_relative_improvement,
    )
    return fitted, source, reference, test, test_labels, readout_weight


class PseudoOrientationTests(unittest.TestCase):
    def test_low_rank_soft_anchors_recover_a_conditional_rotation(self):
        fitted, _, _, test, labels, weight = _synthetic_fit()

        base_prediction = (fitted.base_map.apply(test) @ weight.T).argmax(axis=1) // 2
        corrected_prediction = (fitted.apply(test) @ weight.T).argmax(axis=1) // 2
        base_accuracy = float(np.mean(base_prediction == labels))
        corrected_accuracy = float(np.mean(corrected_prediction == labels))

        self.assertEqual(fitted.diagnostics["status"], "PASS")
        self.assertFalse(fitted.diagnostics["fallback_used"])
        self.assertEqual(fitted.diagnostics["selected_rank"], 2)
        self.assertGreater(corrected_accuracy - base_accuracy, 0.10)
        self.assertGreater(corrected_accuracy, 0.94)
        self.assertGreater(
            fitted.diagnostics["validation_relative_improvement"], 0.30
        )
        self.assertGreater(
            fitted.diagnostics["validation_relative_improvement"],
            fitted.diagnostics["required_null_calibrated_relative_improvement"],
        )
        self.assertEqual(
            fitted.diagnostics["null_calibration"]["interpretation"],
            "conservative_heuristic_selection_gate_not_a_significance_test_or_identification_claim",
        )
        self.assertTrue(
            fitted.diagnostics["orthogonal_orientation_refit_on_all_calibration"]
        )
        self.assertEqual(
            fitted.diagnostics["selection_base_map_fit_source_rows"],
            fitted.diagnostics["source_fit_rows"],
        )
        self.assertEqual(fitted.diagnostics["refit_source_rows"], 6000)
        # Diagnostics are intended for an immutable experiment artifact.
        json.dumps(fitted.diagnostics, sort_keys=True, allow_nan=False)

    def test_orientation_is_orthogonal_and_preserves_fitted_bures_covariance(self):
        fitted, source, reference, test, _, _ = _synthetic_fit()
        dimension = source.shape[1]
        _, source_covariance, _ = estimate_covariance(source, "oas", 1e-6)
        _, reference_covariance, _ = estimate_covariance(reference, "oas", 1e-6)
        mapped_covariance = fitted.matrix.T @ source_covariance @ fitted.matrix
        relative_error = np.linalg.norm(
            mapped_covariance - reference_covariance, ord="fro"
        ) / np.linalg.norm(reference_covariance, ord="fro")

        np.testing.assert_allclose(
            fitted.orientation.T @ fitted.orientation,
            np.eye(dimension),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertLess(relative_error, 1e-10)
        self.assertLess(
            fitted.diagnostics["orientation_orthogonality_error"], 1e-12
        )
        self.assertLess(fitted.diagnostics["fitted_covariance_relative_error"], 1e-10)
        self.assertEqual(
            fitted.diagnostics["covariance_preservation_scope"],
            "fitted_regularized_covariances",
        )
        self.assertAlmostEqual(
            fitted.diagnostics["fitted_oas_covariance_relative_error"],
            relative_error,
            places=12,
        )
        empirical_source = np.cov(source, rowvar=False, bias=True)
        empirical_reference = np.cov(reference, rowvar=False, bias=True)
        empirical_mapped = fitted.matrix.T @ empirical_source @ fitted.matrix
        empirical_error = np.linalg.norm(
            empirical_mapped - empirical_reference, ord="fro"
        ) / np.linalg.norm(empirical_reference, ord="fro")
        self.assertAlmostEqual(
            fitted.diagnostics["empirical_covariance_relative_error"],
            empirical_error,
            places=12,
        )
        self.assertAlmostEqual(abs(fitted.diagnostics["orientation_determinant"]), 1.0)

        reference_mean, reference_covariance, _ = estimate_covariance(
            reference, "oas", 1e-6
        )
        # Directly verify the complete affine composition, including the
        # translation induced by whitening about the reference mean.
        correction = (
            symmetric_power(reference_covariance, -0.5)
            @ fitted.orientation
            @ symmetric_power(reference_covariance, 0.5)
        )
        manual = (
            fitted.base_map.apply(test) - reference_mean
        ) @ correction + reference_mean
        np.testing.assert_allclose(fitted.apply(test), manual, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            fitted.offset,
            (fitted.base_map.offset - reference_mean) @ correction + reference_mean,
            rtol=2e-12,
            atol=2e-12,
        )

    def test_independent_row_permutations_cannot_create_calibration_pairs(self):
        fitted, source, reference, test, _, weight = _synthetic_fit()
        rng = np.random.default_rng(991)
        independently_shuffled = fit_bures_pseudo_orientation(
            source[rng.permutation(len(source))],
            reference[rng.permutation(len(reference))],
            readout_weight=weight,
            candidate_ranks=(2,),
            confidence_threshold=0.3,
            minimum_class_mass=10.0,
            minimum_validation_class_mass=2.0,
            minimum_explained_discrepancy_energy=0.5,
            minimum_subspace_stability=0.05,
            minimum_relative_improvement=0.001,
        )

        np.testing.assert_allclose(
            independently_shuffled.apply(test),
            fitted.apply(test),
            rtol=2e-12,
            atol=2e-12,
        )
        self.assertEqual(
            independently_shuffled.diagnostics["source_partition_sha256"],
            fitted.diagnostics["source_partition_sha256"],
        )
        self.assertEqual(
            independently_shuffled.diagnostics["reference_partition_sha256"],
            fitted.diagnostics["reference_partition_sha256"],
        )
        self.assertFalse(fitted.diagnostics["labels_used"])
        self.assertFalse(fitted.diagnostics["pairing_used"])
        self.assertFalse(fitted.diagnostics["evaluation_data_used_for_selection"])

        parameters = set(inspect.signature(fit_bures_pseudo_orientation).parameters)
        self.assertTrue(
            parameters.isdisjoint(
                {
                    "labels",
                    "source_labels",
                    "reference_labels",
                    "source_ids",
                    "reference_ids",
                    "pairs",
                    "evaluation_features",
                }
            )
        )

    def test_unidentified_anchors_fall_back_to_exact_bures(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(500, 5))
        reference = rng.normal(size=(500, 5))
        # Uniform soft predictions never pass the confidence gate.
        fitted = fit_bures_pseudo_orientation(
            source,
            reference,
            readout_weight=np.zeros((4, 5)),
            confidence_threshold=0.9,
            candidate_ranks=(2,),
        )

        self.assertEqual(fitted.diagnostics["status"], "FALLBACK")
        self.assertTrue(fitted.diagnostics["fallback_used"])
        self.assertEqual(fitted.diagnostics["selected_rank"], 0)
        np.testing.assert_array_equal(fitted.matrix, fitted.base_map.matrix)
        np.testing.assert_array_equal(fitted.offset, fitted.base_map.offset)
        np.testing.assert_array_equal(fitted.orientation, np.eye(5))

    def test_calibration_selection_rejects_a_candidate_below_declared_gain(self):
        fitted, _, _, test, _, _ = _synthetic_fit(
            minimum_relative_improvement=0.99
        )
        self.assertEqual(fitted.diagnostics["status"], "FALLBACK")
        self.assertIn(
            "held_out_gain_does_not_exceed_permutation_null_plus_margin",
            fitted.diagnostics["fallback_reason"],
        )
        np.testing.assert_array_equal(fitted.apply(test), fitted.base_map.apply(test))

    def test_responsibility_permutation_null_rejects_identical_marginals(self):
        fitted, _, _, _, _, _ = _synthetic_fit(source_angle=0.0)
        self.assertEqual(fitted.diagnostics["status"], "FALLBACK")
        self.assertEqual(
            fitted.diagnostics["fallback_reason"],
            "held_out_gain_does_not_exceed_permutation_null_plus_margin",
        )
        self.assertEqual(len(fitted.diagnostics["null_relative_improvements"]), 19)
        self.assertLessEqual(
            fitted.diagnostics["validation_relative_improvement"],
            fitted.diagnostics["required_null_calibrated_relative_improvement"],
        )
        self.assertNotIn("identified", json.dumps(fitted.diagnostics).lower())

    def test_rank_dependent_anchor_support_fails_closed(self):
        rng = np.random.default_rng(88)
        reference, _, prototypes = _rotated_mixture(
            rng, 4000, 0.0, classes=4
        )
        source, _, _ = _rotated_mixture(rng, 4000, 20.0, classes=4)
        fitted = fit_bures_pseudo_orientation(
            source,
            reference,
            readout_weight=4.0 * np.repeat(prototypes, 2, axis=0),
            candidate_ranks=(4,),
            confidence_threshold=0.3,
            minimum_class_mass=10.0,
            minimum_validation_class_mass=2.0,
            minimum_explained_discrepancy_energy=0.5,
            minimum_subspace_stability=0.0,
        )
        self.assertEqual(fitted.diagnostics["status"], "FALLBACK")
        records = fitted.diagnostics["candidate_records"]
        self.assertEqual(records[0]["reason"], "rank_dependent_anchor_support_not_met")
        self.assertEqual(records[0]["required_active_classes"], 9)
        self.assertGreaterEqual(fitted.diagnostics["fit_effective_anchor_mass_min"], 1.0)
        self.assertGreaterEqual(
            fitted.diagnostics["validation_effective_anchor_mass_min"], 1.0
        )

    def test_nonfinite_frozen_readout_fails_closed(self):
        rng = np.random.default_rng(9)
        source = rng.normal(size=(300, 4))
        reference = rng.normal(size=(300, 4))
        weight = np.ones((3, 4))
        weight[0, 0] = np.inf
        fitted = fit_bures_pseudo_orientation(
            source,
            reference,
            readout_weight=weight,
            candidate_ranks=(2,),
        )
        self.assertEqual(fitted.diagnostics["status"], "FALLBACK")
        self.assertEqual(
            fitted.diagnostics["fallback_reason"], "nonfinite_frozen_readout"
        )
        np.testing.assert_array_equal(fitted.matrix, fitted.base_map.matrix)


if __name__ == "__main__":
    unittest.main()
