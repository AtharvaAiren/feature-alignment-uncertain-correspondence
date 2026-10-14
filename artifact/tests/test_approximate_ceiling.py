from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_alignment.ceiling import (
    bayes_orbit_ceiling,
    binary_gaussian_two_world_ceiling,
    two_world_ceiling,
)
from marginal_alignment.campaign.approximate_ceiling import (
    JOINT_OBSERVATION_SEMANTICS,
    coarse_joint_tv_ceiling,
    discrete_two_world_ceiling,
    gaussian_halfspace_indistinguishable_ceiling,
    orbit_indistinguishable_ceiling,
)
from marginal_alignment.campaign.synthetic_ceiling import (
    approximate_orientation_ceiling,
    asymmetric_population_skewness,
    binomial_count_mass,
    run_approximate_symmetry_experiment,
    run_exact_indistinguishable_experiment,
    run_identifiable_nongaussian_experiment,
    run_synthetic_ceiling_suite,
)


class ApproximateCeilingTests(unittest.TestCase):
    def test_exact_overlap_disagreement_formula(self) -> None:
        result = discrete_two_world_ceiling(
            np.asarray([0.6, 0.4]),
            np.asarray([0.4, 0.6]),
            np.asarray([0, 0]),
            np.asarray([1, 0]),
        )
        self.assertAlmostEqual(result["overlap_disagreement"], 0.4)
        self.assertAlmostEqual(result["exact_equal_prior_bayes_accuracy"], 0.8)
        self.assertAlmostEqual(result["joint_observation_total_variation"], 0.2)
        self.assertAlmostEqual(result["coarse_joint_tv_accuracy_upper_bound"], 0.8)
        self.assertIn("joint law", result["joint_observation_semantics"])

    def test_indistinguishable_limit_matches_v1(self) -> None:
        labels0 = np.asarray([0, 0, 1, 1])
        labels1 = np.asarray([0, 1, 0, 1])
        masses = np.full(4, 0.25)
        result = discrete_two_world_ceiling(masses, masses, labels0, labels1)
        delta = float(np.mean(labels0 != labels1))
        self.assertAlmostEqual(
            result["exact_equal_prior_bayes_accuracy"], two_world_ceiling(delta)
        )

    def test_coarse_bound_requires_joint_observation_tv(self) -> None:
        one_draw = approximate_orientation_ceiling(1, 0.05, 1.0)
        many_draws = approximate_orientation_ceiling(64, 0.05, 1.0)
        self.assertGreater(
            many_draws["joint_observation_total_variation"],
            one_draw["joint_observation_total_variation"],
        )
        self.assertGreater(
            many_draws["exact_equal_prior_bayes_accuracy"],
            one_draw["exact_equal_prior_bayes_accuracy"],
        )
        self.assertIn("all calibration samples", JOINT_OBSERVATION_SEMANTICS)

    def test_coarse_relaxation_is_never_below_exact(self) -> None:
        for n in (0, 1, 4, 16, 64):
            for epsilon in (0.0, 0.01, 0.05, 0.1):
                for delta in (0.25, 0.5, 1.0):
                    result = approximate_orientation_ceiling(n, epsilon, delta)
                    self.assertLessEqual(
                        result["exact_equal_prior_bayes_accuracy"],
                        result["coarse_joint_tv_accuracy_upper_bound"] + 1e-12,
                    )
                    if delta == 1.0:
                        self.assertAlmostEqual(
                            result["exact_equal_prior_bayes_accuracy"],
                            result["coarse_joint_tv_accuracy_upper_bound"],
                        )
        self.assertAlmostEqual(coarse_joint_tv_ceiling(1.0, 1.0, 0.0), 0.5)
        self.assertAlmostEqual(coarse_joint_tv_ceiling(1.0, 1.0, 1.0), 1.0)

    def test_gaussian_and_orbit_compatibility(self) -> None:
        for correlation in (-1.0, -0.25, 0.0, 0.75, 1.0):
            self.assertAlmostEqual(
                gaussian_halfspace_indistinguishable_ceiling(correlation),
                binary_gaussian_two_world_ceiling(correlation),
            )
        labels = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        self.assertAlmostEqual(
            orbit_indistinguishable_ceiling(labels), bayes_orbit_ceiling(labels)
        )

    def test_binomial_mass_is_normalized_for_large_sample(self) -> None:
        values = binomial_count_mass(4096, 0.47)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertAlmostEqual(float(values.sum()), 1.0)


class SyntheticBoundaryTests(unittest.TestCase):
    def test_exact_gaussian_and_orbit_validation(self) -> None:
        result = run_exact_indistinguishable_experiment(
            readout_angles_degrees=(0, 90, 180),
            orbit_sizes=(2, 4),
            seeds=tuple(range(3)),
            samples_per_seed=100_000,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["scientific_gates"].values()))
        self.assertLess(result["maximum_absolute_error"], 0.01)

    def test_population_skewness_boundary(self) -> None:
        self.assertEqual(asymmetric_population_skewness(0.0), 0.0)
        self.assertGreater(asymmetric_population_skewness(2.0), 0.0)
        self.assertGreater(
            asymmetric_population_skewness(2.0), asymmetric_population_skewness(0.5)
        )

    def test_approximate_monte_carlo_tracks_exact_value(self) -> None:
        result = run_approximate_symmetry_experiment(
            calibration_sizes=(0, 16, 64),
            epsilons=(0.0, 0.05),
            task_disagreements=(0.5, 1.0),
            seeds=tuple(range(4)),
            trials_per_seed=20_000,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["maximum_aggregate_monte_carlo_z"], 6.0)

    def test_non_gaussian_positive_and_negative_regimes(self) -> None:
        result = run_identifiable_nongaussian_experiment(
            calibration_sizes=(16, 1024),
            asymmetries=(0.0, 2.0),
            seeds=tuple(range(8)),
            evaluation_size=1000,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(result["scientific_gates"].values()))
        lookup = {
            (group["calibration_samples"], group["asymmetry"], group["method"]): group
            for group in result["groups"]
        }
        third = lookup[(1024, 2.0, "third_moment_permutation")]
        second = lookup[(1024, 2.0, "second_moment_permutation")]
        self.assertGreater(third["task_agreement_mean"], second["task_agreement_mean"] + 0.15)
        self.assertGreaterEqual(third["orientation_correct_mean"], 0.9)

    def test_suite_writes_machine_verifiable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_synthetic_ceiling_suite(
                temporary,
                exact_kwargs={
                    "readout_angles_degrees": (0, 90, 180),
                    "orbit_sizes": (2, 4),
                    "seeds": tuple(range(3)),
                    "samples_per_seed": 100_000,
                },
                approximate_kwargs={
                    "calibration_sizes": (0, 16),
                    "epsilons": (0.0, 0.05),
                    "task_disagreements": (1.0,),
                    "seeds": tuple(range(3)),
                    "trials_per_seed": 10_000,
                },
                identifiable_kwargs={
                    "calibration_sizes": (16, 1024),
                    "asymmetries": (0.0, 2.0),
                    "seeds": tuple(range(8)),
                    "evaluation_size": 1000,
                },
            )
            root = Path(result["output_dir"])
            for name in (
                "exact_indistinguishable_rows.jsonl",
                "approximate_two_world_rows.jsonl",
                "identifiable_nongaussian_rows.jsonl",
                "summary.json",
                "manifest.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertGreater(manifest["row_counts"]["exact_indistinguishable"], 0)
            self.assertGreater(manifest["row_counts"]["identifiable_nongaussian"], 0)


if __name__ == "__main__":
    unittest.main()
