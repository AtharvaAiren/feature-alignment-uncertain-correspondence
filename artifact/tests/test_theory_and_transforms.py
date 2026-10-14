from __future__ import annotations

import unittest

import numpy as np

from marginal_alignment.ceiling import (
    bayes_orbit_ceiling,
    binary_gaussian_two_world_ceiling,
    candidate_mixture_world_accuracies,
    cyclic_permutation_group,
    two_world_ceiling,
)
from marginal_alignment.linalg import population_mean_cov
from marginal_alignment.transforms import fit_transform


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self.reference = rng.normal(size=(20000, 6)) @ np.diag([0.4, 0.7, 1.0, 1.3, 1.7, 2.1])
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.target = self.reference @ (rotation @ np.diag([1.2, 0.8, 1.4, 0.6, 1.1, 0.9])).T
        self.target += np.linspace(-0.5, 0.5, 6)

    def test_diagonal_matches_mean_and_variance(self) -> None:
        fitted = fit_transform("diagonal", self.target, self.reference)
        aligned = fitted.apply(self.target)
        np.testing.assert_allclose(aligned.mean(0), self.reference.mean(0), atol=1e-10)
        np.testing.assert_allclose(aligned.std(0), self.reference.std(0), atol=1e-10)

    def test_diagonal_preserves_standardized_vectors(self) -> None:
        fitted = fit_transform("diagonal", self.target, self.reference)
        aligned = fitted.apply(self.target)
        standardized_target = (self.target - self.target.mean(0)) / self.target.std(0)
        standardized_aligned = (aligned - aligned.mean(0)) / aligned.std(0)
        np.testing.assert_allclose(standardized_target, standardized_aligned, atol=1e-10)

    def test_coral_and_bures_match_regularized_population_covariance(self) -> None:
        # With a large sample and tiny shrinkage, both representatives should
        # make the empirical covariance residual negligible.
        _, reference_cov = population_mean_cov(self.reference)
        for method in ("coral", "bures"):
            fitted = fit_transform(method, self.target, self.reference, shrinkage=1e-8)
            _, aligned_cov = population_mean_cov(fitted.apply(self.target))
            relative = np.linalg.norm(aligned_cov - reference_cov, ord="fro") / np.linalg.norm(
                reference_cov, ord="fro"
            )
            self.assertLess(relative, 5e-6, method)

    def test_paired_oracle_recovers_linear_correspondence(self) -> None:
        rng = np.random.default_rng(22)
        reference = rng.normal(size=(10000, 5))
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        target = reference @ q.T + np.arange(5) / 10
        fitted = fit_transform("paired_oracle", target, reference, shrinkage=1e-8)
        self.assertLess(float(np.mean(np.square(fitted.apply(target) - reference))), 1e-10)

    def test_readout_weighted_formula_and_optimum(self) -> None:
        rng = np.random.default_rng(99)
        reference = rng.normal(size=(60000, 5)) @ np.diag([0.6, 0.9, 1.2, 1.6, 2.0])
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        # Independent noise makes the optimal paired-gauge loss nonzero, so
        # the formula check cannot pass vacuously through exact recovery.
        target = reference @ q.T + 0.35 * rng.normal(size=reference.shape)
        weight = rng.normal(size=(4, 5))
        fitted = fit_transform(
            "paired_readout_oracle", target, reference, readout_weight=weight,
            shrinkage=0.0, relative_floor=1e-10,
        )
        self.assertLess(fitted.diagnostics["formula_absolute_error"], 2e-6)
        candidate_loss = fitted.diagnostics["centered_logit_mse_empirical"]
        centering = np.eye(4) - np.ones((4, 4)) / 4
        centered_weight = centering @ weight
        source_mean, source_covariance = population_mean_cov(target)
        reference_mean, reference_covariance = population_mean_cov(reference)
        source_values, source_vectors = np.linalg.eigh(source_covariance)
        reference_values, reference_vectors = np.linalg.eigh(reference_covariance)
        source_inverse_sqrt = (
            source_vectors * (1.0 / np.sqrt(source_values))
        ) @ source_vectors.T
        reference_sqrt = (
            reference_vectors * np.sqrt(reference_values)
        ) @ reference_vectors.T
        for _ in range(20):
            random_q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
            # Every control is in the same covariance-matching gauge family.
            random_matrix = source_inverse_sqrt @ random_q.T @ reference_sqrt
            random_aligned = (target - source_mean) @ random_matrix + reference_mean
            random_loss = float(
                np.mean(np.square((random_aligned - reference) @ centered_weight.T))
            )
            self.assertLessEqual(candidate_loss, random_loss + 1e-8)


class CeilingTests(unittest.TestCase):
    def test_two_world_ceiling(self) -> None:
        self.assertAlmostEqual(two_world_ceiling(0.0), 1.0)
        self.assertAlmostEqual(two_world_ceiling(1.0), 0.5)
        self.assertAlmostEqual(binary_gaussian_two_world_ceiling(0.0), 0.75)
        self.assertAlmostEqual(binary_gaussian_two_world_ceiling(-1.0), 0.5)

    def test_cyclic_orbit_ceiling_is_inverse_class_count(self) -> None:
        rng = np.random.default_rng(7)
        for classes in (2, 4, 8):
            x = rng.normal(size=(5000, 8))
            group = cyclic_permutation_group(8, classes)
            labels = np.stack([np.argmax((x @ action)[:, :classes], axis=1) for action in group])
            self.assertAlmostEqual(bayes_orbit_ceiling(labels), 1.0 / classes, places=12)
            equalizer = candidate_mixture_world_accuracies(labels)
            np.testing.assert_allclose(equalizer, np.full(classes, 1.0 / classes), atol=1e-12)
            # A fixed identity representative has the same uniform-world mean
            # but is not a minimax equalizer: a non-identity world drives it to zero.
            fixed_identity = np.mean(labels == labels[0], axis=1)
            self.assertAlmostEqual(float(fixed_identity.mean()), 1.0 / classes, places=12)
            self.assertAlmostEqual(float(fixed_identity.min()), 0.0, places=12)

    def test_two_world_equalizer_attains_ceiling_in_both_worlds(self) -> None:
        labels = np.asarray([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0],
        ])
        delta = float(np.mean(labels[0] != labels[1]))
        expected = two_world_ceiling(delta)
        equalizer = candidate_mixture_world_accuracies(labels)
        np.testing.assert_allclose(equalizer, np.full(2, expected), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
