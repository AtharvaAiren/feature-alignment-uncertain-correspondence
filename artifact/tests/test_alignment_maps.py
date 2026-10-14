import unittest

import numpy as np

from marginal_alignment.linalg import gram_square_root
from marginal_alignment.campaign.maps import (
    _rank_restrict,
    fit_coordinate_quantile,
    fit_covariance_gauge,
    fit_paired_procrustes,
    fit_paired_ridge,
    fit_sliced_quantile,
)
from marginal_alignment.campaign.method_registry import (
    fit_registered_map,
    method_capabilities,
)
from marginal_alignment.campaign.metrics import (
    evaluate_alignment_with_context,
    evaluate_alignment,
    prepare_evaluation_context,
    sliced_w2_squared,
)


class IclrMapTests(unittest.TestCase):
    def test_symmetric_rank_restriction_rejects_nonzero_cutoff_ties(self):
        tied = np.diag([3.0, 3.0, 1.0, 1.0])
        with self.assertRaisesRegex(
            np.linalg.LinAlgError, "rank boundary splits"
        ):
            _rank_restrict(tied, 1, symmetric_correction=True)

        distinct = np.diag([4.0, 2.0, 1.0, 1.0])
        restricted, retained, diagnostics = _rank_restrict(
            distinct, 1, symmetric_correction=True
        )
        np.testing.assert_allclose(restricted, np.diag([4.0, 1.0, 1.0, 1.0]))
        self.assertGreater(retained, 0.0)
        self.assertFalse(diagnostics["boundary_tie_split"])
        self.assertEqual(diagnostics["effective_rank"], 1)
        self.assertGreater(diagnostics["restricted_map_minimum_eigenvalue"], 0.0)

    def test_gram_square_root_uses_unsquared_factor_conditioning(self):
        rng = np.random.default_rng(2)
        factor = rng.normal(size=(2, 64))
        root, diagnostics = gram_square_root(factor)
        residual = np.linalg.norm(
            root @ root - factor.T @ factor, ord="fro"
        ) / np.linalg.norm(factor.T @ factor, ord="fro")
        self.assertLess(residual, 1e-12)
        self.assertEqual(diagnostics["effective_rank"], 2)
        self.assertEqual(
            diagnostics["algorithm"], "svd_of_factor_without_forming_gram"
        )

    def test_coordinate_quantile_is_out_of_sample_and_matches_coordinates(self):
        rng = np.random.default_rng(3)
        source_fit = rng.standard_t(4, size=(2000, 3)) * [0.5, 2.0, 1.0] + [2.0, -1.0, 3.0]
        reference_fit = rng.laplace(size=(1800, 3)) * [1.0, 0.7, 2.0]
        source_test = rng.standard_t(4, size=(1000, 3)) * [0.5, 2.0, 1.0] + [2.0, -1.0, 3.0]
        fitted = fit_coordinate_quantile(source_fit, reference_fit)
        mapped = fitted.apply(source_test)
        self.assertEqual(mapped.shape, source_test.shape)
        self.assertLess(np.linalg.norm(mapped.mean(0) - reference_fit.mean(0)), 0.25)
        self.assertLess(np.linalg.norm(mapped.std(0) - reference_fit.std(0)), 0.3)

    def test_declared_quantile_knot_budget_is_enforced(self):
        rng = np.random.default_rng(5)
        source = rng.normal(size=(2000, 4))
        reference = rng.standard_t(5, size=(1800, 4))
        fitted = fit_coordinate_quantile(source, reference, knots=257)
        self.assertEqual(fitted.diagnostics["requested_knots"], 257)
        self.assertLessEqual(max(map(len, fitted.source_knots)), 257)
        self.assertEqual(max(map(len, fitted.reference_values)), 257)

    def test_covariance_gauges_share_information_and_match_fitted_covariance(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(3000, 5)) @ rng.normal(size=(5, 5))
        reference = rng.normal(size=(3000, 5)) @ rng.normal(size=(5, 5))
        for gauge in ("coral", "bures"):
            fitted = fit_covariance_gauge(
                source, reference, gauge=gauge, estimator="empirical", relative_floor=1e-9
            )
            mapped = fitted.apply(source)
            residual = np.linalg.norm(np.cov(mapped, rowvar=False, bias=True) - np.cov(
                reference, rowvar=False, bias=True
            )) / np.linalg.norm(np.cov(reference, rowvar=False, bias=True))
            self.assertLess(residual, 1e-6)
            self.assertEqual(method_capabilities(gauge).information_tier, "joint_second_moment")

        rank_four = fit_registered_map(
            "bures", source, reference, covariance_estimator="oas", rank=4
        )
        self.assertEqual(rank_four.diagnostics["rank"], 4)
        self.assertEqual(rank_four.method, "bures_rank4")
        self.assertEqual(
            rank_four.diagnostics["rank_restriction"]["algorithm"],
            "symmetric_eigen_largest_absolute",
        )
        self.assertLess(
            np.linalg.norm(rank_four.matrix - rank_four.matrix.T, ord="fro"),
            1e-12,
        )

    def test_bures_avoids_indefinite_gram_roundoff_without_extra_jitter(self):
        """The Bures middle matrix is SPD even when its formed Gram is not.

        This fixed rank-one construction makes the former Gram/eigendecomposition
        implementation produce a tiny negative eigenvalue on ordinary float64
        LAPACK.  The factor-SVD implementation evaluates the same principal
        square root without squaring the condition number.
        """

        dimension = 16
        rng = np.random.default_rng(0)
        source_basis = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
        reference_basis = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
        eigenvalues = np.zeros(dimension)
        eigenvalues[-1] = dimension

        def samples(basis):
            positive = (basis * np.sqrt(dimension * eigenvalues)).T
            return np.concatenate([positive, -positive], axis=0)

        fitted = fit_covariance_gauge(
            samples(source_basis), samples(reference_basis), gauge="bures",
            estimator="empirical", relative_floor=1.6e-8,
        )
        self.assertTrue(np.all(np.isfinite(fitted.matrix)))
        diagnostics = fitted.diagnostics["bures_middle_square_root"]
        self.assertEqual(
            diagnostics["algorithm"], "svd_of_factor_without_forming_gram"
        )
        self.assertLess(fitted.diagnostics["fitted_covariance_relative_error"], 1e-7)

    def test_sliced_quantile_reduces_declared_projection_discrepancy(self):
        rng = np.random.default_rng(9)
        source = np.column_stack([
            rng.normal(size=2000), rng.normal(size=2000) ** 2,
            rng.choice([-3.0, 3.0], size=2000) + rng.normal(scale=0.2, size=2000),
        ])
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        reference = source @ rotation + rng.normal(scale=0.05, size=source.shape)
        fit_rng = np.random.default_rng(12)
        fitted = fit_sliced_quantile(source[:1200], reference[:1200], rng=fit_rng, projections=16)
        mapped = fitted.apply(source[1200:])
        evaluation_reference = reference[1200:]
        metric_rng_before = np.random.default_rng(22)
        metric_rng_after = np.random.default_rng(22)
        before = sliced_w2_squared(
            source[1200:], evaluation_reference, rng=metric_rng_before, projections=64
        )
        after = sliced_w2_squared(mapped, evaluation_reference, rng=metric_rng_after, projections=64)
        self.assertLess(after, before)

    def test_ranked_sliced_quantile_is_truthful_and_preserves_pca_residual(self):
        rng = np.random.default_rng(10)
        source = rng.normal(size=(600, 12))
        reference = source @ np.linalg.qr(rng.normal(size=(12, 12)))[0]
        fitted = fit_registered_map(
            "sliced_quantile", source[:400], reference[:400],
            rng=np.random.default_rng(11), rank=4, knots=33,
            projections=6, sweeps=2,
        )
        mapped = fitted.apply(source[400:])
        delta = mapped - source[400:]
        basis = fitted.basis
        self.assertIsNotNone(basis)
        assert basis is not None
        orthogonal_delta = delta - (delta @ basis) @ basis.T
        self.assertLess(np.linalg.norm(orthogonal_delta), 1e-10)
        self.assertEqual(fitted.method, "sliced_quantile_rank4")
        self.assertEqual(fitted.diagnostics["requested_rank"], 4)
        self.assertEqual(fitted.diagnostics["effective_rank"], 4)
        self.assertTrue(fitted.diagnostics["orthogonal_residual_preserved"])
        self.assertLessEqual(fitted.diagnostics["maximum_source_knots"], 33)
        self.assertLessEqual(fitted.diagnostics["maximum_reference_knots"], 33)

    def test_true_and_estimator_matched_pair_controls(self):
        rng = np.random.default_rng(11)
        reference = rng.normal(size=(1200, 6))
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        source = reference @ rotation.T
        source_test = rng.normal(size=(500, 6))
        reference_test = source_test @ rotation
        true_map = fit_registered_map(
            "paired_ridge", source, reference, rng=np.random.default_rng(1), ridge=1e-8
        )
        shuffled = fit_registered_map(
            "paired_ridge_shuffled", source, reference,
            rng=np.random.default_rng(1), ridge=1e-8,
        )
        true_error = np.mean(np.square(true_map.apply(source_test) - reference_test))
        shuffled_error = np.mean(np.square(shuffled.apply(source_test) - reference_test))
        self.assertLess(true_error, 1e-8)
        self.assertGreater(shuffled_error, 0.2)
        self.assertEqual(shuffled.diagnostics["pair_control"], "shuffled")

    def test_positive_ridge_dual_solve_matches_primal_estimator(self):
        rng = np.random.default_rng(12)
        source = rng.normal(size=(14, 40))
        reference = source @ rng.normal(size=(40, 40)) + 0.05 * rng.normal(
            size=(14, 40)
        )
        ridge = 0.07
        fitted = fit_paired_ridge(source, reference, ridge=ridge)
        centered_source = source - source.mean(0)
        centered_reference = reference - reference.mean(0)
        scale = max(float(np.square(centered_source).sum() / source.shape[1]), 1e-15)
        expected = np.linalg.solve(
            centered_source.T @ centered_source
            + ridge * scale * np.eye(source.shape[1]),
            centered_source.T @ centered_reference,
        )
        np.testing.assert_allclose(fitted.matrix, expected, rtol=2e-11, atol=2e-11)
        self.assertEqual(fitted.diagnostics["solve_form"], "dual")
        self.assertEqual(fitted.diagnostics["linear_system_dimension"], len(source))

        primal = fit_paired_ridge(source.T, reference.T, ridge=ridge)
        self.assertEqual(primal.diagnostics["solve_form"], "primal")
        self.assertEqual(
            primal.diagnostics["linear_system_dimension"], source.shape[0]
        )

    def test_procrustes_full_rank_gate_rejects_unidentified_completion(self):
        rng = np.random.default_rng(991)
        source = rng.normal(size=(4, 8))
        reference = source + 0.01 * rng.normal(size=(4, 8))
        with self.assertRaisesRegex(
            np.linalg.LinAlgError, "full paired Procrustes is not identified"
        ):
            fit_paired_procrustes(
                source, reference, require_full_centered_rank=True
            )

        source = rng.normal(size=(20, 8))
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        reference = source @ rotation
        fitted = fit_paired_procrustes(
            source, reference, require_full_centered_rank=True
        )
        self.assertTrue(
            fitted.diagnostics["full_orthogonal_gauge_identified"]
        )
        self.assertEqual(fitted.diagnostics["centered_cross_rank"], 8)
        self.assertIn(
            fitted.diagnostics["svd_driver"],
            {"numpy_default", "scipy_gesvd_fallback"},
        )

    def test_low_pair_reduced_rank_ridge_is_finite_and_declares_rank(self):
        rng = np.random.default_rng(992)
        source = rng.normal(size=(4, 32))
        reference = source @ rng.normal(size=(32, 32))
        fitted = fit_paired_ridge(source, reference, ridge=0.01, rank=64)
        self.assertTrue(np.all(np.isfinite(fitted.matrix)))
        self.assertEqual(fitted.diagnostics["effective_fitted_rank"], 3)
        self.assertFalse(fitted.diagnostics["rrr_boundary_tie_split"])

    def test_reduced_rank_ridge_rejects_material_cutoff_tie(self):
        source = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        with self.assertRaisesRegex(
            np.linalg.LinAlgError, "reduced-rank ridge boundary splits"
        ):
            fit_paired_ridge(source, source, ridge=0.01, rank=1)

    def test_zero_ridge_uses_minimum_norm_lstsq_when_underdetermined(self):
        rng = np.random.default_rng(993)
        source = rng.normal(size=(4, 8))
        reference = rng.normal(size=(4, 8))
        fitted = fit_paired_ridge(source, reference, ridge=0.0)
        expected = np.linalg.lstsq(
            source - source.mean(0), reference - reference.mean(0), rcond=None
        )[0]
        np.testing.assert_allclose(fitted.matrix, expected)
        self.assertEqual(fitted.diagnostics["solve_form"], "minimum_norm_lstsq")

    def test_noisy_pair_fraction_and_reduced_rank_are_declared(self):
        rng = np.random.default_rng(14)
        source = rng.normal(size=(100, 8))
        reference = source @ rng.normal(size=(8, 8))
        noisy = fit_registered_map(
            "paired_ridge_noisy", source, reference, rng=np.random.default_rng(8),
            pair_noise_fraction=0.2,
        )
        reduced = fit_registered_map(
            "paired_reduced_rank", source, reference, rng=np.random.default_rng(8),
            rank=3,
        )
        self.assertAlmostEqual(noisy.diagnostics["corrupted_pair_fraction"], 0.2)
        self.assertEqual(reduced.diagnostics["rank"], 3)
        self.assertLessEqual(np.linalg.matrix_rank(reduced.matrix, tol=1e-8), 3)

    def test_capabilities_fail_closed(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(20, 4))
        reference = rng.normal(size=(20, 4))
        with self.assertRaises(ValueError):
            fit_registered_map("mean", source, reference, pair_control="shuffled")
        with self.assertRaises(ValueError):
            fit_registered_map("paired_readout_gauge", source, reference)
        with self.assertRaises(ValueError):
            fit_registered_map("sliced_quantile", source, reference)
        with self.assertRaises(ValueError):
            fit_registered_map(
                "paired_ridge_noisy", source, reference, rng=np.random.default_rng(2),
                pair_noise_fraction=1.5,
            )

    def test_rich_metrics_smoke(self):
        rng = np.random.default_rng(31)
        reference = rng.normal(size=(300, 4))
        source = reference + 0.5
        weight = rng.normal(size=(3, 4))
        labels = np.argmax(reference @ weight.T, axis=1)
        aligned = source - source.mean(0) + reference.mean(0)
        metrics = evaluate_alignment(
            reference, source, aligned, labels=labels, readout_weight=weight,
            rng=np.random.default_rng(32), projections=8, mmd_features=32,
        )
        for key in (
            "aligned_coordinate_variance_relative_error", "aligned_sliced_w2_squared",
            "aligned_rff_mmd_squared", "aligned_accuracy", "negative_log_likelihood",
            "expected_calibration_error",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_prepared_evaluation_context_is_metric_equivalent(self):
        rng = np.random.default_rng(33)
        reference = rng.normal(size=(220, 7))
        source = reference @ rng.normal(size=(7, 7)) + 0.4
        aligned = source - source.mean(0) + reference.mean(0)
        weight = rng.normal(size=(5, 7))
        bias = rng.normal(size=5)
        labels = np.argmax(reference @ weight.T + bias, axis=1)
        context = prepare_evaluation_context(
            reference, source, labels=labels, readout_weight=weight,
            readout_bias=bias, rng=np.random.default_rng(34),
            projections=11, mmd_features=48, distribution_limit=120,
        )
        cached = evaluate_alignment_with_context(aligned, context)
        direct = evaluate_alignment(
            reference, source, aligned, labels=labels, readout_weight=weight,
            readout_bias=bias, rng=np.random.default_rng(34),
            projections=11, mmd_features=48, distribution_limit=120,
        )
        self.assertEqual(set(cached), set(direct))
        for key in cached:
            np.testing.assert_allclose(
                cached[key], direct[key], rtol=0.0, atol=0.0, equal_nan=True,
                err_msg=key,
            )

    def test_distribution_metric_randomness_is_matched(self):
        rng = np.random.default_rng(40)
        reference = rng.normal(size=(250, 5))
        candidate = rng.normal(size=(250, 5))
        weight = rng.normal(size=(3, 5))
        labels = np.argmax(reference @ weight.T, axis=1)
        metrics = evaluate_alignment(
            reference, candidate, candidate.copy(), labels=labels,
            readout_weight=weight, rng=np.random.default_rng(41),
            projections=16, mmd_features=64,
        )
        self.assertEqual(
            metrics["source_sliced_w2_squared"], metrics["aligned_sliced_w2_squared"]
        )
        self.assertEqual(metrics["source_rff_mmd_squared"], metrics["aligned_rff_mmd_squared"])

    def test_distribution_limit_does_not_subsample_task_metrics(self):
        rng = np.random.default_rng(44)
        reference = rng.normal(size=(240, 5))
        source = reference.copy()
        # Damage only rows outside the distribution-diagnostic prefix.  Task
        # agreement must still see the full fixed evaluation set.
        source[80:] *= -1.0
        weight = rng.normal(size=(4, 5))
        labels = np.argmax(reference @ weight.T, axis=1)
        limited = evaluate_alignment(
            reference, source, source, labels=labels, readout_weight=weight,
            rng=np.random.default_rng(45), projections=8, mmd_features=32,
            distribution_limit=80,
        )
        self.assertEqual(limited["distribution_evaluation_rows"], 80.0)
        self.assertLess(limited["reference_prediction_agreement"], 0.8)
        self.assertAlmostEqual(limited["source_mean_l2"], 0.0, places=12)

    def test_diagonal_exposes_exact_source_dependence_preservation(self):
        rng = np.random.default_rng(46)
        source = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
        reference = rng.normal(size=(500, 6)) * np.linspace(0.5, 2.0, 6) + 2.0
        fitted = fit_registered_map("diagonal", source, reference)
        aligned = fitted.apply(source)
        weight = rng.normal(size=(4, 6))
        labels = np.argmax(reference @ weight.T, axis=1)
        metrics = evaluate_alignment(
            reference, source, aligned, labels=labels, readout_weight=weight,
            rng=np.random.default_rng(47), projections=8, mmd_features=32,
        )
        self.assertLess(metrics["source_to_aligned_standardized_vector_mse"], 1e-24)
        self.assertLess(metrics["source_to_aligned_correlation_relative_change"], 1e-12)
        self.assertEqual(metrics["source_to_aligned_coordinate_rank_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
