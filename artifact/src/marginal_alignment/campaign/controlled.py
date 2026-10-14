"""Controlled real-geometry and finite-pair phase-diagram experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file, write_json_atomic
from .method_registry import fit_registered_map


def _centered_logits(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    centering = np.eye(weight.shape[0]) - np.ones((weight.shape[0], weight.shape[0])) / weight.shape[0]
    return features @ (centering @ weight).T


def _centered_weight(weight: np.ndarray) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float64)
    centering = np.eye(value.shape[0]) - np.ones((value.shape[0], value.shape[0])) / value.shape[0]
    return centering @ value


def _row_span_basis(features: np.ndarray) -> tuple[np.ndarray, int]:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(features).mean(0)
    _, singular, right_t = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * (
        float(singular[0]) if len(singular) else 0.0
    )
    rank = int(np.sum(singular > tolerance))
    return right_t[:rank].T, rank


def _identified_readout_energy(
    source_fit: np.ndarray,
    centered_reference_weight: np.ndarray,
    source_from_reference_rotation: np.ndarray,
) -> float:
    """Fraction of task-normal energy lying in the pair-identified source span."""

    basis, _ = _row_span_basis(source_fit)
    # evaluation_source = reference @ rotation.T, so the frozen reference
    # readout expressed in source coordinates is W_c @ rotation.T.
    source_weight = centered_reference_weight @ source_from_reference_rotation.T
    denominator = float(np.square(source_weight).sum())
    if denominator <= 0.0:
        return 1.0
    projected = source_weight @ basis
    return float(np.square(projected).sum() / denominator)


def _task_metrics(reference: np.ndarray, candidate: np.ndarray,
                  weight: np.ndarray) -> dict[str, float]:
    reference_logits = reference @ weight.T
    candidate_logits = candidate @ weight.T
    return {
        "paired_feature_mse": float(np.mean(np.square(candidate - reference))),
        "centered_logit_mse": float(np.mean(np.square(
            _centered_logits(candidate - reference, weight)
        ))),
        "prediction_agreement": float(np.mean(
            candidate_logits.argmax(1) == reference_logits.argmax(1)
        )),
    }


def run_pair_rank_phase_diagram(
    *, seeds: tuple[int, ...] = tuple(range(20)),
    dimensions: tuple[int, ...] = (32, 128, 512),
    classes: tuple[int, ...] = (2, 10, 100),
    pair_ratios: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0),
    threshold_offsets: tuple[int, ...] = (-1, 0, 1, 8),
    pair_noise: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
    evaluation_size: int = 4096,
) -> list[dict[str, Any]]:
    """Empirical orientation recovery versus pair rank, not a claimed theorem."""

    if not seeds or not dimensions or not classes or not pair_noise:
        raise ValueError("seed, dimension, class, and pair-noise axes must be nonempty")
    if any(int(value) < 3 for value in dimensions):
        raise ValueError("pair-rank dimensions must be at least three")
    if any(int(value) < 2 for value in classes):
        raise ValueError("class counts must be at least two")
    if any(not 0.0 <= float(value) <= 1.0 for value in pair_noise):
        raise ValueError("pair-noise rates must lie in [0,1]")
    if any(float(value) <= 0.0 for value in pair_ratios):
        raise ValueError("pair ratios must be positive")
    if set(int(value) for value in threshold_offsets) != {-1, 0, 1, 8}:
        raise ValueError("threshold_offsets must be exactly (-1,0,1,8)")
    if evaluation_size < 2:
        raise ValueError("evaluation_size must be at least two")
    rows = []
    methods = ("paired_procrustes", "paired_ridge", "paired_reduced_rank")
    for seed in seeds:
        for dimension in dimensions:
            rng = np.random.default_rng(9_000_019 + seed * 1009 + dimension)
            rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            ratio_budgets = {
                max(2, int(np.ceil(ratio * dimension))): float(ratio) for ratio in pair_ratios
            }
            threshold_budgets = {
                max(2, dimension + int(offset)): int(offset) for offset in threshold_offsets
            }
            budgets = sorted(set(ratio_budgets) | set(threshold_budgets))
            for class_count in classes:
                weight = rng.normal(size=(class_count, dimension)) / np.sqrt(dimension)
                centered_weight = _centered_weight(weight)
                evaluation_reference = rng.normal(size=(evaluation_size, dimension))
                evaluation_source = evaluation_reference @ rotation.T
                maximum = max(budgets)
                fit_reference_all = rng.normal(size=(maximum, dimension))
                fit_source_all = fit_reference_all @ rotation.T
                for budget in budgets:
                    source_fit = fit_source_all[:budget]
                    reference_fit = fit_reference_all[:budget]
                    _, centered_rank = _row_span_basis(source_fit)
                    identified_energy = _identified_readout_energy(
                        source_fit, centered_weight, rotation
                    )
                    sources = []
                    if budget in ratio_budgets:
                        sources.append(f"ratio:{ratio_budgets[budget]:g}")
                    if budget in threshold_budgets:
                        sources.append(f"threshold:d{threshold_budgets[budget]:+d}")
                    for method in methods:
                        for noise in pair_noise:
                            control = None if noise == 0 else "noisy"
                            rank = min(64, dimension) if method == "paired_reduced_rank" else None
                            fit_rng = np.random.default_rng(
                                int.from_bytes(hashlib.sha256(
                                    f"{seed}|{dimension}|{class_count}|{budget}|{method}|{noise}".encode()
                                ).digest()[:8], "little")
                            )
                            parameters: dict[str, Any] = {}
                            if method != "paired_procrustes":
                                parameters["ridge"] = 1e-5
                            if rank is not None:
                                parameters["rank"] = rank
                            fitted = fit_registered_map(
                                method, source_fit, reference_fit, rng=fit_rng,
                                pair_control=control, pair_noise_fraction=noise,
                                **parameters,
                            )
                            candidate = fitted.apply(evaluation_source)
                            metrics = _task_metrics(evaluation_reference, candidate, weight)
                            rows.append({
                                "schema_version": "marginal_alignment.campaign.pair_phase_row.v2",
                                "seed": seed,
                                "dimension": dimension,
                                "classes": class_count,
                                "raw_readout_rank": int(np.linalg.matrix_rank(weight)),
                                "centered_readout_rank": int(np.linalg.matrix_rank(centered_weight)),
                                "pair_budget": budget,
                                "pairs_per_dimension": float(budget / dimension),
                                "budget_sources": sources,
                                "centered_pair_rank": centered_rank,
                                "identified_centered_readout_energy": identified_energy,
                                "method": method,
                                "pair_noise_rate": noise,
                                "diagnostics": fitted.diagnostics,
                                **metrics,
                            })
    return rows


def pair_phase_scientific_gates(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Machine-check the exact noiseless rank transition and matched controls."""

    if not rows:
        return {"nonempty": False}
    cells: dict[tuple[Any, ...], dict[tuple[str, float], dict[str, Any]]] = {}
    for row in rows:
        key = (row["seed"], row["dimension"], row["classes"], row["pair_budget"])
        treatment = (str(row["method"]), float(row["pair_noise_rate"]))
        if treatment in cells.setdefault(key, {}):
            raise ValueError(f"duplicate pair-phase cell: {key}/{treatment}")
        cells[key][treatment] = row
    dimensions = sorted({int(row["dimension"]) for row in rows})
    unit_keys = sorted({
        (int(row["seed"]), int(row["dimension"]), int(row["classes"])) for row in rows
    })
    thresholds_present = True
    ranks_correct = True
    exact_recovery = True
    full_task_energy = True
    centered_readout_ranks_correct = True
    identified_energy_monotone = True
    matched_controls = True
    full_rank_true: list[float] = []
    full_rank_corrupt: list[float] = []
    all_methods = sorted({str(row["method"]) for row in rows})
    all_noise = sorted({float(row["pair_noise_rate"]) for row in rows})
    for seed, dimension, classes in unit_keys:
        budgets = {int(row["pair_budget"]) for row in rows
                   if (row["seed"], row["dimension"], row["classes"])
                   == (seed, dimension, classes)}
        required = {max(2, dimension - 1), dimension, dimension + 1, dimension + 8}
        thresholds_present &= required.issubset(budgets)
        representative_by_budget = {
            budget: next(
                row for row in rows
                if row["seed"] == seed and row["dimension"] == dimension
                and row["classes"] == classes and row["pair_budget"] == budget
            )
            for budget in budgets
        }
        centered_readout_ranks_correct &= all(
            int(row["centered_readout_rank"]) == min(classes - 1, dimension)
            for row in representative_by_budget.values()
        )
        ordered_energy = [
            float(representative_by_budget[budget]["identified_centered_readout_energy"])
            for budget in sorted(budgets)
        ]
        identified_energy_monotone &= bool(np.all(np.diff(ordered_energy) >= -1e-9))
        if dimension in budgets:
            row_d = next(row for row in rows if row["seed"] == seed
                         and row["dimension"] == dimension and row["classes"] == classes
                         and row["pair_budget"] == dimension)
            ranks_correct &= int(row_d["centered_pair_rank"]) == dimension - 1
        if dimension + 1 in budgets:
            row_full = next(row for row in rows if row["seed"] == seed
                            and row["dimension"] == dimension and row["classes"] == classes
                            and row["pair_budget"] == dimension + 1)
            ranks_correct &= int(row_full["centered_pair_rank"]) == dimension
            full_task_energy &= float(
                row_full["identified_centered_readout_energy"]
            ) >= 1.0 - 1e-9
            true_key = (seed, dimension, classes, dimension + 1)
            true = cells[true_key].get(("paired_procrustes", 0.0))
            exact_recovery &= bool(
                true is not None
                and float(true["paired_feature_mse"]) < 1e-16
                and float(true["prediction_agreement"]) > 1.0 - 1e-10
            )
            if true is not None:
                full_rank_true.append(float(true["prediction_agreement"]))
            corrupt = cells[true_key].get(("paired_procrustes", 1.0))
            if corrupt is not None:
                full_rank_corrupt.append(float(corrupt["prediction_agreement"]))
        for budget in budgets:
            observed = set(cells[(seed, dimension, classes, budget)])
            matched_controls &= observed == {
                (method, noise) for method in all_methods for noise in all_noise
            }
    corruption_gap = (
        bool(full_rank_true and len(full_rank_true) == len(full_rank_corrupt))
        and float(np.mean(full_rank_true) - np.mean(full_rank_corrupt)) > 0.05
    )
    expected_methods = {"paired_procrustes", "paired_ridge", "paired_reduced_rank"}
    gates = {
        "nonempty": True,
        "all_pair_estimators_present": set(all_methods) == expected_methods,
        "true_and_fully_corrupted_controls_present": {0.0, 1.0}.issubset(all_noise),
        "exact_threshold_budgets_present": bool(thresholds_present),
        "centered_rank_changes_between_d_and_d_plus_1": bool(ranks_correct),
        "centered_readout_rank_is_task_relevant_rank": bool(centered_readout_ranks_correct),
        "identified_centered_readout_energy_is_monotone": bool(identified_energy_monotone),
        "noiseless_full_rank_procrustes_recovers": bool(exact_recovery),
        "full_rank_identifies_centered_readout_energy": bool(full_task_energy),
        "matched_method_noise_grid_complete": bool(matched_controls),
        "true_pairs_beat_fully_corrupted_at_full_rank": bool(corruption_gap),
    }
    return gates


def real_orbit_scientific_gates(records: list[dict[str, Any]]) -> dict[str, bool]:
    """Verify exact endpoint invariance and the two distinct decision values."""

    if not records:
        return {"nonempty": False}
    finite_invariance = True
    moment_invariance = True
    valid_mixtures = True
    orbit_completion_sizes = True
    lp_value_attained = True
    bayes_is_upper = True
    task_ambiguity = False
    for record in records:
        finite_invariance &= bool(record.get("finite_multiset_identical", False))
        moment_invariance &= (
            float(record.get("endpoint_mean_residual", np.inf)) <= 1e-10
            and float(record.get("endpoint_covariance_relative_residual", np.inf)) <= 1e-10
        )
        worlds = int(record.get("worlds", 0))
        orbit_completion_sizes &= (
            int(record.get("controlled_endpoint_rows", -1))
            == worlds * int(record.get("geometry_seed_rows", -2))
        )
        mixture = np.asarray(record.get("group_map_mixture", []), dtype=np.float64)
        world_accuracies = np.asarray(
            record.get("group_map_world_accuracies", []), dtype=np.float64
        )
        minimax = float(record.get("achievable_group_map_randomized_minimax", np.nan))
        bayes = float(
            record.get("unrestricted_equal_prior_bayes_prediction_upper", np.nan)
        )
        valid_mixtures &= bool(
            worlds >= 2
            and mixture.shape == (worlds,)
            and np.all(np.isfinite(mixture))
            and np.all(mixture >= -1e-10)
            and abs(float(mixture.sum()) - 1.0) <= 1e-8
            and world_accuracies.shape == (worlds,)
            and np.all(np.isfinite(world_accuracies))
        )
        lp_value_attained &= bool(
            world_accuracies.shape == (worlds,)
            and np.isfinite(minimax)
            and abs(float(world_accuracies.min()) - minimax) <= 1e-8
        )
        # The minimum world accuracy of a restricted randomized map is no
        # larger than its equal-prior average, which is no larger than the
        # unrestricted pointwise posterior-mode value.
        bayes_is_upper &= bool(
            np.isfinite(bayes) and np.isfinite(minimax) and minimax <= bayes + 1e-8
        )
        task_ambiguity |= bool(
            float(record.get("fixed_identity_worst_world_accuracy", 1.0)) < 0.99
            and bayes < 0.999
        )
    return {
        "nonempty": True,
        "finite_endpoint_multisets_are_identical": bool(finite_invariance),
        "endpoint_moments_are_identical": bool(moment_invariance),
        "group_map_mixtures_are_probabilities": bool(valid_mixtures),
        "controlled_reference_is_complete_group_orbit": bool(orbit_completion_sizes),
        "reported_group_minimax_is_attained": bool(lp_value_attained),
        "unrestricted_bayes_value_upper_bounds_group_minimax": bool(bayes_is_upper),
        "at_least_one_orbit_has_nontrivial_task_ambiguity": bool(task_ambiguity),
    }


def controlled_scientific_gates(
    pair_rows: list[dict[str, Any]], orbit_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Nested fail-closed gates for the complete controlled stage."""

    pair = pair_phase_scientific_gates(pair_rows)
    orbit = real_orbit_scientific_gates(orbit_records)
    return {
        "pair_rank_phase": pair,
        "real_embedding_orbit": orbit,
        "all_passed": bool(pair and orbit and all(pair.values()) and all(orbit.values())),
    }


def run_real_embedding_orbit(
    reference: np.ndarray, weight: np.ndarray, *, seed: int,
    bias: np.ndarray | None = None,
    rank: int = 64, worlds: int = 4, evaluation_rows: int = 4096,
    dataset: str | None = None, encoder: str | None = None,
    input_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Orbit-complete real-embedding stress test with distinct ceiling objects.

    The unrestricted posterior-mode value permits an arbitrary task predictor.
    The group-map minimax value restricts the randomized correction to the
    declared inverse group actions and is computed by a finite zero-sum LP.
    These values can differ for a generic, non-equivariant frozen head.
    """

    features = np.asarray(reference, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    readout_bias = (
        np.zeros(weight.shape[0], dtype=np.float64)
        if bias is None else np.asarray(bias, dtype=np.float64)
    )
    if features.ndim != 2 or weight.ndim != 2 or weight.shape[1] != features.shape[1]:
        raise ValueError("invalid real-embedding orbit inputs")
    if readout_bias.shape != (weight.shape[0],):
        raise ValueError("readout bias must have one value per class")
    if len(features) < 3 or features.shape[1] < 2 or weight.shape[0] < 2:
        raise ValueError("real-embedding orbit needs >=3 rows, >=2 dimensions, and >=2 classes")
    if worlds < 2:
        raise ValueError("worlds must be at least two")
    if rank < 2:
        raise ValueError("rank must be at least two")
    if evaluation_rows < 3:
        raise ValueError("evaluation_rows must be at least three")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))[:evaluation_rows]
    x = features[indices]
    mean = x.mean(0)
    centered = x - mean
    _, singular, right_t = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * (
        float(singular[0]) if len(singular) else 0.0
    )
    numerical_rank = int(np.sum(singular > tolerance))
    retained = min(int(rank), numerical_rank)
    if retained < 2:
        raise ValueError("selected real embeddings have fewer than two nondegenerate PCA axes")
    basis = right_t[:retained].T
    scales = np.maximum(singular[:retained] / np.sqrt(len(centered)), 1e-8)
    whitened = centered @ basis / scales
    # Cyclic rotations act only in the whitened retained subspace. The
    # controlled *reference law* is the orbit completion (not the original
    # seed rows alone). It unions every rotated copy, so every group inverse is
    # an admissible marginal-preserving map and the finite endpoint multiset is
    # exactly invariant to the hidden starting world.
    angle = 2.0 * np.pi / worlds
    generator = np.eye(retained)
    if retained >= 2:
        generator[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]]
    rotations = [np.linalg.matrix_power(generator, world) for world in range(worlds)]
    orbit = np.concatenate([whitened @ rotation.T for rotation in rotations], axis=0)

    def recolor(values: np.ndarray) -> np.ndarray:
        return values * scales @ basis.T + mean

    recolored = recolor(orbit)
    # At a common observed orbit point y, hidden world h requires label
    # g(y R_h). Candidate inverse j predicts g(y R_j). Rows therefore encode
    # both the hidden desired tasks and the candidate group-map predictions.
    label_matrix = np.stack([
        (recolor(orbit @ rotation) @ weight.T + readout_bias).argmax(1)
        for rotation in rotations
    ])
    unrestricted_bayes_upper = float(np.mean([
        np.max(np.bincount(label_matrix[:, column], minlength=weight.shape[0])) / worlds
        for column in range(label_matrix.shape[1])
    ]))
    agreement = np.mean(
        label_matrix[:, None, :] == label_matrix[None, :, :], axis=2
    )
    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError("real-embedding group minimax requires scipy") from error
    # maximize v subject to A p >= v, p in simplex.
    objective = np.concatenate([np.zeros(worlds), np.asarray([-1.0])])
    inequalities = np.column_stack([-agreement, np.ones(worlds)])
    optimum = linprog(
        objective, A_ub=inequalities, b_ub=np.zeros(worlds),
        A_eq=np.asarray([[*np.ones(worlds), 0.0]]), b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * worlds + [(0.0, 1.0)], method="highs",
    )
    if not optimum.success:
        raise RuntimeError(f"group-map minimax LP failed: {optimum.message}")
    mixture = np.asarray(optimum.x[:worlds], dtype=np.float64)
    group_map_minimax = float(optimum.x[-1])
    achieved_world_accuracies = agreement @ mixture
    uniform_world_accuracies = agreement @ np.full(worlds, 1.0 / worlds)
    fixed_identity_accuracies = agreement[:, 0]
    sorted_hashes = []
    endpoint_means = []
    endpoint_covariances = []
    for start in range(worlds):
        world_order = np.concatenate([
            whitened @ rotations[(start + offset) % worlds].T for offset in range(worlds)
        ], axis=0)
        endpoint = recolor(world_order)
        endpoint_means.append(endpoint.mean(0))
        endpoint_covariances.append(np.cov(endpoint, rowvar=False, bias=True))
        sorted_rows = np.sort(np.ascontiguousarray(world_order).view(
            np.dtype((np.void, world_order.dtype.itemsize * world_order.shape[1]))
        ).ravel())
        sorted_hashes.append(hashlib.sha256(sorted_rows.tobytes()).hexdigest())
    denominator = max(float(np.square(centered).sum()), 1e-15)
    retained_energy = float(np.square(centered @ basis).sum() / denominator)
    mean_residual = max(float(np.linalg.norm(value - endpoint_means[0]))
                        for value in endpoint_means)
    covariance_denominator = max(
        float(np.linalg.norm(endpoint_covariances[0], ord="fro")), 1e-15
    )
    covariance_residual = max(float(
        np.linalg.norm(value - endpoint_covariances[0], ord="fro") / covariance_denominator
    ) for value in endpoint_covariances)
    return {
        "schema_version": "marginal_alignment.campaign.real_orbit.v2",
        "dataset": dataset,
        "encoder": encoder,
        "input_manifest_sha256": input_manifest_sha256,
        "seed": seed,
        "worlds": worlds,
        "requested_pca_rank": int(rank),
        "available_numerical_rank": numerical_rank,
        "retained_pca_rank": retained,
        "orthogonal_action_rank": 2,
        "geometry_seed_rows": len(x),
        "controlled_endpoint_rows": len(orbit),
        "input_feature_sha256": hashlib.sha256(
            np.ascontiguousarray(features).view(np.uint8)
        ).hexdigest(),
        "readout_weight_sha256": hashlib.sha256(
            np.ascontiguousarray(weight).view(np.uint8)
        ).hexdigest(),
        "readout_bias_sha256": hashlib.sha256(
            np.ascontiguousarray(readout_bias).view(np.uint8)
        ).hexdigest(),
        "retained_pca_energy_fraction": retained_energy,
        "unrestricted_equal_prior_bayes_prediction_upper": unrestricted_bayes_upper,
        "achievable_group_map_randomized_minimax": group_map_minimax,
        "group_map_mixture": mixture.tolist(),
        "group_map_world_accuracies": achieved_world_accuracies.tolist(),
        "uniform_inverse_world_accuracies": uniform_world_accuracies.tolist(),
        "fixed_identity_worst_world_accuracy": float(fixed_identity_accuracies.min()),
        "candidate_inverse_agreement_matrix": agreement.tolist(),
        "finite_multiset_hashes": sorted_hashes,
        "finite_multiset_identical": len(set(sorted_hashes)) == 1,
        "endpoint_mean_residual": mean_residual,
        "endpoint_covariance_relative_residual": covariance_residual,
        "interpretation": (
            "This is a controlled PCA-subspace frozen-head geometry stress test whose reference "
            "law is the orbit completion of sampled real embeddings; it is not a natural "
            "corruption or an external-label ceiling. Each declared group inverse preserves that "
            "controlled endpoint marginal. The unrestricted Bayes prediction upper and the "
            "achievable randomized group-map minimax are reported separately; neither is assumed "
            "to be 1/K for a generic head."
        ),
    }


def write_controlled_results(output_dir: str | Path, pair_rows: list[dict[str, Any]],
                             orbit_records: list[dict[str, Any]]) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "pair_rank_phase.jsonl"
    temporary_rows = rows_path.with_suffix(rows_path.suffix + ".tmp")
    temporary_rows.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    temporary_rows.replace(rows_path)
    orbit_path = output / "real_embedding_orbits.json"
    orbit_gates = real_orbit_scientific_gates(orbit_records)
    write_json_atomic(orbit_path, {
        "schema_version": "marginal_alignment.campaign.real_orbit_collection.v2",
        "status": "PASS" if all(orbit_gates.values()) else "FAIL",
        "scientific_gates": orbit_gates,
        "records": orbit_records,
    })
    manifest = output / "manifest.json"
    gates = controlled_scientific_gates(pair_rows, orbit_records)
    write_json_atomic(manifest, {
        "schema_version": "marginal_alignment.campaign.controlled_manifest.v2",
        "status": "PASS" if gates["all_passed"] else "FAIL",
        "scientific_gates": gates,
        "pair_rows": len(pair_rows),
        "pair_rows_sha256": sha256_file(rows_path),
        "orbit_records": len(orbit_records),
        "orbit_sha256": sha256_file(orbit_path),
    })
    return [rows_path, orbit_path, manifest]
