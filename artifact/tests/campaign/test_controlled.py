import json
import tempfile
from pathlib import Path

import numpy as np

from marginal_alignment.campaign.controlled import (
    controlled_scientific_gates,
    pair_phase_scientific_gates,
    real_orbit_scientific_gates,
    run_pair_rank_phase_diagram,
    run_real_embedding_orbit,
    write_controlled_results,
)


def test_pair_phase_recovers_after_full_centered_rank():
    rows = run_pair_rank_phase_diagram(
        seeds=(0,), dimensions=(8,), classes=(2,), pair_ratios=(0.5, 2.0),
        pair_noise=(0.0, 1.0), evaluation_size=256,
    )
    full = [row for row in rows if row["method"] == "paired_procrustes"
            and row["pairs_per_dimension"] == 2.0 and row["pair_noise_rate"] == 0.0][0]
    shuffled = [row for row in rows if row["method"] == "paired_procrustes"
                and row["pairs_per_dimension"] == 2.0 and row["pair_noise_rate"] == 1.0][0]
    assert full["prediction_agreement"] > 0.99
    assert full["paired_feature_mse"] < shuffled["paired_feature_mse"]
    budgets = {row["pair_budget"] for row in rows}
    assert {7, 8, 9, 16}.issubset(budgets)
    at_d = next(row for row in rows if row["pair_budget"] == 8)
    after_d = next(row for row in rows if row["pair_budget"] == 9)
    assert at_d["centered_pair_rank"] == 7
    assert after_d["centered_pair_rank"] == 8
    assert after_d["identified_centered_readout_energy"] > 1.0 - 1e-9
    assert all(pair_phase_scientific_gates(rows).values())


def test_real_orbit_computes_not_assumes_ceiling():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(300, 6))
    weight = rng.normal(size=(4, 6))
    bias = rng.normal(size=4)
    result = run_real_embedding_orbit(
        features, weight, bias=bias, seed=3, rank=4, worlds=4,
        evaluation_rows=128,
    )
    assert result["finite_multiset_identical"]
    assert result["endpoint_mean_residual"] < 1e-10
    assert result["endpoint_covariance_relative_residual"] < 1e-10
    assert (
        result["achievable_group_map_randomized_minimax"]
        <= result["unrestricted_equal_prior_bayes_prediction_upper"] + 1e-10
    )
    assert np.isclose(
        min(result["group_map_world_accuracies"]),
        result["achievable_group_map_randomized_minimax"],
    )
    assert all(real_orbit_scientific_gates([result]).values())
    assert len(result["readout_bias_sha256"]) == 64
    assert "not a natural corruption" in result["interpretation"]


def test_controlled_manifest_pass_is_earned():
    rows = run_pair_rank_phase_diagram(
        seeds=(0,), dimensions=(8,), classes=(2,), pair_ratios=(0.5, 2.0),
        pair_noise=(0.0, 1.0), evaluation_size=256,
    )
    rng = np.random.default_rng(7)
    orbit = run_real_embedding_orbit(
        rng.normal(size=(300, 6)), rng.normal(size=(4, 6)), seed=9,
        rank=4, worlds=4, evaluation_rows=128,
    )
    gates = controlled_scientific_gates(rows, [orbit])
    assert gates["all_passed"]
    with tempfile.TemporaryDirectory() as temporary:
        outputs = write_controlled_results(temporary, rows, [orbit])
        assert all(path.is_file() for path in outputs)
        manifest = json.loads(
            (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "PASS"
        assert manifest["scientific_gates"]["all_passed"]
