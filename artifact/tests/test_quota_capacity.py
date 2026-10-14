from __future__ import annotations

import numpy as np
import pytest

from marginal_alignment.quota_capacity import (
    QuotaCapacityError,
    finite_quota_minimax,
    prior_average_capacities,
    three_world_quota_control,
)
from scripts.iclr_extension.run_quota_capacity_control import build_control


def test_three_world_control_has_strict_output_mass_gap_and_exact_dual() -> None:
    result = three_world_quota_control()

    assert result["unrestricted_prior_average_value"] == pytest.approx(3.0 / 4.0)
    assert result["quota_prior_average_value"] == pytest.approx(2.0 / 3.0)
    assert result["quota_penalty"] == pytest.approx(1.0 / 12.0)
    assert result["primal_value"] == pytest.approx(2.0 / 3.0)
    assert result["dual_value"] == pytest.approx(2.0 / 3.0)
    assert result["absolute_primal_dual_gap"] <= 1e-10
    assert np.allclose(result["least_favorable_prior"], np.full(3, 1.0 / 3.0))

    actions = result["actions"]
    assert actions.shape == (6, 4)
    assert all(np.bincount(action, minlength=2).tolist() == [2, 2] for action in actions)


def test_quota_control_is_invariant_to_joint_atom_permutation() -> None:
    desired = np.asarray(
        [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=np.int64
    )
    permutation = np.asarray([3, 1, 0, 2])

    original = prior_average_capacities(desired, (2, 2))
    permuted = prior_average_capacities(desired[:, permutation], (2, 2))
    original_game = finite_quota_minimax(desired, (2, 2))
    permuted_game = finite_quota_minimax(desired[:, permutation], (2, 2))

    assert permuted["unrestricted_prior_average_value"] == pytest.approx(
        original["unrestricted_prior_average_value"]
    )
    assert permuted["quota_prior_average_value"] == pytest.approx(
        original["quota_prior_average_value"]
    )
    assert permuted_game["primal_value"] == pytest.approx(original_game["primal_value"])
    assert permuted_game["dual_value"] == pytest.approx(original_game["dual_value"])


def test_quota_control_rejects_world_that_changes_reference_label_mass() -> None:
    invalid = np.asarray([[0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.int64)

    with pytest.raises(QuotaCapacityError, match="preserve the reference label counts"):
        prior_average_capacities(invalid, (2, 2))


def test_serialized_control_is_sealed_and_scoped() -> None:
    control = build_control()

    assert control["status"] == "PASS"
    assert control["scope"] == "finite_indexed_atom_analogue_not_population_transport_evidence"
    assert control["exact_values"]["strict_quota_penalty"] == "1/12"
    assert len(control["manifest_sha256"]) == 64
