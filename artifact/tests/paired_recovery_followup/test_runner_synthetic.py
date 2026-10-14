from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from marginal_alignment.paired_recovery_bounds.geometry import fit_recovery_geometry
from marginal_alignment.paired_recovery_bounds.splitting import (
    RecoveryBatch,
    RecoveryPartition,
)
from marginal_alignment.paired_recovery_followup.calibration import (
    calibrate_direct_recovery_bounds,
)
from marginal_alignment.paired_recovery_followup.runner import (
    PreparedContext,
    _build_unit,
)


class _Endpoint:
    def __init__(self, features: np.ndarray) -> None:
        self.features = features


class _Store:
    def __init__(self, target: np.ndarray, reference: np.ndarray) -> None:
        self._target = _Endpoint(target)
        self._reference = _Endpoint(reference)

    def endpoint(self, role: str) -> _Endpoint:
        return self._target if role == "target" else self._reference


def test_complete_unit_keeps_rewards_physical_across_both_geometries() -> None:
    rng = np.random.default_rng(20260914)
    reference_fit = rng.normal(size=(80, 4))
    target_fit = reference_fit @ np.diag([1.15, 0.85, 1.1, 0.9]) + 0.1
    fitted = fit_recovery_geometry(
        target_fit,
        reference_fit,
        rng.normal(size=(5, 4)),
        rng.normal(size=5),
    )
    underlying = rng.normal(size=(20, 4))
    reference_order = rng.permutation(20)
    target_order = rng.permutation(20)
    reference_position = np.empty(20, dtype=np.int64)
    reference_position[reference_order] = np.arange(20, dtype=np.int64)
    true_assignment = reference_position[target_order]
    reference = underlying[reference_order]
    target = (underlying @ np.diag([1.15, 0.85, 1.1, 0.9]) + 0.1)[target_order]
    batch = RecoveryBatch(
        reference_rows=np.arange(20, dtype=np.int64),
        target_rows=np.arange(20, dtype=np.int64),
        true_assignment=true_assignment,
        reference_ids=tuple(f"r{value}" for value in reference_order),
        target_ids=tuple(f"t{value}" for value in target_order),
    )
    partition = RecoveryPartition(
        map_fit=batch,
        calibration=(),
        evaluation=(batch,),
        seed=2026091401,
        eligible_pair_count=20,
        unused_pair_count=0,
        pairing_sha256="synthetic",
        head_fit_ids_sha256="synthetic",
    )
    fitting_errors = {
        "identity": np.linspace(0.35, 0.50, 16),
        "diagonal": np.linspace(0.25, 0.40, 16),
        "bures": np.linspace(0.24, 0.39, 16),
    }
    calibration_errors = [
        {
            "identity": 0.40 + index / 1000,
            "diagonal": 0.30 + index / 1000,
            "bures": 0.29 + index / 1000,
        }
        for index in range(29)
    ]
    direct = calibrate_direct_recovery_bounds(
        fitting_errors, calibration_errors, delta=0.1
    )
    protocol = SimpleNamespace(
        config={
            "study_id": "synthetic_followup",
            "common_cost_geometry": {
                "name": "full_reference_oas_mahalanobis_after_bures"
            },
            "optimization": {
                "dual_bisection_iterations": 24,
                "exact_diagnostic": {
                    "evaluation_batch_index": 0,
                    "rows": 16,
                    "time_limit_seconds": 30.0,
                },
            },
        }
    )
    context = PreparedContext(
        protocol=protocol,
        cell={
            "encoder": "clip",
            "corruption": "gaussian_noise",
            "repetition": 1,
            "phase": "synthetic",
            "control_seed": 3026091401,
        },
        store=_Store(target, reference),
        partition=partition,
        fitted=fitted,
        physical_qhat=0.5,
        control_qhat=0.5,
        direct=direct,
        prepared_payload={"bindings": {"synthetic": True}},
    )
    unit = _build_unit(context, 0)
    assert unit["status"] == "PASS"
    assert unit["ground_truth_class_labels_used"] is False
    assert unit["shared_feasible_sets"]["column_permuted"][
        "physical_rewards_and_true_assignment_reindexed"
    ] is False
    for contrast in (
        "diagonal_minus_identity",
        "bures_minus_identity",
        "bures_minus_diagonal",
    ):
        assert (
            unit["physical_geometry"]["contrasts"][contrast]["reward_matrix_sha256"]
            == unit["column_permuted_geometry"]["contrasts"][contrast][
                "reward_matrix_sha256"
            ]
        )
    exact = unit["exact_diagnostic"]
    assert exact["status"] == "PASS"
    assert sum(
        len(exact["geometries"][geometry][kind])
        for geometry in ("physical", "column_permuted")
        for kind in ("methods", "contrasts")
    ) == 12
