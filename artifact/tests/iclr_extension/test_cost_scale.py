from __future__ import annotations

import copy
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, seal_manifest
from marginal_alignment.iclr_extension.cost_scale import (
    assignment_cost,
    summarize_cost_scale,
    uniform_random_bijection_cost_moments,
)
from scripts.iclr_extension.audit_correspondence_cost_scale import (
    _expected_unit_specifications,
    _sealed_unit_paths,
    _summaries,
    _validated_output_path,
    _verify_historical_source,
    verify_cost_scale_result,
)


def test_random_bijection_moments_equal_exhaustive_enumeration() -> None:
    target = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [2.0, -1.0, 0.5],
            [1.5, 0.25, -2.0],
            [-0.5, 1.25, 1.0],
        ]
    )
    reference = np.asarray(
        [
            [0.5, -0.25, 1.0],
            [-1.0, 0.0, 2.5],
            [2.0, 1.5, -0.5],
            [1.0, -2.0, 0.0],
        ]
    )
    observed = np.asarray(
        [
            assignment_cost(target, reference, np.asarray(permutation))
            for permutation in itertools.permutations(range(len(target)))
        ]
    )
    exact = uniform_random_bijection_cost_moments(target, reference)
    assert exact["expected_total_cost"] == pytest.approx(float(observed.mean()))
    assert exact["variance_total_cost"] == pytest.approx(float(observed.var()))
    assert exact["standard_deviation_total_cost"] == pytest.approx(
        float(observed.std())
    )


def test_cost_scale_uses_random_minus_minimum_gap() -> None:
    result = summarize_cost_scale(
        minimum_cost=100.0,
        random_moments={
            "expected_total_cost": 500.0,
            "variance_total_cost": 25.0,
            "standard_deviation_total_cost": 5.0,
            "coefficient_of_variation": 0.01,
        },
        budgets=(0.01, 0.05, 0.10),
        identity_cost=110.0,
    )
    assert result["random_bijection"]["expected_to_minimum_ratio"] == 5.0
    assert (
        result["identity_pairing"]["excess_fraction_of_random_minus_minimum"] == 0.025
    )
    assert [
        row["allowed_excess_fraction_of_random_minus_minimum"]
        for row in result["budgets"]
    ] == pytest.approx([0.0025, 0.0125, 0.025])


def test_assignment_must_be_a_bijection() -> None:
    endpoint = np.eye(3)
    with pytest.raises(ValueError, match="permutation"):
        assignment_cost(endpoint, endpoint, np.asarray([0, 0, 2]))


def test_sealed_inventory_is_exact_and_hash_bound(tmp_path: Path) -> None:
    results = tmp_path / "completed"
    (results / "natural").mkdir(parents=True)
    (results / "positive_control").mkdir()
    artifacts = []
    expected = _expected_unit_specifications()
    for unit_id, specification in expected.items():
        payload = f"sealed:{unit_id}\n".encode()
        path = results / specification["panel"] / f"{unit_id}.json"
        path.write_bytes(payload)
        artifacts.append(
            {
                "unit_id": unit_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker = {"unit_artifacts": artifacts}
    assert len(_sealed_unit_paths(results, marker)) == 126

    first = (
        results
        / expected[artifacts[0]["unit_id"]]["panel"]
        / (artifacts[0]["unit_id"] + ".json")
    )
    first.write_bytes(b"changed\n")
    with pytest.raises(IntegrityError, match="bytes changed"):
        _sealed_unit_paths(results, marker)


def test_output_cannot_resolve_inside_historical_results(tmp_path: Path) -> None:
    results = (tmp_path / "completed").resolve()
    results.mkdir()
    with pytest.raises(IntegrityError, match="outside"):
        _validated_output_path(results / "audit.json", results)

    redirect = tmp_path / "redirect"
    redirect.symlink_to(results, target_is_directory=True)
    with pytest.raises(IntegrityError, match="outside"):
        _validated_output_path(redirect / "audit.json", results)


def test_historical_source_rejects_in_root_symlink(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "real.py"
    target.write_text("pass\n", encoding="utf-8")
    (project / "link.py").symlink_to(target)
    with pytest.raises(IntegrityError, match="symbolic"):
        _verify_historical_source(project.resolve(), {"link.py": "0" * 64})


def _sealed_cost_result_fixture() -> dict:
    grouped = {}
    for unit_id, specification in _expected_unit_specifications().items():
        identity = (
            specification["panel"],
            specification["dataset"],
            specification["encoder"],
            specification["corruption"],
            specification["severity"],
            specification["split_seed"],
        )
        grouped.setdefault(identity, []).append((unit_id, specification))
    conditions = []
    for identity, units in sorted(grouped.items(), key=lambda item: str(item[0])):
        identity_cost = None if identity[0] == "natural" else 3.0
        scale = summarize_cost_scale(
            minimum_cost=2.0,
            random_moments={
                "expected_total_cost": 10.0,
                "variance_total_cost": 4.0,
                "standard_deviation_total_cost": 2.0,
                "coefficient_of_variation": 0.2,
            },
            budgets=(0.01, 0.05, 0.10),
            identity_cost=identity_cost,
        )
        by_budget = {
            specification["cost_budget_relative"]: (unit_id, specification)
            for unit_id, specification in units
        }
        budgets = []
        for row in scale["budgets"]:
            unit_id, specification = by_budget[row["cost_budget_relative"]]
            budgets.append(
                {
                    **row,
                    "feasible_disagreement": 0.25,
                    "two_world_recovery_ceiling": 0.875,
                    "unit_result": str(
                        Path(specification["panel"]) / f"{unit_id}.json"
                    ),
                    "unit_result_sha256": "c" * 64,
                }
            )
        conditions.append(
            {
                "panel": identity[0],
                "dataset": identity[1],
                "encoder": identity[2],
                "corruption": identity[3],
                "severity": identity[4],
                "split_seed": identity[5],
                "calibration_rows": 4096,
                "witness_rows": 1024,
                "feature_dimension": 4,
                "store_manifest": "store/FEATURE_STORE_MANIFEST.json",
                "store_manifest_sha256": "a" * 64,
                "fitted_geometry_sha256": "b" * 64,
                "minimum_cost": 2.0,
                "random_bijection": scale["random_bijection"],
                "identity_pairing": scale["identity_pairing"],
                "budgets": budgets,
            }
        )
    return seal_manifest(
        {
            "schema_version": "marginal_alignment.iclr_extension.cost_scale_audit.v1",
            "status": "PASS",
            "scientific_result": True,
            "analysis_timing": "post_hoc_after_inspection_of_correspondence_outcomes",
            "historical_source_files": 26,
            "historical_source_identity_verified": True,
            "condition_geometries": 42,
            "conditions": conditions,
            "summaries": _summaries(conditions),
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["conditions"][17].__setitem__("condition", 999),
        lambda value: value["summaries"][1][
            "random_expected_to_minimum_ratio"
        ].__setitem__("mean", 3.0),
    ],
)
def test_cost_result_seal_rejects_nested_tampering(mutation) -> None:
    value = _sealed_cost_result_fixture()
    verify_cost_scale_result(value)
    changed = copy.deepcopy(value)
    mutation(changed)
    with pytest.raises(IntegrityError, match="manifest checksum mismatch"):
        verify_cost_scale_result(changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["conditions"][17].__setitem__("split_seed", 999),
            "condition grid",
        ),
        (
            lambda value: value["summaries"][1].__setitem__("condition_geometries", 35),
            "summaries differ",
        ),
    ],
)
def test_cost_result_semantics_reject_resealed_nested_tampering(
    mutation, message
) -> None:
    changed = _sealed_cost_result_fixture()
    changed.pop("manifest_sha256")
    mutation(changed)
    resealed = seal_manifest(changed)
    with pytest.raises(IntegrityError, match=message):
        verify_cost_scale_result(resealed)
