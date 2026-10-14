from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError, seal_manifest
from marginal_alignment.iclr_extension.cost_scale import summarize_cost_scale
from marginal_alignment.iclr_extension.pair_structure import (
    PairStructureProtocol,
    build_pair_structure_selection,
    build_pair_structure_split,
    random_cost_moments_from_cost_matrix,
    random_gap_budget_table,
    random_gap_fraction_to_relative_budget,
    relative_budget_for_random_gap_share,
    relative_budget_to_random_gap_fraction,
    validate_pair_structure_manifest,
    verify_pair_structure_selection,
)


def _inputs(rows: int = 32) -> dict:
    return {
        "reference_ids": np.asarray([f"clean-{index:03d}" for index in range(rows)]),
        "target_ids": np.asarray([f"corrupt-{index:03d}" for index in range(rows)]),
        "paired_reference_ids": np.asarray(
            [f"clean-{index:03d}" for index in range(rows)]
        ),
        "paired_target_ids": np.asarray(
            [f"corrupt-{index:03d}" for index in range(rows)]
        ),
        "head_fit_reference_ids": np.asarray(
            [f"clean-{index:03d}" for index in range(4)]
        ),
    }


def _selection(seed: int = 201):
    return build_pair_structure_selection(
        **_inputs(),
        seed=seed,
        selection_salt="unit-test-control",
        protocol=PairStructureProtocol(calibration_rows=4, witness_rows=3),
    )


def test_selection_builds_paired_and_exactly_disjoint_unpaired_arms() -> None:
    selected = _selection()
    paired = selected.paired
    unpaired = selected.disjoint_unpaired

    assert paired.reference_ids == unpaired.reference_ids
    assert np.array_equal(paired.reference_rows, unpaired.reference_rows)
    assert (
        paired.true_pair_overlap_all_rows,
        paired.true_pair_overlap_calibration,
        paired.true_pair_overlap_witness,
    ) == (7, 4, 3)
    assert (
        unpaired.true_pair_overlap_all_rows,
        unpaired.true_pair_overlap_calibration,
        unpaired.true_pair_overlap_witness,
    ) == (0, 0, 0)
    assert set(paired.target_ids).isdisjoint(set(unpaired.target_ids))
    assert not set(_inputs()["head_fit_reference_ids"]).intersection(
        paired.reference_ids
    )

    target_to_reference = dict(
        zip(_inputs()["paired_target_ids"], _inputs()["paired_reference_ids"])
    )
    for target_position, reference_position in enumerate(
        paired.witness_true_pair_assignment
    ):
        target_id = paired.target_witness_ids[target_position]
        reference_id = paired.reference_witness_ids[int(reference_position)]
        assert target_to_reference[target_id] == reference_id

    assert paired.reference_rows.flags.writeable is False
    assert unpaired.target_rows.flags.writeable is False
    validate_pair_structure_manifest(selected.manifest)


def test_selection_is_deterministic_and_pair_table_order_invariant() -> None:
    inputs = _inputs()
    first = _selection()
    second = _selection()
    assert dict(first.manifest) == dict(second.manifest)
    assert np.array_equal(first.paired.reference_rows, second.paired.reference_rows)
    assert np.array_equal(
        first.disjoint_unpaired.target_rows,
        second.disjoint_unpaired.target_rows,
    )

    permutation = np.asarray(list(reversed(range(len(inputs["paired_reference_ids"])))))
    inputs["paired_reference_ids"] = inputs["paired_reference_ids"][permutation]
    inputs["paired_target_ids"] = inputs["paired_target_ids"][permutation]
    reordered = build_pair_structure_selection(
        **inputs,
        seed=201,
        selection_salt="unit-test-control",
        protocol=PairStructureProtocol(calibration_rows=4, witness_rows=3),
    )
    assert dict(first.manifest) == dict(reordered.manifest)
    assert first.paired.reference_ids == reordered.paired.reference_ids
    assert first.disjoint_unpaired.target_ids == reordered.disjoint_unpaired.target_ids


def test_selection_changes_with_seed_and_reconstructs_exactly() -> None:
    first = _selection(201)
    second = _selection(202)
    assert (
        first.manifest["block_a_pairs_sha256"]
        != second.manifest["block_a_pairs_sha256"]
    )
    verify_pair_structure_selection(first, **_inputs())


def test_selection_fails_closed_on_invalid_id_inputs() -> None:
    inputs = _inputs(rows=16)
    inputs["paired_target_ids"] = np.asarray(
        [*inputs["paired_target_ids"][:-1], "absent-target"]
    )
    with pytest.raises(IntegrityError, match="outside the endpoint"):
        build_pair_structure_selection(
            **inputs,
            seed=201,
            selection_salt="unit-test-control",
            protocol=PairStructureProtocol(calibration_rows=3, witness_rows=2),
        )

    insufficient = _inputs(rows=12)
    with pytest.raises(IntegrityError, match="needs 14 eligible identities"):
        build_pair_structure_selection(
            **insufficient,
            seed=201,
            selection_salt="unit-test-control",
            protocol=PairStructureProtocol(calibration_rows=4, witness_rows=3),
        )

    mixed = _inputs(rows=16)
    mixed["reference_ids"] = [
        *mixed["reference_ids"][:-1].tolist(),
        15,
    ]
    with pytest.raises(IntegrityError, match="mixes integer and string IDs"):
        build_pair_structure_selection(
            **mixed,
            seed=201,
            selection_salt="unit-test-control",
            protocol=PairStructureProtocol(calibration_rows=3, witness_rows=2),
        )


def test_selection_manifest_rejects_nested_tampering() -> None:
    manifest = copy.deepcopy(dict(_selection().manifest))
    manifest["arms"]["disjoint_unpaired"]["true_pair_overlap_witness"] = 1
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        validate_pair_structure_manifest(manifest)

    resealed = copy.deepcopy(dict(_selection().manifest))
    resealed.pop("manifest_sha256")
    resealed["arms"]["paired"]["endpoint_order_policy"] = "row_order"
    with pytest.raises(IntegrityError, match="scope drifted"):
        validate_pair_structure_manifest(seal_manifest(resealed))

    extra = copy.deepcopy(dict(_selection().manifest))
    extra.pop("manifest_sha256")
    extra["undeclared"] = True
    with pytest.raises(IntegrityError, match="field inventory drifted"):
        validate_pair_structure_manifest(seal_manifest(extra))


class _Pairing:
    def __init__(self, reference_ids: np.ndarray, target_ids: np.ndarray) -> None:
        self.reference_ids = reference_ids
        self.target_ids = target_ids

    def validated(self):
        return self.reference_ids.tolist(), self.target_ids.tolist()


class _Store:
    def __init__(self, *, corrupt_labels: bool = False) -> None:
        rows = 20
        reference_ids = np.asarray([f"clean-{index:03d}" for index in range(rows)])
        target_ids = np.asarray([f"corrupt-{index:03d}" for index in range(rows)])
        reference_labels = np.arange(rows, dtype=np.int64) % 5
        target_labels = np.array(reference_labels, copy=True)
        if corrupt_labels:
            target_labels = (target_labels + 1) % 5
        self.dataset = "imagenet_c"
        self.encoder = "clip"
        self.hashes_verified = True
        self.manifest = {
            "schema_version": "fixture",
            "metadata": {"model_provenance": {"head_fit_ids_role": "head_fit_ids"}},
        }
        self._reference = SimpleNamespace(
            features=np.arange(rows * 4, dtype=np.float32).reshape(rows, 4),
            ids=reference_ids,
            labels=reference_labels,
        )
        self._target = SimpleNamespace(
            features=(1000 + np.arange(rows * 4, dtype=np.float32)).reshape(rows, 4),
            ids=target_ids,
            labels=target_labels,
        )
        self._pairing = _Pairing(reference_ids, target_ids)
        self._head_fit = reference_ids[:2]

    def endpoint(self, name: str):
        return self._reference if name == "reference" else self._target

    def pairing(self):
        return self._pairing

    def array(self, role: str):
        assert role == "head_fit_ids"
        return self._head_fit


@pytest.mark.parametrize(
    ("arm", "expected_overlap"),
    [("paired", 5), ("disjoint_unpaired", 0)],
)
def test_runner_facing_split_materializes_features_ids_and_labels(
    arm: str, expected_overlap: int
) -> None:
    split = build_pair_structure_split(
        _Store(),
        arm,
        seed=201,
        calibration_rows=3,
        witness_rows=2,
        salt_prefix="unit-test-materialized",
        validate_pair_labels=True,
    )
    assert split.features_reference.shape == (5, 4)
    assert split.features_target.shape == (5, 4)
    assert split.calibration_reference_features.shape == (3, 4)
    assert split.witness_target_features.shape == (2, 4)
    assert split.ids_reference.shape == (5,)
    assert split.labels_target.shape == (5,)
    assert split.features_reference.dtype == np.float64
    assert split.features_reference.flags.writeable is False
    assert split.diagnostics["true_pair_overlap_all_rows"] == expected_overlap
    assert (
        split.diagnostics["pairing_available_to_geometry_cost_reward_or_optimizer"]
        is False
    )
    assert split.diagnostics["labels_used_for_administrative_pair_validation"] is True


def test_runner_facing_split_rejects_invalid_arm_and_pair_labels() -> None:
    with pytest.raises(ValueError, match="arm"):
        build_pair_structure_split(
            _Store(), "unknown", 201, 3, 2, "unit-test-materialized"
        )
    with pytest.raises(IntegrityError, match="inconsistent endpoint labels"):
        build_pair_structure_split(
            _Store(corrupt_labels=True),
            "paired",
            201,
            3,
            2,
            "unit-test-materialized",
            validate_pair_labels=True,
        )


def test_cost_matrix_random_moments_match_exhaustive_bijections() -> None:
    cost = np.asarray(
        [
            [0.0, 1.5, 2.0, 4.0],
            [2.5, 0.25, 1.0, 3.0],
            [1.0, 2.0, 0.5, 2.75],
            [3.0, 0.75, 2.25, 1.0],
        ]
    )
    totals = np.asarray(
        [
            sum(cost[row, column] for row, column in enumerate(permutation))
            for permutation in itertools.permutations(range(len(cost)))
        ]
    )
    result = random_cost_moments_from_cost_matrix(cost)
    assert result["expected_total_cost"] == pytest.approx(float(totals.mean()))
    assert result["variance_total_cost"] == pytest.approx(float(totals.var()))
    assert result["standard_deviation_total_cost"] == pytest.approx(float(totals.std()))
    scale = summarize_cost_scale(
        minimum_cost=1.0,
        random_moments=result,
        budgets=(0.01, 0.05, 0.10),
    )
    assert scale["random_bijection"]["expected_total_cost"] == pytest.approx(
        float(totals.mean())
    )


def test_random_gap_budget_conversions_are_exact_inverses() -> None:
    relative = relative_budget_for_random_gap_share(100.0, 500.0, 0.025)
    assert relative == pytest.approx(0.10)
    assert random_gap_fraction_to_relative_budget(
        0.025, minimum_cost=100.0, expected_random_cost=500.0
    ) == pytest.approx(relative)
    assert relative_budget_to_random_gap_fraction(
        relative, minimum_cost=100.0, expected_random_cost=500.0
    ) == pytest.approx(0.025)

    table = random_gap_budget_table(
        minimum_cost=100.0,
        expected_random_cost=500.0,
        relative_budgets=(0.01, 0.05, 0.10),
    )
    assert [
        row["allowed_excess_fraction_of_random_minus_minimum"]
        for row in table["budgets"]
    ] == pytest.approx([0.0025, 0.0125, 0.025])


@pytest.mark.parametrize(
    "cost",
    [np.ones((2, 3)), np.asarray([[0.0, np.nan], [1.0, 0.0]]), -np.eye(2)],
)
def test_cost_matrix_moments_reject_invalid_inputs(cost: np.ndarray) -> None:
    with pytest.raises(ValueError, match="cost_matrix"):
        random_cost_moments_from_cost_matrix(cost)
