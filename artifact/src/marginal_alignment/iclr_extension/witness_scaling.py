"""Nested endpoint administration for the witness-size scaling follow-up.

The scaling study freezes one maximum-size selection and derives every smaller
endpoint as a literal prefix of its witness segment.  Calibration rows never
move.  For ImageNet-C, the maximum selection is built once from two disjoint
identity blocks, so the clean reference identities and corrupted target
identities remain disjoint at every reported size.

This module receives identifiers and features only for pre-outcome selection.
Class labels may be summarized later with :func:`class_coverage`, after the
certificate has been frozen; they are never inputs to geometry or assignment
optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    seal_manifest,
    sha256_json,
    verify_manifest_checksum,
)
from marginal_alignment.iclr_extension.optimization_certificate import (
    INFORMATION_SCOPE,
    OptimizationCertificateResult,
    PreparedCertificateProblem,
    OptimizationCertificateProtocol,
    _array_sha256,
    _linear_assignment,
    _unique_canonical_rows,
    prepare_certificate_problem,
    solve_exact_budgeted_disagreement_milp,
)
from marginal_alignment.iclr_extension.pair_structure import (
    PairStructureProtocol,
    build_pair_structure_selection,
)


NESTED_SELECTION_SCHEMA = (
    "marginal_alignment.iclr_extension.nested_witness_selection.v1"
)
RANDOM_MOMENTS_SCHEMA = (
    "marginal_alignment.iclr_extension.blockwise_random_cost_moments.v1"
)


def _readonly_int(values: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


def _readonly_ids(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.hasobject:
        raise IntegrityError("endpoint IDs must form a non-object rank-one array")
    result.setflags(write=False)
    return result


def _array_record(values: np.ndarray, *, namespace: str) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": sha256_json(
            {
                "namespace": namespace,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "values": array.tolist(),
            }
        ),
    }


@dataclass(frozen=True)
class WitnessSizeProtocol:
    """Locked row counts for one nested scaling family."""

    calibration_rows: int = 3840
    witness_sizes: tuple[int, ...] = (1024, 4096, 8192)

    def __post_init__(self) -> None:
        if (
            isinstance(self.calibration_rows, bool)
            or not isinstance(self.calibration_rows, (int, np.integer))
            or int(self.calibration_rows) < 2
        ):
            raise ValueError("calibration_rows must be an integer of at least two")
        sizes = tuple(self.witness_sizes)
        if (
            not sizes
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 2
                for value in sizes
            )
            or tuple(sorted(set(map(int, sizes)))) != tuple(map(int, sizes))
        ):
            raise ValueError(
                "witness_sizes must be unique, strictly increasing integers"
            )

    @property
    def maximum_witness_rows(self) -> int:
        return int(self.witness_sizes[-1])

    @property
    def maximum_block_rows(self) -> int:
        return int(self.calibration_rows) + self.maximum_witness_rows

    @property
    def required_disjoint_identity_rows(self) -> int:
        return 2 * self.maximum_block_rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_rows": int(self.calibration_rows),
            "witness_sizes": list(map(int, self.witness_sizes)),
            "maximum_witness_rows": self.maximum_witness_rows,
            "maximum_block_rows": self.maximum_block_rows,
            "required_disjoint_identity_rows": (self.required_disjoint_identity_rows),
        }


@dataclass(frozen=True)
class EndpointPrefix:
    """One materialized calibration-plus-witness prefix."""

    witness_rows: int
    reference_rows: np.ndarray
    target_rows: np.ndarray
    reference_ids: np.ndarray
    target_ids: np.ndarray

    @property
    def calibration_rows(self) -> int:
        return int(len(self.reference_rows) - self.witness_rows)

    @property
    def reference_calibration_rows(self) -> np.ndarray:
        return self.reference_rows[: self.calibration_rows]

    @property
    def target_calibration_rows(self) -> np.ndarray:
        return self.target_rows[: self.calibration_rows]

    @property
    def reference_witness_rows(self) -> np.ndarray:
        return self.reference_rows[self.calibration_rows :]

    @property
    def target_witness_rows(self) -> np.ndarray:
        return self.target_rows[self.calibration_rows :]


@dataclass(frozen=True)
class NestedEndpointSelection:
    """Maximum-size endpoint selection with immutable nested prefixes."""

    protocol: WitnessSizeProtocol
    reference_rows: np.ndarray
    target_rows: np.ndarray
    reference_ids: np.ndarray
    target_ids: np.ndarray
    manifest: Mapping[str, Any]

    def prefix(self, witness_rows: int) -> EndpointPrefix:
        requested = int(witness_rows)
        if requested not in set(map(int, self.protocol.witness_sizes)):
            raise ValueError(f"witness size is outside the lock: {requested}")
        calibration = int(self.protocol.calibration_rows)
        indices = np.concatenate(
            (
                np.arange(calibration, dtype=np.int64),
                calibration + np.arange(requested, dtype=np.int64),
            )
        )
        return EndpointPrefix(
            witness_rows=requested,
            reference_rows=_readonly_int(self.reference_rows[indices]),
            target_rows=_readonly_int(self.target_rows[indices]),
            reference_ids=_readonly_ids(self.reference_ids[indices]),
            target_ids=_readonly_ids(self.target_ids[indices]),
        )


def _selection_manifest(
    *,
    protocol: WitnessSizeProtocol,
    selection_kind: str,
    seed: int,
    selection_salt: str,
    reference_rows: np.ndarray,
    target_rows: np.ndarray,
    reference_ids: np.ndarray,
    target_ids: np.ndarray,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    maximum = NestedEndpointSelection(
        protocol=protocol,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=reference_ids,
        target_ids=target_ids,
        manifest={},
    )
    prefixes: dict[str, Any] = {}
    for size in protocol.witness_sizes:
        prefix = maximum.prefix(int(size))
        key = str(int(size))
        prefixes[key] = {
            "witness_rows": int(size),
            "reference_rows": _array_record(
                prefix.reference_rows,
                namespace=f"nested-witness:{selection_kind}:{seed}:{key}:reference-rows",
            ),
            "target_rows": _array_record(
                prefix.target_rows,
                namespace=f"nested-witness:{selection_kind}:{seed}:{key}:target-rows",
            ),
            "reference_ids": _array_record(
                prefix.reference_ids,
                namespace=f"nested-witness:{selection_kind}:{seed}:{key}:reference-ids",
            ),
            "target_ids": _array_record(
                prefix.target_ids,
                namespace=f"nested-witness:{selection_kind}:{seed}:{key}:target-ids",
            ),
        }
    return seal_manifest(
        {
            "schema_version": NESTED_SELECTION_SCHEMA,
            "status": "PASS",
            "scientific_result": False,
            "selection_kind": selection_kind,
            "selection_phase": "frozen_before_witness_size_outcomes",
            "seed": int(seed),
            "selection_salt": selection_salt,
            "protocol": protocol.as_dict(),
            "calibration_rows_identical_across_sizes": True,
            "witness_endpoints_are_literal_prefixes_of_one_maximum_selection": True,
            "labels_or_predictions_used_for_selection": False,
            "parent_selection": dict(parent),
            "prefixes": prefixes,
        }
    )


def validate_nested_selection_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on malformed or post-hoc-modified selection records."""

    if not isinstance(manifest, Mapping):
        raise IntegrityError("nested selection manifest must be a mapping")
    verify_manifest_checksum(manifest)
    expected = {
        "schema_version",
        "status",
        "scientific_result",
        "selection_kind",
        "selection_phase",
        "seed",
        "selection_salt",
        "protocol",
        "calibration_rows_identical_across_sizes",
        "witness_endpoints_are_literal_prefixes_of_one_maximum_selection",
        "labels_or_predictions_used_for_selection",
        "parent_selection",
        "prefixes",
        "manifest_sha256",
    }
    if set(manifest) != expected:
        raise IntegrityError("nested selection manifest field inventory changed")
    if (
        manifest.get("schema_version") != NESTED_SELECTION_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_result") is not False
        or manifest.get("selection_phase") != "frozen_before_witness_size_outcomes"
        or manifest.get("calibration_rows_identical_across_sizes") is not True
        or manifest.get(
            "witness_endpoints_are_literal_prefixes_of_one_maximum_selection"
        )
        is not True
        or manifest.get("labels_or_predictions_used_for_selection") is not False
    ):
        raise IntegrityError("nested selection status or leakage contract changed")
    raw_protocol = manifest.get("protocol")
    if not isinstance(raw_protocol, Mapping):
        raise IntegrityError("nested selection protocol is absent")
    try:
        protocol = WitnessSizeProtocol(
            calibration_rows=int(raw_protocol["calibration_rows"]),
            witness_sizes=tuple(map(int, raw_protocol["witness_sizes"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("nested selection protocol is invalid") from error
    if dict(raw_protocol) != protocol.as_dict():
        raise IntegrityError("nested selection protocol contains undeclared fields")
    prefixes = manifest.get("prefixes")
    expected_keys = {str(int(size)) for size in protocol.witness_sizes}
    if not isinstance(prefixes, Mapping) or set(prefixes) != expected_keys:
        raise IntegrityError("nested selection prefix inventory changed")
    for size in protocol.witness_sizes:
        record = prefixes[str(int(size))]
        if not isinstance(record, Mapping) or set(record) != {
            "witness_rows",
            "reference_rows",
            "target_rows",
            "reference_ids",
            "target_ids",
        }:
            raise IntegrityError("nested selection prefix record is malformed")
        if int(record.get("witness_rows", -1)) != int(size):
            raise IntegrityError("nested selection prefix size changed")
        expected_shape = [protocol.calibration_rows + int(size)]
        for field in (
            "reference_rows",
            "target_rows",
            "reference_ids",
            "target_ids",
        ):
            array_record = record.get(field)
            if (
                not isinstance(array_record, Mapping)
                or set(array_record) != {"shape", "dtype", "sha256"}
                or array_record.get("shape") != expected_shape
                or not isinstance(array_record.get("sha256"), str)
                or len(array_record["sha256"]) != 64
            ):
                raise IntegrityError("nested selection array record is malformed")


def validate_nested_selection(selection: NestedEndpointSelection) -> None:
    """Bind the live endpoint arrays to every hash in the sealed manifest."""

    validate_nested_selection_manifest(selection.manifest)
    manifest = selection.manifest
    if selection.protocol.as_dict() != dict(manifest["protocol"]):
        raise IntegrityError("live nested-selection protocol differs from manifest")
    maximum_rows = selection.protocol.maximum_block_rows
    arrays = {
        "reference_rows": np.asarray(selection.reference_rows),
        "target_rows": np.asarray(selection.target_rows),
        "reference_ids": np.asarray(selection.reference_ids),
        "target_ids": np.asarray(selection.target_ids),
    }
    if any(
        value.ndim != 1 or len(value) != maximum_rows or value.dtype.hasobject
        for value in arrays.values()
    ):
        raise IntegrityError("live nested-selection arrays are malformed")
    if (
        not np.issubdtype(arrays["reference_rows"].dtype, np.integer)
        or not np.issubdtype(arrays["target_rows"].dtype, np.integer)
        or np.any(arrays["reference_rows"] < 0)
        or np.any(arrays["target_rows"] < 0)
    ):
        raise IntegrityError("live nested-selection row indices are invalid")
    selection_kind = str(manifest["selection_kind"])
    seed = int(manifest["seed"])
    for size in selection.protocol.witness_sizes:
        key = str(int(size))
        prefix = selection.prefix(int(size))
        observed = {
            "witness_rows": int(size),
            "reference_rows": _array_record(
                prefix.reference_rows,
                namespace=(
                    f"nested-witness:{selection_kind}:{seed}:{key}:reference-rows"
                ),
            ),
            "target_rows": _array_record(
                prefix.target_rows,
                namespace=(f"nested-witness:{selection_kind}:{seed}:{key}:target-rows"),
            ),
            "reference_ids": _array_record(
                prefix.reference_ids,
                namespace=(
                    f"nested-witness:{selection_kind}:{seed}:{key}:reference-ids"
                ),
            ),
            "target_ids": _array_record(
                prefix.target_ids,
                namespace=(f"nested-witness:{selection_kind}:{seed}:{key}:target-ids"),
            ),
        }
        if observed != manifest["prefixes"][key]:
            raise IntegrityError(
                "live nested-selection arrays differ from sealed prefix hashes"
            )


def build_nested_pair_selection(
    *,
    reference_ids: Sequence[Any] | np.ndarray,
    target_ids: Sequence[Any] | np.ndarray,
    paired_reference_ids: Sequence[Any] | np.ndarray,
    paired_target_ids: Sequence[Any] | np.ndarray,
    head_fit_reference_ids: Sequence[Any] | np.ndarray,
    seed: int,
    selection_salt: str,
    protocol: WitnessSizeProtocol | None = None,
) -> NestedEndpointSelection:
    """Freeze one maximum ImageNet-C disjoint selection and its prefixes."""

    declared = WitnessSizeProtocol() if protocol is None else protocol
    maximum = build_pair_structure_selection(
        reference_ids=reference_ids,
        target_ids=target_ids,
        paired_reference_ids=paired_reference_ids,
        paired_target_ids=paired_target_ids,
        head_fit_reference_ids=head_fit_reference_ids,
        seed=seed,
        selection_salt=selection_salt,
        protocol=PairStructureProtocol(
            calibration_rows=declared.calibration_rows,
            witness_rows=declared.maximum_witness_rows,
        ),
    )
    arm = maximum.disjoint_unpaired
    reference_rows = _readonly_int(arm.reference_rows)
    target_rows = _readonly_int(arm.target_rows)
    reference_values = _readonly_ids(arm.reference_ids)
    target_values = _readonly_ids(arm.target_ids)
    manifest = _selection_manifest(
        protocol=declared,
        selection_kind="imagenet_c_identity_disjoint_unpaired",
        seed=int(seed),
        selection_salt=selection_salt,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=reference_values,
        target_ids=target_values,
        parent={
            "schema_version": maximum.manifest["schema_version"],
            "manifest_sha256": maximum.manifest["manifest_sha256"],
            "eligible_pair_count": int(maximum.eligible_pair_count),
            "required_disjoint_identity_rows": (
                declared.required_disjoint_identity_rows
            ),
            "arm": "disjoint_unpaired",
            "true_pair_overlap_all_rows": int(arm.true_pair_overlap_all_rows),
        },
    )
    validate_nested_selection_manifest(manifest)
    return NestedEndpointSelection(
        protocol=declared,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=reference_values,
        target_ids=target_values,
        manifest=manifest,
    )


def build_nested_natural_selection(
    *,
    reference_features: np.ndarray,
    target_features: np.ndarray,
    reference_ids: Sequence[Any] | np.ndarray,
    target_ids: Sequence[Any] | np.ndarray,
    dataset: str,
    seed: int,
    protocol: WitnessSizeProtocol | None = None,
) -> NestedEndpointSelection:
    """Freeze content-canonical maximum endpoints for a natural shift."""

    declared = WitnessSizeProtocol() if protocol is None else protocol
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a nonempty string")
    requested = declared.maximum_block_rows
    reference = np.asarray(reference_features)
    target = np.asarray(target_features)
    reference_id_array = _readonly_ids(reference_ids)
    target_id_array = _readonly_ids(target_ids)
    _, reference_order, reference_digest, _ = _unique_canonical_rows(
        reference,
        name="reference_store_features",
        salt=f"play3-v2:{dataset}:{int(seed)}:reference-split".encode("ascii"),
        sample_ids=reference_id_array,
    )
    _, target_order, target_digest, _ = _unique_canonical_rows(
        target,
        name="target_store_features",
        salt=f"play3-v2:{dataset}:{int(seed)}:target-split".encode("ascii"),
        sample_ids=target_id_array,
    )
    if requested > len(reference_order) or requested > len(target_order):
        raise IntegrityError(
            f"{dataset} cannot supply {requested} calibration-plus-witness rows"
        )
    reference_rows = _readonly_int(reference_order[:requested])
    target_rows = _readonly_int(target_order[:requested])
    selected_reference_ids = _readonly_ids(reference_id_array[reference_rows])
    selected_target_ids = _readonly_ids(target_id_array[target_rows])
    selection_salt = f"play3-v2:{dataset}:content-canonical-max-prefix-v1"
    manifest = _selection_manifest(
        protocol=declared,
        selection_kind=f"natural_unpaired:{dataset}",
        seed=int(seed),
        selection_salt=selection_salt,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=selected_reference_ids,
        target_ids=selected_target_ids,
        parent={
            "reference_parent_indexed_atom_sha256": reference_digest,
            "target_parent_indexed_atom_sha256": target_digest,
            "reference_endpoint_rows": int(len(reference)),
            "target_endpoint_rows": int(len(target)),
            "pairing_used_for_split_administration": False,
        },
    )
    validate_nested_selection_manifest(manifest)
    return NestedEndpointSelection(
        protocol=declared,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=selected_reference_ids,
        target_ids=selected_target_ids,
        manifest=manifest,
    )


def materialize_prefix(
    selection: NestedEndpointSelection,
    *,
    reference_features: np.ndarray,
    target_features: np.ndarray,
    witness_rows: int,
) -> dict[str, np.ndarray]:
    """Slice feature arrays only after the locked prefix is authenticated."""

    validate_nested_selection(selection)
    prefix = selection.prefix(witness_rows)
    reference = np.asarray(reference_features)
    target = np.asarray(target_features)
    if (
        reference.ndim != 2
        or target.ndim != 2
        or reference.shape[1] != target.shape[1]
        or np.max(prefix.reference_rows) >= len(reference)
        or np.max(prefix.target_rows) >= len(target)
    ):
        raise IntegrityError("locked prefix is incompatible with endpoint features")
    return {
        "reference_features": np.asarray(
            reference[prefix.reference_rows], dtype=np.float64
        ),
        "target_features": np.asarray(target[prefix.target_rows], dtype=np.float64),
        "reference_ids": np.asarray(prefix.reference_ids),
        "target_ids": np.asarray(prefix.target_ids),
        "reference_rows": np.asarray(prefix.reference_rows, dtype=np.int64),
        "target_rows": np.asarray(prefix.target_rows, dtype=np.int64),
    }


def uniform_random_cost_moments_blockwise(
    cost_matrix: np.ndarray, *, block_rows: int = 256
) -> dict[str, Any]:
    """Exact random-bijection moments without an additional dense copy.

    The identity

    ``||C - rowmean - colmean + grandmean||_F^2``

    is evaluated from row sums, column sums, and ``sum(C**2)``.  Peak scratch
    memory is therefore ``O(n + block_rows*n)`` rather than another ``O(n^2)``
    centered matrix, which matters for an 8192-by-8192 float64 cost matrix.
    """

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if (
        cost.ndim != 2
        or cost.shape[0] != cost.shape[1]
        or cost.shape[0] < 2
        or isinstance(block_rows, bool)
        or not isinstance(block_rows, (int, np.integer))
        or int(block_rows) < 1
    ):
        raise ValueError("cost_matrix must be square and block_rows positive")
    rows = int(cost.shape[0])
    row_sums = np.zeros(rows, dtype=np.float64)
    column_sums = np.zeros(rows, dtype=np.float64)
    sum_squares = 0.0
    for start in range(0, rows, int(block_rows)):
        block = cost[start : start + int(block_rows)]
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("cost_matrix must be finite and nonnegative")
        row_sums[start : start + len(block)] = np.sum(block, axis=1, dtype=np.float64)
        column_sums += np.sum(block, axis=0, dtype=np.float64)
        sum_squares += float(np.einsum("ij,ij->", block, block, dtype=np.float64))
    total = float(np.sum(row_sums, dtype=np.float64))
    centered_energy = (
        sum_squares
        - float(np.dot(row_sums, row_sums)) / rows
        - float(np.dot(column_sums, column_sums)) / rows
        + total * total / (rows * rows)
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * max(sum_squares, 1.0)
    if centered_energy < -tolerance:
        raise RuntimeError("blockwise centered-energy identity became negative")
    centered_energy = max(centered_energy, 0.0)
    expected = total / rows
    variance = centered_energy / (rows - 1)
    standard_deviation = math.sqrt(variance)
    return {
        "schema_version": RANDOM_MOMENTS_SCHEMA,
        "expected_total_cost": float(expected),
        "variance_total_cost": float(variance),
        "standard_deviation_total_cost": float(standard_deviation),
        "coefficient_of_variation": (
            float(standard_deviation / expected) if expected > 0.0 else 0.0
        ),
        "evaluation": {
            "method": "algebraic_double_centering_identity",
            "block_rows": int(block_rows),
            "additional_dense_n_by_n_copy_created": False,
        },
    }


def class_coverage(class_ids: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Summarize post-freeze endpoint class coverage without changing a fit."""

    values = np.asarray(class_ids)
    if values.ndim != 1 or len(values) < 1 or values.dtype.hasobject:
        raise ValueError("class_ids must be a nonempty non-object rank-one array")
    _, counts = np.unique(values, return_counts=True)
    quantiles = np.quantile(counts.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "rows": int(len(values)),
        "unique_classes": int(len(counts)),
        "singleton_classes": int(np.sum(counts == 1)),
        "classes_with_at_least_two_rows": int(np.sum(counts >= 2)),
        "rows_per_observed_class": {
            "minimum": int(np.min(counts)),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "maximum": int(np.max(counts)),
            "mean": float(np.mean(counts, dtype=np.float64)),
        },
    }


def cross_endpoint_class_coverage(
    target_class_ids: Sequence[Any] | np.ndarray,
    reference_class_ids: Sequence[Any] | np.ndarray,
) -> dict[str, Any]:
    """Post-freeze same-class partner coverage across two witness endpoints."""

    target = np.asarray(target_class_ids)
    reference = np.asarray(reference_class_ids)
    if (
        target.ndim != 1
        or reference.ndim != 1
        or len(target) != len(reference)
        or len(target) < 1
        or target.dtype.hasobject
        or reference.dtype.hasobject
    ):
        raise ValueError("class endpoints must be equal nonempty rank-one arrays")
    target_values, target_counts = np.unique(target, return_counts=True)
    reference_values, reference_counts = np.unique(reference, return_counts=True)
    reference_by_class = {
        value.item() if isinstance(value, np.generic) else value: int(count)
        for value, count in zip(reference_values, reference_counts)
    }
    partner_counts = np.asarray(
        [
            reference_by_class.get(
                value.item() if isinstance(value, np.generic) else value, 0
            )
            for value in target
        ],
        dtype=np.int64,
    )
    target_by_class = {
        value.item() if isinstance(value, np.generic) else value: int(count)
        for value, count in zip(target_values, target_counts)
    }
    common = set(target_by_class).intersection(reference_by_class)
    same_class_pairs = sum(
        target_by_class[value] * reference_by_class[value] for value in common
    )
    maximum_class_preserving_matching_rows = sum(
        min(target_by_class[value], reference_by_class[value]) for value in common
    )
    partner_quantiles = np.quantile(
        partner_counts.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    return {
        "rows_per_endpoint": int(len(target)),
        "common_classes": int(len(common)),
        "target_rows_with_at_least_one_reference_same_class_partner": int(
            np.sum(partner_counts > 0)
        ),
        "target_fraction_with_at_least_one_reference_same_class_partner": float(
            np.mean(partner_counts > 0)
        ),
        "target_same_class_reference_partner_count": {
            "minimum": int(np.min(partner_counts)),
            "q25": float(partner_quantiles[1]),
            "median": float(partner_quantiles[2]),
            "q75": float(partner_quantiles[3]),
            "maximum": int(np.max(partner_counts)),
            "mean": float(np.mean(partner_counts, dtype=np.float64)),
        },
        "maximum_class_preserving_matching_rows": int(
            maximum_class_preserving_matching_rows
        ),
        "maximum_class_preserving_matching_fraction": float(
            maximum_class_preserving_matching_rows / len(target)
        ),
        "minimum_rows_forced_to_cross_class_in_any_bijection": int(
            len(target) - maximum_class_preserving_matching_rows
        ),
        "minimum_fraction_forced_to_cross_class_in_any_bijection": float(
            1.0 - maximum_class_preserving_matching_rows / len(target)
        ),
        "same_class_candidate_pairs": int(same_class_pairs),
        "same_class_candidate_pair_fraction": float(
            same_class_pairs / (len(target) * len(reference))
        ),
    }


def construct_scaling_certificate_from_problem(
    problem: PreparedCertificateProblem,
    *,
    protocol: OptimizationCertificateProtocol,
    require_gap_target: bool = True,
) -> OptimizationCertificateResult:
    """Solve a fresh scaling certificate with bounded scratch memory.

    Unlike the historical follow-up solver, this separately frozen protocol
    does not build an eight-million-entry Python greedy set at ``n=8192``.
    The minimum-cost bijection is the initial feasible primal.  Lagrangian LAP
    solutions improve it, and evaluation stops at the first valid gap at or
    below the declared target, with the declared iteration count as a hard
    ceiling.  Every returned upper bound is a sampled Lagrangian bound; early
    stopping therefore changes runtime, not validity.
    """

    geometry = problem.geometry
    if (
        protocol.cost_geometry != geometry.protocol.cost_geometry
        or protocol.covariance_estimator != geometry.protocol.covariance_estimator
        or float(protocol.covariance_relative_floor)
        != float(geometry.protocol.covariance_relative_floor)
        or int(protocol.maximum_witness_rows)
        != int(geometry.protocol.maximum_witness_rows)
    ):
        raise ValueError("scaling solve changed the prepared geometry contract")
    cost = np.asarray(problem.cost, dtype=np.float64)
    assignment_zero = np.asarray(problem.assignment_world_zero, dtype=np.int64)
    reference_prediction = np.asarray(problem.reference_prediction)
    prediction_zero = np.asarray(problem.prediction_world_zero)
    rows = len(assignment_zero)
    row = np.arange(rows, dtype=np.int64)
    baseline = float(problem.baseline_cost)
    allowed = baseline * (1.0 + float(protocol.cost_budget_relative))
    epsilon = np.finfo(np.float64).eps
    tolerance = float(
        512.0 * rows * epsilon * max(allowed, baseline, float(np.max(cost)), 1.0)
    )
    reward = prediction_zero[:, None] != reference_prediction[None, :]
    if baseline > tolerance:
        scale = baseline
    else:
        positive = cost[cost > tolerance]
        scale = float(np.min(positive)) if positive.size else 1.0
    normalized_cost = cost / scale
    normalized_budget = allowed / scale
    objective = np.empty_like(normalized_cost)

    best_assignment = np.array(assignment_zero, copy=True)
    best_primal = 0.0
    best_primal_cost = baseline
    best_dual = 1.0
    best_multiplier = 0.0
    trace: list[dict[str, Any]] = []

    def current_gap() -> float:
        return float(max(min(1.0, max(best_primal, best_dual)) - best_primal, 0.0))

    def evaluate(scaled_multiplier: float) -> float:
        nonlocal best_assignment, best_primal, best_primal_cost
        nonlocal best_dual, best_multiplier
        np.multiply(normalized_cost, scaled_multiplier, out=objective)
        # This objective is ``n`` times the conventional mean-reward
        # Lagrangian objective.  Boolean subtraction avoids another dense
        # float64 reward allocation.
        np.subtract(objective, reward, out=objective, casting="unsafe")
        assignment = _linear_assignment(objective)
        assignment_cost = float(cost[row, assignment].sum(dtype=np.float64))
        primal = float(np.mean(reward[row, assignment], dtype=np.float64))
        multiplier = float(scaled_multiplier / rows)
        normalized_assignment_cost = assignment_cost / scale
        raw_dual = float(
            primal + multiplier * (normalized_budget - normalized_assignment_cost)
        )
        cushion = float(
            2048.0
            * rows
            * epsilon
            * max(
                1.0,
                abs(raw_dual),
                abs(multiplier)
                * (abs(normalized_budget) + abs(normalized_assignment_cost)),
            )
        )
        dual = min(1.0, raw_dual + cushion)
        if dual < best_dual:
            best_dual = dual
            best_multiplier = multiplier
        feasible = assignment_cost <= allowed
        tie = max(
            tolerance,
            float(protocol.assignment_tie_relative_tolerance)
            * max(abs(best_primal_cost), abs(assignment_cost), 1.0),
        )
        if feasible and (
            primal > best_primal + 8.0 * epsilon
            or (
                abs(primal - best_primal) <= 8.0 * epsilon
                and (
                    assignment_cost < best_primal_cost - tie
                    or (
                        abs(assignment_cost - best_primal_cost) <= tie
                        and tuple(assignment.tolist()) < tuple(best_assignment.tolist())
                    )
                )
            )
        ):
            best_assignment = np.array(assignment, copy=True)
            best_primal = primal
            best_primal_cost = assignment_cost
        trace.append(
            {
                "scaled_multiplier": float(scaled_multiplier),
                "mean_reward_multiplier": multiplier,
                "assignment_cost": assignment_cost,
                "disagreement": primal,
                "dual_upper_bound": dual,
                "budget_feasible": bool(feasible),
                "assignment_sha256": _array_sha256(assignment),
                "best_gap_after_evaluation": current_gap(),
            }
        )
        return assignment_cost

    stopped = False
    zero_cost = evaluate(0.0)
    lower = 0.0
    upper = float(rows)
    doublings = 0
    if current_gap() <= float(protocol.primal_dual_gap_target):
        stopped = True
    if zero_cost <= allowed:
        upper = 0.0
    elif not stopped:
        while True:
            upper_cost = evaluate(upper)
            if current_gap() <= float(protocol.primal_dual_gap_target):
                stopped = True
                break
            if upper_cost <= allowed:
                break
            lower = upper
            upper *= 2.0
            doublings += 1
            if doublings > 64 or not np.isfinite(upper):
                raise RuntimeError("could not bracket a feasible scaling dual")
        if not stopped:
            for _ in range(int(protocol.dual_bisection_iterations)):
                midpoint = float(lower + (upper - lower) / 2.0)
                midpoint_cost = evaluate(midpoint)
                if midpoint_cost > allowed:
                    lower = midpoint
                else:
                    upper = midpoint
                if current_gap() <= float(protocol.primal_dual_gap_target):
                    stopped = True
                    break

    dual_upper = float(min(1.0, max(best_primal, best_dual)))
    gap = float(max(dual_upper - best_primal, 0.0))
    if require_gap_target and gap > float(protocol.primal_dual_gap_target):
        raise RuntimeError(
            f"scaling certificate gap {gap} exceeds {protocol.primal_dual_gap_target}"
        )
    if len(np.unique(best_assignment)) != rows:
        raise RuntimeError("scaling world-one assignment is not a bijection")
    primal_cost = float(cost[row, best_assignment].sum(dtype=np.float64))
    if primal_cost > allowed:
        raise RuntimeError("scaling world-one assignment violates its cost budget")
    prediction_one = reference_prediction[best_assignment]
    selected = float(np.mean(prediction_zero != prediction_one))
    if abs(selected - best_primal) > 8.0 * epsilon:
        raise RuntimeError("scaling serialized disagreement is inconsistent")
    return OptimizationCertificateResult(
        assignment_world_zero=assignment_zero,
        assignment_world_one=best_assignment,
        prediction_world_zero=prediction_zero,
        prediction_world_one=prediction_one,
        target_canonical_order=problem.target_canonical_order,
        reference_canonical_order=problem.reference_canonical_order,
        baseline_cost=baseline,
        primal_cost=primal_cost,
        allowed_cost=allowed,
        selected_world_disagreement=selected,
        selected_two_world_minimax_recovery=float(1.0 - selected / 2.0),
        maximum_disagreement_primal_lower_bound=best_primal,
        maximum_disagreement_dual_upper_bound=dual_upper,
        primal_dual_gap=gap,
        diagnostics={
            "information_scope": INFORMATION_SCOPE,
            "external_ground_truth_labels_used_for_optimization": False,
            "frozen_readout_predictions_used_for_optimization": True,
            "pairing_used": False,
            "witness_rows": rows,
            "feature_dimension": int(geometry.readout_weight.shape[1]),
            "target_witness_indexed_atom_sha256": (problem.target_indexed_atom_sha256),
            "reference_witness_indexed_atom_sha256": (
                problem.reference_indexed_atom_sha256
            ),
            "cost_matrix_sha256": _array_sha256(cost),
            "reference_prediction_sha256": _array_sha256(reference_prediction),
            "protocol": protocol.as_dict(),
            "geometry": {
                **dict(geometry.diagnostics),
                "protocol": protocol.as_dict(),
            },
            "budget_independent_problem_reused": True,
            "primal_initializer": "minimum_cost_assignment",
            "python_greedy_pair_set_constructed": False,
            "optimization": {
                "primal_solver": "best_feasible_lagrangian_assignment",
                "dual_solver": "scipy_linear_sum_assignment_lagrangian_search",
                "cost_normalization_scale": float(scale),
                "normalized_budget": float(normalized_budget),
                "feasibility_tolerance": float(tolerance),
                "early_stop_target": float(protocol.primal_dual_gap_target),
                "early_stop_reached": bool(stopped),
                "maximum_bisection_iterations": int(protocol.dual_bisection_iterations),
                "evaluations": len(trace),
                "best_dual_multiplier": best_multiplier,
                "bracket": {
                    "lower_scaled": float(lower),
                    "upper_scaled": float(upper),
                    "doublings": int(doublings),
                },
                "trace": trace,
                "objective_buffer_reused": True,
                "dense_float_reward_matrix_created": False,
                "upper_bounds_lp_relaxation_optimum": True,
                "optimized_lagrangian_dual_value_claimed": False,
                "formal_real_arithmetic_certificate_claimed": False,
                "floating_point_solver_not_formally_verified": True,
                "roundoff_upper_cushion_applied": True,
            },
        },
    )


def run_scaling_exact_subset_validation(
    geometry: Any,
    target_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    target_sample_ids: np.ndarray | None = None,
    reference_sample_ids: np.ndarray | None = None,
    validation_rows: int = 64,
    time_limit_seconds: float = 900.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Validate the *scaling* solver against a proven small exact MILP."""

    target, target_order, _, _ = _unique_canonical_rows(
        target_features,
        name="scaling_exact_validation_target_features",
        salt=b"witness-scaling-exact-target-v1",
        sample_ids=target_sample_ids,
    )
    reference, reference_order, _, _ = _unique_canonical_rows(
        reference_features,
        name="scaling_exact_validation_reference_features",
        salt=b"witness-scaling-exact-reference-v1",
        sample_ids=reference_sample_ids,
    )
    count = int(validation_rows)
    if count < 2 or count > 64 or count > min(len(target), len(reference)):
        raise ValueError("scaling exact-validation row count is invalid")
    selected_target_ids = (
        None
        if target_sample_ids is None
        else np.asarray(target_sample_ids)[target_order][:count]
    )
    selected_reference_ids = (
        None
        if reference_sample_ids is None
        else np.asarray(reference_sample_ids)[reference_order][:count]
    )
    problem = prepare_certificate_problem(
        geometry,
        target[:count],
        reference[:count],
        target_sample_ids=selected_target_ids,
        reference_sample_ids=selected_reference_ids,
    )
    # Exhaust the locked search for this diagnostic rather than allowing the
    # production 1 pp early-stop rule to truncate it.  Small integer problems
    # may have a Lagrangian relaxation gap above 1 pp even when the MILP is
    # solved exactly, so containment—not a diagnostic duality-gap threshold—is
    # the relevant implementation check.
    diagnostic_protocol = replace(geometry.protocol, primal_dual_gap_target=0.0)
    approximate = construct_scaling_certificate_from_problem(
        problem,
        protocol=diagnostic_protocol,
        require_gap_target=False,
    )
    exact_assignment, exact = solve_exact_budgeted_disagreement_milp(
        problem.cost,
        problem.reference_prediction,
        problem.prediction_world_zero,
        allowed_cost=approximate.allowed_cost,
        time_limit_seconds=time_limit_seconds,
    )
    optimum = float(exact["maximum_disagreement_exact"])
    tolerance = 1e-10
    contains = bool(
        approximate.maximum_disagreement_primal_lower_bound - tolerance
        <= optimum
        <= approximate.maximum_disagreement_dual_upper_bound + tolerance
    )
    if not contains:
        raise RuntimeError("scaling solver bounds do not contain exact optimum")
    problem_arrays = {
        "cost": np.asarray(problem.cost, dtype=np.float64),
        "reference_prediction": np.asarray(problem.reference_prediction),
        "prediction_world_zero": np.asarray(problem.prediction_world_zero),
        "scaling_assignment_world_zero": np.asarray(
            approximate.assignment_world_zero, dtype=np.int64
        ),
        "scaling_assignment_world_one": np.asarray(
            approximate.assignment_world_one, dtype=np.int64
        ),
        "scaling_prediction_world_zero": np.asarray(approximate.prediction_world_zero),
        "scaling_prediction_world_one": np.asarray(approximate.prediction_world_one),
        "scaling_target_canonical_order": np.asarray(
            approximate.target_canonical_order, dtype=np.int64
        ),
        "scaling_reference_canonical_order": np.asarray(
            approximate.reference_canonical_order, dtype=np.int64
        ),
    }
    return (
        exact_assignment,
        {
            **exact,
            "certificate_solver": "witness_scaling_no_greedy_early_stop_v1",
            "certificate_primal_lower_bound": float(
                approximate.maximum_disagreement_primal_lower_bound
            ),
            "certificate_dual_upper_bound": float(
                approximate.maximum_disagreement_dual_upper_bound
            ),
            "certificate_primal_dual_gap": float(approximate.primal_dual_gap),
            "certificate_contains_exact_optimum": contains,
            "cost_matrix_sha256": _array_sha256(problem_arrays["cost"]),
            "reference_prediction_sha256": _array_sha256(
                problem_arrays["reference_prediction"]
            ),
            "prediction_world_zero_sha256": _array_sha256(
                problem_arrays["prediction_world_zero"]
            ),
            "scaling_certificate": approximate.summary(),
        },
        problem_arrays,
    )
