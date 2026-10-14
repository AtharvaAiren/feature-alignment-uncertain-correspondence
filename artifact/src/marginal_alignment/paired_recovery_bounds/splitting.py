"""Outcome-blind paired identity partition for recovery-bound calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_json
from marginal_alignment.campaign.feature_store import FeatureStore


PARTITION_SCHEMA = "marginal_alignment.paired_recovery_bounds.partition.v1"


def _scalar(value: Any) -> int | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError("paired recovery IDs must be integer or UTF-8 text scalars")
    return value


def _id_key(value: int | str) -> tuple[str, str]:
    return ("integer", str(value)) if isinstance(value, int) else ("text", value)


def _unique_ids(values: Sequence[Any] | np.ndarray, *, name: str) -> list[int | str]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.hasobject:
        raise IntegrityError(f"{name} must be a non-object rank-one array")
    result = [_scalar(value) for value in array]
    if len(set(result)) != len(result):
        raise IntegrityError(f"{name} contains duplicate IDs")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RecoveryBatch:
    reference_rows: np.ndarray
    target_rows: np.ndarray
    true_assignment: np.ndarray
    reference_ids: tuple[int | str, ...]
    target_ids: tuple[int | str, ...]

    @property
    def size(self) -> int:
        return int(len(self.reference_rows))

    @property
    def identity_sha256(self) -> str:
        return sha256_json({
            "reference_ids": list(self.reference_ids),
            "target_ids": list(self.target_ids),
        })

    def summary(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "identity_sha256": self.identity_sha256,
            "reference_rows_sha256": _array_sha256(self.reference_rows),
            "target_rows_sha256": _array_sha256(self.target_rows),
            "true_assignment_sha256": _array_sha256(self.true_assignment),
            "endpoint_orders_independently_permuted": True,
        }


@dataclass(frozen=True)
class RecoveryPartition:
    map_fit: RecoveryBatch
    calibration: tuple[RecoveryBatch, ...]
    evaluation: tuple[RecoveryBatch, ...]
    seed: int
    eligible_pair_count: int
    unused_pair_count: int
    pairing_sha256: str
    head_fit_ids_sha256: str

    def summary(self) -> dict[str, Any]:
        payload = {
            "schema_version": PARTITION_SCHEMA,
            "selection_inputs": "authenticated_opaque_pair_ids_only_no_features_labels_or_predictions",
            "selection_rng": "numpy.random.Generator(PCG64)",
            "seed": int(self.seed),
            "eligible_pair_count": int(self.eligible_pair_count),
            "unused_pair_count": int(self.unused_pair_count),
            "pairing_sha256": self.pairing_sha256,
            "head_fit_ids_sha256": self.head_fit_ids_sha256,
            "map_fit": self.map_fit.summary(),
            "calibration": [batch.summary() for batch in self.calibration],
            "evaluation": [batch.summary() for batch in self.evaluation],
        }
        payload["partition_sha256"] = sha256_json(payload)
        return payload


def build_recovery_partition(
    store: FeatureStore,
    *,
    seed: int,
    map_fit_rows: int,
    calibration_batches: int,
    evaluation_batches: int,
    batch_size: int,
    expected_eligible_pairs: int | None = None,
) -> RecoveryPartition:
    """Partition verified clean-corrupt identities without reading outcomes.

    Pair records are first sorted by opaque scalar value, making the selected
    partition invariant to the physical order of every endpoint and pairing
    array.  The RNG permutation is then split into mutually disjoint roles.
    """

    if not store.hashes_verified or store.kind != "native_v2":
        raise IntegrityError("paired recovery requires a fully hash-verified native V2 store")
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    reference_ids = _unique_ids(reference.ids, name="reference endpoint IDs")
    target_ids = _unique_ids(target.ids, name="target endpoint IDs")
    reference_row = {value: index for index, value in enumerate(reference_ids)}
    target_row = {value: index for index, value in enumerate(target_ids)}
    pairing = store.pairing()
    paired_reference, paired_target = pairing.validated()
    paired_reference = [_scalar(value) for value in paired_reference]
    paired_target = [_scalar(value) for value in paired_target]
    head_role = str(store.manifest["metadata"]["model_provenance"]["head_fit_ids_role"])
    head_fit_ids = _unique_ids(store.array(head_role), name="head-fit IDs")
    excluded = set(head_fit_ids)

    records: list[tuple[int | str, int | str, int, int]] = []
    for ref_id, tgt_id in zip(paired_reference, paired_target):
        if ref_id in excluded or tgt_id in excluded:
            continue
        if ref_id not in reference_row or tgt_id not in target_row:
            raise IntegrityError("pairing contains an ID outside an authenticated endpoint")
        records.append((ref_id, tgt_id, reference_row[ref_id], target_row[tgt_id]))
    records.sort(key=lambda row: (_id_key(row[0]), _id_key(row[1])))
    if len({row[0] for row in records}) != len(records) or len(
        {row[1] for row in records}
    ) != len(records):
        raise IntegrityError("eligible pairing is not one-to-one")
    if expected_eligible_pairs is not None and len(records) != int(expected_eligible_pairs):
        raise IntegrityError(
            f"eligible pair count changed: {len(records)} != {expected_eligible_pairs}"
        )

    map_count = int(map_fit_rows)
    cal_count = int(calibration_batches)
    eval_count = int(evaluation_batches)
    width = int(batch_size)
    if map_count < 2 or cal_count < 1 or eval_count < 1 or width < 2:
        raise ValueError("invalid map-fit or batch cardinality")
    required = map_count + (cal_count + eval_count) * width
    if required > len(records):
        raise IntegrityError(
            f"study requires {required} eligible pairs but only {len(records)} exist"
        )
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    order = generator.permutation(len(records))
    selected = [records[int(index)] for index in order[:required]]

    def make_batch(rows: list[tuple[int | str, int | str, int, int]]) -> RecoveryBatch:
        # Independently erase the two endpoint orders.  In particular, the
        # true assignment is not the identity merely because pairs happened
        # to be serialized together in the authenticated source table.
        reference_order = generator.permutation(len(rows))
        target_order = generator.permutation(len(rows))
        reference_position = np.empty(len(rows), dtype=np.int64)
        reference_position[reference_order] = np.arange(len(rows), dtype=np.int64)
        true_assignment = reference_position[target_order]
        return RecoveryBatch(
            reference_rows=np.asarray(
                [rows[int(index)][2] for index in reference_order], dtype=np.int64
            ),
            target_rows=np.asarray(
                [rows[int(index)][3] for index in target_order], dtype=np.int64
            ),
            true_assignment=true_assignment,
            reference_ids=tuple(rows[int(index)][0] for index in reference_order),
            target_ids=tuple(rows[int(index)][1] for index in target_order),
        )

    cursor = map_count
    map_fit = make_batch(selected[:cursor])
    calibration: list[RecoveryBatch] = []
    for _ in range(cal_count):
        calibration.append(make_batch(selected[cursor : cursor + width]))
        cursor += width
    evaluation: list[RecoveryBatch] = []
    for _ in range(eval_count):
        evaluation.append(make_batch(selected[cursor : cursor + width]))
        cursor += width

    all_hashes = [map_fit.identity_sha256]
    all_hashes.extend(batch.identity_sha256 for batch in calibration)
    all_hashes.extend(batch.identity_sha256 for batch in evaluation)
    if len(set(all_hashes)) != len(all_hashes):
        raise IntegrityError("role batches unexpectedly have identical ordered ID payloads")
    used_reference = set(map_fit.reference_ids)
    used_target = set(map_fit.target_ids)
    for batch in (*calibration, *evaluation):
        if used_reference.intersection(batch.reference_ids) or used_target.intersection(
            batch.target_ids
        ):
            raise IntegrityError("map-fit, calibration, and evaluation roles overlap")
        used_reference.update(batch.reference_ids)
        used_target.update(batch.target_ids)

    return RecoveryPartition(
        map_fit=map_fit,
        calibration=tuple(calibration),
        evaluation=tuple(evaluation),
        seed=int(seed),
        eligible_pair_count=len(records),
        unused_pair_count=len(records) - required,
        pairing_sha256=pairing.content_hash,
        head_fit_ids_sha256=str(store.records[head_role]["sha256"]),
    )
