"""Deterministic paired and identity-disjoint ImageNet-C split administration.

This module implements the administrative part of the pair-structure control.
Its selection stage receives identifiers only; its materialization helper then
slices endpoint features, while label validation is explicit and disabled by
default.  It never receives readout predictions or optimization outcomes.  The
selected endpoint arrays must still be canonicalized independently by the
certificate implementation before fitting geometry or optimizing an
assignment.

The two primary arms deliberately share their reference endpoint:

``paired``
    Reference block A and the corrupted counterparts of block A.

``disjoint_unpaired``
    The same reference block A and corrupted block B, where A and B contain
    disjoint underlying ImageNet identities.

The verified pairing table is therefore used for split administration and for
proving identity disjointness, but is not an input to the fitted map, cost,
reward, or assignment optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    canonical_json_bytes,
    seal_manifest,
    sha256_json,
    verify_manifest_checksum,
)


SELECTION_SCHEMA = "marginal_alignment.iclr_extension.pair_structure_selection.v1"
BUDGET_SCALE_SCHEMA = "marginal_alignment.random_gap_budget_conversion.v1"
_SELECTION_PURPOSE = "pre_outcome_pair_structure_split_administration"
_PAIRING_SCOPE = (
    "verified_pair_ids_used_only_to_exclude_head_fit_rows_select_common_"
    "identity_blocks_enforce_disjointness_and_enable_posthoc_pair_checks"
)
_ENDPOINT_ORDER_POLICY = (
    "calibration_and_witness_segments_hash_ordered_independently_by_endpoint;_"
    "certificate_must_still_apply_its_own_independent_feature_primary_"
    "canonicalization"
)

Id = int | str


def _id_value(value: Any, *, name: str) -> Id:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError(
            f"{name} IDs must be integer or string scalars; got "
            f"{type(value).__name__}"
        )
    return value


def _ids(values: Sequence[Any] | np.ndarray, *, name: str) -> tuple[Id, ...]:
    if isinstance(values, (str, bytes)):
        raise IntegrityError(f"{name} must be a rank-one ID sequence")
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise IntegrityError(f"{name} must be rank one; got {values.shape}")
        if values.dtype.hasobject:
            raise IntegrityError(f"{name} may not use object dtype")
        raw_values = values.tolist()
    else:
        try:
            raw_values = list(values)
        except TypeError as error:
            raise IntegrityError(f"{name} must be a rank-one ID sequence") from error
    # Normalize each scalar before constructing any NumPy array.  In
    # particular, ``np.asarray([1, "1"])`` silently coerces the integer to a
    # string and would otherwise conceal a mixed identifier type.
    result = tuple(_id_value(value, name=name) for value in raw_values)
    if len(set(result)) != len(result):
        raise IntegrityError(f"{name} contains duplicate IDs")
    if result and len({type(value) for value in result}) != 1:
        raise IntegrityError(f"{name} mixes integer and string IDs")
    return result


def _typed_id(value: Id) -> dict[str, Any]:
    return {
        "kind": "integer" if isinstance(value, int) else "string",
        "value": value,
    }


def _id_sequence_sha256(values: Sequence[Id], *, namespace: str) -> str:
    return sha256_json(
        {
            "namespace": namespace,
            "values": [_typed_id(value) for value in values],
        }
    )


def _row_sequence_sha256(values: np.ndarray, *, namespace: str) -> str:
    rows = np.asarray(values)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise IntegrityError(f"{namespace} rows must be a rank-one integer array")
    return sha256_json(
        {
            "namespace": namespace,
            "rows": [int(value) for value in rows],
        }
    )


def _pair_record(reference_id: Id, target_id: Id) -> dict[str, Any]:
    return {
        "reference_id": _typed_id(reference_id),
        "target_id": _typed_id(target_id),
    }


def _pair_sequence_sha256(pairs: Sequence[tuple[Id, Id]], *, namespace: str) -> str:
    return sha256_json(
        {
            "namespace": namespace,
            "pairs": [_pair_record(reference, target) for reference, target in pairs],
        }
    )


def _canonical_pairs(
    pairs: Sequence[tuple[Id, Id]],
) -> list[tuple[Id, Id]]:
    return sorted(
        pairs,
        key=lambda pair: (
            sha256_json(_pair_record(pair[0], pair[1])),
            canonical_json_bytes(_pair_record(pair[0], pair[1])),
        ),
    )


def _canonical_id_set(values: Sequence[Id]) -> list[Id]:
    return sorted(
        values,
        key=lambda value: (
            sha256_json(_typed_id(value)),
            canonical_json_bytes(_typed_id(value)),
        ),
    )


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    result = int(seed)
    if not 0 <= result < 2**63:
        raise ValueError("seed must lie in [0, 2**63)")
    return result


def _validate_salt(salt: str) -> str:
    if not isinstance(salt, str) or not salt or "\x00" in salt:
        raise ValueError("selection_salt must be a nonempty NUL-free string")
    return salt


@dataclass(frozen=True)
class PairStructureProtocol:
    """Row-count contract shared by the paired and disjoint-unpaired arms."""

    calibration_rows: int = 3840
    witness_rows: int = 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("calibration_rows", self.calibration_rows),
            ("witness_rows", self.witness_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if int(value) < 2:
                raise ValueError(f"{name} must be at least two")

    @property
    def block_rows(self) -> int:
        return int(self.calibration_rows) + int(self.witness_rows)

    def as_dict(self) -> dict[str, int]:
        return {
            "calibration_rows": int(self.calibration_rows),
            "witness_rows": int(self.witness_rows),
            "block_rows": self.block_rows,
            "required_disjoint_identity_rows": 2 * self.block_rows,
        }


def _readonly_rows(values: Sequence[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ArmSelection:
    """Endpoint rows in calibration-then-witness order for one arm."""

    name: str
    reference_rows: np.ndarray
    target_rows: np.ndarray
    reference_ids: tuple[Id, ...]
    target_ids: tuple[Id, ...]
    calibration_rows: int
    witness_rows: int
    true_pair_overlap_all_rows: int
    true_pair_overlap_calibration: int
    true_pair_overlap_witness: int
    witness_true_pair_assignment: np.ndarray | None

    @property
    def reference_calibration_rows(self) -> np.ndarray:
        return self.reference_rows[: self.calibration_rows]

    @property
    def reference_witness_rows(self) -> np.ndarray:
        return self.reference_rows[self.calibration_rows :]

    @property
    def target_calibration_rows(self) -> np.ndarray:
        return self.target_rows[: self.calibration_rows]

    @property
    def target_witness_rows(self) -> np.ndarray:
        return self.target_rows[self.calibration_rows :]

    @property
    def reference_calibration_ids(self) -> tuple[Id, ...]:
        return self.reference_ids[: self.calibration_rows]

    @property
    def reference_witness_ids(self) -> tuple[Id, ...]:
        return self.reference_ids[self.calibration_rows :]

    @property
    def target_calibration_ids(self) -> tuple[Id, ...]:
        return self.target_ids[: self.calibration_rows]

    @property
    def target_witness_ids(self) -> tuple[Id, ...]:
        return self.target_ids[self.calibration_rows :]

    def sample_id_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return safe non-object arrays for certificate tie administration."""

        reference = np.asarray(self.reference_ids)
        target = np.asarray(self.target_ids)
        if reference.dtype.hasobject or target.dtype.hasobject:
            raise IntegrityError(
                "selected endpoint IDs unexpectedly require object dtype"
            )
        return reference, target


@dataclass(frozen=True)
class PairStructureSelection:
    """A common-reference paired/disjoint-unpaired administrative selection."""

    protocol: PairStructureProtocol
    seed: int
    selection_salt: str
    eligible_pair_count: int
    paired: ArmSelection
    disjoint_unpaired: ArmSelection
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class PairStructureSplit:
    """Materialized endpoint data for one arm.

    Labels remain ``None`` unless ``validate_pair_labels=True`` is explicitly
    requested.  In that opt-in mode they are administrative validation data and
    must not be passed to ``fit_certificate_geometry``,
    ``prepare_certificate_problem``, or any assignment routine.
    """

    arm: str
    features_reference: np.ndarray
    features_target: np.ndarray
    ids_reference: np.ndarray
    ids_target: np.ndarray
    labels_reference: np.ndarray | None
    labels_target: np.ndarray | None
    selection: ArmSelection
    selection_manifest: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    @property
    def calibration_reference_features(self) -> np.ndarray:
        return self.features_reference[: self.selection.calibration_rows]

    @property
    def calibration_target_features(self) -> np.ndarray:
        return self.features_target[: self.selection.calibration_rows]

    @property
    def witness_reference_features(self) -> np.ndarray:
        return self.features_reference[self.selection.calibration_rows :]

    @property
    def witness_target_features(self) -> np.ndarray:
        return self.features_target[self.selection.calibration_rows :]

    @property
    def calibration_reference_ids(self) -> np.ndarray:
        return self.ids_reference[: self.selection.calibration_rows]

    @property
    def calibration_target_ids(self) -> np.ndarray:
        return self.ids_target[: self.selection.calibration_rows]

    @property
    def witness_reference_ids(self) -> np.ndarray:
        return self.ids_reference[self.selection.calibration_rows :]

    @property
    def witness_target_ids(self) -> np.ndarray:
        return self.ids_target[self.selection.calibration_rows :]


def _hash_order(
    pairs: Sequence[tuple[Id, Id]], *, seed: int, salt: str, purpose: str
) -> list[int]:
    keys: list[tuple[str, bytes, int]] = []
    for index, (reference_id, target_id) in enumerate(pairs):
        pair = _pair_record(reference_id, target_id)
        key = sha256_json(
            {
                "schema_version": SELECTION_SCHEMA,
                "seed": seed,
                "selection_salt": salt,
                "purpose": purpose,
                "pair": pair,
            }
        )
        # The serialized pair is an input-order-independent tie breaker even
        # under a hypothetical SHA-256 collision.
        keys.append((key, canonical_json_bytes(pair), index))
    return [index for _, _, index in sorted(keys)]


def _ordered_segment(
    pairs: Sequence[tuple[Id, Id]],
    *,
    seed: int,
    salt: str,
    purpose: str,
) -> list[tuple[Id, Id]]:
    return [
        pairs[index]
        for index in _hash_order(pairs, seed=seed, salt=salt, purpose=purpose)
    ]


def _arm(
    *,
    name: str,
    reference_pairs: Sequence[tuple[Id, Id]],
    target_pairs: Sequence[tuple[Id, Id]],
    reference_lookup: Mapping[Id, int],
    target_lookup: Mapping[Id, int],
    protocol: PairStructureProtocol,
    seed: int,
    salt: str,
    paired: bool,
) -> ArmSelection:
    if (
        len(reference_pairs) != protocol.block_rows
        or len(target_pairs) != protocol.block_rows
    ):
        raise IntegrityError("arm blocks do not match the declared row protocol")
    calibration = int(protocol.calibration_rows)

    reference_calibration = _ordered_segment(
        reference_pairs[:calibration],
        seed=seed,
        salt=salt,
        purpose="common-reference-calibration-order",
    )
    reference_witness = _ordered_segment(
        reference_pairs[calibration:],
        seed=seed,
        salt=salt,
        purpose="common-reference-witness-order",
    )
    target_calibration = _ordered_segment(
        target_pairs[:calibration],
        seed=seed,
        salt=salt,
        purpose=f"{name}-target-calibration-order",
    )
    target_witness = _ordered_segment(
        target_pairs[calibration:],
        seed=seed,
        salt=salt,
        purpose=f"{name}-target-witness-order",
    )
    ordered_reference = reference_calibration + reference_witness
    ordered_target = target_calibration + target_witness

    reference_rows = _readonly_rows(
        [reference_lookup[reference_id] for reference_id, _ in ordered_reference]
    )
    target_rows = _readonly_rows(
        [target_lookup[target_id] for _, target_id in ordered_target]
    )
    reference_ids = tuple(reference_id for reference_id, _ in ordered_reference)
    target_ids = tuple(target_id for _, target_id in ordered_target)

    reference_pair_by_id = {
        reference_id: (reference_id, target_id)
        for reference_id, target_id in reference_pairs
    }
    target_pair_by_id = {
        target_id: (reference_id, target_id) for reference_id, target_id in target_pairs
    }
    reference_keys = {reference_pair_by_id[value] for value in reference_ids}
    target_keys = {target_pair_by_id[value] for value in target_ids}
    reference_calibration_keys = {
        reference_pair_by_id[value] for value in reference_ids[:calibration]
    }
    target_calibration_keys = {
        target_pair_by_id[value] for value in target_ids[:calibration]
    }
    reference_witness_keys = {
        reference_pair_by_id[value] for value in reference_ids[calibration:]
    }
    target_witness_keys = {
        target_pair_by_id[value] for value in target_ids[calibration:]
    }
    overlaps = (
        len(reference_keys.intersection(target_keys)),
        len(reference_calibration_keys.intersection(target_calibration_keys)),
        len(reference_witness_keys.intersection(target_witness_keys)),
    )
    witness_assignment: np.ndarray | None
    if paired:
        expected = (protocol.block_rows, calibration, int(protocol.witness_rows))
        if overlaps != expected:
            raise IntegrityError(
                f"paired arm lost true pairs: {overlaps} != {expected}"
            )
        reference_position = {
            reference_pair_by_id[value]: position
            for position, value in enumerate(reference_ids[calibration:])
        }
        witness_assignment = _readonly_rows(
            [
                reference_position[target_pair_by_id[value]]
                for value in target_ids[calibration:]
            ]
        )
        if not np.array_equal(
            np.sort(witness_assignment), np.arange(protocol.witness_rows)
        ):
            raise IntegrityError("paired witness reconstruction is not a bijection")
    else:
        if overlaps != (0, 0, 0):
            raise IntegrityError(f"unpaired arm retains true-pair overlap: {overlaps}")
        witness_assignment = None

    return ArmSelection(
        name=name,
        reference_rows=reference_rows,
        target_rows=target_rows,
        reference_ids=reference_ids,
        target_ids=target_ids,
        calibration_rows=calibration,
        witness_rows=int(protocol.witness_rows),
        true_pair_overlap_all_rows=overlaps[0],
        true_pair_overlap_calibration=overlaps[1],
        true_pair_overlap_witness=overlaps[2],
        witness_true_pair_assignment=witness_assignment,
    )


def _arm_manifest(arm: ArmSelection) -> dict[str, Any]:
    namespace = f"pair-structure:{arm.name}"
    record: dict[str, Any] = {
        "name": arm.name,
        "calibration_rows": int(arm.calibration_rows),
        "witness_rows": int(arm.witness_rows),
        "reference_original_rows_sha256": _row_sequence_sha256(
            arm.reference_rows, namespace=f"{namespace}:reference-rows"
        ),
        "target_original_rows_sha256": _row_sequence_sha256(
            arm.target_rows, namespace=f"{namespace}:target-rows"
        ),
        "reference_ids_sha256": _id_sequence_sha256(
            arm.reference_ids, namespace=f"{namespace}:reference-ids"
        ),
        "target_ids_sha256": _id_sequence_sha256(
            arm.target_ids, namespace=f"{namespace}:target-ids"
        ),
        "reference_calibration_ids_sha256": _id_sequence_sha256(
            arm.reference_calibration_ids,
            namespace=f"{namespace}:reference-calibration-ids",
        ),
        "target_calibration_ids_sha256": _id_sequence_sha256(
            arm.target_calibration_ids,
            namespace=f"{namespace}:target-calibration-ids",
        ),
        "reference_witness_ids_sha256": _id_sequence_sha256(
            arm.reference_witness_ids,
            namespace=f"{namespace}:reference-witness-ids",
        ),
        "target_witness_ids_sha256": _id_sequence_sha256(
            arm.target_witness_ids,
            namespace=f"{namespace}:target-witness-ids",
        ),
        "true_pair_overlap_all_rows": int(arm.true_pair_overlap_all_rows),
        "true_pair_overlap_calibration": int(arm.true_pair_overlap_calibration),
        "true_pair_overlap_witness": int(arm.true_pair_overlap_witness),
        "pairing_available_to_geometry_cost_reward_or_optimizer": False,
        "labels_available_to_geometry_cost_reward_or_optimizer": False,
        "endpoint_order_policy": _ENDPOINT_ORDER_POLICY,
    }
    if arm.witness_true_pair_assignment is None:
        record["posthoc_witness_true_pair_assignment_sha256"] = None
    else:
        record["posthoc_witness_true_pair_assignment_sha256"] = _row_sequence_sha256(
            arm.witness_true_pair_assignment,
            namespace=f"{namespace}:posthoc-true-pair-assignment",
        )
    return record


def _manifest(
    *,
    protocol: PairStructureProtocol,
    seed: int,
    salt: str,
    reference_ids: tuple[Id, ...],
    target_ids: tuple[Id, ...],
    paired_reference_ids: tuple[Id, ...],
    paired_target_ids: tuple[Id, ...],
    head_fit_ids: tuple[Id, ...],
    eligible: Sequence[tuple[Id, Id]],
    block_a: Sequence[tuple[Id, Id]],
    block_b: Sequence[tuple[Id, Id]],
    paired: ArmSelection,
    unpaired: ArmSelection,
) -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": SELECTION_SCHEMA,
            "status": "PASS",
            "scientific_result": False,
            "purpose": _SELECTION_PURPOSE,
            "protocol": protocol.as_dict(),
            "seed": seed,
            "selection_salt": salt,
            "pairing_information_scope": _PAIRING_SCOPE,
            "pairing_used_for_split_administration": True,
            "pairing_available_to_geometry_cost_reward_or_optimizer": False,
            "labels_used_for_selection": False,
            "reference_universe_ids_sha256": _id_sequence_sha256(
                reference_ids, namespace="pair-structure:reference-universe"
            ),
            "target_universe_ids_sha256": _id_sequence_sha256(
                target_ids, namespace="pair-structure:target-universe"
            ),
            "pair_table_sha256": _pair_sequence_sha256(
                _canonical_pairs(list(zip(paired_reference_ids, paired_target_ids))),
                namespace="pair-structure:verified-pair-table",
            ),
            "head_fit_reference_ids_sha256": _id_sequence_sha256(
                _canonical_id_set(head_fit_ids),
                namespace="pair-structure:head-fit-reference-ids",
            ),
            "eligible_pair_count": len(eligible),
            "eligible_pairs_sha256": _pair_sequence_sha256(
                _canonical_pairs(eligible), namespace="pair-structure:eligible-pairs"
            ),
            "block_a_pairs_sha256": _pair_sequence_sha256(
                block_a, namespace="pair-structure:block-a"
            ),
            "block_b_pairs_sha256": _pair_sequence_sha256(
                block_b, namespace="pair-structure:block-b"
            ),
            "blocks_are_identity_disjoint": set(block_a).isdisjoint(set(block_b)),
            "arms_share_identical_reference_rows": (
                paired.reference_ids == unpaired.reference_ids
                and np.array_equal(paired.reference_rows, unpaired.reference_rows)
            ),
            "arms": {
                "paired": _arm_manifest(paired),
                "disjoint_unpaired": _arm_manifest(unpaired),
            },
        }
    )


def build_pair_structure_selection(
    *,
    reference_ids: Sequence[Any] | np.ndarray,
    target_ids: Sequence[Any] | np.ndarray,
    paired_reference_ids: Sequence[Any] | np.ndarray,
    paired_target_ids: Sequence[Any] | np.ndarray,
    head_fit_reference_ids: Sequence[Any] | np.ndarray,
    seed: int,
    selection_salt: str,
    protocol: PairStructureProtocol | None = None,
) -> PairStructureSelection:
    """Build paired and exactly identity-disjoint endpoint selections.

    The result is invariant to the row order of the supplied pairing table.
    Endpoint universe order is retained only to return original feature-store
    row indices; selection priority depends on authenticated ID values.
    """

    declared = PairStructureProtocol() if protocol is None else protocol
    if not isinstance(declared, PairStructureProtocol):
        raise TypeError("protocol must be a PairStructureProtocol")
    locked_seed = _validate_seed(seed)
    salt = _validate_salt(selection_salt)
    reference = _ids(reference_ids, name="reference universe")
    target = _ids(target_ids, name="target universe")
    paired_reference = _ids(paired_reference_ids, name="paired reference IDs")
    paired_target = _ids(paired_target_ids, name="paired target IDs")
    head_fit = _ids(head_fit_reference_ids, name="head-fit reference IDs")

    if len(paired_reference) != len(paired_target) or not paired_reference:
        raise IntegrityError("pairing table must contain equally many nonempty IDs")
    reference_set = set(reference)
    target_set = set(target)
    outside_reference = [
        value for value in paired_reference if value not in reference_set
    ]
    outside_target = [value for value in paired_target if value not in target_set]
    outside_head = [value for value in head_fit if value not in reference_set]
    if outside_reference:
        raise IntegrityError(
            f"pair table contains reference IDs outside the endpoint: {outside_reference[:3]}"
        )
    if outside_target:
        raise IntegrityError(
            f"pair table contains target IDs outside the endpoint: {outside_target[:3]}"
        )
    if outside_head:
        raise IntegrityError(
            f"head-fit IDs are outside the reference endpoint: {outside_head[:3]}"
        )

    head_fit_set = set(head_fit)
    eligible = [
        (reference_id, target_id)
        for reference_id, target_id in zip(paired_reference, paired_target)
        if reference_id not in head_fit_set
    ]
    required = 2 * declared.block_rows
    if len(eligible) < required:
        raise IntegrityError(
            f"pair structure needs {required} eligible identities, found {len(eligible)}"
        )
    ordered = _ordered_segment(
        eligible,
        seed=locked_seed,
        salt=salt,
        purpose="disjoint-identity-block-selection",
    )
    block_a = ordered[: declared.block_rows]
    block_b = ordered[declared.block_rows : required]
    if not set(block_a).isdisjoint(set(block_b)):
        raise IntegrityError("selected identity blocks are not disjoint")

    reference_lookup = {value: index for index, value in enumerate(reference)}
    target_lookup = {value: index for index, value in enumerate(target)}
    paired_arm = _arm(
        name="paired",
        reference_pairs=block_a,
        target_pairs=block_a,
        reference_lookup=reference_lookup,
        target_lookup=target_lookup,
        protocol=declared,
        seed=locked_seed,
        salt=salt,
        paired=True,
    )
    unpaired_arm = _arm(
        name="disjoint_unpaired",
        reference_pairs=block_a,
        target_pairs=block_b,
        reference_lookup=reference_lookup,
        target_lookup=target_lookup,
        protocol=declared,
        seed=locked_seed,
        salt=salt,
        paired=False,
    )
    manifest = _manifest(
        protocol=declared,
        seed=locked_seed,
        salt=salt,
        reference_ids=reference,
        target_ids=target,
        paired_reference_ids=paired_reference,
        paired_target_ids=paired_target,
        head_fit_ids=head_fit,
        eligible=eligible,
        block_a=block_a,
        block_b=block_b,
        paired=paired_arm,
        unpaired=unpaired_arm,
    )
    validate_pair_structure_manifest(manifest)
    return PairStructureSelection(
        protocol=declared,
        seed=locked_seed,
        selection_salt=salt,
        eligible_pair_count=len(eligible),
        paired=paired_arm,
        disjoint_unpaired=unpaired_arm,
        manifest=manifest,
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a mapping")
    return value


def _readonly_array(
    value: Any, *, dtype: np.dtype[Any] | type | None = None
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def build_pair_structure_split(
    store: Any,
    arm: str,
    seed: int,
    calibration_rows: int = 3840,
    witness_rows: int = 1024,
    salt_prefix: str = "imagenet-c-pair-structure-v1",
    *,
    validate_pair_labels: bool = False,
) -> PairStructureSplit:
    """Materialize one authenticated ImageNet-C pair-structure arm.

    ``store`` is expected to be a hash-verified native ``FeatureStore``.  The
    public signature is intentionally runner-friendly, while
    :func:`build_pair_structure_selection` remains available for pre-outcome
    lock generation without reading feature or label values.
    ``validate_pair_labels`` is false by default so ordinary scientific runs
    do not slice or interpret label values during endpoint construction.
    """

    if arm not in {"paired", "disjoint_unpaired"}:
        raise ValueError("arm must be 'paired' or 'disjoint_unpaired'")
    if getattr(store, "dataset", None) != "imagenet_c":
        raise IntegrityError("pair-structure control requires an ImageNet-C store")
    if getattr(store, "hashes_verified", None) is not True:
        raise IntegrityError("pair-structure control requires verified store hashes")
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    pairing = store.pairing()
    paired_reference_ids, paired_target_ids = pairing.validated()
    metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
    provenance = _mapping(metadata.get("model_provenance"), name="model provenance")
    head_fit_role = provenance.get("head_fit_ids_role")
    if not isinstance(head_fit_role, str) or not head_fit_role:
        raise IntegrityError("model provenance lacks head_fit_ids_role")
    head_fit_ids = store.array(head_fit_role)
    protocol = PairStructureProtocol(
        calibration_rows=calibration_rows,
        witness_rows=witness_rows,
    )
    selection = build_pair_structure_selection(
        reference_ids=reference.ids,
        target_ids=target.ids,
        paired_reference_ids=paired_reference_ids,
        paired_target_ids=paired_target_ids,
        head_fit_reference_ids=head_fit_ids,
        seed=seed,
        selection_salt=salt_prefix,
        protocol=protocol,
    )
    selected = selection.paired if arm == "paired" else selection.disjoint_unpaired
    features_reference = _readonly_array(
        np.asarray(reference.features)[selected.reference_rows], dtype=np.float64
    )
    features_target = _readonly_array(
        np.asarray(target.features)[selected.target_rows], dtype=np.float64
    )
    ids_reference, ids_target = selected.sample_id_arrays()
    ids_reference = _readonly_array(ids_reference)
    ids_target = _readonly_array(ids_target)
    labels_reference = (
        _readonly_array(np.asarray(reference.labels)[selected.reference_rows])
        if validate_pair_labels and reference.labels is not None
        else None
    )
    labels_target = (
        _readonly_array(np.asarray(target.labels)[selected.target_rows])
        if validate_pair_labels and target.labels is not None
        else None
    )
    if (labels_reference is None) != (labels_target is None):
        raise IntegrityError("only one ImageNet-C endpoint exposes labels")

    label_pairs_checked = 0
    if labels_reference is not None and labels_target is not None:
        selected_reference_ids = set(selected.reference_ids)
        selected_target_ids = set(selected.target_ids)
        all_reference_labels = np.asarray(reference.labels)
        all_target_labels = np.asarray(target.labels)
        reference_lookup = {
            _id_value(value, name="reference universe"): index
            for index, value in enumerate(np.asarray(reference.ids))
        }
        target_lookup = {
            _id_value(value, name="target universe"): index
            for index, value in enumerate(np.asarray(target.ids))
        }
        for reference_id, target_id in zip(paired_reference_ids, paired_target_ids):
            if (
                reference_id in selected_reference_ids
                or target_id in selected_target_ids
            ):
                reference_label = all_reference_labels[reference_lookup[reference_id]]
                target_label = all_target_labels[target_lookup[target_id]]
                if (
                    np.asarray(reference_label).item()
                    != np.asarray(target_label).item()
                ):
                    raise IntegrityError(
                        "verified ImageNet-C pair has inconsistent endpoint labels"
                    )
                label_pairs_checked += 1

    diagnostics = {
        "schema_version": "marginal_alignment.iclr_extension.pair_structure_split.v1",
        "status": "PASS",
        "scientific_result": False,
        "arm": arm,
        "dataset": "imagenet_c",
        "encoder": str(getattr(store, "encoder", "")),
        "store_manifest_payload_sha256": sha256_json(store.manifest),
        "selection_manifest_sha256": selection.manifest["manifest_sha256"],
        "calibration_rows": int(calibration_rows),
        "witness_rows": int(witness_rows),
        "pairing_used_for_split_administration": True,
        "pairing_available_to_geometry_cost_reward_or_optimizer": False,
        "labels_used_for_administrative_pair_validation": labels_reference is not None,
        "pair_label_validation_requested": bool(validate_pair_labels),
        "labels_available_to_geometry_cost_reward_or_optimizer": False,
        "administrative_label_pairs_checked": label_pairs_checked,
        "true_pair_overlap_all_rows": selected.true_pair_overlap_all_rows,
        "true_pair_overlap_calibration": selected.true_pair_overlap_calibration,
        "true_pair_overlap_witness": selected.true_pair_overlap_witness,
    }
    return PairStructureSplit(
        arm=arm,
        features_reference=features_reference,
        features_target=features_target,
        ids_reference=ids_reference,
        ids_target=ids_target,
        labels_reference=labels_reference,
        labels_target=labels_target,
        selection=selected,
        selection_manifest=selection.manifest,
        diagnostics=diagnostics,
    )


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IntegrityError(f"{name} is not a lowercase SHA-256 digest")
    return value


def validate_pair_structure_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a sealed selection manifest without trusting its producer."""

    if not isinstance(manifest, Mapping):
        raise IntegrityError("pair-structure manifest must be a mapping")
    verify_manifest_checksum(manifest)
    expected_manifest_fields = {
        "schema_version",
        "status",
        "scientific_result",
        "purpose",
        "protocol",
        "seed",
        "selection_salt",
        "pairing_information_scope",
        "pairing_used_for_split_administration",
        "pairing_available_to_geometry_cost_reward_or_optimizer",
        "labels_used_for_selection",
        "reference_universe_ids_sha256",
        "target_universe_ids_sha256",
        "pair_table_sha256",
        "head_fit_reference_ids_sha256",
        "eligible_pair_count",
        "eligible_pairs_sha256",
        "block_a_pairs_sha256",
        "block_b_pairs_sha256",
        "blocks_are_identity_disjoint",
        "arms_share_identical_reference_rows",
        "arms",
        "manifest_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise IntegrityError("pair-structure manifest field inventory drifted")
    if (
        manifest.get("schema_version") != SELECTION_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_result") is not False
        or manifest.get("purpose") != _SELECTION_PURPOSE
        or manifest.get("pairing_information_scope") != _PAIRING_SCOPE
        or manifest.get("pairing_used_for_split_administration") is not True
        or manifest.get("pairing_available_to_geometry_cost_reward_or_optimizer")
        is not False
        or manifest.get("labels_used_for_selection") is not False
        or manifest.get("blocks_are_identity_disjoint") is not True
        or manifest.get("arms_share_identical_reference_rows") is not True
    ):
        raise IntegrityError(
            "pair-structure manifest status or information scope drifted"
        )
    try:
        _validate_seed(manifest.get("seed"))
        _validate_salt(manifest.get("selection_salt"))
    except ValueError as error:
        raise IntegrityError(
            "pair-structure seed or selection salt is invalid"
        ) from error
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise IntegrityError("pair-structure protocol is absent")
    try:
        declared = PairStructureProtocol(
            calibration_rows=int(protocol["calibration_rows"]),
            witness_rows=int(protocol["witness_rows"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("pair-structure protocol is invalid") from error
    if dict(protocol) != declared.as_dict():
        raise IntegrityError("pair-structure protocol contains undeclared fields")
    eligible = manifest.get("eligible_pair_count")
    if (
        isinstance(eligible, bool)
        or not isinstance(eligible, int)
        or eligible < 2 * declared.block_rows
    ):
        raise IntegrityError("eligible pair count cannot support both blocks")
    for name in (
        "reference_universe_ids_sha256",
        "target_universe_ids_sha256",
        "pair_table_sha256",
        "head_fit_reference_ids_sha256",
        "eligible_pairs_sha256",
        "block_a_pairs_sha256",
        "block_b_pairs_sha256",
    ):
        _digest(manifest.get(name), name=name)
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"paired", "disjoint_unpaired"}:
        raise IntegrityError("pair-structure arm inventory drifted")
    digest_fields = {
        "reference_original_rows_sha256",
        "target_original_rows_sha256",
        "reference_ids_sha256",
        "target_ids_sha256",
        "reference_calibration_ids_sha256",
        "target_calibration_ids_sha256",
        "reference_witness_ids_sha256",
        "target_witness_ids_sha256",
    }
    expected_arm_fields = {
        "name",
        "calibration_rows",
        "witness_rows",
        *digest_fields,
        "true_pair_overlap_all_rows",
        "true_pair_overlap_calibration",
        "true_pair_overlap_witness",
        "pairing_available_to_geometry_cost_reward_or_optimizer",
        "labels_available_to_geometry_cost_reward_or_optimizer",
        "endpoint_order_policy",
        "posthoc_witness_true_pair_assignment_sha256",
    }
    for name, expected_overlap in (
        (
            "paired",
            (declared.block_rows, declared.calibration_rows, declared.witness_rows),
        ),
        ("disjoint_unpaired", (0, 0, 0)),
    ):
        arm = arms[name]
        if not isinstance(arm, Mapping) or arm.get("name") != name:
            raise IntegrityError(f"{name} arm is malformed")
        if set(arm) != expected_arm_fields:
            raise IntegrityError(f"{name} arm field inventory drifted")
        if (
            arm.get("calibration_rows") != declared.calibration_rows
            or arm.get("witness_rows") != declared.witness_rows
            or arm.get("pairing_available_to_geometry_cost_reward_or_optimizer")
            is not False
            or arm.get("labels_available_to_geometry_cost_reward_or_optimizer")
            is not False
            or arm.get("endpoint_order_policy") != _ENDPOINT_ORDER_POLICY
        ):
            raise IntegrityError(f"{name} arm protocol or information scope drifted")
        observed_overlap = (
            arm.get("true_pair_overlap_all_rows"),
            arm.get("true_pair_overlap_calibration"),
            arm.get("true_pair_overlap_witness"),
        )
        if observed_overlap != expected_overlap:
            raise IntegrityError(
                f"{name} true-pair overlap is {observed_overlap}, expected {expected_overlap}"
            )
        for field in digest_fields:
            _digest(arm.get(field), name=f"{name}.{field}")
        assignment = arm.get("posthoc_witness_true_pair_assignment_sha256")
        if name == "paired":
            _digest(assignment, name=f"{name}.posthoc assignment")
        elif assignment is not None:
            raise IntegrityError("unpaired arm unexpectedly has a true-pair assignment")


def verify_pair_structure_selection(
    selection: PairStructureSelection,
    **inputs: Any,
) -> None:
    """Rebuild a selection from authenticated inputs and require exact identity."""

    if not isinstance(selection, PairStructureSelection):
        raise TypeError("selection must be a PairStructureSelection")
    reproduced = build_pair_structure_selection(
        **inputs,
        seed=selection.seed,
        selection_salt=selection.selection_salt,
        protocol=selection.protocol,
    )
    if dict(reproduced.manifest) != dict(selection.manifest):
        raise IntegrityError("reconstructed pair-structure manifest differs")
    for observed, expected in (
        (selection.paired, reproduced.paired),
        (selection.disjoint_unpaired, reproduced.disjoint_unpaired),
    ):
        if (
            observed.reference_ids != expected.reference_ids
            or observed.target_ids != expected.target_ids
            or not np.array_equal(observed.reference_rows, expected.reference_rows)
            or not np.array_equal(observed.target_rows, expected.target_rows)
        ):
            raise IntegrityError(f"reconstructed {observed.name} rows differ")


def uniform_random_cost_matrix_moments(cost_matrix: np.ndarray) -> dict[str, Any]:
    """Return exact moments of total cost under a uniform random bijection.

    This is geometry-agnostic and can be applied directly to the immutable
    dense cost matrix in ``PreparedCertificateProblem``.  If ``A`` is the
    doubly centered matrix, ``Var(sum_i C[i,pi(i)]) = ||A||_F^2/(n-1)``.
    """

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if (
        cost.ndim != 2
        or cost.shape[0] != cost.shape[1]
        or cost.shape[0] < 2
        or not np.all(np.isfinite(cost))
        or np.any(cost < 0.0)
    ):
        raise ValueError("cost_matrix must be a finite nonnegative square matrix")
    rows = int(cost.shape[0])
    row_mean = np.mean(cost, axis=1, dtype=np.float64)
    column_mean = np.mean(cost, axis=0, dtype=np.float64)
    grand_mean = float(np.mean(row_mean, dtype=np.float64))
    centered = cost - row_mean[:, None] - column_mean[None, :] + grand_mean
    expected = float(np.sum(cost, dtype=np.float64) / rows)
    variance = max(
        float(np.sum(np.square(centered), dtype=np.float64) / (rows - 1)),
        0.0,
    )
    standard_deviation = float(math.sqrt(variance))
    return {
        "expected_total_cost": expected,
        "variance_total_cost": variance,
        "standard_deviation_total_cost": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / expected if expected > 0.0 else 0.0
        ),
    }


def random_cost_moments_from_cost_matrix(
    cost_matrix: np.ndarray,
) -> dict[str, Any]:
    """Public runner-facing name for exact random-bijection cost moments."""

    return uniform_random_cost_matrix_moments(cost_matrix)


def _cost_scale(
    minimum_cost: float, expected_random_cost: float
) -> tuple[float, float]:
    minimum = float(minimum_cost)
    random = float(expected_random_cost)
    if (
        not math.isfinite(minimum)
        or minimum <= 0.0
        or not math.isfinite(random)
        or random <= minimum
    ):
        raise ValueError(
            "minimum_cost must be positive and expected_random_cost must exceed it"
        )
    return minimum, random


def relative_budget_to_random_gap_fraction(
    relative_budget: float,
    *,
    minimum_cost: float,
    expected_random_cost: float,
) -> float:
    """Express a multiplicative minimum-cost budget on the random-gap scale."""

    minimum, random = _cost_scale(minimum_cost, expected_random_cost)
    budget = float(relative_budget)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("relative_budget must be finite and nonnegative")
    return float(minimum * budget / (random - minimum))


def random_gap_fraction_to_relative_budget(
    random_gap_fraction: float,
    *,
    minimum_cost: float,
    expected_random_cost: float,
) -> float:
    """Convert a fraction of random-minus-minimum cost to a relative budget."""

    minimum, random = _cost_scale(minimum_cost, expected_random_cost)
    fraction = float(random_gap_fraction)
    if not math.isfinite(fraction) or fraction < 0.0:
        raise ValueError("random_gap_fraction must be finite and nonnegative")
    return float(fraction * (random - minimum) / minimum)


def relative_budget_for_random_gap_share(
    baseline: float,
    random_mean: float,
    share: float,
) -> float:
    """Return the multiplicative budget matching ``share`` of the random gap."""

    return random_gap_fraction_to_relative_budget(
        share,
        minimum_cost=baseline,
        expected_random_cost=random_mean,
    )


def random_gap_budget_table(
    *,
    minimum_cost: float,
    expected_random_cost: float,
    relative_budgets: Sequence[float],
) -> dict[str, Any]:
    """Serialize relative budgets and their exact random-gap equivalents."""

    minimum, random = _cost_scale(minimum_cost, expected_random_cost)
    rows = []
    for raw_budget in relative_budgets:
        budget = float(raw_budget)
        fraction = relative_budget_to_random_gap_fraction(
            budget,
            minimum_cost=minimum,
            expected_random_cost=random,
        )
        rows.append(
            {
                "cost_budget_relative": budget,
                "allowed_cost": minimum * (1.0 + budget),
                "allowed_excess_fraction_of_random_minus_minimum": fraction,
            }
        )
    return {
        "schema_version": BUDGET_SCALE_SCHEMA,
        "minimum_cost": minimum,
        "expected_random_bijection_cost": random,
        "random_minus_minimum_cost": random - minimum,
        "budgets": rows,
    }
