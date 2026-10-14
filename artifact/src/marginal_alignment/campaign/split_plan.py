"""Deterministic V2 split and correspondence-dose plans.

Evaluation identities are selected once and remain fixed for every calibration
size.  Marginal reference and target endpoint samples use independent RNG
streams.  True, shuffled, and noisy pairing controls use identical endpoint
sets at each pair budget so the estimator and sample budget stay matched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    IntegrityError,
    seal_manifest,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)


SPLIT_SCHEMA = "marginal_alignment.split_plan.v2"


def _id_value(value: Any) -> int | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError(f"IDs must be integer or string scalars, got {type(value).__name__}")
    return value


def _ids(values: Sequence[Any] | np.ndarray, *, name: str) -> list[int | str]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise IntegrityError(f"{name} must be rank one; got {array.shape}")
    if array.dtype.hasobject:
        raise IntegrityError(f"{name} may not use object dtype")
    result = [_id_value(value) for value in array]
    if len(set(result)) != len(result):
        raise IntegrityError(f"{name} contains duplicate IDs")
    return result


def _sequence_hash(values: Sequence[int | str], *, namespace: str) -> str:
    return sha256_json({"namespace": namespace, "values": list(values)})


@dataclass(frozen=True)
class PairingTable:
    reference_ids: Sequence[Any] | np.ndarray
    target_ids: Sequence[Any] | np.ndarray
    provenance: Mapping[str, Any]
    reference_universe: Sequence[Any] | np.ndarray | None = None
    target_universe: Sequence[Any] | np.ndarray | None = None

    def validated(self) -> tuple[list[int | str], list[int | str]]:
        reference = _ids(self.reference_ids, name="paired reference IDs")
        target = _ids(self.target_ids, name="paired target IDs")
        if len(reference) != len(target) or not reference:
            raise IntegrityError("pairing table must contain equally many nonempty endpoint IDs")
        if self.provenance.get("verified") is not True:
            raise IntegrityError("pairing provenance is not verified")
        if not self.provenance.get("kind") or not self.provenance.get("source"):
            raise IntegrityError("pairing provenance requires kind and source")
        if self.reference_universe is not None:
            universe = set(_ids(self.reference_universe, name="reference universe"))
            outside = [value for value in reference if value not in universe]
            if outside:
                raise IntegrityError(f"pairing references IDs outside the reference endpoint: {outside[:3]}")
        if self.target_universe is not None:
            universe = set(_ids(self.target_universe, name="target universe"))
            outside = [value for value in target if value not in universe]
            if outside:
                raise IntegrityError(f"pairing references IDs outside the target endpoint: {outside[:3]}")
        return reference, target

    @property
    def content_hash(self) -> str:
        reference, target = self.validated()
        return sha256_json({
            "reference_ids": reference,
            "target_ids": target,
            "provenance": dict(self.provenance),
        })


def _positive_sorted_unique(values: Sequence[int], *, name: str) -> list[int]:
    result = sorted({int(value) for value in values})
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(result) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _deranged(values: Sequence[int | str], rng: np.random.Generator) -> list[int | str]:
    source = list(values)
    if len(source) < 2:
        raise ValueError("a matched shuffled-pair control requires at least two pairs")
    order = rng.permutation(len(source))
    # Rejection is cheap for the intended budgets; rotation is a deterministic
    # bounded fallback and remains a derangement for unique endpoint IDs.
    for _ in range(128):
        if np.all(order != np.arange(len(source))):
            return [source[int(index)] for index in order]
        order = rng.permutation(len(source))
    return source[1:] + source[:1]


def _pair_map(
    reference: Sequence[int | str],
    target: Sequence[int | str],
    true_target: Sequence[int | str],
    *,
    kind: str,
    requested_noise_fraction: float | None = None,
) -> dict[str, Any]:
    mismatches = int(sum(left != right for left, right in zip(target, true_target)))
    result: dict[str, Any] = {
        "kind": kind,
        "reference_ids": list(reference),
        "target_ids": list(target),
        "reference_set_sha256": sha256_json(sorted(reference, key=lambda value: (str(type(value)), str(value)))),
        "target_set_sha256": sha256_json(sorted(target, key=lambda value: (str(type(value)), str(value)))),
        "assignment_sha256": sha256_json(list(zip(reference, target))),
        "mismatched_pairs": mismatches,
        "actual_noise_fraction": float(mismatches / len(reference)) if reference else 0.0,
    }
    if requested_noise_fraction is not None:
        result["requested_noise_fraction"] = float(requested_noise_fraction)
    return result


def _noisy_assignment(
    true_target: Sequence[int | str],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int | str], int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("noise fractions must lie in [0,1]")
    result = list(true_target)
    if fraction == 0.0 or not result:
        return result, 0
    changed = int(round(fraction * len(result)))
    # Exactly one mismatch is impossible while preserving the target endpoint
    # multiset.  Use the nearest feasible nonzero matched control and record
    # its actual fraction explicitly.
    changed = min(len(result), max(2, changed))
    if changed < 2:
        return result, 0
    positions = sorted(int(value) for value in rng.choice(len(result), size=changed, replace=False))
    selected = [result[index] for index in positions]
    permuted = _deranged(selected, rng)
    for index, value in zip(positions, permuted):
        result[index] = value
    return result, changed


def build_split_plan(
    *,
    reference_ids: Sequence[Any] | np.ndarray,
    target_ids: Sequence[Any] | np.ndarray,
    pairing: PairingTable,
    seed: int,
    evaluation_size: int,
    calibration_sizes: Sequence[int],
    pair_budgets: Sequence[int],
    noisy_pair_fractions: Sequence[float] = (0.10, 0.25, 0.50),
    excluded_reference_ids: Sequence[Any] | np.ndarray | None = None,
    excluded_target_ids: Sequence[Any] | np.ndarray | None = None,
    head_fit_exclusion_required: bool = False,
    head_fit_ids_count: int = 0,
    reference_namespace: str = "reference",
    target_namespace: str = "target",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    reference = _ids(reference_ids, name="reference endpoint IDs")
    target = _ids(target_ids, name="target endpoint IDs")
    if not reference_namespace or not target_namespace:
        raise ValueError("endpoint namespaces must be nonempty")
    paired_reference, paired_target = pairing.validated()
    excluded_reference = set(_ids(
        [] if excluded_reference_ids is None else excluded_reference_ids,
        name="excluded reference IDs",
    ))
    excluded_target = set(_ids(
        [] if excluded_target_ids is None else excluded_target_ids,
        name="excluded target IDs",
    ))
    reference_set, target_set = set(reference), set(target)
    unknown_reference = excluded_reference - reference_set
    unknown_target = excluded_target - target_set
    if unknown_reference or unknown_target:
        raise IntegrityError(
            "excluded head-fit IDs must belong to their declared endpoint namespaces"
        )
    if head_fit_exclusion_required:
        if int(head_fit_ids_count) <= 0:
            raise IntegrityError("required head-fit exclusion declared an empty ID set")
        if len(excluded_reference) != int(head_fit_ids_count):
            raise IntegrityError(
                "required head-fit ID count does not match excluded reference identities"
            )
        if len(excluded_target) != int(head_fit_ids_count):
            raise IntegrityError(
                "required head-fit counterpart count does not match excluded target identities"
            )
    eligible_positions = [
        index for index, (reference_id, target_id) in enumerate(
            zip(paired_reference, paired_target)
        )
        if reference_id not in excluded_reference and target_id not in excluded_target
    ]
    paired_reference = [paired_reference[index] for index in eligible_positions]
    paired_target = [paired_target[index] for index in eligible_positions]
    if not set(paired_reference).issubset(reference_set):
        raise IntegrityError("pairing contains IDs outside the reference endpoint")
    if not set(paired_target).issubset(target_set):
        raise IntegrityError("pairing contains IDs outside the target endpoint")
    evaluation_size = int(evaluation_size)
    if evaluation_size < 2 or evaluation_size >= len(paired_reference):
        raise ValueError("evaluation_size must leave at least one paired row for fitting")
    calibration = _positive_sorted_unique(calibration_sizes, name="calibration_sizes")
    budgets = _positive_sorted_unique(pair_budgets, name="pair_budgets") if pair_budgets else []
    fractions = sorted({float(value) for value in noisy_pair_fractions})
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("noise fractions must lie in [0,1]")

    seeds = np.random.SeedSequence(int(seed)).spawn(6)
    rng_evaluation = np.random.default_rng(seeds[0])
    rng_reference = np.random.default_rng(seeds[1])
    rng_target = np.random.default_rng(seeds[2])
    rng_pairs = np.random.default_rng(seeds[3])
    rng_shuffled = np.random.default_rng(seeds[4])
    rng_noisy = np.random.default_rng(seeds[5])

    pair_order = rng_evaluation.permutation(len(paired_reference))
    eval_positions = pair_order[:evaluation_size]
    evaluation_reference = [paired_reference[int(index)] for index in eval_positions]
    evaluation_target = [paired_target[int(index)] for index in eval_positions]
    evaluation_reference_set = set(evaluation_reference)
    evaluation_target_set = set(evaluation_target)

    fit_reference = [
        value for value in reference
        if value not in evaluation_reference_set and value not in excluded_reference
    ]
    fit_target = [
        value for value in target
        if value not in evaluation_target_set and value not in excluded_target
    ]
    maximum_calibration = max(calibration)
    if maximum_calibration > len(fit_reference) or maximum_calibration > len(fit_target):
        raise ValueError("largest calibration size does not fit outside the fixed evaluation set")
    reference_order = [fit_reference[int(index)] for index in rng_reference.permutation(len(fit_reference))]
    target_order = [fit_target[int(index)] for index in rng_target.permutation(len(fit_target))]

    remaining_pair_positions = pair_order[evaluation_size:]
    remaining_pair_positions = remaining_pair_positions[
        rng_pairs.permutation(len(remaining_pair_positions))
    ]
    if budgets and max(budgets) > len(remaining_pair_positions):
        raise ValueError("largest pair budget does not fit outside the fixed evaluation set")

    evaluation = {
        "reference_ids": evaluation_reference,
        "target_ids": evaluation_target,
        "reference_ids_sha256": _sequence_hash(evaluation_reference, namespace=reference_namespace),
        "target_ids_sha256": _sequence_hash(evaluation_target, namespace=target_namespace),
        "pair_assignment_sha256": sha256_json(list(zip(evaluation_reference, evaluation_target))),
        "size": evaluation_size,
    }
    evaluation_hash = sha256_json(evaluation)
    calibration_records: dict[str, Any] = {}
    for size in calibration:
        selected_reference = reference_order[:size]
        selected_target = target_order[:size]
        calibration_records[str(size)] = {
            "size": size,
            "reference_ids": selected_reference,
            "target_ids": selected_target,
            "reference_ids_sha256": _sequence_hash(selected_reference, namespace=reference_namespace),
            "target_ids_sha256": _sequence_hash(selected_target, namespace=target_namespace),
            "evaluation_sha256": evaluation_hash,
            "selection": "independent_endpoint_rng_streams",
        }

    pair_records: dict[str, Any] = {}
    for budget in budgets:
        positions = remaining_pair_positions[:budget]
        true_reference = [paired_reference[int(index)] for index in positions]
        true_target = [paired_target[int(index)] for index in positions]
        true_map = _pair_map(true_reference, true_target, true_target, kind="true")
        controls: dict[str, Any] = {"true": true_map}
        if budget >= 2:
            shuffled_target = _deranged(true_target, rng_shuffled)
            controls["shuffled"] = _pair_map(
                true_reference, shuffled_target, true_target, kind="shuffled"
            )
            noisy: dict[str, Any] = {}
            for fraction in fractions:
                noisy_target, _ = _noisy_assignment(true_target, fraction, rng_noisy)
                label = format(fraction, ".6g")
                noisy[label] = _pair_map(
                    true_reference,
                    noisy_target,
                    true_target,
                    kind="noisy",
                    requested_noise_fraction=fraction,
                )
            controls["noisy"] = noisy
        else:
            controls["shuffled"] = {
                "status": "unavailable",
                "reason": "a one-pair endpoint set has no false bijection",
            }
            controls["noisy"] = {}
        pair_records[str(budget)] = {
            "budget": budget,
            "evaluation_sha256": evaluation_hash,
            "controls": controls,
        }

    input_provenance = {
        "reference_namespace": reference_namespace,
        "target_namespace": target_namespace,
        "reference_ids_sha256": _sequence_hash(reference, namespace=reference_namespace),
        "target_ids_sha256": _sequence_hash(target, namespace=target_namespace),
        "pairing_sha256": pairing.content_hash,
        "pairing_provenance": dict(pairing.provenance),
    }
    payload = {
        "schema_version": SPLIT_SCHEMA,
        "seed": int(seed),
        "rng": "numpy.random.PCG64_via_SeedSequence_spawn",
        "input_provenance": input_provenance,
        "policy": {
            "fixed_evaluation_across_calibration_sizes": True,
            "nested_calibration_prefixes": True,
            "independent_reference_target_endpoint_sampling": True,
            "nested_true_pair_endpoint_sets": True,
            "matched_pair_controls_reuse_exact_endpoint_sets": True,
            "evaluation_disjoint_from_every_fit_set": True,
            "marginal_and_paired_fit_sets_may_overlap": True,
            "head_fit_identities_excluded_from_fit_and_evaluation": bool(
                head_fit_exclusion_required
            ),
            "head_fit_exclusion_required": bool(head_fit_exclusion_required),
            "head_fit_ids_count": int(head_fit_ids_count),
        },
        "evaluation": evaluation,
        "evaluation_sha256": evaluation_hash,
        "calibration_sizes": calibration,
        "calibrations": calibration_records,
        "pair_budgets": budgets,
        "pair_controls": pair_records,
        "noisy_pair_fractions": fractions,
    }
    plan = seal_manifest(payload)
    validate_split_plan(plan)
    if output_path is not None:
        write_json_atomic(output_path, plan)
    return plan


def _assert_prefix(smaller: Sequence[Any], larger: Sequence[Any], *, name: str) -> None:
    if list(larger[: len(smaller)]) != list(smaller):
        raise IntegrityError(f"{name} is not nested by prefix")


def validate_split_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SPLIT_SCHEMA:
        raise IntegrityError(f"not a {SPLIT_SCHEMA} manifest")
    verify_manifest_checksum(plan)
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise IntegrityError("split plan lacks evaluation identities")
    eval_reference = _ids(evaluation.get("reference_ids", []), name="evaluation reference IDs")
    eval_target = _ids(evaluation.get("target_ids", []), name="evaluation target IDs")
    if len(eval_reference) != len(eval_target) or len(eval_reference) != int(evaluation.get("size", -1)):
        raise IntegrityError("evaluation endpoint counts differ")
    if sha256_json(evaluation) != plan.get("evaluation_sha256"):
        raise IntegrityError("evaluation checksum mismatch")
    eval_reference_set, eval_target_set = set(eval_reference), set(eval_target)

    calibration_sizes = [int(value) for value in plan.get("calibration_sizes", [])]
    calibrations = plan.get("calibrations")
    if not isinstance(calibrations, Mapping) or sorted(calibration_sizes) != calibration_sizes:
        raise IntegrityError("calibration manifest is malformed")
    previous_reference: list[int | str] = []
    previous_target: list[int | str] = []
    for size in calibration_sizes:
        row = calibrations.get(str(size))
        if not isinstance(row, Mapping) or row.get("evaluation_sha256") != plan["evaluation_sha256"]:
            raise IntegrityError(f"calibration n={size} does not reference the fixed evaluation set")
        reference = _ids(row.get("reference_ids", []), name=f"calibration n={size} reference IDs")
        target = _ids(row.get("target_ids", []), name=f"calibration n={size} target IDs")
        if len(reference) != size or len(target) != size:
            raise IntegrityError(f"calibration n={size} has the wrong endpoint count")
        if eval_reference_set.intersection(reference) or eval_target_set.intersection(target):
            raise IntegrityError(f"calibration n={size} leaks fixed evaluation identities")
        _assert_prefix(previous_reference, reference, name="reference calibration")
        _assert_prefix(previous_target, target, name="target calibration")
        previous_reference, previous_target = reference, target

    pair_budgets = [int(value) for value in plan.get("pair_budgets", [])]
    pair_controls = plan.get("pair_controls")
    if not isinstance(pair_controls, Mapping) or sorted(pair_budgets) != pair_budgets:
        raise IntegrityError("pair-control manifest is malformed")
    previous_pair_reference: list[int | str] = []
    previous_pair_target: list[int | str] = []
    for budget in pair_budgets:
        row = pair_controls.get(str(budget))
        if not isinstance(row, Mapping) or row.get("evaluation_sha256") != plan["evaluation_sha256"]:
            raise IntegrityError(f"pair budget m={budget} does not reference fixed evaluation")
        controls = row.get("controls")
        if not isinstance(controls, Mapping) or not isinstance(controls.get("true"), Mapping):
            raise IntegrityError(f"pair budget m={budget} lacks a true pairing map")
        true = controls["true"]
        reference = _ids(true.get("reference_ids", []), name=f"true m={budget} reference IDs")
        target = _ids(true.get("target_ids", []), name=f"true m={budget} target IDs")
        if len(reference) != budget or len(target) != budget:
            raise IntegrityError(f"pair budget m={budget} has the wrong endpoint count")
        if true.get("mismatched_pairs") != 0:
            raise IntegrityError("true pairing map contains mismatches")
        if eval_reference_set.intersection(reference) or eval_target_set.intersection(target):
            raise IntegrityError(f"pair budget m={budget} leaks evaluation identities")
        _assert_prefix(previous_pair_reference, reference, name="true paired reference IDs")
        _assert_prefix(previous_pair_target, target, name="true paired target IDs")
        previous_pair_reference, previous_pair_target = reference, target
        if budget >= 2:
            shuffled = controls.get("shuffled")
            if not isinstance(shuffled, Mapping):
                raise IntegrityError(f"pair budget m={budget} lacks a shuffled control")
            shuffled_reference = _ids(
                shuffled.get("reference_ids", []), name=f"shuffled m={budget} reference IDs"
            )
            shuffled_target = _ids(
                shuffled.get("target_ids", []), name=f"shuffled m={budget} target IDs"
            )
            if shuffled_reference != reference or set(shuffled_target) != set(target):
                raise IntegrityError("shuffled control changed the estimator endpoint set")
            if any(left == right for left, right in zip(shuffled_target, target)):
                raise IntegrityError("shuffled control retained a true assignment")
            for noisy in controls.get("noisy", {}).values():
                noisy_reference = _ids(noisy.get("reference_ids", []), name="noisy reference IDs")
                noisy_target = _ids(noisy.get("target_ids", []), name="noisy target IDs")
                if noisy_reference != reference or set(noisy_target) != set(target):
                    raise IntegrityError("noisy control changed the estimator endpoint set")
                mismatches = sum(left != right for left, right in zip(noisy_target, target))
                if mismatches != int(noisy.get("mismatched_pairs", -1)):
                    raise IntegrityError("noisy control mismatch count is inconsistent")
