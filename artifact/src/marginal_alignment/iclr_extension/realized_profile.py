"""Post-hoc realized accuracy for a frozen correspondence profile.

The correspondence profile is deliberately constructed without target labels.
This module preserves that boundary with a two-stage API: first freeze the four
prespecified marginal-map predictions, then supply structural WNIDs only to the
scoring function.  It also provides a cluster-level association summary that
nests split repeats and encoders inside shifts instead of treating them as
independent observations.

The helpers are intended for an explicitly post-hoc, reviewer-requested audit.
They do not select a map, geometry, cost budget, or shift from outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import IntegrityError
from marginal_alignment.iclr_extension.frontier import (
    FrozenMarginalFrontier,
    frozen_frontier_predictions,
)


SCHEMA_VERSION = "marginal_alignment.iclr_extension.realized_profile.v1"
SCORE_SCHEMA_VERSION = "marginal_alignment.iclr_extension.realized_profile_score.v1"
ASSOCIATION_SCHEMA_VERSION = (
    "marginal_alignment.iclr_extension.realized_profile_association.v1"
)

# This order is part of the audit contract.  In particular, no post-hoc oracle
# or best-of-method score is constructed.
REALIZED_PROFILE_METHODS = ("none", "mean", "diagonal", "bures_oas")
_WNID = re.compile(r"n[0-9]{8}\Z")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(value: np.ndarray) -> str:
    """Hash an array with the optimization-certificate serialization rule."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(np.dtype(array.dtype).str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenRealizedProfilePredictions:
    """Opaque result of the label-free prediction-freeze stage.

    ``target_original_rows`` are administrative row indices, not labels.  The
    scoring stage uses them to select WNIDs from the full endpoint and thereby
    prevents a caller from accidentally pairing predictions with a differently
    ordered label vector.
    """

    dataset: str
    encoder: str
    split_seed: int
    target_endpoint_rows: int
    target_original_rows: np.ndarray
    class_count: int
    profile_binding_sha256: str
    target_witness_features_sha256: str
    target_original_rows_sha256: str
    bures_matrix_sha256: str
    bures_offset_sha256: str
    prediction_sha256: Mapping[str, str]
    predictions: Mapping[str, np.ndarray]
    phase: str = "fixed_method_predictions_frozen_before_target_wnids"

    def binding(self) -> dict[str, Any]:
        """Return the JSON-safe, outcome-free binding for an audit manifest."""

        return {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "scientific_result": False,
            "dataset": self.dataset,
            "encoder": self.encoder,
            "split_seed": self.split_seed,
            "phase": self.phase,
            "target_endpoint_rows": self.target_endpoint_rows,
            "witness_rows": int(len(self.target_original_rows)),
            "class_count": self.class_count,
            "profile_binding_sha256": self.profile_binding_sha256,
            "target_witness_features_sha256": (self.target_witness_features_sha256),
            "target_original_rows_sha256": self.target_original_rows_sha256,
            "bures_matrix_sha256": self.bures_matrix_sha256,
            "bures_offset_sha256": self.bures_offset_sha256,
            "prediction_sha256": dict(self.prediction_sha256),
            "methods": list(REALIZED_PROFILE_METHODS),
            "target_wnids_loaded": False,
            "labels_used_for_map_fitting": False,
            "labels_used_for_prediction_freeze": False,
            "pairing_used_for_map_fitting": False,
            "posthoc_method_selection": False,
        }


def freeze_realized_profile_predictions(
    frontier: FrozenMarginalFrontier,
    target_witness_features: np.ndarray,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    *,
    dataset: str,
    encoder: str,
    split_seed: int,
    target_endpoint_rows: int,
    target_witness_original_rows: np.ndarray,
    profile_binding_sha256: str,
    expected_bures_matrix_sha256: str | None = None,
    expected_bures_offset_sha256: str | None = None,
    expected_witness_rows: int | None = None,
) -> FrozenRealizedProfilePredictions:
    """Freeze four fixed-method predictions without accepting labels or WNIDs.

    For an existing profile, callers should provide the expected Bures matrix
    and offset hashes from its authenticated geometry.  A newly constructed
    profile may omit them, but its own pre-label artifact hash remains mandatory
    through ``profile_binding_sha256``.
    """

    if not isinstance(dataset, str) or not dataset:
        raise IntegrityError("realized-profile dataset must be nonempty text")
    if not isinstance(encoder, str) or not encoder:
        raise IntegrityError("realized-profile encoder must be nonempty text")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise IntegrityError("realized-profile split seed must be an integer")
    if not _valid_sha256(profile_binding_sha256):
        raise IntegrityError("profile binding must be a lowercase SHA-256")

    diagnostics = frontier.diagnostics
    if (
        diagnostics.get("labels_used") is not False
        or diagnostics.get("pairing_used") is not False
        or diagnostics.get("information_scope") != "unpaired_calibration_features_only"
    ):
        raise IntegrityError(
            "marginal frontier does not authenticate label-free unpaired fitting"
        )
    if not set(REALIZED_PROFILE_METHODS).issubset(frontier.maps):
        raise IntegrityError("marginal frontier lacks a fixed requested method")

    target = np.asarray(target_witness_features, dtype=np.float64)
    weight = np.asarray(readout_weight, dtype=np.float64)
    bias = np.asarray(readout_bias, dtype=np.float64)
    if (
        target.ndim != 2
        or target.shape[0] < 1
        or target.shape[1] < 1
        or not np.all(np.isfinite(target))
    ):
        raise IntegrityError("target witness features must be a finite matrix")
    if (
        weight.ndim != 2
        or weight.shape[0] < 2
        or weight.shape[1] != target.shape[1]
        or bias.shape != (weight.shape[0],)
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(bias))
    ):
        raise IntegrityError("frozen readout has an invalid shape or value")

    if not isinstance(target_endpoint_rows, int) or isinstance(
        target_endpoint_rows, bool
    ):
        raise IntegrityError("target endpoint row count must be an integer")
    endpoint_rows = target_endpoint_rows
    original_rows = np.asarray(target_witness_original_rows)
    if (
        endpoint_rows < len(target)
        or original_rows.shape != (len(target),)
        or not np.issubdtype(original_rows.dtype, np.integer)
    ):
        raise IntegrityError("target witness row binding has an invalid shape")
    original_rows = original_rows.astype(np.int64, copy=False)
    if (
        np.any(original_rows < 0)
        or np.any(original_rows >= endpoint_rows)
        or len(np.unique(original_rows)) != len(original_rows)
    ):
        raise IntegrityError("target witness row binding is out of range or repeated")
    if expected_witness_rows is not None and len(target) != int(expected_witness_rows):
        raise IntegrityError("target witness row count differs from the audit contract")

    bures = frontier.maps["bures_oas"]
    matrix = getattr(bures, "matrix", None)
    offset = getattr(bures, "offset", None)
    if matrix is None or offset is None:
        raise IntegrityError("Bures frontier map is not an authenticated affine map")
    matrix_sha256 = _array_sha256(np.asarray(matrix))
    offset_sha256 = _array_sha256(np.asarray(offset))
    one_expected_hash = (
        expected_bures_matrix_sha256 is not None
        or expected_bures_offset_sha256 is not None
    )
    if one_expected_hash:
        if not (
            _valid_sha256(expected_bures_matrix_sha256)
            and _valid_sha256(expected_bures_offset_sha256)
        ):
            raise IntegrityError("expected Bures hashes must be supplied as a pair")
        if (
            matrix_sha256 != expected_bures_matrix_sha256
            or offset_sha256 != expected_bures_offset_sha256
        ):
            raise IntegrityError("Bures map differs from the frozen profile geometry")

    all_predictions = frozen_frontier_predictions(frontier, target, weight, bias)
    predictions: dict[str, np.ndarray] = {}
    prediction_hashes: dict[str, str] = {}
    for method in REALIZED_PROFILE_METHODS:
        value = np.asarray(all_predictions.get(method))
        if (
            value.shape != (len(target),)
            or not np.issubdtype(value.dtype, np.integer)
            or np.any(value < 0)
            or np.any(value >= weight.shape[0])
        ):
            raise IntegrityError(f"invalid frozen prediction vector for {method}")
        immutable = _immutable_array(value, dtype=np.dtype(np.int64))
        predictions[method] = immutable
        prediction_hashes[method] = _array_sha256(immutable)

    immutable_rows = _immutable_array(original_rows, dtype=np.dtype(np.int64))
    return FrozenRealizedProfilePredictions(
        dataset=dataset,
        encoder=encoder,
        split_seed=split_seed,
        target_endpoint_rows=endpoint_rows,
        target_original_rows=immutable_rows,
        class_count=int(weight.shape[0]),
        profile_binding_sha256=profile_binding_sha256,
        target_witness_features_sha256=_array_sha256(target),
        target_original_rows_sha256=_array_sha256(immutable_rows),
        bures_matrix_sha256=matrix_sha256,
        bures_offset_sha256=offset_sha256,
        prediction_sha256=MappingProxyType(prediction_hashes),
        predictions=MappingProxyType(predictions),
    )


def _validated_class_index(
    class_index_wnids: Sequence[str],
    *,
    expected_classes: int | None,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in class_index_wnids)
    if (
        len(values) < 2
        or len(set(values)) != len(values)
        or any(_WNID.fullmatch(value) is None for value in values)
    ):
        raise IntegrityError("official class-index WNIDs are malformed or repeated")
    if expected_classes is not None and len(values) != int(expected_classes):
        raise IntegrityError(
            "official class-index cardinality differs from the contract"
        )
    return values


def score_frozen_realized_profile(
    frozen: FrozenRealizedProfilePredictions,
    target_endpoint_wnids: np.ndarray,
    class_index_wnids: Sequence[str],
    *,
    expected_classes: int | None = 1000,
) -> dict[str, Any]:
    """Map WNIDs to head rows and score predictions after the freeze stage.

    The full class index is required even when the target supports only a
    subset (for example, ImageNet-R).  Predictions outside that subset remain
    wrong; this function never restricts or renormalizes the frozen head.
    """

    if not isinstance(frozen, FrozenRealizedProfilePredictions):
        raise TypeError("scoring requires a frozen realized-profile object")
    if frozen.phase != "fixed_method_predictions_frozen_before_target_wnids":
        raise IntegrityError("realized-profile prediction phase is not frozen")
    original_rows = np.asarray(frozen.target_original_rows)
    if (
        original_rows.ndim != 1
        or not np.issubdtype(original_rows.dtype, np.integer)
        or np.any(original_rows < 0)
        or np.any(original_rows >= frozen.target_endpoint_rows)
        or len(np.unique(original_rows)) != len(original_rows)
    ):
        raise IntegrityError("frozen target row binding is malformed")
    if _array_sha256(original_rows) != frozen.target_original_rows_sha256:
        raise IntegrityError("frozen target row binding changed before scoring")
    if (
        tuple(frozen.predictions) != REALIZED_PROFILE_METHODS
        or tuple(frozen.prediction_sha256) != REALIZED_PROFILE_METHODS
    ):
        raise IntegrityError("frozen prediction inventory differs from the contract")
    class_wnids = _validated_class_index(
        class_index_wnids, expected_classes=expected_classes
    )
    if len(class_wnids) != frozen.class_count:
        raise IntegrityError(
            "full class index differs from the frozen head output space"
        )

    raw_wnids = np.asarray(target_endpoint_wnids)
    if raw_wnids.ndim != 1 or len(raw_wnids) != frozen.target_endpoint_rows:
        raise IntegrityError("target endpoint WNIDs do not cover the frozen endpoint")
    endpoint_values = [str(value) for value in raw_wnids.tolist()]
    if any(_WNID.fullmatch(value) is None for value in endpoint_values):
        raise IntegrityError("target endpoint contains a malformed WNID")
    endpoint_wnids = np.asarray(endpoint_values, dtype="<U9")
    selected_wnids = endpoint_wnids[original_rows]
    index_by_wnid = {wnid: index for index, wnid in enumerate(class_wnids)}
    unknown = sorted(set(selected_wnids.tolist()).difference(index_by_wnid))
    if unknown:
        raise IntegrityError(
            f"target WNID is outside the full class index: {unknown[:3]}"
        )
    labels = np.asarray(
        [index_by_wnid[value] for value in selected_wnids.tolist()], dtype=np.int64
    )

    method_correct: dict[str, np.ndarray] = {}
    method_accuracy: dict[str, float] = {}
    for method in REALIZED_PROFILE_METHODS:
        prediction = frozen.predictions[method]
        if _array_sha256(prediction) != frozen.prediction_sha256[method]:
            raise IntegrityError("a frozen prediction vector changed before scoring")
        correct = np.asarray(prediction == labels, dtype=bool)
        method_correct[method] = correct
        method_accuracy[method] = float(np.mean(correct))

    per_wnid: list[dict[str, Any]] = []
    for class_index, wnid in enumerate(class_wnids):
        mask = selected_wnids == wnid
        rows = int(np.count_nonzero(mask))
        if rows:
            per_wnid.append(
                {
                    "wnid": wnid,
                    "class_index": class_index,
                    "rows": rows,
                    "method_correct_counts": {
                        method: int(np.count_nonzero(method_correct[method][mask]))
                        for method in REALIZED_PROFILE_METHODS
                    },
                }
            )
    if sum(row["rows"] for row in per_wnid) != len(labels):
        raise IntegrityError("post-hoc WNID sufficient statistics do not close")

    gains = {
        f"{method}_minus_none": method_accuracy[method] - method_accuracy["none"]
        for method in REALIZED_PROFILE_METHODS[1:]
    }
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_requested",
        "dataset": frozen.dataset,
        "encoder": frozen.encoder,
        "split_seed": frozen.split_seed,
        "rows": int(len(labels)),
        "class_count": frozen.class_count,
        "target_supported_class_count": len(set(selected_wnids.tolist())),
        "profile_binding_sha256": frozen.profile_binding_sha256,
        "prediction_freeze_binding": frozen.binding(),
        "target_endpoint_wnids_sha256": _array_sha256(endpoint_wnids),
        "selected_target_wnids_sha256": _array_sha256(selected_wnids),
        "selected_target_labels_sha256": _array_sha256(labels),
        "methods": list(REALIZED_PROFILE_METHODS),
        "per_method_target_accuracy": method_accuracy,
        "fixed_method_gains": gains,
        "bures_minus_none_target_accuracy": gains["bures_oas_minus_none"],
        "per_wnid_sufficient_statistics": per_wnid,
        "evaluation_phase": "after_profile_map_and_predictions_frozen",
        "target_wnids_used_for_map_fitting": False,
        "target_wnids_used_for_prediction_freeze": False,
        "target_wnids_used_for_posthoc_scoring": True,
        "labels_used_for_map_fitting": False,
        "pairing_used_for_map_fitting": False,
        "full_frozen_head_retained": True,
        "posthoc_method_selection": False,
    }


def profile_gain_record(
    score: Mapping[str, Any],
    *,
    cluster: str,
    family: str,
    profile_disagreement: float,
    profile_binding_sha256: str,
) -> dict[str, Any]:
    """Bind one frozen profile scalar to its sample-matched realized gain."""

    if (
        score.get("schema_version") != SCORE_SCHEMA_VERSION
        or score.get("status") != "PASS"
        or score.get("evaluation_phase") != "after_profile_map_and_predictions_frozen"
        or score.get("labels_used_for_map_fitting") is not False
        or score.get("target_wnids_used_for_prediction_freeze") is not False
        or score.get("posthoc_method_selection") is not False
    ):
        raise IntegrityError("realized score lacks the required information boundary")
    if not isinstance(cluster, str) or not cluster:
        raise IntegrityError("association cluster must be nonempty text")
    if not isinstance(family, str) or not family:
        raise IntegrityError("association family must be nonempty text")
    if (
        not _valid_sha256(profile_binding_sha256)
        or score.get("profile_binding_sha256") != profile_binding_sha256
    ):
        raise IntegrityError("score and profile binding differ")
    disagreement = float(profile_disagreement)
    gain = float(score.get("bures_minus_none_target_accuracy", math.nan))
    if not math.isfinite(disagreement) or not 0.0 <= disagreement <= 1.0:
        raise IntegrityError("profile disagreement must lie in [0, 1]")
    if not math.isfinite(gain) or not -1.0 <= gain <= 1.0:
        raise IntegrityError("realized Bures gain must lie in [-1, 1]")
    return {
        "cluster": cluster,
        "family": family,
        "dataset": str(score.get("dataset")),
        "encoder": str(score.get("encoder")),
        "split_seed": int(score.get("split_seed")),
        "profile_disagreement": disagreement,
        "profile_ceiling": 1.0 - disagreement / 2.0,
        "bures_gain": gain,
        "profile_binding_sha256": profile_binding_sha256,
        "profile_frozen_before_scoring": True,
        "predictions_frozen_before_target_wnids": True,
        "labels_used_for_map_fitting": False,
        "posthoc_method_selection": False,
    }


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise IntegrityError("association inputs contain no finite values")
    return float(np.mean(array))


def aggregate_profile_gain_clusters(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_encoders: Sequence[str] | None = None,
    expected_split_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Average seeds, then encoders, to make shifts the independent units."""

    if not records:
        raise IntegrityError("profile-gain association has no records")
    encoder_contract = (
        None if expected_encoders is None else tuple(map(str, expected_encoders))
    )
    seed_contract = (
        None
        if expected_split_seeds is None
        else tuple(int(value) for value in expected_split_seeds)
    )
    if encoder_contract is not None and len(set(encoder_contract)) != len(
        encoder_contract
    ):
        raise IntegrityError("expected encoders contain duplicates")
    if seed_contract is not None and len(set(seed_contract)) != len(seed_contract):
        raise IntegrityError("expected split seeds contain duplicates")

    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    cluster_family: dict[str, str] = {}
    for record in records:
        if (
            record.get("profile_frozen_before_scoring") is not True
            or record.get("predictions_frozen_before_target_wnids") is not True
            or record.get("labels_used_for_map_fitting") is not False
            or record.get("posthoc_method_selection") is not False
        ):
            raise IntegrityError("association record violates the information boundary")
        cluster = str(record.get("cluster", ""))
        family = str(record.get("family", ""))
        encoder = str(record.get("encoder", ""))
        seed = record.get("split_seed")
        if (
            not cluster
            or not family
            or not encoder
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not _valid_sha256(record.get("profile_binding_sha256"))
        ):
            raise IntegrityError("association identity fields are malformed")
        previous_family = cluster_family.setdefault(cluster, family)
        if previous_family != family:
            raise IntegrityError("one shift cluster appears in multiple families")
        key = (cluster, encoder)
        if seed in grouped[key]:
            raise IntegrityError("duplicate cluster/encoder/split-seed record")
        disagreement = float(record.get("profile_disagreement", math.nan))
        gain = float(record.get("bures_gain", math.nan))
        if (
            not math.isfinite(disagreement)
            or not 0.0 <= disagreement <= 1.0
            or not math.isfinite(gain)
            or not -1.0 <= gain <= 1.0
        ):
            raise IntegrityError("association values are outside their valid ranges")
        grouped[key][seed] = record

    observed_encoders = tuple(sorted({key[1] for key in grouped}))
    observed_seeds = tuple(sorted({seed for rows in grouped.values() for seed in rows}))
    encoders = observed_encoders if encoder_contract is None else encoder_contract
    seeds = observed_seeds if seed_contract is None else seed_contract
    if encoder_contract is not None and set(observed_encoders) != set(encoder_contract):
        raise IntegrityError("observed encoders differ from the association contract")
    if seed_contract is not None and set(observed_seeds) != set(seed_contract):
        raise IntegrityError(
            "observed split seeds differ from the association contract"
        )
    clusters = tuple(sorted(cluster_family))
    if not encoders or not seeds:
        raise IntegrityError("association panel has an empty encoder or seed axis")

    cluster_results: list[dict[str, Any]] = []
    for cluster in clusters:
        cell_results: list[dict[str, Any]] = []
        for encoder in encoders:
            by_seed = grouped.get((cluster, encoder), {})
            if set(by_seed) != set(seeds):
                raise IntegrityError(
                    "association panel is not complete over split seeds and encoders"
                )
            cell_results.append(
                {
                    "encoder": encoder,
                    "split_seeds": list(seeds),
                    "profile_disagreement_mean": _mean(
                        by_seed[seed]["profile_disagreement"] for seed in seeds
                    ),
                    "bures_gain_mean": _mean(
                        by_seed[seed]["bures_gain"] for seed in seeds
                    ),
                }
            )
        cluster_disagreement = _mean(
            value["profile_disagreement_mean"] for value in cell_results
        )
        cluster_results.append(
            {
                "cluster": cluster,
                "family": cluster_family[cluster],
                "encoder_cells": cell_results,
                "encoders_nested": list(encoders),
                "split_seeds_nested": list(seeds),
                "profile_disagreement_mean": cluster_disagreement,
                "profile_ceiling_mean": 1.0 - cluster_disagreement / 2.0,
                "bures_gain_mean": _mean(
                    value["bures_gain_mean"] for value in cell_results
                ),
            }
        )
    return cluster_results


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        return None
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _percentile_interval(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"defined": False, "lower": None, "upper": None}
    return {
        "defined": True,
        "lower": float(np.quantile(array, 0.025)),
        "upper": float(np.quantile(array, 0.975)),
    }


def cluster_aware_profile_gain_association(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_encoders: Sequence[str] | None = None,
    expected_split_seeds: Sequence[int] | None = None,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 10_000,
    rng_seed: int = 20260910,
) -> dict[str, Any]:
    """Summarize profile/gain association at the shift-cluster level.

    Bootstrap resampling is stratified by shift family.  The Monte Carlo null
    permutes gains among whole clusters *within* family.  Both encoders and all
    split repeats have already been averaged inside a cluster; neither is
    counted as an independent sample.
    """

    clusters = aggregate_profile_gain_clusters(
        records,
        expected_encoders=expected_encoders,
        expected_split_seeds=expected_split_seeds,
    )
    if len(clusters) < 3:
        raise IntegrityError("at least three independent shift clusters are required")
    bootstrap_count = int(bootstrap_replicates)
    permutation_count = int(permutation_replicates)
    if bootstrap_count < 0 or permutation_count < 0:
        raise IntegrityError("resampling replicate counts must be nonnegative")
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise IntegrityError("association RNG seed must be an integer")

    disagreement = [row["profile_disagreement_mean"] for row in clusters]
    ceiling = [row["profile_ceiling_mean"] for row in clusters]
    gain = [row["bures_gain_mean"] for row in clusters]
    point = _spearman(disagreement, gain)
    ceiling_point = _spearman(ceiling, gain)
    if point is None or ceiling_point is None:
        raise IntegrityError("cluster-level profile or gain is constant")

    leave_one_out: list[float] = []
    if len(clusters) >= 4:
        for omitted in range(len(clusters)):
            value = _spearman(
                disagreement[:omitted] + disagreement[omitted + 1 :],
                gain[:omitted] + gain[omitted + 1 :],
            )
            if value is not None:
                leave_one_out.append(value)

    rng = np.random.default_rng(rng_seed)
    family_indices: dict[str, np.ndarray] = {}
    for family in sorted({row["family"] for row in clusters}):
        family_indices[family] = np.asarray(
            [index for index, row in enumerate(clusters) if row["family"] == family],
            dtype=np.int64,
        )

    bootstrap_values: list[float] = []
    for _ in range(bootstrap_count):
        selected: list[int] = []
        for indices in family_indices.values():
            selected.extend(
                rng.choice(indices, size=len(indices), replace=True).tolist()
            )
        value = _spearman(
            [disagreement[index] for index in selected],
            [gain[index] for index in selected],
        )
        if value is not None:
            bootstrap_values.append(value)

    permutation_values: list[float] = []
    for _ in range(permutation_count):
        permuted_gain = list(gain)
        for indices in family_indices.values():
            values = [gain[index] for index in indices]
            shuffled = rng.permutation(values).tolist()
            for index, value in zip(indices.tolist(), shuffled):
                permuted_gain[index] = float(value)
        value = _spearman(disagreement, permuted_gain)
        if value is not None:
            permutation_values.append(value)
    permutation_p = (
        (1.0 + sum(abs(value) >= abs(point) for value in permutation_values))
        / (1.0 + len(permutation_values))
        if permutation_values
        else None
    )

    return {
        "schema_version": ASSOCIATION_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_requested",
        "independent_unit": "shift_or_corruption_cluster",
        "cluster_count": len(clusters),
        "family_counts": {
            family: int(len(indices)) for family, indices in family_indices.items()
        },
        "encoders_and_split_seeds_nested_within_cluster": True,
        "cluster_rows": clusters,
        "primary_statistic": "spearman_profile_disagreement_vs_bures_gain",
        "spearman_profile_disagreement_vs_bures_gain": point,
        "spearman_profile_ceiling_vs_bures_gain": ceiling_point,
        "ceiling_is_deterministic_monotone_transform_not_separate_evidence": True,
        "leave_one_cluster_out": {
            "defined_replicates": len(leave_one_out),
            "minimum": min(leave_one_out) if leave_one_out else None,
            "maximum": max(leave_one_out) if leave_one_out else None,
        },
        "cluster_bootstrap": {
            "stratified_by_family": True,
            "attempted_replicates": bootstrap_count,
            "defined_replicates": len(bootstrap_values),
            "dropped_undefined_replicates": bootstrap_count - len(bootstrap_values),
            "interval": "percentile_95",
            **_percentile_interval(bootstrap_values),
        },
        "cluster_permutation": {
            "within_family": True,
            "attempted_replicates": permutation_count,
            "defined_replicates": len(permutation_values),
            "two_sided_plus_one_p_value": permutation_p,
            "validity_requires_cluster_exchangeability_within_family": True,
        },
        "rng": "numpy.random.Generator(PCG64)",
        "rng_seed": rng_seed,
        "interpretation": (
            "exploratory cluster-level association; not a calibrated deployment "
            "rule and not evidence from independent encoder or split-seed samples"
        ),
    }
