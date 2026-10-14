"""Strict prequential streaming audit for frozen marginal alignment.

The protocol is intentionally secondary to the fixed-endpoint campaign.  A
block is always predicted with target statistics from *earlier* blocks; only
after every strategy has been evaluated is the current block admitted to the
cumulative Welford state and fixed FIFO windows.  Clean/reference counterparts
are used solely for held-out evaluation and never for fitting target state.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_alignment.linalg import (
    gram_square_root, stable_hash_arrays, sym, symmetric_power,
)

from .artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
    write_json_atomic,
)
from .feature_store import FeatureStore, load_feature_store
from .maps import AffineMap
from .metrics import evaluate_alignment


STREAMING_SCHEMA = "marginal_alignment.campaign.streaming_plan.v1"
STREAMING_WINDOWS = (64, 256, 1024)
STREAMING_METHODS = ("none", "mean", "diagonal", "bures_oas")
MINIMUM_ADAPTIVE_HISTORY = 2
AGGREGATE_METRICS = (
    "aligned_mean_l2",
    "aligned_coordinate_variance_relative_error",
    "aligned_covariance_relative_error",
    "aligned_correlation_relative_error",
    "aligned_coordinate_w2_squared",
    "aligned_sliced_w2_squared",
    "aligned_rff_mmd_squared",
    "paired_feature_mse",
    "centered_logit_mse",
    "reference_prediction_agreement",
    "aligned_accuracy",
    "negative_log_likelihood",
    "brier_score",
    "expected_calibration_error",
)


def _id_value(value: Any) -> int | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError(f"stream IDs must be integers or strings, got {type(value).__name__}")
    return value


def _ids(values: Sequence[Any] | np.ndarray, *, name: str) -> list[int | str]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.hasobject:
        raise IntegrityError(f"{name} must be a safe rank-one ID array")
    result = [_id_value(value) for value in array]
    if len(result) != len(set(result)):
        raise IntegrityError(f"{name} contains duplicate IDs")
    return result


def _id_hash(values: Sequence[int | str], *, namespace: str) -> str:
    return sha256_json({"namespace": namespace, "values": list(values)})


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _lookup_ids(universe: np.ndarray, selected: Sequence[int | str], *, name: str) -> np.ndarray:
    values = _ids(universe, name=f"{name} universe")
    lookup = {value: index for index, value in enumerate(values)}
    try:
        return np.asarray([lookup[value] for value in selected], dtype=np.int64)
    except KeyError as error:
        raise IntegrityError(f"{name} references absent ID {error.args[0]!r}") from error


def _strategy_grid() -> list[dict[str, Any]]:
    result = [{"strategy": "none", "method": "none", "history_mode": "none",
               "window_size": None}]
    for method in STREAMING_METHODS[1:]:
        result.append({
            "strategy": f"{method}__cumulative",
            "method": method,
            "history_mode": "cumulative_welford",
            "window_size": None,
        })
        for window in STREAMING_WINDOWS:
            result.append({
                "strategy": f"{method}__window{window}",
                "method": method,
                "history_mode": "fixed_window",
                "window_size": int(window),
            })
    return result


@dataclass
class WelfordMoments:
    """Mergeable population mean/covariance state without retaining samples."""

    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, dimension: int) -> "WelfordMoments":
        if int(dimension) < 1:
            raise ValueError("feature dimension must be positive")
        return cls(0, np.zeros(int(dimension)), np.zeros((int(dimension), int(dimension))))

    @classmethod
    def from_array(cls, values: np.ndarray) -> "WelfordMoments":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
            raise ValueError("Welford input must be a nonempty finite matrix")
        state = cls.empty(array.shape[1])
        state.update(array)
        return state

    def copy(self) -> "WelfordMoments":
        return WelfordMoments(self.count, self.mean.copy(), self.m2.copy())

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.mean) or not len(array):
            raise ValueError("Welford update has an invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError("Welford update contains non-finite features")
        batch_count = len(array)
        batch_mean = array.mean(0)
        centered = array - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = sym(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 = sym(
            self.m2 + batch_m2
            + np.outer(delta, delta) * (self.count * batch_count / total)
        )
        self.mean = self.mean + delta * (batch_count / total)
        self.count = total

    @property
    def covariance(self) -> np.ndarray:
        if self.count < 1:
            raise ValueError("empty Welford state has no covariance")
        return sym(self.m2 / self.count)

    @property
    def content_hash(self) -> str:
        return sha256_json({
            "count": self.count,
            "arrays_sha256": stable_hash_arrays(self.mean, self.m2),
        })


def _oas_covariance(state: WelfordMoments, relative_floor: float = 1e-6
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    covariance = state.covariance
    dimension = covariance.shape[0]
    mu = float(np.trace(covariance) / dimension)
    alpha = float(np.mean(np.square(covariance)))
    denominator = (state.count + 1.0) * (alpha - (mu * mu) / dimension)
    shrinkage = 1.0 if denominator <= 0.0 else min(
        (alpha + mu * mu) / denominator, 1.0
    )
    covariance = sym(
        (1.0 - shrinkage) * covariance + shrinkage * mu * np.eye(dimension)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.trace(covariance) / dimension), np.finfo(float).eps)
    floor = max(float(relative_floor) * scale, np.finfo(float).eps)
    clipped = np.maximum(eigenvalues, floor)
    regularized = sym((eigenvectors * clipped) @ eigenvectors.T)
    return regularized, {
        "covariance_estimator": "oas",
        "shrinkage": float(shrinkage),
        "eigenvalue_floor": float(floor),
        "clipped_eigenvalues": int(np.sum(eigenvalues < floor)),
    }


def _fit_from_moments(
    method: str,
    source: WelfordMoments,
    reference: WelfordMoments,
    *,
    standard_deviation_floor: float = 1e-6,
    reference_oas: tuple[np.ndarray, Mapping[str, Any]] | None = None,
) -> AffineMap:
    if source.count < MINIMUM_ADAPTIVE_HISTORY:
        raise ValueError("adaptive maps require at least two previous observations")
    dimension = len(reference.mean)
    if method == "mean":
        matrix = np.eye(dimension)
        diagnostics: dict[str, Any] = {}
    elif method == "diagonal":
        source_std_raw = np.sqrt(np.maximum(np.diag(source.covariance), 0.0))
        reference_std_raw = np.sqrt(np.maximum(np.diag(reference.covariance), 0.0))
        source_std = np.maximum(source_std_raw, standard_deviation_floor)
        reference_std = np.maximum(reference_std_raw, standard_deviation_floor)
        matrix = np.diag(reference_std / source_std)
        diagnostics = {
            "standard_deviation_floor": float(standard_deviation_floor),
            "floored_source_dimensions": int(np.sum(source_std_raw < standard_deviation_floor)),
            "floored_reference_dimensions": int(
                np.sum(reference_std_raw < standard_deviation_floor)
            ),
        }
    elif method == "bures_oas":
        source_covariance, source_diagnostics = _oas_covariance(source)
        if reference_oas is None:
            reference_covariance, reference_diagnostics = _oas_covariance(reference)
        else:
            reference_covariance = np.asarray(reference_oas[0], dtype=np.float64)
            reference_diagnostics = dict(reference_oas[1])
        source_sqrt = symmetric_power(source_covariance, 0.5)
        source_inverse_sqrt = symmetric_power(source_covariance, -0.5)
        reference_sqrt = symmetric_power(reference_covariance, 0.5)
        middle_sqrt, bures_sqrt_diagnostics = gram_square_root(
            reference_sqrt @ source_sqrt
        )
        matrix = sym(
            source_inverse_sqrt @ middle_sqrt @ source_inverse_sqrt
        )
        diagnostics = {
            "source_covariance": source_diagnostics,
            "reference_covariance": reference_diagnostics,
            "bures_middle_square_root": bures_sqrt_diagnostics,
            "gauge": "bures",
        }
    else:
        raise ValueError(f"unsupported streaming method: {method}")
    offset = reference.mean - source.mean @ matrix
    diagnostics.update({
        "source_history_rows": source.count,
        "reference_rows": reference.count,
        "source_state_sha256": source.content_hash,
        "reference_state_sha256": reference.content_hash,
    })
    return AffineMap(np.asarray(matrix), np.asarray(offset), method, diagnostics)


def _head(store: FeatureStore) -> tuple[np.ndarray, np.ndarray, str]:
    declaration = store.manifest.get("head")
    if not isinstance(declaration, Mapping):
        raise IntegrityError("streaming store lacks a frozen head declaration")
    weight_role, bias_role = declaration.get("weight_role"), declaration.get("bias_role")
    if not weight_role or not bias_role:
        raise IntegrityError("streaming requires frozen weight and bias roles")
    weight = np.asarray(store.array(str(weight_role)), dtype=np.float64)
    bias = np.asarray(store.array(str(bias_role)), dtype=np.float64)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise IntegrityError("frozen streaming head has invalid shapes")
    return weight, bias, stable_hash_arrays(weight, bias)


def build_streaming_plan(
    store_path: str | Path,
    *,
    target_order_ids: Sequence[Any] | np.ndarray,
    reference_calibration_ids: Sequence[Any] | np.ndarray,
    order_provenance: Mapping[str, Any],
    block_size: int = 256,
    metric_seed: int = 20260812,
    metric_projections: int = 16,
    mmd_features: int = 128,
    distribution_limit: int = 256,
    execution_provenance: Mapping[str, Any] | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Seal a store-bound stream order and disjoint frozen reference subset."""

    store = load_feature_store(store_path, verify_hashes=True)
    if not store.hashes_verified:
        raise IntegrityError("streaming refuses an unverified feature store")
    if order_provenance.get("verified") is not True:
        raise IntegrityError("stream order provenance must be explicitly verified")
    if not order_provenance.get("kind") or not order_provenance.get("source"):
        raise IntegrityError("stream order provenance requires kind and source")
    target = store.endpoint("target")
    reference = store.endpoint("reference")
    target_order = _ids(np.asarray(target_order_ids), name="locked target order")
    reference_ids = _ids(
        np.asarray(reference_calibration_ids), name="reference calibration IDs"
    )
    _lookup_ids(target.ids, target_order, name="locked target order")
    reference_index = _lookup_ids(
        reference.ids, reference_ids, name="reference calibration IDs"
    )
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError("stream block_size must be at least two")
    if len(target_order) <= block_size:
        raise ValueError("stream order must contain at least two prediction blocks")
    if len(target_order) % block_size == 1:
        raise ValueError("the final streaming block may not be a singleton")
    if len(reference_ids) < 2:
        raise ValueError("frozen reference calibration needs at least two rows")
    if min(int(metric_projections), int(mmd_features), int(distribution_limit)) < 2:
        raise ValueError("stream metric budgets must be at least two")
    execution_provenance = dict(execution_provenance or {})
    for key in ("config_sha256", "source_sha256", "runtime_sha256"):
        value = execution_provenance.get(key)
        if execution_provenance and (
            not isinstance(value, str) or len(value) != 64
        ):
            raise IntegrityError(f"stream execution provenance lacks exact {key}")

    paired_reference, paired_target = store.pairing().validated()
    target_to_reference = dict(zip(paired_target, paired_reference))
    try:
        evaluation_reference = [target_to_reference[value] for value in target_order]
    except KeyError as error:
        raise IntegrityError(
            f"locked target stream lacks verified evaluation counterpart {error.args[0]!r}"
        ) from error
    overlap = set(reference_ids).intersection(evaluation_reference)
    if overlap:
        raise IntegrityError(
            "frozen reference calibration overlaps clean evaluation counterparts: "
            f"{sorted(overlap, key=str)[:3]}"
        )

    model = store.manifest["metadata"]["model_provenance"]
    head_fit_ids = set(_ids(
        store.array(str(model["head_fit_ids_role"])), name="head-fit IDs"
    ))
    head_namespace = str(model["head_fit_id_namespace"])
    if head_namespace == reference.id_namespace and head_fit_ids.intersection(reference_ids):
        raise IntegrityError("reference calibration overlaps frozen-head fitting IDs")
    if head_namespace == target.id_namespace and head_fit_ids.intersection(target_order):
        raise IntegrityError("target stream overlaps frozen-head fitting IDs")

    reference_features = np.asarray(reference.features[reference_index], dtype=np.float64)
    weight, bias, head_hash = _head(store)
    plan = seal_manifest({
        "schema_version": STREAMING_SCHEMA,
        "status": "PASS",
        "feature_store_manifest": str(store.manifest_path),
        "feature_store_manifest_sha256": sha256_file(store.manifest_path),
        "dataset": store.dataset,
        "encoder": store.encoder,
        "target_namespace": target.id_namespace,
        "reference_namespace": reference.id_namespace,
        "target_order_ids": target_order,
        "target_order_sha256": _id_hash(target_order, namespace=target.id_namespace),
        "reference_calibration_ids": reference_ids,
        "reference_calibration_ids_sha256": _id_hash(
            reference_ids, namespace=reference.id_namespace
        ),
        "reference_calibration_features_sha256": stable_hash_arrays(reference_features),
        "evaluation_reference_ids_sha256": _id_hash(
            evaluation_reference, namespace=reference.id_namespace
        ),
        "order_provenance": dict(order_provenance),
        "execution_provenance": execution_provenance,
        "block_size": block_size,
        "block_count": int(math.ceil(len(target_order) / block_size)),
        "stream_rows": len(target_order),
        "minimum_adaptive_history": MINIMUM_ADAPTIVE_HISTORY,
        "windows": list(STREAMING_WINDOWS),
        "methods": list(STREAMING_METHODS),
        "strategies": _strategy_grid(),
        "metric_seed": int(metric_seed),
        "metric_projections": int(metric_projections),
        "mmd_features": int(mmd_features),
        "distribution_limit": int(distribution_limit),
        "frozen_head_value_sha256": head_hash,
        "protocol": "strict_predict_before_update",
    })
    if destination is not None:
        write_json_atomic(destination, plan)
    return plan


def _load_plan(plan: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(plan) if isinstance(plan, (str, Path)) else dict(plan)
    if value.get("schema_version") != STREAMING_SCHEMA or value.get("status") != "PASS":
        raise IntegrityError("invalid or non-passing streaming plan")
    verify_manifest_checksum(value)
    if tuple(value.get("windows", ())) != STREAMING_WINDOWS:
        raise IntegrityError("streaming windows differ from the locked 64/256/1024 grid")
    if tuple(value.get("methods", ())) != STREAMING_METHODS:
        raise IntegrityError("streaming methods differ from the locked method grid")
    if value.get("strategies") != _strategy_grid():
        raise IntegrityError("streaming strategy grid is not exact")
    if value.get("protocol") != "strict_predict_before_update":
        raise IntegrityError("streaming plan is not predict-before-update")
    target_order = [_id_value(item) for item in value.get("target_order_ids", [])]
    reference_ids = [
        _id_value(item) for item in value.get("reference_calibration_ids", [])
    ]
    block_size = int(value.get("block_size", 0))
    if block_size < 2 or len(target_order) <= block_size:
        raise IntegrityError("streaming plan has invalid block/order dimensions")
    if len(target_order) % block_size == 1:
        raise IntegrityError("streaming plan contains a singleton final block")
    if int(value.get("stream_rows", -1)) != len(target_order):
        raise IntegrityError("stream_rows disagrees with the locked order")
    if int(value.get("block_count", -1)) != math.ceil(len(target_order) / block_size):
        raise IntegrityError("block_count disagrees with the locked order")
    if int(value.get("minimum_adaptive_history", -1)) != MINIMUM_ADAPTIVE_HISTORY:
        raise IntegrityError("minimum adaptive history differs from the locked protocol")
    if value.get("target_order_sha256") != _id_hash(
        target_order, namespace=str(value.get("target_namespace"))
    ):
        raise IntegrityError("target order does not match its sealed hash")
    if value.get("reference_calibration_ids_sha256") != _id_hash(
        reference_ids, namespace=str(value.get("reference_namespace"))
    ):
        raise IntegrityError("reference calibration IDs do not match their sealed hash")
    provenance = value.get("order_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("verified") is not True:
        raise IntegrityError("locked order provenance is absent or unverified")
    if not provenance.get("kind") or not provenance.get("source"):
        raise IntegrityError("locked order provenance requires kind and source")
    if min(int(value.get(key, 0)) for key in (
        "metric_projections", "mmd_features", "distribution_limit"
    )) < 2:
        raise IntegrityError("streaming metric budgets are invalid")
    return value


def _history_for_strategy(
    strategy: Mapping[str, Any], cumulative: WelfordMoments,
    history_features: np.ndarray,
) -> WelfordMoments | None:
    mode = strategy["history_mode"]
    if mode == "none":
        return None
    if mode == "cumulative_welford":
        return cumulative.copy()
    if mode != "fixed_window":
        raise AssertionError(mode)
    window = int(strategy["window_size"])
    selected = history_features[-window:]
    return WelfordMoments.empty(history_features.shape[1]) if not len(selected) \
        else WelfordMoments.from_array(selected)


def _history_ids(
    target_order: Sequence[int | str], start: int, strategy: Mapping[str, Any]
) -> list[int | str]:
    mode = strategy["history_mode"]
    if mode == "none":
        return []
    first = 0 if mode == "cumulative_welford" else max(0, start - int(strategy["window_size"]))
    return list(target_order[first:start])


def _after_history_ids(
    target_order: Sequence[int | str], stop: int, strategy: Mapping[str, Any]
) -> list[int | str]:
    mode = strategy["history_mode"]
    if mode == "none":
        return []
    first = 0 if mode == "cumulative_welford" else max(0, stop - int(strategy["window_size"]))
    return list(target_order[first:stop])


def _metric_seed(plan_seed: int, block: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{plan_seed}|{block}".encode()).digest()[:8],
        "little",
    ) % (2**63 - 1)


def _safe_metrics(values: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        result[key] = number if np.isfinite(number) else None
    return result


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["strategy"]), []).append(row)
    result = []
    for strategy in _strategy_grid():
        values = buckets.get(str(strategy["strategy"]), [])
        if not values:
            continue
        weights = np.asarray([row["block_rows"] for row in values], dtype=np.float64)
        record = {
            **strategy,
            "blocks": len(values),
            "stream_rows": int(weights.sum()),
            "fitted_blocks": sum(row["fit_status"] == "FITTED" for row in values),
            "cold_start_blocks": sum(
                row["fit_status"] == "COLD_START_IDENTITY" for row in values
            ),
            "aggregation": "block-row-weighted prequential mean; unweighted SD across blocks",
        }
        for metric in AGGREGATE_METRICS:
            available = [(row.get(metric), weight) for row, weight in zip(values, weights)
                         if row.get(metric) is not None]
            vector = np.asarray([item[0] for item in available], dtype=np.float64)
            metric_weights = np.asarray([item[1] for item in available], dtype=np.float64)
            record[f"{metric}_prequential_mean"] = (
                float(np.average(vector, weights=metric_weights)) if len(vector) else None
            )
            record[f"{metric}_block_std"] = (
                float(vector.std(ddof=1)) if len(vector) > 1
                else 0.0 if len(vector) else None
            )
        result.append(record)
    return result


def streaming_scientific_gates(
    rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> dict[str, bool]:
    """Recompute protocol-integrity gates from persisted rows and the locked plan."""

    target_order = list(plan["target_order_ids"])
    namespace = str(plan["target_namespace"])
    block_size = int(plan["block_size"])
    strategies = _strategy_grid()
    expected_cells = {
        (block, str(strategy["strategy"]))
        for block in range(int(plan["block_count"])) for strategy in strategies
    }
    cells: dict[tuple[int, str], Mapping[str, Any]] = {}
    duplicates = False
    for row in rows:
        key = (int(row.get("block_index", -1)), str(row.get("strategy")))
        if key in cells:
            duplicates = True
        cells[key] = row
    prefix_exact = True
    update_exact = True
    block_exact = True
    cold_start_exact = True
    strategy_metadata_exact = True
    provenance_exact = True
    metrics_logged = True
    for block, strategy_name in expected_cells:
        row = cells.get((block, strategy_name))
        if row is None:
            prefix_exact = update_exact = block_exact = cold_start_exact = False
            strategy_metadata_exact = False
            provenance_exact = metrics_logged = False
            continue
        strategy = next(item for item in strategies if item["strategy"] == strategy_name)
        strategy_metadata_exact &= all(
            row.get(key) == strategy[key]
            for key in ("method", "history_mode", "window_size")
        )
        start = block * block_size
        stop = min(start + block_size, len(target_order))
        current = target_order[start:stop]
        before = _history_ids(target_order, start, strategy)
        after = _after_history_ids(target_order, stop, strategy)
        prefix_exact &= bool(
            int(row.get("history_count_before", -1)) == len(before)
            and row.get("history_ids_sha256_before") == _id_hash(before, namespace=namespace)
            and int(row.get("current_block_history_overlap_count", -1)) == 0
        )
        update_exact &= bool(
            int(row.get("history_count_after", -1)) == len(after)
            and row.get("history_ids_sha256_after") == _id_hash(after, namespace=namespace)
            and row.get("state_transition") == "predict_then_update"
            and row.get("prediction_used_current_block") is False
        )
        block_exact &= bool(
            int(row.get("block_start", -1)) == start
            and int(row.get("block_stop", -1)) == stop
            and int(row.get("block_rows", -1)) == len(current)
            and row.get("block_ids_sha256") == _id_hash(current, namespace=namespace)
        )
        expected_status = (
            "STATIC_IDENTITY" if strategy["method"] == "none"
            else "COLD_START_IDENTITY" if len(before) < MINIMUM_ADAPTIVE_HISTORY
            else "FITTED"
        )
        cold_start_exact &= row.get("fit_status") == expected_status
        provenance_exact &= bool(
            row.get("plan_manifest_sha256") == plan["manifest_sha256"]
            and row.get("feature_store_manifest_sha256")
            == plan["feature_store_manifest_sha256"]
            and row.get("frozen_head_value_sha256") == plan["frozen_head_value_sha256"]
            and row.get("frozen_reference_features_sha256")
            == plan["reference_calibration_features_sha256"]
        )
        metrics_logged &= all(_finite(row.get(metric)) for metric in (
            "aligned_covariance_relative_error", "aligned_sliced_w2_squared",
            "paired_feature_mse", "reference_prediction_agreement", "aligned_accuracy",
        ))
    grid_exact = not duplicates and set(cells) == expected_cells
    gates = {
        "locked_strategy_grid_complete": bool(grid_exact),
        "history_is_exact_strict_prefix": bool(prefix_exact),
        "current_block_is_excluded_until_update": bool(update_exact),
        "locked_blocks_cover_order_once": bool(block_exact),
        "cold_start_policy_is_explicit": bool(cold_start_exact),
        "strategy_metadata_matches_lock": bool(strategy_metadata_exact),
        "frozen_store_reference_and_head_are_bound": bool(provenance_exact),
        "distribution_and_task_metrics_logged": bool(metrics_logged),
        "cumulative_and_all_fixed_windows_present": {
            row.get("history_mode") for row in rows
        } == {"none", "cumulative_welford", "fixed_window"} and {
            row.get("window_size") for row in rows if row.get("history_mode") == "fixed_window"
        } == set(STREAMING_WINDOWS),
    }
    gates["all_passed"] = all(gates.values())
    return gates


def run_prequential_stream(
    store_path: str | Path,
    plan: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact locked strategy grid on one verified feature store."""

    locked = _load_plan(plan)
    store = load_feature_store(store_path, verify_hashes=True)
    if sha256_file(store.manifest_path) != locked["feature_store_manifest_sha256"]:
        raise IntegrityError("streaming plan is bound to a different feature store")
    if store.dataset != locked["dataset"] or store.encoder != locked["encoder"]:
        raise IntegrityError("streaming store identity differs from the plan")
    target = store.endpoint("target")
    reference = store.endpoint("reference")
    if target.id_namespace != locked["target_namespace"]:
        raise IntegrityError("target namespace differs from the streaming plan")
    if reference.id_namespace != locked["reference_namespace"]:
        raise IntegrityError("reference namespace differs from the streaming plan")
    target_order = [_id_value(value) for value in locked["target_order_ids"]]
    reference_calibration_ids = [
        _id_value(value) for value in locked["reference_calibration_ids"]
    ]
    if _id_hash(target_order, namespace=target.id_namespace) != locked["target_order_sha256"]:
        raise IntegrityError("stream target order hash is inconsistent")
    target_index = _lookup_ids(target.ids, target_order, name="stream target order")
    reference_calibration_index = _lookup_ids(
        reference.ids, reference_calibration_ids, name="stream reference calibration"
    )
    target_features = np.asarray(target.features[target_index], dtype=np.float64)
    frozen_reference_features = np.asarray(
        reference.features[reference_calibration_index], dtype=np.float64
    )
    if (stable_hash_arrays(frozen_reference_features)
            != locked["reference_calibration_features_sha256"]):
        raise IntegrityError("frozen reference calibration features changed")
    reference_state = WelfordMoments.from_array(frozen_reference_features)
    # This endpoint is frozen, so its regularized covariance is computed once.
    # Source-side eigendecompositions remain strategy/state specific.
    reference_oas = _oas_covariance(reference_state)
    weight, bias, head_hash = _head(store)
    if head_hash != locked["frozen_head_value_sha256"]:
        raise IntegrityError("frozen head values changed")

    paired_reference, paired_target = store.pairing().validated()
    target_to_reference = dict(zip(paired_target, paired_reference))
    evaluation_reference_ids = [target_to_reference[value] for value in target_order]
    if (_id_hash(evaluation_reference_ids, namespace=reference.id_namespace)
            != locked["evaluation_reference_ids_sha256"]):
        raise IntegrityError("evaluation pairing differs from the locked plan")
    evaluation_reference_index = _lookup_ids(
        reference.ids, evaluation_reference_ids, name="stream evaluation reference"
    )
    evaluation_reference_features = np.asarray(
        reference.features[evaluation_reference_index], dtype=np.float64
    )
    if target.labels is None or reference.labels is None:
        raise IntegrityError("streaming task metrics require endpoint labels")
    target_labels = np.asarray(target.labels[target_index], dtype=np.int64)
    reference_labels = np.asarray(
        reference.labels[evaluation_reference_index], dtype=np.int64
    )
    if not np.array_equal(target_labels, reference_labels):
        raise IntegrityError("verified streaming evaluation pairs have unequal labels")

    cumulative = WelfordMoments.empty(target_features.shape[1])
    history_features = np.empty((0, target_features.shape[1]), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    block_size = int(locked["block_size"])
    for block, start in enumerate(range(0, len(target_order), block_size)):
        stop = min(start + block_size, len(target_order))
        block_ids = target_order[start:stop]
        source_block = target_features[start:stop]
        reference_block = evaluation_reference_features[start:stop]
        labels_block = target_labels[start:stop]
        block_rows: list[dict[str, Any]] = []
        fitted_cache: dict[tuple[str, str], AffineMap] = {}
        metric_cache: dict[str, dict[str, float | None]] = {}
        for strategy in _strategy_grid():
            before_ids = _history_ids(target_order, start, strategy)
            after_ids = _after_history_ids(target_order, stop, strategy)
            source_state = _history_for_strategy(strategy, cumulative, history_features)
            method = str(strategy["method"])
            if method == "none":
                fitted = AffineMap(
                    np.eye(target_features.shape[1]), np.zeros(target_features.shape[1]),
                    "none", {"static": True},
                )
                fit_status = "STATIC_IDENTITY"
                source_state_hash = None
            elif source_state is None or source_state.count < MINIMUM_ADAPTIVE_HISTORY:
                fitted = AffineMap(
                    np.eye(target_features.shape[1]), np.zeros(target_features.shape[1]),
                    method, {"cold_start_identity": True},
                )
                fit_status = "COLD_START_IDENTITY"
                source_state_hash = None if source_state is None else source_state.content_hash
            else:
                cache_key = (method, source_state.content_hash)
                fitted = fitted_cache.get(cache_key)
                if fitted is None:
                    fitted = _fit_from_moments(
                        method, source_state, reference_state,
                        reference_oas=reference_oas,
                    )
                    fitted_cache[cache_key] = fitted
                fit_status = "FITTED"
                source_state_hash = source_state.content_hash
            map_hash = stable_hash_arrays(fitted.matrix, fitted.offset)
            metrics = metric_cache.get(map_hash)
            if metrics is None:
                aligned = fitted.apply(source_block)
                metrics = _safe_metrics(evaluate_alignment(
                    reference_block, source_block, aligned,
                    labels=labels_block, readout_weight=weight, readout_bias=bias,
                    rng=np.random.default_rng(
                        _metric_seed(int(locked["metric_seed"]), block)
                    ),
                    projections=int(locked["metric_projections"]),
                    mmd_features=int(locked["mmd_features"]),
                    distribution_limit=min(
                        int(locked["distribution_limit"]), len(source_block)
                    ),
                ))
                metric_cache[map_hash] = metrics
            before_set = set(before_ids)
            row = {
                "schema_version": "marginal_alignment.campaign.streaming_row.v1",
                "dataset": store.dataset,
                "encoder": store.encoder,
                "corruption": store.manifest.get("metadata", {}).get("corruption"),
                "severity": store.manifest.get("metadata", {}).get("severity"),
                "plan_manifest_sha256": locked["manifest_sha256"],
                "feature_store_manifest_sha256": locked["feature_store_manifest_sha256"],
                "target_order_sha256": locked["target_order_sha256"],
                "frozen_reference_features_sha256": locked[
                    "reference_calibration_features_sha256"
                ],
                "frozen_reference_state_sha256": reference_state.content_hash,
                "frozen_head_value_sha256": head_hash,
                "block_index": block,
                "block_start": start,
                "block_stop": stop,
                "block_rows": len(source_block),
                "block_ids_sha256": _id_hash(block_ids, namespace=target.id_namespace),
                **strategy,
                "history_count_before": len(before_ids),
                "history_ids_sha256_before": _id_hash(
                    before_ids, namespace=target.id_namespace
                ),
                "history_count_after": len(after_ids),
                "history_ids_sha256_after": _id_hash(
                    after_ids, namespace=target.id_namespace
                ),
                "current_block_history_overlap_count": len(
                    before_set.intersection(block_ids)
                ),
                "prediction_used_current_block": False,
                "state_transition": "predict_then_update",
                "fit_status": fit_status,
                "source_state_sha256_before": source_state_hash,
                "map_sha256": map_hash,
                "fit_diagnostics": fitted.diagnostics,
                **metrics,
            }
            block_rows.append(row)
        # This mutation is deliberately below the complete strategy loop.
        cumulative.update(source_block)
        history_features = np.concatenate([history_features, source_block], axis=0)
        history_features = history_features[-max(STREAMING_WINDOWS):]
        rows.extend(block_rows)

    gates = streaming_scientific_gates(rows, locked)
    return {
        "schema_version": "marginal_alignment.campaign.streaming_result.v1",
        "status": "PASS" if gates["all_passed"] else "FAIL",
        "plan_manifest_sha256": locked["manifest_sha256"],
        "feature_store_manifest_sha256": locked["feature_store_manifest_sha256"],
        "protocol": "strict_predict_before_update",
        "rows": rows,
        "aggregates": _aggregate_rows(rows),
        "scientific_gates": gates,
    }


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_streaming_results(
    output_dir: str | Path, result: Mapping[str, Any]
) -> list[Path]:
    """Persist raw blocks, aggregates, and a content-bound PASS/FAIL manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = result.get("rows")
    aggregates = result.get("aggregates")
    gates = result.get("scientific_gates")
    if not isinstance(rows, list) or not isinstance(aggregates, list) or not isinstance(gates, Mapping):
        raise IntegrityError("streaming result is malformed")
    expected_status = "PASS" if gates.get("all_passed") is True else "FAIL"
    if result.get("status") != expected_status:
        raise IntegrityError("streaming result status disagrees with its scientific gates")
    rows_path = output / "rows.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    _atomic_jsonl(rows_path, rows)
    write_json_atomic(summary_path, {
        "schema_version": "marginal_alignment.campaign.streaming_summary.v1",
        "status": expected_status,
        "protocol": result.get("protocol"),
        "plan_manifest_sha256": result.get("plan_manifest_sha256"),
        "feature_store_manifest_sha256": result.get("feature_store_manifest_sha256"),
        "row_count": len(rows),
        "aggregates": aggregates,
        "scientific_gates": dict(gates),
    })
    write_json_atomic(manifest_path, {
        "schema_version": "marginal_alignment.campaign.streaming_manifest.v1",
        "status": expected_status,
        "plan_manifest_sha256": result.get("plan_manifest_sha256"),
        "feature_store_manifest_sha256": result.get("feature_store_manifest_sha256"),
        "record_count": len(rows),
        "aggregate_count": len(aggregates),
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
        "scientific_gates": dict(gates),
    })
    return [rows_path, summary_path, manifest_path]
