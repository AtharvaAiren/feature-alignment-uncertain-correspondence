"""Run V2 real-feature suites with enforced information separation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .artifacts import IntegrityError, sha256_file, sha256_json, write_json_atomic
from .feature_store import FeatureStore, load_feature_store
from .inference import macro_summary
from .method_registry import fit_registered_map, method_capabilities
from .metrics import (
    EvaluationContext,
    classification_metrics,
    evaluate_alignment_with_context,
    evaluate_alignment,
    prepare_evaluation_context,
)
from .split_plan import PairingTable, build_split_plan


METRICS = (
    "aligned_mean_l2", "aligned_coordinate_variance_relative_error",
    "aligned_covariance_relative_error", "aligned_correlation_relative_error",
    "aligned_standardized_skew_relative_error", "aligned_standardized_kurtosis_relative_error",
    "aligned_coordinate_w2_squared", "aligned_sliced_w2_squared", "aligned_rff_mmd_squared",
    "paired_feature_mse", "centered_logit_mse", "reference_prediction_agreement",
    "aligned_accuracy", "negative_log_likelihood", "brier_score",
    "expected_calibration_error", "accuracy_recovery_fraction",
    "source_to_aligned_standardized_vector_mse",
    "source_to_aligned_correlation_relative_change",
    "source_to_aligned_coordinate_rank_mse",
)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _method_kwargs(specification: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    implementation = str(specification["implementation"])
    parameters = {
        key: value for key, value in specification.items()
        if key not in {
            "implementation", "information", "pair_control", "scope", "noise_sweep",
            "fit_on_pair_endpoint_sets", "minimum_calibration_size",
        }
    }
    renames = {
        "covariance": "covariance_estimator",
        "iterations": "sweeps",
    }
    parameters = {renames.get(key, key): value for key, value in parameters.items()}
    return implementation, parameters


def _method_available_at_calibration(
    specification: Mapping[str, Any], calibration_size: int,
) -> bool:
    minimum = specification.get("minimum_calibration_size")
    return minimum is None or int(calibration_size) >= int(minimum)


def _indices(ids: np.ndarray, selected: list[Any], *, name: str) -> np.ndarray:
    lookup = {value.item() if isinstance(value, np.generic) else value: index
              for index, value in enumerate(np.asarray(ids))}
    try:
        return np.asarray([lookup[value] for value in selected], dtype=np.int64)
    except KeyError as error:
        raise IntegrityError(f"{name} references absent endpoint ID {error.args[0]!r}") from error


def _head(store: FeatureStore) -> tuple[np.ndarray, np.ndarray]:
    roles = store.manifest.get("head", {})
    if not isinstance(roles, Mapping):
        raise IntegrityError("feature store lacks frozen-readout declaration")
    weight_role = roles.get("weight_role")
    bias_role = roles.get("bias_role")
    if not weight_role or not bias_role:
        raise IntegrityError("frozen readout must declare weight_role and bias_role")
    weight = np.asarray(store.array(str(weight_role)), dtype=np.float64)
    bias = np.asarray(store.array(str(bias_role)), dtype=np.float64)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise IntegrityError("frozen readout arrays have invalid shapes")
    return weight, bias


def _labels(reference_labels: np.ndarray | None, target_labels: np.ndarray | None,
            reference_index: np.ndarray, target_index: np.ndarray) -> np.ndarray:
    if reference_labels is None or target_labels is None:
        raise IntegrityError("evaluation requires labels on both endpoints")
    reference = np.asarray(reference_labels[reference_index], dtype=np.int64)
    target = np.asarray(target_labels[target_index], dtype=np.int64)
    if not np.array_equal(reference, target):
        raise IntegrityError("verified evaluation pairs have unequal endpoint labels")
    return target


def _seed(*values: Any) -> int:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _path_token(value: Any) -> str:
    """Return a conservative, deterministic token for split-plan filenames."""

    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    return token or "missing"


def split_plan_path(
    output_dir: Path, *, suite_name: str, dataset: str, encoder: str,
    corruption: Any, severity: Any, split_seed: int,
) -> Path:
    """Name split plans by their complete experimental unit.

    Including the suite prevents two suites that happen to reuse a split-seed
    integer from silently overwriting one another's identity manifest.
    """

    return output_dir / "splits" / (
        f"{_path_token(suite_name)}__{_path_token(dataset)}__{_path_token(encoder)}__"
        f"{_path_token(corruption)}__s{_path_token(severity)}__seed{int(split_seed)}.json"
    )


def _row_base(
    *, suite: str, store: FeatureStore, condition: Mapping[str, Any], split_seed: int,
    calibration_size: int, evaluation_size: int, method: str,
    pair_budget: int | None, pair_noise_rate: float | None,
    pair_noise_rate_actual: float | None,
    split_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "marginal_alignment.campaign.real_row.v2",
        "suite": suite,
        "dataset": store.dataset,
        "encoder": store.encoder,
        "corruption": condition.get("corruption"),
        "severity": condition.get("severity"),
        "split_seed": int(split_seed),
        "calibration_size": int(calibration_size),
        "evaluation_size": int(evaluation_size),
        "method": method,
        "pair_budget": pair_budget,
        # Keep the legacy key as the requested experimental dose so existing
        # grouping/inference remains stable, but never confuse it with the
        # feasible realized mismatch fraction at small pair budgets.
        "pair_noise_rate": pair_noise_rate,
        "pair_noise_rate_requested": pair_noise_rate,
        "pair_noise_rate_actual": pair_noise_rate_actual,
        "feature_store_manifest": str(store.manifest_path),
        "feature_store_manifest_sha256": sha256_file(store.manifest_path),
        "split_plan_sha256": split_plan["manifest_sha256"],
        "evaluation_sha256": split_plan["evaluation_sha256"],
    }


def _fit_and_evaluate(
    *, method_name: str, method_specification: Mapping[str, Any], store: FeatureStore,
    source_fit: np.ndarray, reference_fit: np.ndarray, source_eval: np.ndarray,
    reference_eval: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray,
    fit_rng: np.random.Generator, metric_rng: np.random.Generator | None = None,
    evaluation_context: EvaluationContext | None = None,
    pair_control: str | None = None, pair_noise_fraction: float = 0.0,
    requested_pair_noise_fraction: float = 0.0, metrics_projections: int = 64,
    mmd_features: int = 512, distribution_evaluation_limit: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    implementation, parameters = _method_kwargs(method_specification)
    capabilities = method_capabilities(implementation)
    if capabilities.uses_labels or capabilities.changes_consumer:
        raise NotImplementedError(
            "labeled consumer adaptation is isolated from frozen-map evaluation; "
            "run it through the dedicated comparator stage"
        )
    if pair_control is not None:
        parameters["pair_control"] = None if pair_control == "true" else pair_control
        parameters["pair_noise_fraction"] = pair_noise_fraction
        parameters["pair_assignment_materialized"] = True
    start = time.time()
    try:
        fitted = fit_registered_map(
            implementation, source_fit, reference_fit,
            readout_weight=weight if capabilities.requires_readout else None,
            rng=fit_rng, **parameters,
        )
    except Exception as error:
        raise RuntimeError(
            "registered-map fit failed for "
            f"method_alias={method_name!r}, implementation={implementation!r}, "
            f"source_rows={len(source_fit)}, reference_rows={len(reference_fit)}, "
            f"dimension={source_fit.shape[1]}"
        ) from error
    aligned = fitted.apply(source_eval)
    if evaluation_context is None:
        if metric_rng is None:
            raise ValueError("metric_rng is required without an evaluation context")
        metrics = evaluate_alignment(
            reference_eval, source_eval, aligned, labels=labels,
            readout_weight=weight, readout_bias=bias, rng=metric_rng,
            projections=metrics_projections, mmd_features=mmd_features,
            distribution_limit=distribution_evaluation_limit,
        )
    else:
        metrics = evaluate_alignment_with_context(aligned, evaluation_context)
    diagnostics = dict(fitted.diagnostics)
    if pair_control is not None:
        diagnostics["requested_pair_noise_fraction"] = float(
            requested_pair_noise_fraction
        )
        diagnostics["actual_pair_noise_fraction"] = float(pair_noise_fraction)
    diagnostics["fit_seconds"] = time.time() - start
    return metrics, diagnostics


def _fit_labeled_er(
    *, source_fit: np.ndarray, fit_labels: np.ndarray,
    reference_replay_fit: np.ndarray, reference_replay_labels: np.ndarray,
    source_eval: np.ndarray, reference_eval: np.ndarray, eval_labels: np.ndarray,
    weight: np.ndarray, bias: np.ndarray, specification: Mapping[str, Any],
    fit_rng: np.random.Generator, metric_rng: np.random.Generator | None = None,
    evaluation_context: EvaluationContext | None = None,
    metrics_projections: int = 64, mmd_features: int = 512,
    distribution_evaluation_limit: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Labeled experience replay comparator with an explicit information audit.

    The feature map is the identity.  The reference-trained head is copied
    exactly, then updated on the union of a labeled target buffer and a labeled
    clean/reference replay buffer.  Consequently distribution and feature
    residuals remain the raw target residuals, while task metrics are evaluated
    under the adapted consumer.
    """

    started = time.time()
    source_fit = np.asarray(source_fit, dtype=np.float64)
    reference_replay_fit = np.asarray(reference_replay_fit, dtype=np.float64)
    fit_labels = np.asarray(fit_labels, dtype=np.int64)
    reference_replay_labels = np.asarray(reference_replay_labels, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if source_fit.ndim != 2 or reference_replay_fit.ndim != 2:
        raise ValueError("ER target and replay features must be matrices")
    if source_fit.shape[1] != reference_replay_fit.shape[1]:
        raise ValueError("ER target and replay feature dimensions differ")
    if fit_labels.shape != (len(source_fit),):
        raise ValueError("ER target labels do not match target features")
    if reference_replay_labels.shape != (len(reference_replay_fit),):
        raise ValueError("ER replay labels do not match replay features")
    if weight.ndim != 2 or weight.shape[1] != source_fit.shape[1]:
        raise ValueError("ER frozen weight has the wrong shape")
    if bias.shape != (weight.shape[0],):
        raise ValueError("ER frozen bias has the wrong shape")
    for name, labels in (("target", fit_labels), ("reference replay", reference_replay_labels)):
        if np.any(labels < 0) or np.any(labels >= weight.shape[0]):
            raise ValueError(f"ER {name} labels fall outside the frozen-head class range")

    requested_buffer_size = int(specification.get("buffer_size", 1024))
    if requested_buffer_size < 1:
        raise ValueError("ER buffer_size must be positive")
    target_buffer_size = min(requested_buffer_size, len(source_fit))
    replay_ratio = float(specification.get("replay_ratio", 1.0))
    if not np.isfinite(replay_ratio) or replay_ratio <= 0.0:
        raise ValueError("ER replay_ratio must be positive")
    requested_replay_size = int(specification.get(
        "replay_buffer_size", round(target_buffer_size * replay_ratio)
    ))
    if requested_replay_size < 1:
        raise ValueError("ER replay buffer must contain at least one clean example")
    replay_buffer_size = min(requested_replay_size, len(reference_replay_fit))
    if target_buffer_size < 1 or replay_buffer_size < 1:
        raise ValueError("ER requires nonempty target and clean replay buffers")

    target_order = fit_rng.permutation(len(source_fit))[:target_buffer_size]
    replay_order = fit_rng.permutation(len(reference_replay_fit))[:replay_buffer_size]
    target_x, target_y = source_fit[target_order], fit_labels[target_order]
    replay_x = reference_replay_fit[replay_order]
    replay_y = reference_replay_labels[replay_order]
    train_x = np.concatenate([target_x, replay_x], axis=0)
    train_y = np.concatenate([target_y, replay_y], axis=0)

    epochs = int(specification.get("steps", 5))
    learning_rate = float(specification.get("learning_rate", 1e-2))
    anchored_weight_decay = float(specification.get("weight_decay", 1e-4))
    batch_size = min(int(specification.get("batch_size", 256)), len(train_x))
    if epochs < 1 or batch_size < 1:
        raise ValueError("ER steps and batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("ER learning_rate must be positive")
    if not np.isfinite(anchored_weight_decay) or anchored_weight_decay < 0.0:
        raise ValueError("ER weight_decay must be nonnegative")

    initial_weight = weight.copy()
    initial_bias = bias.copy()
    adapted_weight = initial_weight.copy()
    adapted_bias = initial_bias.copy()

    def training_nll(features: np.ndarray, labels: np.ndarray,
                     candidate_weight: np.ndarray, candidate_bias: np.ndarray) -> float:
        return classification_metrics(
            features @ candidate_weight.T + candidate_bias, labels
        )["negative_log_likelihood"]

    initial_target_nll = training_nll(target_x, target_y, adapted_weight, adapted_bias)
    initial_replay_nll = training_nll(replay_x, replay_y, adapted_weight, adapted_bias)
    optimizer_updates = 0
    for _ in range(epochs):
        epoch_order = fit_rng.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            indices = epoch_order[start:start + batch_size]
            x_batch, y_batch = train_x[indices], train_y[indices]
            batch_logits = x_batch @ adapted_weight.T + adapted_bias
            batch_logits -= batch_logits.max(axis=1, keepdims=True)
            probabilities = np.exp(batch_logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            errors = probabilities
            errors[np.arange(len(indices)), y_batch] -= 1.0
            gradient_weight = errors.T @ x_batch / len(indices)
            gradient_bias = errors.mean(axis=0)
            # Anchoring to the frozen head is distinct from replay: replay
            # supplies clean labeled examples, while this optional penalty
            # merely discourages an unnecessarily large consumer displacement.
            gradient_weight += anchored_weight_decay * (adapted_weight - initial_weight)
            gradient_bias += anchored_weight_decay * (adapted_bias - initial_bias)
            adapted_weight -= learning_rate * gradient_weight
            adapted_bias -= learning_rate * gradient_bias
            optimizer_updates += 1

    final_target_nll = training_nll(target_x, target_y, adapted_weight, adapted_bias)
    final_replay_nll = training_nll(replay_x, replay_y, adapted_weight, adapted_bias)

    # Identity-aligned evaluation preserves all raw feature/distribution
    # residuals and the frozen-head source/reference baselines.
    if evaluation_context is None:
        if metric_rng is None:
            raise ValueError("metric_rng is required without an evaluation context")
        metrics = evaluate_alignment(
            reference_eval, source_eval, source_eval, labels=eval_labels,
            readout_weight=weight, readout_bias=bias, rng=metric_rng,
            projections=metrics_projections, mmd_features=mmd_features,
            distribution_limit=distribution_evaluation_limit,
        )
        reference_logits = reference_eval @ weight.T + bias
    else:
        metrics = evaluate_alignment_with_context(source_eval, evaluation_context)
        reference_logits = evaluation_context.reference_logits
    adapted_logits = source_eval @ adapted_weight.T + adapted_bias
    adapted_task = classification_metrics(
        adapted_logits, eval_labels, reference_logits=reference_logits
    )
    aligned_accuracy = adapted_task["accuracy"]
    source_accuracy = float(metrics["source_accuracy"])
    reference_accuracy = float(metrics["reference_accuracy"])
    denominator = reference_accuracy - source_accuracy
    metrics.update({
        "reference_prediction_agreement": adapted_task[
            "reference_prediction_agreement"
        ],
        "aligned_accuracy": aligned_accuracy,
        "accuracy_recovery_fraction": (
            (aligned_accuracy - source_accuracy) / denominator
            if abs(denominator) > 1e-12 else float("nan")
        ),
        "negative_log_likelihood": adapted_task["negative_log_likelihood"],
        "brier_score": adapted_task["brier_score"],
        "expected_calibration_error": adapted_task["expected_calibration_error"],
    })
    return metrics, {
        "method": "labeled_experience_replay",
        "changes_frozen_consumer": True,
        "feature_transform_applied": False,
        "uses_target_labels": True,
        "uses_reference_replay_labels": True,
        "initialization": "exact_frozen_reference_head_weight_and_bias",
        "buffer_size": target_buffer_size,
        "target_buffer_rows": target_buffer_size,
        "clean_reference_replay_rows": replay_buffer_size,
        "requested_replay_ratio": replay_ratio,
        "realized_replay_to_target_ratio": replay_buffer_size / target_buffer_size,
        "steps": epochs,
        "step_semantics": "full shuffled epochs over target-plus-clean-replay union",
        "batch_size": batch_size,
        "optimizer_updates": optimizer_updates,
        "optimizer": "minibatch_multiclass_softmax_sgd",
        "learning_rate": learning_rate,
        "anchored_weight_decay": anchored_weight_decay,
        "initial_target_buffer_nll": initial_target_nll,
        "final_target_buffer_nll": final_target_nll,
        "initial_clean_replay_nll": initial_replay_nll,
        "final_clean_replay_nll": final_replay_nll,
        "weight_update_frobenius": float(np.linalg.norm(
            adapted_weight - initial_weight, ord="fro"
        )),
        "bias_update_l2": float(np.linalg.norm(adapted_bias - initial_bias)),
        "feature_metric_semantics": (
            "identity feature map: aligned distribution, paired-feature, and "
            "frozen centered-logit residuals equal the raw target residuals"
        ),
        "task_metric_semantics": (
            "agreement, accuracy, NLL, Brier, and ECE use adapted consumer logits"
        ),
        "comparator_scope": (
            "labeled target experience replay with clean/reference replay; consumer "
            "changes, encoder and features do not"
        ),
        "fit_seconds": time.time() - started,
    }


def run_condition(
    *, store_path: Path, suite_name: str, suite: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]], condition: Mapping[str, Any],
    output_dir: Path, global_seed: int, metrics_projections: int = 64,
    mmd_features: int = 512, distribution_evaluation_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run one dataset/encoder/corruption/severity store and persist its split plans."""

    store = load_feature_store(store_path, verify_hashes=True)
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    pairing = store.pairing()
    weight, bias = _head(store)
    model_provenance = store.manifest["metadata"]["model_provenance"]
    head_fit_ids = np.asarray(store.array(str(model_provenance["head_fit_ids_role"])))
    head_fit_namespace = str(model_provenance["head_fit_id_namespace"])
    head_fit_shares_reference_namespace = head_fit_namespace == reference.id_namespace
    if str(model_provenance["head_fit_split"]) == str(
        model_provenance["reference_split"]
    ) and not head_fit_shares_reference_namespace:
        raise IntegrityError(
            "head fit and endpoint share a split, but their ID namespaces differ"
        )
    excluded_reference_ids = np.asarray([], dtype=reference.ids.dtype)
    excluded_target_ids = np.asarray([], dtype=target.ids.dtype)
    paired_reference_ids, paired_target_ids = pairing.validated()
    if head_fit_shares_reference_namespace:
        target_by_reference = {
            reference_id.item() if isinstance(reference_id, np.generic) else reference_id:
            target_id.item() if isinstance(target_id, np.generic) else target_id
            for reference_id, target_id in zip(paired_reference_ids, paired_target_ids)
        }
        missing = [
            value.item() if isinstance(value, np.generic) else value
            for value in head_fit_ids
            if (value.item() if isinstance(value, np.generic) else value)
            not in target_by_reference
        ]
        if missing:
            raise IntegrityError(
                f"head-fit reference IDs lack verified target counterparts: {missing[:3]}"
            )
        excluded_reference_ids = np.asarray(head_fit_ids, dtype=reference.ids.dtype)
        excluded_target_ids = np.asarray([
            target_by_reference[value.item() if isinstance(value, np.generic) else value]
            for value in head_fit_ids
        ], dtype=target.ids.dtype)
    elif head_fit_namespace == target.id_namespace:
        reference_by_target = {
            target_id.item() if isinstance(target_id, np.generic) else target_id:
            reference_id.item() if isinstance(reference_id, np.generic) else reference_id
            for reference_id, target_id in zip(paired_reference_ids, paired_target_ids)
        }
        missing = [
            value.item() if isinstance(value, np.generic) else value
            for value in head_fit_ids
            if (value.item() if isinstance(value, np.generic) else value)
            not in reference_by_target
        ]
        if missing:
            raise IntegrityError(
                f"head-fit target IDs lack verified reference counterparts: {missing[:3]}"
            )
        excluded_target_ids = np.asarray(head_fit_ids, dtype=target.ids.dtype)
        excluded_reference_ids = np.asarray([
            reference_by_target[value.item() if isinstance(value, np.generic) else value]
            for value in head_fit_ids
        ], dtype=reference.ids.dtype)
    evaluation_size = int(condition["evaluation_size"])
    calibration_sizes = [int(value) for value in suite["calibration_sizes"]]
    pair_budgets = [int(value) for value in suite.get("pair_budgets", [])]
    noise_rates = [float(value) for value in suite.get("pair_noise_rates", [0.0])]
    rows: list[dict[str, Any]] = []
    for split_seed in suite["split_seeds"]:
        split_path = split_plan_path(
            output_dir, suite_name=suite_name, dataset=store.dataset,
            encoder=store.encoder, corruption=condition.get("corruption"),
            severity=condition.get("severity"), split_seed=int(split_seed),
        )
        plan = build_split_plan(
            reference_ids=reference.ids, target_ids=target.ids, pairing=pairing,
            seed=_seed(global_seed, suite_name, store.dataset, store.encoder,
                       condition.get("corruption"), condition.get("severity"), split_seed),
            evaluation_size=evaluation_size, calibration_sizes=calibration_sizes,
            pair_budgets=pair_budgets, noisy_pair_fractions=noise_rates,
            reference_namespace=reference.id_namespace, target_namespace=target.id_namespace,
            excluded_reference_ids=excluded_reference_ids,
            excluded_target_ids=excluded_target_ids,
            head_fit_exclusion_required=(
                head_fit_namespace in {reference.id_namespace, target.id_namespace}
            ),
            head_fit_ids_count=int(len(head_fit_ids)),
            output_path=split_path,
        )
        eval_ref_index = _indices(reference.ids, plan["evaluation"]["reference_ids"],
                                  name="evaluation reference")
        eval_target_index = _indices(target.ids, plan["evaluation"]["target_ids"],
                                     name="evaluation target")
        reference_eval = np.asarray(reference.features[eval_ref_index], dtype=np.float64)
        source_eval = np.asarray(target.features[eval_target_index], dtype=np.float64)
        eval_labels = _labels(reference.labels, target.labels, eval_ref_index, eval_target_index)
        # Every method/calibration budget evaluated on this fixed identity set
        # receives an independently instantiated RNG with the same seed.  Thus
        # sliced-W2 directions and RFF features are common random numbers and
        # cannot add method-specific Monte Carlo noise to a contrast.
        metric_seed = _seed(
            global_seed, "evaluation_metrics", plan["evaluation_sha256"],
            metrics_projections, mmd_features,
        )
        evaluation_context = prepare_evaluation_context(
            reference_eval, source_eval, labels=eval_labels,
            readout_weight=weight, readout_bias=bias,
            rng=np.random.default_rng(metric_seed),
            projections=metrics_projections, mmd_features=mmd_features,
            distribution_limit=distribution_evaluation_limit,
        )
        for calibration_size in calibration_sizes:
            calibration = plan["calibrations"][str(calibration_size)]
            ref_index = _indices(reference.ids, calibration["reference_ids"], name="reference fit")
            target_index = _indices(target.ids, calibration["target_ids"], name="target fit")
            reference_marginal_fit = np.asarray(reference.features[ref_index], dtype=np.float64)
            source_marginal_fit = np.asarray(target.features[target_index], dtype=np.float64)
            for method_name in suite["methods"]:
                specification = methods[method_name]
                if not _method_available_at_calibration(
                    specification, calibration_size
                ):
                    continue
                implementation, _ = _method_kwargs(specification)
                if implementation == "labeled_er":
                    if target.labels is None or reference.labels is None:
                        raise IntegrityError(
                            "labeled_er requires target-buffer and clean-reference replay labels"
                        )
                    fit_labels = np.asarray(target.labels[target_index], dtype=np.int64)
                    replay_labels = np.asarray(reference.labels[ref_index], dtype=np.int64)
                    fit_rng = np.random.default_rng(_seed(
                        global_seed, "fit", plan["manifest_sha256"], method_name,
                        calibration_size,
                    ))
                    metrics, diagnostics = _fit_labeled_er(
                        source_fit=source_marginal_fit, fit_labels=fit_labels,
                        reference_replay_fit=reference_marginal_fit,
                        reference_replay_labels=replay_labels,
                        source_eval=source_eval, reference_eval=reference_eval,
                        eval_labels=eval_labels, weight=weight, bias=bias,
                        specification=specification, fit_rng=fit_rng,
                        evaluation_context=evaluation_context,
                        metrics_projections=metrics_projections,
                        mmd_features=mmd_features,
                        distribution_evaluation_limit=distribution_evaluation_limit,
                    )
                    rows.append({
                        **_row_base(suite=suite_name, store=store, condition=condition,
                            split_seed=split_seed, calibration_size=calibration_size,
                            evaluation_size=evaluation_size, method=method_name,
                            pair_budget=None, pair_noise_rate=None,
                            pair_noise_rate_actual=None, split_plan=plan),
                        "fit_information": "target_labels_plus_clean_reference_replay",
                        "configured_information_label": specification["information"],
                        "pair_control": None,
                        "changes_frozen_consumer": True,
                        "diagnostics": diagnostics,
                        **metrics,
                    })
                    continue
                capabilities = method_capabilities(implementation)
                if specification.get("fit_on_pair_endpoint_sets") is True:
                    if capabilities.requires_pairing or capabilities.uses_labels:
                        raise ValueError(
                            "fit_on_pair_endpoint_sets is only valid for an unpaired "
                            "marginal method"
                        )
                    if not pair_budgets:
                        raise ValueError(
                            f"{method_name} requests pair endpoint sets but the suite "
                            "declares no pair_budgets"
                        )
                    for pair_budget in pair_budgets:
                        true_control = plan["pair_controls"][str(pair_budget)]["controls"]["true"]
                        # The two endpoint *sets* are deliberately canonicalized
                        # independently.  No row ordering or assignment relation is
                        # supplied to the marginal estimator.
                        pair_reference_ids = sorted(
                            true_control["reference_ids"],
                            key=lambda value: (type(value).__name__, str(value)),
                        )
                        pair_target_ids = sorted(
                            true_control["target_ids"],
                            key=lambda value: (type(value).__name__, str(value)),
                        )
                        pair_ref_index = _indices(
                            reference.ids, pair_reference_ids,
                            name="assignment-erased reference endpoint fit",
                        )
                        pair_target_index = _indices(
                            target.ids, pair_target_ids,
                            name="assignment-erased target endpoint fit",
                        )
                        pair_reference = np.asarray(
                            reference.features[pair_ref_index], dtype=np.float64
                        )
                        pair_source = np.asarray(
                            target.features[pair_target_index], dtype=np.float64
                        )
                        fit_rng = np.random.default_rng(_seed(
                            global_seed, "fit", plan["manifest_sha256"], method_name,
                            calibration_size, pair_budget, "assignment_erased",
                        ))
                        metrics, diagnostics = _fit_and_evaluate(
                            method_name=method_name,
                            method_specification=specification,
                            store=store, source_fit=pair_source,
                            reference_fit=pair_reference, source_eval=source_eval,
                            reference_eval=reference_eval, labels=eval_labels,
                            weight=weight, bias=bias, fit_rng=fit_rng,
                            evaluation_context=evaluation_context,
                            metrics_projections=metrics_projections,
                            mmd_features=mmd_features,
                            distribution_evaluation_limit=distribution_evaluation_limit,
                        )
                        diagnostics.update({
                            "pair_assignment_available_to_estimator": False,
                            "pair_assignment_erased": True,
                            "endpoint_set_budget": int(pair_budget),
                            "reference_endpoint_set_sha256": true_control[
                                "reference_set_sha256"
                            ],
                            "target_endpoint_set_sha256": true_control[
                                "target_set_sha256"
                            ],
                        })
                        rows.append({
                            **_row_base(
                                suite=suite_name, store=store, condition=condition,
                                split_seed=split_seed,
                                calibration_size=calibration_size,
                                evaluation_size=evaluation_size, method=method_name,
                                pair_budget=pair_budget, pair_noise_rate=None,
                                pair_noise_rate_actual=None, split_plan=plan,
                            ),
                            "fit_information": specification["information"],
                            "pair_control": "assignment_erased",
                            "diagnostics": diagnostics,
                            **metrics,
                        })
                elif capabilities.requires_pairing:
                    if not pair_budgets:
                        continue
                    for pair_budget in pair_budgets:
                        pair_row = plan["pair_controls"][str(pair_budget)]["controls"]
                        declared_control = str(specification.get("pair_control", "true"))
                        controls: list[tuple[str, Mapping[str, Any], float, float]] = [
                            ("true", pair_row["true"], 0.0,
                             float(pair_row["true"]["actual_noise_fraction"]))
                        ]
                        if declared_control == "shuffled":
                            if pair_row["shuffled"].get("status") == "unavailable":
                                continue
                            controls = [(
                                "shuffled", pair_row["shuffled"], 1.0,
                                float(pair_row["shuffled"]["actual_noise_fraction"]),
                            )]
                        elif declared_control != "true":
                            raise ValueError(f"unsupported declared pair control: {declared_control}")
                        elif (specification.get("noise_sweep") is True
                              and noise_rates and any(value > 0 for value in noise_rates)):
                            # True method row plus separately named noisy diagnostic rows.
                            controls.extend(
                                (
                                    "noisy", pair_row["noisy"][format(rate, ".6g")],
                                    rate,
                                    float(pair_row["noisy"][format(rate, ".6g")][
                                        "actual_noise_fraction"
                                    ]),
                                )
                                for rate in noise_rates if rate > 0 and pair_budget >= 2
                            )
                        for control, control_row, requested_noise, actual_noise in controls:
                            pair_ref_index = _indices(reference.ids, control_row["reference_ids"],
                                                      name="paired reference fit")
                            pair_target_index = _indices(target.ids, control_row["target_ids"],
                                                         name="paired target fit")
                            pair_reference = np.asarray(reference.features[pair_ref_index], dtype=np.float64)
                            pair_source = np.asarray(target.features[pair_target_index], dtype=np.float64)
                            fit_rng = np.random.default_rng(_seed(
                                global_seed, "fit", plan["manifest_sha256"], method_name,
                                calibration_size, pair_budget, requested_noise,
                            ))
                            # Pair assignment is already materialized by the split plan; pass it as
                            # true rows to the estimator so no second random corruption is introduced.
                            metrics, diagnostics = _fit_and_evaluate(
                                method_name=method_name, method_specification=specification,
                                store=store, source_fit=pair_source,
                                reference_fit=pair_reference, source_eval=source_eval,
                                reference_eval=reference_eval, labels=eval_labels, weight=weight,
                                bias=bias, fit_rng=fit_rng,
                                evaluation_context=evaluation_context,
                                pair_control=control,
                                pair_noise_fraction=actual_noise,
                                requested_pair_noise_fraction=requested_noise,
                                metrics_projections=metrics_projections,
                                mmd_features=mmd_features,
                                distribution_evaluation_limit=distribution_evaluation_limit,
                            )
                            label = (method_name if control != "noisy"
                                     else f"{method_name}_noise{requested_noise:g}")
                            rows.append({
                                **_row_base(suite=suite_name, store=store, condition=condition,
                                    split_seed=split_seed, calibration_size=calibration_size,
                                    evaluation_size=evaluation_size, method=label,
                                    pair_budget=pair_budget,
                                    pair_noise_rate=requested_noise,
                                    pair_noise_rate_actual=actual_noise,
                                    split_plan=plan),
                                "fit_information": specification["information"],
                                "pair_control": control,
                                "diagnostics": diagnostics,
                                **metrics,
                            })
                else:
                    fit_rng = np.random.default_rng(_seed(
                        global_seed, "fit", plan["manifest_sha256"], method_name,
                        calibration_size,
                    ))
                    metrics, diagnostics = _fit_and_evaluate(
                        method_name=method_name, method_specification=specification,
                        store=store, source_fit=source_marginal_fit,
                        reference_fit=reference_marginal_fit, source_eval=source_eval,
                        reference_eval=reference_eval, labels=eval_labels, weight=weight,
                        bias=bias, fit_rng=fit_rng,
                        evaluation_context=evaluation_context,
                        metrics_projections=metrics_projections,
                        mmd_features=mmd_features,
                        distribution_evaluation_limit=distribution_evaluation_limit,
                    )
                    rows.append({
                        **_row_base(suite=suite_name, store=store, condition=condition,
                            split_seed=split_seed, calibration_size=calibration_size,
                            evaluation_size=evaluation_size, method=method_name,
                            pair_budget=None, pair_noise_rate=None,
                            pair_noise_rate_actual=None, split_plan=plan),
                        "fit_information": specification["information"],
                        "pair_control": None,
                        "diagnostics": diagnostics,
                        **metrics,
                    })
    return rows


SEMANTIC_KEY_FIELDS = (
    "suite", "dataset", "encoder", "corruption", "severity", "split_seed",
    "calibration_size", "evaluation_size", "method", "pair_budget",
    "pair_control", "pair_noise_rate_requested", "pair_noise_rate_actual",
)


def semantic_row_key(row: Mapping[str, Any]) -> str:
    """Canonical key for one scientifically distinct result cell."""

    payload: dict[str, Any] = {}
    for field in SEMANTIC_KEY_FIELDS:
        value = row.get(field)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not np.isfinite(value):
                raise IntegrityError(f"semantic field {field} must be finite or null")
            value = 0.0 if value == 0.0 else value
        payload[field] = value
    return json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )


def _expected_noise_actual(pair_budget: int, requested: float) -> float:
    if requested == 0.0:
        return 0.0
    changed = min(int(pair_budget), max(2, int(round(float(requested) * pair_budget))))
    return float(changed / pair_budget)


def expected_unit_semantic_keys(
    *, suite_name: str, suite: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]], condition: Mapping[str, Any],
) -> set[str]:
    """Construct the exact locked result grid for one suite/condition unit."""

    budgets = [int(value) for value in suite.get("pair_budgets", [])]
    noise_rates = [float(value) for value in suite.get("pair_noise_rates", [0.0])]
    result: set[str] = set()

    def add(
        *, split_seed: int, calibration_size: int, method: str,
        pair_budget: int | None, pair_control: str | None,
        requested_noise: float | None, actual_noise: float | None,
    ) -> None:
        key = semantic_row_key({
            "suite": suite_name,
            "dataset": condition["dataset"],
            "encoder": condition["encoder"],
            "corruption": condition.get("corruption"),
            "severity": condition.get("severity"),
            "split_seed": int(split_seed),
            "calibration_size": int(calibration_size),
            "evaluation_size": int(condition["evaluation_size"]),
            "method": method,
            "pair_budget": pair_budget,
            "pair_control": pair_control,
            "pair_noise_rate_requested": requested_noise,
            "pair_noise_rate_actual": actual_noise,
        })
        if key in result:
            raise IntegrityError(f"configured semantic grid contains duplicate cell: {key}")
        result.add(key)

    for split_seed in suite["split_seeds"]:
        for calibration_size in suite["calibration_sizes"]:
            for method_name in suite["methods"]:
                specification = methods[method_name]
                if not _method_available_at_calibration(
                    specification, int(calibration_size)
                ):
                    continue
                implementation, _ = _method_kwargs(specification)
                if implementation == "labeled_er":
                    add(
                        split_seed=int(split_seed),
                        calibration_size=int(calibration_size),
                        method=method_name, pair_budget=None, pair_control=None,
                        requested_noise=None, actual_noise=None,
                    )
                    continue
                capabilities = method_capabilities(implementation)
                endpoint_matched = specification.get("fit_on_pair_endpoint_sets") is True
                if endpoint_matched:
                    if capabilities.requires_pairing or capabilities.uses_labels:
                        raise IntegrityError(
                            f"{method_name}: endpoint-matched control must be marginal-only"
                        )
                    if not budgets:
                        raise IntegrityError(
                            f"{suite_name}/{method_name}: pair_budgets are required"
                        )
                    for budget in budgets:
                        add(
                            split_seed=int(split_seed),
                            calibration_size=int(calibration_size),
                            method=method_name, pair_budget=budget,
                            pair_control="assignment_erased",
                            requested_noise=None, actual_noise=None,
                        )
                elif capabilities.requires_pairing:
                    if not budgets:
                        raise IntegrityError(
                            f"{suite_name}/{method_name}: pair_budgets are required"
                        )
                    declared_control = str(specification.get("pair_control", "true"))
                    if declared_control not in {"true", "shuffled"}:
                        raise IntegrityError(
                            f"{method_name}: unsupported pair_control={declared_control!r}"
                        )
                    for budget in budgets:
                        if budget < 2 and declared_control == "shuffled":
                            continue
                        if declared_control == "shuffled":
                            add(
                                split_seed=int(split_seed),
                                calibration_size=int(calibration_size),
                                method=method_name, pair_budget=budget,
                                pair_control="shuffled", requested_noise=1.0,
                                actual_noise=1.0,
                            )
                            continue
                        add(
                            split_seed=int(split_seed),
                            calibration_size=int(calibration_size),
                            method=method_name, pair_budget=budget,
                            pair_control="true", requested_noise=0.0,
                            actual_noise=0.0,
                        )
                        if specification.get("noise_sweep") is True:
                            for rate in noise_rates:
                                if rate <= 0.0 or budget < 2:
                                    continue
                                add(
                                    split_seed=int(split_seed),
                                    calibration_size=int(calibration_size),
                                    method=f"{method_name}_noise{rate:g}",
                                    pair_budget=budget, pair_control="noisy",
                                    requested_noise=rate,
                                    actual_noise=_expected_noise_actual(budget, rate),
                                )
                else:
                    add(
                        split_seed=int(split_seed),
                        calibration_size=int(calibration_size),
                        method=method_name, pair_budget=None, pair_control=None,
                        requested_noise=None, actual_noise=None,
                    )
    return result


def validate_semantic_grid(
    rows: Iterable[Mapping[str, Any]], expected_keys: Iterable[str],
    *, scope: str = "prespecified_expected",
) -> dict[str, Any]:
    """Fail on every missing, extra, or duplicate scientific result cell."""

    expected = set(expected_keys)
    observed_list = [semantic_row_key(row) for row in rows]
    counts: dict[str, int] = {}
    for key in observed_list:
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    observed = set(observed_list)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if duplicates or missing or extra:
        raise IntegrityError(
            "real semantic grid mismatch: "
            f"missing={len(missing)} extra={len(extra)} duplicates={len(duplicates)}; "
            f"missing_preview={missing[:2]} extra_preview={extra[:2]} "
            f"duplicate_preview={duplicates[:2]}"
        )
    return {
        "status": "PASS",
        "scope": scope,
        "expected_count": len(expected),
        "observed_count": len(observed_list),
        "duplicate_count": 0,
        "missing_count": 0,
        "extra_count": 0,
        "expected_grid_sha256": sha256_json(sorted(expected)),
        "observed_grid_sha256": sha256_json(sorted(observed)),
    }


def finalize_rows(
    rows: list[dict[str, Any]], output_dir: Path, *,
    expected_semantic_keys: Iterable[str] | None = None,
    unit_records: Iterable[Mapping[str, Any]] = (),
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    rows = sorted(rows, key=semantic_row_key)
    observed_keys = [semantic_row_key(row) for row in rows]
    explicit_expected = expected_semantic_keys is not None
    grid = validate_semantic_grid(
        rows,
        observed_keys if expected_semantic_keys is None else expected_semantic_keys,
        scope="prespecified_expected" if explicit_expected else "observed_only",
    )
    _atomic_jsonl(rows_path, rows)
    summary = macro_summary(rows, METRICS)
    write_json_atomic(summary_path, {
        "schema_version": "marginal_alignment.campaign.real_summary.v2",
        "rows": len(rows),
        "groups": summary,
    })
    write_json_atomic(manifest_path, {
        "schema_version": "marginal_alignment.campaign.real_manifest.v2",
        "status": "PASS",
        "record_count": len(rows),
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
        "semantic_grid_validation": grid,
        "semantic_grid_sha256": grid["observed_grid_sha256"],
        "units": sorted(
            [dict(record) for record in unit_records],
            key=lambda record: str(record.get("unit_key", "")),
        ),
    })
    return [rows_path, summary_path, manifest_path]
