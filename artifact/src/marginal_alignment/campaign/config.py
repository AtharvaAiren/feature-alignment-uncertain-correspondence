"""Strict configuration loading for the evaluation campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised before compute when a campaign is ambiguous or incomplete."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def validate_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == "marginal_alignment.campaign.campaign.v2",
             "unsupported or missing schema_version")
    _require(bool(config.get("campaign_id")), "campaign_id is required")
    _require(isinstance(config.get("datasets"), list) and config["datasets"],
             "at least one dataset is required")
    _require(isinstance(config.get("suites"), dict) and config["suites"],
             "at least one experimental suite is required")
    dataset_names: set[str] = set()
    for dataset in config["datasets"]:
        name = str(dataset.get("name", ""))
        _require(bool(name) and name not in dataset_names, f"invalid/duplicate dataset {name!r}")
        dataset_names.add(name)
        _require(dataset.get("reference_key") and dataset.get("target_key"),
                 f"{name}: reference_key and target_key are required")
        _require(dataset.get("encoders"), f"{name}: encoders cannot be empty")
        _require(dataset.get("corruptions"), f"{name}: corruptions cannot be empty")
        _require(dataset.get("severities"), f"{name}: severities cannot be empty")
        _require(int(dataset.get("evaluation_size", 0)) >= 128,
                 f"{name}: evaluation_size must be >=128")
        _require(isinstance(dataset.get("head_protocol"), dict),
                 f"{name}: a structured head_protocol is required")
    methods = config.get("methods", {})
    _require(isinstance(methods, dict) and methods, "methods registry is required")
    diagnostics = config.get("diagnostic_methods", {})
    _require(isinstance(diagnostics, dict), "diagnostic_methods must be an object")
    all_method_names = set(methods) | set(diagnostics)
    for suite_name, suite in config["suites"].items():
        _require(suite.get("datasets"), f"{suite_name}: datasets cannot be empty")
        unknown = set(suite["datasets"]) - dataset_names
        _require(not unknown, f"{suite_name}: unknown datasets {sorted(unknown)}")
        _require(suite.get("methods"), f"{suite_name}: methods cannot be empty")
        unknown_methods = set(suite["methods"]) - all_method_names
        _require(not unknown_methods,
                 f"{suite_name}: unknown methods {sorted(unknown_methods)}")
        sizes = [int(item) for item in suite.get("calibration_sizes", [])]
        _require(sizes and sizes == sorted(set(sizes)),
                 f"{suite_name}: calibration_sizes must be sorted and unique")
        seeds = [int(item) for item in suite.get("split_seeds", [])]
        _require(seeds and len(seeds) == len(set(seeds)),
                 f"{suite_name}: split_seeds must be unique")
        budgets = [int(item) for item in suite.get("pair_budgets", [])]
        if budgets:
            _require(budgets == sorted(set(budgets)) and min(budgets) >= 2,
                     f"{suite_name}: pair_budgets must be sorted, unique, and >=2")
        rates = [float(item) for item in suite.get("pair_noise_rates", [])]
        _require(all(0.0 <= item <= 1.0 for item in rates),
                 f"{suite_name}: pair_noise_rates must lie in [0,1]")
    primary = config.get("inference", {}).get("primary_contrasts", [])
    _require(primary, "at least one primary contrast must be locked")
    for contrast in primary:
        def valid_contrast_arm(value: Any) -> bool:
            label = str(value)
            return label in all_method_names or any(
                label.startswith(f"{name}_noise") for name in all_method_names
            )
        _require(valid_contrast_arm(contrast.get("left"))
                 and valid_contrast_arm(contrast.get("right")),
                 f"invalid primary contrast {contrast}")
        _require(contrast.get("metric"), f"contrast metric missing: {contrast}")
    synthetic = config.get("synthetic")
    _require(isinstance(synthetic, dict), "synthetic grid is required")
    required_synthetic = {
        "exact_indistinguishable", "approximate_two_world",
        "identifiable_nongaussian", "pair_rank_phase", "real_embedding_orbit",
    }
    _require(required_synthetic.issubset(synthetic),
             f"synthetic grid is missing {sorted(required_synthetic - set(synthetic))}")
    streaming = config.get("streaming_secondary")
    _require(isinstance(streaming, dict), "streaming_secondary protocol is required")
    if streaming.get("enabled") is True:
        unknown_datasets = set(streaming.get("datasets", [])) - dataset_names
        _require(streaming.get("datasets") and not unknown_datasets,
                 f"streaming_secondary has invalid datasets {sorted(unknown_datasets)}")
        declared_encoders = {
            str(encoder) for dataset in config["datasets"] for encoder in dataset["encoders"]
        }
        unknown_encoders = set(streaming.get("encoders", [])) - declared_encoders
        _require(streaming.get("encoders") and not unknown_encoders,
                 f"streaming_secondary has invalid encoders {sorted(unknown_encoders)}")
        _require(int(streaming.get("stream_rows", 0)) >= 2 * int(
            streaming.get("block_size", 0)
        ), "streaming_secondary requires at least two blocks")
        _require(int(streaming.get("reference_calibration_rows", 0)) >= 2,
                 "streaming_secondary reference calibration is too small")
        order_seeds = [int(value) for value in streaming.get("order_seeds", [])]
        _require(order_seeds and len(order_seeds) == len(set(order_seeds)),
                 "streaming_secondary order_seeds must be unique")
    lock_path = config.get("lock_path")
    if lock_path:
        _require(Path(lock_path).expanduser().is_file(), f"lock_path is missing: {lock_path}")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError("top-level configuration must be an object")
    config = _expand(raw)
    if "lock_path" not in config:
        candidate = source.parents[2] / "protocol" / "experiment_lock.json"
        if candidate.is_file():
            config["lock_path"] = str(candidate.resolve())
    validate_config(config)
    return config, source, config_sha256(config)
