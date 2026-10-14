#!/usr/bin/env python3
"""Deterministically enumerate and launch the locked Play-3 unit panel.

This is intentionally a thin orchestration layer.  Scientific work remains in
``run_play3.py``; this module only validates the complete store inventory,
partitions immutable unit identities, chooses fresh output paths, and delegates
one unit at a time without reading any resulting outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_json,
    verify_manifest_checksum,
)
from marginal_alignment.campaign.feature_store import NATIVE_MANIFEST, NATIVE_SCHEMA
from marginal_alignment.iclr_extension.natural_shift import (
    NATURAL_SHIFT_RESULT_SCHEMA,
    capture_play3_runtime,
    load_locked_play3_config,
)
from scripts.iclr_extension.run_play3 import main as run_play3_main


PANEL_PLAN_SCHEMA = "marginal_alignment.iclr_extension.play3_panel_plan.v1"
STORE_MAP_SCHEMA = "marginal_alignment.iclr_extension.play3_store_map.v1"
EXPECTED_ENCODERS = ("dinov3", "clip", "vitb16")
EXPECTED_SEEDS = (201, 202, 203, 204, 205)
EXPECTED_BUDGETS = (0.05, 0.01, 0.1)
EXPECTED_CORRUPTIONS = (
    "brightness",
    "defocus_blur",
    "fog",
    "gaussian_noise",
    "jpeg_compression",
    "motion_blur",
)
EXPECTED_NATURAL_UNITS = 45
EXPECTED_POSITIVE_UNITS = 270
EXPECTED_TOTAL_UNITS = EXPECTED_NATURAL_UNITS + EXPECTED_POSITIVE_UNITS


@dataclass(frozen=True)
class PanelUnit:
    global_index: int
    panel: str
    dataset: str
    encoder: str
    corruption: str | None
    severity: int | None
    split_seed: int
    cost_budget_relative: float
    unit_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_index": int(self.global_index),
            "panel": self.panel,
            "dataset": self.dataset,
            "encoder": self.encoder,
            "corruption": self.corruption,
            "severity": self.severity,
            "split_seed": int(self.split_seed),
            "cost_budget_relative": float(self.cost_budget_relative),
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True)
class ValidatedStores:
    natural: Mapping[str, Path]
    positive: Mapping[tuple[str, str], Path]
    records: tuple[Mapping[str, Any], ...]
    protected_roots: tuple[Path, ...]

    def manifest_for(self, unit: PanelUnit) -> Path:
        if unit.panel == "natural":
            return self.natural[unit.encoder]
        if unit.corruption is None:
            raise IntegrityError("positive-control unit lacks a corruption")
        return self.positive[(unit.encoder, unit.corruption)]


def _budget_code(value: float) -> str:
    scaled = int(round(10000 * float(value)))
    if abs(float(value) - scaled / 10000.0) > 1e-15:
        raise IntegrityError(f"cost budget lacks an exact basis-point identity: {value}")
    return f"{scaled:04d}bp"


def _locked_axes(config: Mapping[str, Any]) -> tuple[
    tuple[str, ...], tuple[int, ...], tuple[float, ...], tuple[str, ...], int
]:
    encoders = tuple(map(str, config.get("encoders", ())))
    ceiling = config.get("ceiling_protocol")
    control = config.get("positive_control")
    dataset = config.get("dataset")
    if not all(isinstance(value, Mapping) for value in (ceiling, control, dataset)):
        raise IntegrityError("Play-3 panel declarations are malformed")
    seeds = tuple(int(value) for value in ceiling.get("split_seeds", ()))
    budgets = (
        float(ceiling.get("primary_cost_budget_relative", float("nan"))),
        *(float(value) for value in ceiling.get("cost_budget_relative_sensitivity", ())),
    )
    corruptions = tuple(map(str, control.get("heldout_corruptions", ())))
    severity = int(control.get("severity", -1))
    if encoders != EXPECTED_ENCODERS:
        raise IntegrityError(f"locked Play-3 encoders changed: {encoders}")
    if seeds != EXPECTED_SEEDS:
        raise IntegrityError(f"locked Play-3 split seeds changed: {seeds}")
    if budgets != EXPECTED_BUDGETS or len({_budget_code(x) for x in budgets}) != 3:
        raise IntegrityError(f"locked Play-3 cost budgets changed: {budgets}")
    if corruptions != EXPECTED_CORRUPTIONS:
        raise IntegrityError(f"locked Play-3 positive controls changed: {corruptions}")
    if (
        dataset.get("name") != "imagenet_sketch"
        or control.get("dataset") != "imagenet_c"
        or severity != 5
    ):
        raise IntegrityError("locked Play-3 dataset or severity changed")
    return encoders, seeds, budgets, corruptions, severity


def enumerate_locked_panel(config: Mapping[str, Any]) -> tuple[PanelUnit, ...]:
    """Return the exact prespecified 45 natural and 270 positive units."""

    encoders, seeds, budgets, corruptions, severity = _locked_axes(config)
    units: list[PanelUnit] = []
    for seed in seeds:
        for encoder in encoders:
            for budget in budgets:
                units.append(PanelUnit(
                    global_index=len(units),
                    panel="natural",
                    dataset="imagenet_sketch",
                    encoder=encoder,
                    corruption=None,
                    severity=None,
                    split_seed=seed,
                    cost_budget_relative=budget,
                    unit_id=(
                        f"imagenet_sketch__{encoder}__seed_{seed}__"
                        f"budget_{_budget_code(budget)}"
                    ),
                ))
    natural_count = len(units)
    for seed in seeds:
        for corruption in corruptions:
            for encoder in encoders:
                for budget in budgets:
                    units.append(PanelUnit(
                        global_index=len(units),
                        panel="positive_control",
                        dataset="imagenet_c",
                        encoder=encoder,
                        corruption=corruption,
                        severity=severity,
                        split_seed=seed,
                        cost_budget_relative=budget,
                        unit_id=(
                            f"positive__{corruption}__{encoder}__seed_{seed}__"
                            f"budget_{_budget_code(budget)}"
                        ),
                    ))
    if (
        natural_count != EXPECTED_NATURAL_UNITS
        or len(units) - natural_count != EXPECTED_POSITIVE_UNITS
        or len(units) != EXPECTED_TOTAL_UNITS
        or len({unit.unit_id for unit in units}) != EXPECTED_TOTAL_UNITS
    ):
        raise IntegrityError("Play-3 panel enumeration is incomplete or nonunique")
    return tuple(units)


def select_shard(
    units: Sequence[PanelUnit], *, shard_index: int, num_shards: int
) -> tuple[PanelUnit, ...]:
    """Round-robin partition fixed global identities without data dependence."""

    if num_shards < 1 or num_shards > len(units):
        raise IntegrityError(
            f"num_shards must be in [1,{len(units)}], found {num_shards}"
        )
    if shard_index < 0 or shard_index >= num_shards:
        raise IntegrityError(
            f"shard_index must be in [0,{num_shards}), found {shard_index}"
        )
    selected = tuple(
        unit for unit in units if unit.global_index % num_shards == shard_index
    )
    if not selected:
        raise IntegrityError("the requested deterministic shard is empty")
    return selected


def _absolute_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{name} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise IntegrityError(f"{name} must be absolute: {value}")
    return path.resolve()


def validate_store_map(
    store_map_path: str | Path, config: Mapping[str, Any]
) -> ValidatedStores:
    """Validate all 21 manifest identities without opening unselected arrays.

    The delegated lower runner performs full body-hash verification for every
    store actually selected by this shard.  This structural pass prevents a
    missing or mis-keyed complete-panel map without multiplying large-array
    I/O by the number of shards.
    """

    source = Path(store_map_path).expanduser().resolve()
    if not source.is_file():
        raise IntegrityError(f"Play-3 store map is missing: {source}")
    value = read_json(source)
    if not isinstance(value, Mapping):
        raise IntegrityError("Play-3 store map must be a JSON object")
    if set(value) != {"schema_version", "natural_stores", "positive_control_root"}:
        raise IntegrityError("Play-3 store map has missing or undeclared fields")
    if value.get("schema_version") != STORE_MAP_SCHEMA:
        raise IntegrityError("unsupported Play-3 store-map schema")
    natural_values = value.get("natural_stores")
    if not isinstance(natural_values, Mapping):
        raise IntegrityError("natural_stores must be an encoder-to-path object")
    encoders, _, _, corruptions, severity = _locked_axes(config)
    if set(map(str, natural_values)) != set(encoders):
        raise IntegrityError("natural store map must contain exactly the locked encoders")
    positive_root = _absolute_path(
        value.get("positive_control_root"), name="positive_control_root"
    )
    if not positive_root.is_dir():
        raise IntegrityError(f"positive-control store root is missing: {positive_root}")

    def structural_store(declared_path: Path) -> tuple[Path, Path, dict[str, Any]]:
        manifest_path = (
            declared_path / NATIVE_MANIFEST if declared_path.is_dir() else declared_path
        ).resolve()
        if manifest_path.name != NATIVE_MANIFEST or not manifest_path.is_file():
            raise IntegrityError(f"native feature-store manifest is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != NATIVE_SCHEMA:
            raise IntegrityError(f"feature-store manifest has the wrong schema: {manifest_path}")
        verify_manifest_checksum(manifest)
        arrays = manifest.get("arrays")
        if not isinstance(arrays, Mapping) or not arrays:
            raise IntegrityError(f"feature-store arrays are not declared: {manifest_path}")
        root = manifest_path.parent.resolve()
        for role, record in arrays.items():
            if not isinstance(record, Mapping):
                raise IntegrityError(f"malformed feature-store array record: {role}")
            relative = safe_relative_path(str(record.get("path", "")))
            body = (root / relative).resolve()
            if body.parent != root and root not in body.parents:
                raise IntegrityError(f"feature-store array escapes its root: {role}")
            digest = str(record.get("sha256", ""))
            if (
                not body.is_file()
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or int(record.get("size_bytes", -1)) < 0
            ):
                raise IntegrityError(f"feature-store array declaration is invalid: {role}")
            if body.stat().st_size != int(record["size_bytes"]):
                raise IntegrityError(f"feature-store array size mismatch: {role}")
        return root, manifest_path, manifest

    natural: dict[str, Path] = {}
    positive: dict[tuple[str, str], Path] = {}
    records: list[Mapping[str, Any]] = []
    protected: set[Path] = {positive_root}
    seen_roots: set[Path] = set()
    for encoder in encoders:
        declared_path = _absolute_path(
            natural_values[encoder], name=f"natural_stores.{encoder}"
        )
        root, manifest_path, manifest = structural_store(declared_path)
        pairing = manifest.get("pairing", {})
        if (
            manifest.get("dataset") != "imagenet_sketch"
            or manifest.get("encoder") != encoder
            or not isinstance(pairing, Mapping)
            or pairing.get("status") != "unavailable"
        ):
            raise IntegrityError(f"natural store identity is invalid for {encoder}")
        if root in seen_roots:
            raise IntegrityError("two locked natural encoders resolve to one store")
        seen_roots.add(root)
        protected.add(root)
        natural[encoder] = manifest_path
        records.append({
            "panel": "natural",
            "dataset": "imagenet_sketch",
            "encoder": encoder,
            "corruption": None,
            "severity": None,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "body_hash_verification": "delegated_lower_runner_for_selected_shard_only",
        })

    for encoder in encoders:
        for corruption in corruptions:
            declared_path = (
                positive_root / "imagenet_c" / encoder / corruption
                / f"severity_{severity}"
            )
            root, manifest_path, manifest = structural_store(declared_path)
            metadata = manifest.get("metadata", {})
            pairing = manifest.get("pairing", {})
            if (
                manifest.get("dataset") != "imagenet_c"
                or manifest.get("encoder") != encoder
                or metadata.get("corruption") != corruption
                or int(metadata.get("severity", -1)) != severity
                or not isinstance(pairing, Mapping)
                or pairing.get("status") != "available"
            ):
                raise IntegrityError(
                    f"positive-control store identity is invalid: {encoder}/{corruption}"
                )
            if root in seen_roots:
                raise IntegrityError("two Play-3 panel conditions resolve to one store")
            seen_roots.add(root)
            positive[(encoder, corruption)] = manifest_path
            records.append({
                "panel": "positive_control",
                "dataset": "imagenet_c",
                "encoder": encoder,
                "corruption": corruption,
                "severity": severity,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "body_hash_verification": "delegated_lower_runner_for_selected_shard_only",
            })
    if len(natural) != 3 or len(positive) != 18 or len(records) != 21:
        raise IntegrityError("Play-3 store inventory is incomplete")
    return ValidatedStores(
        natural=natural,
        positive=positive,
        records=tuple(records),
        protected_roots=tuple(sorted(protected, key=str)),
    )


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _unit_output(output_root: Path, unit: PanelUnit) -> Path:
    folder = "natural" if unit.panel == "natural" else "positive_control"
    return output_root / folder / f"{unit.unit_id}.json"


def _lower_arguments(
    unit: PanelUnit,
    *,
    config_path: Path,
    stores: ValidatedStores,
    provenance: Path,
    dataset_root: Path,
    archive: Path,
    output: Path,
) -> list[str]:
    arguments = [
        "natural" if unit.panel == "natural" else "positive-control",
        "--config", str(config_path),
        "--store", str(stores.manifest_for(unit)),
        "--split-seed", str(unit.split_seed),
        "--cost-budget-relative", repr(float(unit.cost_budget_relative)),
        "--output", str(output),
    ]
    if unit.panel == "natural":
        arguments.extend([
            "--provenance", str(provenance),
            "--dataset-root", str(dataset_root),
            "--archive", str(archive),
        ])
    return arguments


def _validate_completed_unit_identity(
    path: Path,
    *,
    unit: PanelUnit,
    config_path: Path,
    lock: Mapping[str, Any],
    store_manifest: Path,
    provenance: Path,
    runtime: Mapping[str, Any],
) -> None:
    """Validate resume identity/provenance without reading scientific metrics."""

    result = read_json(path)
    expected_schema = (
        NATURAL_SHIFT_RESULT_SCHEMA
        if unit.panel == "natural"
        else "marginal_alignment.iclr_extension.positive_control.v2"
    )
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != expected_schema
        or result.get("status") != "PASS"
        or result.get("scientific_result") is not True
        or result.get("immutable_fresh_output") is not True
    ):
        raise IntegrityError(f"completed Play-3 unit is malformed: {path}")
    verify_manifest_checksum(result)
    plan = result.get("unit_plan")
    if not isinstance(plan, Mapping):
        raise IntegrityError(f"completed Play-3 unit lacks its immutable plan: {path}")
    exact = {
        "unit_id": unit.unit_id,
        "dataset": unit.dataset,
        "encoder": unit.encoder,
        "split_seed": unit.split_seed,
        "cost_budget_relative": unit.cost_budget_relative,
        "store_manifest": str(store_manifest),
        "store_manifest_sha256": sha256_file(store_manifest),
        "config_sha256": sha256_file(config_path),
        "runtime": dict(runtime),
    }
    if unit.panel == "positive_control":
        exact.update({"corruption": unit.corruption, "severity": unit.severity})
    mismatches = {
        key: {"expected": expected, "actual": plan.get(key)}
        for key, expected in exact.items() if plan.get(key) != expected
    }
    if mismatches:
        raise IntegrityError(
            f"completed Play-3 unit identity drifted: {unit.unit_id}: {mismatches}"
        )
    if unit.panel == "natural":
        if (
            plan.get("lock_manifest_sha256") != sha256_json(lock)
            or plan.get("dataset_provenance_manifest_sha256") != sha256_file(provenance)
        ):
            raise IntegrityError(
                f"completed natural unit lock or provenance drifted: {unit.unit_id}"
            )
        source_binding = plan.get("source_campaign_reference_store")
    else:
        if plan.get("extension_lock") != lock:
            raise IntegrityError(
                f"completed positive-control unit lock drifted: {unit.unit_id}"
            )
        source_binding = plan.get("source_campaign_store")
    if not isinstance(source_binding, Mapping) or source_binding.get("status") != "PASS":
        raise IntegrityError(
            f"completed Play-3 source-store provenance drifted: {unit.unit_id}"
        )
    if unit.panel == "natural":
        generated_store = read_json(store_manifest)
        generated_binding = generated_store.get("metadata", {}).get(
            "source_campaign_reference_store"
        ) if isinstance(generated_store, Mapping) else None
        declared = lock.get("base_campaign_feature_stores")
        condition = source_binding.get("condition")
        if (
            not isinstance(generated_binding, Mapping)
            or dict(source_binding) != dict(generated_binding)
            or source_binding.get("schema_version")
            != "marginal_alignment.iclr_extension.base_campaign_store_binding.v1"
            or source_binding.get("actual_manifest_sha256")
            != source_binding.get("expected_manifest_sha256")
            or not isinstance(declared, Mapping)
            or not isinstance(condition, Mapping)
            or condition != {
                "dataset": str(declared.get("natural_reference_condition", {}).get(
                    "dataset", ""
                )),
                "encoder": unit.encoder,
                "corruption": str(declared.get("natural_reference_condition", {}).get(
                    "corruption", ""
                )),
                "severity": int(declared.get("natural_reference_condition", {}).get(
                    "severity", -1
                )),
            }
            or source_binding.get("external_input_manifest_file_sha256")
            != declared.get("external_input_manifest_file_sha256")
            or source_binding.get("external_input_sha256")
            != declared.get("external_input_sha256")
        ):
            raise IntegrityError(
                f"completed natural source-store provenance drifted: {unit.unit_id}"
            )
    elif source_binding.get("actual_manifest_sha256") != sha256_file(store_manifest):
        raise IntegrityError(
            f"completed positive-control source-store provenance drifted: {unit.unit_id}"
        )


def orchestrate_panel(
    *,
    config_path: str | Path,
    store_map_path: str | Path,
    provenance_path: str | Path,
    dataset_root: str | Path,
    archive_path: str | Path,
    output_root: str | Path,
    shard_index: int,
    num_shards: int,
    plan_only: bool,
    resume: bool = False,
    lower_main: Callable[[list[str]], int] = run_play3_main,
) -> dict[str, Any]:
    """Validate, partition, and optionally delegate a complete panel shard."""

    config, resolved_config, lock, _ = load_locked_play3_config(config_path)
    units = enumerate_locked_panel(config)
    selected = select_shard(
        units, shard_index=int(shard_index), num_shards=int(num_shards)
    )
    stores = validate_store_map(store_map_path, config)
    provenance = Path(provenance_path).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    archive = Path(archive_path).expanduser().resolve()
    store_map = Path(store_map_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    runtime = capture_play3_runtime()
    if not provenance.is_file() or not archive.is_file() or not dataset.is_dir():
        raise IntegrityError("natural-shift provenance, archive, or dataset root is missing")
    if output.exists() and not output.is_dir():
        raise IntegrityError("Play-3 output root exists and is not a directory")
    protected = {
        resolved_config.resolve(), store_map, provenance, archive, dataset,
        *stores.protected_roots,
    }
    for value in protected:
        if _overlap(output, value):
            raise IntegrityError(f"Play-3 output root overlaps a protected input: {value}")

    planned_outputs = [_unit_output(output, unit) for unit in selected]
    if len(set(planned_outputs)) != len(planned_outputs):
        raise IntegrityError("selected Play-3 units do not have unique output paths")
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not resume:
        raise IntegrityError(
            f"selected Play-3 output is not fresh: {existing[0]}"
        )
    completed: set[Path] = set()
    if resume:
        for unit, path in zip(selected, planned_outputs):
            if not path.exists():
                continue
            _validate_completed_unit_identity(
                path,
                unit=unit,
                config_path=resolved_config,
                lock=lock,
                store_manifest=stores.manifest_for(unit),
                provenance=provenance,
                runtime=runtime,
            )
            completed.add(path)
    plan = {
        "schema_version": PANEL_PLAN_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "compute_started": False,
        "outcomes_inspected": False,
        "resume_mode": bool(resume),
        "resume_validation_scope": "identity_provenance_runtime_only_not_metrics",
        "runtime": runtime,
        "config": str(resolved_config),
        "config_sha256": sha256_file(resolved_config),
        "extension_lock_manifest_sha256": sha256_json(lock),
        "store_map": str(store_map),
        "store_map_sha256": sha256_file(store_map),
        "complete_panel": {
            "natural_units": EXPECTED_NATURAL_UNITS,
            "positive_control_units": EXPECTED_POSITIVE_UNITS,
            "total_units": EXPECTED_TOTAL_UNITS,
        },
        "shard": {
            "index": int(shard_index),
            "count": int(num_shards),
            "selection_rule": "global_index_mod_num_shards_equals_shard_index",
            "selected_units": len(selected),
        },
        "output_root": str(output),
        "fresh_selected_outputs_verified": not completed,
        "completed_units_verified_for_resume": len(completed),
        "store_inventory": list(stores.records),
        "units": [
            {
                **unit.as_dict(),
                "store_manifest": str(stores.manifest_for(unit)),
                "store_manifest_sha256": sha256_file(stores.manifest_for(unit)),
                "output": str(path),
            }
            for unit, path in zip(selected, planned_outputs)
        ],
    }
    if plan_only:
        return plan

    delegated = 0
    for unit, destination in zip(selected, planned_outputs):
        if destination in completed:
            continue
        status = lower_main(_lower_arguments(
            unit,
            config_path=resolved_config,
            stores=stores,
            provenance=provenance,
            dataset_root=dataset,
            archive=archive,
            output=destination,
        ))
        if status != 0:
            raise RuntimeError(
                f"lower-level Play-3 runner failed for {unit.unit_id}: {status}"
            )
        delegated += 1
    return {
        **plan,
        "status": "PASS",
        "compute_started": True,
        "delegated_units": delegated,
        "resumed_units": len(completed),
        "lower_level_outcomes_read_by_orchestrator": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or launch a deterministic shard of the locked Play-3 panel"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--store-map", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip only checksum-verified units with exact current identity bindings",
    )
    arguments = parser.parse_args(argv)
    result = orchestrate_panel(
        config_path=arguments.config,
        store_map_path=arguments.store_map,
        provenance_path=arguments.provenance,
        dataset_root=arguments.dataset_root,
        archive_path=arguments.archive,
        output_root=arguments.output_root,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
        plan_only=arguments.plan_only,
        resume=arguments.resume,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
