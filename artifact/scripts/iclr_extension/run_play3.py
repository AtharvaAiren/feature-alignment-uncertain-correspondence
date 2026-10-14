#!/usr/bin/env python3
"""Build, preflight, and run immutable Play-3 experiment units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.artifacts import IntegrityError, sha256_file
from marginal_alignment.campaign.feature_store import load_feature_store
from marginal_alignment.iclr_extension.aggregation import aggregate_play3_panel
from marginal_alignment.iclr_extension.ceiling import CorrespondenceCeilingProtocol
from marginal_alignment.iclr_extension.natural_shift import (
    _load_locked_base_campaign_feature_store_evidence,
    build_imagenet_sketch_feature_store,
    capture_play3_runtime,
    load_locked_play3_config,
    main as natural_unit_main,
    run_paired_positive_control,
    verify_base_campaign_feature_store,
    write_immutable_unit_result,
)


def _protocol(config: dict, budget: float) -> CorrespondenceCeilingProtocol:
    declared = config["ceiling_protocol"]
    allowed = {
        float(declared["primary_cost_budget_relative"]),
        *map(float, declared["cost_budget_relative_sensitivity"]),
    }
    if float(budget) not in allowed:
        raise IntegrityError(f"cost budget is outside the Play-3 lock: {budget}")
    return CorrespondenceCeilingProtocol(
        cost_budget_relative=float(budget),
        covariance_estimator=declared["covariance_estimator"],
        covariance_relative_floor=float(declared["covariance_relative_floor"]),
        maximum_witness_rows=int(declared["maximum_witness_rows"]),
        swap_neighbors=int(declared["swap_neighbors"]),
        swap_neighbor_sensitivity=tuple(
            int(value) for value in declared["swap_neighbor_sensitivity"]
        ),
        assignment_tie_relative_tolerance=float(
            declared["assignment_tie_relative_tolerance"]
        ),
    )


def _positive_control(arguments: argparse.Namespace) -> int:
    config, config_path, lock, repository = load_locked_play3_config(arguments.config)
    store = load_feature_store(arguments.store, verify_hashes=True)
    panel = config["positive_control"]
    metadata = store.manifest.get("metadata", {})
    corruption = str(metadata.get("corruption", ""))
    severity = int(metadata.get("severity", -1))
    if store.dataset != panel["dataset"]:
        raise IntegrityError("positive-control store has the wrong dataset")
    if store.encoder not in config["encoders"]:
        raise IntegrityError("positive-control encoder is outside the Play-3 lock")
    if corruption not in panel["heldout_corruptions"]:
        raise IntegrityError("corruption is outside the locked positive-control panel")
    if severity != int(panel["severity"]):
        raise IntegrityError("positive-control severity is outside the Play-3 lock")
    if store.pairing_spec.get("status") != "available":
        raise IntegrityError("positive-control store lacks verified pairing")
    source_evidence = _load_locked_base_campaign_feature_store_evidence(
        config, lock, repository
    )
    source_store_binding = verify_base_campaign_feature_store(
        store,
        source_evidence,
        dataset=str(panel["dataset"]),
        encoder=store.encoder,
        corruption=corruption,
        severity=severity,
    )
    seed = int(arguments.split_seed)
    if seed not in [int(value) for value in config["ceiling_protocol"]["split_seeds"]]:
        raise IntegrityError("positive-control seed is outside the Play-3 lock")
    protocol = _protocol(config, float(arguments.cost_budget_relative))
    unit_id = (
        f"positive__{corruption}__{store.encoder}__seed_{seed}__"
        f"budget_{int(round(10000 * float(arguments.cost_budget_relative))):04d}bp"
    )
    plan = {
        "schema_version": "marginal_alignment.iclr_extension.positive_control_plan.v1",
        "status": "PASS",
        "scientific_result": False,
        "compute_started": False,
        "unit_id": unit_id,
        "dataset": store.dataset,
        "encoder": store.encoder,
        "corruption": corruption,
        "severity": severity,
        "split_seed": seed,
        "cost_budget_relative": float(arguments.cost_budget_relative),
        "calibration_rows": int(config["ceiling_protocol"]["calibration_rows"]),
        "witness_rows": int(config["ceiling_protocol"]["witness_rows"]),
        "store_manifest": str(store.manifest_path),
        "store_manifest_sha256": sha256_file(store.manifest_path),
        "source_campaign_store": source_store_binding,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "extension_lock": lock,
        "pairing_used_for_split_administration_before_fit": True,
        "labels_used_for_administrative_pair_validation_before_fit": True,
        "pairing_unavailable_to_estimator": True,
        "labels_unavailable_to_estimator": True,
        "labels_used_by_estimator": False,
        "protocol": protocol.as_dict(),
        "runtime": capture_play3_runtime(),
    }
    if arguments.plan_only:
        if arguments.output is not None:
            raise IntegrityError("--plan-only does not accept --output")
        print(json.dumps(plan, sort_keys=True))
        return 0
    if arguments.output is None:
        raise IntegrityError("a full positive-control unit requires --output")
    output = arguments.output.expanduser().resolve()
    if output.exists() or store.root.resolve() in output.parents:
        raise IntegrityError("positive-control output is not a fresh disjoint path")
    declared = config["ceiling_protocol"]
    result = run_paired_positive_control(
        store,
        source_campaign_store=source_store_binding,
        split_seed=seed,
        calibration_rows=int(declared["calibration_rows"]),
        witness_rows=int(declared["witness_rows"]),
        protocol=protocol,
    )
    destination = write_immutable_unit_result(output, {
        **result,
        "unit_plan": plan,
        "scientific_result": True,
        "immutable_fresh_output": True,
    })
    print(json.dumps({
        "status": "PASS",
        "unit_id": unit_id,
        "output": str(destination),
        "output_sha256": sha256_file(destination),
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-store")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--provenance", required=True, type=Path)
    build.add_argument("--dataset-root", required=True, type=Path)
    build.add_argument("--archive", required=True, type=Path)
    build.add_argument("--reference-store", required=True, type=Path)
    build.add_argument("--encoder", required=True)
    build.add_argument("--output-store", required=True, type=Path)
    build.add_argument("--device", default="cuda")
    build.add_argument("--batch-size", type=int)
    build.add_argument("--workers", type=int)
    build.add_argument(
        "--work-dir",
        type=Path,
        help="deterministic resumable work directory (default: .<output>.work)",
    )
    build.add_argument("--checkpoint-rows", type=int, default=5000)

    natural = subparsers.add_parser("natural")
    natural.add_argument("--config", required=True, type=Path)
    natural.add_argument("--store", required=True, type=Path)
    natural.add_argument("--provenance", required=True, type=Path)
    natural.add_argument("--dataset-root", required=True, type=Path)
    natural.add_argument("--archive", required=True, type=Path)
    natural.add_argument("--split-seed", required=True, type=int)
    natural.add_argument("--cost-budget-relative", required=True, type=float)
    natural.add_argument("--output", type=Path)
    natural.add_argument("--plan-only", action="store_true")

    positive = subparsers.add_parser("positive-control")
    positive.add_argument("--config", required=True, type=Path)
    positive.add_argument("--store", required=True, type=Path)
    positive.add_argument("--split-seed", required=True, type=int)
    positive.add_argument("--cost-budget-relative", required=True, type=float)
    positive.add_argument("--output", type=Path)
    positive.add_argument("--plan-only", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", required=True, type=Path)
    aggregate.add_argument("--natural-units-root", required=True, type=Path)
    aggregate.add_argument("--positive-units-root", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "build-store":
        output = build_imagenet_sketch_feature_store(
            play3_config_path=arguments.config,
            provenance_manifest=arguments.provenance,
            reference_store_path=arguments.reference_store,
            destination=arguments.output_store,
            encoder_id=arguments.encoder,
            dataset_root=arguments.dataset_root,
            archive=arguments.archive,
            device=arguments.device,
            batch_size=arguments.batch_size,
            workers=arguments.workers,
            work_dir=arguments.work_dir,
            checkpoint_rows=arguments.checkpoint_rows,
        )
        print(json.dumps({
            "status": "PASS",
            "encoder": arguments.encoder,
            "store_manifest": str(output),
            "store_manifest_sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    if arguments.command == "positive-control":
        return _positive_control(arguments)
    if arguments.command == "aggregate":
        output = aggregate_play3_panel(
            config_path=arguments.config,
            natural_units_root=arguments.natural_units_root,
            positive_units_root=arguments.positive_units_root,
            output=arguments.output,
        )
        print(json.dumps({
            "status": "PASS",
            "aggregate": str(output),
            "aggregate_sha256": sha256_file(output),
        }, sort_keys=True))
        return 0
    delegated = [
        "--config", str(arguments.config),
        "--store", str(arguments.store),
        "--provenance", str(arguments.provenance),
        "--dataset-root", str(arguments.dataset_root),
        "--archive", str(arguments.archive),
        "--split-seed", str(arguments.split_seed),
        "--cost-budget-relative", str(arguments.cost_budget_relative),
    ]
    if arguments.output is not None:
        delegated.extend(["--output", str(arguments.output)])
    if arguments.plan_only:
        delegated.append("--plan-only")
    return natural_unit_main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
