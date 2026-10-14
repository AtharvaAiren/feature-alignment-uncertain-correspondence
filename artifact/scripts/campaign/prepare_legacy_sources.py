#!/usr/bin/env python3
"""Audit or prepare strict legacy sources without mutating shared caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.feature_protocol import load_feature_build_config
from marginal_alignment.campaign.legacy_prepare import (
    audit_all_reuse,
    prepare_reused_pair,
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="read-only qualification of existing caches")
    prepare = subparsers.add_parser(
        "prepare", help="hard-link qualified arrays and fit exact clean heads"
    )
    for child in (audit, prepare):
        child.add_argument("--config", required=True)
        child.add_argument("--source-root", required=True)
        child.add_argument("--datasets", default="cifar100_c,imagenet_c")
        child.add_argument("--encoders", default="dinov3,clip,vitb16")
    audit.add_argument("--output", required=True)
    prepare.add_argument("--destination-root", required=True)
    prepare.add_argument("--seal", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    datasets = _csv(arguments.datasets)
    encoders = _csv(arguments.encoders)
    if arguments.command == "audit":
        report = audit_all_reuse(
            source_root=arguments.source_root, config_path=arguments.config,
            datasets=datasets, encoders=encoders,
        )
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "READY" else 3

    config = load_feature_build_config(arguments.config)
    results = []
    for dataset in datasets:
        for encoder in encoders:
            print(f"[prepare] {dataset}/{encoder}", flush=True)
            results.append(prepare_reused_pair(
                source_root=arguments.source_root,
                destination_root=arguments.destination_root,
                dataset=dataset, encoder=encoder, config=config,
                seal=bool(arguments.seal),
            ))
    report = {
        "schema_version": "marginal_alignment.campaign.prepared_legacy_campaign.v1",
        "status": "PASS", "pairs": len(results), "results": results,
    }
    output = Path(arguments.destination_root) / "PREPARATION_MANIFEST.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "pairs": len(results), "manifest": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

