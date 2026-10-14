#!/usr/bin/env python3
"""Build strict legacy feature arrays from raw images and exact local weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marginal_alignment.campaign.raw_extract import build_raw_pair


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", default="cifar100_c,imagenet_c")
    parser.add_argument("--encoders", default="dinov3,clip,vitb16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args(argv)
    results = []
    for encoder in _csv(arguments.encoders):
        for dataset in _csv(arguments.datasets):
            print(f"[raw] {dataset}/{encoder}", flush=True)
            results.append(build_raw_pair(
                config_path=arguments.config, output_root=arguments.output_root,
                dataset=dataset, encoder_id=encoder, device=arguments.device,
                batch_size=arguments.batch_size, workers=arguments.workers,
            ))
    report = {
        "schema_version": "marginal_alignment.campaign.raw_feature_campaign.v1",
        "status": "PASS", "pairs": len(results), "results": results,
    }
    output = Path(arguments.output_root) / "RAW_EXTRACTION_MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "pairs": len(results), "manifest": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

