#!/usr/bin/env python3
"""Emit the exact finite quota-capacity control used in the theory audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from marginal_alignment.campaign.artifacts import seal_manifest, write_json_atomic
from marginal_alignment.quota_capacity import three_world_quota_control


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_control() -> dict[str, Any]:
    result = {key: _json_value(value) for key, value in three_world_quota_control().items()}
    expected = {
        "unrestricted_prior_average_value": 3.0 / 4.0,
        "quota_prior_average_value": 2.0 / 3.0,
        "minimax_value": 2.0 / 3.0,
    }
    checks = {
        "unrestricted_value_is_three_quarters": bool(
            np.isclose(
                result["unrestricted_prior_average_value"],
                expected["unrestricted_prior_average_value"],
            )
        ),
        "quota_value_is_two_thirds": bool(
            np.isclose(result["quota_prior_average_value"], expected["quota_prior_average_value"])
        ),
        "minimax_primal_is_two_thirds": bool(
            np.isclose(result["primal_value"], expected["minimax_value"])
        ),
        "minimax_dual_is_two_thirds": bool(
            np.isclose(result["dual_value"], expected["minimax_value"])
        ),
        "primal_dual_gap_at_most_1e_10": bool(
            result["absolute_primal_dual_gap"] <= 1e-10
        ),
        "strict_output_mass_penalty": bool(result["quota_penalty"] > 0.0),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return seal_manifest(
        {
            "schema_version": "marginal_alignment.quota_capacity_control.v1",
            "status": status,
            "scope": "finite_indexed_atom_analogue_not_population_transport_evidence",
            "worlds": 3,
            "atoms": 4,
            "reference_output_label_counts": [2, 2],
            "exact_values": {
                "unrestricted_prior_average_value": "3/4",
                "quota_prior_average_value": "2/3",
                "randomized_quota_minimax_value": "2/3",
                "strict_quota_penalty": "1/12",
            },
            "checks": checks,
            "computed": result,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    control = build_control()
    if args.output is not None:
        write_json_atomic(args.output, control)
    print(json.dumps(control, indent=2, sort_keys=True))
    return 0 if control["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
