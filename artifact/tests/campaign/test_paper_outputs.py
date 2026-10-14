from __future__ import annotations

import json
from pathlib import Path

from marginal_alignment.campaign.artifacts import read_json, sha256_file, write_json_atomic
from marginal_alignment.campaign.inference import macro_summary
from marginal_alignment.campaign.paper_outputs import generate_paper_outputs
from marginal_alignment.campaign.real_runner import METRICS


def _metrics(value: float) -> dict[str, float]:
    return {metric: float(value) for metric in METRICS}


def _real_row(
    *, suite: str, method: str, calibration_size: int, split_seed: int,
    severity: int = 1, value: float = 0.5, **extra: object,
) -> dict[str, object]:
    return {
        "suite": suite,
        "dataset": "toy_c",
        "encoder": "toy_encoder",
        "corruption": "fog",
        "severity": severity,
        "split_seed": split_seed,
        "calibration_size": calibration_size,
        "method": method,
        "pair_budget": None,
        "pair_noise_rate": None,
        "pair_noise_rate_requested": None,
        "pair_noise_rate_actual": None,
        "pair_control": None,
        "diagnostics": {},
        **_metrics(value),
        **extra,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_real_fixture(root: Path) -> None:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for severity in (1, 3):
            rows.extend([
                _real_row(
                    suite="primary_marginal", method="none", calibration_size=4096,
                    split_seed=seed, severity=severity, value=0.40 + 0.01 * seed,
                ),
                _real_row(
                    suite="primary_marginal", method="diagonal", calibration_size=4096,
                    split_seed=seed, severity=severity, value=0.60 + 0.01 * seed,
                ),
            ])
    for seed in (40, 41):
        for severity in (1, 3, 5):
            rows.append(_real_row(
                suite="severity_breadth", method="diagonal", calibration_size=4096,
                split_seed=seed, severity=severity, value=0.50 + 0.05 * severity,
            ))
    # A method that exists only at another n must never leak into the locked main table.
    rows.append(_real_row(
        suite="primary_marginal", method="legacy_only", calibration_size=1024,
        split_seed=0, value=0.99,
    ))
    for seed in (30, 31):
        common = {
            "suite": "correspondence_dose", "calibration_size": 4096,
            "split_seed": seed, "severity": 5,
        }
        rows.extend([
            _real_row(
                method="paired_ridge", value=0.8, **common,
                pair_budget=16, pair_noise_rate=0.0,
                pair_noise_rate_requested=0.0, pair_noise_rate_actual=0.0,
                pair_control="true",
            ),
            _real_row(
                method="paired_ridge_shuffled", value=0.3, **common,
                pair_budget=16, pair_noise_rate=1.0,
                pair_noise_rate_requested=1.0, pair_noise_rate_actual=1.0,
                pair_control="shuffled",
            ),
            _real_row(
                method="paired_ridge_noise0.1", value=0.5, **common,
                pair_budget=16, pair_noise_rate=0.1,
                pair_noise_rate_requested=0.1, pair_noise_rate_actual=0.125,
                pair_control="noisy",
            ),
            _real_row(
                method="labeled_er", value=0.7, **common,
                changes_frozen_consumer=True,
                diagnostics={
                    "changes_frozen_consumer": True,
                    "target_buffer_rows": 512,
                    "clean_reference_replay_rows": 256,
                },
            ),
        ])
    rows_path = root / "real/rows.jsonl"
    summary_path = root / "real/summary.json"
    manifest_path = root / "real/manifest.json"
    _write_jsonl(rows_path, rows)
    write_json_atomic(summary_path, {
        "schema_version": "fixture.real_summary",
        "rows": len(rows),
        "groups": macro_summary(rows, METRICS),
    })
    write_json_atomic(manifest_path, {
        "schema_version": "fixture.real_manifest",
        "status": "PASS",
        "record_count": len(rows),
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    })


def _assert_output_manifest_complete(root: Path) -> None:
    output = root / "paper_outputs"
    manifest = read_json(output / "manifest.json")
    declared = {record["path"]: record for record in manifest["outputs"]}
    actual = {path.name for path in output.iterdir() if path.name != "manifest.json"}
    assert set(declared) == actual
    for relative, record in declared.items():
        path = output / relative
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_file(path)


def test_outputs_lock_primary_n_preserve_actual_noise_and_mark_missing_optional(tmp_path):
    _write_real_fixture(tmp_path)
    generate_paper_outputs(tmp_path)

    main = read_json(tmp_path / "paper_outputs/main_marginal_ladder.json")
    assert main["status"] == "PASS"
    assert main["methods_present"] == ["none", "diagonal"]
    assert {row["calibration_size"] for row in main["rows"]} == {4096}
    assert all("aligned_covariance_relative_error_mean" in row for row in main["rows"])
    assert all("reference_prediction_agreement_mean" in row for row in main["rows"])

    correspondence = read_json(tmp_path / "paper_outputs/correspondence_dose.json")
    assert {row["pair_control"] for row in correspondence["rows"]} == {
        "true", "shuffled", "noisy",
    }
    noisy = next(row for row in correspondence["rows"] if row["pair_control"] == "noisy")
    assert noisy["pair_noise_rate_requested"] == 0.1
    assert noisy["pair_noise_rate_actual"] == 0.125

    er = read_json(tmp_path / "paper_outputs/labeled_er.json")
    assert er["rows"][0]["target_labeled_rows"] == 512
    assert er["rows"][0]["clean_replay_rows"] == 256
    assert "aligned_covariance_relative_error_mean" in er["rows"][0]

    breadth = read_json(tmp_path / "paper_outputs/encoder_severity_breadth.json")
    assert breadth["status"] == "PASS"
    assert {row["severity"] for row in breadth["rows"]} == {1, 3, 5}

    for name in (
        "finite_sample", "rank_shrinkage", "higher_order", "primary_inference",
        "strict_prequential_streaming",
        "synthetic_exact", "synthetic_approximate", "synthetic_nongaussian",
        "controlled_pair_phase", "controlled_real_orbit",
    ):
        assert read_json(tmp_path / f"paper_outputs/{name}.json")["status"] == "UNAVAILABLE"
        assert (tmp_path / f"paper_outputs/{name}.md").is_file()
        assert not (tmp_path / f"paper_outputs/{name}.png").exists()
        assert not (tmp_path / f"paper_outputs/{name}.pdf").exists()
    _assert_output_manifest_complete(tmp_path)


def test_optional_theory_and_controlled_panels_use_persisted_hashed_evidence(tmp_path):
    _write_real_fixture(tmp_path)
    synthetic = tmp_path / "synthetic"
    exact_rows = [
        {
            "family": "binary_gaussian_rotation", "seed": seed,
            "angle_degrees": 0,
            "theoretical_equal_prior_bayes_accuracy": 1.0,
            "empirical_equal_prior_bayes_accuracy": 1.0,
            "uniform_inverse_world_gap": 0.0, "absolute_error": 0.0,
        }
        for seed in (0, 1)
    ]
    approximate_rows = [{"seed": 0, "family": "approximate_two_world"}]
    nongaussian_rows = [{"seed": 0, "family": "identifiable_nongaussian_swap"}]
    _write_jsonl(synthetic / "exact_indistinguishable_rows.jsonl", exact_rows)
    _write_jsonl(synthetic / "approximate_two_world_rows.jsonl", approximate_rows)
    _write_jsonl(synthetic / "identifiable_nongaussian_rows.jsonl", nongaussian_rows)
    write_json_atomic(synthetic / "summary.json", {
        "status": "PASS",
        "approximate_two_world": {"groups": [
            {
                "calibration_samples": n, "epsilon": 0.1, "task_disagreement": 0.5,
                "joint_observation_total_variation_mean": 0.2 + n / 100.0,
                "joint_observation_total_variation_std": 0.0,
                "exact_equal_prior_bayes_accuracy_mean": 0.6 + n / 100.0,
                "exact_equal_prior_bayes_accuracy_std": 0.0,
                "empirical_bayes_accuracy_mean": 0.6 + n / 100.0,
                "empirical_bayes_accuracy_std": 0.0,
            }
            for n in (0, 4)
        ]},
        "identifiable_nongaussian": {"groups": [
            {
                "calibration_samples": n, "asymmetry": 1.0,
                "method": "third_moment_permutation",
                "orientation_correct_mean": 0.7 + n / 1000.0,
                "orientation_correct_std": 0.01,
                "task_agreement_mean": 0.75, "task_agreement_std": 0.01,
                "heldout_covariance_relative_error_mean": 0.0,
                "heldout_covariance_relative_error_std": 0.0,
            }
            for n in (16, 64)
        ]},
    })
    source_names = (
        "exact_indistinguishable_rows.jsonl", "approximate_two_world_rows.jsonl",
        "identifiable_nongaussian_rows.jsonl", "summary.json",
    )
    write_json_atomic(synthetic / "manifest.json", {
        "status": "PASS",
        "outputs": [
            {"path": f"/server/run/synthetic/{name}", "sha256": sha256_file(synthetic / name)}
            for name in source_names
        ],
    })

    controlled = tmp_path / "controlled"
    pair_rows = [
        {
            "seed": seed, "dimension": 8, "classes": 2,
            "method": "paired_procrustes", "pair_budget": 9,
            "pairs_per_dimension": 1.125, "pair_noise_rate": 0.0,
            "prediction_agreement": 0.98, "paired_feature_mse": 0.001,
            "centered_logit_mse": 0.001,
            "identified_centered_readout_energy": 1.0,
        }
        for seed in (0, 1)
    ]
    _write_jsonl(controlled / "pair_rank_phase.jsonl", pair_rows)
    orbit_record = {
        "dataset": "toy_c", "encoder": "toy_encoder", "worlds": 4, "rank": 4,
        "unrestricted_equal_prior_bayes_prediction_upper": 0.8,
        "achievable_group_map_randomized_minimax": 0.6,
        "endpoint_mean_residual": 0.0,
        "endpoint_covariance_relative_residual": 0.0,
    }
    write_json_atomic(controlled / "real_embedding_orbits.json", {
        "status": "PASS", "records": [orbit_record],
    })
    write_json_atomic(controlled / "manifest.json", {
        "status": "PASS", "pair_rows": len(pair_rows),
        "pair_rows_sha256": sha256_file(controlled / "pair_rank_phase.jsonl"),
        "orbit_records": 1,
        "orbit_sha256": sha256_file(controlled / "real_embedding_orbits.json"),
    })

    generate_paper_outputs(tmp_path)
    for name in (
        "synthetic_exact", "synthetic_approximate", "synthetic_nongaussian",
        "controlled_pair_phase", "controlled_real_orbit",
    ):
        assert read_json(tmp_path / f"paper_outputs/{name}.json")["status"] == "PASS"
        assert (tmp_path / f"paper_outputs/{name}.png").is_file()
        assert (tmp_path / f"paper_outputs/{name}.pdf").is_file()
    exact = read_json(tmp_path / "paper_outputs/synthetic_exact.json")
    pair = read_json(tmp_path / "paper_outputs/controlled_pair_phase.json")
    assert exact["rows"][0]["repeats"] == 2
    assert exact["rows"][0]["repeat_key"] == "seed"
    assert pair["rows"][0]["repeats"] == 2
    assert pair["rows"][0]["repeat_key"] == "seed"
    _assert_output_manifest_complete(tmp_path)
