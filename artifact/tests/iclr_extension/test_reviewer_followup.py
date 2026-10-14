from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    read_json,
    seal_manifest,
    sha256_file,
    write_json_atomic,
)
from marginal_alignment.iclr_extension.realized_profile import (
    REALIZED_PROFILE_METHODS,
)


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "configs/iclr_extension/reviewer_followup_20260910.json"
RUNNER = REPOSITORY / "scripts/iclr_extension/run_reviewer_followup.py"


def _load_runner():
    module_spec = importlib.util.spec_from_file_location(
        "tested_reviewer_followup", RUNNER
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _config() -> dict:
    return copy.deepcopy(runner._load_config(CONFIG))


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _result_summary(value: float = 0.25) -> dict:
    return {
        "maximum_disagreement_primal_lower_bound": value,
        "maximum_disagreement_dual_upper_bound": value + 0.01,
        "selected_two_world_minimax_recovery": 0.5,
        "selected_world_disagreement": value,
    }


def _profile_scientific(spec) -> dict:
    return {
        "group_id": spec.group_id,
        "panel": spec.panel,
        "dataset": spec.dataset,
        "corruption": spec.corruption,
        "encoder": spec.encoder,
        "split_seed": spec.seed,
        "certificates": [
            {
                "budget_key": "shared_random_gap__0500bp",
                "budget_family": "shared_random_gap",
                "declared_budget_value": 0.05,
                "result": _result_summary(),
            }
        ],
        # Alphabetical insertion order matches canonical JSON, not the public
        # method tuple. Validation must therefore be order independent.
        "accuracy": {
            "per_method_target_accuracy": {
                "bures_oas": 0.4,
                "diagonal": 0.3,
                "mean": 0.2,
                "none": 0.1,
            },
            "bures_minus_none_target_accuracy": 0.3,
            "target_identity_source": {"kind": "fixture"},
        },
        "profile_gain_association_record": {
            "profile_binding_sha256": "a" * 64,
        },
    }


def _minimal_group(tmp_path: Path):
    spec = runner.GroupSpec(
        runner.CONTROL_UNPAIRED,
        "imagenet_c",
        "contrast",
        "clip",
        201,
        (runner.BudgetSpec("shared_random_gap", 0.05),),
        True,
    )
    lock_path = tmp_path / "REVIEWER_FOLLOWUP_LOCK.json"
    lock_path.write_text("locked fixture\n", encoding="utf-8")
    group_path = tmp_path / "groups" / spec.group_id
    group_path.mkdir(parents=True)
    arrays = {
        name: runner._save_array(
            group_path, name, np.asarray([0, 1, 2], dtype=np.int64)
        )
        for name in sorted(runner._expected_array_names(spec))
    }
    scientific = _profile_scientific(spec)
    scientific.update(
        {
            "pairing_available_to_map_or_certificate_optimizer": False,
            "labels_available_to_map_or_certificate_optimizer": False,
            "target_label_values_interpreted_only_after_predictions_and_certificates_frozen": True,
            "pair_structure_postfit": None,
        }
    )
    manifest = seal_manifest(
        {
            "schema_version": runner.GROUP_SCHEMA,
            "status": "PASS",
            "scientific_result": True,
            "lock_file_sha256": sha256_file(lock_path),
            "scientific": scientific,
            "arrays": arrays,
            "compute": {"fixture": True},
        }
    )
    write_json_atomic(group_path / "GROUP_MANIFEST.json", manifest)
    return lock_path, spec, group_path


def _reseal(path: Path, mutate) -> None:
    payload = dict(read_json(path))
    payload.pop("manifest_sha256")
    mutate(payload)
    write_json_atomic(path, seal_manifest(payload))


def test_config_enumerates_exact_locked_inventory_and_budget_families() -> None:
    groups = runner.enumerate_groups(_config())
    assert len(groups) == 144
    assert len({group.group_id for group in groups}) == 144
    assert sum(len(group.budgets) for group in groups) == 720
    assert sum(group.score_accuracy for group in groups) == 108

    panel_groups = Counter(group.panel for group in groups)
    panel_certificates = Counter()
    family_certificates = Counter()
    for group in groups:
        panel_certificates[group.panel] += len(group.budgets)
        family_certificates.update(budget.family for budget in group.budgets)
    assert panel_groups == {
        runner.NATURAL_PANEL: 18,
        runner.CONTROL_UNPAIRED: 90,
        runner.CONTROL_PAIRED: 36,
    }
    assert panel_certificates == {
        runner.NATURAL_PANEL: 90,
        runner.CONTROL_UNPAIRED: 342,
        runner.CONTROL_PAIRED: 288,
    }
    assert family_certificates == {
        "shared_random_gap": 504,
        "direct_nominal": 216,
    }
    assert _config()["expected_counts"]["sketch_paired_bound_separation_rows"] == 180


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["common_protocol"].__setitem__(
                "maximum_primal_dual_gap", 0.06
            ),
            "critical follow-up protocol",
        ),
        (
            lambda value: value["panels"][runner.CONTROL_UNPAIRED].__setitem__(
                "expected_focus_certificate_rows", 287
            ),
            "disjoint-unpaired",
        ),
        (
            lambda value: value["panels"][runner.CONTROL_UNPAIRED][
                "corruptions"
            ].__setitem__(1, "replacement_corruption"),
            "disjoint-unpaired",
        ),
        (
            lambda value: value["reporting_profiles"][
                "paired_structure_contrast"
            ].__setitem__("primary_budget_value", 0.1),
            "reporting profile",
        ),
    ],
)
def test_config_rejects_scientific_protocol_or_schema_drift(
    tmp_path: Path, mutate, match: str
) -> None:
    value = _config()
    mutate(value)
    with pytest.raises(IntegrityError, match=match):
        runner._load_config(_write_config(tmp_path, value))


def test_shared_random_gap_and_direct_budget_conversion_are_distinct() -> None:
    shared = runner._budget_relative(
        runner.BudgetSpec("shared_random_gap", 0.05),
        baseline_cost=100.0,
        random_mean_cost=300.0,
    )
    direct = runner._budget_relative(
        runner.BudgetSpec("direct_nominal", 0.05),
        baseline_cost=100.0,
        random_mean_cost=300.0,
    )
    assert shared == pytest.approx(0.1)
    assert 100.0 * (1.0 + shared) == pytest.approx(110.0)
    assert direct == pytest.approx(0.05)
    assert 100.0 * (1.0 + direct) == pytest.approx(105.0)
    with pytest.raises(IntegrityError, match="unknown budget family"):
        runner._budget_relative(
            runner.BudgetSpec("unknown", 0.05),
            baseline_cost=100.0,
            random_mean_cost=300.0,
        )


def test_store_aware_sharding_is_complete_deterministic_and_never_splits_store() -> (
    None
):
    groups = runner.enumerate_groups(_config())
    first = runner.shard_groups(groups, 8)
    second = runner.shard_groups(list(reversed(groups)), 8)
    assert [[value.group_id for value in shard] for shard in first] == [
        [value.group_id for value in shard] for shard in second
    ]
    assert Counter(value.group_id for shard in first for value in shard) == Counter(
        value.group_id for value in groups
    )
    store_shards: dict[str, set[int]] = defaultdict(set)
    for shard_index, shard in enumerate(first):
        for value in shard:
            store_shards[
                runner._store_key(value.dataset, value.encoder, value.corruption)
            ].add(shard_index)
    assert all(len(indices) == 1 for indices in store_shards.values())


def test_profile_rows_accept_canonical_method_key_order_and_close_at_108() -> None:
    groups = [
        _profile_scientific(spec)
        for spec in runner.enumerate_groups(_config())
        if spec.score_accuracy
    ]
    rows = runner._profile_rows(groups)
    assert len(rows) == 108
    assert {row["panel"] for row in rows} == {
        runner.NATURAL_PANEL,
        runner.CONTROL_UNPAIRED,
    }
    assert rows[0]["accuracy_none"] == pytest.approx(0.1)
    assert rows[0]["accuracy_bures_oas"] == pytest.approx(0.4)

    groups[0]["accuracy"]["per_method_target_accuracy"].pop("mean")
    with pytest.raises(IntegrityError, match="method inventory"):
        runner._profile_rows(groups)


def _pair_contrasts() -> list[dict]:
    corruptions = _config()["panels"][runner.CONTROL_PAIRED]["corruptions"]
    return [
        {
            "corruption": corruption,
            "encoder": encoder,
            "split_seed": seed,
            "budget_family": "shared_random_gap",
            "declared_budget_value": 0.05,
            "selected_feasible_paired_minus_disjoint_unpaired": 0.25,
        }
        for corruption in corruptions
        for encoder in ("clip", "dinov3")
        for seed in (201, 202, 203)
    ]


def test_primary_pair_inference_uses_paired_minus_unpaired_and_exact_sign_flip() -> (
    None
):
    corruptions = _config()["panels"][runner.CONTROL_PAIRED]["corruptions"]
    result = runner._primary_pair_inference(
        _pair_contrasts(), expected_corruptions=corruptions, rng_seed=7
    )
    assert result["estimand"] == "selected_feasible_paired_minus_disjoint_unpaired"
    assert result["cluster_count"] == 6
    assert result["cluster_mean"] == pytest.approx(0.25)
    assert result["exhaustive_two_sided_sign_flip"] == {
        "assignments": 64,
        "p_value": pytest.approx(0.03125),
        "minimum_attainable_two_sided_p_value": 0.03125,
    }
    for cluster in result["cluster_rows"]:
        assert cluster["nested_encoder_seed_cells"] == 6
        assert len(cluster["encoder_rows"]) == 2
        assert all(row["split_seed_repeats"] == 3 for row in cluster["encoder_rows"])


def test_primary_pair_inference_rejects_duplicate_or_missing_nested_cells() -> None:
    rows = _pair_contrasts()
    corruptions = _config()["panels"][runner.CONTROL_PAIRED]["corruptions"]
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(IntegrityError, match="duplicate"):
        runner._primary_pair_inference(duplicate, expected_corruptions=corruptions)
    with pytest.raises(IntegrityError, match="36 cells"):
        runner._primary_pair_inference(rows[:-1], expected_corruptions=corruptions)


def test_group_validation_binds_exact_relative_array_paths_and_inventory(
    tmp_path: Path,
) -> None:
    lock_path, spec, group_path = _minimal_group(tmp_path)
    runner._validate_group(lock_path, spec, group_path)

    extra = group_path / "undeclared.npy"
    np.save(extra, np.zeros(1), allow_pickle=False)
    with pytest.raises(IntegrityError, match="file inventory changed"):
        runner._validate_group(lock_path, spec, group_path)
    extra.unlink()

    manifest_path = group_path / "GROUP_MANIFEST.json"
    names = sorted(runner._expected_array_names(spec))
    _reseal(
        manifest_path,
        lambda value: value["arrays"][names[0]].__setitem__("path", f"{names[1]}.npy"),
    )
    with pytest.raises(IntegrityError, match="array path changed"):
        runner._validate_group(lock_path, spec, group_path)


def test_group_validation_rejects_resealed_label_contract_tampering(
    tmp_path: Path,
) -> None:
    lock_path, spec, group_path = _minimal_group(tmp_path)
    manifest_path = group_path / "GROUP_MANIFEST.json"
    _reseal(
        manifest_path,
        lambda value: value["scientific"].__setitem__(
            "labels_available_to_map_or_certificate_optimizer", True
        ),
    )
    with pytest.raises(IntegrityError, match="invalid follow-up group"):
        runner._validate_group(lock_path, spec, group_path)


def test_group_resume_fails_closed_on_any_stale_partial_temporary(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "REVIEWER_FOLLOWUP_LOCK.json"
    lock_path.write_text("locked fixture\n", encoding="utf-8")
    spec = runner.GroupSpec(
        runner.NATURAL_PANEL,
        "imagenet_r",
        None,
        "clip",
        201,
        (runner.BudgetSpec("shared_random_gap", 0.01),),
        True,
    )
    stale = tmp_path / "groups" / f".{spec.group_id}.tmp.999999"
    stale.mkdir(parents=True)
    with pytest.raises(IntegrityError, match="partial temporary group"):
        runner._write_group(lock_path, spec)


def test_compute_group_withholds_pair_labels_until_postfreeze_scoring() -> None:
    source = inspect.getsource(runner._compute_group)
    assert "validate_pair_labels=False" in source
    assert "target.labels" not in source
    assert source.index(
        "construct_optimization_certificate_from_problem"
    ) < source.index("freeze_realized_profile_predictions")
    assert source.index("freeze_realized_profile_predictions") < source.index(
        "_posthoc_endpoint_wnids"
    )
    assert source.index("_posthoc_endpoint_wnids") < source.index(
        "score_frozen_realized_profile"
    )


def test_aggregate_rejects_group_changed_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "REVIEWER_FOLLOWUP_LOCK.json"
    lock_path.write_text("lock fixture\n", encoding="utf-8")
    spec = runner.GroupSpec(
        runner.NATURAL_PANEL,
        "imagenet_r",
        None,
        "clip",
        201,
        (runner.BudgetSpec("shared_random_gap", 0.01),),
        True,
    )
    group_path = runner._group_path(tmp_path, spec)
    group_path.mkdir(parents=True)
    manifest_path = group_path / "GROUP_MANIFEST.json"
    manifest_path.write_text("current group bytes\n", encoding="utf-8")
    current_manifest = {"manifest_sha256": "c" * 64, "scientific": {}}
    report = seal_manifest(
        {
            "schema_version": runner.VERIFICATION_SCHEMA,
            "status": "PASS",
            "lock_file_sha256": sha256_file(lock_path),
            "shard_index": 0,
            "shard_count": 1,
            "groups": [
                {
                    "group_id": spec.group_id,
                    "manifest_file_sha256": "a" * 64,
                    "manifest_sha256": "c" * 64,
                }
            ],
        }
    )
    write_json_atomic(tmp_path / "verification" / "shard_000.json", report)
    monkeypatch.setattr(runner, "_load_lock", lambda _: {"config": {}})
    monkeypatch.setattr(runner, "enumerate_groups", lambda _: [spec])
    monkeypatch.setattr(runner, "_validate_group", lambda *_: current_manifest)
    with pytest.raises(IntegrityError, match="changed after full reconstruction"):
        runner.aggregate(lock_path, verification_shards=1)


def test_aggregate_completion_publishes_truthful_root_state() -> None:
    source = inspect.getsource(runner.aggregate)
    assert 'source.parent / "STATE.json"' in source
    assert '"status": "COMPLETE"' in source
    assert '"completed_groups": len(group_records)' in source
    assert '"done_file_sha256": sha256_file(done_path)' in source


def test_realized_profile_method_contract_remains_exact() -> None:
    assert set(REALIZED_PROFILE_METHODS) == {
        "none",
        "mean",
        "diagonal",
        "bures_oas",
    }
