#!/usr/bin/env python3
"""Run the reviewer-requested correspondence follow-up without altering R5.

The follow-up has three deliberately separated roles:

* the completed Play-3 v2 R5 stores supply the three natural-shift panels;
* verified ImageNet-C stores supply paired and identity-disjoint controls; and
* label/WNID values are not interpreted as scientific outcomes until the
  post-freeze accuracy evaluation.

Every expensive group shares one OAS/Bures fit and one dense cost matrix across
its locked budget grid.  A second pass reconstructs every scientific payload
and array before aggregation.  The script is CPU-only by protocol; launchers
must set ``CUDA_VISIBLE_DEVICES`` to the empty string.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_alignment.campaign.artifacts import (
    IntegrityError,
    describe_npy,
    read_json,
    seal_manifest,
    sha256_file,
    sha256_json,
    verify_file_record,
    verify_manifest_checksum,
    write_json_atomic,
)
from marginal_alignment.campaign.feature_store import load_feature_store
from marginal_alignment.iclr_extension.frontier import fit_marginal_frontier
from marginal_alignment.iclr_extension.cost_scale import summarize_cost_scale
from marginal_alignment.iclr_extension.natural_shift_store import (
    load_official_class_index,
)
from marginal_alignment.iclr_extension.optimization_certificate import (
    OptimizationCertificateProtocol,
    _array_sha256,
    _unique_canonical_rows,
    construct_optimization_certificate_from_problem,
    fit_certificate_geometry,
    prepare_certificate_problem,
    run_exact_subset_validation,
)
from marginal_alignment.iclr_extension.pair_structure import (
    PairStructureProtocol,
    build_pair_structure_selection,
    build_pair_structure_split,
    random_cost_moments_from_cost_matrix,
    relative_budget_for_random_gap_share,
    validate_pair_structure_manifest,
)
from marginal_alignment.iclr_extension.realized_profile import (
    REALIZED_PROFILE_METHODS,
    cluster_aware_profile_gain_association,
    freeze_realized_profile_predictions,
    profile_gain_record,
    score_frozen_realized_profile,
)


CONFIG_SCHEMA = "marginal_alignment.iclr_extension.reviewer_followup.v1"
LOCK_SCHEMA = "marginal_alignment.reviewer_followup_lock.v1"
GROUP_SCHEMA = "marginal_alignment.reviewer_followup_group.v1"
VERIFICATION_SCHEMA = "marginal_alignment.reviewer_followup_verification.v1"
AGGREGATE_SCHEMA = "marginal_alignment.reviewer_followup_aggregate.v1"
STATE_SCHEMA = "marginal_alignment.reviewer_followup_state.v1"

NATURAL_PANEL = "natural_shared_profile"
CONTROL_PAIRED = "imagenet_c_paired_structure_control"
CONTROL_UNPAIRED = "imagenet_c_disjoint_unpaired_profile"
PANELS = (NATURAL_PANEL, CONTROL_PAIRED, CONTROL_UNPAIRED)
PRIMARY_GEOMETRY = "full_reference_oas_mahalanobis_after_bures"
IMAGENET_C_CORRUPTIONS = (
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
)
FOCUS_CORRUPTIONS = (
    "brightness",
    "defocus_blur",
    "fog",
    "gaussian_noise",
    "jpeg_compression",
    "motion_blur",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SOURCE_FILES = (
    "configs/iclr_extension/reviewer_followup_20260910.json",
    "docs/reviewer_followup_20260910.md",
    "scripts/iclr_extension/run_reviewer_followup.py",
    "src/marginal_alignment/__init__.py",
    "src/marginal_alignment/ceiling.py",
    "src/marginal_alignment/campaign/__init__.py",
    "src/marginal_alignment/iclr_extension/pair_structure.py",
    "src/marginal_alignment/iclr_extension/realized_profile.py",
    "src/marginal_alignment/iclr_extension/cost_scale.py",
    "src/marginal_alignment/iclr_extension/frontier.py",
    "src/marginal_alignment/iclr_extension/natural_shift_store.py",
    "src/marginal_alignment/iclr_extension/optimization_certificate.py",
    "src/marginal_alignment/iclr_extension/__init__.py",
    "src/marginal_alignment/iclr_extension/ceiling.py",
    "src/marginal_alignment/iclr_extension/natural_shift.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/cache_import.py",
    "src/marginal_alignment/campaign/feature_protocol.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/method_registry.py",
    "src/marginal_alignment/campaign/raw_extract.py",
    "src/marginal_alignment/campaign/split_plan.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/transforms.py",
    "tests/iclr_extension/test_pair_structure.py",
    "tests/iclr_extension/test_realized_profile.py",
    "tests/iclr_extension/test_reviewer_followup.py",
)


@dataclass(frozen=True)
class BudgetSpec:
    family: str
    value: float

    @property
    def key(self) -> str:
        basis_points = int(round(10_000.0 * self.value))
        return f"{self.family}__{basis_points:04d}bp"


@dataclass(frozen=True)
class GroupSpec:
    panel: str
    dataset: str
    corruption: str | None
    encoder: str
    seed: int
    budgets: tuple[BudgetSpec, ...]
    score_accuracy: bool

    @property
    def group_id(self) -> str:
        condition = self.dataset if self.corruption is None else self.corruption
        return f"{self.panel}__{condition}__{self.encoder}__seed_{self.seed}"


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "src/marginal_alignment").is_dir():
        raise IntegrityError("cannot resolve repository root")
    return root


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{name} must be a JSON object")
    return value


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _source_hashes() -> dict[str, str]:
    repository = _repository_root()
    values: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = repository / relative
        if not source.is_file() or source.is_symlink():
            raise IntegrityError(f"follow-up source is absent or symbolic: {relative}")
        values[relative] = sha256_file(source)
    return values


@lru_cache(maxsize=None)
def _load_verified_store(path_text: str):
    """Hash one immutable store once per worker, then reuse its memmaps."""

    return load_feature_store(Path(path_text), verify_hashes=True)


def _runtime_contract() -> dict[str, Any]:
    try:
        import scipy
        import scipy.optimize._highspy._core as highs_core
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError("the follow-up requires SciPy with bundled HiGHS") from error
    observed_threads = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}
    if observed_threads != THREAD_ENVIRONMENT:
        raise RuntimeError(
            "all locked numerical thread variables must equal '1': "
            f"observed={observed_threads}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("reviewer follow-up is locked to CPU-only execution")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "highs": (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        ),
        "thread_environment": observed_threads,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = dict(_mapping(read_json(source), name="follow-up config"))
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("status")
        != "PRESPECIFIED_POST_HOC_REVIEWER_MOTIVATED_BEFORE_NEW_OUTCOMES"
        or value.get("scientific_result") is not False
    ):
        raise IntegrityError("follow-up config is not eligible for freezing")
    timing = _mapping(value.get("analysis_timing"), name="analysis timing")
    if (
        timing.get("classification") != "post_hoc_reviewer_motivated"
        or timing.get("prior_outcomes_inspected") is not True
    ):
        raise IntegrityError("follow-up timing disclosure changed")
    protocol = _mapping(value.get("common_protocol"), name="common protocol")
    budget_arms = _mapping(value.get("budget_arms"), name="budget arms")
    shared = _mapping(budget_arms.get("shared_random_gap"), name="shared budget")
    direct = _mapping(budget_arms.get("direct_nominal"), name="direct budget")
    if (
        int(protocol.get("calibration_rows", -1)) != 3840
        or int(protocol.get("witness_rows", -1)) != 1024
        or int(protocol.get("exact_validation_rows", -1)) != 64
        or float(protocol.get("exact_milp_time_limit_seconds", -1.0)) != 900.0
        or float(protocol.get("exact_reported_mip_gap_tolerance", -1.0)) != 1e-12
        or str(protocol.get("cost_geometry")) != PRIMARY_GEOMETRY
        or str(protocol.get("covariance_estimator")) != "oas"
        or float(protocol.get("covariance_relative_floor", -1.0)) != 1e-6
        or int(protocol.get("greedy_neighbors", -1)) != 1024
        or int(protocol.get("dual_bisection_iterations", -1)) != 48
        or float(protocol.get("maximum_primal_dual_gap", -1.0)) != 0.05
        or float(protocol.get("assignment_tie_relative_tolerance", -1.0)) != 1e-12
        or list(map(int, protocol.get("split_seeds", ()))) != [201, 202, 203]
        or list(map(str, protocol.get("encoders", ()))) != ["clip", "dinov3"]
        or list(map(float, shared.get("values", ()))) != [0.01, 0.05, 0.1, 0.2, 0.4]
        or float(shared.get("primary_value", -1.0)) != 0.05
        or list(map(float, direct.get("values", ()))) != [0.01, 0.05, 0.1]
        or protocol.get(
            "external_ground_truth_labels_used_for_geometry_or_optimization"
        )
        is not False
        or protocol.get("frozen_readout_predictions_used_as_optimization_reward")
        is not True
    ):
        raise IntegrityError("critical follow-up protocol values changed")
    panels = _mapping(value.get("panels"), name="panels")
    controls = _mapping(panels.get(CONTROL_PAIRED), name="pair control")
    unpaired = _mapping(panels.get(CONTROL_UNPAIRED), name="unpaired control")
    natural = _mapping(panels.get(NATURAL_PANEL), name="natural panel")
    focus = list(map(str, unpaired.get("focus_corruptions", ())))
    paired_corruptions = list(map(str, controls.get("corruptions", ())))
    if (
        natural.get("endpoint_arm") != "natural_unpaired"
        or list(map(str, natural.get("datasets", ())))
        != ["imagenet_sketch", "imagenet_v2", "imagenet_r"]
        or natural.get("corruptions") is not None
        or natural.get("budget_schedule")
        != {"shared_random_gap": "all", "direct_nominal": "none"}
        or int(natural.get("expected_geometry_groups", -1)) != 18
        or int(natural.get("expected_certificate_rows", -1)) != 90
    ):
        raise IntegrityError("natural-shift panel changed")
    if (
        unpaired.get("endpoint_arm") != "disjoint_unpaired"
        or unpaired.get("dataset") != "imagenet_c"
        or int(unpaired.get("severity", -1)) != 5
        or tuple(map(str, unpaired.get("corruptions", ()))) != IMAGENET_C_CORRUPTIONS
        or tuple(focus) != FOCUS_CORRUPTIONS
        or focus != paired_corruptions
        or unpaired.get("nonfocus_budget_schedule")
        != {"shared_random_gap": [0.05], "direct_nominal": "none"}
        or unpaired.get("focus_budget_schedule")
        != {"shared_random_gap": "all", "direct_nominal": "all"}
        or int(unpaired.get("expected_focus_geometry_groups", -1)) != 36
        or int(unpaired.get("expected_nonfocus_geometry_groups", -1)) != 54
        or int(unpaired.get("expected_geometry_groups", -1)) != 90
        or int(unpaired.get("expected_focus_certificate_rows", -1)) != 288
        or int(unpaired.get("expected_nonfocus_certificate_rows", -1)) != 54
        or int(unpaired.get("expected_certificate_rows", -1)) != 342
    ):
        raise IntegrityError("disjoint-unpaired ImageNet-C panel changed")
    if (
        controls.get("endpoint_arm")
        != "paired_endpoint_sets_pairing_withheld_until_postfit_scoring"
        or controls.get("dataset") != "imagenet_c"
        or int(controls.get("severity", -1)) != 5
        or tuple(paired_corruptions) != FOCUS_CORRUPTIONS
        or controls.get("budget_schedule")
        != {"shared_random_gap": "all", "direct_nominal": "all"}
        or int(controls.get("expected_geometry_groups", -1)) != 36
        or int(controls.get("expected_certificate_rows", -1)) != 288
        or controls.get("pairing_used_for_geometry_or_assignment_optimization")
        is not False
    ):
        raise IntegrityError("paired ImageNet-C control changed")
    expected = _mapping(value.get("expected_counts"), name="expected counts")
    required_counts = {
        "natural_geometry_groups": 18,
        "imagenet_c_disjoint_unpaired_focus_geometry_groups": 36,
        "imagenet_c_disjoint_unpaired_nonfocus_geometry_groups": 54,
        "imagenet_c_disjoint_unpaired_geometry_groups": 90,
        "imagenet_c_paired_geometry_groups": 36,
        "total_geometry_groups": 144,
        "shared_random_gap_certificate_rows": 504,
        "direct_nominal_certificate_rows": 216,
        "natural_certificate_rows": 90,
        "imagenet_c_disjoint_unpaired_certificate_rows": 342,
        "imagenet_c_paired_certificate_rows": 288,
        "total_certificate_rows": 720,
        "primary_realized_profile_groups": 108,
        "primary_pair_contrast_geometry_pairs": 36,
        "primary_pair_contrast_corruption_clusters": 6,
        "sketch_paired_bound_separation_rows": 180,
    }
    if any(
        int(expected.get(key, -1)) != count for key, count in required_counts.items()
    ):
        raise IntegrityError("follow-up inventory changed")
    profiles = _mapping(value.get("reporting_profiles"), name="reporting profiles")
    realized = _mapping(
        profiles.get("primary_realized_profile"), name="primary realized profile"
    )
    pair_contrast = _mapping(
        profiles.get("paired_structure_contrast"), name="paired contrast"
    )
    if (
        realized.get("budget_arm") != "shared_random_gap"
        or float(realized.get("budget_value", -1.0)) != 0.05
        or list(map(str, realized.get("included_panels", ())))
        != [NATURAL_PANEL, CONTROL_UNPAIRED]
        or int(realized.get("expected_geometry_groups", -1)) != 108
        or pair_contrast.get("left_panel") != CONTROL_PAIRED
        or pair_contrast.get("right_panel") != CONTROL_UNPAIRED
        or pair_contrast.get("primary_budget_arm") != "shared_random_gap"
        or float(pair_contrast.get("primary_budget_value", -1.0)) != 0.05
        or list(map(str, pair_contrast.get("scope_corruptions", ()))) != focus
        or int(pair_contrast.get("expected_matched_geometry_pairs", -1)) != 36
    ):
        raise IntegrityError("follow-up reporting profile changed")
    return value


def enumerate_groups(config: Mapping[str, Any]) -> list[GroupSpec]:
    protocol = _mapping(config["common_protocol"], name="common protocol")
    seeds = tuple(map(int, protocol["split_seeds"]))
    encoders = tuple(map(str, protocol["encoders"]))
    arms = _mapping(config["budget_arms"], name="budget arms")
    shared = tuple(
        BudgetSpec("shared_random_gap", float(value))
        for value in arms["shared_random_gap"]["values"]
    )
    direct = tuple(
        BudgetSpec("direct_nominal", float(value))
        for value in arms["direct_nominal"]["values"]
    )
    primary = (
        BudgetSpec(
            "shared_random_gap",
            float(arms["shared_random_gap"]["primary_value"]),
        ),
    )
    panels = _mapping(config["panels"], name="panels")
    focused = tuple(map(str, panels[CONTROL_UNPAIRED]["focus_corruptions"]))
    paired_focus = tuple(map(str, panels[CONTROL_PAIRED]["corruptions"]))
    all_corruptions = tuple(map(str, panels[CONTROL_UNPAIRED]["corruptions"]))
    if focused != paired_focus:
        raise IntegrityError("paired and unpaired focus panels differ")
    if not set(focused).issubset(set(all_corruptions)):
        raise IntegrityError("focused corruptions are outside the profile panel")

    groups: list[GroupSpec] = []
    for dataset in map(str, panels[NATURAL_PANEL]["datasets"]):
        for encoder in encoders:
            for seed in seeds:
                groups.append(
                    GroupSpec(NATURAL_PANEL, dataset, None, encoder, seed, shared, True)
                )
    for corruption in all_corruptions:
        budgets = shared + direct if corruption in focused else primary
        for encoder in encoders:
            for seed in seeds:
                groups.append(
                    GroupSpec(
                        CONTROL_UNPAIRED,
                        "imagenet_c",
                        corruption,
                        encoder,
                        seed,
                        budgets,
                        True,
                    )
                )
    for corruption in focused:
        for encoder in encoders:
            for seed in seeds:
                groups.append(
                    GroupSpec(
                        CONTROL_PAIRED,
                        "imagenet_c",
                        corruption,
                        encoder,
                        seed,
                        shared + direct,
                        False,
                    )
                )
    groups.sort(key=lambda value: value.group_id)
    if len(groups) != 144 or len({group.group_id for group in groups}) != 144:
        raise IntegrityError(
            "follow-up grouped-compute inventory must contain 144 groups"
        )
    if sum(len(group.budgets) for group in groups) != 720:
        raise IntegrityError("follow-up certificate inventory must contain 720 rows")
    if sum(group.score_accuracy for group in groups) != 108:
        raise IntegrityError(
            "follow-up realized-profile inventory must contain 108 groups"
        )
    return groups


def _store_key(dataset: str, encoder: str, corruption: str | None) -> str:
    condition = dataset if corruption is None else f"{dataset}/{corruption}/severity_5"
    return f"{condition}::{encoder}"


def _pair_selection_key(corruption: str, encoder: str, seed: int) -> str:
    return f"{corruption}::{encoder}::seed_{seed}"


def _build_pair_selection_manifest(
    store, *, seed: int, calibration_rows: int, witness_rows: int
) -> Mapping[str, Any]:
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    pairing = store.pairing()
    paired_reference_ids, paired_target_ids = pairing.validated()
    metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
    provenance = _mapping(metadata.get("model_provenance"), name="model provenance")
    role = str(provenance.get("head_fit_ids_role", ""))
    selection = build_pair_structure_selection(
        reference_ids=reference.ids,
        target_ids=target.ids,
        paired_reference_ids=paired_reference_ids,
        paired_target_ids=paired_target_ids,
        head_fit_reference_ids=store.array(role),
        seed=seed,
        selection_salt="bm-reviewer-followup-20260910-v1",
        protocol=PairStructureProtocol(
            calibration_rows=calibration_rows,
            witness_rows=witness_rows,
        ),
    )
    return selection.manifest


def _external_imagenet_c_hashes(path: Path) -> dict[tuple[str, str], str]:
    value = _mapping(read_json(path), name="external-input manifest")
    if (
        value.get("schema_version") != "marginal_alignment.iclr.external_inputs.v1"
        or value.get("status") != "PASS"
        or value.get("external_input_sha256")
        != "3a8348746ec7a924968c68a29c34e9fa4ac0ea32c6d728d40dc82e0e6844ba97"
    ):
        raise IntegrityError("base external-input manifest identity changed")
    hashes: dict[tuple[str, str], str] = {}
    for record in value.get("native_feature_stores", ()):
        item = _mapping(record, name="external feature-store record")
        if item.get("dataset") == "imagenet_c" and int(item.get("severity", -1)) == 5:
            key = (str(item["corruption"]), str(item["encoder"]))
            hashes[key] = str(item["manifest_sha256"])
    return hashes


def _bind_store(
    path: Path, *, dataset: str, encoder: str, corruption: str | None
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise IntegrityError(f"feature-store manifest is absent or symbolic: {path}")
    store = _load_verified_store(str(path))
    metadata = _mapping(store.manifest.get("metadata"), name="store metadata")
    if store.dataset != dataset or store.encoder != encoder:
        raise IntegrityError(f"feature-store identity mismatch: {path}")
    if corruption is None:
        if store.pairing_spec.get("status") != "unavailable":
            raise IntegrityError("natural-shift stores must not expose pairing")
        intersection = _mapping(
            metadata.get("class_intersection"), name="class intersection"
        )
        target_wnids_role = str(intersection.get("target_wnids_role", ""))
        if target_wnids_role not in store.records:
            raise IntegrityError("natural-shift store lacks target WNIDs")
    else:
        if (
            store.pairing_spec.get("status") != "available"
            or str(metadata.get("corruption")) != corruption
            or int(metadata.get("severity", -1)) != 5
        ):
            raise IntegrityError("ImageNet-C store condition or pairing changed")
        store.pairing().validated()
        target_wnids_role = None
    head = _mapping(store.manifest.get("head"), name="frozen head")
    model = _mapping(metadata.get("model_provenance"), name="model provenance")
    for role in (
        head.get("weight_role"),
        head.get("bias_role"),
        model.get("head_fit_ids_role"),
    ):
        if str(role) not in store.records:
            raise IntegrityError("feature store lacks a frozen-head role")
    return {
        "dataset": dataset,
        "encoder": encoder,
        "corruption": corruption,
        "severity": None if corruption is None else 5,
        "manifest": str(path),
        "manifest_file_sha256": sha256_file(path),
        "manifest_payload_sha256": sha256_json(store.manifest),
        "pairing_status": str(store.pairing_spec.get("status")),
        "target_wnids_role": target_wnids_role,
        "head_weight_role": str(head["weight_role"]),
        "head_bias_role": str(head["bias_role"]),
        "head_fit_ids_role": str(model["head_fit_ids_role"]),
    }


def freeze_lock(
    config_path: str | Path,
    r5_root: str | Path,
    feature_root: str | Path,
    external_input_manifest: str | Path,
    class_index: str | Path,
    output_root: str | Path,
) -> Path:
    """Freeze all source/data/runtime identities before new outcomes exist."""

    config_source = Path(config_path).expanduser().resolve()
    r5 = Path(r5_root).expanduser().resolve()
    features = Path(feature_root).expanduser().resolve()
    external = Path(external_input_manifest).expanduser().resolve()
    class_index_path = Path(class_index).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to reuse follow-up root: {destination}")
    config = _load_config(config_source)
    groups = enumerate_groups(config)
    for path, name in (
        (r5 / "PLAY3_OPTIMIZATION_LOCK.json", "R5 lock"),
        (r5 / "PLAY3_OPTIMIZATION_AGGREGATE.json", "R5 aggregate"),
        (external, "external-input manifest"),
        (class_index_path, "ImageNet class index"),
    ):
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"{name} is absent or symbolic: {path}")
    r5_lock = _mapping(read_json(r5 / "PLAY3_OPTIMIZATION_LOCK.json"), name="R5 lock")
    r5_aggregate = _mapping(
        read_json(r5 / "PLAY3_OPTIMIZATION_AGGREGATE.json"), name="R5 aggregate"
    )
    verify_manifest_checksum(r5_lock)
    verify_manifest_checksum(r5_aggregate)
    if (
        r5_aggregate.get("status") != "PASS"
        or r5_aggregate.get("artifact_status") != "PASS"
        or r5_aggregate.get("scientific_gate_status") != "PASS"
        or r5_aggregate.get("all_required_gates_pass") is not True
    ):
        raise IntegrityError("R5 is not an authenticated all-gates PASS artifact")
    external_hashes = _external_imagenet_c_hashes(external)
    official_index = load_official_class_index(class_index_path)

    stores: dict[str, dict[str, Any]] = {}
    anchors: dict[str, dict[str, Any]] = {}
    pair_selections: dict[str, Mapping[str, Any]] = {}
    required_stores = {
        _store_key(group.dataset, group.encoder, group.corruption) for group in groups
    }
    for key in sorted(required_stores):
        condition, encoder = key.split("::", 1)
        if condition.startswith("imagenet_c/"):
            _, corruption, _ = condition.split("/")
            manifest = (
                features
                / "imagenet_c"
                / encoder
                / corruption
                / "severity_5"
                / "FEATURE_STORE_MANIFEST.json"
            )
            record = _bind_store(
                manifest, dataset="imagenet_c", encoder=encoder, corruption=corruption
            )
            if record["manifest_file_sha256"] != external_hashes.get(
                (corruption, encoder)
            ):
                raise IntegrityError(
                    f"ImageNet-C store is outside the base input lock: {key}"
                )
        else:
            dataset = condition
            manifest = r5 / "stores" / dataset / encoder / "FEATURE_STORE_MANIFEST.json"
            record = _bind_store(
                manifest, dataset=dataset, encoder=encoder, corruption=None
            )
        stores[key] = record

    common = _mapping(config["common_protocol"], name="common protocol")
    c_specs = {
        (str(group.corruption), group.encoder, group.seed)
        for group in groups
        if group.corruption is not None
    }
    for corruption, encoder, seed in sorted(c_specs):
        store_record = stores[_store_key("imagenet_c", encoder, corruption)]
        store = _load_verified_store(str(store_record["manifest"]))
        pair_selections[_pair_selection_key(corruption, encoder, seed)] = dict(
            _build_pair_selection_manifest(
                store,
                seed=seed,
                calibration_rows=int(common["calibration_rows"]),
                witness_rows=int(common["witness_rows"]),
            )
        )

    for group in groups:
        if group.panel != NATURAL_PANEL:
            continue
        anchor_id = (
            f"{group.dataset}__{group.encoder}__{PRIMARY_GEOMETRY}__"
            f"seed_{group.seed}__budget_0100bp"
        )
        anchor_path = r5 / "units" / anchor_id / "UNIT_MANIFEST.json"
        if not anchor_path.is_file() or anchor_path.is_symlink():
            raise IntegrityError(f"R5 natural anchor is absent: {anchor_path}")
        anchor = _mapping(read_json(anchor_path), name="R5 natural anchor")
        verify_manifest_checksum(anchor)
        if (
            anchor.get("status") != "PASS"
            or anchor.get("dataset") != group.dataset
            or anchor.get("encoder") != group.encoder
            or int(anchor.get("split_seed", -1)) != group.seed
            or anchor.get("cost_geometry") != PRIMARY_GEOMETRY
        ):
            raise IntegrityError(f"R5 natural anchor identity changed: {anchor_id}")
        geometry = _mapping(
            _mapping(anchor.get("result"), name="anchor result").get("diagnostics", {}),
            name="anchor diagnostics",
        ).get("geometry")
        geometry = _mapping(geometry, name="anchor geometry")
        anchors[group.group_id] = {
            "path": str(anchor_path),
            "file_sha256": sha256_file(anchor_path),
            "manifest_sha256": str(anchor["manifest_sha256"]),
            "bures_matrix_sha256": str(geometry["bures_matrix_sha256"]),
            "bures_offset_sha256": str(geometry["bures_offset_sha256"]),
            "reference_mean_sha256": str(geometry["reference_mean_sha256"]),
        }

    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_REVIEWER_FOLLOWUP_OUTCOMES",
        "analysis_status": "post_hoc_reviewer_requested",
        "historical_outcomes_inspected_before_design": True,
        "new_followup_outcomes_read_during_freeze": False,
        "config": config,
        "config_file_sha256": sha256_file(config_source),
        "source_files": _source_hashes(),
        "freeze_runtime": _runtime_contract(),
        "r5": {
            "root": str(r5),
            "lock_file_sha256": sha256_file(r5 / "PLAY3_OPTIMIZATION_LOCK.json"),
            "lock_manifest_sha256": str(r5_lock["manifest_sha256"]),
            "aggregate_file_sha256": sha256_file(
                r5 / "PLAY3_OPTIMIZATION_AGGREGATE.json"
            ),
            "aggregate_manifest_sha256": str(r5_aggregate["manifest_sha256"]),
        },
        "external_input_manifest": {
            "path": str(external),
            "file_sha256": sha256_file(external),
            "semantic_sha256": "3a8348746ec7a924968c68a29c34e9fa4ac0ea32c6d728d40dc82e0e6844ba97",
        },
        "class_index": {
            "path": str(class_index_path),
            "file_sha256": official_index.file_sha256,
            "mapping_sha256": official_index.canonical_mapping_sha256,
        },
        "stores": stores,
        "natural_r5_anchors": anchors,
        "pair_structure_selections": pair_selections,
        "expected_group_count": len(groups),
        "expected_certificate_count": sum(len(group.budgets) for group in groups),
        "expected_accuracy_group_count": sum(group.score_accuracy for group in groups),
    }
    destination.mkdir(parents=True)
    lock_path = destination / "REVIEWER_FOLLOWUP_LOCK.json"
    write_json_atomic(lock_path, seal_manifest(payload))
    write_json_atomic(
        destination / "STATE.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "LOCKED",
            "completed_groups": 0,
            "expected_groups": len(groups),
        },
    )
    return lock_path


def _load_lock(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    lock = dict(_mapping(read_json(source), name="follow-up lock"))
    verify_manifest_checksum(lock)
    config_path = (
        _repository_root() / "configs/iclr_extension/reviewer_followup_20260910.json"
    )
    source_config = _load_config(config_path)
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "FROZEN_BEFORE_REVIEWER_FOLLOWUP_OUTCOMES"
        or lock.get("new_followup_outcomes_read_during_freeze") is not False
        or lock.get("config") != source_config
        or lock.get("config_file_sha256") != sha256_file(config_path)
        or lock.get("source_files") != _source_hashes()
        or lock.get("freeze_runtime") != _runtime_contract()
    ):
        raise IntegrityError("follow-up lock identity or runtime changed")
    groups = enumerate_groups(source_config)
    selections = _mapping(lock.get("pair_structure_selections"), name="pair selections")
    stores = _mapping(lock.get("stores"), name="locked stores")
    anchors = _mapping(lock.get("natural_r5_anchors"), name="natural anchors")
    expected_selection_keys = {
        _pair_selection_key(str(group.corruption), group.encoder, group.seed)
        for group in groups
        if group.corruption is not None
    }
    expected_store_keys = {
        _store_key(group.dataset, group.encoder, group.corruption) for group in groups
    }
    expected_anchor_keys = {
        group.group_id for group in groups if group.panel == NATURAL_PANEL
    }
    if (
        int(lock.get("expected_group_count", -1)) != 144
        or int(lock.get("expected_certificate_count", -1)) != 720
        or int(lock.get("expected_accuracy_group_count", -1)) != 108
        or set(anchors) != expected_anchor_keys
        or set(stores) != expected_store_keys
        or set(selections) != expected_selection_keys
        or len(groups) != 144
    ):
        raise IntegrityError("follow-up lock inventory changed")
    for selection in selections.values():
        validate_pair_structure_manifest(
            _mapping(selection, name="pair-selection manifest")
        )
    r5 = _mapping(lock.get("r5"), name="R5 binding")
    r5_root = Path(str(r5["root"])).resolve()
    for relative, field in (
        ("PLAY3_OPTIMIZATION_LOCK.json", "lock_file_sha256"),
        ("PLAY3_OPTIMIZATION_AGGREGATE.json", "aggregate_file_sha256"),
    ):
        bound = r5_root / relative
        if not bound.is_file() or bound.is_symlink() or sha256_file(bound) != r5[field]:
            raise IntegrityError(f"locked R5 input changed: {relative}")
    external = _mapping(
        lock.get("external_input_manifest"), name="external-input binding"
    )
    external_path = Path(str(external["path"])).resolve()
    if (
        not external_path.is_file()
        or external_path.is_symlink()
        or sha256_file(external_path) != external.get("file_sha256")
        or external.get("semantic_sha256")
        != "3a8348746ec7a924968c68a29c34e9fa4ac0ea32c6d728d40dc82e0e6844ba97"
    ):
        raise IntegrityError("locked external-input manifest changed")
    _external_imagenet_c_hashes(external_path)
    class_index = _mapping(lock.get("class_index"), name="class-index binding")
    class_index_path = Path(str(class_index["path"])).resolve()
    if not class_index_path.is_file() or class_index_path.is_symlink():
        raise IntegrityError("locked ImageNet class index is absent or symbolic")
    observed_index = load_official_class_index(class_index_path)
    if (
        observed_index.file_sha256 != class_index["file_sha256"]
        or observed_index.canonical_mapping_sha256 != class_index["mapping_sha256"]
    ):
        raise IntegrityError("locked ImageNet class index changed")
    return lock


def _load_bound_store(lock: Mapping[str, Any], spec: GroupSpec):
    key = _store_key(spec.dataset, spec.encoder, spec.corruption)
    record = _mapping(
        _mapping(lock.get("stores"), name="locked stores").get(key),
        name=f"locked store {key}",
    )
    manifest = Path(str(record["manifest"])).resolve()
    store = _load_verified_store(str(manifest))
    if (
        sha256_file(manifest) != record["manifest_file_sha256"]
        or sha256_json(store.manifest) != record["manifest_payload_sha256"]
        or store.dataset != spec.dataset
        or store.encoder != spec.encoder
    ):
        raise IntegrityError(f"execution store differs from its lock: {key}")
    return store, record


def _content_select(
    features: np.ndarray,
    sample_ids: np.ndarray,
    *,
    requested: int,
    name: str,
    salt: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    canonical, order, digest, _ = _unique_canonical_rows(
        features, name=name, salt=salt, sample_ids=sample_ids
    )
    if requested > len(canonical):
        raise IntegrityError(f"{name} has fewer than {requested} rows")
    selected_order = np.asarray(order[:requested], dtype=np.int64)
    return (
        np.array(canonical[:requested], copy=True),
        selected_order,
        np.asarray(sample_ids)[selected_order],
        digest,
    )


def _natural_split(store, spec: GroupSpec, *, calibration_rows: int, witness_rows: int):
    requested = calibration_rows + witness_rows
    reference = store.endpoint("reference")
    target = store.endpoint("target")
    reference_rows, reference_selection, reference_ids, reference_digest = (
        _content_select(
            reference.features,
            reference.ids,
            requested=requested,
            name="reference_store_features",
            salt=f"play3-v2:{store.dataset}:{spec.seed}:reference-split".encode(
                "ascii"
            ),
        )
    )
    target_rows, target_selection, target_ids, target_digest = _content_select(
        target.features,
        target.ids,
        requested=requested,
        name="target_store_features",
        salt=f"play3-v2:{store.dataset}:{spec.seed}:target-split".encode("ascii"),
    )
    return {
        "reference_features": reference_rows,
        "target_features": target_rows,
        "reference_ids": reference_ids,
        "target_ids": target_ids,
        "reference_selection": reference_selection,
        "target_selection": target_selection,
        "diagnostics": {
            "selection_mode": "r5_exact_independent_content_split",
            "reference_parent_indexed_atom_sha256": reference_digest,
            "target_parent_indexed_atom_sha256": target_digest,
            "pairing_used_for_split_administration": False,
            "underlying_identity_overlap": None,
        },
    }


def _base_protocol(config: Mapping[str, Any]) -> OptimizationCertificateProtocol:
    protocol = _mapping(config["common_protocol"], name="common protocol")
    return OptimizationCertificateProtocol(
        cost_geometry=PRIMARY_GEOMETRY,
        cost_budget_relative=0.0,
        covariance_estimator=str(protocol["covariance_estimator"]),
        covariance_relative_floor=float(protocol["covariance_relative_floor"]),
        maximum_witness_rows=int(protocol["witness_rows"]),
        greedy_neighbors=int(protocol["greedy_neighbors"]),
        dual_bisection_iterations=int(protocol["dual_bisection_iterations"]),
        primal_dual_gap_target=float(protocol["maximum_primal_dual_gap"]),
        assignment_tie_relative_tolerance=float(
            protocol["assignment_tie_relative_tolerance"]
        ),
    )


def _budget_relative(
    budget: BudgetSpec, *, baseline_cost: float, random_mean_cost: float
) -> float:
    if budget.family == "direct_nominal":
        return float(budget.value)
    if budget.family == "shared_random_gap":
        return relative_budget_for_random_gap_share(
            baseline=baseline_cost,
            random_mean=random_mean_cost,
            share=float(budget.value),
        )
    raise IntegrityError(f"unknown budget family: {budget.family}")


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    np.save(path, np.asarray(value), allow_pickle=False)
    return describe_npy(path, relative_to=root)


def _posthoc_endpoint_wnids(
    lock: Mapping[str, Any], store, record: Mapping[str, Any], spec: GroupSpec
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Interpret structural target identities only after outcomes are frozen."""

    official = load_official_class_index(lock["class_index"]["path"])
    target = store.endpoint("target")
    if spec.panel == NATURAL_PANEL:
        role = str(record["target_wnids_role"])
        endpoint_wnids = np.asarray(store.array(role))
        source = {
            "kind": "postfreeze_authenticated_target_wnids",
            "role": role,
            "role_sha256": str(store.records[role]["sha256"]),
        }
    else:
        if target.labels is None:
            raise IntegrityError("ImageNet-C target labels are absent")
        labels = np.asarray(target.labels, dtype=np.int64)
        if (
            labels.ndim != 1
            or len(labels) != len(target.features)
            or np.any(labels < 0)
            or np.any(labels >= len(official.wnids_by_index))
        ):
            raise IntegrityError("ImageNet-C target labels violate the class index")
        endpoint_wnids = np.asarray(
            [official.wnids_by_index[int(value)] for value in labels], dtype="<U9"
        )
        label_role = str(store.endpoint_roles["target"]["labels"])
        source = {
            "kind": "postfreeze_sealed_target_labels_mapped_through_class_index",
            "role": label_role,
            "role_sha256": str(store.records[label_role]["sha256"]),
        }
    if endpoint_wnids.ndim != 1 or len(endpoint_wnids) != len(target.features):
        raise IntegrityError("postfreeze target WNIDs do not cover the endpoint")
    return endpoint_wnids, official.wnids_by_index, source


def _compute_group(lock: Mapping[str, Any], spec: GroupSpec):
    config = _mapping(lock["config"], name="config")
    protocol = _mapping(config["common_protocol"], name="common protocol")
    calibration = int(protocol["calibration_rows"])
    witness_rows = int(protocol["witness_rows"])
    store, store_record = _load_bound_store(lock, spec)
    pair_split = None
    if spec.panel == NATURAL_PANEL:
        split = _natural_split(
            store, spec, calibration_rows=calibration, witness_rows=witness_rows
        )
    else:
        pair_split = build_pair_structure_split(
            store,
            arm="paired" if spec.panel == CONTROL_PAIRED else "disjoint_unpaired",
            seed=spec.seed,
            calibration_rows=calibration,
            witness_rows=witness_rows,
            salt_prefix="bm-reviewer-followup-20260910-v1",
            validate_pair_labels=False,
        )
        locked_selection = _mapping(
            lock["pair_structure_selections"].get(
                _pair_selection_key(str(spec.corruption), spec.encoder, spec.seed)
            ),
            name="locked pair-structure selection",
        )
        if dict(pair_split.selection_manifest) != dict(locked_selection):
            raise IntegrityError(
                "pair-structure split differs from its frozen selection"
            )
        split = {
            "reference_features": pair_split.features_reference,
            "target_features": pair_split.features_target,
            "reference_ids": pair_split.ids_reference,
            "target_ids": pair_split.ids_target,
            "reference_selection": pair_split.selection.reference_rows,
            "target_selection": pair_split.selection.target_rows,
            "diagnostics": {
                **dict(pair_split.diagnostics),
                "selection_manifest_sha256": pair_split.selection_manifest[
                    "manifest_sha256"
                ],
                "labels_materialized_during_split": False,
            },
        }

    reference = np.asarray(split["reference_features"], dtype=np.float64)
    target = np.asarray(split["target_features"], dtype=np.float64)
    reference_ids = np.asarray(split["reference_ids"])
    target_ids = np.asarray(split["target_ids"])
    head = _mapping(store.manifest.get("head"), name="frozen head")
    weight = np.asarray(store.array(str(head["weight_role"])), dtype=np.float64)
    bias = np.asarray(store.array(str(head["bias_role"])), dtype=np.float64)
    base_protocol = _base_protocol(config)
    geometry = fit_certificate_geometry(
        target[:calibration],
        reference[:calibration],
        weight,
        bias,
        protocol=base_protocol,
        target_calibration_ids=target_ids[:calibration],
        reference_calibration_ids=reference_ids[:calibration],
    )
    if spec.panel == NATURAL_PANEL:
        anchor = _mapping(
            lock["natural_r5_anchors"].get(spec.group_id), name="natural R5 anchor"
        )
        for field in (
            "bures_matrix_sha256",
            "bures_offset_sha256",
            "reference_mean_sha256",
        ):
            if geometry.diagnostics.get(field) != anchor[field]:
                raise IntegrityError(
                    f"natural split/Bures fit does not reconstruct R5: {field}"
                )

    witness_target = target[calibration:]
    witness_reference = reference[calibration:]
    witness_target_ids = target_ids[calibration:]
    witness_reference_ids = reference_ids[calibration:]
    frontier = fit_marginal_frontier(
        target[:calibration],
        reference[:calibration],
        prefit_bures_map=geometry.bures_map,
        covariance_relative_floor=float(protocol["covariance_relative_floor"]),
    )
    problem = prepare_certificate_problem(
        geometry,
        witness_target,
        witness_reference,
        target_sample_ids=witness_target_ids,
        reference_sample_ids=witness_reference_ids,
    )
    random_moments = random_cost_moments_from_cost_matrix(problem.cost)
    certificates: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for budget in spec.budgets:
        relative = _budget_relative(
            budget,
            baseline_cost=float(problem.baseline_cost),
            random_mean_cost=float(random_moments["expected_total_cost"]),
        )
        unit_protocol = replace(base_protocol, cost_budget_relative=relative)
        result = construct_optimization_certificate_from_problem(
            problem, protocol=unit_protocol
        )
        exact_geometry = replace(
            geometry,
            protocol=unit_protocol,
            diagnostics={
                **dict(geometry.diagnostics),
                "protocol": unit_protocol.as_dict(),
            },
        )
        exact_assignment, exact = run_exact_subset_validation(
            exact_geometry,
            witness_target,
            witness_reference,
            target_sample_ids=witness_target_ids,
            reference_sample_ids=witness_reference_ids,
            validation_rows=int(protocol["exact_validation_rows"]),
            time_limit_seconds=float(protocol["exact_milp_time_limit_seconds"]),
        )
        if (
            exact.get("status") != "PROVEN_OPTIMAL"
            or float(exact.get("reported_mip_gap", float("inf")))
            > float(protocol["exact_reported_mip_gap_tolerance"])
            or exact.get("certificate_contains_exact_optimum") is not True
        ):
            raise IntegrityError(
                "exact subset validation did not satisfy its tolerance"
            )
        prefix = budget.key
        arrays.update(
            {
                f"{prefix}__assignment_world_zero": result.assignment_world_zero,
                f"{prefix}__assignment_world_one": result.assignment_world_one,
                f"{prefix}__prediction_world_zero": result.prediction_world_zero,
                f"{prefix}__prediction_world_one": result.prediction_world_one,
                f"{prefix}__target_canonical_order": result.target_canonical_order,
                f"{prefix}__reference_canonical_order": result.reference_canonical_order,
                f"{prefix}__exact_subset_assignment": exact_assignment,
            }
        )
        certificates.append(
            {
                "budget_key": budget.key,
                "budget_family": budget.family,
                "declared_budget_value": float(budget.value),
                "derived_cost_budget_relative_to_minimum": float(relative),
                "result": result.summary(),
                "exact_small_instance_validation": exact,
            }
        )

    pair_structure_postfit = None
    if pair_split is not None and spec.panel == CONTROL_PAIRED:
        raw_assignment = pair_split.selection.witness_true_pair_assignment
        if raw_assignment is None or len(raw_assignment) != witness_rows:
            raise IntegrityError("paired witness lacks its administrative pairing")
        target_order = np.asarray(problem.target_canonical_order, dtype=np.int64)
        reference_order = np.asarray(problem.reference_canonical_order, dtype=np.int64)
        inverse_reference = np.empty(witness_rows, dtype=np.int64)
        inverse_reference[reference_order] = np.arange(witness_rows, dtype=np.int64)
        canonical_assignment = inverse_reference[
            np.asarray(raw_assignment, dtype=np.int64)[target_order]
        ]
        canonical_row = np.arange(witness_rows, dtype=np.int64)
        true_pair_cost = float(
            problem.cost[canonical_row, canonical_assignment].sum(dtype=np.float64)
        )
        pair_structure_postfit = {
            "status": "PASS",
            "scientific_result": True,
            "phase": "after_geometry_cost_reward_budgets_and_assignments_frozen",
            "true_pair_assignment_cost": true_pair_cost,
            "true_pair_cost_relative_to_minimum": (
                true_pair_cost / float(problem.baseline_cost) - 1.0
            ),
            "true_pair_excess_fraction_of_random_minus_minimum": (
                (true_pair_cost - float(problem.baseline_cost))
                / (
                    float(random_moments["expected_total_cost"])
                    - float(problem.baseline_cost)
                )
            ),
            "true_pair_assignment_sha256": _array_sha256(canonical_assignment),
            "pairing_used_for_geometry_cost_reward_or_optimization": False,
        }
        arrays["postfit__true_pair_assignment"] = canonical_assignment

    accuracy = None
    association_record = None
    if spec.score_accuracy:
        primary = next(
            row
            for row in certificates
            if row["budget_family"] == "shared_random_gap"
            and float(row["declared_budget_value"]) == 0.05
        )
        profile_binding_sha256 = sha256_json(
            {
                "schema_version": "marginal_alignment.reviewer_followup_profile_binding.v1",
                "lock_manifest_sha256": lock["manifest_sha256"],
                "group_id": spec.group_id,
                "selection": split["diagnostics"],
                "geometry": geometry.diagnostics,
                "frontier": frontier.diagnostics,
                "prepared_problem": {
                    "target_indexed_atom_sha256": problem.target_indexed_atom_sha256,
                    "reference_indexed_atom_sha256": problem.reference_indexed_atom_sha256,
                    "baseline_cost": float(problem.baseline_cost),
                    "random_bijection": random_moments,
                },
                "primary_certificate": primary,
            }
        )
        frozen = freeze_realized_profile_predictions(
            frontier,
            witness_target,
            weight,
            bias,
            dataset=(
                spec.dataset
                if spec.corruption is None
                else f"imagenet_c:{spec.corruption}"
            ),
            encoder=spec.encoder,
            split_seed=spec.seed,
            target_endpoint_rows=len(store.endpoint("target").features),
            target_witness_original_rows=np.asarray(
                split["target_selection"], dtype=np.int64
            )[calibration:],
            profile_binding_sha256=profile_binding_sha256,
            expected_bures_matrix_sha256=str(
                geometry.diagnostics["bures_matrix_sha256"]
            ),
            expected_bures_offset_sha256=str(
                geometry.diagnostics["bures_offset_sha256"]
            ),
            expected_witness_rows=witness_rows,
        )
        # This call is the first outcome-bearing access to target WNIDs/labels.
        endpoint_wnids, class_index_wnids, label_source = _posthoc_endpoint_wnids(
            lock, store, store_record, spec
        )
        accuracy = score_frozen_realized_profile(
            frozen,
            endpoint_wnids,
            class_index_wnids,
            expected_classes=1000,
        )
        accuracy["target_identity_source"] = label_source
        cluster = (
            f"natural:{spec.dataset}"
            if spec.corruption is None
            else f"imagenet_c:{spec.corruption}"
        )
        family = "natural" if spec.corruption is None else "imagenet_c"
        association_record = profile_gain_record(
            accuracy,
            cluster=cluster,
            family=family,
            profile_disagreement=float(
                primary["result"]["maximum_disagreement_primal_lower_bound"]
            ),
            profile_binding_sha256=profile_binding_sha256,
        )
        for method, prediction in frozen.predictions.items():
            arrays[f"accuracy__prediction__{method}"] = np.asarray(prediction)

    cost_scale_summary = summarize_cost_scale(
        minimum_cost=float(problem.baseline_cost),
        random_moments=dict(random_moments),
        budgets=[
            float(certificate["derived_cost_budget_relative_to_minimum"])
            for certificate in certificates
        ],
        identity_cost=(
            None
            if pair_structure_postfit is None
            else float(pair_structure_postfit["true_pair_assignment_cost"])
        ),
    )

    scientific = {
        "schema_version": "marginal_alignment.reviewer_followup_scientific.v1",
        "group_id": spec.group_id,
        "panel": spec.panel,
        "dataset": spec.dataset,
        "corruption": spec.corruption,
        "encoder": spec.encoder,
        "split_seed": spec.seed,
        "calibration_rows": calibration,
        "witness_rows": witness_rows,
        "cost_geometry": PRIMARY_GEOMETRY,
        "pairing_used_for_split_administration": spec.panel != NATURAL_PANEL,
        "pairing_available_to_map_or_certificate_optimizer": False,
        "labels_available_to_map_or_certificate_optimizer": False,
        "target_label_values_interpreted_only_after_predictions_and_certificates_frozen": True,
        "selection": dict(split["diagnostics"]),
        "geometry": dict(geometry.diagnostics),
        "frontier": dict(frontier.diagnostics),
        "prediction_hashes_before_label_access": (
            None
            if accuracy is None
            else accuracy["prediction_freeze_binding"]["prediction_sha256"]
        ),
        "cost_scale": {
            "baseline_minimum_cost": float(problem.baseline_cost),
            "random_bijection": random_moments,
            "normalized_diagnostics": cost_scale_summary,
        },
        "pair_structure_postfit": pair_structure_postfit,
        "certificates": certificates,
        "accuracy": accuracy,
        "profile_gain_association_record": association_record,
    }
    return scientific, arrays


def _group_path(root: Path, spec: GroupSpec) -> Path:
    return root / "groups" / spec.group_id


def _expected_array_names(spec: GroupSpec) -> set[str]:
    names: set[str] = set()
    for budget in spec.budgets:
        prefix = budget.key
        names.update(
            {
                f"{prefix}__assignment_world_zero",
                f"{prefix}__assignment_world_one",
                f"{prefix}__prediction_world_zero",
                f"{prefix}__prediction_world_one",
                f"{prefix}__target_canonical_order",
                f"{prefix}__reference_canonical_order",
                f"{prefix}__exact_subset_assignment",
            }
        )
    if spec.score_accuracy:
        names.update(
            f"accuracy__prediction__{method}" for method in REALIZED_PROFILE_METHODS
        )
    if spec.panel == CONTROL_PAIRED:
        names.add("postfit__true_pair_assignment")
    return names


def _write_group(lock_path: Path, spec: GroupSpec) -> Path:
    root = lock_path.parent
    destination = _group_path(root, spec)
    if destination.exists() or destination.is_symlink():
        _validate_group(lock_path, spec, destination)
        return destination / "GROUP_MANIFEST.json"
    stale_temporaries = (
        sorted(destination.parent.glob(f".{destination.name}.tmp.*"))
        if destination.parent.is_dir()
        else []
    )
    if stale_temporaries:
        raise IntegrityError(
            "partial temporary group requires explicit preservation and cleanup: "
            + ", ".join(str(path) for path in stale_temporaries)
        )
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale temporary group exists: {temporary}")
    started = time.perf_counter_ns()
    scientific, arrays = _compute_group(_load_lock(lock_path), spec)
    temporary.mkdir(parents=True)
    artifact_records = {
        name: _save_array(temporary, name, value)
        for name, value in sorted(arrays.items())
    }
    payload = {
        "schema_version": GROUP_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "lock_file_sha256": sha256_file(lock_path),
        "scientific": scientific,
        "arrays": artifact_records,
        "compute": {
            "elapsed_seconds_before_manifest_write": (time.perf_counter_ns() - started)
            / 1e9,
            "runtime": _runtime_contract(),
        },
    }
    write_json_atomic(temporary / "GROUP_MANIFEST.json", seal_manifest(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return destination / "GROUP_MANIFEST.json"


def _validate_group(
    lock_path: Path, spec: GroupSpec, group_path: Path
) -> dict[str, Any]:
    if not group_path.is_dir() or group_path.is_symlink():
        raise IntegrityError(f"group path is absent or symbolic: {group_path}")
    manifest_path = group_path / "GROUP_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise IntegrityError(f"group manifest is absent or symbolic: {manifest_path}")
    manifest = dict(_mapping(read_json(manifest_path), name="group manifest"))
    verify_manifest_checksum(manifest)
    if set(manifest) != {
        "schema_version",
        "status",
        "scientific_result",
        "lock_file_sha256",
        "scientific",
        "arrays",
        "compute",
        "manifest_sha256",
    }:
        raise IntegrityError(
            f"follow-up group manifest inventory changed: {spec.group_id}"
        )
    scientific = _mapping(manifest.get("scientific"), name="scientific payload")
    certificates = scientific.get("certificates")
    expected_budgets = [
        (budget.key, budget.family, float(budget.value)) for budget in spec.budgets
    ]
    observed_budgets = (
        [
            (
                str(row.get("budget_key")),
                str(row.get("budget_family")),
                float(row.get("declared_budget_value", float("nan"))),
            )
            for row in certificates
        ]
        if isinstance(certificates, list)
        and all(isinstance(row, Mapping) for row in certificates)
        else None
    )
    accuracy = scientific.get("accuracy")
    association = scientific.get("profile_gain_association_record")
    if (
        manifest.get("schema_version") != GROUP_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("scientific_result") is not True
        or manifest.get("lock_file_sha256") != sha256_file(lock_path)
        or scientific.get("group_id") != spec.group_id
        or scientific.get("panel") != spec.panel
        or scientific.get("dataset") != spec.dataset
        or scientific.get("corruption") != spec.corruption
        or scientific.get("encoder") != spec.encoder
        or scientific.get("split_seed") != spec.seed
        or scientific.get("pairing_available_to_map_or_certificate_optimizer")
        is not False
        or scientific.get("labels_available_to_map_or_certificate_optimizer")
        is not False
        or scientific.get(
            "target_label_values_interpreted_only_after_predictions_and_certificates_frozen"
        )
        is not True
        or observed_budgets != expected_budgets
        or (isinstance(accuracy, Mapping) and isinstance(association, Mapping))
        is not spec.score_accuracy
        or (scientific.get("pair_structure_postfit") is not None)
        is not (spec.panel == CONTROL_PAIRED)
    ):
        raise IntegrityError(f"invalid follow-up group: {spec.group_id}")
    arrays = _mapping(manifest.get("arrays"), name="group arrays")
    expected_arrays = _expected_array_names(spec)
    if set(arrays) != expected_arrays:
        raise IntegrityError(
            f"follow-up group array inventory changed: {spec.group_id}"
        )
    for name, raw_record in arrays.items():
        record = _mapping(raw_record, name="array record")
        if (
            set(record) != {"path", "sha256", "size_bytes", "format", "shape", "dtype"}
            or record.get("path") != f"{name}.npy"
        ):
            raise IntegrityError(
                f"follow-up group array path changed: {spec.group_id}/{name}"
            )
        verify_file_record(group_path, record)
    expected_files = {"GROUP_MANIFEST.json"}.union(
        f"{name}.npy" for name in expected_arrays
    )
    entries = list(group_path.iterdir())
    observed_files = {path.name for path in entries}
    if observed_files != expected_files or any(
        not path.is_file() or path.is_symlink() for path in entries
    ):
        raise IntegrityError(f"follow-up group file inventory changed: {spec.group_id}")
    return manifest


def shard_groups(
    groups: Sequence[GroupSpec], shard_count: int
) -> list[list[GroupSpec]]:
    """Keep each verified feature store in one balanced worker process."""

    if shard_count < 1:
        raise ValueError("shard count must be positive")
    bundles: dict[str, list[GroupSpec]] = {}
    for group in groups:
        key = _store_key(group.dataset, group.encoder, group.corruption)
        bundles.setdefault(key, []).append(group)
    weighted = sorted(
        bundles.items(),
        key=lambda item: (-sum(len(group.budgets) for group in item[1]), item[0]),
    )
    shards: list[list[GroupSpec]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for _, bundle in weighted:
        destination = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[destination].extend(sorted(bundle, key=lambda group: group.group_id))
        loads[destination] += sum(len(group.budgets) for group in bundle)
    return shards


def run_shard(lock_path: str | Path, *, shard_index: int, shard_count: int) -> Path:
    source = Path(lock_path).expanduser().resolve()
    lock = _load_lock(source)
    groups = enumerate_groups(lock["config"])
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    selected = shard_groups(groups, shard_count)[shard_index]
    for completed, group in enumerate(selected, start=1):
        _write_group(source, group)
        write_json_atomic(
            source.parent / "shards" / f"run_{shard_index:03d}.json",
            {
                "schema_version": STATE_SCHEMA,
                "status": "RUNNING" if completed < len(selected) else "PASS",
                "shard_index": shard_index,
                "shard_count": shard_count,
                "completed_groups": completed,
                "expected_groups": len(selected),
                "latest_group": group.group_id,
            },
        )
        print(
            f"Shard {shard_index}: {completed}/{len(selected)}: {group.group_id}",
            flush=True,
        )
    return source.parent / "shards" / f"run_{shard_index:03d}.json"


def verify_shard(lock_path: str | Path, *, shard_index: int, shard_count: int) -> Path:
    source = Path(lock_path).expanduser().resolve()
    lock = _load_lock(source)
    groups = enumerate_groups(lock["config"])
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    selected = shard_groups(groups, shard_count)[shard_index]
    records = []
    for completed, group in enumerate(selected, start=1):
        group_path = _group_path(source.parent, group)
        manifest = _validate_group(source, group, group_path)
        expected_scientific, expected_arrays = _compute_group(lock, group)
        if manifest["scientific"] != expected_scientific:
            raise IntegrityError(f"scientific reconstruction failed: {group.group_id}")
        for name, expected in expected_arrays.items():
            record = _mapping(manifest["arrays"].get(name), name=f"array {name}")
            observed = np.load(
                verify_file_record(group_path, record), allow_pickle=False
            )
            if not np.array_equal(observed, expected):
                raise IntegrityError(
                    f"array reconstruction failed: {group.group_id}/{name}"
                )
        records.append(
            {
                "group_id": group.group_id,
                "manifest_file_sha256": sha256_file(group_path / "GROUP_MANIFEST.json"),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
        print(
            f"Verify {shard_index}: {completed}/{len(selected)}: {group.group_id}",
            flush=True,
        )
    output = source.parent / "verification" / f"shard_{shard_index:03d}.json"
    write_json_atomic(
        output,
        seal_manifest(
            {
                "schema_version": VERIFICATION_SCHEMA,
                "status": "PASS",
                "lock_file_sha256": sha256_file(source),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "groups": records,
            }
        ),
    )
    return output


def _certificate_index(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scientific in groups:
        for certificate in scientific["certificates"]:
            result = certificate["result"]
            exact = certificate["exact_small_instance_validation"]
            cost_scale = scientific["cost_scale"]
            pair_diagnostic = scientific.get("pair_structure_postfit")
            rows.append(
                {
                    "group_id": scientific["group_id"],
                    "panel": scientific["panel"],
                    "dataset": scientific["dataset"],
                    "corruption": scientific["corruption"],
                    "encoder": scientific["encoder"],
                    "split_seed": scientific["split_seed"],
                    "budget_key": certificate["budget_key"],
                    "budget_family": certificate["budget_family"],
                    "declared_budget_value": certificate["declared_budget_value"],
                    "derived_cost_budget_relative_to_minimum": certificate[
                        "derived_cost_budget_relative_to_minimum"
                    ],
                    "baseline_minimum_cost": cost_scale["baseline_minimum_cost"],
                    "expected_uniform_random_bijection_cost": cost_scale[
                        "random_bijection"
                    ]["expected_total_cost"],
                    "allowed_cost": result["allowed_cost"],
                    "primal_cost": result["primal_cost"],
                    "selected_world_disagreement": result[
                        "selected_world_disagreement"
                    ],
                    "selected_two_world_minimax_recovery": result[
                        "selected_two_world_minimax_recovery"
                    ],
                    "lower_disagreement": result[
                        "maximum_disagreement_primal_lower_bound"
                    ],
                    "upper_disagreement": result[
                        "maximum_disagreement_dual_upper_bound"
                    ],
                    "gap": result["primal_dual_gap"],
                    "exact_64_row_maximum_disagreement": exact[
                        "maximum_disagreement_exact"
                    ],
                    "exact_64_row_reported_mip_gap": exact["reported_mip_gap"],
                    "exact_64_row_certificate_contains_optimum": exact[
                        "certificate_contains_exact_optimum"
                    ],
                    "postfit_true_pair_cost": (
                        None
                        if pair_diagnostic is None
                        else pair_diagnostic["true_pair_assignment_cost"]
                    ),
                    "postfit_true_pair_excess_fraction_of_random_gap": (
                        None
                        if pair_diagnostic is None
                        else pair_diagnostic[
                            "true_pair_excess_fraction_of_random_minus_minimum"
                        ]
                    ),
                }
            )
    return rows


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    values = list(records)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for record in values:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _profile_rows(scientific: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in scientific:
        association = group.get("profile_gain_association_record")
        accuracy = group.get("accuracy")
        if association is None and accuracy is None:
            continue
        if not isinstance(association, Mapping) or not isinstance(accuracy, Mapping):
            raise IntegrityError("realized profile is only partially populated")
        primary = [
            value
            for value in group["certificates"]
            if value["budget_family"] == "shared_random_gap"
            and float(value["declared_budget_value"]) == 0.05
        ]
        if len(primary) != 1:
            raise IntegrityError("realized profile lacks one primary certificate")
        result = primary[0]["result"]
        method_accuracy = _mapping(
            accuracy.get("per_method_target_accuracy"), name="method accuracy"
        )
        if len(method_accuracy) != len(REALIZED_PROFILE_METHODS) or set(
            method_accuracy
        ) != set(REALIZED_PROFILE_METHODS):
            raise IntegrityError("realized profile method inventory changed")
        rows.append(
            {
                "group_id": group["group_id"],
                "panel": group["panel"],
                "dataset": group["dataset"],
                "corruption": group["corruption"],
                "encoder": group["encoder"],
                "split_seed": group["split_seed"],
                "shared_random_gap_q": 0.05,
                "maximum_disagreement_primal_lower_bound": result[
                    "maximum_disagreement_primal_lower_bound"
                ],
                "maximum_disagreement_dual_upper_bound": result[
                    "maximum_disagreement_dual_upper_bound"
                ],
                "selected_two_world_minimax_recovery": result[
                    "selected_two_world_minimax_recovery"
                ],
                "selected_world_prediction_agreement": (
                    1.0 - float(result["selected_world_disagreement"])
                ),
                "profile_binding_sha256": association["profile_binding_sha256"],
                "accuracy_none": method_accuracy["none"],
                "accuracy_mean": method_accuracy["mean"],
                "accuracy_diagonal": method_accuracy["diagonal"],
                "accuracy_bures_oas": method_accuracy["bures_oas"],
                "bures_minus_none_target_accuracy": accuracy[
                    "bures_minus_none_target_accuracy"
                ],
                "profile_gain_association_record": dict(association),
                "target_identity_source": accuracy["target_identity_source"],
            }
        )
    rows.sort(key=lambda value: value["group_id"])
    if len(rows) != 108:
        raise IntegrityError("realized-profile inventory must contain 108 rows")
    return rows


def _primary_pair_inference(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    expected_corruptions: Sequence[str] | None = None,
    rng_seed: int = 20260910,
) -> dict[str, Any]:
    primary = [
        row
        for row in contrasts
        if row["budget_family"] == "shared_random_gap"
        and float(row["declared_budget_value"]) == 0.05
    ]
    if len(primary) != 36:
        raise IntegrityError("primary paired contrast must contain 36 cells")
    expected_encoders = ("clip", "dinov3")
    expected_seeds = (201, 202, 203)
    by_corruption: dict[str, dict[tuple[str, int], float]] = {}
    for row in primary:
        corruption = str(row["corruption"])
        identity = (str(row["encoder"]), int(row["split_seed"]))
        value = float(row["selected_feasible_paired_minus_disjoint_unpaired"])
        if not np.isfinite(value):
            raise IntegrityError("pair inference contains a nonfinite contrast")
        cells = by_corruption.setdefault(corruption, {})
        if identity in cells:
            raise IntegrityError(
                "pair inference contains a duplicate encoder-seed cell"
            )
        cells[identity] = value
    required_cells = {
        (encoder, seed) for encoder in expected_encoders for seed in expected_seeds
    }
    expected_cluster_set = (
        set(map(str, expected_corruptions))
        if expected_corruptions is not None
        else set(by_corruption)
    )
    if (
        len(expected_cluster_set) != 6
        or set(by_corruption) != expected_cluster_set
        or any(set(cells) != required_cells for cells in by_corruption.values())
    ):
        raise IntegrityError(
            "pair inference must nest 3 seeds x 2 encoders per corruption"
        )
    cluster_rows = []
    for corruption, cells in sorted(by_corruption.items()):
        encoder_rows = [
            {
                "encoder": encoder,
                "split_seed_repeats": len(expected_seeds),
                "paired_minus_disjoint_unpaired_seed_mean": float(
                    np.mean([cells[(encoder, seed)] for seed in expected_seeds])
                ),
            }
            for encoder in expected_encoders
        ]
        cluster_rows.append(
            {
                "corruption": corruption,
                "nested_encoder_seed_cells": len(cells),
                "encoder_rows": encoder_rows,
                "paired_minus_disjoint_unpaired_mean": float(
                    np.mean(
                        [
                            row["paired_minus_disjoint_unpaired_seed_mean"]
                            for row in encoder_rows
                        ]
                    )
                ),
            }
        )
    cluster_values = np.asarray(
        [row["paired_minus_disjoint_unpaired_mean"] for row in cluster_rows],
        dtype=np.float64,
    )
    observed = float(np.mean(cluster_values))
    sign_flip = []
    for mask in range(1 << len(cluster_values)):
        signs = np.asarray(
            [
                1.0 if mask & (1 << index) else -1.0
                for index in range(len(cluster_values))
            ]
        )
        sign_flip.append(float(np.mean(signs * cluster_values)))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(observed), 1.0)
    p_value = float(np.mean(np.abs(np.asarray(sign_flip)) >= abs(observed) - tolerance))
    rng = np.random.default_rng(rng_seed)
    bootstrap = np.asarray(
        [
            np.mean(rng.choice(cluster_values, size=len(cluster_values), replace=True))
            for _ in range(10_000)
        ]
    )
    return {
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_motivated_supportive",
        "estimand": "selected_feasible_paired_minus_disjoint_unpaired",
        "independent_unit": "fixed_imagenet_c_corruption_cluster",
        "cluster_count": 6,
        "encoders_and_overlapping_split_seeds_nested_within_cluster": True,
        "cluster_rows": cluster_rows,
        "cluster_mean": observed,
        "exhaustive_two_sided_sign_flip": {
            "assignments": 64,
            "p_value": p_value,
            "minimum_attainable_two_sided_p_value": 0.03125,
        },
        "descriptive_cluster_bootstrap": {
            "replicates": 10_000,
            "rng_seed": rng_seed,
            "lower": float(np.quantile(bootstrap, 0.025)),
            "upper": float(np.quantile(bootstrap, 0.975)),
            "fixed_panel_not_population_confidence_interval": True,
        },
        "selected_lower_bound_contrast_does_not_order_exact_maxima": True,
    }


def aggregate(lock_path: str | Path, *, verification_shards: int) -> Path:
    source = Path(lock_path).expanduser().resolve()
    lock = _load_lock(source)
    specs = enumerate_groups(lock["config"])
    expected_shards = shard_groups(specs, verification_shards)
    verification_bindings: dict[str, tuple[str, str]] = {}
    for index in range(verification_shards):
        path = source.parent / "verification" / f"shard_{index:03d}.json"
        report = _mapping(read_json(path), name="verification report")
        verify_manifest_checksum(report)
        if set(report) != {
            "schema_version",
            "status",
            "lock_file_sha256",
            "shard_index",
            "shard_count",
            "groups",
            "manifest_sha256",
        }:
            raise IntegrityError("verification report field inventory changed")
        raw_rows = report.get("groups")
        if not isinstance(raw_rows, list) or not all(
            isinstance(row, Mapping) for row in raw_rows
        ):
            raise IntegrityError("verification report group inventory is malformed")
        expected_ids = [spec.group_id for spec in expected_shards[index]]
        observed_ids = [str(row.get("group_id")) for row in raw_rows]
        if (
            report.get("schema_version") != VERIFICATION_SCHEMA
            or report.get("status") != "PASS"
            or report.get("lock_file_sha256") != sha256_file(source)
            or int(report.get("shard_index", -1)) != index
            or int(report.get("shard_count", -1)) != verification_shards
            or observed_ids != expected_ids
        ):
            raise IntegrityError("verification report identity changed")
        for row in raw_rows:
            if set(row) != {
                "group_id",
                "manifest_file_sha256",
                "manifest_sha256",
            }:
                raise IntegrityError(
                    "verification group binding field inventory changed"
                )
            group_id = str(row["group_id"])
            file_sha256 = str(row.get("manifest_file_sha256"))
            manifest_sha256 = str(row.get("manifest_sha256"))
            if (
                group_id in verification_bindings
                or not _valid_sha256(file_sha256)
                or not _valid_sha256(manifest_sha256)
            ):
                raise IntegrityError(
                    "verification report binding is malformed or repeated"
                )
            verification_bindings[group_id] = (file_sha256, manifest_sha256)
    if set(verification_bindings) != {spec.group_id for spec in specs}:
        raise IntegrityError("full-reconstruction verification coverage is incomplete")

    scientific: list[Mapping[str, Any]] = []
    group_records = []
    for spec in specs:
        group_path = _group_path(source.parent, spec)
        manifest = _validate_group(source, spec, group_path)
        observed_binding = (
            sha256_file(group_path / "GROUP_MANIFEST.json"),
            str(manifest["manifest_sha256"]),
        )
        if observed_binding != verification_bindings[spec.group_id]:
            raise IntegrityError(
                f"group changed after full reconstruction: {spec.group_id}"
            )
        scientific.append(_mapping(manifest["scientific"], name="scientific payload"))
        group_records.append(
            {
                "group_id": spec.group_id,
                "manifest_file_sha256": observed_binding[0],
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    certificate_rows = _certificate_index(scientific)
    if len(certificate_rows) != 720:
        raise IntegrityError("aggregate certificate inventory is incomplete")
    protocol = _mapping(lock["config"]["common_protocol"], name="common protocol")
    if any(
        float(row["gap"]) > float(protocol["maximum_primal_dual_gap"])
        for row in certificate_rows
    ):
        raise IntegrityError("at least one certificate misses the locked gap target")
    _write_jsonl(source.parent / "CERTIFICATE_ROWS.jsonl", certificate_rows)
    profile_rows = _profile_rows(scientific)
    _write_jsonl(source.parent / "REALIZED_PROFILE_ROWS.jsonl", profile_rows)
    profile_config = _mapping(
        lock["config"]["reporting_profiles"]["primary_realized_profile"],
        name="realized profile config",
    )
    profile_association = cluster_aware_profile_gain_association(
        [row["profile_gain_association_record"] for row in profile_rows],
        expected_encoders=tuple(map(str, protocol["encoders"])),
        expected_split_seeds=tuple(map(int, protocol["split_seeds"])),
        bootstrap_replicates=int(profile_config["association_bootstrap_replicates"]),
        permutation_replicates=int(
            profile_config["association_permutation_replicates"]
        ),
        rng_seed=int(profile_config["association_rng_seed"]),
    )

    # Bound-aware pair-structure contrasts.  For each matched cell, the
    # identified difference paired-minus-unpaired is [L_p-U_u, U_p-L_u].
    index = {
        (
            row["panel"],
            row["dataset"],
            row["corruption"],
            row["encoder"],
            row["split_seed"],
            row["budget_key"],
        ): row
        for row in certificate_rows
    }
    contrasts = []
    focused = set(lock["config"]["panels"][CONTROL_PAIRED]["corruptions"])
    for row in certificate_rows:
        if row["panel"] != CONTROL_UNPAIRED or row["corruption"] not in focused:
            continue
        paired = index[
            (
                CONTROL_PAIRED,
                "imagenet_c",
                row["corruption"],
                row["encoder"],
                row["split_seed"],
                row["budget_key"],
            )
        ]
        contrasts.append(
            {
                "corruption": row["corruption"],
                "encoder": row["encoder"],
                "split_seed": row["split_seed"],
                "budget_key": row["budget_key"],
                "budget_family": row["budget_family"],
                "declared_budget_value": row["declared_budget_value"],
                "selected_feasible_paired_minus_disjoint_unpaired": (
                    paired["lower_disagreement"] - row["lower_disagreement"]
                ),
                "exact_maximum_difference_lower_bound": (
                    paired["lower_disagreement"] - row["upper_disagreement"]
                ),
                "exact_maximum_difference_upper_bound": (
                    paired["upper_disagreement"] - row["lower_disagreement"]
                ),
                "numerical_bound_intervals_separate": bool(
                    paired["lower_disagreement"] > row["upper_disagreement"]
                    or paired["upper_disagreement"] < row["lower_disagreement"]
                ),
            }
        )
    if len(contrasts) != 288:
        raise IntegrityError("pair-structure contrast inventory is incomplete")
    _write_jsonl(source.parent / "PAIR_STRUCTURE_CONTRASTS.jsonl", contrasts)
    pair_inference = _primary_pair_inference(
        contrasts,
        expected_corruptions=lock["config"]["panels"][CONTROL_PAIRED]["corruptions"],
    )

    # Directly answer the reviewer's requested check U_paired < L_Sketch.
    sketch_separations = []
    for paired in certificate_rows:
        if (
            paired["panel"] != CONTROL_PAIRED
            or paired["budget_family"] != "shared_random_gap"
        ):
            continue
        sketch = index[
            (
                NATURAL_PANEL,
                "imagenet_sketch",
                None,
                paired["encoder"],
                paired["split_seed"],
                paired["budget_key"],
            )
        ]
        sketch_separations.append(
            {
                "corruption": paired["corruption"],
                "encoder": paired["encoder"],
                "split_seed": paired["split_seed"],
                "budget_key": paired["budget_key"],
                "budget_family": paired["budget_family"],
                "declared_budget_value": paired["declared_budget_value"],
                "sketch_lower_disagreement": sketch["lower_disagreement"],
                "sketch_upper_disagreement": sketch["upper_disagreement"],
                "paired_imagenet_c_lower_disagreement": paired["lower_disagreement"],
                "paired_imagenet_c_upper_disagreement": paired["upper_disagreement"],
                "sketch_minus_paired_exact_maximum_lower_bound": (
                    sketch["lower_disagreement"] - paired["upper_disagreement"]
                ),
                "sketch_minus_paired_exact_maximum_upper_bound": (
                    sketch["upper_disagreement"] - paired["lower_disagreement"]
                ),
                "paired_upper_below_sketch_lower": bool(
                    paired["upper_disagreement"] < sketch["lower_disagreement"]
                ),
            }
        )
    if len(sketch_separations) != 180:
        raise IntegrityError("Sketch/paired bound-separation inventory is incomplete")
    _write_jsonl(
        source.parent / "SKETCH_PAIRED_BOUND_SEPARATIONS.jsonl",
        sketch_separations,
    )

    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "analysis_status": "post_hoc_reviewer_requested",
        "lock_file_sha256": sha256_file(source),
        "lock_manifest_sha256": lock["manifest_sha256"],
        "groups": len(group_records),
        "certificates": len(certificate_rows),
        "realized_profile_groups": len(profile_rows),
        "pair_structure_contrasts": len(contrasts),
        "all_certificate_gaps_pass": True,
        "full_reconstruction_verified": True,
        "profile_gain_association": profile_association,
        "primary_pair_structure_inference": pair_inference,
        "sketch_paired_bound_separations": len(sketch_separations),
        "sketch_paired_strict_separations": sum(
            row["paired_upper_below_sketch_lower"] for row in sketch_separations
        ),
        "artifacts": {
            "certificate_rows_sha256": sha256_file(
                source.parent / "CERTIFICATE_ROWS.jsonl"
            ),
            "realized_profile_rows_sha256": sha256_file(
                source.parent / "REALIZED_PROFILE_ROWS.jsonl"
            ),
            "pair_structure_contrasts_sha256": sha256_file(
                source.parent / "PAIR_STRUCTURE_CONTRASTS.jsonl"
            ),
            "sketch_paired_bound_separations_sha256": sha256_file(
                source.parent / "SKETCH_PAIRED_BOUND_SEPARATIONS.jsonl"
            ),
        },
        "group_manifests": group_records,
    }
    output = source.parent / "REVIEWER_FOLLOWUP_AGGREGATE.json"
    write_json_atomic(output, seal_manifest(payload))
    done_path = source.parent / "DONE.json"
    write_json_atomic(
        done_path,
        seal_manifest(
            {
                "schema_version": "marginal_alignment.reviewer_followup_done.v1",
                "status": "PASS",
                "aggregate_file_sha256": sha256_file(output),
                "aggregate_manifest_sha256": read_json(output)["manifest_sha256"],
            }
        ),
    )
    write_json_atomic(
        source.parent / "STATE.json",
        {
            "schema_version": STATE_SCHEMA,
            "status": "COMPLETE",
            "completed_groups": len(group_records),
            "expected_groups": len(specs),
            "aggregate_file_sha256": sha256_file(output),
            "done_file_sha256": sha256_file(done_path),
        },
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--r5-root", required=True)
    freeze.add_argument("--feature-root", required=True)
    freeze.add_argument("--external-input-manifest", required=True)
    freeze.add_argument("--class-index", required=True)
    freeze.add_argument("--output-root", required=True)
    for name in ("run-shard", "verify-shard"):
        command = sub.add_parser(name)
        command.add_argument("--lock", required=True)
        command.add_argument("--shard-index", type=int, required=True)
        command.add_argument("--shard-count", type=int, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--lock", required=True)
    aggregate_parser.add_argument("--verification-shards", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        output = freeze_lock(
            arguments.config,
            arguments.r5_root,
            arguments.feature_root,
            arguments.external_input_manifest,
            arguments.class_index,
            arguments.output_root,
        )
    elif arguments.command == "run-shard":
        output = run_shard(
            arguments.lock,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    elif arguments.command == "verify-shard":
        output = verify_shard(
            arguments.lock,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    else:
        output = aggregate(
            arguments.lock, verification_shards=arguments.verification_shards
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
