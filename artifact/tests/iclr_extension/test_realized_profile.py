from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from marginal_alignment.campaign.artifacts import IntegrityError
from marginal_alignment.campaign.maps import AffineMap
from marginal_alignment.iclr_extension.frontier import FrozenMarginalFrontier
from marginal_alignment.iclr_extension.realized_profile import (
    REALIZED_PROFILE_METHODS,
    FrozenRealizedProfilePredictions,
    aggregate_profile_gain_clusters,
    cluster_aware_profile_gain_association,
    freeze_realized_profile_predictions,
    profile_gain_record,
    score_frozen_realized_profile,
)


def _sha(character: str) -> str:
    return character * 64


def _frontier(*, labels_used: bool = False) -> FrozenMarginalFrontier:
    identity = np.eye(1, dtype=np.float64)
    zero = np.zeros(1, dtype=np.float64)
    shift = np.ones(1, dtype=np.float64)
    maps = {
        "none": AffineMap(identity, zero, "none"),
        "mean": AffineMap(identity, shift, "mean"),
        "diagonal": AffineMap(identity, shift, "diagonal"),
        "coordinate_quantile_257": AffineMap(identity, zero, "quantile"),
        "coral_oas": AffineMap(identity, zero, "coral"),
        "bures_oas": AffineMap(identity, shift, "bures"),
    }
    return FrozenMarginalFrontier(
        maps=maps,
        diagnostics={
            "information_scope": "unpaired_calibration_features_only",
            "labels_used": labels_used,
            "pairing_used": False,
        },
    )


def _frozen() -> FrozenRealizedProfilePredictions:
    # None predicts [0, 0, 0, 1], while the shifted methods predict
    # [0, 0, 1, 1] under this two-class readout.
    return freeze_realized_profile_predictions(
        _frontier(),
        np.asarray([[-3.0], [-2.0], [0.0], [1.0]]),
        np.asarray([[-1.0], [1.0]]),
        np.asarray([0.5, -0.5]),
        dataset="imagenet_v2",
        encoder="clip",
        split_seed=201,
        target_endpoint_rows=6,
        target_witness_original_rows=np.asarray([3, 1, 4, 0]),
        profile_binding_sha256=_sha("a"),
        expected_witness_rows=4,
    )


def test_prediction_freeze_api_cannot_accept_labels_or_wnids() -> None:
    names = set(signature(freeze_realized_profile_predictions).parameters)
    assert not any("label" in name or "wnid" in name for name in names)
    frozen = _frozen()
    assert frozen.phase == "fixed_method_predictions_frozen_before_target_wnids"
    assert tuple(frozen.predictions) == REALIZED_PROFILE_METHODS
    assert frozen.binding()["target_wnids_loaded"] is False
    with pytest.raises(ValueError):
        frozen.predictions["none"][0] = 1


def test_freeze_rejects_a_frontier_that_used_labels() -> None:
    with pytest.raises(IntegrityError, match="label-free unpaired"):
        freeze_realized_profile_predictions(
            _frontier(labels_used=True),
            np.zeros((2, 1)),
            np.asarray([[-1.0], [1.0]]),
            np.zeros(2),
            dataset="imagenet_r",
            encoder="clip",
            split_seed=201,
            target_endpoint_rows=2,
            target_witness_original_rows=np.asarray([0, 1]),
            profile_binding_sha256=_sha("b"),
        )


def test_freeze_checks_profile_bures_hash_binding() -> None:
    with pytest.raises(IntegrityError, match="Bures map differs"):
        freeze_realized_profile_predictions(
            _frontier(),
            np.zeros((2, 1)),
            np.asarray([[-1.0], [1.0]]),
            np.zeros(2),
            dataset="imagenet_r",
            encoder="clip",
            split_seed=201,
            target_endpoint_rows=2,
            target_witness_original_rows=np.asarray([0, 1]),
            profile_binding_sha256=_sha("b"),
            expected_bures_matrix_sha256=_sha("c"),
            expected_bures_offset_sha256=_sha("d"),
        )


def test_posthoc_scoring_uses_bound_original_rows_and_fixed_full_head() -> None:
    frozen = _frozen()
    class_index = ["n00000001", "n00000002"]
    endpoint_wnids = np.asarray(
        [
            "n00000002",  # selected fourth
            "n00000001",  # selected second
            "n00000002",  # not selected
            "n00000001",  # selected first
            "n00000002",  # selected third
            "n00000001",  # not selected
        ]
    )
    score = score_frozen_realized_profile(
        frozen, endpoint_wnids, class_index, expected_classes=2
    )
    assert score["per_method_target_accuracy"] == {
        "none": 0.75,
        "mean": 1.0,
        "diagonal": 1.0,
        "bures_oas": 1.0,
    }
    assert score["bures_minus_none_target_accuracy"] == 0.25
    assert score["target_supported_class_count"] == 2
    assert score["full_frozen_head_retained"] is True
    assert score["target_wnids_used_for_map_fitting"] is False
    assert sum(row["rows"] for row in score["per_wnid_sufficient_statistics"]) == 4


def test_posthoc_scoring_requires_full_endpoint_and_full_class_index() -> None:
    frozen = _frozen()
    with pytest.raises(IntegrityError, match="cover the frozen endpoint"):
        score_frozen_realized_profile(
            frozen,
            np.asarray(["n00000001"] * 4),
            ["n00000001", "n00000002"],
            expected_classes=2,
        )
    with pytest.raises(IntegrityError, match="frozen head output space"):
        score_frozen_realized_profile(
            frozen,
            np.asarray(["n00000001"] * 6),
            ["n00000001", "n00000002", "n00000003"],
            expected_classes=3,
        )
    with pytest.raises(IntegrityError, match="malformed WNID"):
        score_frozen_realized_profile(
            frozen,
            np.asarray(["n00000001suffix"] * 6),
            ["n00000001", "n00000002"],
            expected_classes=2,
        )


def _association_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    clusters = [
        ("imagenet_c:brightness", "imagenet_c", 0.10, 0.01),
        ("imagenet_c:fog", "imagenet_c", 0.30, 0.03),
        ("natural:imagenet_v2", "natural", 0.60, 0.06),
        ("natural:imagenet_r", "natural", 0.90, 0.09),
    ]
    for cluster, family, disagreement, gain in clusters:
        for encoder_offset, encoder in enumerate(("clip", "dinov3")):
            for seed_offset, seed in enumerate((201, 202, 203)):
                records.append(
                    {
                        "cluster": cluster,
                        "family": family,
                        "dataset": cluster.split(":", 1)[-1],
                        "encoder": encoder,
                        "split_seed": seed,
                        "profile_disagreement": disagreement
                        + 0.001 * encoder_offset
                        + 0.0001 * seed_offset,
                        "profile_ceiling": 1.0 - disagreement / 2.0,
                        "bures_gain": gain
                        + 0.001 * encoder_offset
                        + 0.0001 * seed_offset,
                        "profile_binding_sha256": _sha("e"),
                        "profile_frozen_before_scoring": True,
                        "predictions_frozen_before_target_wnids": True,
                        "labels_used_for_map_fitting": False,
                        "posthoc_method_selection": False,
                    }
                )
    return records


def test_cluster_aggregation_nests_seeds_and_encoders() -> None:
    clusters = aggregate_profile_gain_clusters(
        _association_records(),
        expected_encoders=("clip", "dinov3"),
        expected_split_seeds=(201, 202, 203),
    )
    assert len(clusters) == 4
    assert all(len(row["encoder_cells"]) == 2 for row in clusters)
    assert all(row["split_seeds_nested"] == [201, 202, 203] for row in clusters)


def test_cluster_association_is_cluster_level_and_deterministic() -> None:
    kwargs = {
        "expected_encoders": ("clip", "dinov3"),
        "expected_split_seeds": (201, 202, 203),
        "bootstrap_replicates": 100,
        "permutation_replicates": 100,
        "rng_seed": 17,
    }
    first = cluster_aware_profile_gain_association(_association_records(), **kwargs)
    second = cluster_aware_profile_gain_association(_association_records(), **kwargs)
    assert first == second
    assert first["cluster_count"] == 4
    assert first["spearman_profile_disagreement_vs_bures_gain"] == pytest.approx(1.0)
    assert first["spearman_profile_ceiling_vs_bures_gain"] == pytest.approx(-1.0)
    assert first["encoders_and_split_seeds_nested_within_cluster"] is True
    assert first["cluster_bootstrap"]["attempted_replicates"] == 100
    assert first["cluster_permutation"]["within_family"] is True


def test_cluster_panel_rejects_duplicates_and_missing_repeats() -> None:
    records = _association_records()
    with pytest.raises(IntegrityError, match="duplicate"):
        aggregate_profile_gain_clusters([*records, records[0]])
    with pytest.raises(IntegrityError, match="not complete"):
        aggregate_profile_gain_clusters(
            records[:-1],
            expected_encoders=("clip", "dinov3"),
            expected_split_seeds=(201, 202, 203),
        )
    extra_encoder = dict(records[0], encoder="unlocked_encoder")
    with pytest.raises(IntegrityError, match="observed encoders differ"):
        aggregate_profile_gain_clusters(
            [*records, extra_encoder],
            expected_encoders=("clip", "dinov3"),
            expected_split_seeds=(201, 202, 203),
        )


def test_profile_gain_record_rejects_a_postfit_profile_binding() -> None:
    frozen = _frozen()
    score = score_frozen_realized_profile(
        frozen,
        np.asarray(
            [
                "n00000002",
                "n00000001",
                "n00000002",
                "n00000001",
                "n00000002",
                "n00000001",
            ]
        ),
        ["n00000001", "n00000002"],
        expected_classes=2,
    )
    record = profile_gain_record(
        score,
        cluster="natural:imagenet_v2",
        family="natural",
        profile_disagreement=0.5,
        profile_binding_sha256=_sha("a"),
    )
    assert record["profile_ceiling"] == 0.75
    with pytest.raises(IntegrityError, match="binding differ"):
        profile_gain_record(
            score,
            cluster="natural:imagenet_v2",
            family="natural",
            profile_disagreement=0.5,
            profile_binding_sha256=_sha("f"),
        )
