import json

import pytest

from marginal_alignment.campaign.config import ConfigurationError, config_sha256, load_config


def _minimal():
    return {
        "schema_version": "marginal_alignment.campaign.campaign.v2",
        "campaign_id": "test",
        "datasets": [{
            "name": "toy", "reference_key": "clean", "target_key": "corrupt",
            "encoders": ["tiny"], "corruptions": ["noise"], "severities": [1],
            "evaluation_size": 128, "head_protocol": {"kind": "fixture"},
        }],
        "methods": {"none": {}, "mean": {}},
        "suites": {"main": {"datasets": ["toy"], "methods": ["none", "mean"],
                              "calibration_sizes": [16, 32], "split_seeds": [1, 2]}},
        "inference": {"primary_contrasts": [
            {"left": "mean", "right": "none", "metric": "accuracy"}
        ]},
        "synthetic": {
            "exact_indistinguishable": {}, "approximate_two_world": {},
            "identifiable_nongaussian": {}, "pair_rank_phase": {},
            "real_embedding_orbit": {},
        },
        "streaming_secondary": {"enabled": False},
    }


def test_config_is_canonical_and_expands_environment(tmp_path, monkeypatch):
    config = _minimal()
    config["cache_roots"] = ["$CAMPAIGN_TEST_ROOT/cache"]
    monkeypatch.setenv("CAMPAIGN_TEST_ROOT", str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded, source, digest = load_config(path)
    assert source == path.resolve()
    assert loaded["cache_roots"] == [str(tmp_path / "cache")]
    assert digest == config_sha256(loaded)


def test_config_rejects_unknown_method(tmp_path):
    config = _minimal()
    config["suites"]["main"]["methods"].append("not_registered")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown methods"):
        load_config(path)
