import numpy as np
import pytest

from marginal_alignment.campaign.inference import (
    holm_adjust, infer_contrast, macro_summary, paired_effect_rows,
)


def _rows():
    rows = []
    for corruption, effect in (("a", 0.1), ("b", 0.2), ("c", 0.3)):
        for seed in (0, 1):
            base = dict(suite="main", dataset="toy", encoder="e", corruption=corruption,
                        severity=1, split_seed=seed, calibration_size=16,
                        pair_budget=None, pair_noise_rate=None)
            rows.append({**base, "method": "left", "accuracy": 0.5 + effect})
            rows.append({**base, "method": "right", "accuracy": 0.5})
    return rows


def test_paired_cluster_effect_and_bootstrap():
    effects = paired_effect_rows(_rows(), left="left", right="right", metric="accuracy")
    result = infer_contrast(effects, bootstrap_repetitions=2000, seed=1)
    assert result["effect"] == pytest.approx(0.2)
    assert result["clusters"] == 3
    assert result["ci_lower"] <= result["effect"] <= result["ci_upper"]


def test_duplicate_cells_fail_closed():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        paired_effect_rows(rows, left="left", right="right", metric="accuracy")


def test_holm_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.04, 0.001, 0.02])
    assert adjusted[1] == pytest.approx(0.003)
    assert adjusted[2] >= 0.02
    assert all(0 <= value <= 1 for value in adjusted)


def test_macro_sd_uses_split_macros_not_corruption_pool():
    rows = _rows()
    summary = macro_summary(rows, ["accuracy"])
    left = next(row for row in summary if row["method"] == "left")
    assert left["accuracy_mean"] == pytest.approx(0.7)
    assert left["accuracy_std"] == pytest.approx(0.0)
    assert "split-level" in left["variation_unit"]
