from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_alignment.experiment import run_config, summarize
from marginal_alignment.schema import load_cache, write_cache


class SchemaRunnerTests(unittest.TestCase):
    def test_hidden_world_summary_separates_bayes_mean_and_worst_world(self) -> None:
        rows = []
        metric_names = [
            "aligned_accuracy", "reference_accuracy", "source_accuracy",
            "accuracy_recovery_fraction", "reference_prediction_agreement",
            "aligned_mean_error", "aligned_covariance_relative_error",
            "aligned_gaussian_w2_squared", "paired_feature_mse", "centered_logit_mse",
        ]
        for data_seed in (0, 1):
            for world, value in enumerate((1.0, 0.0, 0.0, 0.0)):
                row = {
                    "dataset": "synthetic_cyclic_k4", "encoder": "identity",
                    "method": "none", "calibration_size": 128,
                    "data_seed": data_seed, "split_seed": 7,
                    "hidden_world": world, "worlds": 4,
                    "theoretical_bayes_ceiling": 0.25,
                }
                row.update({metric: value for metric in metric_names})
                # External-label accuracy and frozen-task agreement are
                # deliberately different so the theorem-facing aggregation
                # cannot silently use the former.
                row["reference_prediction_agreement"] = (0.8, 0.6, 0.4, 0.2)[world]
                rows.append(row)
        group = summarize(rows)["groups"][0]
        self.assertAlmostEqual(group["ambiguity_uniform_accuracy_mean"], 0.25)
        self.assertAlmostEqual(group["ambiguity_worst_accuracy_mean"], 0.0)
        self.assertAlmostEqual(group["ambiguity_uniform_accuracy_std"], 0.0)
        self.assertAlmostEqual(group["ambiguity_uniform_task_agreement_mean"], 0.5)
        self.assertAlmostEqual(group["ambiguity_worst_task_agreement_mean"], 0.2)
        self.assertAlmostEqual(group["ambiguity_uniform_task_agreement_std"], 0.0)

    def test_cache_and_runner_are_disjoint_and_reproducible(self) -> None:
        rng = np.random.default_rng(3)
        reference = rng.normal(size=(600, 6))
        action = np.eye(6)
        action[:2, :2] = [[0, -1], [1, 0]]
        target = reference @ action.T
        weight = np.zeros((2, 6))
        weight[:, :2] = np.eye(2)
        labels = np.argmax(reference[:, :2], axis=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = write_cache(
                root / "cache.npz", reference_features=reference,
                target_features=target, labels=labels, head_weight=weight,
                sample_ids=np.arange(len(reference)),
                metadata={"dataset": "unit", "encoder": "identity"},
            )
            cache = load_cache(cache_path)
            self.assertEqual(cache.reference.shape, (600, 6))
            config = {
                "cache_paths": [str(cache_path)],
                "methods": ["none", "diagonal", "bures", "paired_oracle", "shuffled_oracle"],
                "split_seeds": [0, 1],
                "calibration_sizes": [100],
                "evaluation_size": 300,
                "shrinkage": 1e-4,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_config(config_path, root / "out")
            self.assertEqual(result["manifest"]["record_count"], 10)
            rows = [json.loads(line) for line in (root / "out/rows.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["calibration_id_hash"] != row["evaluation_id_hash"] for row in rows))
            oracle = [row for row in rows if row["method"] == "paired_oracle"]
            unpaired = [row for row in rows if row["method"] == "bures"]
            self.assertGreater(
                np.mean([row["aligned_accuracy"] for row in oracle]),
                np.mean([row["aligned_accuracy"] for row in unpaired]),
            )

    def test_duplicate_ids_fail_closed(self) -> None:
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(
                path,
                reference_features=rng.normal(size=(10, 3)),
                target_features=rng.normal(size=(10, 3)),
                labels=np.zeros(10, dtype=np.int64),
                head_weight=np.ones((2, 3)),
                sample_ids=np.zeros(10, dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cache(path)


if __name__ == "__main__":
    unittest.main()
