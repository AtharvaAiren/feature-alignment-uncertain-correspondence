import json
import tempfile
import unittest
from pathlib import Path

from marginal_alignment.statistics import analyze_real_rows


class ClusteredStatisticsTests(unittest.TestCase):
    def test_pairs_cells_and_normalizes_metric_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = Path(directory) / "rows.jsonl"
            payload = []
            methods = [
                "none", "diagonal", "coral", "bures", "paired_procrustes",
                "paired_oracle", "paired_readout_oracle", "shuffled_pairs", "er",
            ]
            quality = {
                "none": 0.0, "diagonal": 1.0, "coral": 2.0, "bures": 3.0,
                "paired_procrustes": 5.0, "paired_oracle": 6.0,
                "paired_readout_oracle": 7.0, "shuffled_pairs": 1.0, "er": 4.0,
            }
            for corruption_index in range(3):
                for split_seed in range(2):
                    for method in methods:
                        method_quality = quality[method]
                        payload.append({
                            "encoder": "test_encoder",
                            "corruption": f"c{corruption_index}",
                            "severity": 5,
                            "split_seed": split_seed,
                            "calibration_size": 1024,
                            "method": method,
                            "aligned_accuracy": 0.4 + 0.01 * method_quality,
                            "reference_prediction_agreement": 0.5 + 0.01 * method_quality,
                            # Lower is better, so later methods improve here too.
                            "centered_logit_mse": 1.0 - 0.01 * method_quality,
                            "aligned_covariance_relative_error": 0.8 - 0.01 * method_quality,
                        })
            rows.write_text("".join(json.dumps(row) + "\n" for row in payload))
            output = Path(directory) / "inference.json"
            result = analyze_real_rows(
                rows, output, calibration_size=1024, bootstrap_replicates=100, seed=7
            )
            self.assertTrue(output.is_file())
            self.assertEqual(len(result["records"]), 28)
            self.assertTrue(all(
                row["effect_positive_means_candidate_better"] > 0
                for row in result["records"]
            ))
            self.assertTrue(all("holm_adjusted_p" in row for row in result["records"]))


if __name__ == "__main__":
    unittest.main()
