from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from marginal_alignment.campaign.artifacts import IntegrityError
from marginal_alignment.campaign.split_plan import PairingTable, build_split_plan, validate_split_plan


class SplitPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.arange(240, dtype=np.int64)
        self.target = np.arange(1000, 1240, dtype=np.int64)
        self.pairing = PairingTable(
            reference_ids=self.reference,
            target_ids=self.target,
            provenance={
                "verified": True,
                "kind": "natural_clean_corruption_identity",
                "source": "unit construction",
            },
            reference_universe=self.reference,
            target_universe=self.target,
        )

    def _plan(self, output: Path | None = None):
        return build_split_plan(
            reference_ids=self.reference,
            target_ids=self.target,
            pairing=self.pairing,
            seed=20260812,
            evaluation_size=40,
            calibration_sizes=[16, 32, 64],
            pair_budgets=[4, 8, 16, 32],
            noisy_pair_fractions=[0.25, 0.5],
            reference_namespace="clean",
            target_namespace="corrupted",
            output_path=output,
        )

    def test_fixed_evaluation_independent_endpoints_and_nested_budgets(self) -> None:
        plan = self._plan()
        evaluation_hash = plan["evaluation_sha256"]
        for row in plan["calibrations"].values():
            self.assertEqual(row["evaluation_sha256"], evaluation_hash)
        c16, c32, c64 = (plan["calibrations"][str(size)] for size in (16, 32, 64))
        self.assertEqual(c32["reference_ids"][:16], c16["reference_ids"])
        self.assertEqual(c64["target_ids"][:32], c32["target_ids"])
        # Independent endpoint RNG streams are not a disguised matched-ID draw.
        self.assertNotEqual(c64["reference_ids"], [value - 1000 for value in c64["target_ids"]])
        eval_reference = set(plan["evaluation"]["reference_ids"])
        eval_target = set(plan["evaluation"]["target_ids"])
        self.assertFalse(eval_reference.intersection(c64["reference_ids"]))
        self.assertFalse(eval_target.intersection(c64["target_ids"]))

        true16 = plan["pair_controls"]["16"]["controls"]["true"]
        true32 = plan["pair_controls"]["32"]["controls"]["true"]
        self.assertEqual(true32["reference_ids"][:16], true16["reference_ids"])
        self.assertEqual(true32["target_ids"][:16], true16["target_ids"])

    def test_shuffled_and_noisy_controls_are_estimator_matched(self) -> None:
        plan = self._plan()
        controls = plan["pair_controls"]["16"]["controls"]
        true = controls["true"]
        shuffled = controls["shuffled"]
        self.assertEqual(shuffled["reference_ids"], true["reference_ids"])
        self.assertEqual(set(shuffled["target_ids"]), set(true["target_ids"]))
        self.assertTrue(all(
            shuffled_value != true_value
            for shuffled_value, true_value in zip(shuffled["target_ids"], true["target_ids"])
        ))
        noisy25 = controls["noisy"]["0.25"]
        self.assertEqual(noisy25["mismatched_pairs"], 4)
        self.assertEqual(set(noisy25["target_ids"]), set(true["target_ids"]))
        self.assertEqual(noisy25["actual_noise_fraction"], 0.25)

    def test_manifest_is_byte_deterministic_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            plan_a = self._plan(first)
            plan_b = self._plan(second)
            self.assertEqual(plan_a, plan_b)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            tampered = dict(plan_a)
            tampered["seed"] = 7
            with self.assertRaisesRegex(IntegrityError, "checksum mismatch"):
                validate_split_plan(tampered)

    def test_head_fit_identities_are_never_selected(self) -> None:
        excluded_reference = self.reference[:12]
        excluded_target = self.target[:12]
        plan = build_split_plan(
            reference_ids=self.reference,
            target_ids=self.target,
            pairing=self.pairing,
            seed=3,
            evaluation_size=40,
            calibration_sizes=[16, 64],
            pair_budgets=[16, 64],
            excluded_reference_ids=excluded_reference,
            excluded_target_ids=excluded_target,
            head_fit_exclusion_required=True,
            head_fit_ids_count=12,
        )
        selected_reference = set(plan["evaluation"]["reference_ids"])
        selected_target = set(plan["evaluation"]["target_ids"])
        selected_reference.update(plan["calibrations"]["64"]["reference_ids"])
        selected_target.update(plan["calibrations"]["64"]["target_ids"])
        selected_reference.update(
            plan["pair_controls"]["64"]["controls"]["true"]["reference_ids"]
        )
        selected_target.update(
            plan["pair_controls"]["64"]["controls"]["true"]["target_ids"]
        )
        self.assertFalse(selected_reference.intersection(excluded_reference))
        self.assertFalse(selected_target.intersection(excluded_target))
        self.assertTrue(
            plan["policy"]["head_fit_identities_excluded_from_fit_and_evaluation"]
        )

    def test_invalid_pair_provenance_and_outside_ids_fail_closed(self) -> None:
        unverified = PairingTable(
            self.reference,
            self.target,
            provenance={"verified": False, "kind": "natural", "source": "fixture"},
        )
        with self.assertRaisesRegex(IntegrityError, "not verified"):
            unverified.validated()
        outside = PairingTable(
            reference_ids=[0, 999],
            target_ids=[1000, 1001],
            provenance={"verified": True, "kind": "natural", "source": "fixture"},
            reference_universe=self.reference,
            target_universe=self.target,
        )
        with self.assertRaisesRegex(IntegrityError, "outside"):
            outside.validated()


if __name__ == "__main__":
    unittest.main()
