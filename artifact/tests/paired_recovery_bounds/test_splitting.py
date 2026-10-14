from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marginal_alignment.campaign.feature_store import Endpoint
from marginal_alignment.campaign.split_plan import PairingTable
from marginal_alignment.paired_recovery_bounds.splitting import (
    build_recovery_partition,
)


@dataclass
class _Store:
    hashes_verified = True
    kind = "native_v2"

    def __post_init__(self):
        ids = np.arange(80, dtype=np.int64)
        self._head = np.arange(10, dtype=np.int64)
        self._reference = Endpoint(
            "reference", np.zeros((80, 3)), ids, None, "r", {}
        )
        self._target = Endpoint("target", np.zeros((80, 3)), ids, None, "t", {})
        self.manifest = {
            "metadata": {"model_provenance": {"head_fit_ids_role": "head_fit_ids"}}
        }
        self.records = {"head_fit_ids": {"sha256": "a" * 64}}

    def endpoint(self, name):
        return self._reference if name == "reference" else self._target

    def pairing(self):
        ids = np.arange(80, dtype=np.int64)
        return PairingTable(
            ids,
            ids,
            {"verified": True, "kind": "test", "source": "fixture"},
            reference_universe=ids,
            target_universe=ids,
        )

    def array(self, role):
        assert role == "head_fit_ids"
        return self._head


def test_partition_roles_are_disjoint_deterministic_and_outcome_blind() -> None:
    first = build_recovery_partition(
        _Store(),
        seed=17,
        map_fit_rows=10,
        calibration_batches=3,
        evaluation_batches=2,
        batch_size=5,
        expected_eligible_pairs=70,
    )
    second = build_recovery_partition(
        _Store(),
        seed=17,
        map_fit_rows=10,
        calibration_batches=3,
        evaluation_batches=2,
        batch_size=5,
        expected_eligible_pairs=70,
    )
    assert first.summary() == second.summary()
    assert first.unused_pair_count == 35
    roles = [first.map_fit, *first.calibration, *first.evaluation]
    seen = set()
    for batch in roles:
        assert np.array_equal(
            np.sort(batch.true_assignment), np.arange(batch.size)
        )
        for target_position, reference_position in enumerate(batch.true_assignment):
            assert (
                batch.target_ids[target_position]
                == batch.reference_ids[reference_position]
            )
        assert not seen.intersection(batch.reference_ids)
        seen.update(batch.reference_ids)
    assert not seen.intersection(range(10))
