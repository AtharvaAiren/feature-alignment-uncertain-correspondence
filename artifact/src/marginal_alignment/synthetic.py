"""Synthetic generators with an exact group-ambiguity task ceiling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ceiling import cyclic_permutation_group
from .schema import write_cache


def generate_cyclic_suite(
    output_dir: str | Path,
    *,
    class_counts: tuple[int, ...] = (2, 4, 8),
    dimension: int = 16,
    samples: int = 12000,
    seeds: tuple[int, ...] = tuple(range(20)),
    anisotropic: bool = True,
) -> list[Path]:
    """Generate every hidden cyclic world; all worlds have identical marginals."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for classes in class_counts:
        if classes > dimension:
            raise ValueError("class count cannot exceed feature dimension")
        group = cyclic_permutation_group(dimension, classes)
        for seed in seeds:
            rng = np.random.default_rng(seed + classes * 1000)
            scales = np.ones(dimension)
            if anisotropic and dimension > classes:
                scales[classes:] = np.linspace(0.4, 2.0, dimension - classes)
            if samples % classes:
                raise ValueError("samples must be divisible by every class count for orbit completion")
            # Complete every finite group orbit.  The resulting empirical
            # multiset, not only its population law, is exactly invariant to
            # every hidden world.  This prevents finite-sample covariance noise
            # from leaking the latent action to an unpaired estimator.
            base = rng.normal(size=(samples // classes, dimension)) * scales
            reference = np.concatenate([base @ action.T for action in group], axis=0)
            # The frozen K-way readout and ground-truth task are defined in the
            # reference coordinate system.
            weight = np.zeros((classes, dimension), dtype=np.float64)
            weight[:, :classes] = np.eye(classes)
            labels = np.argmax(reference[:, :classes], axis=1)
            for world, action in enumerate(group):
                target = reference @ action.T
                path = root / f"cyclic_k{classes}_seed{seed}_world{world}.npz"
                write_cache(
                    path,
                    reference_features=reference,
                    target_features=target,
                    labels=labels,
                    head_weight=weight,
                    sample_ids=np.arange(samples),
                    metadata={
                        "dataset": f"synthetic_cyclic_k{classes}",
                        "encoder": "identity",
                        "classes": classes,
                        "seed": seed,
                        "hidden_world": world,
                        "worlds": classes,
                        "theoretical_bayes_ceiling": 1.0 / classes,
                        "marginals_identical_by_construction": True,
                        "empirical_multiset_orbit_complete": True,
                        "first_k_eigenvalues_repeated": True,
                    },
                )
                written.append(path)
    return written
