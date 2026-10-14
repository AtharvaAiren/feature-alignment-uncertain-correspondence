"""Decision-theoretic ceilings under marginally indistinguishable drifts.

The naive quantity ``sup_T`` over maps satisfying a push-forward constraint is
an oracle quantity, not a marginal-only ceiling: it permits choosing the true
inverse correspondence.  Here a marginal-only decision rule must make the same
choice in every latent world that produces the same observed marginals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


LabelFunction = Callable[[np.ndarray], np.ndarray]


def task_disagreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("label vectors must have the same one-dimensional shape")
    return float(np.mean(a != b))


def two_world_ceiling(disagreement: float) -> float:
    """Sharp Bayes/minimax ceiling for two equally likely indistinguishable worlds.

    The ceiling applies to arbitrary task predictions, hence also upper-bounds
    any subclass induced by ``g(T(x))``.  In the map-constrained problem it is
    attained by mixing the two inverse maps only when those inverses are
    admissible and the recovered reference decision is the task of interest.
    """

    delta = float(disagreement)
    if not 0.0 <= delta <= 1.0:
        raise ValueError("disagreement must lie in [0,1]")
    return 1.0 - 0.5 * delta


def binary_gaussian_two_world_ceiling(correlation: float) -> float:
    """Closed form for an isotropic Gaussian and a frozen binary halfspace.

    If ``rho = <w,Hw>/||w||^2``, the two halfspaces disagree with probability
    ``arccos(rho)/pi``.  The marginal-only two-world ceiling is therefore
    ``1-arccos(rho)/(2*pi)``.
    """

    rho = float(correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must lie in [-1,1]")
    return float(1.0 - np.arccos(np.clip(rho, -1.0, 1.0)) / (2.0 * np.pi))


def bayes_orbit_ceiling(label_matrix: np.ndarray, prior: np.ndarray | None = None) -> float:
    """Exact task-prediction ceiling for a finite indistinguishable-world prior.

    ``label_matrix[h,i]`` is the true task label of observed sample ``i`` in
    latent world ``h``.  The unlabeled marginal law is identical across rows,
    so the Bayes-optimal marginal-only prediction chooses the posterior modal
    label independently at each observed point.

    This is an upper bound for every alignment of the form ``g o T`` because it
    permits arbitrary task predictions.  It is exact for the unrestricted
    prediction problem and requires an explicit ambiguity set and prior.
    """

    labels = np.asarray(label_matrix)
    if labels.ndim != 2 or min(labels.shape) < 1:
        raise ValueError("label_matrix must have non-empty shape [worlds,samples]")
    worlds, samples = labels.shape
    if prior is None:
        probabilities = np.full(worlds, 1.0 / worlds)
    else:
        probabilities = np.asarray(prior, dtype=np.float64)
        if probabilities.shape != (worlds,) or np.any(probabilities < 0.0):
            raise ValueError("prior must be non-negative with one value per world")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("prior must have positive mass")
        probabilities = probabilities / total
    posterior_modes = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        _, inverse = np.unique(labels[:, index], return_inverse=True)
        mass = np.bincount(inverse, weights=probabilities)
        posterior_modes[index] = mass.max()
    return float(posterior_modes.mean())


def candidate_mixture_world_accuracies(
    label_matrix: np.ndarray,
    mixture: np.ndarray | None = None,
) -> np.ndarray:
    """Expected accuracy in every world for a mixture of candidate inverses.

    ``label_matrix[h, i]`` is the desired label of observed point ``i`` in
    hidden world ``h``.  Candidate inverse ``j`` predicts row ``j``.  Therefore
    the pairwise agreement matrix, multiplied by ``mixture``, is the vector of
    world-specific expected accuracies.  In a finite group experiment, the
    uniform mixture is the discrete Haar equalizer used by the minimax proof.

    This diagnostic is intentionally different from :func:`bayes_orbit_ceiling`:
    the latter is a prior-averaged posterior-mode value, while this function
    verifies that a claimed randomized minimax construction attains the value
    in *each fixed world*.
    """

    labels = np.asarray(label_matrix)
    if labels.ndim != 2 or min(labels.shape) < 1:
        raise ValueError("label_matrix must have non-empty shape [worlds,samples]")
    worlds = labels.shape[0]
    if mixture is None:
        probabilities = np.full(worlds, 1.0 / worlds)
    else:
        probabilities = np.asarray(mixture, dtype=np.float64)
        if probabilities.shape != (worlds,) or np.any(probabilities < 0.0):
            raise ValueError("mixture must be non-negative with one value per candidate")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("mixture must have positive mass")
        probabilities = probabilities / total
    pairwise = np.mean(labels[:, None, :] == labels[None, :, :], axis=2)
    return np.asarray(pairwise @ probabilities, dtype=np.float64)


def empirical_group_ceiling(
    features: np.ndarray,
    group: Sequence[np.ndarray],
    readout: LabelFunction,
    prior: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Estimate the finite-group orbit ceiling and diagnostic disagreement."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape [n,d]")
    matrices = [np.asarray(item, dtype=np.float64) for item in group]
    if not matrices:
        raise ValueError("group must be non-empty")
    d = x.shape[1]
    for matrix in matrices:
        if matrix.shape != (d, d):
            raise ValueError("every group action must have shape [d,d]")
    # A world h observes x=h z.  Given an observed x, its latent reference is
    # h^{-1}x, which is x @ h for orthogonal row-vector actions x=z@h.T.
    label_matrix = np.stack([np.asarray(readout(x @ matrix)) for matrix in matrices])
    ceiling = bayes_orbit_ceiling(label_matrix, prior)
    baseline = label_matrix[0]
    mean_disagreement = float(
        np.mean([task_disagreement(baseline, row) for row in label_matrix])
    )
    return {
        "worlds": len(matrices),
        "samples": x.shape[0],
        "bayes_orbit_ceiling": ceiling,
        "mean_identity_disagreement": mean_disagreement,
    }


def cyclic_permutation_group(dimension: int, orbit_size: int) -> list[np.ndarray]:
    """Orthogonal cyclic actions on the first ``orbit_size`` coordinates."""

    if not 2 <= orbit_size <= dimension:
        raise ValueError("orbit_size must lie in [2,dimension]")
    result: list[np.ndarray] = []
    for shift in range(orbit_size):
        matrix = np.eye(dimension)
        for index in range(orbit_size):
            matrix[index, index] = 0.0
            matrix[index, (index + shift) % orbit_size] = 1.0
        result.append(matrix)
    return result
