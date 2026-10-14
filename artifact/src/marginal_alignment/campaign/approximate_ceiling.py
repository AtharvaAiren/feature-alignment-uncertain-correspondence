"""Two-world task ceilings when the observable laws are only approximately equal.

The important object is the law of the *entire observation available to the
decision rule*.  If an aligner sees ``n`` reference samples, ``n`` target
samples, a query feature, and fixed side information, then ``Q_0`` and ``Q_1``
below are the joint laws of that complete object.  Substituting the TV distance
of one endpoint draw generally gives an invalid bound because product laws can
become distinguishable as the calibration sample grows.

For a common countable support, let ``q_h(o)`` be the mass of observation
``o`` in world ``h`` and let ``ell_h(o)`` be its desired task label.  Under an
equal world prior, unrestricted Bayes task accuracy is exactly

    1 - 1/2 sum_o 1{ell_0(o) != ell_1(o)} min(q_0(o), q_1(o)).

The same number upper-bounds worst-world accuracy because the minimum of two
world accuracies cannot exceed their average.  It is an exact minimax value
only with an additional equalizer/attainability argument; this module does not
silently assume one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ceiling import bayes_orbit_ceiling, binary_gaussian_two_world_ceiling


JOINT_OBSERVATION_SEMANTICS = (
    "Q_h is the joint law of every random observation available to the rule "
    "(all calibration samples, the query when usable, and random side "
    "information), not the marginal law of one endpoint draw."
)


def _probability_mass(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite non-negative masses")
    total = float(result.sum())
    if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one, got {total:.16g}")
    # Normalize the final floating-point ulps so identities such as overlap
    # plus non-overlap remain stable for long binomial support vectors.
    return result / total


def coarse_joint_tv_ceiling(
    disagreement_world0: float,
    disagreement_world1: float,
    joint_observation_total_variation: float,
) -> float:
    """Coarse two-world upper bound from task disagreement and joint-law TV.

    Write ``D={ell_0 != ell_1}``, ``delta_h=Q_h(D)``, and
    ``epsilon=TV(Q_0,Q_1)``.  Since

    ``int_D min(dQ_0,dQ_1) >= max(delta_0,delta_1)-epsilon``, clipped at zero,

    equal-prior Bayes accuracy, and therefore worst-world accuracy, is at most

    ``1 - 1/2 (max(delta_0,delta_1)-epsilon)_+``.

    ``epsilon`` must concern the complete joint observation.  Passing a
    one-draw endpoint TV when the algorithm sees multiple draws is a semantic
    error that this function's name and argument intentionally make visible.
    """

    delta0 = float(disagreement_world0)
    delta1 = float(disagreement_world1)
    epsilon = float(joint_observation_total_variation)
    for name, value in (
        ("disagreement_world0", delta0),
        ("disagreement_world1", delta1),
        ("joint_observation_total_variation", epsilon),
    ):
        if not np.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
            raise ValueError(f"{name} must lie in [0,1]")
    delta0 = float(np.clip(delta0, 0.0, 1.0))
    delta1 = float(np.clip(delta1, 0.0, 1.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    overlap_lower_bound = max(0.0, max(delta0, delta1) - epsilon)
    return 1.0 - 0.5 * overlap_lower_bound


def discrete_two_world_ceiling(
    probability_world0: np.ndarray,
    probability_world1: np.ndarray,
    labels_world0: np.ndarray,
    labels_world1: np.ndarray,
) -> dict[str, Any]:
    """Exact equal-prior Bayes ceiling on a common finite/countable support.

    The four vectors enumerate the same joint-observation support.  Labels may
    be strings, integers, or any NumPy-comparable scalar type.  The returned
    exact value is for unrestricted task prediction.  A class such as
    ``g(T(x))`` can only do worse, and exact marginal matching can add further
    capacity constraints.
    """

    q0 = _probability_mass(probability_world0, name="probability_world0")
    q1 = _probability_mass(probability_world1, name="probability_world1")
    ell0 = np.asarray(labels_world0)
    ell1 = np.asarray(labels_world1)
    if ell0.ndim != 1 or ell1.ndim != 1:
        raise ValueError("label vectors must be one-dimensional")
    if not (q0.shape == q1.shape == ell0.shape == ell1.shape):
        raise ValueError("probability and label vectors must share one support shape")

    disagreement = ell0 != ell1
    overlap_mass = np.minimum(q0, q1)
    overlap_disagreement = float(overlap_mass[disagreement].sum())
    delta0 = float(q0[disagreement].sum())
    delta1 = float(q1[disagreement].sum())
    total_variation = float(0.5 * np.abs(q0 - q1).sum())
    exact_bayes = float(1.0 - 0.5 * overlap_disagreement)
    coarse = coarse_joint_tv_ceiling(delta0, delta1, total_variation)
    if exact_bayes > coarse + 2e-12:
        raise AssertionError("the exact overlap ceiling exceeded its coarse TV relaxation")

    return {
        "schema_version": "marginal_alignment.campaign.two_world_ceiling.v1",
        "support_size": int(q0.size),
        "disagreement_world0": delta0,
        "disagreement_world1": delta1,
        "joint_observation_total_variation": total_variation,
        "joint_observation_overlap": float(overlap_mass.sum()),
        "overlap_disagreement": overlap_disagreement,
        "exact_equal_prior_bayes_accuracy": exact_bayes,
        "worst_world_accuracy_upper_bound": exact_bayes,
        "coarse_joint_tv_accuracy_upper_bound": coarse,
        "joint_observation_semantics": JOINT_OBSERVATION_SEMANTICS,
        "attainability_scope": (
            "Exact for unrestricted equal-prior Bayes task prediction. It is "
            "not automatically an exact minimax value for a restricted "
            "alignment family."
        ),
    }


def gaussian_halfspace_indistinguishable_ceiling(correlation: float) -> float:
    """Compatibility wrapper for the V1 identical-Gaussian closed form."""

    return binary_gaussian_two_world_ceiling(correlation)


def orbit_indistinguishable_ceiling(
    label_matrix: np.ndarray,
    prior: np.ndarray | None = None,
) -> float:
    """Compatibility wrapper for a finite exactly indistinguishable orbit."""

    return bayes_orbit_ceiling(label_matrix, prior=prior)
