"""Capability-enforcing registry for V2 alignment methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .maps import (
    FittedMap,
    fit_coordinate_quantile,
    fit_covariance_gauge,
    fit_diagonal,
    fit_identity,
    fit_mean,
    fit_paired_procrustes,
    fit_paired_readout_gauge,
    fit_paired_ridge,
    fit_sliced_quantile,
    fit_third_moment_ica,
)


@dataclass(frozen=True)
class MethodCapabilities:
    name: str
    information_tier: str
    requires_pairing: bool = False
    requires_readout: bool = False
    uses_labels: bool = False
    changes_consumer: bool = False
    supports_out_of_sample: bool = True
    changes_joint_dependence: bool = False
    pair_assignment_materialized: bool = False


_CAPABILITIES = {
    "none": MethodCapabilities("none", "none"),
    "mean": MethodCapabilities("mean", "coordinate_mean"),
    "diagonal": MethodCapabilities("diagonal", "coordinate_second_moment"),
    "coordinate_quantile": MethodCapabilities(
        "coordinate_quantile", "all_coordinate_marginals"
    ),
    "coral": MethodCapabilities("coral", "joint_second_moment", changes_joint_dependence=True),
    "bures": MethodCapabilities("bures", "joint_second_moment", changes_joint_dependence=True),
    "sliced_quantile": MethodCapabilities(
        "sliced_quantile", "finite_projected_joint_marginals", changes_joint_dependence=True
    ),
    "third_moment_ica": MethodCapabilities(
        "third_moment_ica", "higher_order_joint_marginals", changes_joint_dependence=True
    ),
    "paired_procrustes": MethodCapabilities(
        "paired_procrustes", "sample_correspondence", requires_pairing=True,
        changes_joint_dependence=True,
    ),
    "paired_ridge": MethodCapabilities(
        "paired_ridge", "sample_correspondence", requires_pairing=True,
        changes_joint_dependence=True,
    ),
    "paired_reduced_rank": MethodCapabilities(
        "paired_reduced_rank", "sample_correspondence", requires_pairing=True,
        changes_joint_dependence=True,
    ),
    "paired_readout_gauge": MethodCapabilities(
        "paired_readout_gauge", "sample_correspondence_plus_readout",
        requires_pairing=True, requires_readout=True, changes_joint_dependence=True,
    ),
}


_ALIASES = {
    "identity": "none", "bn": "diagonal", "bn_style": "diagonal",
    "quantile": "coordinate_quantile", "coordinate_quantiles": "coordinate_quantile",
    "sliced": "sliced_quantile", "sliced_quantile_flow": "sliced_quantile",
    "ica": "third_moment_ica", "fobi_ica": "third_moment_ica",
    "procrustes": "paired_procrustes", "paired_rrr": "paired_reduced_rank",
    "readout_gauge": "paired_readout_gauge",
}


def _base_and_control(name: str) -> tuple[str, str | None]:
    normalized = name.lower().replace("-", "_")
    control = None
    for suffix, value in (("_shuffled", "shuffled"), ("_noisy", "noisy")):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            control = value
            break
    # Configs may use ``base_shuffled`` (preferred for readable paired
    # estimator names) or ``shuffled_base``.  Both resolve to the same exact
    # estimator with only the assignment map changed.
    for prefix, value in (("shuffled_", "shuffled"), ("noisy_", "noisy")):
        if normalized.startswith(prefix):
            if control is not None:
                raise ValueError("method requests two pair-control modes")
            normalized = normalized[len(prefix):]
            control = value
            break
    normalized = _ALIASES.get(normalized, normalized)
    return normalized, control


def method_capabilities(name: str) -> MethodCapabilities:
    base, control = _base_and_control(name)
    if base not in _CAPABILITIES:
        raise KeyError(f"unknown campaign method: {name}")
    capabilities = _CAPABILITIES[base]
    if control is not None and not capabilities.requires_pairing:
        raise ValueError("shuffled/noisy controls are valid only for paired estimators")
    return capabilities


def registered_methods() -> dict[str, MethodCapabilities]:
    return dict(_CAPABILITIES)


def _controlled_reference(reference: np.ndarray, *, control: str | None,
                          rng: np.random.Generator | None,
                          noise_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    if control is None:
        return reference, {"pair_control": "true", "corrupted_pair_fraction": 0.0}
    if rng is None:
        raise ValueError("paired shuffled/noisy controls require an explicit RNG")
    if control == "shuffled":
        permutation = rng.permutation(len(reference))
        return reference[permutation], {
            "pair_control": "shuffled", "corrupted_pair_fraction": 1.0,
            "pair_permutation_fixed_points": int(np.sum(permutation == np.arange(len(reference)))),
        }
    if control != "noisy":
        raise AssertionError(control)
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("pair_noise_fraction must lie in [0,1]")
    count = int(round(noise_fraction * len(reference)))
    if noise_fraction > 0.0 and len(reference) > 1:
        count = max(2, count)
    count = min(count, len(reference))
    indices = rng.choice(len(reference), size=count, replace=False)
    result = np.array(reference, copy=True)
    if count > 1:
        # A cyclic shift is a derangement of the selected indices, unlike a
        # raw permutation which can silently retain an uncontrolled number of
        # true pairs.
        result[indices] = reference[np.roll(indices, 1)]
    return result, {
        "pair_control": "noisy", "requested_pair_noise_fraction": float(noise_fraction),
        "corrupted_pair_fraction": float(count / len(reference)) if count > 1 else 0.0,
    }


def fit_registered_map(name: str, source: np.ndarray, reference: np.ndarray, *,
                       readout_weight: np.ndarray | None = None,
                       rng: np.random.Generator | None = None,
                       **parameters: Any) -> FittedMap:
    """Fit a registered map after validating its declared information access."""

    base, suffix_control = _base_and_control(name)
    capabilities = method_capabilities(name)
    if not capabilities.supports_out_of_sample:
        raise ValueError(f"{name} is not eligible for held-out evaluation")
    if capabilities.requires_readout and readout_weight is None:
        raise ValueError(f"{name} requires a frozen readout weight")
    if capabilities.requires_pairing and len(source) != len(reference):
        raise ValueError(f"{name} requires equal paired calibration row counts")
    assignment_materialized = bool(parameters.pop("pair_assignment_materialized", False))
    requested_control = parameters.pop("pair_control", suffix_control)
    if suffix_control is not None and requested_control != suffix_control:
        raise ValueError("method suffix and pair_control disagree")
    if not capabilities.requires_pairing and requested_control is not None:
        raise ValueError("marginal-only methods cannot request a pair control")
    control_diagnostics: dict[str, Any] = {}
    fit_reference = reference
    if capabilities.requires_pairing and assignment_materialized:
        if requested_control not in {None, "shuffled", "noisy"}:
            raise ValueError("unknown materialized pair control")
        fit_reference = np.asarray(reference)
        control_diagnostics = {
            "pair_control": "true" if requested_control is None else requested_control,
            "pair_assignment_materialized": True,
            "corrupted_pair_fraction": float(parameters.pop("pair_noise_fraction", 0.0)),
        }
    elif capabilities.requires_pairing:
        fit_reference, control_diagnostics = _controlled_reference(
            np.asarray(reference), control=requested_control, rng=rng,
            noise_fraction=float(parameters.pop("pair_noise_fraction", 0.25)),
        )

    if base == "none":
        fitted = fit_identity(source, fit_reference)
    elif base == "mean":
        fitted = fit_mean(source, fit_reference)
    elif base == "diagonal":
        fitted = fit_diagonal(
            source, fit_reference,
            standard_deviation_floor=float(parameters.pop("standard_deviation_floor", 1e-6)),
        )
    elif base == "coordinate_quantile":
        fitted = fit_coordinate_quantile(
            source, fit_reference, knots=parameters.pop("knots", None)
        )
    elif base in {"coral", "bures"}:
        fitted = fit_covariance_gauge(
            source, fit_reference, gauge=base,
            estimator=parameters.pop("covariance_estimator", "oas"),
            relative_floor=float(parameters.pop("relative_floor", 1e-6)),
            rank=parameters.pop("rank", None),
        )
    elif base == "sliced_quantile":
        if rng is None:
            raise ValueError("sliced_quantile requires an explicit RNG")
        fitted = fit_sliced_quantile(
            source, fit_reference, rng=rng,
            projections=int(parameters.pop("projections", 64)),
            sweeps=int(parameters.pop("sweeps", 1)),
            rank=parameters.pop("rank", None),
            knots=parameters.pop("knots", None),
        )
    elif base == "third_moment_ica":
        if rng is None:
            raise ValueError("third_moment_ica requires an explicit RNG")
        fitted = fit_third_moment_ica(
            source, fit_reference, rng=rng,
            rank=int(parameters.pop("rank", 64)),
            max_iterations=int(parameters.pop("max_iterations", 1000)),
        )
    elif base == "paired_procrustes":
        fitted = fit_paired_procrustes(
            source, fit_reference,
            require_full_centered_rank=bool(
                parameters.pop("require_full_centered_rank", False)
            ),
        )
    elif base == "paired_ridge":
        fitted = fit_paired_ridge(
            source, fit_reference, ridge=float(parameters.pop("ridge", 1e-3))
        )
    elif base == "paired_reduced_rank":
        fitted = fit_paired_ridge(
            source, fit_reference, ridge=float(parameters.pop("ridge", 1e-3)),
            rank=int(parameters.pop("rank", 64)),
        )
    elif base == "paired_readout_gauge":
        assert readout_weight is not None
        fitted = fit_paired_readout_gauge(
            source, fit_reference, readout_weight,
            estimator=parameters.pop("covariance_estimator", "oas"),
            relative_floor=float(parameters.pop("relative_floor", 1e-6)),
        )
    else:
        raise AssertionError(base)
    if parameters:
        raise ValueError(f"unused parameters for {name}: {sorted(parameters)}")
    # Fitted maps are dataclasses with mutable diagnostic dictionaries by
    # design; annotating the information audit does not alter their action.
    fitted.diagnostics.update(control_diagnostics)
    fitted.diagnostics.update({
        "registered_method": base,
        "information_tier": capabilities.information_tier,
        "requires_pairing": capabilities.requires_pairing,
        "requires_readout": capabilities.requires_readout,
        "supports_out_of_sample": capabilities.supports_out_of_sample,
    })
    return fitted
