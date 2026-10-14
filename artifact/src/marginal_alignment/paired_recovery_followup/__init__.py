"""Prospectively frozen follow-up utilities for paired recovery bounds.

This namespace is additive.  It deliberately leaves the completed
``paired_recovery_bounds`` study and its source lock byte-for-byte unchanged.
"""

from .calibration import (
    DirectCalibration,
    DirectStandardizer,
    calibrate_direct_recovery_bounds,
    fit_direct_standardizer,
    simultaneous_direct_score,
)
from .controls import (
    column_permutation,
    column_permuted_cost_problem,
)
from .signed_optimization import (
    LinearRewardCertificate,
    certify_linear_reward,
    signed_contrast_reward_matrix,
    verify_linear_reward_certificate,
)

__all__ = [
    "DirectCalibration",
    "DirectStandardizer",
    "LinearRewardCertificate",
    "calibrate_direct_recovery_bounds",
    "certify_linear_reward",
    "column_permutation",
    "column_permuted_cost_problem",
    "fit_direct_standardizer",
    "signed_contrast_reward_matrix",
    "simultaneous_direct_score",
    "verify_linear_reward_certificate",
]
