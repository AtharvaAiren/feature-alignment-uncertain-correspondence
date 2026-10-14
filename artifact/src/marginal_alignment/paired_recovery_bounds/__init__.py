"""Paired, corruption-specific conformal recovery-bound study.

This namespace is deliberately additive.  It does not change the completed
base campaign or the unpaired natural-shift correspondence ceiling.
"""

from .conformal import conformal_rank, split_conformal_quantile
from .geometry import (
    COMMON_COST_GEOMETRY,
    FittedRecoveryGeometry,
    batch_cost_problem,
    fit_recovery_geometry,
    frozen_predictions,
)
from .optimization import (
    RecoveryCertificate,
    certify_recovery_error,
    unconstrained_label_count_bound,
)
from .splitting import RecoveryPartition, build_recovery_partition

__all__ = [
    "COMMON_COST_GEOMETRY",
    "FittedRecoveryGeometry",
    "RecoveryCertificate",
    "RecoveryPartition",
    "batch_cost_problem",
    "build_recovery_partition",
    "certify_recovery_error",
    "conformal_rank",
    "fit_recovery_geometry",
    "frozen_predictions",
    "split_conformal_quantile",
    "unconstrained_label_count_bound",
]
