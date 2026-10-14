"""Marginal feature alignment and correspondence-identifiability audit."""

from .ceiling import bayes_orbit_ceiling, empirical_group_ceiling
from .transforms import fit_transform

__all__ = ["bayes_orbit_ceiling", "empirical_group_ceiling", "fit_transform"]
__version__ = "0.1.0"
