"""campaign-scale experimental components.

This namespace is deliberately separate from the frozen workshop/V1 runner.
It contains maps and measurements whose information requirements are explicit
and machine checked.
"""

from .method_registry import MethodCapabilities, fit_registered_map, method_capabilities

__all__ = ["MethodCapabilities", "fit_registered_map", "method_capabilities"]
