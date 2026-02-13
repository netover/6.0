"""
Top-level package for RAG and microservice components.

This package provides lazy imports to avoid circular dependencies.
"""

__all__ = ["settings", "core", "api", "services", "models"]

# PEP 562 __getattr__ for lazy imports to avoid circular dependencies
_LAZY_MODULES = {
    # Core modules that may have circular dependencies
    "settings": ("resync.settings", "settings"),
    "core": ("resync.core", None),
    "api": ("resync.api", None),
    "services": ("resync.services", None),
    "models": ("resync.models", None),
}

_LOADED_MODULES = {}


def __getattr__(name):
    """PEP 562 lazy imports for resync package."""
    if name in _LAZY_MODULES:
        module_name, attr_name = _LAZY_MODULES[name]

        if name not in _LOADED_MODULES:
            try:
                module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
                _LOADED_MODULES[name] = module if attr_name is None else getattr(module, attr_name)
            except ImportError as e:
                _LOADED_MODULES[name] = None
                raise ImportError(f"Failed to lazy import {name}: {e}") from None

        return _LOADED_MODULES[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


# Version info
__version__ = "6.2.0"
__author__ = "Resync Team"
