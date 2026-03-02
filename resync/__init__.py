"""
Top-level package for RAG and microservice components.

This package provides lazy imports to avoid circular dependencies.
"""

import importlib  # [P2-06 FIX] Use modern import API
import threading  # [P1-06 FIX] Thread-safety for lazy imports

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

_LOADED_MODULES: dict[str, object] = {}
# [P1-06 FIX] Lock to prevent race conditions in multi-threaded environments
# (e.g., Gunicorn/Uvicorn workers during startup, pytest-xdist)
_IMPORT_LOCK = threading.Lock()


def __getattr__(name: str) -> object:
    """PEP 562 lazy imports for resync package.

    Thread-safe lazy loading with double-checked locking pattern.

    Args:
        name: Attribute name to import (must be in _LAZY_MODULES)

    Returns:
        The imported module or attribute

    Raises:
        AttributeError: If name is not in _LAZY_MODULES
        ImportError: If the module import fails

    Example:
        >>> from resync import settings  # Triggers __getattr__("settings")
    """
    if name not in _LAZY_MODULES:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        ) from None

    # [P1-06 FIX] Double-checked locking: fast path without lock
    if name in _LOADED_MODULES:
        return _LOADED_MODULES[name]

    # [P1-06 FIX] Slow path: acquire lock for thread-safe import
    with _IMPORT_LOCK:
        # Check again inside lock (another thread may have loaded it)
        if name in _LOADED_MODULES:
            return _LOADED_MODULES[name]

        module_name, attr_name = _LAZY_MODULES[name]

        try:
            # [P2-06 FIX] Use importlib.import_module (modern API)
            module = importlib.import_module(module_name)
            result = module if attr_name is None else getattr(module, attr_name)
            _LOADED_MODULES[name] = result
            return result
        except ImportError as exc:
            raise ImportError(
                f"Failed to lazy import '{name}' from '{module_name}'"
            ) from exc


# Version info
__version__ = "6.2.0"
__author__ = "Resync Team"
