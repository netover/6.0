"""
API Gateway package for Resync.

Expose submodules lazily so tests can patch `resync.api_gateway.container.*`
without importing the full gateway stack during module resolution.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["container"]


def __getattr__(name: str) -> ModuleType:
    if name == "container":
        module = import_module(".container", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
