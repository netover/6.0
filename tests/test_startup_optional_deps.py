from __future__ import annotations

import builtins
from typing import Any

import pytest


class _BlockImports:
    """Context manager to force ImportError for selected top-level modules.

    This allows us to verify that the startup path is resilient to *optional*
    dependencies being absent (e.g., slowapi/jinja2/orjson).
    """

    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked
        self._real_import = builtins.__import__

    def __enter__(self) -> None:
        def _import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0):
            base = name.split(".", 1)[0]
            if base in self.blocked:
                raise ImportError(f"blocked optional dependency: {base}")
            return self._real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _import  # type: ignore[assignment]

    def __exit__(self, exc_type, exc, tb) -> None:
        builtins.__import__ = self._real_import  # type: ignore[assignment]


def test_create_app_without_optional_deps() -> None:
    blocked = {"slowapi", "jinja2", "orjson"}
    with _BlockImports(blocked):
        # Import inside the blocked context so module-level optional imports are exercised
        from resync.app_factory import create_app

        app = create_app()
        assert app is not None
        # Basic contract: app has state object and routes list
        assert hasattr(app, "state")
        assert hasattr(app, "routes")


def test_startup_module_imports_valkey_exceptions() -> None:
    # The startup module imports Valkey* exception names; these should exist
    from resync.core import exceptions as exc

    assert hasattr(exc, "ValkeyAuthError")
    assert hasattr(exc, "ValkeyConnectionError")
    assert hasattr(exc, "ValkeyInitializationError")
    assert hasattr(exc, "ValkeyTimeoutError")
