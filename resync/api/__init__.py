from typing import Any

_LAZY_API_EXPORTS: dict[str, tuple[str, str]] = {
    "create_app": ("resync.api.app_factory", "create_app"),
    "router": ("resync.api.app_factory", "router"),
    "register_routes": ("resync.api.routes", "register_routes"),
}
_LOADED: dict[str, Any] = {}


def __getattr__(name: str):
    if name in _LAZY_API_EXPORTS:
        mod, attr = _LAZY_API_EXPORTS[name]
        if name not in _LOADED:
            module = __import__(mod, fromlist=[attr])
            _LOADED[name] = getattr(module, attr)
        return _LOADED[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# =============================================================================
# v5.8.0: Application factory compatibility layer
# =============================================================================

def create_app():
    """
    Create and configure the FastAPI application.

    v5.8.0: Unified API entry point.

    Usage:
        from resync.api import create_app
        app = create_app()
    """
    from resync.app_factory import ApplicationFactory
    factory = ApplicationFactory()
    return factory.create_app()

