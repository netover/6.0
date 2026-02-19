"""
LiteLLM initialization for Resync TWS application.

This module sets up LiteLLM with proper configuration for
TWS-specific use cases, including local Ollama and remote API models.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Protocol, runtime_checkable

from resync.settings import settings

logger = logging.getLogger(__name__)


class LiteLLMMetrics:
    """Class to manage LiteLLM operational metrics in a thread-safe manner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.init_success: int = 0
        self.init_fail_reason: dict[str, int] = {}
        self.cost_calc_fail: int = 0

    def increment_init_success(self) -> None:
        """Increment initialization success counter."""
        with self._lock:
            self.init_success += 1

    def increment_init_fail_reason(self, reason: str) -> None:
        """Increment initialization failure counter by reason."""
        with self._lock:
            self.init_fail_reason[reason] = self.init_fail_reason.get(reason, 0) + 1

    def increment_cost_calc_fail(self) -> None:
        """Increment cost calculation failure counter."""
        with self._lock:
            self.cost_calc_fail += 1

    def get_metrics(self) -> dict[str, Any]:
        """Return thread-safe copy of metrics."""
        with self._lock:
            return {
                "init_success": self.init_success,
                "init_fail_reason": self.init_fail_reason.copy(),
                "cost_calc_fail": self.cost_calc_fail,
            }


class LiteLLMManager:
    """Thread-safe LiteLLM Router manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._router: RouterLike | None = None
        self._metrics = LiteLLMMetrics()

    def get_router(self) -> RouterLike | None:
        """Return singleton instance (thread-safe)."""
        with self._lock:
            if self._router is None:
                self._router = self._initialize_litellm()
            return self._router

    def reset_router(self) -> None:
        """Reset the singleton (useful for tests)."""
        with self._lock:
            self._router = None

    def _initialize_litellm(self) -> RouterLike | None:
        """Initialize LiteLLM Router based on settings."""
        strict = bool(getattr(settings, "LITELLM_STRICT_INIT", False))
        try:
            from litellm import Router as LiteLLMRouter  # type: ignore

            _apply_env_from_settings()

            enable_checks = getattr(settings, "LITELLM_PRE_CALL_CHECKS", True)
            kwargs: dict[str, Any] = {"enable_pre_call_checks": enable_checks}

            model_list = getattr(settings, "LITELLM_MODEL_LIST", None)
            if model_list is not None:
                kwargs["model_list"] = model_list  # apenas se houver

            # Optional additional parameters (when present in settings)
            for k_setting, k_arg in [
                ("LITELLM_NUM_RETRIES", "num_retries"),
                ("LITELLM_TIMEOUT", "timeout"),
            ]:
                val = getattr(settings, k_setting, None)
                if val is not None:
                    kwargs[k_arg] = val

            router: RouterLike = LiteLLMRouter(**kwargs)  # type: ignore[call-arg]
            logger.info(
                "LiteLLM initialized (pre_checks=%s, model_list=%s)",
                enable_checks,
                "provided" if model_list is not None else "default",
            )

            # Update metrics
            self._metrics.increment_init_success()

            return router

        except ImportError as import_err:
            logger.warning("LiteLLM not installed: %s", import_err, exc_info=False)
            self._metrics.increment_init_fail_reason("ImportError")
            if strict:
                raise
            return None
        except (ValueError, TypeError, KeyError, AttributeError) as err:
            # Configuration or programming errors should always fail fast
            logger.error("Failed to initialize LiteLLM due to configuration error: %s", err)
            self._metrics.increment_init_fail_reason(type(err).__name__)
            raise

        except RuntimeError:
            logger.exception("Runtime error initializing LiteLLM")
            self._metrics.increment_init_fail_reason("RuntimeError")
            if strict:
                raise
            return None
        except (OSError, ConnectionError, TimeoutError) as err:
            # Defensive umbrella for heterogeneous environments
            logger.exception("Unexpected error initializing LiteLLM")
            err_type = type(err).__name__
            self._metrics.increment_init_fail_reason(err_type)
            if strict:
                raise
            return None

    def get_metrics(self) -> dict[str, Any]:
        """Return LiteLLM operational metrics."""
        metrics = self._metrics.get_metrics()
        metrics["router_initialized"] = self._router is not None
        return metrics

    def calculate_completion_cost(self, completion_response: Any) -> float:
        """
        Calculate response cost using LiteLLM. Return 0.0 if unavailable/error.
        """
        try:
            from litellm import completion_cost  # type: ignore
        except ImportError as import_err:
            logger.info(
                "Cost calculation unavailable (litellm not installed?): %s",
                import_err,
                exc_info=False,
            )
            return 0.0

        try:
            return float(completion_cost(completion_response=completion_response))
        except (ValueError, TypeError, KeyError) as err:
            logger.warning("Could not calculate completion cost: %s", err, exc_info=False)
            self._metrics.increment_cost_calc_fail()
            return 0.0


# Singleton manager instance
__LITELLM_MANAGER_instance: LiteLLMManager | None = None


class _LazyLiteLLMManager:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance = None

    def get_instance(self) -> LiteLLMManager:
        if self._instance is None:
            self._instance = LiteLLMManager()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)


_LITELLM_MANAGER = _LazyLiteLLMManager()


def get__LITELLM_MANAGER() -> LiteLLMManager:
    """Return the singleton instance (preferred over using the proxy directly)."""
    return _LITELLM_MANAGER.get_instance()

@runtime_checkable
class RouterLike(Protocol):
    """Minimal protocol for LiteLLM Router."""

    def completion(self, *args, **kwargs):  # type: ignore
        """LiteLLM router completion method."""


def _set_env(name: str, value: str | None, *, overwrite: bool = False) -> bool:
    """
    Set an environment variable if value is present. By default, do not overwrite.
    Return True if variable was set/changed.
    """
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return False
    if overwrite or name not in os.environ:
        os.environ[name] = value  # type: ignore[arg-type]
        return True
    return False


def _apply_env_from_settings(*, overwrite: bool = False) -> dict[str, bool]:
    """
    Populate env vars from settings. Do not overwrite by default.
    Return a map {VAR: changed?} to facilitate testing/observability.
    """
    changed: dict[str, bool] = {}

    endpoint = getattr(settings, "LLM_ENDPOINT", None)
    # Source of truth: OPENAI_BASE_URL; only fill OPENAI_API_BASE if not exists
    changed["OPENAI_BASE_URL"] = _set_env("OPENAI_BASE_URL", endpoint, overwrite=overwrite)
    changed["OPENAI_API_BASE"] = _set_env(
        "OPENAI_API_BASE",
        endpoint if ("OPENAI_API_BASE" not in os.environ or overwrite) else None,
        overwrite=overwrite,
    )

    llm_key = getattr(settings, "llm_api_key", None)
    if hasattr(llm_key, "get_secret_value"):
        llm_key = llm_key.get_secret_value()
    changed["OPENAI_API_KEY"] = _set_env("OPENAI_API_KEY", llm_key, overwrite=overwrite)

    or_key = getattr(settings, "OPENROUTER_API_KEY", None)
    changed["OPENROUTER_API_KEY"] = _set_env("OPENROUTER_API_KEY", or_key, overwrite=overwrite)

    or_base = getattr(settings, "OPENROUTER_API_BASE", None)
    changed["OPENROUTER_API_BASE"] = _set_env("OPENROUTER_API_BASE", or_base, overwrite=overwrite)

    # Log which variables were set (never the values)
    set_vars = [k for k, v in changed.items() if v]
    if set_vars:
        logger.debug("LLM env set: %s", ", ".join(set_vars))
    return changed


# Compatibility functions maintaining original public API
def get_litellm_router() -> RouterLike | None:
    """
    Return singleton instance (thread-safe).
    """
    return _LITELLM_MANAGER.get_router()


def reset_litellm_router() -> None:
    """
    Reset the singleton (useful for tests).
    """
    _LITELLM_MANAGER.reset_router()


def initialize_litellm(*, overwrite_env: bool = False) -> RouterLike | None:
    """
    Initialize LiteLLM Router based on settings.
    overwrite_env: if True, overwrite existing environment variables.
    May raise in strict mode via settings.LITELLM_STRICT_INIT.
    """
    # Force re-initialization when overwrite_env is True
    if overwrite_env:
        _LITELLM_MANAGER.reset_router()

    return _LITELLM_MANAGER.get_router()


def calculate_completion_cost(completion_response: Any) -> float:
    """
    Calculate response cost using LiteLLM. Return 0.0 if unavailable/error.
    """
    return _LITELLM_MANAGER.calculate_completion_cost(completion_response)


def get_litellm_metrics() -> dict[str, Any]:
    """
    Return LiteLLM operational metrics.
    """
    return _LITELLM_MANAGER.get_metrics()
