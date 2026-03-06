"""resync.core.litellm_init

Thread-safe LiteLLM Router initialization + hot-reload + metrics.

Design goals:
- Lazy init (gunicorn --preload safe)
- Fail-fast on programming/config errors; optionally soft-fail when LiteLLM missing
- Safe router reload when resync/core/litellm_config.yaml changes
- First-class observability hooks (callbacks + cache)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import hashlib
from typing import Any, Protocol, runtime_checkable

from resync.core.litellm_config_store import load_litellm_config
from resync.settings import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class RouterLike(Protocol):
    """Minimal protocol used across the codebase."""

    # sync
    def completion(self, *args, **kwargs):  # type: ignore
        ...

    # async
    async def acompletion(self, *args, **kwargs):  # type: ignore
        ...

    # optional async embeddings
    async def aembedding(self, *args, **kwargs):  # type: ignore
        ...


class LiteLLMMetrics:
    """Thread-safe operational metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.init_success: int = 0
        self.init_fail_reason: dict[str, int] = {}
        self.cost_calc_fail: int = 0
        self.reloads: int = 0
        self.reload_fail: int = 0

    def inc(self, field: str, reason: str | None = None) -> None:
        with self._lock:
            if field == "init_success":
                self.init_success += 1
            elif field == "cost_calc_fail":
                self.cost_calc_fail += 1
            elif field == "reloads":
                self.reloads += 1
            elif field == "reload_fail":
                self.reload_fail += 1
            elif field == "init_fail_reason" and reason:
                self.init_fail_reason[reason] = self.init_fail_reason.get(reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "init_success": self.init_success,
                "init_fail_reason": dict(self.init_fail_reason),
                "cost_calc_fail": self.cost_calc_fail,
                "reloads": self.reloads,
                "reload_fail": self.reload_fail,
            }


def _set_env(name: str, value: str | None, *, overwrite: bool = False) -> bool:
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
    """Populate env vars (never logs secrets)."""
    changed: dict[str, bool] = {}

    endpoint = getattr(settings, "llm_endpoint", None)
    changed["OPENAI_BASE_URL"] = _set_env("OPENAI_BASE_URL", endpoint, overwrite=overwrite)
    changed["OPENAI_API_BASE"] = _set_env(
        "OPENAI_API_BASE",
        endpoint if ("OPENAI_API_BASE" not in os.environ or overwrite) else None,
        overwrite=overwrite,
    )

    llm_key = getattr(settings, "llm_api_key", None)
    if llm_key is not None and hasattr(llm_key, "get_secret_value"):
        llm_key = llm_key.get_secret_value()
    changed["OPENAI_API_KEY"] = _set_env("OPENAI_API_KEY", llm_key, overwrite=overwrite)

    or_key = getattr(settings, "OPENROUTER_API_KEY", None)
    changed["OPENROUTER_API_KEY"] = _set_env("OPENROUTER_API_KEY", or_key, overwrite=overwrite)

    or_base = getattr(settings, "OPENROUTER_API_BASE", None)
    changed["OPENROUTER_API_BASE"] = _set_env("OPENROUTER_API_BASE", or_base, overwrite=overwrite)

    set_vars = [k for k, v in changed.items() if v]

    # Aliases for OpenAI-compatible OpenRouter usage: allow OPENAI_* to drive OPENROUTER_* when unset.
    if "OPENROUTER_API_KEY" not in os.environ and "OPENAI_API_KEY" in os.environ:
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]
        changed.setdefault("OPENROUTER_API_KEY", True)

    if not os.environ.get("OPENROUTER_API_BASE"):
        base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if base:
            os.environ["OPENROUTER_API_BASE"] = base
            changed.setdefault("OPENROUTER_API_BASE", True)
    if set_vars:
        logger.debug("LLM env set: %s", ", ".join(set_vars))
    return changed


def _setup_litellm_cache_and_callbacks() -> None:
    """Configure semantic cache + callbacks (best-effort)."""
    try:
        import litellm  # type: ignore
        from litellm.caching import Cache  # type: ignore
    except Exception:
        return

    # Cache via Valkey (Valkey-compatible). Normalize scheme for valkey-py.
    try:
        url = settings.valkey_url.get_secret_value()
    except Exception:
        url = "valkey://localhost:6379/0"

    if url.startswith("valkey://"):
        redis_url = "valkey://" + url[len("valkey://") :]
    elif url.startswith("valkeys://"):
        redis_url = "rediss://" + url[len("valkeys://") :]
    else:
        redis_url = url

    ttl = int(getattr(settings, "llm_cache_ttl_seconds", 3600))
    enabled = bool(getattr(settings, "llm_cache_enabled", True))
    if enabled:
        try:
            litellm.cache = Cache(type="valkey", redis_url=redis_url, ttl=ttl)
        except TypeError:
            # older litellm versions
            litellm.cache = Cache(type="valkey", host=redis_url, ttl=ttl)

    try:
        from resync.core.litellm_hooks import on_litellm_failure, on_litellm_success

        litellm.success_callback = [on_litellm_success]
        litellm.failure_callback = [on_litellm_failure]
    except Exception:
        return


class LiteLLMManager:
    """Singleton router manager with safe hot-reload."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._router: RouterLike | None = None
        self._metrics = LiteLLMMetrics()
        self._config_mtime_ns: int | None = None
        self._config_hash: str | None = None
        self._last_maybe_reload_check_s: float = 0.0

    def get_router(self) -> RouterLike | None:
        with self._lock:
            if self._router is None:
                self._router = self._initialize_litellm()
            return self._router

    def reset_router(self) -> None:
        with self._lock:
            self._router = None
            self._config_mtime_ns = None
            self._last_maybe_reload_check_s = 0.0

    def reload_router(self) -> RouterLike | None:
        """Force reload."""
        with self._lock:
            self._router = None
            self._config_mtime_ns = None
            self._config_hash = None
            self._last_maybe_reload_check_s = 0.0
            self._metrics.inc("reloads")
        try:
            return self.get_router()
        except Exception:
            self._metrics.inc("reload_fail")
            return None

    def maybe_reload_router(self, *, min_interval_seconds: float = 5.0) -> RouterLike | None:
        """Reload router if config changed on disk (rate-limited check).

        P0 FIX: Avoid calling get_router() while holding self._lock (would deadlock).
        P1 FIX: Use a content hash instead of relying only on mtime (mtime can miss rapid swaps).
        """
        now = time.monotonic()

        # Fast path: if we're within the rate-limit window, return existing router if present.
        with self._lock:
            within_window = (now - self._last_maybe_reload_check_s) < min_interval_seconds
            router = self._router
            if within_window and router is not None:
                return router
            if not within_window:
                self._last_maybe_reload_check_s = now

        # If no router yet, just ensure it's initialized.
        if router is None and within_window:
            return self.get_router()

        try:
            cfg = load_litellm_config()
        except Exception:
            return self.get_router()

        cfg_hash = hashlib.sha256(cfg.text.encode("utf-8")).hexdigest()

        with self._lock:
            # First load will set mtime/hash on initialization.
            if self._config_hash is None and self._config_mtime_ns is None:
                return self._router if self._router is not None else self.get_router()

            changed = False
            if self._config_hash is not None and cfg_hash != self._config_hash:
                changed = True
            elif self._config_hash is None and self._config_mtime_ns is not None and cfg.mtime_ns != self._config_mtime_ns:
                changed = True

            if changed:
                logger.info("litellm_config_changed_reloading_router")
                self._router = None
                self._config_mtime_ns = None
                self._config_hash = None
                self._metrics.inc("reloads")

        return self.get_router()

    def get_metrics(self) -> dict[str, Any]:
        m = self._metrics.snapshot()
        m["router_initialized"] = self._router is not None
        m["config_mtime_ns"] = self._config_mtime_ns
        return m

    def calculate_completion_cost(self, completion_response: Any) -> float:
        """Best-effort cost calculation."""
        try:
            from litellm import completion_cost  # type: ignore
        except ImportError:
            return 0.0

        try:
            return float(completion_cost(completion_response=completion_response))
        except Exception:
            self._metrics.inc("cost_calc_fail")
            return 0.0

    def _initialize_litellm(self) -> RouterLike | None:
        strict = bool(getattr(settings, "LITELLM_STRICT_INIT", False))
        try:
            from litellm import Router as LiteLLMRouter  # type: ignore

            _apply_env_from_settings()
            _setup_litellm_cache_and_callbacks()

            cfg_obj = None
            try:
                cfg_obj = load_litellm_config()
            except Exception:
                cfg_obj = None
            yaml_cfg = (cfg_obj.raw if cfg_obj else {})
            if cfg_obj:
                self._config_mtime_ns = cfg_obj.mtime_ns
                self._config_hash = hashlib.sha256(cfg_obj.text.encode("utf-8")).hexdigest()

            enable_checks = getattr(settings, "LITELLM_PRE_CALL_CHECKS", True)
            kwargs: dict[str, Any] = {"enable_pre_call_checks": enable_checks}

            # Prefer explicit settings override; else YAML.
            model_list = getattr(settings, "LITELLM_MODEL_LIST", None)
            if model_list is None:
                model_list = yaml_cfg.get("model_list")
            if model_list is not None:
                kwargs["model_list"] = model_list

            model_aliases = yaml_cfg.get("model_aliases")
            if isinstance(model_aliases, dict):
                kwargs["model_aliases"] = model_aliases

            router_settings = yaml_cfg.get("router_settings")
            if isinstance(router_settings, dict):
                kwargs["router_settings"] = router_settings

            # Optional parameters
            for k_setting, k_arg in [
                ("LITELLM_NUM_RETRIES", "num_retries"),
                ("LITELLM_TIMEOUT", "timeout"),
            ]:
                val = getattr(settings, k_setting, None)
                if val is not None:
                    kwargs[k_arg] = val

            router: RouterLike = LiteLLMRouter(**kwargs)  # type: ignore[call-arg]
            self._metrics.inc("init_success")
            logger.info(
                "LiteLLM initialized (pre_checks=%s, model_list=%s)",
                enable_checks,
                "provided" if model_list is not None else "default",
            )
            return router

        except ImportError as import_err:
            logger.warning("LiteLLM not installed: %s", import_err, exc_info=False)
            self._metrics.inc("init_fail_reason", reason="ImportError")
            if strict:
                raise
            return None
        except (ValueError, TypeError, KeyError, AttributeError) as err:
            # Programming/config errors: always fail-fast.
            logger.error("Failed to initialize LiteLLM: %s", err)
            self._metrics.inc("init_fail_reason", reason=type(err).__name__)
            raise
        except Exception as err:  # noqa: BLE001
            logger.exception("Unexpected error initializing LiteLLM")
            self._metrics.inc("init_fail_reason", reason=type(err).__name__)
            if strict:
                raise
            return None


class _LazyLiteLLMManager:
    """Lazy proxy to avoid import-time side effects."""

    __slots__ = ("_instance", "_lock")

    def __init__(self) -> None:
        self._instance: LiteLLMManager | None = None
        self._lock = threading.Lock()

    def get_instance(self) -> LiteLLMManager:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = LiteLLMManager()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)


_LITELLM_MANAGER = _LazyLiteLLMManager()


def get_litellm_manager() -> LiteLLMManager:
    return _LITELLM_MANAGER.get_instance()


def get_litellm_router() -> RouterLike | None:
    return _LITELLM_MANAGER.get_router()


def maybe_reload_litellm_router() -> RouterLike | None:
    return _LITELLM_MANAGER.maybe_reload_router()


def reset_litellm_router() -> None:
    _LITELLM_MANAGER.reset_router()


def initialize_litellm(*, overwrite_env: bool = False) -> RouterLike | None:
    if overwrite_env:
        _LITELLM_MANAGER.reset_router()
        _apply_env_from_settings(overwrite=True)
    return _LITELLM_MANAGER.get_router()


def calculate_completion_cost(completion_response: Any) -> float:
    return _LITELLM_MANAGER.calculate_completion_cost(completion_response)


def get_litellm_metrics() -> dict[str, Any]:
    return _LITELLM_MANAGER.get_metrics()


def reload_litellm_router() -> RouterLike | None:
    return get_litellm_manager().reload_router()
