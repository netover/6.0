"""resync.core.startup

Canonical startup sequence for the Resync ASGI application.

This module is designed to be called from the ASGI *lifespan* (startup
section, before the `yield`). That ensures the same validation and
startup health semantics apply in the real production path.

Design goals
------------
* **Single source of truth** for startup validation/health checks.
* Health checks return explicit **ok/fail/skipped** statuses.
* Only **critical** services affect fail-fast.
* Startup retries are bounded by a **total time budget**.
* Startup failures are **observable** (structured logs + clear reason codes).

The public entrypoint here is :func:`run_startup_checks`.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Literal

import httpx

from resync.core.exceptions import ConfigurationError
from resync.core.exceptions import (
    RedisAuthError,
    RedisConnectionError,
    RedisInitializationError,
    RedisTimeoutError,
)
from resync.core.structured_logger import get_logger
from resync.core.redis_init import RedisInitError, get_redis_initializer
from resync.settings import Settings, get_settings


Status = Literal["ok", "fail", "skipped"]


@dataclass(frozen=True)
class StartupCheck:
    """Result of a single startup check."""

    name: str
    status: Status
    critical: bool
    reason_code: str
    detail: str | None = None
    attempts: int = 1
    duration_ms: int = 0


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def get_startup_policy(settings: Settings) -> dict[str, Any]:
    """Resolve startup policy from env vars + settings.

    Env vars are intentionally used as a fast rollback mechanism.
    """

    strict_env = _parse_bool(os.getenv("STARTUP_STRICT"))
    strict = strict_env if strict_env is not None else bool(getattr(settings, "is_production", False))

    # Total startup time budget (seconds)
    max_total_seconds = float(os.getenv("STARTUP_MAX_TOTAL_SECONDS", "30"))

    # Generic retry knobs for external services (TWS/LLM). Redis has its own retry
    # parameters in settings + initialize_redis_with_retry().
    retries = int(os.getenv("STARTUP_SERVICE_RETRIES", "3"))
    base_delay = float(os.getenv("STARTUP_RETRY_BASE_DELAY_SECONDS", "0.2"))
    max_delay = float(os.getenv("STARTUP_RETRY_MAX_DELAY_SECONDS", "2.0"))

    return {
        "strict": strict,
        "max_total_seconds": max(1.0, max_total_seconds),
        "service_retries": max(1, retries),
        "retry_base_delay": max(0.0, base_delay),
        "retry_max_delay": max(0.0, max_delay),
    }


async def _tcp_reachable(host: str, port: int, timeout: float) -> tuple[bool, str | None]:
    """Simple TCP reachability check."""

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        # Explicitly close immediately.
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            # Some loop implementations can raise here; ignore.
            pass
        # Silence unused variable warning.
        _ = reader
        return True, None
    except asyncio.TimeoutError:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error:{type(e).__name__}"
    except Exception as e:  # pragma: no cover
        return False, f"unexpected:{type(e).__name__}"


async def _http_healthy(url: str, timeout: float) -> tuple[bool, str | None, int | None, str | None]:
    """HTTP health check.

    We prefer strict 2xx success. Redirects are treated as unhealthy because
    they often indicate misrouting/auth.
    """

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.get(url)
            if 200 <= resp.status_code < 300:
                return True, None, resp.status_code, None
            if 300 <= resp.status_code < 400:
                # Don't log raw URL/credentials; return minimal signal.
                return False, "redirect", resp.status_code, resp.headers.get("location")
            return False, "http_status", resp.status_code, None
    except httpx.TimeoutException:
        return False, "timeout", None, None
    except httpx.RequestError as e:
        return False, f"request_error:{type(e).__name__}", None, None
    except Exception as e:  # pragma: no cover
        return False, f"unexpected:{type(e).__name__}", None, None


async def _retry(
    name: str,
    fn: Callable[[int], Awaitable[StartupCheck]],
    *,
    retries: int,
    base_delay: float,
    max_delay: float,
    deadline: float,
) -> StartupCheck:
    """Retry a check until ok/skip or retries exhausted or deadline reached."""

    last: StartupCheck | None = None
    for attempt in range(1, retries + 1):
        if time.monotonic() >= deadline:
            break
        last = await fn(attempt)
        if last.status in ("ok", "skipped"):
            return last
        if attempt < retries:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # Respect global deadline.
            remaining = max(0.0, deadline - time.monotonic())
            await asyncio.sleep(min(delay, remaining))

    if last is None:
        # Should be impossible, but keep safe.
        return StartupCheck(
            name=name,
            status="fail",
            critical=True,
            reason_code="startup_budget_exhausted",
            detail="Startup budget exhausted before first attempt",
        )
    return last


async def _check_tws(settings: Settings, *, critical: bool, deadline: float, policy: dict[str, Any]) -> StartupCheck:
    start = time.monotonic()
    host = getattr(settings, "tws_host", None)
    port = getattr(settings, "tws_port", None)
    timeout = float(getattr(settings, "STARTUP_TCP_CHECK_TIMEOUT", 3.0))

    if not host or not port:
        status: Status = "fail" if critical else "skipped"
        reason = "not_configured"
        detail = "TWS host/port not configured"
        return StartupCheck(
            name="tws_reachability",
            status=status,
            critical=critical,
            reason_code=reason,
            detail=detail,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def attempt_fn(attempt: int) -> StartupCheck:
        ok, reason = await _tcp_reachable(str(host), int(port), timeout)
        dur_ms = int((time.monotonic() - start) * 1000)
        if ok:
            return StartupCheck(
                name="tws_reachability",
                status="ok",
                critical=critical,
                reason_code="reachable",
                attempts=attempt,
                duration_ms=dur_ms,
            )
        return StartupCheck(
            name="tws_reachability",
            status="fail",
            critical=critical,
            reason_code=reason or "unreachable",
            detail=f"TWS not reachable ({host}:{port})",
            attempts=attempt,
            duration_ms=dur_ms,
        )

    return await _retry(
        "tws_reachability",
        attempt_fn,
        retries=int(policy["service_retries"]),
        base_delay=float(policy["retry_base_delay"]),
        max_delay=float(policy["retry_max_delay"]),
        deadline=deadline,
    )


async def _check_llm(settings: Settings, *, critical: bool, deadline: float, policy: dict[str, Any]) -> StartupCheck:
    start = time.monotonic()
    endpoint = getattr(settings, "llm_endpoint", None)
    timeout = float(getattr(settings, "STARTUP_LLM_HEALTH_TIMEOUT", 5.0))

    if not endpoint:
        status: Status = "fail" if critical else "skipped"
        return StartupCheck(
            name="llm_service",
            status=status,
            critical=critical,
            reason_code="not_configured",
            detail="LLM endpoint not configured",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def attempt_fn(attempt: int) -> StartupCheck:
        ok, reason, status_code, location = await _http_healthy(str(endpoint), timeout)
        dur_ms = int((time.monotonic() - start) * 1000)
        if ok:
            return StartupCheck(
                name="llm_service",
                status="ok",
                critical=critical,
                reason_code="healthy",
                attempts=attempt,
                duration_ms=dur_ms,
            )

        detail = None
        if status_code is not None:
            detail = f"HTTP {status_code}"
            if reason == "redirect" and location:
                detail = f"HTTP {status_code} redirect"

        return StartupCheck(
            name="llm_service",
            status="fail",
            critical=critical,
            reason_code=reason or "unhealthy",
            detail=detail,
            attempts=attempt,
            duration_ms=dur_ms,
        )

    return await _retry(
        "llm_service",
        attempt_fn,
        retries=int(policy["service_retries"]),
        base_delay=float(policy["retry_base_delay"]),
        max_delay=float(policy["retry_max_delay"]),
        deadline=deadline,
    )


async def _check_redis(
    settings: Settings, *, deadline: float, logger: Any
) -> StartupCheck:
    start = time.monotonic()

    # Operators can explicitly disable Redis (e.g. CI, lightweight deployments).
    # Accept common boolean values to match .env / container conventions.
    if _parse_bool(os.getenv("RESYNC_DISABLE_REDIS")) is True:
        return StartupCheck(
            name="redis_connection",
            status="skipped",
            critical=False,
            reason_code="disabled",
            detail="Redis disabled by RESYNC_DISABLE_REDIS",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        # Canonical Redis bootstrap is owned by resync.core.redis_init.
        # This ensures the worker has exactly one Redis client instance,
        # shared by all code paths via the module-level lazy accessors.
        initializer = get_redis_initializer()
        await initializer.initialize(
            max_retries=int(getattr(settings, "redis_max_startup_retries", 5)),
            health_check_interval=int(getattr(settings, "redis_health_check_interval", 5)),
            redis_url=getattr(settings, "redis_url", None),
        )
        return StartupCheck(
            name="redis_connection",
            status="ok",
            critical=True,
            reason_code="initialized",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except ConfigurationError as e:
        logger.error(
            "startup_redis_configuration_error",
            error_message=str(e),
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code="configuration_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except RedisAuthError as e:
        logger.error(
            "startup_redis_authentication_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code="redis_auth_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except RedisTimeoutError as e:
        logger.error(
            "startup_redis_timeout_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code="redis_timeout",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except RedisConnectionError as e:
        logger.error(
            "startup_redis_connection_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code="redis_connection_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except RedisInitializationError as e:
        logger.error(
            "startup_redis_initialization_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code="redis_initialization_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as e:
        # Prefer a stable reason code for known initializer failures.
        if isinstance(e, RedisInitError):
            reason = "redis_init_error"
        else:
            # If startup budget is exhausted we still want a deterministic failure.
            reason = "startup_budget_exhausted" if time.monotonic() >= deadline else type(e).__name__
        logger.error(
            "startup_redis_unexpected_error",
            error_message=str(e),
            reason_code=reason,
        )
        return StartupCheck(
            name="redis_connection",
            status="fail",
            critical=True,
            reason_code=reason,
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )


async def run_startup_checks(*, settings: Settings | None = None) -> dict[str, Any]:
    """Run the canonical startup sequence.

    Returns a dict suitable for logging and/or surfacing through a diagnostics
    endpoint.
    """

    settings_obj = settings or get_settings()
    logger = get_logger("resync.startup")
    policy = get_startup_policy(settings_obj)

    started_at = time.monotonic()
    deadline = started_at + float(policy["max_total_seconds"])

    redis_disabled = _parse_bool(os.getenv("RESYNC_DISABLE_REDIS")) is True
    critical_services: list[str] = [] if redis_disabled else ["redis_connection"]
    if getattr(settings_obj, "require_tws_at_boot", False):
        critical_services.append("tws_reachability")
    if getattr(settings_obj, "require_llm_at_boot", False):
        critical_services.append("llm_service")

    logger.info(
        "startup_checks_begin",
        strict=policy["strict"],
        max_total_seconds=policy["max_total_seconds"],
        critical_services=critical_services,
    )

    # Execute checks. Redis init is sequential (it sets up global redis/idempotency).
    # TWS/LLM checks can run in parallel.
    results: list[StartupCheck] = []

    redis_result = await _check_redis(settings_obj, deadline=deadline, logger=logger)
    results.append(redis_result)

    # Parallel checks (bounded by remaining time budget)
    remaining = max(0.0, deadline - time.monotonic())
    if remaining > 0:
        try:
            tws_task = _check_tws(
                settings_obj,
                critical=getattr(settings_obj, "require_tws_at_boot", False),
                deadline=deadline,
                policy=policy,
            )
            llm_task = _check_llm(
                settings_obj,
                critical=getattr(settings_obj, "require_llm_at_boot", False),
                deadline=deadline,
                policy=policy,
            )
            # Cap gather by the global remaining budget.
            tws_res, llm_res = await asyncio.wait_for(
                asyncio.gather(tws_task, llm_task), timeout=remaining
            )
            results.extend([tws_res, llm_res])
        except asyncio.TimeoutError:
            # Budget exhausted while waiting for parallel checks.
            results.extend(
                [
                    StartupCheck(
                        name="tws_reachability",
                        status="fail" if getattr(settings_obj, "require_tws_at_boot", False) else "skipped",
                        critical=getattr(settings_obj, "require_tws_at_boot", False),
                        reason_code="startup_budget_exhausted",
                        detail="Startup budget exhausted during TWS check",
                    ),
                    StartupCheck(
                        name="llm_service",
                        status="fail" if getattr(settings_obj, "require_llm_at_boot", False) else "skipped",
                        critical=getattr(settings_obj, "require_llm_at_boot", False),
                        reason_code="startup_budget_exhausted",
                        detail="Startup budget exhausted during LLM check",
                    ),
                ]
            )

    # Compute overall health: only critical services matter.
    overall_health = True
    for r in results:
        if r.critical and r.status != "ok":
            overall_health = False

    duration_ms = int((time.monotonic() - started_at) * 1000)
    results_payload = [asdict(r) for r in results]

    logger.info(
        "startup_checks_end",
        overall_health=overall_health,
        duration_ms=duration_ms,
        results=results_payload,
    )

    return {
        "strict": bool(policy["strict"]),
        "critical_services": critical_services,
        "overall_health": overall_health,
        "duration_ms": duration_ms,
        "results": results_payload,
    }


def enforce_startup_policy(result: dict[str, Any]) -> None:
    """Fail the current process if strict startup is enabled and overall health failed."""

    if result.get("strict") and not result.get("overall_health"):
        # NOTE: Raising SystemExit ensures a non-zero exit code.
        # This is important because some server/proc-manager combinations have
        # historically swallowed startup failures.
        raise SystemExit(1)
