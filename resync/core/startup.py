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



import asyncio
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fastapi import FastAPI

import httpx

from resync.core.exceptions import (
    ConfigurationError,
    RedisAuthError,
    RedisConnectionError,
    RedisInitializationError,
    RedisTimeoutError,
)
from resync.core.redis_init import RedisInitError, get_redis_initializer
from resync.core.structured_logger import get_logger
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
    strict = (
        strict_env
        if strict_env is not None
        else bool(getattr(settings, "is_production", False))
    )

    # Total startup time budget (seconds)
    # Prefer settings.startup_timeout, fallback to env var, default 30
    timeout_default = float(getattr(settings, "startup_timeout", 30))
    max_total_seconds = float(
        os.getenv("STARTUP_MAX_TOTAL_SECONDS", str(timeout_default))
    )

    # Valores de retry com clamp para evitar absurdos
    retries = max(1, min(20, int(os.getenv("STARTUP_SERVICE_RETRIES", "3"))))
    base_delay = max(
        0.1,
        min(10.0, float(os.getenv("STARTUP_RETRY_BASE_DELAY_SECONDS", "0.2"))),
    )
    max_delay = max(
        base_delay,
        min(30.0, float(os.getenv("STARTUP_RETRY_MAX_DELAY_SECONDS", "2.0"))),
    )

    return {
        "strict": strict,
        "max_total_seconds": max(1.0, max_total_seconds),
        "service_retries": retries,
        "retry_base_delay": base_delay,
        "retry_max_delay": max_delay,
    }


async def _tcp_reachable(
    host: str, port: int, timeout: float
) -> tuple[bool, str | None]:
    """Simple TCP reachability check."""

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        # Explicitly close immediately.
        writer.close()
        try:
            # apenas erros de I/O/loop são ignorados
            await writer.wait_closed()
        except (OSError, AttributeError):
            pass
        # Silence unused variable warning.
        _ = reader
        return True, None
    except TimeoutError:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error:{type(e).__name__}"
    except Exception as e:  # pragma: no cover
        return False, f"unexpected:{type(e).__name__}"


async def _http_healthy(
    url: str, timeout: float
) -> tuple[bool, str | None, int | None, str | None]:
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
                loc = resp.headers.get("location")
                # sanitiza querystring para não vazar tokens
                safe_loc = None
                if loc:
                    safe_loc = loc.split("?", 1)[0] if "?" in loc else loc
                return False, "redirect", resp.status_code, safe_loc
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
    logger = get_logger("resync.startup")
    last: StartupCheck | None = None
    for attempt in range(1, retries + 1):
        now = time.monotonic()
        if now >= deadline:
            logger.warning(f"{name}_retry_deadline_reached", attempt=attempt)
            break
        logger.debug(f"{name}_check_attempt", attempt=attempt, max_retries=retries)
        last = await fn(attempt)
        if last.status in ("ok", "skipped"):
            return last
        if attempt < retries:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                break
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


async def _check_tws(
    settings: Settings, *, critical: bool, deadline: float, policy: dict[str, Any]
) -> StartupCheck:
    start = time.monotonic()
    host = getattr(settings, "tws_host", None)
    port = getattr(settings, "tws_port", None)
    raw_timeout = getattr(settings, "STARTUP_TCP_CHECK_TIMEOUT", "3.0")
    timeout = float(raw_timeout) if raw_timeout else 3.0

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
        # sanitiza host básico
        safe_host = str(host).split("@")[-1]
        return StartupCheck(
            name="tws_reachability",
            status="fail",
            critical=critical,
            reason_code=reason or "unreachable",
            detail=f"TWS not reachable ({safe_host}:{port})",
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


async def _check_llm(
    settings: Settings, *, critical: bool, deadline: float, policy: dict[str, Any]
) -> StartupCheck:
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


async def _check_rag(
    settings: Settings, *, critical: bool, deadline: float, policy: dict[str, Any]
) -> StartupCheck:
    start = time.monotonic()
    url = getattr(settings, "rag_service_url", None)
    timeout = float(getattr(settings, "RAG_SERVICE_TIMEOUT", 5.0))

    if not url:
        status: Status = "skipped"
        return StartupCheck(
            name="rag_service",
            status=status,
            critical=critical,
            reason_code="not_configured",
            detail="RAG service URL not configured",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def attempt_fn(attempt: int) -> StartupCheck:
        # Check if URL is valid by trying to connect/ping
        ok, reason, status_code, _ = await _http_healthy(str(url), timeout)
        dur_ms = int((time.monotonic() - start) * 1000)
        if ok:
            return StartupCheck(
                name="rag_service",
                status="ok",
                critical=critical,
                reason_code="healthy",
                attempts=attempt,
                duration_ms=dur_ms,
            )

        return StartupCheck(
            name="rag_service",
            status="fail",
            critical=critical,
            reason_code=reason or "unhealthy",
            detail=f"HTTP {status_code}" if status_code else reason,
            attempts=attempt,
            duration_ms=dur_ms,
        )

    return await _retry(
        "rag_service",
        attempt_fn,
        retries=int(policy["service_retries"]),
        base_delay=float(policy["retry_base_delay"]),
        max_delay=float(policy["retry_max_delay"]),
        deadline=deadline,
    )


async def _check_redis(
    settings: Settings, *, deadline: float, logger: Any, disabled: bool
) -> StartupCheck:
    start = time.monotonic()

    if disabled:
        logger.info(
            "redis_startup_disabled",
            reason="RESYNC_DISABLE_REDIS=true, skipping redis initialization",
        )
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
            health_check_interval=int(
                getattr(settings, "redis_health_check_interval", 5)
            ),
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
            reason = (
                "startup_budget_exhausted"
                if time.monotonic() >= deadline
                else type(e).__name__
            )
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

    redis_disabled = _parse_bool(os.getenv("RESYNC_DISABLE_REDIS")) or False
    if redis_disabled:
        logger.warning(
            "redis_marked_noncritical",
            reason="RESYNC_DISABLE_REDIS=true",
        )

    critical_services: list[str] = [] if redis_disabled else ["redis_connection"]
    if getattr(settings_obj, "require_tws_at_boot", False):
        critical_services.append("tws_reachability")
    if getattr(settings_obj, "require_llm_at_boot", False):
        critical_services.append("llm_service")
    if getattr(settings_obj, "require_rag_at_boot", False):
        critical_services.append("rag_service")

    logger.info(
        "startup_checks_begin",
        strict=policy["strict"],
        max_total_seconds=policy["max_total_seconds"],
        critical_services=critical_services,
    )

    # Execute checks. Redis init is sequential (it sets up global redis/idempotency).
    # TWS/LLM checks can run in parallel.
    results: list[StartupCheck] = []

    redis_result = await _check_redis(
        settings_obj, deadline=deadline, logger=logger, disabled=redis_disabled
    )
    results.append(redis_result)

    # Parallel checks (bounded by remaining time budget)
    remaining = max(0.0, deadline - time.monotonic())
    if remaining > 0:
        tasks: dict[str, asyncio.Task[StartupCheck]] = {}
        try:
            async with asyncio.timeout(remaining):
                try:
                    async with asyncio.TaskGroup() as tg:
                        tasks["tws"] = tg.create_task(
                            _check_tws(
                                settings_obj,
                                critical=getattr(settings_obj, "require_tws_at_boot", False),
                                deadline=deadline,
                                policy=policy,
                            ),
                            name="tws_check"
                        )
                        tasks["llm"] = tg.create_task(
                            _check_llm(
                                settings_obj,
                                critical=getattr(settings_obj, "require_llm_at_boot", False),
                                deadline=deadline,
                                policy=policy,
                            ),
                            name="llm_check"
                        )
                        tasks["rag"] = tg.create_task(
                            _check_rag(
                                settings_obj,
                                critical=getattr(settings_obj, "require_rag_at_boot", False),
                                deadline=deadline,
                                policy=policy,
                            ),
                            name="rag_check"
                        )
                except* asyncio.CancelledError:
                    raise
                except* Exception:
                    # Results will be extracted from tasks below
                    pass
        except TimeoutError:
            # Handle timeout for the entire check group
            pass
        except Exception:
            # Fallback for unexpected global failures
            pass

        now = time.monotonic()
        elapsed_ms = int((now - started_at) * 1000)

        for name, key in [
            ("tws_reachability", "tws"),
            ("llm_service", "llm"),
            ("rag_service", "rag"),
        ]:
            task = tasks.get(key)
            if task and task.done() and not task.cancelled() and task.exception() is None:
                results.append(task.result())
            else:
                is_critical = getattr(settings_obj, f"require_{key}_at_boot", False)
                results.append(
                    StartupCheck(
                        name=name,
                        status="fail" if is_critical else "skipped",
                        critical=is_critical,
                        reason_code="startup_budget_exhausted" if now >= deadline else "task_failed",
                        detail=f"Budget exhausted or task failed during {key.upper()} check",
                        duration_ms=elapsed_ms,
                    )
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle with proper startup and shutdown.

    Consolidates logic previously in ApplicationFactory for better separation
    of concerns.
    """
    logger = get_logger("resync.lifespan")
    logger.info("application_startup_initiated")

    # Initialize readiness event for optional parallel services
    app.state.singletons_ready_event = asyncio.Event()

    # Cache VERSION_TEAMS_WEBHOOK at startup (read once, use many times)
    # This avoids reading from disk on every request
    _version_file = Path(__file__).parent.parent / "VERSION_TEAMS_WEBHOOK.txt"
    try:
        if _version_file.exists():
            app.state.teams_webhook_version = _version_file.read_text().strip()
            logger.info(
                "teams_webhook_version_loaded", version=app.state.teams_webhook_version
            )
        else:
            app.state.teams_webhook_version = "unknown"
            logger.warning("teams_webhook_version_file_not_found")
    except Exception as e:
        app.state.teams_webhook_version = "error"
        logger.error("teams_webhook_version_load_error", error=str(e))

    try:
        settings = get_settings()
        startup_timeout = getattr(settings, "startup_timeout", 60)

        async with asyncio.timeout(startup_timeout):
            # 1. Canonical startup validation/health checks
            startup_result = await run_startup_checks(settings=settings)
            if not startup_result.get("overall_health"):
                logger.warning(
                    "startup_health_failed",
                    critical_services=startup_result.get("critical_services"),
                    results=startup_result.get("results"),
                    strict=startup_result.get("strict"),
                )
            enforce_startup_policy(startup_result)

            # 2. Core initialization (failures here are fatal)
            from resync.api_gateway.container import setup_dependencies
            from resync.core.tws_monitor import get_tws_monitor
            from resync.core.types.app_state import enterprise_state_from_app
            from resync.core.wiring import init_domain_singletons

            await init_domain_singletons(app)
            st = enterprise_state_from_app(app)

            # Record startup time for admin dashboard uptime display
            from resync.core.startup_time import set_startup_time
            set_startup_time()

            async with asyncio.TaskGroup() as bg_tasks:
                app.state.bg_tasks = bg_tasks

                # Move TWS monitor initialization here to use the bg_tasks group
                await get_tws_monitor(st.tws_client, tg=bg_tasks)
                setup_dependencies(st.tws_client, st.agent_manager, st.knowledge_graph)

                # Signal that core singletons are ready for optional services
                app.state.singletons_ready_event.set()
                logger.info("core_services_initialized")

                # 3. Optional services...
                optional_timeout = max(0.5, min(3.0, float(startup_timeout) / 2.0))
                optional_results: list[Exception] = []
                try:
                    async with asyncio.timeout(optional_timeout):
                        # Scoped TG for initialization; some of these might spawn background tasks in bg_tasks
                        try:
                            async with asyncio.TaskGroup() as init_tg:
                                init_tg.create_task(_init_proactive_monitoring(app, bg_tasks))
                                init_tg.create_task(_init_metrics_collector(app))
                                init_tg.create_task(_init_cache_warmup(app))
                                init_tg.create_task(_init_graphrag(app))
                                init_tg.create_task(_init_config_system(app))
                                init_tg.create_task(_init_enterprise_systems(app, bg_tasks))
                                init_tg.create_task(_init_health_monitoring(app, bg_tasks))
                                init_tg.create_task(_init_backup_scheduler(app, bg_tasks))
                                init_tg.create_task(_init_security_dashboard(app, bg_tasks))
                                init_tg.create_task(_init_event_bus(app, bg_tasks))
                                init_tg.create_task(_init_service_discovery(app, bg_tasks))
                        except* asyncio.CancelledError:
                            raise
                        except* Exception:
                            # Log and continue; these are optional
                            get_logger("resync.startup").warning("optional_services_init_partial_failure")
                except TimeoutError:
                    get_logger("resync.startup").warning("optional_services_init_timeout")
                except Exception as e:
                    optional_results.append(e)

                # Re-raise critical programming errors
                for res in optional_results:
                    if isinstance(res, (TypeError, KeyError, AttributeError, IndexError, NameError)):
                        logger.critical("critical_startup_programming_error", error=str(res), type=type(res).__name__)
                        raise res

                logger.info("application_startup_completed")
                st.startup_complete = True

                yield  # Application runs here, bg_tasks are active

        # When lifespan exits, bg_tasks (TaskGroup) will automatically cancel and wait for tasks.

    except TimeoutError:
        logger.critical(
            "application_startup_timeout",
            timeout_seconds=startup_timeout,
            hint=f"Startup exceeded {startup_timeout}s. Check Redis/DB connectivity and network firewalls.",
            troubleshooting={
                "redis": "Verify REDIS_URL and Redis server status.",
                "database": "Check DATABASE_URL and PostgreSQL availability.",
                "network": "Ensure no firewalls block outbound TWS/LLM connections.",
            },
        )
        raise ConfigurationError(
            f"Application startup exceeded {startup_timeout}s timeout"
        )
    except Exception as exc:
        if not isinstance(exc, ConfigurationError):
            logger.critical("application_startup_failed", error=str(exc))
        raise
    finally:
        await _shutdown_services(app)


# --- Helper methods for optional services (moved from ApplicationFactory) ---


async def _init_proactive_monitoring(app: FastAPI, bg_tasks: asyncio.TaskGroup | None = None) -> None:
    try:
        async with asyncio.timeout(10):
            # Wait for core singletons to be ready
            await app.state.singletons_ready_event.wait()

            from resync.core.monitoring_integration import (
                initialize_proactive_monitoring,
            )

            await initialize_proactive_monitoring(app, tg=bg_tasks)
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning(
            "proactive_monitoring_init_failed", error=str(exc)
        )


async def _init_metrics_collector(app: FastAPI) -> None:
    try:
        async with asyncio.timeout(10):
            # Wait for core singletons to be ready
            await app.state.singletons_ready_event.wait()

            from resync.api import monitoring_dashboard

            # Use global TaskGroup if available
            bg_tasks = getattr(app.state, "bg_tasks", None)
            if bg_tasks:
                monitoring_dashboard._collector_task = bg_tasks.create_task(
                    monitoring_dashboard.metrics_collector_loop(), name="metrics-collector"
                )
            else:
                from resync.core.task_tracker import create_tracked_task
                monitoring_dashboard._collector_task = await create_tracked_task(
                    monitoring_dashboard.metrics_collector_loop(), name="metrics-collector"
                )
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning(
            "metrics_collector_start_failed", error=str(exc)
        )


async def _init_cache_warmup(app: FastAPI) -> None:
    try:
        async with asyncio.timeout(10):
            # Wait for core singletons to be ready
            await app.state.singletons_ready_event.wait()

            from resync.core.cache_utils import warmup_cache_on_startup

            await warmup_cache_on_startup()
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("cache_warming_failed", error=str(exc))


async def _init_graphrag(app: FastAPI) -> None:
    try:
        async with asyncio.timeout(15):
            # Wait for core singletons to be ready
            await app.state.singletons_ready_event.wait()

            settings = get_settings()
            graphrag_enabled = getattr(settings, "GRAPHRAG_ENABLED", False)
            if not graphrag_enabled:
                return

            from resync.core.graphrag_integration import initialize_graphrag
            from resync.core.redis_init import get_redis_client, is_redis_available
            from resync.knowledge.retrieval.graph import get_knowledge_graph
            from resync.services.llm_service import get_llm_service
            from resync.services.tws_service import get_tws_client
            initialize_graphrag(
                llm_service=await get_llm_service(),
                knowledge_graph=get_knowledge_graph(),
                tws_client=get_tws_client(),
                redis_client=get_redis_client() if is_redis_available() else None,
                enabled=True,
            )
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, IndexError)):
            raise
        get_logger("resync.startup").warning(
            "graphrag_initialization_failed", error=str(exc)
        )


async def _init_config_system(app: FastAPI) -> None:
    try:
        async with asyncio.timeout(10):
            # Wait for core singletons to be ready
            await app.state.singletons_ready_event.wait()

            from resync.core.unified_config import initialize_config_system

            await initialize_config_system()
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning(
            "unified_config_initialization_failed", error=str(exc)
        )


async def _init_enterprise_systems(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(15):
            await app.state.singletons_ready_event.wait()
            from resync.core.enterprise.manager import get_enterprise_manager
            manager = get_enterprise_manager()
            await manager.initialize(tg=bg_tasks)
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("enterprise_systems_init_failed", error=str(exc))


async def _init_health_monitoring(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(10):
            await app.state.singletons_ready_event.wait()
            from resync.core.health import get_unified_health_service
            service = get_unified_health_service()
            service.start_monitoring(tg=bg_tasks)
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("health_monitoring_init_failed", error=str(exc))


async def _init_backup_scheduler(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(10):
            await app.state.singletons_ready_event.wait()
            from resync.core.backup.backup_service import get_backup_service
            service = get_backup_service()
            service.start_scheduler(tg=bg_tasks)
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("backup_scheduler_init_failed", error=str(exc))


async def _init_security_dashboard(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(10):
            await app.state.singletons_ready_event.wait()
            from resync.core.security_dashboard import get_security_dashboard
            _dashboard = get_security_dashboard(tg=bg_tasks)
            # dashboard is automatically started via get_security_dashboard and its lazy init
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("security_dashboard_init_failed", error=str(exc))


async def _init_event_bus(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(10):
            await app.state.singletons_ready_event.wait()
            from resync.core.event_bus import get_event_bus
            bus = get_event_bus()
            bus.start(tg=bg_tasks)
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("event_bus_init_failed", error=str(exc))


async def _init_service_discovery(app: FastAPI, bg_tasks: asyncio.TaskGroup) -> None:
    try:
        async with asyncio.timeout(15):
            await app.state.singletons_ready_event.wait()
            from resync.core.service_discovery import get_service_discovery_manager
            _manager = get_service_discovery_manager(tg=bg_tasks)
            # manager is automatically started via get_service_discovery_manager and its lazy init
    except Exception as exc:
        if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        get_logger("resync.startup").warning("service_discovery_init_failed", error=str(exc))


async def _shutdown_services(app: FastAPI) -> None:
    """Shutdown services in correct order with individual timeouts.

    Order:
    1. Cancel background tasks first (prevents tasks from using closing resources)
    2. Shutdown domain singletons and TWS monitor in parallel (independent)

    Each step has individual timeout to prevent hanging.
    """
    logger = get_logger("resync.lifespan")
    logger.info("application_shutdown_initiated")

    # Imports inside to avoid early circular deps
    from resync.core.task_tracker import cancel_all_tasks
    from resync.core.tws_monitor import shutdown_tws_monitor
    from resync.core.wiring import shutdown_domain_singletons

    # 1. Cancel background tasks FIRST (priority: stop active work)
    async def _cancel_tasks():
        try:
            await asyncio.wait_for(cancel_all_tasks(timeout=5.0), timeout=7.0)
            logger.info("background_tasks_cancelled")
        except TimeoutError:
            logger.warning(
                "task_cancel_timeout", hint="Some tasks did not cancel within 7s"
            )
        except Exception as e:
            logger.warning("task_cancel_error", error=str(e))

    # 2. Shutdown domain singletons (connections, clients)
    async def _shutdown_singletons():
        try:
            await asyncio.wait_for(shutdown_domain_singletons(app), timeout=10.0)
            logger.info("domain_singletons_shutdown")
        except TimeoutError:
            logger.error(
                "singleton_shutdown_timeout", hint="Singleton shutdown exceeded 10s"
            )
        except Exception as e:
            logger.error("domain_shutdown_error", error=str(e))

    # 3. Shutdown TWS monitor
    async def _shutdown_tws():
        try:
            await asyncio.wait_for(shutdown_tws_monitor(), timeout=5.0)
            logger.info("tws_monitor_shutdown")
        except TimeoutError:
            logger.warning("tws_monitor_shutdown_timeout")
        except Exception as e:
            logger.warning("tws_monitor_shutdown_error", error=str(e))

    # 4. Shutdown Health Service
    async def _shutdown_health():
        from resync.core.health import shutdown_unified_health_service

        try:
            await asyncio.wait_for(shutdown_unified_health_service(), timeout=5.0)
            logger.info("health_service_shutdown")
        except Exception as e:
            logger.warning("health_service_shutdown_error", error=str(e))

    # Execute: tasks first, then resources in parallel
    await _cancel_tasks()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_shutdown_singletons())
            tg.create_task(_shutdown_tws())
            tg.create_task(_shutdown_health())
    except* Exception:
        pass

    logger.info("application_shutdown_completed")
