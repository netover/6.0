import asyncio
import os
import sys
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
    ValkeyAuthError,
    ValkeyConnectionError,
    ValkeyInitializationError,
    ValkeyTimeoutError,
)
from resync.core.exception_guard import guard_programming_errors
from resync.core.structured_logger import get_logger
from resync.settings import Settings, get_settings

Status = Literal["ok", "fail", "skipped"]

@dataclass(frozen=True)
class StartupCheck:
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
    get_logger("resync.startup").warning("invalid_boolean_env_var", value=value)
    return None

def get_startup_policy(settings: Settings) -> dict[str, Any]:
    strict_env = _parse_bool(os.getenv("STARTUP_STRICT"))
    strict = (
        strict_env
        if strict_env is not None
        else bool(getattr(settings, "is_production", False))
    )

    timeout_default = float(getattr(settings, "startup_timeout", 30))
    max_total_seconds = float(
        os.getenv("STARTUP_MAX_TOTAL_SECONDS", str(timeout_default))
    )

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
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, AttributeError):
            pass
        _ = reader
        return True, None
    except TimeoutError:
        return False, "timeout"
    except OSError as e:
        return False, f"os_error:{type(e).__name__}"
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, ConnectionError) as e:
        return False, f"unexpected:{type(e).__name__}"

async def _http_healthy(
    url: str, timeout: float
) -> tuple[bool, str | None, int | None, str | None]:
    # Check if SSRF protection is disabled via environment variable
    ssrf_disabled = os.getenv("RESYNC_DISABLE_SSRF", "true").lower() == "true"
    
    if not ssrf_disabled:
        from resync.core.ssrf_protection import SSRFProtection
        is_safe, reason = SSRFProtection.is_safe_url(url)
        if not is_safe:
            return False, f"ssrf_blocked:{reason}", None, None

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.get(url)
            if 200 <= resp.status_code < 300:
                return True, None, resp.status_code, None
            if 300 <= resp.status_code < 400:
                loc = resp.headers.get("location")
                safe_loc = None
                if loc:
                    from urllib.parse import urlparse
                    parsed = urlparse(loc)
                    safe_loc = f"{parsed.scheme}://{parsed.netloc}{parsed.path[:32]}"
                    if len(parsed.path) > 32:
                        safe_loc += "..."
                return False, "redirect", resp.status_code, safe_loc
            return False, "http_status", resp.status_code, None
    except httpx.TimeoutException:
        return False, "timeout", None, None
    except httpx.RequestError as e:
        return False, f"request_error:{type(e).__name__}", None, None
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, ConnectionError) as e:
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
    raw_timeout = getattr(settings, "tws_timeout_connect", 3.0)
    timeout = float(raw_timeout) if raw_timeout else 3.0

    if not host or not port:
        status: Status = "fail" if critical else "skipped"
        return StartupCheck(
            name="tws_reachability",
            status=status,
            critical=critical,
            reason_code="not_configured",
            detail="TWS host/port not configured",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def attempt_fn(attempt: int) -> StartupCheck:
        attempt_start = time.monotonic()
        ok, reason = await _tcp_reachable(str(host), int(port), timeout)
        dur_ms = int((time.monotonic() - attempt_start) * 1000)
        if ok:
            return StartupCheck(
                name="tws_reachability",
                status="ok",
                critical=critical,
                reason_code="reachable",
                attempts=attempt,
                duration_ms=dur_ms,
            )
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
    timeout = float(
        os.getenv(
            "STARTUP_LLM_HEALTH_TIMEOUT",
            str(getattr(settings, "startup_llm_health_timeout", 5.0)),
        )
    )

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
        attempt_start = time.monotonic()
        ok, reason, status_code, location = await _http_healthy(str(endpoint), timeout)
        dur_ms = int((time.monotonic() - attempt_start) * 1000)
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
    timeout = float(getattr(settings, "rag_service_timeout", 5.0))

    if not url:
        status: Status = "fail" if critical else "skipped"
        return StartupCheck(
            name="rag_service",
            status=status,
            critical=critical,
            reason_code="not_configured",
            detail="RAG service URL not configured",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def attempt_fn(attempt: int) -> StartupCheck:
        attempt_start = time.monotonic()
        ok, reason, status_code, _ = await _http_healthy(str(url), timeout)
        dur_ms = int((time.monotonic() - attempt_start) * 1000)
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

async def _check_valkey(
    settings: Settings, *, deadline: float, logger: Any, disabled: bool
) -> StartupCheck:
    start = time.monotonic()
    if disabled:
        logger.info(
            "valkey_startup_disabled",
            reason="RESYNC_DISABLE_VALKEY=true, skipping valkey initialization",
        )
        return StartupCheck(
            name="valkey_connection",
            status="skipped",
            critical=False,
            reason_code="disabled",
            detail="Valkey disabled by RESYNC_DISABLE_VALKEY",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        from resync.core.valkey_init import ValkeyInitError, get_valkey_initializer
        initializer = get_valkey_initializer()
        with guard_programming_errors():
            await initializer.initialize(
                max_retries=int(getattr(settings, "valkey_max_startup_retries", 5)),
                health_check_interval=int(
                    getattr(settings, "valkey_health_check_interval", 5)
                ),
                valkey_url=(
                    getattr(settings, "valkey_url", None)
                ),
            )
        return StartupCheck(
            name="valkey_connection",
            status="ok",
            critical=True,
            reason_code="initialized",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    except ConfigurationError as e:
        logger.error("startup_valkey_configuration_error", error_message=str(e))
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code="configuration_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except ValkeyAuthError as e:
        logger.error(
            "startup_valkey_authentication_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code="valkey_auth_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except ValkeyTimeoutError as e:
        logger.error(
            "startup_valkey_timeout_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code="valkey_timeout",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except ValkeyConnectionError as e:
        logger.error(
            "startup_valkey_connection_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code="valkey_connection_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except ValkeyInitializationError as e:
        logger.error(
            "startup_valkey_initialization_error",
            error_message=str(e),
            error_details=getattr(e, "details", None),
        )
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code="valkey_initialization_error",
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
        if isinstance(e, ValkeyInitError):
            reason = "valkey_init_error"
        else:
            reason = (
                "startup_budget_exhausted"
                if deadline and time.monotonic() >= deadline
                else type(e).__name__
            )
        logger.error(
            "startup_valkey_unexpected_error",
            error_message=str(e),
            reason_code=reason,
        )
        return StartupCheck(
            name="valkey_connection",
            status="fail",
            critical=True,
            reason_code=reason,
            detail=str(e),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

async def run_startup_checks(*, settings: Settings | None = None) -> dict[str, Any]:
    settings_obj = settings or get_settings()
    logger = get_logger("resync.startup")
    policy = get_startup_policy(settings_obj)

    started_at = time.monotonic()
    deadline = started_at + float(policy["max_total_seconds"])

    valkey_disabled = _parse_bool(os.getenv("RESYNC_DISABLE_VALKEY")) or False
    if valkey_disabled:
        logger.warning("valkey_marked_noncritical", reason="RESYNC_DISABLE_VALKEY=true")

    critical_services: list[str] = [] if valkey_disabled else ["valkey_connection"]
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

    results: list[StartupCheck] = []

    valkey_result = await _check_valkey(
        settings_obj, deadline=deadline, logger=logger, disabled=valkey_disabled
    )
    results.append(valkey_result)

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
                            name="tws_check",
                        )
                        tasks["llm"] = tg.create_task(
                            _check_llm(
                                settings_obj,
                                critical=getattr(settings_obj, "require_llm_at_boot", False),
                                deadline=deadline,
                                policy=policy,
                            ),
                            name="llm_check",
                        )
                        tasks["rag"] = tg.create_task(
                            _check_rag(
                                settings_obj,
                                critical=getattr(settings_obj, "require_rag_at_boot", False),
                                deadline=deadline,
                                policy=policy,
                            ),
                            name="rag_check",
                        )
                except* asyncio.CancelledError:
                    raise
                except* Exception as exc_group:
                    for exc in exc_group.exceptions:
                        logger.warning(
                            "startup_check_task_group_error",
                            error=str(exc),
                            type=type(exc).__name__,
                            exc_info=exc,
                        )
        except TimeoutError:
            logger.warning("startup_checks_total_timeout")
        except (OSError, ValueError, RuntimeError, ConnectionError) as exc:
            _exc_type, _exc, _tb = sys.exc_info()
            from resync.core.exception_guard import maybe_reraise_programming_error
            maybe_reraise_programming_error(_exc, _tb)
            logger.error("startup_checks_unexpected_error", error=str(exc))

        now = time.monotonic()
        elapsed_ms = int((now - started_at) * 1000)

        for name, key in [
            ("tws_reachability", "tws"),
            ("llm_service", "llm"),
            ("rag_service", "rag"),
        ]:
            task = tasks.get(key)
            if (
                task
                and task.done()
                and not task.cancelled()
                and task.exception() is None
            ):
                results.append(task.result())
            else:
                is_critical = getattr(settings_obj, f"require_{key}_at_boot", False)
                results.append(
                    StartupCheck(
                        name=name,
                        status="fail" if is_critical else "skipped",
                        critical=is_critical,
                        reason_code="startup_budget_exhausted"
                        if now >= deadline
                        else "task_failed",
                        detail=(
                            "Budget exhausted or task failed during "
                            f"{key.upper()} check"
                        ),
                        duration_ms=elapsed_ms,
                    )
                )

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
    if result.get("strict") and not result.get("overall_health"):
        raise SystemExit(1)

@asynccontextmanager
async def lifespan(app: "FastAPI") -> AsyncIterator[None]:
    """Manage application lifecycle with proper startup and shutdown."""
    logger = get_logger("resync.lifespan")
    logger.info("application_startup_initiated")

    # -------------------------------------------------------------------------
    # Sentry SDK — initialize early so all subsequent exceptions are captured.
    # -------------------------------------------------------------------------
    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        settings_for_sentry = get_settings()
        sentry_dsn_secret = getattr(settings_for_sentry, "sentry_dsn", None)
        sentry_dsn = (
            sentry_dsn_secret.get_secret_value()
            if sentry_dsn_secret is not None
            else None
        )
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[
                    StarletteIntegration(transaction_style="endpoint"),
                    FastApiIntegration(transaction_style="endpoint"),
                    AsyncioIntegration(),
                ],
                traces_sample_rate=getattr(settings_for_sentry, "sentry_traces_sample_rate", 0.1),
                profiles_sample_rate=getattr(settings_for_sentry, "sentry_profiles_sample_rate", 0.0),
                environment=(
                    getattr(settings_for_sentry, "environment", "development").value
                    if hasattr(getattr(settings_for_sentry, "environment", ""), "value")
                    else str(getattr(settings_for_sentry, "environment", "development"))
                ),
                release=getattr(settings_for_sentry, "project_version", "unknown"),
                send_default_pii=False,
            )
            logger.info(
                "sentry_initialized",
                dsn_host=sentry_dsn.split("@")[-1].split("/")[0]
                if "@" in sentry_dsn
                else "configured",
            )
        else:
            logger.info("sentry_disabled_no_dsn_configured")
    except ImportError:
        logger.info(
            "sentry_sdk_not_installed",
            hint="Install sentry-sdk[fastapi] to enable error tracking.",
        )
    except (OSError, ValueError, RuntimeError, ConnectionError) as _sentry_err:
        # FIX-01 + FIX-06: top-level sys import, TimeoutError removed from tuple.
        _exc_type, _exc, _tb = sys.exc_info()
        from resync.core.exception_guard import maybe_reraise_programming_error
        maybe_reraise_programming_error(_exc, _tb)
        logger.warning("sentry_init_failed", error=str(_sentry_err))

    app.state.singletons_ready_event = asyncio.Event()

    # FIX-04: I/O-blocking Path.exists() / read_text() offloaded to thread.
    _version_file = Path(__file__).parent.parent / "VERSION_TEAMS_WEBHOOK.txt"
    try:
        exists = await asyncio.to_thread(_version_file.exists)
        if exists:
            raw = await asyncio.to_thread(_version_file.read_text)
            app.state.teams_webhook_version = raw.strip()
            logger.info(
                "teams_webhook_version_loaded",
                version=app.state.teams_webhook_version,
            )
        else:
            app.state.teams_webhook_version = "unknown"
            logger.warning("teams_webhook_version_file_not_found")
    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
        # FIX-01 + FIX-06: top-level sys, TimeoutError removed.
        _exc_type, _exc, _tb = sys.exc_info()
        from resync.core.exception_guard import maybe_reraise_programming_error
        maybe_reraise_programming_error(_exc, _tb)
        app.state.teams_webhook_version = "error"
        logger.error("teams_webhook_version_load_error", error=str(e))

    try:
        settings = get_settings()
        startup_timeout = getattr(settings, "startup_timeout", 60)

        # [FIX BUG #4] Use single timeout layer - simpler and more predictable
        async with asyncio.timeout(float(startup_timeout)):
            startup_result = await run_startup_checks(settings=settings)
            if not startup_result.get("overall_health"):
                logger.warning(
                    "startup_health_failed",
                    critical_services=startup_result.get("critical_services"),
                    results=startup_result.get("results"),
                    strict=startup_result.get("strict"),
                )
            enforce_startup_policy(startup_result)

            from resync.api_gateway.container import setup_dependencies
            from resync.core.tws_monitor import get_tws_monitor
            from resync.core.types.app_state import enterprise_state_from_app
            from resync.core.wiring import init_domain_singletons

            await init_domain_singletons(app)
            st = enterprise_state_from_app(app)

            from resync.core.startup_time import set_startup_time
            set_startup_time()

            from resync.core.bg_tasks import ManagedTaskGroup
            async with asyncio.TaskGroup() as _tg:
                bg_tasks = ManagedTaskGroup(_tg)
                # FIX (Bug #10): assign before any optional service accesses it.
                app.state.bg_tasks = bg_tasks

                await get_tws_monitor(st.tws_client, tg=bg_tasks)
                setup_dependencies(st.tws_client, st.agent_manager, st.knowledge_graph)

                app.state.singletons_ready_event.set()
                logger.info("core_services_initialized")

                # [P2 FIX] Use settings field directly instead of getattr with default
                enable_optional_services = settings.startup_enable_optional_services is True
                optional_timeout = max(1.0, min(5.0, float(startup_timeout) * 0.4))

                # FIX (Bug #5): list is now read and logged after init loop.
                optional_results: list[Exception] = []

                if enable_optional_services:
                    try:
                        async with asyncio.timeout(optional_timeout):
                            # FIX-03: except* blocks are OUTSIDE the async with,
                            # not nested inside it (was SyntaxError).
                            try:
                                async with asyncio.TaskGroup() as init_tg:
                                    init_tg.create_task(
                                        _init_proactive_monitoring(app, bg_tasks)
                                    )
                                    # [P2 FIX] Only start metrics collector if explicitly enabled
                                    if settings.startup_enable_metrics_collector is True:
                                        init_tg.create_task(_init_metrics_collector(app))
                                    init_tg.create_task(_init_cache_warmup(app))
                                    init_tg.create_task(_init_graphrag(app))
                                    init_tg.create_task(_init_config_system(app))
                                    init_tg.create_task(
                                        _init_enterprise_systems(app, bg_tasks)
                                    )
                                    init_tg.create_task(
                                        _init_health_monitoring(app, bg_tasks)
                                    )
                                    init_tg.create_task(
                                        _init_backup_scheduler(app, bg_tasks)
                                    )
                                    init_tg.create_task(
                                        _init_security_dashboard(app, bg_tasks)
                                    )
                                    init_tg.create_task(_init_event_bus(app, bg_tasks))
                                    init_tg.create_task(
                                        _init_service_discovery(app, bg_tasks)
                                    )
                            except* asyncio.CancelledError:
                                raise
                            except* Exception as exc_group:
                                for exc in exc_group.exceptions:
                                    optional_results.append(exc)
                                    get_logger("resync.startup").warning(
                                        "optional_service_init_failed",
                                        error=str(exc),
                                        type=type(exc).__name__,
                                    )

                        # FIX (Bug #5): log summary of optional failures.
                        if optional_results:
                            get_logger("resync.startup").warning(
                                "optional_services_init_summary",
                                failed_count=len(optional_results),
                                failed_types=[type(e).__name__ for e in optional_results],
                            )
                    except TimeoutError:
                        get_logger("resync.startup").warning(
                            "optional_services_init_timeout"
                        )
                    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
                        # FIX-06: TimeoutError removed from tuple.
                        optional_results.append(e)
                else:
                    logger.info("optional_services_startup_skipped")

                logger.info("application_startup_completed")
                st.startup_complete = True

                try:
                    yield
                except* Exception as eg:
                    for exc in eg.exceptions:
                        logger.critical(
                            "bg_task_crashed",
                            error=str(exc),
                            type=type(exc).__name__,
                        )
                finally:
                    # Ensure background tasks are cancelled on shutdown.
                    shutdown_timeout = float(
                        getattr(settings, "graceful_shutdown_timeout_seconds", 10.0)
                    )
                    try:
                        async with asyncio.timeout(shutdown_timeout):
                            await bg_tasks.cancel_all()
                    except TimeoutError:
                        logger.warning(
                            "bg_tasks_cancel_timeout",
                            timeout_seconds=shutdown_timeout,
                        )

    except TimeoutError:
        logger.critical(
            "application_startup_timeout",
            timeout_seconds=startup_timeout,
            hint=f"Startup exceeded {startup_timeout}s. Check Valkey/DB/networking.",
            troubleshooting={
                "valkey": "Verify VALKEY_URL and Valkey server status.",
                "database": "Check DATABASE_URL and PostgreSQL availability.",
                "network": "Ensure no firewalls block outbound TWS/LLM connections.",
            },
        )
        raise ConfigurationError(
            f"Application startup exceeded {startup_timeout}s timeout"
        )
    except (OSError, ValueError, RuntimeError, ConnectionError, ConfigurationError) as exc:
        # [FIX BUG #5] ConfigurationError now caught, isinstance check removed (was dead code)
        _exc_type, _exc, _tb = sys.exc_info()
        from resync.core.exception_guard import maybe_reraise_programming_error
        maybe_reraise_programming_error(_exc, _tb)
        logger.critical(
            "application_startup_failed",
            error=str(exc),
            exc_type=type(exc).__name__,
        )
        raise
    finally:
        await _shutdown_services(app)


# ---------------------------------------------------------------------------
# Optional service initializers
# ---------------------------------------------------------------------------

async def _init_proactive_monitoring(
    app: "FastAPI", bg_tasks: asyncio.TaskGroup | None = None
) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.monitoring_integration import initialize_proactive_monitoring
                await initialize_proactive_monitoring(app, tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "proactive_monitoring_init_failed", error=str(exc)
        )


async def _init_metrics_collector(app: "FastAPI") -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.api import monitoring_dashboard
                bg_tasks = getattr(app.state, "bg_tasks", None)
                if bg_tasks:
                    monitoring_dashboard._collector_task = bg_tasks.create_task(
                        monitoring_dashboard.metrics_collector_loop(),
                        name="metrics-collector",
                    )
                else:
                    from resync.core.task_tracker import create_tracked_task
                    monitoring_dashboard._collector_task = create_tracked_task(
                        monitoring_dashboard.metrics_collector_loop(),
                        name="metrics-collector",
                    )
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "metrics_collector_start_failed", error=str(exc)
        )


async def _init_cache_warmup(app: "FastAPI") -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.cache_utils import warmup_cache_on_startup
                await warmup_cache_on_startup()
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning("cache_warming_failed", error=str(exc))


async def _init_graphrag(app: "FastAPI") -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(15):
                await app.state.singletons_ready_event.wait()
                settings = get_settings()
                # FIX-07: snake_case consistent with rest of Settings access.
                graphrag_enabled = getattr(settings, "graphrag_enabled", False)
                if not graphrag_enabled:
                    return
                from resync.core.graphrag_integration import initialize_graphrag
                from resync.core.valkey_init import get_valkey_client, is_valkey_available
                from resync.knowledge.retrieval.graph import get_knowledge_graph
                from resync.services.llm_service import get_llm_service
                from resync.services.tws_service import get_tws_client
                initialize_graphrag(
                    llm_service=await get_llm_service(),
                    knowledge_graph=get_knowledge_graph(),
                    tws_client=get_tws_client(),
                    valkey_client=get_valkey_client() if is_valkey_available() else None,
                    enabled=True,
                )
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "graphrag_initialization_failed", error=str(exc)
        )


async def _init_config_system(app: "FastAPI") -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.unified_config import initialize_config_system
                await initialize_config_system()
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "unified_config_initialization_failed", error=str(exc)
        )


async def _init_enterprise_systems(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(15):
                await app.state.singletons_ready_event.wait()
                from resync.core.enterprise.manager import get_enterprise_manager
                manager = await get_enterprise_manager()
                await manager.initialize(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "enterprise_systems_init_failed", error=str(exc)
        )


async def _init_health_monitoring(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.health import get_unified_health_service
                service = await get_unified_health_service()
                service.start_monitoring(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "health_monitoring_init_failed", error=str(exc)
        )


async def _init_backup_scheduler(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.backup.backup_service import get_backup_service
                service = get_backup_service()
                service.start_scheduler(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "backup_scheduler_init_failed", error=str(exc)
        )


async def _init_security_dashboard(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.security_dashboard import get_security_dashboard
                await get_security_dashboard(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "security_dashboard_init_failed", error=str(exc)
        )


async def _init_event_bus(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(10):
                await app.state.singletons_ready_event.wait()
                from resync.core.event_bus import get_event_bus
                bus = get_event_bus()
                bus.start(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning("event_bus_init_failed", error=str(exc))


async def _init_service_discovery(app: "FastAPI", bg_tasks: asyncio.TaskGroup) -> None:
    try:
        with guard_programming_errors():
            async with asyncio.timeout(15):
                await app.state.singletons_ready_event.wait()
                from resync.core.service_discovery import get_service_discovery_manager
                get_service_discovery_manager(tg=bg_tasks)
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
        get_logger("resync.startup").warning(
            "service_discovery_init_failed", error=str(exc)
        )


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

async def _shutdown_services(app: "FastAPI") -> None:
    """Shutdown services in correct order with individual timeouts."""
    logger = get_logger("resync.lifespan")
    logger.info("application_shutdown_initiated")

    from resync.core.task_tracker import cancel_all_tasks
    from resync.core.tws_monitor import shutdown_tws_monitor
    from resync.core.wiring import shutdown_domain_singletons

    async def _cancel_tasks() -> None:
        try:
            with guard_programming_errors():
                await asyncio.wait_for(cancel_all_tasks(timeout=5.0), timeout=7.0)
                logger.info("background_tasks_cancelled")
        except TimeoutError:
            logger.warning("task_cancel_timeout")
        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.warning("task_cancel_error", error=str(e))

    async def _cancel_bg_tasks() -> None:
        """Cancel tasks in the bg_tasks manager if it exists.

        Avoids accessing asyncio.TaskGroup private attributes (e.g. _tasks).
        """
        bg_tasks = getattr(app.state, "bg_tasks", None)
        if bg_tasks is None:
            return
        try:
            with guard_programming_errors():
                cancel_all = getattr(bg_tasks, "cancel_all", None)
                if callable(cancel_all):
                    await cancel_all()
                else:
                    # Best-effort: if it's a raw TaskGroup or other object, we can't reliably
                    # enumerate tasks without private APIs.
                    logger.warning("bg_tasks_cancel_skipped_unmanaged_type", type=str(type(bg_tasks)))
                logger.info("bg_tasks_cancelled")
        except TimeoutError:
            logger.warning("bg_tasks_cancel_timeout")
        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.warning("bg_tasks_cancel_error", error=str(e))

    async def _shutdown_singletons() -> None:
        try:
            with guard_programming_errors():
                await asyncio.wait_for(shutdown_domain_singletons(app), timeout=10.0)
                logger.info("domain_singletons_shutdown")
        except TimeoutError:
            logger.error("singleton_shutdown_timeout")
        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.error("domain_shutdown_error", error=str(e))

    async def _shutdown_tws() -> None:
        try:
            with guard_programming_errors():
                await asyncio.wait_for(shutdown_tws_monitor(), timeout=5.0)
                logger.info("tws_monitor_shutdown")
        except TimeoutError:
            logger.warning("tws_monitor_shutdown_timeout")
        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.warning("tws_monitor_shutdown_error", error=str(e))

    async def _shutdown_health() -> None:
        from resync.core.health import shutdown_unified_health_service
        try:
            with guard_programming_errors():
                await asyncio.wait_for(shutdown_unified_health_service(), timeout=5.0)
                logger.info("health_service_shutdown")
        except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("health_service_shutdown_error", error=str(e))

    # Cancel bg_tasks first to stop health monitoring and other background tasks
    await _cancel_bg_tasks()
    await _cancel_tasks()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_shutdown_singletons())
            tg.create_task(_shutdown_tws())
            tg.create_task(_shutdown_health())
    except* Exception as exc_group:
        logger.error(
            "shutdown_task_group_error",
            errors=[str(e) for e in exc_group.exceptions],
        )

    logger.info("application_shutdown_completed")
