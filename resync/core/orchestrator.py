# pylint: disable=all
"""
Service Orchestrator (v6.1.2) — parallel fan-out/fan-in for multiple backend services.

Coordinates calls to TWS, Knowledge Graph and other services with:
- Automatic parallelisation of independent calls via ``asyncio.gather``
- Unified retry with exponential backoff + jitter
- Per-call timeout (prevents one slow service from consuming the full budget)
- Global orchestration timeout
- Graceful partial-failure handling (results are collected even when some fail)
- Dynamic ``success_rate`` based on actually-executed tasks

Note:
    Circuit-breaker per service is **not** implemented yet.  If needed,
    consider ``aiocircuitbreaker`` or a custom state machine per dependency.
"""

import asyncio
import random
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resync.core.structured_logger import get_logger

if TYPE_CHECKING:
    from resync.core.interfaces import IKnowledgeGraph, ITWSClient

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default maximum number of retry attempts per service call.
_DEFAULT_MAX_RETRIES: int = 2

#: Default global orchestration timeout (seconds).
_DEFAULT_TIMEOUT_SECONDS: int = 10

#: Default per-call timeout (seconds).  Must be less than global timeout.
_DEFAULT_PER_CALL_TIMEOUT: float = 8.0

#: Default number of log tail lines to fetch.
_DEFAULT_LOG_LINES: int = 100

#: Window (hours) for querying recently-failed jobs.
_FAILED_JOBS_HOURS_WINDOW: int = 24

#: Maximum backoff delay (seconds) to cap exponential growth.
_MAX_BACKOFF_SECONDS: float = 8.0


# =============================================================================
# RESULT MODEL
# =============================================================================


@dataclass
class OrchestrationResult:
    """Result of a multi-service orchestration run.

    Attributes:
        tws_status: Job status dict from TWS (None if call failed).
        tws_logs: Tail of job logs (None if not requested or failed).
        kg_context: Relevant context from the Knowledge Graph.
        job_dependencies: Upstream/downstream dependency list.
        historical_failures: Past failures for the same job.
        errors: Mapping of ``task_name → error_message`` for failed calls.
        attempted_tasks: Number of tasks that were actually dispatched.
    """

    tws_status: dict[str, Any] | None = None
    tws_logs: str | None = None
    kg_context: str | None = None
    job_dependencies: list[dict[str, Any]] | None = None
    historical_failures: list[dict[str, Any]] | None = None
    errors: dict[str, str] = field(default_factory=dict)
    attempted_tasks: int = 0

    @property
    def is_complete(self) -> bool:
        """True when all *critical* data (status + context) was obtained."""
        return self.tws_status is not None and self.kg_context is not None

    @property
    def has_errors(self) -> bool:
        """True when at least one task failed."""
        return bool(self.errors)

    @property
    def success_rate(self) -> float:
        """Fraction of attempted tasks that succeeded (0.0-1.0).

        Computed dynamically from ``attempted_tasks`` and ``errors``, so it
        is always consistent regardless of which optional tasks were included.
        """
        if self.attempted_tasks == 0:
            return 0.0
        failed = len(self.errors)
        return (self.attempted_tasks - failed) / self.attempted_tasks


# =============================================================================
# SERVICE ORCHESTRATOR
# =============================================================================


class ServiceOrchestrator:
    """Fan-out/fan-in orchestrator for backend service calls.

    All calls are executed concurrently via ``asyncio.gather``.  Each call
    is wrapped in a unified retry helper with exponential backoff + jitter
    and an individual timeout.  A global timeout caps total wall-clock time.

    Args:
        tws_client: TWS client instance (must expose ``get_job_status``,
            ``get_job_logs``, ``get_job_dependencies``, ``query_jobs``,
            ``get_engine_info``).
        knowledge_graph: KG instance (must expose ``get_relevant_context``).
        max_retries: Maximum retry attempts per call.
        timeout_seconds: Global orchestration timeout.
        per_call_timeout: Timeout for each individual service call.
    """

    def __init__(
        self,
        tws_client: "ITWSClient",
        knowledge_graph: "IKnowledgeGraph",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        per_call_timeout: float = _DEFAULT_PER_CALL_TIMEOUT,
    ) -> None:
        self.tws = tws_client
        self.kg = knowledge_graph
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.per_call_timeout = per_call_timeout

    # -----------------------------------------------------------------
    # Unified retry helper
    # -----------------------------------------------------------------

    async def _call_with_retry(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        retries: int | None = None,
    ) -> Any:
        """Execute *coro_factory()* with retry, backoff+jitter, and per-call timeout.

        Args:
            name: Human-readable label used in logs and error messages.
            coro_factory: Zero-arg callable that returns a fresh coroutine
                on each invocation (needed because a coroutine object can
                only be awaited once).
            retries: Override for ``self.max_retries``.

        Returns:
            The result of the coroutine on success.

        Raises:
            The last exception if all attempts fail.  The caller
            (``asyncio.gather(return_exceptions=True)``) will capture it.
        """
        max_attempts = (retries if retries is not None else self.max_retries) + 1

        last_exc: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                return await asyncio.wait_for(
                    coro_factory(), timeout=self.per_call_timeout
                )
            except asyncio.CancelledError:
                # Respect cooperative cancellation — never swallow it.
                raise
            except Exception as exc:
                last_exc = exc
                if attempt == max_attempts - 1:
                    logger.error(
                        "service_call_failed",
                        task=name,
                        attempts=max_attempts,
                        error=type(exc).__name__,
                        detail="Internal server error. Check server logs for details.",
                    )
                    raise
                delay = min(2**attempt, _MAX_BACKOFF_SECONDS)
                jitter = random.uniform(0, delay * 0.5)  # noqa: S311
                logger.warning(
                    "service_call_retrying",
                    task=name,
                    attempt=attempt + 1,
                    next_delay=f"{delay + jitter:.2f}s",
                    error=str(exc),
                )
                await asyncio.sleep(delay + jitter)

        # Unreachable, but satisfies the type checker.
        raise last_exc  # type: ignore[misc]

    # -----------------------------------------------------------------
    # Public orchestration methods
    # -----------------------------------------------------------------

    async def investigate_job_failure(
        self,
        job_name: str,
        include_logs: bool = True,
        include_dependencies: bool = True,
    ) -> OrchestrationResult:
        """Investigate a job failure by querying multiple services in parallel.

        Fetches:
            1. TWS job status (with retry)
            2. Knowledge-graph context (with retry)
            3. Historical failures (with retry)
            4. Job logs (optional, with retry)
            5. Upstream/downstream dependencies (optional, with retry)

        Args:
            job_name: Name of the failed job.
            include_logs: Whether to fetch log tail.
            include_dependencies: Whether to fetch dependency graph.

        Returns:
            ``OrchestrationResult`` with partial data on failure.
        """
        result = OrchestrationResult()

        # Build task map (insertion-order stable in Python 3.7+)
        tasks: dict[str, Coroutine[Any, Any, Any]] = {
            "status": self._call_with_retry(
                "tws_job_status",
                lambda jn=job_name: self.tws.get_job_status(jn),
            ),
            "context": self._call_with_retry(
                "kg_context",
                lambda jn=job_name: self.kg.get_relevant_context(f"job failure {jn}"),
            ),
            "history": self._call_with_retry(
                "historical_failures",
                lambda jn=job_name: self._fetch_historical_failures(jn),
            ),
        }

        if include_logs:
            tasks["logs"] = self._call_with_retry(
                "tws_job_logs",
                lambda jn=job_name: self.tws.get_job_logs(jn, lines=_DEFAULT_LOG_LINES),
            )

        if include_dependencies:
            tasks["deps"] = self._call_with_retry(
                "tws_job_deps",
                lambda jn=job_name: self.tws.get_job_dependencies(jn),
            )

        result.attempted_tasks = len(tasks)

        # Fan-out with global timeout
        task_objs: dict[str, asyncio.Task] = {}

        try:
            async with asyncio.timeout(self.timeout):
                try:
                    async with asyncio.TaskGroup() as tg:
                        for name, coro in tasks.items():
                            task_objs[name] = tg.create_task(coro, name=name)
                except* asyncio.CancelledError:
                    # Garantir que cancelamento cooperative seja propagado
                    raise
                except* Exception as eg:
                    # Log detalhado, mas não interrompe o flow (partial failure)
                    logger.warning(
                        "orchestration_tasks_partial_failure",
                        job_name=job_name,
                        count=len(eg.exceptions),
                    )
        except TimeoutError:
            logger.error(
                "orchestration_timeout",
                job_name=job_name,
                timeout=self.timeout,
                attempted=result.attempted_tasks,
            )
            result.errors["_global"] = f"Orchestration timed out after {self.timeout}s"
            # Retorna resultado parcial (tasks canceladas serão tratadas no fan-in)
            return result

        # Fan-in: map results back to task names
        self._assign_results(result, task_objs)

        logger.info(
            "orchestration_complete",
            job_name=job_name,
            success_rate=f"{result.success_rate:.1%}",
            errors=len(result.errors),
            attempted=result.attempted_tasks,
        )

        return result

    async def get_system_health(self) -> dict[str, Any]:
        """Retrieve overall TWS system health in parallel.

        Checks engine status, critical-path jobs, and recently-failed jobs.

        Returns:
            Dict with ``status`` (HEALTHY / DEGRADED / ERROR) and ``details``
            per component.
        """
        tasks: dict[str, Coroutine[Any, Any, Any]] = {
            "engine": self._call_with_retry(
                "engine_status",
                lambda: self.tws.get_engine_info(),
            ),
            "critical_jobs": self._call_with_retry(
                "critical_jobs",
                lambda: self.tws.get_critical_path_status(),
                retries=0,  # fast-fail for non-essential
            ),
            "failed_jobs": self._call_with_retry(
                "failed_jobs",
                lambda: self.tws.query_jobs(
                    status="ABEND", hours=_FAILED_JOBS_HOURS_WINDOW
                ),
            ),
        }

        task_objs: dict[str, asyncio.Task] = {}
        try:
            async with asyncio.timeout(self.timeout):
                try:
                    async with asyncio.TaskGroup() as tg:
                        for name, coro in tasks.items():
                            task_objs[name] = tg.create_task(
                                coro, name=f"health_{name}"
                            )
                except* asyncio.CancelledError:
                    raise
                except* Exception:
                    # Exceptions will be caught when checking task.result()
                    pass
        except TimeoutError:
            logger.error("health_check_timeout", timeout=self.timeout)
            return {
                "status": "ERROR",
                "message": f"Health check timed out after {self.timeout}s",
            }

        health: dict[str, Any] = {"status": "HEALTHY", "details": {}}

        for task_name, task in task_objs.items():
            # Safer result extraction
            if not task.done():
                health["status"] = "DEGRADED"
                health["details"][task_name] = {"status": "TIMEOUT"}
                continue

            try:
                task_result = task.result()
                health["details"][task_name] = {
                    "status": "OK",
                    "data": task_result,
                }
            except (asyncio.CancelledError, Exception) as e:
                health["status"] = "DEGRADED"
                health["details"][task_name] = {
                    "status": "ERROR",
                    "error": str(e),
                }

        return health

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _assign_results(
        self,
        result: OrchestrationResult,
        tasks: dict[str, asyncio.Task],
    ) -> None:
        """Map task results back to ``OrchestrationResult`` fields."""
        _field_map: dict[str, str] = {
            "status": "tws_status",
            "context": "kg_context",
            "logs": "tws_logs",
            "deps": "job_dependencies",
            "history": "historical_failures",
        }

        for task_name, task in tasks.items():
            # Safer extraction: only try .result() if task is actually done
            if not task.done():
                result.errors[task_name] = "Task timed out or not completed"
                continue

            try:
                task_result = task.result()
                attr = _field_map.get(task_name)
                if attr:
                    setattr(result, attr, task_result)
            except (asyncio.CancelledError, Exception) as e:
                result.errors[task_name] = f"{type(e).__name__}: {e}"
                logger.warning(
                    "orchestration_task_failed",
                    task=task_name,
                    error=type(e).__name__,
                    detail=str(e),
                )

    async def _fetch_historical_failures(self, job_name: str) -> list[dict[str, Any]]:
        """Query KG for historical failures related to *job_name*.

        Returns a structured list (possibly empty) rather than raw text.
        """
        context = await self.kg.get_relevant_context(
            f"historical failures job {job_name}"
        )
        if context:
            return [{"summary": context}]
        return []


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "OrchestrationResult",
    "ServiceOrchestrator",
]
