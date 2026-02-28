"""
Background Task Tracking System

Prevents task leaks and ensures proper cleanup on shutdown.

Key guarantees:
- Tasks can be created only when a running event loop exists (fail-fast).
- Tracked tasks are cancelled on shutdown with bounded waiting (two-phase).
- Unhandled task exceptions are surfaced (logged) to avoid silent failures.
- Optional backpressure cap avoids runaway spawning / DoS-by-tasks.

Integration notes:
- [DEPRECATED] Internal components should use `asyncio.TaskGroup`
  for structured concurrency.
- Intended for legacy FastAPI lifespan, request, websocket handlers, etc.
- For cross-thread scheduling, use create_tracked_task_threadsafe().
"""

import asyncio
import functools
import logging
import os
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, cast

from resync.core.utils.async_bridge import run_sync

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HARDENING [P0]: Strict lock required for free-threaded Python (3.13/3.14+ variants)
# where Python code may execute concurrently in multiple OS threads.
_tasks_lock = threading.Lock()
_background_tasks: set[asyncio.Task[Any]] = set()

# Backpressure cap (configurable).
# Keep it high by default to avoid breaking legitimate workloads, but bounded.
MAX_BACKGROUND_TASKS: int = int(os.getenv("MAX_BACKGROUND_TASKS", "10000"))

# Default shutdown timeout (seconds).
DEFAULT_SHUTDOWN_TIMEOUT: float = float(
    os.getenv("TASK_TRACKER_SHUTDOWN_TIMEOUT", "5.0")
)

def _require_running_loop() -> asyncio.AbstractEventLoop:
    """
    Return the running event loop or raise a clear error.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            "create_tracked_task() must be called with a running event loop "
            "(e.g., inside FastAPI lifespan/request/websocket context). "
            "If calling from another thread, use "
            "create_tracked_task_threadsafe()."
        ) from e

def _validate_timeout_positive(
    timeout: float | None, *, param_name: str = "timeout"
) -> None:
    if timeout is None:
        return
    if timeout <= 0:
        raise ValueError(f"{param_name} must be > 0 or None, got: {timeout!r}")

def _try_close_coroutine(coro: Coroutine[Any, Any, Any]) -> None:
    """
    Close a coroutine object if we are going to drop it without scheduling.
    Prevents "coroutine was never awaited" ResourceWarning in some paths.
    """
    try:
        coro.close()
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        # Defensive: coroutine.close() shouldn't normally fail, but never break caller.
        logger.exception("task_tracker_coroutine_close_failed")

def _cap_check_or_reject(coro: Coroutine[Any, Any, Any]) -> None:
    """
    Enforce MAX_BACKGROUND_TASKS for tracked tasks.
    Must be called before creating the asyncio.Task to prevent task leaks.
    """
    with _tasks_lock:
        if len(_background_tasks) >= MAX_BACKGROUND_TASKS:
            _try_close_coroutine(coro)
            raise RuntimeError(
                f"Too many background tasks tracked (limit={MAX_BACKGROUND_TASKS}). "
                "This indicates runaway task spawning or insufficient cleanup."
            )

def _on_task_done(task: asyncio.Task[Any]) -> None:
    """
    Done callback for all tasks created by this module.

    Notes:
    - This callback runs on the event loop thread.
    - It must be fast and must not await.
    """
    # Remove from tracking set if present.
    try:
        with _tasks_lock:
            _background_tasks.discard(task)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        # Never allow callback failure to bubble into event loop.
        logger.exception(
            "task_tracker_done_callback_lock_failed",
            extra={"task": task.get_name()},
        )
        return

    # Surface unhandled exceptions (avoid silent task death).
    if task.cancelled():
        return

    try:
        exc = task.exception()
    except asyncio.CancelledError:
        # Race: in rare cases exception() may raise CancelledError
        return
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        logger.exception(
            "task_tracker_done_callback_exception_failed",
            extra={"task": task.get_name()},
        )
        return

    if exc is not None:
        logger.error(
            "unhandled_background_task_error",
            extra={"task": task.get_name(), "error": str(exc)},
            exc_info=exc,
        )

def create_tracked_task(
    coro: Coroutine[Any, Any, T],
    name: str | None = None,
    *,
    cancel_on_shutdown: bool = True,
) -> asyncio.Task[T]:
    """
    Create a background task that can be cancelled on shutdown and is tracked.

    Hardening:
    - Fail-fast if no running loop in current thread.
    - Backpressure cap for tracked tasks.
    - Done callback logs unhandled exceptions.
    """
    loop = _require_running_loop()

    if cancel_on_shutdown:
        _cap_check_or_reject(coro)

    task = loop.create_task(coro, name=name)
    task.add_done_callback(_on_task_done)

    if cancel_on_shutdown:
        with _tasks_lock:
            _background_tasks.add(cast(asyncio.Task[Any], task))
            total_tasks = len(_background_tasks)
    else:
        with _tasks_lock:
            total_tasks = len(_background_tasks)

    logger.debug(
        "Created task",
        extra={
            "task_name": name or task.get_name(),
            "tracked": cancel_on_shutdown,
            "total_tracked_tasks": total_tasks,
        },
    )
    return task

def create_tracked_task_threadsafe(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    loop: asyncio.AbstractEventLoop,
    *,
    name: str | None = None,
    cancel_on_shutdown: bool = True,
) -> "asyncio.Future[asyncio.Task[T]]":
    """
    Thread-safe API to create a tracked task from a non-event-loop thread.

    Why a factory:
    - Creates the coroutine in the loop thread, avoiding cross-thread surprises.

    Returns:
    - An asyncio.Future resolved in the target loop, containing the created Task.
      If you need a concurrent.futures.Future for use outside asyncio, wrap with
      asyncio.run_coroutine_threadsafe in your caller.
    """
    fut: asyncio.Future[asyncio.Task[T]] = loop.create_future()

    def _create() -> None:
        try:
            coro = coro_factory()
            t = create_tracked_task(
                coro, name=name, cancel_on_shutdown=cancel_on_shutdown
            )
            fut.set_result(t)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            fut.set_exception(e)

    loop.call_soon_threadsafe(_create)
    return fut

def background_task(
    func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., asyncio.Task[T]]:
    """
    Decorator: calling the function schedules it immediately and returns a Task.

    Important:
    - The decorated function no longer returns a coroutine; it returns asyncio.Task[T].
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> asyncio.Task[T]:
        coro = func(*args, **kwargs)
        return create_tracked_task(coro, name=func.__name__, cancel_on_shutdown=True)

    return wrapper

async def cancel_all_tasks(timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> dict[str, int]:
    """
    Cancel all tracked background tasks gracefully, with bounded waiting.

    Two-phase shutdown:
    1) Cancel + wait(timeout)
    2) Re-cancel pending + wait(timeout)
       Remaining pending are considered "stuck" and kept tracked for visibility.

    Returns stats with stable keys + richer observability fields.
    """
    _validate_timeout_positive(timeout, param_name="timeout")

    with _tasks_lock:
        tasks_to_cancel = list(_background_tasks)

    total = len(tasks_to_cancel)
    if total == 0:
        return {
            "total": 0,
            "cancelled": 0,
            "completed": 0,
            "errors": 0,
            "timeout": 0,
            "timed_out": 0,  # alias
            "stuck": 0,
        }

    logger.info("Cancelling %s background tasks...", total)

    for t in tasks_to_cancel:
        t.cancel()

    done1, pending1 = await asyncio.wait(tasks_to_cancel, timeout=timeout)

    cancelled = 0
    completed = 0
    errors = 0

    def _consume_done(done_set: set[asyncio.Task[Any]]) -> None:
        nonlocal cancelled, completed, errors
        for t in done_set:
            if t.cancelled():
                cancelled += 1
                continue
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                cancelled += 1
                continue
            if exc is not None:
                logger.error(
                    "background_task_error",
                    extra={"task": t.get_name(), "error": str(exc)},
                    exc_info=exc,
                )
                errors += 1
            else:
                completed += 1

    _consume_done(set(done1))

    timed_out = len(pending1)
    stuck = 0

    if pending1:
        logger.warning(
            "background_tasks_timeout",
            extra={
                "pending_count": len(pending1),
                "tasks": [t.get_name() for t in pending1],
            },
        )
        # Second bounded window.
        for t in pending1:
            t.cancel()

        done2, pending2 = await asyncio.wait(pending1, timeout=timeout)
        _consume_done(set(done2))

        if pending2:
            stuck = len(pending2)
            logger.error(
                "background_tasks_stuck_after_cancel",
                extra={"stuck_count": stuck, "tasks": [t.get_name() for t in pending2]},
            )
            # IMPORTANT: do NOT discard them here. Keep tracked for operator visibility.

    stats = {
        "total": total,
        "cancelled": cancelled,
        "completed": completed,
        "errors": errors,
        "timeout": timed_out,  # backward-compatible field name
        "timed_out": timed_out,  # clearer alias
        "stuck": stuck,
    }
    logger.info("Background tasks shutdown complete", extra=stats)
    return stats

def get_task_count() -> int:
    """Get number of currently tracked background tasks."""
    with _tasks_lock:
        return len(_background_tasks)

def get_task_names() -> list[str]:
    """Get names of all currently tracked background tasks (sorted for stability)."""
    with _tasks_lock:
        names = [t.get_name() for t in _background_tasks]
    names.sort()
    return names

async def wait_for_tasks(
    timeout: float | None = None,
    *,
    cancel_pending: bool = False,
    cancel_timeout: float | None = None,
) -> bool:
    """
    Wait for tracked tasks to complete.

    By default it only observes (does not cancel).
    If cancel_pending=True, it will request cancellation of pending tasks and
    wait up to cancel_timeout (defaults to timeout if provided,
    else DEFAULT_SHUTDOWN_TIMEOUT).

    Returns:
        True if all tasks finished within timeout, else False.
    """
    _validate_timeout_positive(timeout, param_name="timeout")
    _validate_timeout_positive(cancel_timeout, param_name="cancel_timeout")

    with _tasks_lock:
        tasks = set(_background_tasks)

    if not tasks:
        return True

    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if not pending:
        return True

    if not cancel_pending:
        return False

    # Cancel with bounded wait (do not hang forever).
    bounded = (
        cancel_timeout
        if cancel_timeout is not None
        else (timeout if timeout is not None else DEFAULT_SHUTDOWN_TIMEOUT)
    )

    for t in pending:
        t.cancel()

    await asyncio.wait(pending, timeout=bounded)
    return False

def track_task(
    coro: Coroutine[Any, Any, T],
    name: str | None = None,
    *,
    cancel_on_shutdown: bool = True,
) -> asyncio.Task[T]:
    """Backward-compatibility alias for create_tracked_task()."""
    return create_tracked_task(coro, name=name, cancel_on_shutdown=cancel_on_shutdown)

if __name__ == "__main__":

    async def example_shutdown() -> None:
        @background_task
        async def long_task() -> None:
            try:
                while True:
                    print("Working...")
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("Task cancelled, cleaning up...")
                raise

        long_task()
        long_task()
        await asyncio.sleep(3)
        stats = await cancel_all_tasks(timeout=2.0)
        print(f"Cancelled confirmed: {stats['cancelled']}, stuck: {stats['stuck']}")

    run_sync(example_shutdown())
