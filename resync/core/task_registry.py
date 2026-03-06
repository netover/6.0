"""Task registry for background asyncio tasks.

Centralizes creation/tracking/cancellation of background tasks so we don't
leak orphan tasks across reloads/shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Optional, Set

logger = logging.getLogger(__name__)

_tasks: Set[asyncio.Task] = set()
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def create_tracked_task(aw: Awaitable, *, name: Optional[str] = None) -> asyncio.Task:
    """Create and track a background task.

    This should be used instead of asyncio.create_task for long-lived tasks
    that must be cancelled on shutdown.
    """
    task = asyncio.create_task(aw, name=name)  # type: ignore[arg-type]
    _tasks.add(task)

    def _cleanup(_t: asyncio.Task) -> None:
        _tasks.discard(_t)
        try:
            exc = _t.exception()
            if exc:
                logger.debug("Background task %s finished with error: %s", _t.get_name(), exc, exc_info=True)
        except asyncio.CancelledError:
            # normal on shutdown
            pass
        except Exception:
            logger.debug("Error reading task exception", exc_info=True)

    task.add_done_callback(_cleanup)
    return task


async def cancel_all_tracked_tasks(*, timeout_s: float = 5.0) -> None:
    """Cancel all tracked tasks and wait for completion."""
    if not _tasks:
        return
    # copy to avoid mutation while iterating
    tasks = list(_tasks)
    for t in tasks:
        if not t.done():
            t.cancel()
    try:
        await asyncio.wait(tasks, timeout=timeout_s)
    except Exception:
        logger.debug("Error while waiting tracked tasks", exc_info=True)
    finally:
        _tasks.clear()


def get_task_stats(*, sample_limit: int = 50) -> dict[str, object]:
    """Return lightweight stats about currently tracked tasks.

    Useful for admin dashboards and debugging task leaks.
    """
    tasks = list(_tasks)
    total = len(tasks)
    done = sum(1 for t in tasks if t.done())
    cancelled = sum(1 for t in tasks if t.cancelled())
    running = total - done
    # Stable-ish sample of names
    names: list[str] = []
    for t in tasks[:max(0, sample_limit)]:
        try:
            names.append(t.get_name())
        except Exception:
            names.append("<unnamed>")
    return {
        "total": total,
        "running": running,
        "done": done,
        "cancelled": cancelled,
        "sample_names": names,
    }
