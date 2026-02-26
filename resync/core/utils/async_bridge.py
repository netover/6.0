"""Utilities to bridge between sync and async code safely.

Why this exists
--------------
Resync runs in mixed environments (CLI tools, background threads, and ASGI
servers like FastAPI). In particular, calling ``asyncio.run`` while an event
loop is already running in the *same thread* raises ``RuntimeError``.

These helpers provide a single, well-defined place for that behavior.

Design principles
-----------------
* **Deterministic**: never silently "schedule and continue" when the caller
  expects results synchronously. If a loop is running, we raise with guidance.
* **Minimal deps**: avoids adding third-party dependencies.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine from synchronous code.

    If no event loop is running in the current thread, this runs the coroutine
    to completion using ``asyncio.run``.

    If an event loop is already running in the current thread, this raises a
    RuntimeError with guidance (because a synchronous caller cannot reliably
    "wait" for async work without blocking the loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "run_sync() called from a running event loop. "
        "Convert caller to async and `await` coroutine instead."
    )

def fire_and_forget(
    coro: Coroutine[Any, Any, Any],
    *,
    logger: Any | None = None,
    name: str | None = None,
) -> None:
    """Schedule a coroutine without awaiting it.

    This helper is intended for background refreshes and telemetry. It attaches
    a done-callback that logs exceptions instead of letting them get swallowed.
    If called from a thread without a running loop, this will raise.
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(coro, name=name)

    def _done(t: asyncio.Task) -> None:
        try:
            t.result()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            if logger is not None:
                try:
                    logger.warning(
                        "background_task_failed",
                        extra={"task": name or str(t), "error": str(exc)},
                    )
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as exc:
                    logger.debug(
                        "suppressed_exception", exc_info=True, extra={"error": str(exc)}
                    )

    task.add_done_callback(_done)
