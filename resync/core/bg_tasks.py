from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

class ManagedTaskGroup:
    """A small wrapper around asyncio.TaskGroup that tracks created tasks.

    This avoids touching asyncio.TaskGroup private attributes like ``_tasks`` and
    gives us a stable way to cancel long-running background tasks on shutdown.

    It intentionally mirrors the subset of TaskGroup API we use (create_task).
    """

    def __init__(self, tg: asyncio.TaskGroup) -> None:
        self._tg = tg
        self._tasks: list[asyncio.Task[Any]] = []

    def create_task(self, coro: Awaitable[Any], *, name: str | None = None) -> asyncio.Task[Any]:
        task = self._tg.create_task(coro, name=name)
        self._tasks.append(task)
        return task

    def tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(self._tasks)

    async def cancel_all(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        # Let cancellations propagate
        await asyncio.sleep(0)

    def __getattr__(self, item: str) -> Any:
        # Forward other attributes to the underlying TaskGroup (best-effort).
        return getattr(self._tg, item)
