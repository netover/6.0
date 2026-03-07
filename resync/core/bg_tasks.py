from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


class ManagedTaskGroup:
    """Wrapper around asyncio.TaskGroup with tracked task cancellation."""

    def __init__(self) -> None:
        self._tg: asyncio.TaskGroup | None = None
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> ManagedTaskGroup:
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        if self._tg is None:
            return None
        try:
            return await self._tg.__aexit__(exc_type, exc, tb)
        finally:
            self._tg = None

    def create_task(
        self, coro: Awaitable[Any], *, name: str | None = None
    ) -> asyncio.Task[Any]:
        if self._tg is None:
            raise RuntimeError("ManagedTaskGroup must be entered before use")
        task = self._tg.create_task(coro, name=name)
        self._tasks.append(task)
        return task

    def tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(self._tasks)

    async def cancel_all(self) -> None:
        pending: list[asyncio.Task[Any]] = []
        for task in self._tasks:
            if not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
