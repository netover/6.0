from __future__ import annotations

import asyncio

import pytest

from resync.core.bg_tasks import ManagedTaskGroup


@pytest.mark.asyncio
async def test_cancel_all_waits_for_task_completion() -> None:
    finished = asyncio.Event()

    async def worker() -> None:
        try:
            await asyncio.sleep(60)
        finally:
            finished.set()

    async with ManagedTaskGroup() as tg:
        tg.create_task(worker())
        await asyncio.sleep(0)
        await tg.cancel_all()

    assert finished.is_set()
