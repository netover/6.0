import asyncio

import pytest

from resync.core.resource_manager import ResourcePool

pytestmark = pytest.mark.asyncio(loop_scope="function")


@pytest.mark.asyncio
async def test_acquire_does_not_serialize_factory_execution():
    pool: ResourcePool[object] = ResourcePool(max_resources=10)
    started = 0
    started_event = asyncio.Event()
    proceed_event = asyncio.Event()
    counter_lock = asyncio.Lock()

    async def factory() -> object:
        nonlocal started
        async with counter_lock:
            started += 1
            if started >= 2:
                started_event.set()
        await proceed_event.wait()
        return object()

    t1 = asyncio.create_task(pool.acquire("x", factory))
    t2 = asyncio.create_task(pool.acquire("x", factory))

    await asyncio.wait_for(started_event.wait(), timeout=0.5)
    proceed_event.set()

    (id1, r1), (id2, r2) = await asyncio.gather(t1, t2)
    await pool.release(id1, r1)
    await pool.release(id2, r2)

    assert pool.get_stats()["active_resources"] == 0


@pytest.mark.asyncio
async def test_release_does_not_serialize_close_execution():
    pool: ResourcePool[object] = ResourcePool(max_resources=10)
    close_started = 0
    close_started_event = asyncio.Event()
    proceed_event = asyncio.Event()
    counter_lock = asyncio.Lock()

    class CloseWaiter:
        async def close(self) -> None:
            nonlocal close_started
            async with counter_lock:
                close_started += 1
                if close_started >= 2:
                    close_started_event.set()
            await proceed_event.wait()

    async def factory() -> CloseWaiter:
        return CloseWaiter()

    id1, r1 = await pool.acquire("x", factory)
    id2, r2 = await pool.acquire("x", factory)

    rel1 = asyncio.create_task(pool.release(id1, r1))
    rel2 = asyncio.create_task(pool.release(id2, r2))

    await asyncio.wait_for(close_started_event.wait(), timeout=0.5)
    proceed_event.set()

    await asyncio.gather(rel1, rel2)
    assert pool.get_stats()["active_resources"] == 0


@pytest.mark.asyncio
async def test_factory_failure_releases_capacity():
    pool: ResourcePool[object] = ResourcePool(max_resources=1)

    async def failing_factory() -> object:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await pool.acquire("x", failing_factory)

    async def ok_factory() -> object:
        return object()

    resource_id, resource = await pool.acquire("x", ok_factory)
    await pool.release(resource_id, resource)
    assert pool.get_stats()["active_resources"] == 0


@pytest.mark.asyncio
async def test_capacity_check_includes_inflight_creations():
    pool: ResourcePool[object] = ResourcePool(max_resources=1)
    started_event = asyncio.Event()
    proceed_event = asyncio.Event()

    async def slow_factory() -> object:
        started_event.set()
        await proceed_event.wait()
        return object()

    t1 = asyncio.create_task(pool.acquire("x", slow_factory))
    await asyncio.wait_for(started_event.wait(), timeout=0.5)

    async def other_factory() -> object:
        return object()

    with pytest.raises(RuntimeError):
        await pool.acquire("x", other_factory)

    proceed_event.set()
    resource_id, resource = await t1
    await pool.release(resource_id, resource)
    assert pool.get_stats()["active_resources"] == 0

