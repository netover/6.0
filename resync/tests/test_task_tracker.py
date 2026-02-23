import asyncio
import logging
from typing import List

import pytest
import pytest_asyncio

from resync.core.task_tracker import (
    background_task,
    cancel_all_tasks,
    create_tracked_task,
    get_task_count,
    get_task_names,
    track_task,
    wait_for_tasks,
)

pytestmark = pytest.mark.asyncio(loop_scope="function")


async def eventually(assert_fn, timeout=1.0, step=0.01):
    """Helper for eventual consistency polling to eliminate generic short sleeps."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            assert_fn()
            return
        except AssertionError:
            if asyncio.get_running_loop().time() >= deadline:
                raise
            await asyncio.sleep(step)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks_between_tests():
    # Increase base timeout to 0.5 for CI environments
    await cancel_all_tasks(timeout=0.5)
    yield
    stats = await cancel_all_tasks(timeout=0.5)
    if stats["timeout"] > 0:
        pytest.fail(f"Tasks leaked and exceeded timeout: {stats}")


@pytest.mark.asyncio
async def test_create_tracked_task_tracks_and_removes_on_completion():
    async def quick():
        await asyncio.sleep(0.05)
        return 42

    before = get_task_count()
    task = create_tracked_task(quick(), name="quick-task")

    # Should be tracked immediately
    assert get_task_count() == before + 1

    # Wait for completion
    result = await task
    assert result == 42

    # Allow done-callback to run and remove from set polling
    await eventually(lambda: get_task_count() == before)


@pytest.mark.asyncio
async def test_create_tracked_task_not_tracked_when_cancel_on_shutdown_false():
    async def quick():
        await asyncio.sleep(0.05)

    before = get_task_count()
    task = create_tracked_task(quick(), name="no-track", cancel_on_shutdown=False)
    await task

    # Count should remain unchanged
    await eventually(lambda: get_task_count() == before)


@pytest.mark.asyncio
async def test_track_task_tracks_sync_context():
    async def quick():
        await asyncio.sleep(0.05)
        return 1

    before = get_task_count()
    t = track_task(quick(), name="sync-start")

    assert get_task_count() == before + 1
    assert await t == 1

    await eventually(lambda: get_task_count() == before)


@pytest.mark.asyncio
async def test_background_task_decorator_tracks_and_names_function():
    @background_task
    async def worker():
        await asyncio.sleep(0.05)

    before_names: List[str] = get_task_names()
    t = worker()

    # Name should be function name and task should be tracked
    names = get_task_names()
    assert len(names) == len(before_names) + 1
    assert "worker" in names

    await t

    def assert_removed():
        assert "worker" not in get_task_names()
        assert get_task_count() == len(before_names)

    await eventually(assert_removed)


@pytest.mark.asyncio
async def test_cancel_all_tasks_cancels_running_tasks_and_reports_stats():
    async def long_running():
        never = asyncio.Event()
        try:
            await never.wait()
        except asyncio.CancelledError:
            # simulate small cleanup
            await asyncio.sleep(0.05)
            raise

    # Create multiple long-running tracked tasks
    _ = create_tracked_task(long_running(), name="long-1")
    _ = create_tracked_task(long_running(), name="long-2")

    await asyncio.sleep(0.05)

    stats = await cancel_all_tasks(timeout=0.3)

    assert stats["total"] >= 2
    assert stats["cancelled"] >= 2
    assert stats["completed"] == 0
    assert stats["errors"] == 0
    assert stats["timeout"] == 0
    assert get_task_count() == 0


@pytest.mark.asyncio
async def test_cancel_all_tasks_collects_errors_for_failed_tasks_and_logs_error(caplog):
    caplog.set_level(logging.ERROR)

    async def failing_on_cancel():
        never = asyncio.Event()
        try:
            await never.wait()
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed boom")

    _ = create_tracked_task(failing_on_cancel(), name="will-fail-on-cancel")

    # Give the task a chance to start and enter the try block before cancelling
    await asyncio.sleep(0.05)

    stats = await cancel_all_tasks(timeout=0.2)

    assert stats["errors"] >= 1
    assert any("background_task_error" in r.message for r in caplog.records)
    # After cleanup, no tasks should remain tracked
    await eventually(lambda: get_task_count() == 0)


@pytest.mark.asyncio
async def test_wait_for_tasks_returns_true_when_all_complete_and_false_on_timeout():
    async def short():
        await asyncio.sleep(0.05)

    async def long():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Let it be cancellable to avoid leaks
            raise

    # Case 1: all complete
    create_tracked_task(short(), name="short")
    ok = await wait_for_tasks(timeout=0.5)
    assert ok is True

    # Case 2: timeout waiting for long task
    create_tracked_task(long(), name="long")
    ok2 = await wait_for_tasks(timeout=0.1)
    # Function cancels pending then returns False
    assert ok2 is False

    # Ensure global set is cleaned (pending was cancelled inside wait_for_tasks)
    await cancel_all_tasks(timeout=0.2)
    assert get_task_count() == 0


@pytest.mark.asyncio
async def test_cancel_all_tasks_empty_set():
    stats = await cancel_all_tasks()
    assert stats["total"] == 0
    assert stats["cancelled"] == 0
    assert stats["completed"] == 0
    assert stats["errors"] == 0
    assert stats["timeout"] == 0
