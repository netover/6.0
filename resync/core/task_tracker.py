"""
Background Task Tracking System

Prevents task leaks and ensures proper cleanup on shutdown.

PROBLEM:
========
Creating tasks without tracking leads to:
- Memory leaks (tasks never garbage collected)
- Incomplete shutdown (tasks still running)
- No way to cancel tasks gracefully

Example of problematic code:
```python
asyncio.create_task(long_running_work())  # Task is lost!
```

SOLUTION:
=========
Use tracked task creation:

```python
from resync.core.task_tracker import create_tracked_task, track_task

# Task is automatically tracked and cleaned up
task = track_task(long_running_work())
```

Or use decorator for background functions:

```python
from resync.core.task_tracker import background_task

@background_task
async def periodic_cleanup():
    while True:
        # Do work
        await asyncio.sleep(60)

# Start it
await periodic_cleanup()  # Automatically tracked!
```

SHUTDOWN:
=========
All tracked tasks are automatically cancelled during application shutdown
via the lifespan manager.

Tasks get:
1. Graceful cancellation (CancelledError)
2. 5 second timeout for cleanup
3. Forced termination if timeout exceeded
"""
import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar
logger = logging.getLogger(__name__)
_background_tasks: set[asyncio.Task] = set()
T = TypeVar('T')

async def create_tracked_task(coro: Coroutine[Any, Any, T], name: str | None=None, *, cancel_on_shutdown: bool=True) -> asyncio.Task[T]:
    """
    Create a background task that is automatically tracked and cleaned up.

    Args:
        coro: Coroutine to run as a task
        name: Optional name for the task (for debugging)
        cancel_on_shutdown: If True, task will be cancelled on shutdown

    Returns:
        Task object

    Example:
        ```python
        async def monitor_redis():
            while True:
                await check_redis_health()
                await asyncio.sleep(30)

        # Create tracked background task
        task = await create_tracked_task(
            monitor_redis(),
            name="redis-monitor"
        )
        ```
    """
    task = asyncio.create_task(coro, name=name)
    if cancel_on_shutdown:
        _background_tasks.add(task)

        def remove_task(t: asyncio.Task) -> None:
            _background_tasks.discard(t)
        task.add_done_callback(remove_task)
    logger.debug('Created tracked task', extra={'task_name': name or task.get_name(), 'total_tasks': len(_background_tasks)})
    return task

def background_task(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., asyncio.Task[T]]:
    """
    Decorator to automatically track background tasks.

    Usage:
        ```python
        @background_task
        async def periodic_cleanup():
            while True:
                await cleanup()
                await asyncio.sleep(3600)

        # Start background task
        task = await periodic_cleanup()
        ```
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> asyncio.Task[T]:
        coro = func(*args, **kwargs)
        return await create_tracked_task(coro, name=func.__name__)
    return wrapper

async def cancel_all_tasks(timeout: float=5.0) -> dict[str, int]:
    """
    Cancel all tracked background tasks gracefully.

    Called automatically during application shutdown.

    Args:
        timeout: Maximum time to wait for tasks to cleanup (seconds)

    Returns:
        Dictionary with cancellation statistics

    Example:
        ```python
        # In lifespan shutdown:
        stats = await cancel_all_tasks(timeout=10.0)
        logger.info("Cancelled %s tasks", stats['cancelled'])
        ```
    """
    tasks_to_cancel = list(_background_tasks)
    total = len(tasks_to_cancel)
    if not tasks_to_cancel:
        return {'total': 0, 'cancelled': 0, 'timeout': 0, 'error': 0}
    logger.info('Cancelling %s background tasks...', total)
    for task in tasks_to_cancel:
        task.cancel()
    done, pending = await asyncio.wait(tasks_to_cancel, timeout=timeout)
    cancelled_count = 0
    error_count = 0
    success_count = 0
    for task in done:
        try:
            if task.cancelled():
                cancelled_count += 1
            else:
                exc = task.exception()
                if exc and (not isinstance(exc, asyncio.CancelledError)):
                    logger.error('background_task_error', extra={'task': task.get_name(), 'error': str(exc)})
                    error_count += 1
                else:
                    success_count += 1
        except asyncio.CancelledError:
            cancelled_count += 1
    if pending:
        logger.warning('background_tasks_timeout', extra={'pending_count': len(pending), 'tasks': [t.get_name() for t in pending]})
    stats = {'total': total, 'cancelled': cancelled_count + len(pending), 'completed': success_count, 'errors': error_count}
    logger.info('Background tasks shutdown complete', extra=stats)
    _background_tasks.clear()
    return stats

def get_task_count() -> int:
    """
    Get number of currently running background tasks.

    Returns:
        Number of active tasks
    """
    return len(_background_tasks)

def track_task(coro: Coroutine[Any, Any, T], name: str | None=None) -> asyncio.Task[T]:
    """Create a tracked task from a **sync** context (``def``, not ``async def``).

    Unlike :func:`create_tracked_task` this function is synchronous and can be
    called from any ``def`` function that runs inside the event-loop (e.g. a
    callback, an ``on_modified`` handler, or a class ``start()`` method).

    It is equivalent to::

        task = asyncio.create_task(coro, name=name)
        _background_tasks.add(task)

    The task is automatically removed from ``_background_tasks`` when done
    and will be cancelled by :func:`cancel_all_tasks` during shutdown.
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)

    def _remove(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
    task.add_done_callback(_remove)
    logger.debug('Created tracked task (sync)', extra={'task_name': name or task.get_name(), 'total_tasks': len(_background_tasks)})
    return task

def get_task_names() -> list[str]:
    """
    Get names of all running background tasks.

    Returns:
        List of task names
    """
    return [task.get_name() for task in _background_tasks]

async def wait_for_tasks(timeout: float | None=None) -> bool:
    """
    Wait for all background tasks to complete.

    Args:
        timeout: Maximum time to wait (None = wait forever)

    Returns:
        True if all tasks completed, False if timeout

    Example:
        ```python
        # Wait for all tasks to finish
        completed = await wait_for_tasks(timeout=30.0)
        if not completed:
            logger.warning("Some tasks didn't finish in time")
        ```
    """
    tasks = list(_background_tasks)
    if not tasks:
        return True
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
if __name__ == '__main__':

    async def example_manual():

        async def worker(n: int):
            for i in range(n):
                print(f'Worker iteration {i}')
                await asyncio.sleep(1)
        task = await create_tracked_task(worker(5), name='worker-1')
        await task

    async def example_decorator():

        @background_task
        async def monitor():
            count = 0
            while True:
                count += 1
                print(f'Monitor check {count}')
                await asyncio.sleep(2)
                if count >= 5:
                    break
        task = await monitor()
        await task

    async def example_shutdown():

        @background_task
        async def long_task():
            try:
                while True:
                    print('Working...')
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                print('Task cancelled, cleaning up...')
                raise
        await long_task()
        await long_task()
        await asyncio.sleep(3)
        stats = await cancel_all_tasks(timeout=2.0)
        print(f"Cancelled {stats['cancelled']} tasks")
    asyncio.run(example_shutdown())