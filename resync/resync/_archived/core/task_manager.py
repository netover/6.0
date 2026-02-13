"""Asynchronous task management system.

This module provides a comprehensive task execution framework with:
- Priority-based task scheduling
- Retry logic with configurable delays
- Timeout handling for long-running tasks
- Concurrent execution with configurable worker pools
- Task status tracking and statistics
- Graceful shutdown and cleanup
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task to be executed."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Callable[..., Any] | None = None
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: Exception | None = None
    max_retries: int = 0
    retry_count: int = 0
    retry_delay: float = 1.0
    timeout: float | None = None


class TaskManager:
    """Manages asynchronous task execution with priority and retry support."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.tasks: dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()
        self.running_tasks: dict[str, asyncio.Task[Any]] = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self._shutdown = False

        # Start the task processor
        self._processor_task = None

    async def start(self):
        """Start the task manager."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_tasks())

    async def stop(self):
        """Stop the task manager."""
        self._shutdown = True

        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()

        # Wait for processor to finish
        if self._processor_task and not self._processor_task.done():
            await self._processor_task

    async def submit(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a task for asynchronous execution.

        Args:
            name: Human-readable task name for monitoring
            func: Async callable to execute
            *args: Positional arguments for the function
            priority: Task execution priority (LOW/NORMAL/HIGH)
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retry attempts in seconds
            timeout: Maximum execution time in seconds
            **kwargs: Keyword arguments for the function

        Returns:
            Unique task identifier string

        Raises:
            ValueError: If task manager is not running
        """

        task = Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )

        self.tasks[task.id] = task

        # Use negative priority for max-heap behavior (higher priority = lower number)
        priority_value = -priority.value
        await self.task_queue.put((priority_value, task.id))

        logger.info(f"Task submitted: {task.id} - {name}")
        return task.id

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's pending or running."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]:
            return False

        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            return True

        # Task is still in queue
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        return True

    async def _process_tasks(self):
        """Process tasks from the queue."""

        while not self._shutdown:
            try:
                # Get next task from queue with timeout
                try:
                    priority_value, task_id = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                task = self.tasks.get(task_id)
                if not task or task.status == TaskStatus.CANCELLED:
                    continue

                # Acquire semaphore to limit concurrent tasks
                async with self.semaphore:
                    await self._execute_task(task)

            except Exception as e:
                logger.error(f"Error processing tasks: {e}")

    async def _execute_task(self, task: Task):
        """Execute a single task."""

        if task.status != TaskStatus.PENDING:
            return

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        # Create the actual asyncio task
        async def task_wrapper():
            try:
                result = await self._run_task_function(task)
                await self._handle_task_success(task, result)

            except asyncio.CancelledError:
                await self._handle_task_cancellation(task)

            except Exception as e:
                logger.error("exception_caught", error=str(e), exc_info=True)
                await self._handle_task_failure(task, e)

        # Start the task
        asyncio_task = asyncio.create_task(task_wrapper())
        self.running_tasks[task.id] = asyncio_task

        # Wait for completion
        try:
            await asyncio_task
        finally:
            self.running_tasks.pop(task.id, None)

    async def _run_task_function(self, task: Task) -> Any:
        """Execute the task function with appropriate handling."""
        if asyncio.iscoroutinefunction(task.func):
            return await self._run_async_function(task)
        return await self._run_sync_function(task)

    async def _run_async_function(self, task: Task) -> Any:
        """Run an async task function."""
        if task.func is None:
            raise ValueError("Task function is not set")
        if task.timeout:
            return await asyncio.wait_for(
                task.func(*task.args, **task.kwargs), timeout=task.timeout
            )
        return await task.func(*task.args, **task.kwargs)

    async def _run_sync_function(self, task: Task) -> Any:
        """Run a sync task function in executor."""
        if task.func is None:
            raise ValueError("Task function is not set")
        loop = asyncio.get_event_loop()
        if task.timeout:
            return await asyncio.wait_for(
                loop.run_in_executor(None, task.func, *task.args, **task.kwargs),
                timeout=task.timeout,
            )
        return await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)

    async def _handle_task_success(self, task: Task, result: Any) -> None:
        """Handle successful task completion."""
        task.result = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        logger.info(f"Task completed: {task.id} - {task.name}")

    async def _handle_task_cancellation(self, task: Task) -> None:
        """Handle task cancellation."""
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        logger.info(f"Task cancelled: {task.id} - {task.name}")

    async def _handle_task_failure(self, task: Task, error: Exception) -> None:
        """Handle task failure and retry logic."""
        task.error = error
        logger.error(f"Task failed: {task.id} - {task.name}: {error}")

        # Handle retries
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            # Schedule retry
            asyncio.create_task(self._schedule_retry(task))
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

    async def _schedule_retry(self, task: Task):
        """Schedule a task retry with delay."""

        await asyncio.sleep(task.retry_delay * task.retry_count)

        if task.status == TaskStatus.PENDING:
            # Use negative priority for max-heap behavior
            priority_value = -task.priority.value
            await self.task_queue.put((priority_value, task.id))

    def get_stats(self) -> dict[str, Any]:
        """Get task manager statistics."""

        from collections import Counter

        status_counts = Counter(task.status.value for task in self.tasks.values())

        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(self.running_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "max_workers": self.max_workers,
            "available_workers": (
                self.semaphore._value if hasattr(self.semaphore, "_value") else "unknown"
            ),
            "status_counts": status_counts,
        }
