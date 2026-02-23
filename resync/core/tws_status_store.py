"""
TWS Status Store - PostgreSQL Implementation.

This module provides the same interface as the original SQLite-based
tws_status_store.py but uses PostgreSQL via the database repositories.

Migration Note:
    This file replaces the original SQLite implementation.
    The interface remains the same for backward compatibility.
"""

import logging
from datetime import datetime
from typing import Any

from resync.core.database.models import (
    TWSEvent,
    TWSJobStatus,
    TWSPattern,
    TWSProblemSolution,
)

# Import from new PostgreSQL repositories
from resync.core.database.repositories import (
    JobStatus,
    PatternMatch,
    TWSStore,
)

logger = logging.getLogger(__name__)

# Re-export data classes for compatibility
__all__ = [
    "JobStatus",
    "PatternMatch",
    "TWSStatusStore",
    "get_tws_status_store",
]


class TWSStatusStore:
    """
    TWS Status Store - PostgreSQL Backend.

    Provides storage and retrieval of TWS job status, events,
    patterns, and problem-solutions using PostgreSQL.
    """

    def __init__(self):
        """Initialize - uses PostgreSQL."""
        self._store = TWSStore()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the store."""
        await self._store.initialize()
        self._initialized = True
        logger.info("TWSStatusStore initialized (PostgreSQL)")

    async def close(self) -> None:
        """Close the store."""
        await self._store.close()
        self._initialized = False

    async def update_job_status(
        self, job: JobStatus, snapshot_id: int | None = None
    ) -> TWSJobStatus:
        """Update or insert job status."""
        return await self._store.update_job_status(job)

    async def get_job_status(self, job_name: str) -> TWSJobStatus | None:
        """Get latest status for a job."""
        return await self._store.get_job_status(job_name)

    async def get_job_history(
        self, job_name: str, limit: int = 100
    ) -> list[TWSJobStatus]:
        """Get job status history."""
        return await self._store.get_job_history(job_name, limit)

    async def get_failed_jobs(
        self, hours: int = 24, limit: int = 100
    ) -> list[TWSJobStatus]:
        """Get recently failed jobs."""
        return await self._store.get_failed_jobs(hours, limit)

    async def get_status_summary(self) -> dict[str, int]:
        """Get job count by status."""
        return await self._store.get_status_summary()

    async def get_jobs_by_workstation(
        self, workstation: str, limit: int = 100
    ) -> list[TWSJobStatus]:
        """Get jobs for a workstation."""
        return await self._store.jobs.get_jobs_by_workstation(workstation, limit)

    async def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        job_name: str | None = None,
        workstation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TWSEvent:
        """Log an event."""
        return await self._store.log_event(
            event_type=event_type,
            message=message,
            severity=severity,
            job_name=job_name,
            details=details,
        )

    async def get_events(
        self, limit: int = 100, severity: str | None = None, job_name: str | None = None
    ) -> list[TWSEvent]:
        """Get events with optional filters."""
        if severity:
            return await self._store.events.get_events_by_severity(
                severity, limit=limit
            )
        if job_name:
            return await self._store.events.find({"job_name": job_name}, limit=limit)
        return await self._store.events.get_all(limit=limit)

    async def get_events_in_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> list[TWSEvent]:
        """Get events in time range."""
        return await self._store.get_events_in_range(start, end, limit)

    async def get_unacknowledged_events(self, limit: int = 100) -> list[TWSEvent]:
        """Get unacknowledged events."""
        return await self._store.events.get_unacknowledged(limit)

    async def acknowledge_event(
        self, event_id: int, acknowledged_by: str
    ) -> TWSEvent | None:
        """Acknowledge an event."""
        return await self._store.events.acknowledge_event(event_id, acknowledged_by)

    async def detect_pattern(self, pattern: PatternMatch) -> TWSPattern:
        """Record a detected pattern."""
        return await self._store.detect_pattern(pattern)

    async def get_patterns(
        self, job_name: str | None = None, min_confidence: float = 0.5
    ) -> list[TWSPattern]:
        """Get active patterns."""
        return await self._store.get_patterns(job_name, min_confidence)

    async def add_solution(
        self,
        problem_type: str,
        problem_description: str,
        solution: str,
        job_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TWSProblemSolution:
        """Add a problem-solution pair."""
        return await self._store.add_solution(
            problem_type=problem_type,
            problem_description=problem_description,
            solution=solution,
            job_name=job_name,
        )

    async def find_solution(
        self, problem_type: str, job_name: str | None = None
    ) -> TWSProblemSolution | None:
        """Find a solution for a problem."""
        return await self._store.find_solution(problem_type, job_name)

    async def record_solution_outcome(
        self, solution_id: int, success: bool
    ) -> TWSProblemSolution | None:
        """Record whether a solution worked."""
        return await self._store.solutions.record_outcome(solution_id, success)

    async def cleanup_old_data(self, days: int = 30) -> dict[str, int]:
        """Clean up old data."""
        return await self._store.cleanup_old_data(days)


_instance: TWSStatusStore | None = None


def get_tws_status_store() -> TWSStatusStore:
    """Get the singleton TWSStatusStore instance."""
    global _instance
    if _instance is None:
        _instance = TWSStatusStore()
    return _instance


async def initialize_tws_status_store() -> TWSStatusStore:
    """Initialize and return the TWSStatusStore."""
    store = get_tws_status_store()
    await store.initialize()
    return store
