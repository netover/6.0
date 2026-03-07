"""
TWS Status Store - PostgreSQL Implementation.

This module provides the same interface as the original SQLite-based
tws_status_store.py but uses PostgreSQL via the database repositories.

Migration Note:
    This file replaces the original SQLite implementation.
    The interface remains the same for backward compatibility.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from threading import Lock

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
        self, job_name: str, days: int | None = None, limit: int = 100
    ) -> list[TWSJobStatus]:
        """Get job status history with legacy days+limit compatibility."""
        history = await self._store.get_job_history(job_name, limit)
        if days is None:
            return history

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [entry for entry in history if entry.timestamp and entry.timestamp >= cutoff]

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
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[str] | None = None,
        severity: str | None = None,
    ) -> list[TWSEvent]:
        """Get events in time range with legacy monitoring-route compatibility."""
        effective_start = start_time or start
        effective_end = end_time or end
        if effective_start is None or effective_end is None:
            raise ValueError("start/end timestamps are required")

        events = await self._store.get_events_in_range(effective_start, effective_end, limit)
        if severity:
            severity_lower = severity.lower()
            events = [event for event in events if (event.severity or "").lower() == severity_lower]
        if event_types:
            allowed = {event_type.lower() for event_type in event_types}
            events = [event for event in events if (event.event_type or "").lower() in allowed]
        return events

    async def search_events(self, query: str, limit: int = 50) -> list[TWSEvent]:
        """Best-effort text search compatible with legacy monitoring routes."""
        needle = query.lower()
        events = await self._store.events.get_all(limit=max(limit * 4, 200))
        matches = [
            event
            for event in events
            if needle in (event.message or "").lower()
            or needle in (event.event_type or "").lower()
            or needle in (event.job_name or "").lower()
        ]
        return matches[:limit]

    async def get_daily_summary(self, target_date: datetime) -> dict[str, Any]:
        """Build daily summary compatible with legacy monitoring endpoint."""
        start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        events = await self._store.get_events_in_range(start, end, limit=5000)
        jobs = await self._store.jobs.find({}, limit=5000, order_by="timestamp", desc=True)
        jobs = [job for job in jobs if job.timestamp and start <= job.timestamp < end]

        return {
            "date": start.date().isoformat(),
            "total_jobs": len(jobs),
            "failed_jobs": sum(1 for job in jobs if (job.status or "").upper() in {"FAILED", "ERROR", "ABEND"}),
            "completed_jobs": sum(1 for job in jobs if (job.status or "").upper() in {"COMPLETED", "SUCC", "SUCCESS"}),
            "events_count": len(events),
        }

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
        self,
        pattern_type: str | None = None,
        min_confidence: float = 0.5,
    ) -> list[TWSPattern]:
        """Get active patterns with legacy pattern_type filtering."""
        patterns = await self._store.get_patterns(None, min_confidence)
        if pattern_type:
            pattern_type_lower = pattern_type.lower()
            patterns = [
                pattern
                for pattern in patterns
                if (pattern.pattern_type or "").lower() == pattern_type_lower
            ]
        return patterns

    async def detect_patterns(self) -> list[TWSPattern]:
        """Legacy compatibility shim for manual pattern detection endpoint."""
        return await self.get_patterns()

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
        self, problem_type: str, error_message: str | None = None
    ) -> TWSProblemSolution | None:
        """Find a solution for a problem using legacy route signature."""
        _ = error_message
        return await self._store.find_solution(problem_type, None)

    async def record_solution_outcome(
        self, solution_id: int, success: bool
    ) -> TWSProblemSolution | None:
        """Record whether a solution worked."""
        return await self._store.solutions.record_outcome(solution_id, success)

    async def record_solution_result(
        self, problem_id: int, success: bool
    ) -> TWSProblemSolution | None:
        """Legacy alias expected by monitoring routes."""
        return await self.record_solution_outcome(problem_id, success)

    async def cleanup_old_data(self, days: int = 30) -> dict[str, int]:
        """Clean up old data."""
        return await self._store.cleanup_old_data(days)

    async def get_database_stats(self) -> dict[str, Any]:
        """Compatibility stats payload for monitoring routes."""
        summary = await self.get_status_summary()
        recent_events = await self._store.events.get_all(limit=1000)
        active_patterns = await self.get_patterns()
        solutions = await self._store.solutions.get_all(limit=1000)
        return {
            "job_status_summary": summary,
            "events_count": len(recent_events),
            "active_patterns": len(active_patterns),
            "solutions_count": len(solutions),
        }

_instance: TWSStatusStore | None = None
_init_lock: Lock = Lock()
_async_init_lock: asyncio.Lock | None = None

def _get_async_init_lock() -> asyncio.Lock:
    """Return async init lock lazily bound to the active event loop."""
    global _async_init_lock
    if _async_init_lock is None:
        _async_init_lock = asyncio.Lock()
    return _async_init_lock

def get_tws_status_store() -> TWSStatusStore:
    """Get the singleton TWSStatusStore instance."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = TWSStatusStore()
    return _instance

async def initialize_tws_status_store() -> TWSStatusStore:
    """Initialize and return the TWSStatusStore safely with a lock."""
    store = get_tws_status_store()
    async with _get_async_init_lock():
        if not store._initialized:
            await store.initialize()
    return store

# Alias compatibility
def get_status_store() -> "TWSStatusStore":
    """Alias for get_tws_status_store — backward compat."""
    return get_tws_status_store()

async def init_status_store(**kwargs) -> "TWSStatusStore":
    """Initialize and return the TWSStatusStore singleton."""
    return await initialize_tws_status_store()
