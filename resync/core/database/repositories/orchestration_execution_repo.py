# pylint
# mypy
"""
Orchestration Execution Repository

Provides data access methods for orchestration executions.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.models.orchestration import (
    OrchestrationExecution,
    OrchestrationStepRun,
)

class OrchestrationExecutionRepository:
    """
    Repository for orchestration execution data access.

    Provides CRUD operations and queries for orchestration executions.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self._session = session

    async def create(
        self,
        trace_id: str,
        config_id: UUID | None,
        config_name: str,
        input_data: dict,
        user_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        created_by: str | None = None,
        callback_url: str | None = None,
    ) -> OrchestrationExecution:
        """
        Create a new execution record.

        Args:
            trace_id: Unique trace ID for correlation
            config_id: Configuration ID
            config_name: Configuration name
            input_data: Input data for execution
            user_id: User ID
            session_id: Session ID
            tenant_id: Tenant ID
            created_by: Creator user ID
            callback_url: Callback URL for completion notification

        Returns:
            Created execution
        """
        execution = OrchestrationExecution(
            trace_id=trace_id,
            config_id=config_id,
            config_name=config_name,
            input_data=input_data,
            status="pending",
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            created_by=created_by,
            callback_url=callback_url,
        )
        self._session.add(execution)
        await self._session.commit()
        await self._session.refresh(execution)
        return execution

    async def get_by_id(self, execution_id: UUID) -> OrchestrationExecution | None:
        """Get execution by ID."""
        result = await self._session.execute(
            select(OrchestrationExecution).where(
                OrchestrationExecution.id == execution_id
            )
        )
        return result.scalar_one_or_none()

    async def get_by_trace_id(self, trace_id: str) -> OrchestrationExecution | None:
        """Get execution by trace ID."""
        result = await self._session.execute(
            select(OrchestrationExecution).where(
                OrchestrationExecution.trace_id == trace_id
            )
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        execution_id: UUID,
        status: str,
        output: dict | None = None,
        completed_at: datetime | None = None,
        total_latency_ms: int | None = None,
        estimated_cost: float | None = None,
        meta_data_update: dict | None = None,
    ) -> OrchestrationExecution | None:
        """
        Update execution status and metrics.

        Args:
            execution_id: Execution ID
            status: New status
            output: Optional output data
            completed_at: Completion timestamp
            total_latency_ms: Total latency in milliseconds
            estimated_cost: Estimated cost

        Returns:
            Updated execution
        """
        values = {"status": status}

        if meta_data_update:
            current = await self.get_by_id(execution_id)
            merged_meta = dict(getattr(current, 'meta_data', {}) or {}) if current else {}
            merged_meta.update(meta_data_update)
            values["meta_data"] = merged_meta

        if output is not None:
            values["output_data"] = output

        if completed_at:
            values["completed_at"] = completed_at

        if total_latency_ms is not None:
            values["total_latency_ms"] = total_latency_ms

        if estimated_cost is not None:
            values["estimated_cost"] = estimated_cost

        if status == "running" and not output:
            values["started_at"] = datetime.now(timezone.utc)

        await self._session.execute(
            update(OrchestrationExecution)
            .where(OrchestrationExecution.id == execution_id)
            .values(**values)
        )
        await self._session.commit()
        return await self.get_by_id(execution_id)

    async def list_by_config(
        self,
        config_id: UUID,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrchestrationExecution]:
        """
        List executions for a configuration.

        Args:
            config_id: Configuration ID
            status: Optional status filter
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of executions
        """
        query = (
            select(OrchestrationExecution)
            .where(OrchestrationExecution.config_id == config_id)
            .order_by(OrchestrationExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if status:
            query = query.where(OrchestrationExecution.status == status)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def list_by_user(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OrchestrationExecution]:
        """List executions for a user."""
        query = (
            select(OrchestrationExecution)
            .where(OrchestrationExecution.user_id == user_id)
            .order_by(OrchestrationExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if status:
            query = query.where(OrchestrationExecution.status == status)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def get_step_runs(
        self,
        execution_id: UUID,
    ) -> list[OrchestrationStepRun]:
        """Get all step runs for an execution."""
        result = await self._session.execute(
            select(OrchestrationStepRun)
            .where(OrchestrationStepRun.execution_id == execution_id)
            .order_by(OrchestrationStepRun.step_index)
        )
        return list(result.scalars().all())

class OrchestrationStepRunRepository:
    """Repository for step run data access."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        execution_id: UUID,
        step_index: int,
        step_id: str,
        step_name: str | None = None,
        dependencies_json: list | None = None,
    ) -> OrchestrationStepRun:
        """Create a new step run record."""
        step_run = OrchestrationStepRun(
            execution_id=execution_id,
            step_index=step_index,
            step_id=step_id,
            step_name=step_name,
            status="pending",
            dependencies_json=dependencies_json or [],
        )
        self._session.add(step_run)
        await self._session.commit()
        await self._session.refresh(step_run)
        return step_run

    async def update_status(
        self,
        step_run_id: UUID,
        status: str,
        output: dict | None = None,
        output_truncated: str | None = None,
        error_message: str | None = None,
        error_trace: str | None = None,
        latency_ms: int | None = None,
        retry_count: int | None = None,
        token_count: int | None = None,
        estimated_cost: float | None = None,
        meta_data_update: dict | None = None,
    ) -> OrchestrationStepRun | None:
        """Update step run status and metrics."""
        values = {"status": status}

        if meta_data_update:
            current = await self.get_by_id(execution_id)
            merged_meta = dict(getattr(current, 'meta_data', {}) or {}) if current else {}
            merged_meta.update(meta_data_update)
            values["meta_data"] = merged_meta

        if output is not None:
            values["output"] = output

        if output_truncated is not None:
            values["output_truncated"] = output_truncated

        if error_message is not None:
            values["error_message"] = error_message

        if error_trace is not None:
            values["error_trace"] = error_trace

        if latency_ms is not None:
            values["latency_ms"] = latency_ms

        if retry_count is not None:
            values["retry_count"] = retry_count

        if token_count is not None:
            values["token_count"] = token_count

        if estimated_cost is not None:
            values["estimated_cost"] = estimated_cost

        if status == "running" and latency_ms is None:
            values["started_at"] = datetime.now(timezone.utc)

        if status in ("completed", "failed", "skipped"):
            values["completed_at"] = datetime.now(timezone.utc)

        await self._session.execute(
            update(OrchestrationStepRun)
            .where(OrchestrationStepRun.id == step_run_id)
            .values(**values)
        )
        await self._session.commit()

        result = await self._session.execute(
            select(OrchestrationStepRun).where(OrchestrationStepRun.id == step_run_id)
        )
        return result.scalar_one_or_none()
