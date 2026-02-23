# pylint: disable=all
# mypy: no-rerun
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resync.models.tws import (
    CriticalJob,
    DependencyTree,
    Event,
    JobDetails,
    JobExecution,
    JobStatus,
    PerformanceData,
    PlanDetails,
    ResourceStatus,
    SystemStatus,
    WorkstationStatus,
)

logger = logging.getLogger(__name__)


class MockTWSClient:
    """
    A mock client for the HCL Workload Automation (TWS) API, used for
    development and testing without a live TWS connection.
    It loads static data from a JSON file.

    Args:
        *args: Additional positional arguments (unused)
        **kwargs: Additional keyword arguments (unused)

    Attributes:
        mock_data (Dict[str, Any]): The loaded mock data from the JSON file
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MockTWSClient with default settings."""
        self.mock_data: dict[str, Any] = {}
        self._load_mock_data()
        logger.info("MockTWSClient initialized. Using static mock data.")

    def _load_mock_data(self) -> None:
        """
        Load mock data from a JSON file.

        This method:
        1. Constructs the path to the mock data file
        2. Checks if the file exists
        3. Tries to load and parse the JSON data
        4. Handles different types of errors gracefully

        Returns:
            None
        """
        mock_data_path = Path(__file__).parent.parent.parent / "mock_tws_data.json"
        if not mock_data_path.exists():
            logger.warning(
                "Mock data file not found at %s. Returning empty data.", mock_data_path
            )
            return

        try:
            with mock_data_path.open(encoding="utf-8") as f:
                self.mock_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode mock data JSON from %s: %s",
                mock_data_path,
                e,
            )
            self.mock_data = {}
            # Don't raise here to allow the service to continue with empty data
        except FileNotFoundError as e:
            logger.error("Mock data file not found at %s: %s", mock_data_path, e)
            self.mock_data = {}
            # Don't raise here to allow the service to continue with empty data
        except PermissionError as e:
            logger.error(
                "Permission denied accessing mock data file at %s: %s",
                mock_data_path,
                e,
            )
            self.mock_data = {}
            # Don't raise here to allow the service to continue with empty data
        except (OSError, IsADirectoryError) as e:
            logger.error("Failed to access mock data file at %s: %s", mock_data_path, e)
            self.mock_data = {}
            # Don't raise here to allow the service to continue with empty data
        except UnicodeDecodeError as e:
            logger.error(
                "Unicode decode error reading mock data file at %s: %s",
                mock_data_path,
                e,
            )
            self.mock_data = {}
            # Don't raise here to allow the service to continue with empty data
        except Exception as e:
            logger.error(
                "Unexpected error loading mock data from %s: %s",
                mock_data_path,
                e,
            )
            self.mock_data = {}
            # In a production environment, consider raising a FileProcessingError here
            # but for a mock service, it's better to continue with empty data

    async def check_connection(self) -> bool:
        """
        Mocks checking the connection status.

        Returns:
            bool: The connection status from the mock data

        Note:
            Simulates an asynchronous network delay with a 0.1 second wait

        Example:
            >>> client = MockTWSClient()
            >>> status = client.check_connection()
            >>> print(status)
            True
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        connection_status = self.mock_data.get("connection_status")
        return bool(connection_status) if connection_status is not None else False

    async def ping(self) -> None:
        """
        Performs a lightweight connectivity test for the mock TWS server.

        This method simulates a ping operation for health checks.

        Raises:
            ConnectionError: If the mock server is configured as unreachable
            TimeoutError: If the mock server is configured to timeout
        """
        await asyncio.sleep(0.05)  # Simulate quick network delay

        # Check mock configuration for ping behavior
        ping_config = self.mock_data.get("ping_config", {})

        if ping_config.get("timeout", False):
            raise TimeoutError("Mock TWS server ping timed out")

        if not ping_config.get("reachable", True):
            raise ConnectionError("Mock TWS server is unreachable")

        # Ping successful for mock
        return

    async def get_workstations_status(self) -> list[WorkstationStatus]:
        """
        Mocks retrieving workstation status.

        Returns:
            List[WorkstationStatus]: List of workstation status objects

        Note:
            Simulates an asynchronous delay with a 0.1 second wait
        """
        await asyncio.sleep(0.1)
        workstations_data = self.mock_data.get("workstations_status", [])
        workstations = []
        for ws in workstations_data:
            if isinstance(ws, dict):
                try:
                    workstations.append(WorkstationStatus(**ws))
                except Exception as e:
                    logger.warning(
                        "Failed to create WorkstationStatus from data: %s", e
                    )
        return workstations

    async def get_jobs_status(self) -> list[JobStatus]:
        """
        Mocks retrieving job status.

        Returns:
            List[JobStatus]: List of job status objects

        Note:
            Simulates an asynchronous delay with a 0.1 second wait
        """
        await asyncio.sleep(0.1)
        jobs_data = self.mock_data.get("jobs_status", [])
        jobs = []
        for job in jobs_data:
            if isinstance(job, dict):
                try:
                    jobs.append(JobStatus(**job))
                except Exception as e:
                    logger.warning("Failed to create JobStatus from data: %s", e)
        return jobs

    async def get_critical_path_status(self) -> list[CriticalJob]:
        """
        Mocks retrieving critical path status.

        Returns:
            List[CriticalJob]: List of critical job status objects

        Note:
            Simulates an asynchronous delay with a 0.1 second wait
        """
        await asyncio.sleep(0.1)
        critical_jobs = []
        for job in self.mock_data.get("critical_path_status", []):
            if isinstance(job, dict):
                try:
                    critical_jobs.append(CriticalJob(**job))
                except Exception as e:
                    logger.warning("Failed to create CriticalJob from data: %s", e)
        return critical_jobs

    async def get_system_status(self) -> SystemStatus:
        """
        Mocks retrieving comprehensive system status.

        Returns:
            SystemStatus: Object containing the overall system status

        Note:
            Aggregates status data from multiple sources with appropriate delays
        """
        workstations = await self.get_workstations_status()
        jobs = await self.get_jobs_status()
        critical_jobs = await self.get_critical_path_status()
        return SystemStatus(
            workstations=workstations, jobs=jobs, critical_jobs=critical_jobs
        )

    async def restart_job(self, job_id: str) -> dict[str, Any]:
        """
        Mocks restarting a job.

        Args:
            job_id: The ID of the job to restart

        Returns:
            Dict containing job restart information

        Note:
            Simulates an asynchronous delay and returns mock restart data
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            "job_id": job_id,
            "action": "restarted",
            "status": "PENDING",
            "timestamp": "2024-01-01T12:00:00Z",
        }

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """
        Mocks canceling a job.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            Dict containing job cancellation information

        Note:
            Simulates an asynchronous delay and returns mock cancellation data
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        return {
            "job_id": job_id,
            "action": "canceled",
            "status": "CANCELED",
            "timestamp": "2024-01-01T12:00:00Z",
        }

    async def get_job_details(self, job_id: str) -> JobDetails:
        """
        Mocks getting detailed information about a specific job.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        # Find job data or create mock data
        jobs_data = self.mock_data.get("jobs_status", [])
        job_data = None
        for job in jobs_data:
            if job.get("name") == job_id or job.get("id") == job_id:
                job_data = job
                break

        if not job_data:
            # Create mock job data
            job_data = {
                "name": job_id,
                "workstation": "CPU_WS",
                "status": "SUCC",
                "job_stream": "STREAM_A",
                "full_definition": {
                    "name": job_id,
                    "workstation": "CPU_WS",
                    "status": "SUCC",
                    "job_stream": "STREAM_A",
                    "script": f"#!/bin/bash\necho 'Running job {job_id}'",
                    "dependencies": ["PREV_JOB"],
                    "resource_requirements": {"cpu": "1", "memory": "512MB"},
                },
                "dependencies": ["PREV_JOB"],
                "resource_requirements": {"cpu": "1", "memory": "512MB"},
            }

        # Get execution history
        history = await self.get_job_history(job_id)

        # Get dependencies
        dependencies = await self.get_job_dependencies(job_id)

        return JobDetails(
            job_id=job_id,
            name=job_data.get("name", job_id),
            workstation=job_data.get("workstation", "CPU_WS"),
            status=job_data.get("status", "SUCC"),
            job_stream=job_data.get("job_stream", "STREAM_A"),
            full_definition=job_data.get("full_definition", {}),
            dependencies=dependencies.dependencies,
            resource_requirements=job_data.get("resource_requirements", {}),
            execution_history=history[:10],  # Limit to last 10 executions
        )

    async def get_job_history(self, job_name: str) -> list[JobExecution]:
        """
        Mocks getting the execution history for a specific job.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        # Return mock history data
        return [
            JobExecution(
                job_id=job_name,
                status="SUCC",
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration="5m",
                error_message=None,
            ),
            JobExecution(
                job_id=job_name,
                status="RUNNING",
                start_time=datetime.now(timezone.utc),
                end_time=None,
                duration=None,
                error_message=None,
            ),
        ]

    async def get_job_log(self, job_id: str) -> str:
        """
        Mocks getting the log content for a specific job execution.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        return (
            f"Mock log content for job {job_id}\n"
            "Job started successfully\n"
            "Processing data...\n"
            "Job completed"
        )

    async def get_plan_details(self) -> PlanDetails:
        """
        Mocks getting details about the current TWS plan.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        return PlanDetails(
            plan_id="CURRENT_PLAN",
            creation_date=datetime.now(timezone.utc),
            jobs_count=len(self.mock_data.get("jobs_status", [])),
            estimated_completion=datetime.now(timezone.utc),
            status="ACTIVE",
        )

    async def get_job_dependencies(self, job_id: str) -> DependencyTree:
        """
        Mocks getting the dependency tree for a specific job.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        return DependencyTree(
            job_id=job_id,
            dependencies=["PREV_JOB"],
            dependents=["NEXT_JOB"],
            dependency_graph={
                "PREV_JOB": [],
                job_id: ["PREV_JOB"],
                "NEXT_JOB": [job_id],
            },
        )

    async def get_resource_usage(self) -> list[ResourceStatus]:
        """
        Mocks getting resource usage information.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        return [
            ResourceStatus(
                resource_name="CPU_WS",
                resource_type="WORKSTATION",
                total_capacity=100.0,
                used_capacity=75.0,
                available_capacity=25.0,
                utilization_percentage=75.0,
            ),
            ResourceStatus(
                resource_name="MEMORY_POOL",
                resource_type="MEMORY",
                total_capacity=8192.0,
                used_capacity=6144.0,
                available_capacity=2048.0,
                utilization_percentage=75.0,
            ),
        ]

    async def get_event_log(self, last_hours: int = 24) -> list[Event]:
        """
        Mocks getting TWS event log entries.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        events_data: list[dict[str, Any]] = [
            {
                "event_id": "EVENT_001",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "JOB_STARTED",
                "severity": "INFO",
                "source": "TWS_ENGINE",
                "message": "Job JOB_A started successfully",
                "job_id": "JOB_A",
                "workstation": "CPU_WS",
            },
            {
                "event_id": "EVENT_002",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "JOB_COMPLETED",
                "severity": "INFO",
                "source": "TWS_ENGINE",
                "message": "Job JOB_A completed successfully",
                "job_id": "JOB_A",
                "workstation": "CPU_WS",
            },
        ]

        return [Event(**event_data) for event_data in events_data]

    async def get_performance_metrics(self) -> PerformanceData:
        """
        Mocks getting TWS performance metrics.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        return PerformanceData(
            timestamp=datetime.now(timezone.utc),
            api_response_times={
                "get_system_status": 0.5,
                "get_jobs_status": 0.3,
            },
            cache_hit_rate=85.5,
            memory_usage_mb=256.0,
            cpu_usage_percentage=45.0,
            active_connections=5,
            jobs_per_minute=12.5,
        )

    async def get_job_status(self, job_id: str) -> JobStatus:
        """
        Mocks getting the status of a specific job.

        Args:
            job_id: The ID of the job to get status for

        Returns:
            JobStatus: The status of the requested job

        Note:
            Simulates an asynchronous delay and returns mock job status
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        # Find a job that matches the job_id (for simplicity, return first job)
        jobs = self.mock_data.get("jobs_status", [])
        if jobs:
            return JobStatus(**jobs[0])
        # Return a default job if none found
        return JobStatus(
            name=f"JOB_{job_id}",
            workstation="CPU_WS",
            status="SUCC",
            job_stream="STREAM_A",
        )

    async def list_jobs(self, status_filter: str | None = None) -> list[JobStatus]:
        """
        Mocks listing all jobs, optionally filtered by status.

        Args:
            status_filter: Optional status filter (e.g., 'SUCC', 'ABEND')

        Returns:
            List[JobStatus]: List of job status objects

        Note:
            Simulates an asynchronous delay and returns filtered mock jobs
        """
        await asyncio.sleep(0.1)  # Simulate network delay
        jobs = [JobStatus(**job) for job in self.mock_data.get("jobs_status", [])]

        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]

        return jobs

    async def validate_connection(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        """
        Mocks validating TWS connection parameters.

        Args:
            host: TWS server hostname (optional)
            port: TWS server port (optional)
            user: TWS username (optional)
            password: TWS password (optional)

        Returns:
            Dictionary with validation result
        """
        # Simulate validation process
        await asyncio.sleep(0.05)  # Simulate short network delay for validation

        # Get validation config from mock data or default to success
        validation_config = self.mock_data.get("validation_config", {})
        validation_success = validation_config.get("connection_valid", True)

        if validation_success:
            return {
                "valid": True,
                "message": "Successfully validated connection to mock TWS server",
                "host": host or "mock-tws-server",
                "port": port or 31111,
            }
        return {
            "valid": False,
            "message": "Mock TWS connection validation failed as configured",
            "host": host or "mock-tws-server",
            "port": port or 31111,
        }

    def close(self) -> None:
        """
        Mocks closing the client connection.

        Note:
            Simulates proper cleanup of client resources
        """
        logger.info("MockTWSClient closed.")
