from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TeamsNotification:
    """Standardized Teams notification structure."""

    title: str
    message: str
    severity: str  # "info", "warning", "error"
    job_id: str | None = None
    job_status: str | None = None
    instance_name: str | None = None
    additional_data: dict[str, Any] | None = None


class NotificationError(Exception):
    """Exception raised for notification errors."""


def create_job_status_notification(
    job_data: dict[str, Any], instance_name: str, job_status_filters: list[str]
) -> TeamsNotification:
    """
    Create a standardized job status notification.

    Args:
        job_data: Dictionary containing job information
        instance_name: Name of the TWS instance
        job_status_filters: List of status filters to match against

    Returns:
        TeamsNotification: Standardized notification object
    """
    job_status = job_data.get("status", "").upper()

    # Check if job status matches filters
    if job_status in [status.upper() for status in job_status_filters]:
        return TeamsNotification(
            title=f"Job Status Alert: {job_data.get('job_name', 'Unknown Job')}",
            message=f"Job {job_data.get('job_name', 'Unknown Job')} entered status {job_status}",
            severity="error" if job_status == "ABEND" else "warning",
            job_id=job_data.get("job_id"),
            job_status=job_status,
            instance_name=instance_name,
            additional_data={
                "start_time": job_data.get("start_time"),
                "end_time": job_data.get("end_time"),
                "duration": job_data.get("duration"),
                "owner": job_data.get("owner"),
            },
        )

    return None
