"""Integração do monitor de jobs com Teams Notifications."""

from resync.core.database.session import get_db
from resync.core.teams_notifier import TeamsNotificationManager


async def notify_job_event(job_data: dict):
    """
    Notifica evento de job no Teams.

    Chamado automaticamente pelo monitor quando job tem ABEND, ERROR, etc.
    """
    db = next(get_db())
    manager = TeamsNotificationManager(db)

    await manager.send_job_notification(
        job_name=job_data.get('job_name'),
        job_status=job_data.get('status'),
        instance_name=job_data.get('instance_name'),
        return_code=job_data.get('return_code'),
        error_message=job_data.get('error_message'),
        timestamp=job_data.get('timestamp')
    )
