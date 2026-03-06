from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog

from resync.core.database import Feedback, get_session
from resync.settings import get_settings

logger = structlog.get_logger(__name__)


async def run_feedback_retention_loop() -> None:
    """Delete old feedback rows if retention is configured."""
    settings = get_settings()
    days = int(getattr(settings, "feedback_retention_days", 0) or 0)
    if days <= 0:
        return
    interval = int(getattr(settings, "feedback_retention_interval_seconds", 3600) or 3600)

    from resync.core.loop_utils import run_resilient_loop

    async def _step() -> None:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            async with get_session() as session:
                stmt = Feedback.__table__.delete().where(Feedback.created_at < cutoff)  # type: ignore[attr-defined]
                await session.execute(stmt)
                await session.commit()
            logger.info("feedback_retention_purge_completed", cutoff=cutoff.isoformat())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("feedback_retention_purge_failed", error=str(e))
        await asyncio.sleep(interval)

    await run_resilient_loop("feedback_retention.loop", _step, logger=logger, step_timeout_seconds=None)
