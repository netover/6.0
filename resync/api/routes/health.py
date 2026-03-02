from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from fastapi import APIRouter

from resync.core.database.engine import async_engine
from resync.core.redis_init import get_redis_client

router = APIRouter(tags=["Health"])

@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

@router.get("/health/ready")
async def ready() -> dict[str, str]:
    async with async_engine.begin() as conn:
        await conn.execute("SELECT 1")  # type: ignore[arg-type]
    # Redis readiness: required if APP_REDIS_URL is set in production
    redis_required = bool(os.getenv('APP_REDIS_URL')) and os.getenv('APP_ENV','').lower() in {'prod','production'}
    try:
        redis = get_redis_client()
        await redis.ping()
    except asyncio.CancelledError:
        raise
    except Exception:
        if redis_required:
            raise
    return {"status": "ready", "ts": datetime.now(timezone.utc).isoformat()}
