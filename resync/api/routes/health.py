from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from resync.core.database.engine import get_engine
from resync.core.valkey_init import get_redis_client
from sqlalchemy import text

router = APIRouter(tags=["Health"])

@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

@router.get("/health/ready")
async def ready() -> dict[str, str]:
    # P1-17 FIX: Enforce strict timeouts on readiness checks
    
    # Database check with 3-second timeout
    try:
        engine = get_engine()
        async with asyncio.timeout(3.0):
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Database readiness timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database readiness failed: {e}"
        )
    
    # Redis readiness: required if APP_VALKEY_URL is set in production
    redis_required = bool(os.getenv('APP_VALKEY_URL')) and os.getenv('APP_ENVIRONMENT','').lower() in {'prod','production'}
    
    if redis_required:
        try:
            async with asyncio.timeout(2.0):
                redis = get_redis_client()
                await redis.ping()
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Redis readiness timeout")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Redis connection failed: {e}")
    
    return {"status": "ready", "ts": datetime.now(timezone.utc).isoformat()}
