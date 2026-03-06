from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from resync.core.database.engine import get_engine
from resync.core.valkey_init import get_valkey_client
from sqlalchemy import text
from resync.core.litellm_init import get_litellm_router
from resync.core.async_utils import with_timeout, classify_exception
from resync.settings import get_settings

router = APIRouter(tags=["Health"])

@router.get("/health/live")
async def live() -> dict[str, str]:
    """Cheap liveness probe."""
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

@router.get("/health/ready")
async def ready() -> dict[str, str]:
    settings = get_settings()
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
    
    # Valkey readiness: required if APP_VALKEY_URL is set in production
    valkey_required = bool(os.getenv('APP_VALKEY_URL')) and os.getenv('APP_ENVIRONMENT','').lower() in {'prod','production'}
    
    if valkey_required:
        try:
            async with asyncio.timeout(2.0):
                valkey = get_valkey_client()
                if not valkey:
                    raise RuntimeError('Valkey client not available')
                await with_timeout(valkey.ping(), getattr(settings, 'valkey_health_timeout', 2.0), op='valkey.ping')
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Valkey readiness timeout")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Valkey connection failed: {e}")
    
    return {"status": "ready", "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/health/llm")
async def llm_health(deep: bool = False) -> dict[str, str]:
    """Lightweight LLM health.

    - Does NOT disclose configuration details.
    - Does NOT perform external network calls (keeps it cheap).
    For deep checks, use the Admin LiteLLM health endpoints.
    """
    router_obj = get_litellm_router()
    if router_obj is None:
        raise HTTPException(status_code=503, detail="LLM router not initialized")
    if deep:
        # Deep check performs a tiny external call with a strict timeout.
        try:
            async with asyncio.timeout(5.0):
                model = os.getenv("APP_LLM_MODEL") or os.getenv("LITELLM_DEFAULT_MODEL") or "openrouter/free"
                await router_obj.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="LLM deep health timeout")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM deep health failed: {e}")
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}