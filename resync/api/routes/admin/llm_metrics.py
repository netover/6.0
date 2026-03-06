from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.litellm_hooks import VALKEY_LLM_METRICS_HASH, VALKEY_LLM_RECENT_LIST
from resync.core.valkey_init import get_valkey_client, is_valkey_available

router = APIRouter(prefix="/llm", tags=["Admin - LLM"])
logger = logging.getLogger(__name__)


class LLMMetricsSummary(BaseModel):
    requests: int = 0
    errors: int = 0
    fallbacks: int = 0
    fallbacks_rate_limit: int = 0
    fallbacks_timeout: int = 0
    fallbacks_provider_5xx: int = 0
    fallbacks_network: int = 0
    fallbacks_other: int = 0
    tokens_total: int = 0
    latency_ms_total: int = 0


class LLMMetricsResponse(BaseModel):
    summary: LLMMetricsSummary = Field(default_factory=LLMMetricsSummary)
    recent: list[dict[str, Any]] = Field(default_factory=list)


@router.get(
    "/metrics",
    response_model=LLMMetricsResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_llm_metrics() -> LLMMetricsResponse:
    """Return LiteLLM metrics collected via callbacks (Valkey-backed)."""
    if not is_valkey_available():
        return LLMMetricsResponse()

    client = get_valkey_client()
    if client is None:
        return LLMMetricsResponse()

    h = await client.hgetall(VALKEY_LLM_METRICS_HASH)
    # valkey returns bytes; normalize
    def _i(x: Any) -> int:
        if x is None:
            return 0
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", "ignore")
        try:
            return int(x)
        except (ValueError, TypeError):
            return 0

    summary = LLMMetricsSummary(
        requests=_i(h.get(b"requests") if isinstance(h, dict) else h.get("requests")),
        errors=_i(h.get(b"errors") if isinstance(h, dict) else h.get("errors")),
        fallbacks=_i(h.get(b"fallbacks_total") if isinstance(h, dict) else h.get("fallbacks_total")),
        fallbacks_rate_limit=_i(h.get(b"fallbacks_rate_limit") if isinstance(h, dict) else h.get("fallbacks_rate_limit")),
        fallbacks_timeout=_i(h.get(b"fallbacks_timeout") if isinstance(h, dict) else h.get("fallbacks_timeout")),
        fallbacks_provider_5xx=_i(h.get(b"fallbacks_provider_5xx") if isinstance(h, dict) else h.get("fallbacks_provider_5xx")),
        fallbacks_network=_i(h.get(b"fallbacks_network") if isinstance(h, dict) else h.get("fallbacks_network")),
        fallbacks_other=_i(h.get(b"fallbacks_other") if isinstance(h, dict) else h.get("fallbacks_other")),
        tokens_total=_i(h.get(b"tokens_total") if isinstance(h, dict) else h.get("tokens_total")),
        latency_ms_total=_i(h.get(b"latency_ms_total") if isinstance(h, dict) else h.get("latency_ms_total")),
    )

    raw = await client.lrange(VALKEY_LLM_RECENT_LIST, 0, 50)
    recent: list[dict[str, Any]] = []
    import json
    for r in raw:
        if isinstance(r, (bytes, bytearray)):
            r = r.decode("utf-8", "ignore")
        try:
            obj = json.loads(r)
            if isinstance(obj, dict):
                recent.append(obj)
        except Exception:
            logger.debug("llm_metrics_recent_item_decode_failed", exc_info=True)
            continue

    return LLMMetricsResponse(summary=summary, recent=recent)
