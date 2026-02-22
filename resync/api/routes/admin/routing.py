from fastapi import APIRouter, Depends, Query
from typing import Any
from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.metrics.runtime_metrics import runtime_metrics

router = APIRouter(prefix="/admin/routing", tags=["Admin", "Routing"])


@router.get("/recent", summary="Get recent routing decisions")
async def get_recent_decisions(
    limit: int = Query(50, ge=1, le=50), _: Any = Depends(verify_admin_credentials)
):
    """
    Returns the last N routing decisions from the ring buffer.

    Provides insights into how messages are being classified and routed
    among RAG-only, Agentic, and Diagnostic paths.
    """
    stats = runtime_metrics.get_stats()
    decisions = stats.get("routing", {}).get("recent_decisions", [])
    # Reversing to show most recent first
    return {"decisions": list(reversed(decisions[-limit:]))}


@router.get("/stats", summary="Get routing statistics")
async def get_routing_stats(_: Any = Depends(verify_admin_credentials)):
    """
    Returns aggregated routing statistics.

    Includes total decisions, errors, and average latency.
    """
    stats = runtime_metrics.get_stats()
    return stats.get("routing", {})
