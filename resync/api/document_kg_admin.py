"""Admin API for Document Knowledge Graph.

Mounted under /api/admin/kg (see app_factory.py integration).

Endpoints:
- GET /api/admin/kg/health
- GET /api/admin/kg/stats?tenant=...&graph_version=...
- GET /api/admin/kg/subgraph?tenant=...&graph_version=...&seed=...&depth=...
"""

from __future__ import annotations
# mypy

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from resync.api.routes.core.auth import verify_admin_credentials
from resync.knowledge.kg_store.store import PostgresGraphStore

def get_kg_store(request: Request) -> PostgresGraphStore:
    """Retorna singleton já inicializado do app state."""
    return request.app.state.enterprise_state.kg_store

router = APIRouter(prefix="/api/admin/kg", tags=["KG Admin"])

@router.get("/health", dependencies=[Depends(verify_admin_credentials)])
async def kg_health(
    store: PostgresGraphStore = Depends(get_kg_store),
) -> dict:
    ok = await store.ping()
    return {"status": "ok" if ok else "degraded"}

@router.get("/stats", dependencies=[Depends(verify_admin_credentials)])
async def kg_stats(
    store: PostgresGraphStore = Depends(get_kg_store),
    tenant: Annotated[str, Query()] = "default",
    graph_version: Annotated[int, Query()] = 1,
) -> dict:
    return await store.stats(tenant=tenant, graph_version=graph_version)

@router.get("/subgraph", dependencies=[Depends(verify_admin_credentials)])
async def kg_subgraph(
    seed: Annotated[list[str], Query(description="node_id(s) or name(s)")],
    tenant: Annotated[str, Query()] = "default",
    graph_version: Annotated[int, Query()] = 1,
    depth: Annotated[int, Query(ge=1, le=5)] = 2,
    store: PostgresGraphStore = Depends(get_kg_store),
) -> dict:
    # Normalize seeds: if bare name without type prefix, assume Concept
    seeds: list[str] = []
    for s in seed:
        s = s.strip()
        if ":" not in s:
            seeds.append(f"Concept:{s.lower()}")
        else:
            seeds.append(s)

    return await store.get_subgraph(
        tenant=tenant,
        graph_version=graph_version,
        seed_node_ids=seeds,
        depth=depth,
        max_edges=200,
    )
