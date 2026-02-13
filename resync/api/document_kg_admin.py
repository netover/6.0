"""Admin API for Document Knowledge Graph.

Mounted under /api/admin/kg (see app_factory.py integration).

Endpoints:
- GET /api/admin/kg/health
- GET /api/admin/kg/stats?tenant=...&graph_version=...
- GET /api/admin/kg/subgraph?tenant=...&graph_version=...&seed=...&depth=...
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from resync.knowledge.kg_store.store import PostgresGraphStore

router = APIRouter(prefix="/api/admin/kg", tags=["KG Admin"])


@router.get("/health")
async def kg_health() -> dict:
    store = PostgresGraphStore()
    await store.ensure_schema()
    return {"status": "ok"}


@router.get("/stats")
async def kg_stats(
    tenant: str = Query("default"),
    graph_version: int = Query(1),
) -> dict:
    store = PostgresGraphStore()
    return await store.stats(tenant=tenant, graph_version=graph_version)


@router.get("/subgraph")
async def kg_subgraph(
    tenant: str = Query("default"),
    graph_version: int = Query(1),
    seed: list[str] = Query(..., description="node_id(s) or name(s)"),
    depth: int = Query(2, ge=1, le=5),
) -> dict:
    store = PostgresGraphStore()

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
