"""PostgreSQL-backed Document Knowledge Graph (DKG) store.

Design goals:
- Postgres-only to match Resync stack
- multi-tenant + versioned via graph_version
- evidence JSONB for explainability
- Connection pool shared across operations (not connect-per-call)

This module uses asyncpg (already used in resync.knowledge.store).
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterable
from resync.knowledge.config import CFG
from resync.knowledge.kg_store.ddl import DDL_STATEMENTS

logger = logging.getLogger(__name__)
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    ASYNCPG_AVAILABLE = False


@dataclass(frozen=True)
class KGNode:
    node_id: str
    node_type: str
    name: str
    aliases: list[str] | None = None
    properties: dict[str, Any] | None = None


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.5
    evidence: dict[str, Any] | None = None
    edge_id: str | None = None


class PostgresGraphStore:
    """Async Postgres store for nodes/edges with shared connection pool."""

    _pool: asyncpg.Pool | None = None

    def __init__(self, database_url: str | None = None):
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required. pip install asyncpg")
        self._database_url = database_url or CFG.database_url
        if self._database_url.startswith("postgresql+asyncpg://"):
            self._database_url = self._database_url.replace(
                "postgresql+asyncpg://", "postgresql://"
            )
        self._schema_ensured = False

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create a shared connection pool."""
        if PostgresGraphStore._pool is None or PostgresGraphStore._pool._closed:
            PostgresGraphStore._pool = await asyncpg.create_pool(
                self._database_url, min_size=2, max_size=10, command_timeout=30
            )
        return PostgresGraphStore._pool

    async def ensure_schema(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        if self._schema_ensured:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            for stmt in DDL_STATEMENTS:
                await conn.execute(stmt)
        self._schema_ensured = True

    async def close(self) -> None:
        """Close the shared pool. Call on app shutdown."""
        if PostgresGraphStore._pool is not None and (
            not PostgresGraphStore._pool._closed
        ):
            await PostgresGraphStore._pool.close()
            PostgresGraphStore._pool = None

    async def upsert_nodes(
        self, *, tenant: str, graph_version: int, nodes: Iterable[KGNode]
    ) -> int:
        await self.ensure_schema()
        nodes_list = list(nodes)
        if not nodes_list:
            return 0
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                "\n                INSERT INTO kg_nodes (tenant, graph_version, node_id, node_type, name, aliases, properties)\n                VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb)\n                ON CONFLICT (tenant, graph_version, node_id)\n                DO UPDATE SET\n                    node_type = EXCLUDED.node_type,\n                    name = EXCLUDED.name,\n                    aliases = EXCLUDED.aliases,\n                    properties = EXCLUDED.properties\n                ",
                [
                    (
                        tenant,
                        graph_version,
                        n.node_id,
                        n.node_type,
                        n.name,
                        asyncpg.types.Json(n.aliases or []),
                        asyncpg.types.Json(n.properties or {}),
                    )
                    for n in nodes_list
                ],
            )
            return len(nodes_list)

    async def insert_edges(
        self, *, tenant: str, graph_version: int, edges: Iterable[KGEdge]
    ) -> int:
        await self.ensure_schema()
        edges_list = list(edges)
        if not edges_list:
            return 0
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                "\n                INSERT INTO kg_edges (tenant, graph_version, edge_id, source_id, target_id, relation_type, weight, evidence)\n                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)\n                ",
                [
                    (
                        tenant,
                        graph_version,
                        e.edge_id or str(uuid.uuid4()),
                        e.source_id,
                        e.target_id,
                        e.relation_type,
                        float(e.weight),
                        asyncpg.types.Json(e.evidence or {}),
                    )
                    for e in edges_list
                ],
            )
            return len(edges_list)

    async def upsert_from_extraction(
        self, *, tenant: str, graph_version: int, doc_id: str, extraction: Any
    ) -> dict[str, int]:
        """Convenience method: persist an ExtractionResult (concepts + edges).

        Returns dict with counts: {"nodes": N, "edges": M}.
        """
        from resync.knowledge.kg_extraction.normalizer import make_node_id

        kg_nodes = [
            KGNode(
                node_id=make_node_id(c.node_type, c.name),
                node_type=c.node_type,
                name=c.name,
                aliases=c.aliases,
                properties={**(c.properties or {}), "doc_id": doc_id},
            )
            for c in extraction.concepts or []
        ]
        kg_edges = [
            KGEdge(
                source_id=make_node_id("Concept", e.source),
                target_id=make_node_id("Concept", e.target),
                relation_type=e.relation_type,
                weight=e.weight,
                evidence=e.evidence.model_dump() if e.evidence else {"doc_id": doc_id},
            )
            for e in extraction.edges or []
        ]
        n_nodes = await self.upsert_nodes(
            tenant=tenant, graph_version=graph_version, nodes=kg_nodes
        )
        n_edges = await self.insert_edges(
            tenant=tenant, graph_version=graph_version, edges=kg_edges
        )
        logger.info(
            "kg_extraction_persisted",
            extra={"doc_id": doc_id, "nodes": n_nodes, "edges": n_edges},
        )
        return {"nodes": n_nodes, "edges": n_edges}

    async def stats(self, *, tenant: str, graph_version: int) -> dict[str, Any]:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            nodes = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_nodes WHERE tenant=$1 AND graph_version=$2",
                tenant,
                graph_version,
            )
            edges = await conn.fetchval(
                "SELECT COUNT(*) FROM kg_edges WHERE tenant=$1 AND graph_version=$2",
                tenant,
                graph_version,
            )
            return {
                "tenant": tenant,
                "graph_version": graph_version,
                "nodes": int(nodes),
                "edges": int(edges),
            }

    async def find_nodes_by_name(
        self, *, tenant: str, graph_version: int, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fuzzy search nodes by name (requires pg_trgm)."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "\n                SELECT node_id, node_type, name, aliases, properties\n                FROM kg_nodes\n                WHERE tenant=$1 AND graph_version=$2\n                  AND name % $3\n                ORDER BY similarity(name, $3) DESC\n                LIMIT $4\n                ",
                tenant,
                graph_version,
                query,
                limit,
            )
            return [
                {
                    "node_id": r["node_id"],
                    "node_type": r["node_type"],
                    "name": r["name"],
                    "aliases": r["aliases"],
                    "properties": r["properties"],
                }
                for r in rows
            ]

    async def get_subgraph(
        self,
        *,
        tenant: str,
        graph_version: int,
        seed_node_ids: list[str],
        depth: int = 2,
        max_edges: int = 200,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a subgraph around seed nodes using recursive CTE."""
        await self.ensure_schema()
        if not seed_node_ids:
            return {"nodes": [], "edges": []}
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            params: list[Any] = [tenant, graph_version, seed_node_ids, depth]
            doc_filter = ""
            if doc_id:
                doc_filter = "AND (evidence->>'doc_id') = $5"
                params.append(doc_id)
            rows = await conn.fetch(
                f"\n                WITH RECURSIVE walk AS (\n                    SELECT\n                        e.source_id,\n                        e.target_id,\n                        e.relation_type,\n                        e.weight,\n                        e.evidence,\n                        1 AS lvl,\n                        ARRAY[e.source_id, e.target_id] AS path\n                    FROM kg_edges e\n                    WHERE e.tenant=$1 AND e.graph_version=$2\n                      AND (e.source_id = ANY($3) OR e.target_id = ANY($3))\n                      {doc_filter}\n\n                    UNION ALL\n\n                    SELECT\n                        e.source_id,\n                        e.target_id,\n                        e.relation_type,\n                        e.weight,\n                        e.evidence,\n                        w.lvl + 1 AS lvl,\n                        w.path || ARRAY[e.source_id, e.target_id] AS path\n                    FROM kg_edges e\n                    JOIN walk w\n                      ON (e.source_id = w.target_id OR e.target_id = w.target_id)\n                    WHERE e.tenant=$1 AND e.graph_version=$2\n                      AND w.lvl < $4\n                      AND NOT (e.target_id = ANY(w.path))\n                      {doc_filter}\n                )\n                SELECT source_id, target_id, relation_type, weight, evidence\n                FROM walk\n                LIMIT {int(max_edges)}\n                ",
                *params,
            )
            edges = [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "relation_type": r["relation_type"],
                    "weight": float(r["weight"]),
                    "evidence": r["evidence"],
                }
                for r in rows
            ]
            node_ids = set(seed_node_ids)
            for e in edges:
                node_ids.add(e["source_id"])
                node_ids.add(e["target_id"])
            node_rows = await conn.fetch(
                "\n                SELECT node_id, node_type, name, aliases, properties\n                FROM kg_nodes\n                WHERE tenant=$1 AND graph_version=$2 AND node_id = ANY($3)\n                ",
                tenant,
                graph_version,
                list(node_ids),
            )
            nodes = [
                {
                    "node_id": n["node_id"],
                    "node_type": n["node_type"],
                    "name": n["name"],
                    "aliases": n["aliases"],
                    "properties": n["properties"],
                }
                for n in node_rows
            ]
            return {"nodes": nodes, "edges": edges}
