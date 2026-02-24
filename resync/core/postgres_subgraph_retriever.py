# pylint: disable=all
"""Postgres-based subgraph retriever for Document KG.

Returns a small connected subgraph for a set of seed node_ids or names.
Uses a recursive CTE traversal over kg_edges.
"""

import logging
from typing import Any

from resync.knowledge.kg_store.store import PostgresGraphStore

logger = logging.getLogger(__name__)


class PostgresSubgraphRetriever:
    def __init__(self, store: PostgresGraphStore | None = None):
        self.store = store or PostgresGraphStore()

    async def get_subgraph_for_seeds(
        self,
        *,
        tenant: str,
        graph_version: int,
        seeds: list[str],
        depth: int = 2,
        limit_edges: int = 200,
    ) -> dict[str, Any]:
        return await self.store.get_subgraph(
            tenant=tenant,
            graph_version=graph_version,
            seed_node_ids=seeds,
            depth=depth,
            limit_edges=limit_edges,
        )

    @staticmethod
    def format_for_llm(subgraph: dict[str, Any]) -> str:
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if not nodes:
            return ""

        # Compact textual representation
        lines: list[str] = []
        lines.append("[DOCUMENT KNOWLEDGE GRAPH]\n")
        lines.append("Nodes:")
        for n in nodes[:80]:
            lines.append(
                f"- {n.get('node_id')} ({n.get('node_type')}): {n.get('name')}"
            )
        lines.append("\nEdges:")
        for e in edges[:200]:
            ev = e.get("evidence") or {}
            rationale = ev.get("rationale") or ""
            if len(rationale) > 160:
                rationale = rationale[:160] + "..."
            lines.append(
                f"- {e.get('source_id')} -[{e.get('relation_type')}]"
                f"-> {e.get('target_id')}"
                f" (w={e.get('weight')})" + (f" | {rationale}" if rationale else "")
            )
        return "\n".join(lines)
