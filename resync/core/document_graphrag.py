# pylint: disable=all
# mypy: no-rerun
"""Document GraphRAG context builder.

This component transforms:
- user message + extracted entities (job_name, error codes)
into:
- a small, explainable subgraph context string for LLM prompts.

It is intentionally conservative:
- returns empty string when disabled or on errors
- never raises to caller

Enable via env:
- KG_RETRIEVAL_ENABLED=1
"""

from __future__ import annotations

import logging
from typing import Any

from resync.core.postgres_subgraph_retriever import PostgresSubgraphRetriever
from resync.knowledge.kg_store.store import PostgresGraphStore

logger = logging.getLogger(__name__)


class DocumentGraphRAG:
    def __init__(self) -> None:
        from resync.settings import get_settings

        _s = get_settings()

        self.enabled = _s.KG_RETRIEVAL_ENABLED
        self.depth = _s.KG_RETRIEVAL_DEPTH
        self.max_edges = _s.KG_RETRIEVAL_MAX_EDGES
        self._retriever: PostgresSubgraphRetriever | None = None

    async def _get_retriever(self) -> PostgresSubgraphRetriever:
        if self._retriever is not None:
            return self._retriever
        store = PostgresGraphStore()
        await store.ensure_schema()
        self._retriever = PostgresSubgraphRetriever(store)
        return self._retriever

    def _seeds_from_state(self, state: dict[str, Any]) -> list[str]:
        """Extract seed node IDs from agent state (entities + message heuristics)."""
        seeds: list[str] = []
        entities = state.get("entities") or {}

        job_name = entities.get("job_name")
        if job_name:
            seeds.append(f"Job:{job_name.lower()}")

        # Error codes or keywords from raw_data signals
        raw = state.get("raw_data") or {}
        signals = raw.get("signals") or {}
        abend = signals.get("abend_code")
        if abend:
            seeds.append(f"Error:{str(abend).lower()}")

        rc = signals.get("return_code")
        if rc and str(rc) != "0":
            seeds.append(f"Error:rc_{rc}")

        # Heuristic: short concept from message
        msg = (state.get("message") or "").strip()
        if msg:
            head = msg[:80].lower().replace(" ", "_")
            seeds.append(f"Concept:{head}")

        return seeds[:5]

    async def build_context(
        self,
        *,
        tenant: str = "default",
        graph_version: int = 1,
        state: dict[str, Any],
    ) -> str:
        """Build KG context string for LLM prompts.

        Args:
            tenant: Multi-tenant key.
            graph_version: Graph version number.
            state: The AgentState dict (message, entities, raw_data, etc.).

        Returns:
            Formatted subgraph context string, or empty string if disabled/no data.
        """
        if not self.enabled:
            return ""
        try:
            retriever = await self._get_retriever()
            seeds = self._seeds_from_state(state)
            if not seeds:
                return ""
            sg = await retriever.get_subgraph_for_seeds(
                tenant=tenant,
                graph_version=graph_version,
                seeds=seeds,
                depth=self.depth,
                limit_edges=self.max_edges,
            )
            formatted = retriever.format_for_llm(sg)
            if not formatted:
                return ""
            return "\n\n[Document Knowledge Graph Context]\n" + formatted
        except Exception as e:
            logger.warning("document_graphrag_failed", error=str(e))
            return ""
