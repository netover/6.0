"""LLM + heuristic extraction of a Document Knowledge Graph.

Usage (from ingest):
    extractor = KGExtractor()
    result = await extractor.extract(doc_id=..., chunks=[...])

By default, extraction is gated by env var KG_EXTRACTION_ENABLED.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from pydantic import ValidationError

from resync.core.utils.llm import call_llm

from .prompts import build_concepts_prompt, build_edges_prompt
from .schemas import Concept, Edge, Evidence, ExtractionResult
from .normalizer import dedup_concepts, dedup_edges

logger = logging.getLogger(__name__)


class KGExtractor:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model: str | None = None,
        max_concepts_per_chunk: int | None = None,
        max_edges_per_chunk: int | None = None,
        allowed_node_types: Iterable[str] | None = None,
        allowed_relations: Iterable[str] | None = None,
    ):
        from resync.settings import get_settings
        _s = get_settings()

        self.enabled = enabled if enabled is not None else _s.KG_EXTRACTION_ENABLED
        self.model = model or _s.KG_EXTRACTION_MODEL or (_s.llm_model or "gpt-4o")
        self.max_concepts_per_chunk = max_concepts_per_chunk or _s.KG_EXTRACTION_MAX_CONCEPTS
        self.max_edges_per_chunk = max_edges_per_chunk or _s.KG_EXTRACTION_MAX_EDGES
        self.allowed_node_types = list(allowed_node_types) if allowed_node_types else [
            "Concept",
            "Error",
            "Solution",
            "Procedure",
            "Job",
            "System",
        ]
        self.allowed_relations = list(allowed_relations) if allowed_relations else [
            "RELATED_TO",
            "CAUSES",
            "SOLVED_BY",
            "DEPENDS_ON",
            "MENTIONED_IN",
            "APPLIES_TO",
        ]

    async def extract(
        self,
        *,
        tenant: str,
        graph_version: int,
        doc_id: str,
        chunks: list[dict[str, Any]],
    ) -> ExtractionResult:
        """Extract concepts and edges from chunks.

        chunks: list of dicts with at least keys: chunk_id, content
        """
        if not self.enabled:
            return ExtractionResult()

        all_concepts: list[Concept] = []
        all_edges: list[Edge] = []

        for ch in chunks:
            text = (ch.get("content") or "").strip()
            if not text:
                continue

            chunk_id = ch.get("chunk_id")

            # 1) concepts
            try:
                c_prompt = build_concepts_prompt(
                    text,
                    allowed_node_types=self.allowed_node_types,
                    max_concepts=self.max_concepts_per_chunk,
                )
                raw = await call_llm(c_prompt, temperature=0.0, model=self.model)
                concepts = self._parse_concepts(raw)
            except Exception as e:
                logger.warning("kg_extract_concepts_failed", extra={"error": str(e), "doc_id": doc_id, "chunk_id": chunk_id})
                concepts = []

            # attach provenance
            for c in concepts:
                c.properties = {**c.properties, "doc_id": doc_id, "chunk_id": chunk_id}
            all_concepts.extend(concepts)

            # 2) edges
            try:
                names = [c.name for c in concepts][: self.max_concepts_per_chunk]
                e_prompt = build_edges_prompt(
                    text,
                    concepts=names,
                    allowed_relations=self.allowed_relations,
                    max_edges=self.max_edges_per_chunk,
                )
                raw = await call_llm(e_prompt, temperature=0.0, model=self.model)
                edges = self._parse_edges(raw, doc_id=doc_id, chunk_id=chunk_id)
            except Exception as e:
                logger.warning("kg_extract_edges_failed", extra={"error": str(e), "doc_id": doc_id, "chunk_id": chunk_id})
                edges = []

            # 3) fallback edges by co-occurrence (weak)
            if not edges and len(concepts) >= 2:
                edges = self._cooc_edges(concepts, doc_id=doc_id, chunk_id=chunk_id)

            all_edges.extend(edges)

        # Normalize/dedup
        all_concepts = dedup_concepts(all_concepts)
        all_edges = dedup_edges(all_edges)

        return ExtractionResult(concepts=all_concepts, edges=all_edges)

    def _strip_json_fences(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^```\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
        return s.strip()

    def _parse_concepts(self, raw: str) -> list[Concept]:
        data = json.loads(self._strip_json_fences(raw))
        items = data.get("concepts", []) if isinstance(data, dict) else []
        out: list[Concept] = []
        for it in items:
            try:
                out.append(Concept.model_validate(it))
            except ValidationError:
                continue
        return out

    def _parse_edges(self, raw: str, *, doc_id: str, chunk_id: str | None) -> list[Edge]:
        data = json.loads(self._strip_json_fences(raw))
        items = data.get("edges", []) if isinstance(data, dict) else []
        out: list[Edge] = []
        for it in items:
            try:
                e = Edge.model_validate(it)
                # ensure evidence
                if not e.evidence:
                    e.evidence = Evidence(doc_id=doc_id, chunk_id=chunk_id)
                else:
                    e.evidence.doc_id = e.evidence.doc_id or doc_id
                    e.evidence.chunk_id = e.evidence.chunk_id or chunk_id
                out.append(e)
            except ValidationError:
                continue
        return out

    def _cooc_edges(self, concepts: list[Concept], *, doc_id: str, chunk_id: str | None) -> list[Edge]:
        names = [c.name for c in concepts][:8]
        edges: list[Edge] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                edges.append(
                    Edge(
                        source=names[i],
                        target=names[j],
                        relation_type="RELATED_TO",
                        weight=0.3,
                        evidence=Evidence(doc_id=doc_id, chunk_id=chunk_id, extractor="cooc", confidence=0.3),
                    )
                )
        return edges
