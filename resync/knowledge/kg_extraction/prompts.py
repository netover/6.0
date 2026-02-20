"""LLM prompts for Document KG extraction.

These are adapted from the knowledge_graph-main approach (concepts + relations),
but constrained to be production-safe:
- JSON-only output
- bounded list sizes
- optional allowed types/relations
"""

from __future__ import annotations

from typing import Iterable


def build_concepts_prompt(
    text: str,
    *,
    allowed_node_types: Iterable[str] | None = None,
    max_concepts: int = 10,
) -> str:
    allowed = ", ".join(allowed_node_types) if allowed_node_types else "Concept"
    return f"""You are extracting a small set of key concepts from a technical document chunk.

Return ONLY valid JSON with this schema:
{{
  "concepts": [
    {{"name": string, "node_type": string, "aliases": [string], "properties": object}}
  ]
}}

Rules:
- Max {max_concepts} concepts.
- node_type must be one of: {allowed}
- Prefer concise canonical Portuguese names.
- Include error codes, job names, procedures, systems if present.

TEXT:
{text}
"""


def build_edges_prompt(
    text: str,
    concepts: list[str],
    *,
    allowed_relations: Iterable[str] | None = None,
    max_edges: int = 20,
) -> str:
    rels = ", ".join(allowed_relations) if allowed_relations else "RELATED_TO"
    concepts_list = ", ".join(concepts[:50])
    return f"""You are extracting directed relationships between concepts mentioned in a technical text chunk.

Return ONLY valid JSON with this schema:
{{
  "edges": [
    {{"source": string, "target": string, "relation_type": string, "weight": number, "evidence": {{"rationale": string, "confidence": number}} }}
  ]
}}

Rules:
- Only use concepts from this allowed list (exact match preferred): [{concepts_list}]
- relation_type must be one of: {rels}
- Max {max_edges} edges.
- weight 0..1 where 1 is very strong.
- rationale <= 180 chars.

TEXT:
{text}
"""
