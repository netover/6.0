"""Normalization utilities for extracted concepts/edges."""

from __future__ import annotations

import re
from typing import Iterable

from .schemas import Concept, Edge


def canonicalize_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_node_id(node_type: str, name: str) -> str:
    # keep node_id stable and easy to debug
    n = canonicalize_name(name).lower()
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    t = canonicalize_name(node_type) or "Concept"
    return f"{t}:{n}" if n else f"{t}:unknown"


def dedup_concepts(concepts: Iterable[Concept]) -> list[Concept]:
    seen = {}
    for c in concepts:
        key = make_node_id(c.node_type, c.name)
        if key not in seen:
            # canonicalize
            c.name = canonicalize_name(c.name)
            c.aliases = [canonicalize_name(a) for a in c.aliases if canonicalize_name(a)]
            seen[key] = c
        else:
            # merge aliases/properties
            existing = seen[key]
            existing.aliases = sorted(set(existing.aliases + c.aliases))
            existing.properties.update(c.properties or {})
    return list(seen.values())


def dedup_edges(edges: Iterable[Edge]) -> list[Edge]:
    seen = {}
    for e in edges:
        src = canonicalize_name(e.source)
        tgt = canonicalize_name(e.target)
        rel = canonicalize_name(e.relation_type) or "RELATED_TO"
        key = (src.lower(), tgt.lower(), rel)
        if key not in seen:
            e.source = src
            e.target = tgt
            e.relation_type = rel
            seen[key] = e
        else:
            # keep max weight, prefer llm evidence
            cur = seen[key]
            cur.weight = max(cur.weight, e.weight)
            if (cur.evidence.extractor != "llm") and (e.evidence.extractor == "llm"):
                cur.evidence = e.evidence
    return list(seen.values())
