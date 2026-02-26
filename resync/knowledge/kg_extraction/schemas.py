"""Pydantic schemas for Document KG extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ExtractorType = Literal["llm", "cooc"]

class Evidence(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    rationale: str | None = None
    extractor: ExtractorType = "llm"
    confidence: float | None = None

class Concept(BaseModel):
    name: str = Field(..., description="Canonical name")
    node_type: str = Field(default="Concept", description="Ontology type")
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)

class Edge(BaseModel):
    source: str = Field(..., description="Source concept name")
    target: str = Field(..., description="Target concept name")
    relation_type: str = Field(default="RELATED_TO")
    weight: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: Evidence = Field(default_factory=Evidence)

class ExtractionResult(BaseModel):
    concepts: list[Concept] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
