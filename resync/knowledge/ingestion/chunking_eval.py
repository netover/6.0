"""
Chunking Evaluation Pipeline for RAG Systems.

v6.0: Implements Decision #9 from "9 RAG Chunking Decisions That Beat Models"
"Tuning by failure slices, not global scores"

The biggest trap: tuning chunk size using a single average metric.
RAG failures are lumpy. Chunking that helps "general" queries might hurt edge cases.

This module provides:
- Failure slice taxonomy for categorizing retrieval failures
- Eval workflow for systematic chunking improvement
- Rule adjustment suggestions based on failure patterns

Usage:
    from resync.knowledge.ingestion.chunking_eval import (
        ChunkingEvalPipeline,
        FailureSlice,
        EvalResult,
    )

    pipeline = ChunkingEvalPipeline()
    results = await pipeline.evaluate_queries(queries, ground_truth)
    suggestions = pipeline.suggest_rule_changes(results)
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FailureSlice(str, Enum):
    """
    Taxonomy of RAG retrieval failure types.

    Based on article Decision #9: "Chunking should be tuned by failure slices"
    Each slice represents a specific pattern of retrieval failure.
    """

    MISSING_EXCEPTION = "missing_exception"
    """Retrieved the rule but missed the exception/condition."""
    WRONG_SCOPE_VERSION = "wrong_scope_version"
    """Retrieved content from wrong scope (beta vs prod) or outdated version."""
    LOST_TABLE_HEADER = "lost_table_header"
    """Table row retrieved without its header, making it uninterpretable."""
    REDUNDANT_OVERLAPS = "redundant_overlaps"
    """Multiple near-identical chunks from overlap, wasting context."""
    NEEDS_CROSS_SECTION = "needs_cross_section_context"
    """Answer requires combining information from multiple sections."""
    MISSING_PROCEDURE_STEP = "missing_procedure_step"
    """Procedure chunk missing critical step or prerequisite."""
    CODE_WITHOUT_CONTEXT = "code_without_context"
    """Code block retrieved without explanation or usage context."""
    ERROR_CODE_INCOMPLETE = "error_code_incomplete"
    """Error documentation missing cause, solution, or explanation."""
    DEFINITION_TRUNCATED = "definition_truncated"
    """Definition split mid-way, missing constraints or examples."""
    LIST_ITEM_ORPHANED = "list_item_orphaned"
    """List item retrieved without its intro sentence or context."""
    UNKNOWN = "unknown"
    """Failure doesn't fit other categories."""


class FailureSeverity(str, Enum):
    """Severity level of a retrieval failure."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RetrievedChunk:
    """A chunk that was retrieved for a query."""

    chunk_id: str
    content: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query_id: str
    query_text: str
    expected_answer: str
    actual_answer: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    top_k_used: int = 5
    failure_slice: FailureSlice = FailureSlice.UNKNOWN
    failure_severity: FailureSeverity = FailureSeverity.MEDIUM
    failure_description: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)
    retrieved_relevant: bool = False
    recall_at_k: float = 0.0
    mrr: float = 0.0
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content[:200] + "..."
                    if len(c.content) > 200
                    else c.content,
                    "score": c.score,
                    "rank": c.rank,
                }
                for c in self.retrieved_chunks
            ],
            "failure_slice": self.failure_slice.value,
            "failure_severity": self.failure_severity.value,
            "failure_description": self.failure_description,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "retrieved_relevant": self.retrieved_relevant,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class RuleSuggestion:
    """Suggested chunking rule change based on failure analysis."""

    failure_slice: FailureSlice
    current_rule: str
    suggested_rule: str
    rationale: str
    affected_queries: int
    estimated_impact: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_slice": self.failure_slice.value,
            "current_rule": self.current_rule,
            "suggested_rule": self.suggested_rule,
            "rationale": self.rationale,
            "affected_queries": self.affected_queries,
            "estimated_impact": self.estimated_impact,
        }


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    failure_slice_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    avg_recall: float = 0.0
    avg_mrr: float = 0.0
    rule_suggestions: list[RuleSuggestion] = field(default_factory=list)
    results: list[EvalResult] = field(default_factory=list)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "failure_slice_counts": self.failure_slice_counts,
            "severity_counts": self.severity_counts,
            "avg_recall": self.avg_recall,
            "avg_mrr": self.avg_mrr,
            "rule_suggestions": [s.to_dict() for s in self.rule_suggestions],
            "evaluated_at": self.evaluated_at,
        }


def detect_failure_slice(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    expected_answer: str,
    relevant_chunk_ids: list[str],
) -> tuple[FailureSlice, str]:
    """
    Detect the type of failure slice for a query.

    Uses heuristics to categorize why retrieval failed.
    """
    if not retrieved_chunks:
        return (FailureSlice.UNKNOWN, "No chunks retrieved")
    retrieved_ids = {c.chunk_id for c in retrieved_chunks}
    relevant_retrieved = retrieved_ids.intersection(set(relevant_chunk_ids))
    if relevant_retrieved:
        return (FailureSlice.UNKNOWN, "Relevant chunks were retrieved")
    for chunk in retrieved_chunks:
        content = chunk.content.lower()
        if "|" in content and "header" not in content:
            if not any((h in content for h in ["column", "field", "name"])):
                return (
                    FailureSlice.LOST_TABLE_HEADER,
                    "Table row retrieved without header context",
                )
    for chunk in retrieved_chunks:
        content = chunk.content.lower()
        if any((code in content for code in ["aws", "error", "code"])):
            if not any(
                (
                    phrase in content
                    for phrase in ["cause:", "solution:", "explanation:"]
                )
            ):
                return (
                    FailureSlice.ERROR_CODE_INCOMPLETE,
                    "Error code without complete documentation",
                )
    for chunk in retrieved_chunks:
        content = chunk.content
        step_count = content.count("step") + content.count("1.") + content.count("2.")
        if step_count > 0 and step_count < 3:
            return (FailureSlice.MISSING_PROCEDURE_STEP, "Procedure appears incomplete")
    for chunk in retrieved_chunks:
        content = chunk.content
        if "```" in content or "def " in content or "function " in content:
            if len(content.split("\n")) > 5 and "explanation" not in content.lower():
                return (
                    FailureSlice.CODE_WITHOUT_CONTEXT,
                    "Code block without explanation",
                )
    if len(retrieved_chunks) >= 3:
        contents = [c.content for c in retrieved_chunks[:3]]
        for i in range(len(contents) - 1):
            overlap = _calculate_text_overlap(contents[i], contents[i + 1])
            if overlap > 0.7:
                return (
                    FailureSlice.REDUNDANT_OVERLAPS,
                    f"High overlap ({overlap:.0%}) between chunks {i} and {i + 1}",
                )
    if "exception" in expected_answer.lower() or "unless" in expected_answer.lower():
        for chunk in retrieved_chunks:
            if (
                "exception" not in chunk.content.lower()
                and "unless" not in chunk.content.lower()
            ):
                return (
                    FailureSlice.MISSING_EXCEPTION,
                    "Expected exception/condition not in retrieved chunks",
                )
    for chunk in retrieved_chunks:
        metadata = chunk.metadata
        if metadata.get("is_deprecated"):
            return (FailureSlice.WRONG_SCOPE_VERSION, "Retrieved deprecated content")
        if metadata.get("environment") not in ["all", "prod"]:
            return (
                FailureSlice.WRONG_SCOPE_VERSION,
                f"Retrieved content for {metadata.get('environment')} environment",
            )
    for chunk in retrieved_chunks:
        content = chunk.content
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith(("-", "•", "*")) or line.strip()[0].isdigit():
                if not any(
                    (
                        intro in content.lower()
                        for intro in ["following", "these", "below", "steps"]
                    )
                ):
                    return (
                        FailureSlice.LIST_ITEM_ORPHANED,
                        "List item without intro context",
                    )
    return (FailureSlice.UNKNOWN, "Could not determine specific failure type")


def _calculate_text_overlap(text1: str, text2: str) -> float:
    """Calculate overlap ratio between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    smaller = min(len(words1), len(words2))
    return len(intersection) / smaller if smaller > 0 else 0.0


FAILURE_SLICE_RULES: dict[FailureSlice, dict[str, Any]] = {
    FailureSlice.MISSING_EXCEPTION: {
        "current_rule": "Split at paragraph boundaries",
        "suggested_rule": "Keep 'rule + exception' pairs together; increase overlap at 'unless', 'except', 'however'",
        "impact": "high",
    },
    FailureSlice.WRONG_SCOPE_VERSION: {
        "current_rule": "Index all content equally",
        "suggested_rule": "Add metadata filtering for environment/version; boost non-deprecated content",
        "impact": "high",
    },
    FailureSlice.LOST_TABLE_HEADER: {
        "current_rule": "Split tables by rows",
        "suggested_rule": "Keep header + row groups together; add header context to each row chunk",
        "impact": "high",
    },
    FailureSlice.REDUNDANT_OVERLAPS: {
        "current_rule": "Constant token overlap",
        "suggested_rule": "Use structure-aware overlap at paragraph/heading boundaries only",
        "impact": "medium",
    },
    FailureSlice.NEEDS_CROSS_SECTION: {
        "current_rule": "Single chunk retrieval",
        "suggested_rule": "Enable hierarchical chunking; retrieve parent context for cross-section queries",
        "impact": "high",
    },
    FailureSlice.MISSING_PROCEDURE_STEP: {
        "current_rule": "Split procedures by token count",
        "suggested_rule": "Keep complete procedures together; split only at step boundaries",
        "impact": "high",
    },
    FailureSlice.CODE_WITHOUT_CONTEXT: {
        "current_rule": "Split code blocks independently",
        "suggested_rule": "Keep code + preceding explanation together; add code summary to metadata",
        "impact": "medium",
    },
    FailureSlice.ERROR_CODE_INCOMPLETE: {
        "current_rule": "Split error docs by token count",
        "suggested_rule": "Keep error code + cause + solution as atomic unit",
        "impact": "high",
    },
    FailureSlice.DEFINITION_TRUNCATED: {
        "current_rule": "Split at sentence boundaries",
        "suggested_rule": "Detect definition patterns and keep complete; include examples",
        "impact": "medium",
    },
    FailureSlice.LIST_ITEM_ORPHANED: {
        "current_rule": "Split lists by items",
        "suggested_rule": "Keep list intro + items together; add intro context to each item",
        "impact": "medium",
    },
}


def generate_rule_suggestions(results: list[EvalResult]) -> list[RuleSuggestion]:
    """Generate rule change suggestions based on failure analysis."""
    suggestions: dict[FailureSlice, RuleSuggestion] = {}
    for result in results:
        if result.failure_slice == FailureSlice.UNKNOWN:
            continue
        slice_type = result.failure_slice
        if slice_type not in suggestions:
            rule_info = FAILURE_SLICE_RULES.get(slice_type, {})
            suggestions[slice_type] = RuleSuggestion(
                failure_slice=slice_type,
                current_rule=rule_info.get("current_rule", "Unknown"),
                suggested_rule=rule_info.get("suggested_rule", "Review and adjust"),
                rationale=f"Detected in {result.query_id}: {result.failure_description}",
                affected_queries=1,
                estimated_impact=rule_info.get("impact", "medium"),
            )
        else:
            suggestions[slice_type].affected_queries += 1
    return sorted(suggestions.values(), key=lambda s: s.affected_queries, reverse=True)


class ChunkingEvalPipeline:
    """
    Pipeline for evaluating chunking quality and suggesting improvements.

    Usage:
        pipeline = ChunkingEvalPipeline()

        # Evaluate queries
        results = await pipeline.evaluate_queries(
            queries=[
                {"id": "q1", "text": "What is AWS001E?", "expected": "...", "relevant_ids": ["c1", "c2"]},
            ],
            retriever=my_retriever,
        )

        # Get suggestions
        suggestions = pipeline.suggest_rule_changes(results)

        # Save report
        pipeline.save_report(results, "eval_report.json")
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    async def evaluate_query(
        self,
        query_id: str,
        query_text: str,
        expected_answer: str,
        relevant_chunk_ids: list[str],
        retriever: Any | None = None,
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> EvalResult:
        """
        Evaluate a single query.

        Args:
            query_id: Unique query identifier
            query_text: The query text
            expected_answer: Expected answer for comparison
            relevant_chunk_ids: IDs of chunks that should be retrieved
            retriever: Retriever to use (if retrieved_chunks not provided)
            retrieved_chunks: Pre-retrieved chunks (if available)

        Returns:
            EvalResult with failure analysis
        """
        if retrieved_chunks is None and retriever is not None:
            try:
                raw_results = await retriever.retrieve(query_text, top_k=self.top_k)
                retrieved_chunks = [
                    RetrievedChunk(
                        chunk_id=r.get("id", r.get("chunk_id", "")),
                        content=r.get("content", r.get("text", "")),
                        score=r.get("score", 0.0),
                        rank=i + 1,
                        metadata=r.get("metadata", {}),
                    )
                    for i, r in enumerate(raw_results)
                ]
            except Exception as e:
                logger.error(
                    "eval_retrieve_failed",
                    extra={"query_id": query_id, "error": str(e)},
                )
                retrieved_chunks = []
        retrieved_chunks = retrieved_chunks or []
        retrieved_ids = {c.chunk_id for c in retrieved_chunks}
        relevant_set = set(relevant_chunk_ids)
        relevant_retrieved = retrieved_ids.intersection(relevant_set)
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        mrr = 0.0
        for chunk in retrieved_chunks:
            if chunk.chunk_id in relevant_set:
                mrr = 1.0 / chunk.rank
                break
        failure_slice, failure_desc = detect_failure_slice(
            query_text, retrieved_chunks, expected_answer, relevant_chunk_ids
        )
        severity = FailureSeverity.MEDIUM
        if recall == 0:
            severity = FailureSeverity.CRITICAL
        elif recall < 0.5:
            severity = FailureSeverity.HIGH
        elif recall < 1.0:
            severity = FailureSeverity.MEDIUM
        return EvalResult(
            query_id=query_id,
            query_text=query_text,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks,
            top_k_used=self.top_k,
            failure_slice=failure_slice,
            failure_severity=severity,
            failure_description=failure_desc,
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_relevant=len(relevant_retrieved) > 0,
            recall_at_k=recall,
            mrr=mrr,
        )

    async def evaluate_queries(
        self, queries: list[dict[str, Any]], retriever: Any | None = None
    ) -> EvalReport:
        """
        Evaluate multiple queries and generate a report.

        Args:
            queries: List of query dicts with keys:
                - id: Query ID
                - text: Query text
                - expected: Expected answer
                - relevant_ids: List of relevant chunk IDs
            retriever: Retriever to use

        Returns:
            EvalReport with aggregated results
        """
        results: list[EvalResult] = []
        for query in queries:
            result = await self.evaluate_query(
                query_id=query.get("id", ""),
                query_text=query.get("text", ""),
                expected_answer=query.get("expected", ""),
                relevant_chunk_ids=query.get("relevant_ids", []),
                retriever=retriever,
            )
            results.append(result)
        return self._generate_report(results)

    def _generate_report(self, results: list[EvalResult]) -> EvalReport:
        """Generate aggregated report from results."""
        total = len(results)
        successful = sum((1 for r in results if r.retrieved_relevant))
        failed = total - successful
        slice_counts: dict[str, int] = {}
        for r in results:
            slice_key = r.failure_slice.value
            slice_counts[slice_key] = slice_counts.get(slice_key, 0) + 1
        severity_counts: dict[str, int] = {}
        for r in results:
            sev_key = r.failure_severity.value
            severity_counts[sev_key] = severity_counts.get(sev_key, 0) + 1
        avg_recall = sum((r.recall_at_k for r in results)) / total if total > 0 else 0.0
        avg_mrr = sum((r.mrr for r in results)) / total if total > 0 else 0.0
        suggestions = generate_rule_suggestions(results)
        return EvalReport(
            total_queries=total,
            successful_queries=successful,
            failed_queries=failed,
            failure_slice_counts=slice_counts,
            severity_counts=severity_counts,
            avg_recall=avg_recall,
            avg_mrr=avg_mrr,
            rule_suggestions=suggestions,
            results=results,
        )

    def suggest_rule_changes(self, results: list[EvalResult]) -> list[RuleSuggestion]:
        """Generate rule change suggestions from results."""
        return generate_rule_suggestions(results)

    def save_report(self, report: EvalReport, path: str | Path) -> None:
        """Save evaluation report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("eval_report_saved", extra={"path": str(path)})

    def load_results(self, path: str | Path) -> list[EvalResult]:
        """Load evaluation results from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for item in data.get("results", []):
            result = EvalResult(
                query_id=item["query_id"],
                query_text=item["query_text"],
                expected_answer=item["expected_answer"],
                actual_answer=item.get("actual_answer"),
                failure_slice=FailureSlice(item.get("failure_slice", "unknown")),
                failure_severity=FailureSeverity(
                    item.get("failure_severity", "medium")
                ),
                failure_description=item.get("failure_description", ""),
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                retrieved_relevant=item.get("retrieved_relevant", False),
                recall_at_k=item.get("recall_at_k", 0.0),
                mrr=item.get("mrr", 0.0),
                evaluated_at=item.get("evaluated_at", ""),
            )
            results.append(result)
        return results


def create_eval_query(
    query_id: str, query_text: str, expected_answer: str, relevant_chunk_ids: list[str]
) -> dict[str, Any]:
    """Create a query dict for evaluation."""
    return {
        "id": query_id,
        "text": query_text,
        "expected": expected_answer,
        "relevant_ids": relevant_chunk_ids,
    }


__all__ = [
    "FailureSlice",
    "FailureSeverity",
    "RetrievedChunk",
    "EvalResult",
    "RuleSuggestion",
    "EvalReport",
    "ChunkingEvalPipeline",
    "detect_failure_slice",
    "generate_rule_suggestions",
    "create_eval_query",
    "FAILURE_SLICE_RULES",
]
