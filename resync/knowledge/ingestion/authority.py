"""Authority and document-type inference helpers.

This module is referenced by
:meth:`resync.knowledge.ingestion.ingest.IngestService.ingest_document_advanced`.
Earlier versions of the project shipped the ingestion pipeline without these helpers,
causing runtime ``ImportError``.

The goal of the inference here is intentionally simple and deterministic:
we infer a coarse ``doc_type`` from the source filename/path.

Doc types are used as a retrieval/reranking signal (authority/freshness), not
as a strict schema.
"""

from __future__ import annotations

from pathlib import Path

# Conservative set of doc types used across the ingestion pipeline.
KNOWN_DOC_TYPES: set[str] = {
    "policy",
    "manual",
    "kb",
    "release_notes",
    "blog",
    "forum",
    "code",
    "unknown",
}

def infer_doc_type(source: str) -> str:
    """Infer a document type from a source name or path.

    Args:
        source: A filename or path (e.g. "docs/policy/security.md").

    Returns:
        A string doc_type compatible with the ingestion payload.
    """
    s = str(source or "").lower()
    name = Path(s).name

    # Extensions / formats
    if name.endswith((".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp")):
        return "code"

    # Common folders / keywords
    if any(k in s for k in ("policy", "policies", "security", "compliance")):
        return "policy"
    if any(k in s for k in ("manual", "handbook", "guide", "documentation", "docs/")):
        return "manual"
    if any(k in s for k in ("kb", "knowledge", "troubleshooting", "faq")):
        return "kb"
    if any(k in s for k in ("release", "changelog", "release-notes", "releasenotes")):
        return "release_notes"
    if any(k in s for k in ("blog", "medium", "dev.to")):
        return "blog"
    if any(k in s for k in ("forum", "discuss", "stack", "community")):
        return "forum"

    return "unknown"

def normalize_doc_type(doc_type: str | None) -> str:
    """Normalize doc_type to a known value."""
    if not doc_type:
        return "unknown"
    dt = str(doc_type).strip().lower()
    return dt if dt in KNOWN_DOC_TYPES else "unknown"
