# pylint: disable=all
# mypy: no-rerun
"""
RAG Integration Service - Connects FastAPI routes to RAG microservice.

Provides:
- Document ingestion with automatic chunking and embedding
- Semantic search/retrieval
- File management
- Background processing support
- Automatic fallback to mock mode when pgvector unavailable
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rag_config import RAGConfig, get_rag_config

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """Represents a document in the RAG system."""

    file_id: str
    filename: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks_count: int = 0
    status: str = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    processed_at: str | None = None


@dataclass
class RAGSearchResult:
    """Represents a search result from RAG."""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGIntegrationService:
    """
    Service that integrates FastAPI routes with RAG microservice.

    Provides high-level API for:
    - Document ingestion
    - Semantic search
    - File management
    """

    def __init__(
        self,
        upload_dir: str | None = None,
        tenant: str = "default",
        use_mock: bool | None = None,
        config: RAGConfig | None = None,
    ):
        # Load configuration
        self.config = config or get_rag_config()

        # Set parameters from config or arguments
        self.upload_dir = Path(upload_dir or self.config.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.tenant = tenant
        self.use_mock = use_mock if use_mock is not None else self.config.use_mock

        # In-memory document store (replace with DB in production)
        self._documents: dict[str, RAGDocument] = {}
        self._chunks: dict[str, list[dict]] = {}

        # Try to import real RAG components
        self._ingest_service = None
        self._retriever = None

        if not self.use_mock:
            try:
                self._initialize_rag_services()
            except ImportError as e:
                logger.warning("RAG services not available, using mock: %s", e)
                self.use_mock = True

    def _initialize_rag_services(self):
        """Initialize real RAG services if available."""
        try:
            from resync.knowledge.ingestion.embedding_service import EmbeddingService
            from resync.knowledge.ingestion.ingest import IngestService
            from resync.knowledge.retrieval.retriever import RagRetriever
            from resync.knowledge.store.pgvector_store import (
                ASYNCPG_AVAILABLE,
                PgVectorStore,
            )

            if not ASYNCPG_AVAILABLE:
                logger.warning("asyncpg not available, using mock mode")
                self.use_mock = True
                return

            # Initialize embedder and vector store
            embedder = EmbeddingService()
            store = PgVectorStore()

            self._ingest_service = IngestService(embedder=embedder, store=store)
            self._retriever = RagRetriever(embedder=embedder, store=store)

            logger.info("RAG services initialized with pgvector backend")
        except ImportError as e:
            logger.warning("RAG microservice not fully available: %s", e)
            self.use_mock = True
        except RuntimeError as e:
            logger.warning("pgvector not configured: %s", e)
            self.use_mock = True
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to initialize RAG services: %s", e)
            self.use_mock = True

    async def ingest_document(
        self,
        file_id: str,
        filename: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RAGDocument:
        """
        Ingest a document into the RAG system.

        Args:
            file_id: Unique identifier for the file
            filename: Original filename
            content: Text content to ingest
            tags: Optional tags for filtering
            metadata: Additional metadata

        Returns:
            RAGDocument with ingestion status
        """
        doc = RAGDocument(
            file_id=file_id,
            filename=filename,
            content=content[:500] + "..." if len(content) > 500 else content,
            metadata=metadata or {},
            status="processing",
        )

        self._documents[file_id] = doc

        if self.use_mock:
            # Mock ingestion - simulate chunking
            chunks = self._mock_chunk_text(content)
            self._chunks[file_id] = chunks
            doc.chunks_count = len(chunks)
            doc.status = "completed"
            doc.processed_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Mock ingested document %s with %s chunks", file_id, len(chunks)
            )
        else:
            # Real ingestion via microservice
            try:
                chunks_count = await self._ingest_service.ingest_document(
                    tenant=self.tenant,
                    doc_id=file_id,
                    source=filename,
                    text=content,
                    ts_iso=datetime.now(timezone.utc).isoformat(),
                    tags=tags,
                )
                doc.chunks_count = chunks_count
                doc.status = "completed"
                doc.processed_at = datetime.now(timezone.utc).isoformat()

                logger.info(
                    "Ingested document %s with %s chunks", file_id, chunks_count
                )
            except Exception as e:
                doc.status = "failed"
                doc.metadata["error"] = str(e)
                logger.error("Failed to ingest document %s: %s", file_id, e)

        return doc

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RAGSearchResult]:
        """
        Search for relevant documents using semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (e.g., by tags, date)

        Returns:
            List of search results with scores
        """
        if self.use_mock:
            return self._mock_search(query, top_k, filters)

        try:
            hits = await self._retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            results = []
            for hit in hits:
                results.append(
                    RAGSearchResult(
                        chunk_id=hit.get("chunk_id", ""),
                        doc_id=hit.get("doc_id", ""),
                        content=hit.get("payload", {}).get("text", ""),
                        score=hit.get("score", 0.0),
                        metadata=hit.get("payload", {}),
                    )
                )

            return results
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def _mock_chunk_text(self, text: str, chunk_size: int = 500) -> list[dict]:
        """Mock text chunking."""
        chunks = []
        words = text.split()

        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                        "text": chunk_text,
                        "size": len(chunk_text),
                    }
                )
                current_chunk = []
                current_size = 0

        # Add remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "id": hashlib.sha256(chunk_text.encode()).hexdigest()[:16],
                    "text": chunk_text,
                    "size": len(chunk_text),
                }
            )

        return chunks

    def _mock_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RAGSearchResult]:
        """Mock semantic search using simple keyword matching.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters applied in mock mode via _matches_filters
        """
        results = []
        query_words = set(query.lower().split())

        # Apply filters if provided (mock implementation)
        filtered_docs = self._documents
        if filters:
            # Basic filtering by metadata in mock mode
            filtered_docs = {
                k: v
                for k, v in self._documents.items()
                if self._matches_filters(v, filters)
            }

        for file_id, chunks in self._chunks.items():
            if file_id not in filtered_docs:
                continue
            doc = filtered_docs[file_id]

            for chunk in chunks:
                chunk_words = set(chunk["text"].lower().split())
                overlap = len(query_words & chunk_words)

                if overlap > 0:
                    score = overlap / max(len(query_words), 1)
                    results.append(
                        RAGSearchResult(
                            chunk_id=chunk["id"],
                            doc_id=file_id,
                            content=chunk["text"][:200] + "...",
                            score=score,
                            metadata={
                                "filename": doc.filename,
                                "source": doc.filename,
                            },
                        )
                    )

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _matches_filters(self, doc: RAGDocument, filters: dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        if not filters:
            return True
        doc_metadata = doc.metadata or {}
        for key, value in filters.items():
            if doc_metadata.get(key) != value:
                return False
        return True

    async def get_document(self, file_id: str) -> RAGDocument | None:
        """Get document by ID."""
        return self._documents.get(file_id)

    async def list_documents(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RAGDocument]:
        """List all documents, optionally filtered by status."""
        docs = list(self._documents.values())

        if status:
            docs = [d for d in docs if d.status == status]

        return docs[:limit]

    async def delete_document(self, file_id: str) -> bool:
        """Delete a document and its chunks."""
        if file_id in self._documents:
            del self._documents[file_id]
            if file_id in self._chunks:
                del self._chunks[file_id]

            # Delete file from disk (run blocking I/O in thread pool)
            def _delete_files():
                for path in self.upload_dir.glob(f"{file_id}_*"):
                    try:
                        path.unlink()
                    except Exception as e:
                        logger.warning("Failed to delete file %s: %s", path, e)

            await asyncio.to_thread(_delete_files)

            return True
        return False

    async def get_stats(self) -> dict[str, Any]:
        """Get RAG system statistics."""
        total_chunks = sum(len(c) for c in self._chunks.values())

        return {
            "total_documents": len(self._documents),
            "total_chunks": total_chunks,
            "documents_by_status": {
                "pending": len(
                    [d for d in self._documents.values() if d.status == "pending"]
                ),
                "processing": len(
                    [d for d in self._documents.values() if d.status == "processing"]
                ),
                "completed": len(
                    [d for d in self._documents.values() if d.status == "completed"]
                ),
                "failed": len(
                    [d for d in self._documents.values() if d.status == "failed"]
                ),
            },
            "use_mock": self.use_mock,
        }


# Global service instance (singleton pattern)
_rag_service: RAGIntegrationService | None = None


def get_rag_service() -> RAGIntegrationService:
    """Get or create RAG service instance."""
    global _rag_service

    if _rag_service is None:
        use_mock = os.getenv("RAG_USE_MOCK", "true").lower() == "true"
        _rag_service = RAGIntegrationService(use_mock=use_mock)

    return _rag_service


def reset_rag_service():
    """Reset RAG service (for testing)."""
    global _rag_service
    _rag_service = None
