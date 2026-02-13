"""
Knowledge Ingestion Module.

v6.2.0: Docling-powered document conversion pipeline.

Components:
- document_converter.py: Docling-based file → markdown converter (subprocess-isolated)
- pipeline.py: End-to-end file → RAG pipeline (convert → chunk → embed → pgvector)
- ingest.py: Core ingestion service (chunk → embed → store)
- advanced_chunking.py: Multi-strategy chunking (TWS-optimized, semantic, structure-aware)
- embedding_service.py: Multi-provider embeddings via LiteLLM

Usage:
    from resync.knowledge.ingestion.pipeline import DocumentIngestionPipeline

    pipeline = DocumentIngestionPipeline(embedder, store)
    result = await pipeline.ingest_file("manual.pdf", tenant="default")
"""

__all__ = [
    "DocumentIngestionPipeline",
    "DoclingConverter",
    "IngestService",
]
