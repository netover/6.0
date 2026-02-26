"""
Document Ingestion Pipeline.

End-to-end pipeline: file upload → Docling conversion → chunking → embedding → pgvector.

Orchestrates DoclingConverter and IngestService to provide a single entry point
for ingesting documents of any supported format into the RAG knowledge base.

Usage:
    pipeline = DocumentIngestionPipeline(embedder, store)
    result = await pipeline.ingest_file(
        file_path="manuals/tws_admin.pdf",
        tenant="default",
    )
    print(f"Ingested {result.chunks_stored} chunks from {result.pages} pages")
"""

from __future__ import annotations

import asyncio
import hashlib
import structlog
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from resync.knowledge.interfaces import Embedder, VectorStore

from .document_converter import DoclingConverter
from .ingest import IngestService

logger = structlog.get_logger(__name__)

@dataclass
class IngestionResult:
    """Result of the full ingestion pipeline."""

    doc_id: str
    source: str
    format: str
    pages: int
    tables_extracted: int
    chunks_stored: int
    conversion_time_s: float
    ingestion_time_s: float
    total_time_s: float
    status: str = "success"
    error: str | None = None

class DocumentIngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Steps:
        1. DoclingConverter: file → structured markdown + tables
        2. IngestService: markdown → chunks → embeddings → pgvector

    Config via environment or constructor:
        DOCLING_TABLE_STRUCTURE=true   (ML table extraction)
        DOCLING_OCR=false              (OCR for scanned docs)
        DOCLING_PROCESS_TIMEOUT=300    (subprocess hard timeout)
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        converter: DoclingConverter | None = None,
        batch_size: int = 128,
    ):
        self.ingest_service = IngestService(embedder, store, batch_size=batch_size)
        self.converter = converter or DoclingConverter(
            do_table_structure=True, do_ocr=False, process_timeout=300
        )

    async def ingest_file(
        self,
        file_path: str | Path,
        *,
        tenant: str = "default",
        doc_id: str | None = None,
        tags: list[str] | None = None,
        chunking_strategy: str = "tws_optimized",
        reindex: bool = False,
    ) -> IngestionResult:
        """
        Convert and ingest a document file into the RAG knowledge base.

        Args:
            file_path: Path to the document file.
            tenant: Tenant identifier.
            doc_id: Document ID (auto-generated from filename if not provided).
            tags: Optional tags for metadata.
            chunking_strategy: Chunking strategy for IngestService.
            reindex: If True, delete existing chunks before re-ingesting.

        Returns:
            IngestionResult with stats.
        """
        file_path = Path(file_path)
        source = file_path.name
        if doc_id is None:
            doc_id = await self._generate_doc_id(file_path)
        t_total = time.perf_counter()
        converted = await self.converter.convert(file_path)
        if converted.status != "success":
            return IngestionResult(
                doc_id=doc_id,
                source=source,
                format=converted.format,
                pages=0,
                tables_extracted=0,
                chunks_stored=0,
                conversion_time_s=converted.conversion_time_s,
                ingestion_time_s=0,
                total_time_s=time.perf_counter() - t_total,
                status="error",
                error=converted.error or "Conversion failed",
            )
        if not converted.markdown or not converted.markdown.strip():
            return IngestionResult(
                doc_id=doc_id,
                source=source,
                format=converted.format,
                pages=converted.pages,
                tables_extracted=len(converted.tables) if converted.tables else 0,
                chunks_stored=0,
                conversion_time_s=converted.conversion_time_s,
                ingestion_time_s=0,
                total_time_s=time.perf_counter() - t_total,
                status="warning",
                error="Document converted but no text extracted",
            )
        t_ingest = time.perf_counter()
        ts_iso = datetime.now(timezone.utc).isoformat()
        document_title = converted.metadata.get("title", file_path.stem)
        auto_tags = list(tags or [])
        auto_tags.append(f"format:{converted.format}")
        if converted.tables:
            auto_tags.append(f"tables:{len(converted.tables)}")
        try:
            if reindex:
                chunks_stored = await self.ingest_service.reindex_document(
                    tenant=tenant,
                    doc_id=doc_id,
                    source=source,
                    text=converted.markdown,
                    ts_iso=ts_iso,
                    document_title=document_title,
                    tags=auto_tags,
                    use_advanced=True,
                )
            else:
                chunks_stored = await self.ingest_service.ingest_document_advanced(
                    tenant=tenant,
                    doc_id=doc_id,
                    source=source,
                    text=converted.markdown,
                    ts_iso=ts_iso,
                    document_title=document_title,
                    tags=auto_tags,
                    chunking_strategy=chunking_strategy,
                )
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("ingestion_failed", doc_id=doc_id, error=str(e))
            return IngestionResult(
                doc_id=doc_id,
                source=source,
                format=converted.format,
                pages=converted.pages,
                tables_extracted=len(converted.tables),
                chunks_stored=0,
                conversion_time_s=converted.conversion_time_s,
                ingestion_time_s=time.perf_counter() - t_ingest,
                total_time_s=time.perf_counter() - t_total,
                status="error",
                error=f"Ingestion failed: {e}",
            )
        ingestion_time = time.perf_counter() - t_ingest
        total_time = time.perf_counter() - t_total
        logger.info(
            "document_ingested",
            doc_id=doc_id,
            source=source,
            format=converted.format,
            pages=converted.pages,
            tables=len(converted.tables),
            chunks=chunks_stored,
            convert_s=f"{converted.conversion_time_s:.1f}",
            ingest_s=f"{ingestion_time:.1f}",
            total_s=f"{total_time:.1f}",
        )
        return IngestionResult(
            doc_id=doc_id,
            source=source,
            format=converted.format,
            pages=converted.pages,
            tables_extracted=len(converted.tables),
            chunks_stored=chunks_stored,
            conversion_time_s=converted.conversion_time_s,
            ingestion_time_s=ingestion_time,
            total_time_s=total_time,
        )

    async def ingest_batch(
        self,
        file_paths: list[str | Path],
        *,
        tenant: str = "default",
        tags: list[str] | None = None,
        max_concurrent: int = 2,
    ) -> list[IngestionResult]:
        """
        Ingest multiple documents with controlled concurrency.

        Args:
            file_paths: List of file paths to ingest.
            tenant: Tenant identifier.
            tags: Tags applied to all documents.
            max_concurrent: Max concurrent conversions (each ~4 GB RAM).

        Returns:
            List of IngestionResult (same order as input).
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited_ingest(fp: str | Path) -> IngestionResult:
            async with semaphore:
                return await self.ingest_file(fp, tenant=tenant, tags=tags)

        return list(await asyncio.gather(*(_limited_ingest(fp) for fp in file_paths)))

    @staticmethod
    async def _generate_doc_id(file_path: Path) -> str:
        """Generate a deterministic doc_id from file path and content hash."""
        stat = await asyncio.to_thread(file_path.stat)
        key = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}"
        short_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
        stem = file_path.stem[:40]
        return f"{stem}_{short_hash}"
