"""
Document Ingestion API.

Endpoints for uploading documents and ingesting them into the RAG knowledge base
using Docling for intelligent document conversion.

Routes:
    POST /api/v1/knowledge/ingest       — Upload and ingest a single document
    POST /api/v1/knowledge/ingest/batch  — Ingest multiple documents by path
    GET  /api/v1/knowledge/ingest/status — Check supported formats and converter health
"""

from __future__ import annotations
# pylint: disable=no-name-in-module
# mypy: ignore-errors

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.knowledge.ingestion.document_converter import (
    DoclingConverter,
    FORMAT_EXTENSIONS,
    SupportedFormat,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge/ingest",
    tags=["knowledge-ingestion"],
)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ── Response Models ─────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response from document ingestion."""

    status: str = Field(description="success, warning, or error")
    doc_id: str = Field(description="Document identifier")
    source: str = Field(description="Original filename")
    format: str = Field(description="Detected format")
    pages: int = Field(default=0, description="Number of pages")
    tables_extracted: int = Field(default=0, description="Tables found")
    chunks_stored: int = Field(default=0, description="Chunks ingested into RAG")
    conversion_time_s: float = Field(default=0, description="Docling conversion time")
    total_time_s: float = Field(default=0, description="Total pipeline time")
    error: str | None = Field(default=None, description="Error details if failed")


class BatchIngestRequest(BaseModel):
    """Request to ingest multiple files by path."""

    file_paths: list[str] = Field(description="List of file paths on the server")
    tenant: str = Field(default="default")
    tags: list[str] = Field(default_factory=list)
    chunking_strategy: str = Field(default="tws_optimized")
    reindex: bool = Field(default=False, description="Delete existing chunks before re-ingesting")


class BatchIngestResponse(BaseModel):
    """Response from batch ingestion."""

    total: int
    succeeded: int
    failed: int
    results: list[IngestResponse]


class ConverterStatusResponse(BaseModel):
    """Converter health and capabilities."""

    docling_available: bool
    supported_formats: list[str]
    supported_extensions: list[str]
    table_extraction: bool
    ocr_available: bool


# ── Helper ──────────────────────────────────────────────────────────────────

def _get_pipeline():
    """Lazy-load the pipeline to avoid importing heavy deps at module load."""
    from resync.knowledge.ingestion.pipeline import DocumentIngestionPipeline
    from resync.knowledge.ingestion.embedding_service import get_embedder
    from resync.knowledge.store import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()
    return DocumentIngestionPipeline(embedder, store)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    dependencies=[Depends(verify_admin_credentials)],
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, HTML, MD, XLSX)"),
    tenant: str = Form(default="default"),
    tags: str = Form(default="", description="Comma-separated tags"),
    chunking_strategy: str = Form(default="tws_optimized"),
    reindex: bool = Form(default=False),
    doc_id: str | None = Form(default=None),
):
    """
    Upload a document and ingest it into the RAG knowledge base.

    The document is converted using Docling (PDF tables, layout analysis)
    and then chunked, embedded, and stored in pgvector.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_filename = Path(file.filename).name
    ext = Path(safe_filename).suffix.lower()
    if ext not in FORMAT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {list(FORMAT_EXTENSIONS.keys())}",
        )

    # Save to temp file
    tmp_dir = tempfile.mkdtemp(prefix="resync_ingest_")
    tmp_path = Path(tmp_dir) / safe_filename

    try:
        # Stream upload to disk without blocking the event loop
        import anyio

        async with await anyio.open_file(tmp_path, "wb") as tmp_file:
            while chunk := await file.read(8192):
                await tmp_file.write(chunk)

        file_size = tmp_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size / 1024 / 1024:.1f} MB (max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB)",
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Run pipeline
        pipeline = _get_pipeline()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        result = await pipeline.ingest_file(
            tmp_path,
            tenant=tenant,
            doc_id=doc_id,
            tags=tag_list,
            chunking_strategy=chunking_strategy,
            reindex=reindex,
        )

        return IngestResponse(
            status=result.status,
            doc_id=result.doc_id,
            source=result.source,
            format=result.format,
            pages=result.pages,
            tables_extracted=result.tables_extracted,
            chunks_stored=result.chunks_stored,
            conversion_time_s=round(result.conversion_time_s, 2),
            total_time_s=round(result.total_time_s, 2),
            error=result.error,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post(
    "/batch",
    response_model=BatchIngestResponse,
    summary="Ingest multiple documents by server path",
    dependencies=[Depends(verify_admin_credentials)],
)
async def ingest_batch(request: BatchIngestRequest):
    """
    Ingest multiple documents already on the server filesystem.

    Useful for bulk ingestion of manuals and runbooks from a shared directory.
    """
    pipeline = _get_pipeline()
    results = []
    succeeded = 0
    failed = 0

    for fp in request.file_paths:
        path = Path(fp)
        if not path.exists():
            results.append(IngestResponse(
                status="error",
                doc_id="",
                source=fp,
                format="",
                error=f"File not found: {fp}",
            ))
            failed += 1
            continue

        if not DoclingConverter.is_supported(path):
            results.append(IngestResponse(
                status="error",
                doc_id="",
                source=fp,
                format="",
                error=f"Unsupported format: {path.suffix}",
            ))
            failed += 1
            continue

        result = await pipeline.ingest_file(
            path,
            tenant=request.tenant,
            tags=request.tags,
            chunking_strategy=request.chunking_strategy,
            reindex=request.reindex,
        )

        resp = IngestResponse(
            status=result.status,
            doc_id=result.doc_id,
            source=result.source,
            format=result.format,
            pages=result.pages,
            tables_extracted=result.tables_extracted,
            chunks_stored=result.chunks_stored,
            conversion_time_s=round(result.conversion_time_s, 2),
            total_time_s=round(result.total_time_s, 2),
            error=result.error,
        )
        results.append(resp)

        if result.status == "success":
            succeeded += 1
        else:
            failed += 1

    return BatchIngestResponse(
        total=len(request.file_paths),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.get(
    "/status",
    response_model=ConverterStatusResponse,
    summary="Check converter status and supported formats",
)
async def converter_status():
    """Check Docling status and supported formats."""
    docling_available = False
    try:
        import docling  # noqa: F401
        docling_available = True
    except ImportError:
        logger.error("docling_not_installed — pip install docling is required")

    return ConverterStatusResponse(
        docling_available=docling_available,
        supported_formats=[f.value for f in SupportedFormat],
        supported_extensions=list(FORMAT_EXTENSIONS.keys()),
        table_extraction=docling_available,
        ocr_available=docling_available,
    )
