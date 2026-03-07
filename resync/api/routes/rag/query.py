"""
RAG (Retrieval-Augmented Generation) routes for FastAPI

Provides endpoints for:
- File upload and ingestion
- Semantic search
- Document management
- RAG statistics
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import uuid
import aiofiles
from werkzeug.utils import secure_filename
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from resync.api.dependencies_v2 import get_current_user, get_logger
from resync.api.models.requests import FileUploadValidation
from resync.api.models.responses_v2 import FileUploadResponse
from pydantic import BaseModel
from ...services.rag_service import RAGIntegrationService, get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".json"}


class RagSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict[str, object] | None = None


class RagSearchResponse(BaseModel):
    query: str
    results: list[RagSearchResult]
    total: int


class RagFileSummary(BaseModel):
    file_id: str
    filename: str
    status: str
    chunks_count: int
    created_at: object
    processed_at: object | None = None


class RagFileListResponse(BaseModel):
    files: list[RagFileSummary]
    total: int


class RagFileDetailResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    chunks_count: int
    created_at: object
    processed_at: object | None = None
    metadata: dict[str, object] | None = None


class RagDeleteResponse(BaseModel):
    message: str
    file_id: str


class RagStatsResponse(BaseModel):
    stats: dict[str, object]

def get_rag() -> RAGIntegrationService:
    """Dependency to get RAG service."""
    return get_rag_service()

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file using Pydantic model"""
    try:
        validation_model = FileUploadValidation(
            filename=file.filename or "",
            content_type=file.content_type or "",
            size=file.size or 0,
        )
        validation_model.validate_file()
    except ValueError as e:
        logger.error(
            "request_failed",
            exc_info=True,
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request. Check server logs for details.",
        ) from e
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("file_validation_failed", exc_info=True, extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File validation failed. Check server logs for details.",
        ) from e

async def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """Save uploaded file to disk using streaming (OOM protected)."""
    # P0-18 FIX: Stream the file in chunks instead of reading all into RAM
    # P2-40 FIX: Use aiofiles properly (not anyio.open_file with await)
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    try:
        async with aiofiles.open(destination, "wb") as buffer:
            while chunk := await upload_file.read(CHUNK_SIZE):
                await buffer.write(chunk)
        
        # Return file path for later processing, not decoded content
        return str(destination)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file. Check server logs for details.",
        ) from e

async def process_rag_document(
    rag_service: RAGIntegrationService,
    file_id: str,
    filename: str,
    file_path: str,
    tags: list[str],
):
    """Background task to process document for RAG."""
    try:
        # P0-18 FIX: Read content from file path (not from memory)
        # Use asyncio.to_thread for blocking I/O
        import aiofiles
        
        # Determine if file is text-based
        text_extensions = {'.txt', '.md', '.json', '.log'}
        ext = Path(filename).suffix.lower()
        
        if ext in text_extensions:
            # Read text files with proper encoding
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        else:
            # For binary files (PDF, DOCX), pass the path to the RAG service
            # The service will handle extraction
            content = file_path  # Pass path for non-text files
        
        await rag_service.ingest_document(
            file_id=file_id, filename=filename, content=content, tags=tags
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        import logging

        logging.error(f"RAG processing failed for {file_id}: {e}")

@router.post("/rag/upload", response_model=FileUploadResponse)
async def upload_rag_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: str = Query(default="", description="Comma-separated tags"),
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """
    Upload file for RAG processing.

    The file is saved and queued for background processing which includes:
    - Text extraction
    - Chunking
    - Embedding generation
    - Vector storage
    """
    try:
        validate_file(file)
        
        # P0-19 FIX: Rigid Path Traversal protection
        raw_filename = file.filename if file.filename else "upload.bin"
        safe_filename = secure_filename(raw_filename)
        
        if not safe_filename:
            safe_filename = f"upload_{uuid.uuid4().hex[:8]}.bin"
            
        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}_{safe_filename}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save in chunks - returns file path, not decoded content
        file_path_str = await save_upload_file(file, file_path)
        
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # P0-18 FIX: Pass file path to background task, let it read from disk
        background_tasks.add_task(
            process_rag_document,
            rag_service,
            file_id,
            safe_filename,
            file_path_str,  # Pass path instead of content
            tag_list,
        )
        upload_response = FileUploadResponse(
            filename=safe_filename,
            status="processing",
            file_id=file_id,
            upload_time=datetime.now(timezone.utc).isoformat(),
        )
        logger_instance.info(
            "rag_file_uploaded",
            user_id=current_user.get("user_id"),
            filename=safe_filename,
            file_id=file_id,
            file_size=file.size,
            tags=tag_list,
        )
        return upload_response
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error(
            "rag_upload_error",
            error=str(e),
            filename=Path(file.filename or "").name,
            user_id=current_user.get("user_id"),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process file upload",
        ) from e

@router.get("/rag/search", response_model=RagSearchResponse)
async def search_rag(
    query: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """
    Search for relevant documents using semantic search.

    Returns chunks most similar to the query.
    """
    try:
        results = await rag_service.search(query=query, top_k=top_k)
        logger_instance.info(
            "rag_search",
            user_id=current_user.get("user_id"),
            query=query[:50],
            results_count=len(results),
        )
        return RagSearchResponse(
            query=query,
            results=[
                RagSearchResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_search_error", error=str(e), query=query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        ) from e

@router.get("/rag/files", response_model=RagFileListResponse)
async def list_rag_files(
    status_filter: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """List uploaded RAG files with optional filtering."""
    try:
        docs = await rag_service.list_documents(status=status_filter, limit=limit)
        files = [
            {
                "file_id": doc.file_id,
                "filename": doc.filename,
                "status": doc.status,
                "chunks_count": doc.chunks_count,
                "created_at": doc.created_at,
                "processed_at": doc.processed_at,
            }
            for doc in docs
        ]
        logger_instance.info(
            "rag_files_listed",
            user_id=current_user.get("user_id"),
            file_count=len(files),
        )
        return RagFileListResponse(files=[RagFileSummary(**file_data) for file_data in files], total=len(files))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_files_listing_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list RAG files",
        ) from e

@router.get("/rag/files/{file_id}", response_model=RagFileDetailResponse)
async def get_rag_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Get details of a specific RAG file."""
    doc = await rag_service.get_document(file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {file_id} not found",
        )
    return RagFileDetailResponse(
        file_id=doc.file_id,
        filename=doc.filename,
        status=doc.status,
        chunks_count=doc.chunks_count,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
        metadata=doc.metadata,
    )

@router.delete("/rag/files/{file_id}", response_model=RagDeleteResponse)
async def delete_rag_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Delete RAG file and its associated chunks."""
    try:
        deleted = await rag_service.delete_document(file_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {file_id} not found",
            )
        logger_instance.info(
            "rag_file_deleted", user_id=current_user.get("user_id"), file_id=file_id
        )
        return RagDeleteResponse(message="File deleted successfully", file_id=file_id)
    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_file_deletion_error", error=str(e), file_id=file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete RAG file",
        ) from e

@router.get("/rag/stats", response_model=RagStatsResponse)
async def get_rag_stats(
    current_user: dict = Depends(get_current_user),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Get RAG system statistics."""
    return RagStatsResponse(stats=await rag_service.get_stats())
