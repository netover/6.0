from datetime import datetime, timezone

"""
RAG (Retrieval-Augmented Generation) routes for FastAPI

Provides endpoints for:
- File upload and ingestion
- Semantic search
- Document management
- RAG statistics
"""
from pathlib import Path

import anyio

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

import logging
logger = logging.getLogger(__name__)

from ...services.rag_service import RAGIntegrationService, get_rag_service

router = APIRouter()

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".json"}


def get_rag() -> RAGIntegrationService:
    """Dependency to get RAG service."""
    return get_rag_service()


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file using Pydantic model"""
    try:
        validation_model = FileUploadValidation(
            filename=file.filename or "", content_type=file.content_type or "", size=file.size or 0
        )
        validation_model.validate_file()
    except ValueError as e:
        logger.error("request_failed", error=str(e), error_type=type(e).__name__, exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request. Check server logs for details.") from e
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("file_validation_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File validation failed. Check server logs for details.",
        ) from e


async def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """Save uploaded file to disk and return content."""
    try:
        content = await upload_file.read()
        async with await anyio.open_file(destination, "wb") as buffer:
            await buffer.write(content)
        return content.decode("utf-8", errors="ignore")
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
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
    content: str,
    tags: list[str],
):
    """Background task to process document for RAG."""
    try:
        await rag_service.ingest_document(
            file_id=file_id,
            filename=filename,
            content=content,
            tags=tags,
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        # Log error but don't raise - background task
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

        import uuid


        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / unique_filename

        # Save file and get content
        content = await save_upload_file(file, file_path)

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Queue background processing
        background_tasks.add_task(
            process_rag_document,
            rag_service,
            file_id,
            file.filename,
            content,
            tag_list,
        )

        upload_response = FileUploadResponse(
            filename=file.filename,
            status="processing",
            file_id=file_id,
            upload_time=datetime.now(timezone.utc).isoformat(),
        )

        logger_instance.info(
            "rag_file_uploaded",
            user_id=current_user.get("user_id"),
            filename=file.filename,
            file_id=file_id,
            file_size=file.size,
            tags=tag_list,
        )

        return upload_response

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error(
            "rag_upload_error",
            error=str(e),
            filename=file.filename,
            user_id=current_user.get("user_id"),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process file upload",
        ) from e


@router.get("/rag/search")
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

        return {
            "query": query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_search_error", error=str(e), query=query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        ) from e


@router.get("/rag/files")
async def list_rag_files(
    status_filter: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """List uploaded RAG files with optional filtering."""
    try:
        docs = rag_service.list_documents(status=status_filter, limit=limit)

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
            "rag_files_listed", user_id=current_user.get("user_id"), file_count=len(files)
        )

        return {"files": files, "total": len(files)}

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_files_listing_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list RAG files"
        ) from e


@router.get("/rag/files/{file_id}")
async def get_rag_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Get details of a specific RAG file."""
    doc = rag_service.get_document(file_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {file_id} not found"
        )

    return {
        "file_id": doc.file_id,
        "filename": doc.filename,
        "status": doc.status,
        "chunks_count": doc.chunks_count,
        "created_at": doc.created_at,
        "processed_at": doc.processed_at,
        "metadata": doc.metadata,
    }


@router.delete("/rag/files/{file_id}")
async def delete_rag_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    logger_instance=Depends(get_logger),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Delete RAG file and its associated chunks."""
    try:
        deleted = rag_service.delete_document(file_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {file_id} not found"
            )

        logger_instance.info(
            "rag_file_deleted", user_id=current_user.get("user_id"), file_id=file_id
        )

        return {"message": "File deleted successfully", "file_id": file_id}

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger_instance.error("rag_file_deletion_error", error=str(e), file_id=file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete RAG file"
        ) from e


@router.get("/rag/stats")
async def get_rag_stats(
    current_user: dict = Depends(get_current_user),
    rag_service: RAGIntegrationService = Depends(get_rag),
):
    """Get RAG system statistics."""
    return rag_service.get_stats()
