"""
RAG upload endpoint module.

This module defines a FastAPI route for uploading files to the Retrieval‑Augmented
Generation (RAG) pipeline. The endpoint validates file size and metadata using
Pydantic models and delegates saving and ingestion to the configured file
ingestor. Errors are handled explicitly and returned as HTTP exceptions.

Note: `` must appear at the top of the file
before any other import statements to satisfy Python's import rules. See
PEP 563 and PEP 649 for details.
"""


# resync/api/rag_upload.py

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from resync.core.exceptions import FileProcessingError
from resync.core.fastapi_di import get_file_ingestor
from resync.core.interfaces import IFileIngestor
from resync.models.validation import DocumentUpload

logger = logging.getLogger(__name__)

# Module-level dependencies to avoid B008 errors
file_dependency = File(...)
file_ingestor_dependency = Depends(get_file_ingestor)

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/upload", summary="Upload a document for RAG ingestion")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = file_dependency,
    file_ingestor: IFileIngestor = file_ingestor_dependency,
):
    """
    Accepts a file upload and saves it to the RAG directory for processing.
    """
    try:
        # Check file size by reading and limiting
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

        # Reset file pointer for saving
        await file.seek(0)

        # Validate the document using our validation model
        try:
            document_upload = DocumentUpload(
                filename=file.filename or "",
                content_type=file.content_type or "application/octet-stream",
                size=len(contents),
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve

        # Save the uploaded file
        destination = await file_ingestor.save_uploaded_file(
            file_name=document_upload.filename,
            file_content=file.file,
        )

        # Start file ingestion in the background
        # We don't want to block the response while processing potentially large files
        background_tasks.add_task(file_ingestor.ingest_file, destination)

        # Get the filename from the path
        safe_filename = destination.name

        return {
            "filename": safe_filename,
            "content_type": document_upload.content_type,
            "size": document_upload.size,
            "message": "File uploaded successfully and queued for ingestion.",
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileProcessingError as e:
        logger.error("File processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid request. Check server logs for details.") from e
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to process uploaded file: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Could not process file. Check server logs for details.",
        ) from e
    finally:
        await file.close()
