"""
RAG upload endpoint module.

This module defines a FastAPI route for uploading files to the Retrieval‑Augmented
Generation (RAG) pipeline. The endpoint validates file size and metadata using
Pydantic models and delegates saving and ingestion to the configured file
ingestor. Errors are handled explicitly and returned as HTTP exceptions.

Note: `` must appear at the top of the file
before any other import statements to satisfy Python's import rules. See
PEP\xa0563 and PEP\xa0649 for details.
"""
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from resync.core.exceptions import FileProcessingError
from resync.core.fastapi_di import get_file_ingestor
from resync.core.interfaces import IFileIngestor
from resync.models.validation import DocumentUpload
from resync.api.dependencies_v2 import get_current_user
from resync.settings import get_settings
logger = logging.getLogger(__name__)
file_dependency = File(...)
file_ingestor_dependency = Depends(get_file_ingestor)
router = APIRouter(prefix='/api/rag', tags=['rag'])

@router.post('/upload', summary='Upload a document for RAG ingestion')
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile=file_dependency, file_ingestor: IFileIngestor=file_ingestor_dependency, current_user: dict=Depends(get_current_user)):
    """
    Accepts a file upload and saves it to the RAG directory for processing.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail='Authentication required')
    settings = get_settings()
    try:
        contents = await file.read()
        if len(contents) > settings.max_file_size:
            raise HTTPException(status_code=400, detail=f'File too large. Maximum size is {settings.max_file_size / (1024 * 1024):.1f}MB.')
        await file.seek(0)
        try:
            document_upload = DocumentUpload(filename=file.filename or '', content_type=file.content_type or 'application/octet-stream', size=len(contents))
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        destination = await file_ingestor.save_uploaded_file(file_name=document_upload.filename, file_content=file.file)
        background_tasks.add_task(file_ingestor.ingest_file, destination)
        safe_filename = destination.name
        logger.info('rag_document_uploaded', extra={'user': current_user.get('username'), 'filename': safe_filename, 'size': document_upload.size})
        return {'filename': safe_filename, 'content_type': document_upload.content_type, 'size': document_upload.size, 'message': 'File uploaded successfully and queued for ingestion.'}
    except HTTPException:
        raise
    except FileProcessingError as e:
        logger.error('File processing error: %s', e, exc_info=True)
        raise HTTPException(status_code=400, detail='Invalid request. Check server logs for details.') from e
    except Exception as e:
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error('Failed to process uploaded file: %s', e, exc_info=True)
        raise HTTPException(status_code=500, detail='Could not process file. Check server logs for details.') from e
    finally:
        await file.close()