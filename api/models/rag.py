"""RAG (Retrieval-Augmented Generation) models."""

from pydantic import BaseModel

from .base import BaseModelWithTime


class RAGFileMetaData(BaseModel):
    """Metadata for RAG file uploads."""

    filename: str
    content_type: str
    uploaded_by: str | None = None
    description: str | None = None


class RAGFileCreate(BaseModelWithTime):
    """RAG file creation request model."""

    filename: str
    content_type: str
    metadata: RAGFileMetaData | None = None


class RAGFileDetail(RAGFileCreate):
    """Detailed RAG file information."""

    id: str
    file_size: int
    ingestion_status: str
