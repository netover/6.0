"""File upload validation models for API endpoints."""

import os
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic import StringConstraints as PydanticStringConstraints

from .common import BaseValidatedModel, NumericConstraints, StringConstraints, ValidationPatterns


class FileType(str, Enum):
    """Allowed file types."""

    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    DATA = "data"
    ARCHIVE = "archive"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """File processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileUploadRequest(BaseValidatedModel):
    """File upload request validation."""

    filename: StringConstraints.FILENAME = Field(..., description="Original filename")

    file_size: int = Field(
        ..., ge=1, le=NumericConstraints.MAX_FILE_SIZE, description="File size in bytes"
    )

    content_type: str = Field(..., description="MIME content type")

    file_type: FileType | None = Field(None, description="Categorized file type")

    purpose: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
    ] = Field(..., description="Purpose of file upload")

    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional file metadata", max_length=50
    )

    sanitize_content: bool = Field(default=True, description="Whether to sanitize file content")

    extract_text: bool = Field(default=False, description="Whether to extract text content")

    generate_thumbnail: bool = Field(default=False, description="Whether to generate thumbnail")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        """Validate filename for security and format."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        # Remove path traversal attempts
        v = os.path.basename(v)
        # Check for dangerous patterns
        if ValidationPatterns.PATH_TRAVERSAL_PATTERN.search(v):
            raise ValueError("Filename contains path traversal patterns")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Filename contains potentially malicious content")
        # Check for double extensions (potential evasion)
        if v.count(".") > 2:
            raise ValueError("Filename has too many extensions")
        # Validate extension
        extension = os.path.splitext(v)[1].lower()
        if extension and not re.match(r"^\.[a-zA-Z0-9]{1,10}$", extension):
            raise ValueError("Invalid file extension")
        return v

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v):
        """Validate MIME content type."""
        if not v or not v.strip():
            raise ValueError("Content type cannot be empty")
        # Basic MIME type validation
        if not re.match(r"^[a-zA-Z0-9\-]+\/[a-zA-Z0-9\-\+]+(;.*)?$", v):
            raise ValueError("Invalid MIME content type format")
        # Check for dangerous content types
        dangerous_types = {
            "application/x-php",
            "application/x-javascript",
            "text/javascript",
            "application/javascript",
            "application/x-sh",
            "application/x-csh",
            "application/x-perl",
            "application/x-python",
            "application/x-ruby",
            "application/x-java-jnlp-file",
            "application/x-msdownload",
            "application/x-msdos-program",
            "application/vnd.microsoft.portable-executable",
            "application/x-executable",
            "application/x-sharedlib",
            "application/x-dosexec",
        }
        base_type = v.split(";")[0].lower()
        if base_type in dangerous_types:
            raise ValueError(f"Dangerous content type not allowed: {base_type}")
        return v

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size limits."""
        if v > NumericConstraints.MAX_FILE_SIZE:
            raise ValueError(
                f"File size exceeds maximum allowed: {NumericConstraints.MAX_FILE_SIZE} bytes"
            )
        if v > NumericConstraints.MAX_FILE_SIZE // 2:
            import logging

            logging.warning(f"Large file upload detected: {v} bytes")
        return v

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, v):
        """Validate file upload purpose."""
        if not v or not v.strip():
            raise ValueError("Purpose cannot be empty")
        # Check for script injection
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Purpose contains potentially malicious content")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata dictionary."""
        if not v:
            return v
        # Check metadata keys and values
        for key, value in v.items():
            # Validate key
            if not key.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid metadata key: {key}")
            # Validate string values
            if isinstance(value, str):
                if len(value) > 1000:  # Max metadata value length
                    raise ValueError(f"Metadata value too long for key '{key}'")
                if ValidationPatterns.SCRIPT_PATTERN.search(value):
                    raise ValueError(f"Metadata value contains malicious content for key '{key}'")
            # Validate nested dictionaries
            elif isinstance(value, dict):
                if len(value) > 10:  # Max nested items
                    raise ValueError(f"Too many nested items in metadata key '{key}'")
                for nested_key, nested_value in value.items():
                    if not nested_key.replace("_", "").isalnum():
                        raise ValueError(f"Invalid nested metadata key: {nested_key}")
                    if isinstance(nested_value, str) and ValidationPatterns.SCRIPT_PATTERN.search(
                        nested_value
                    ):
                        raise ValueError("Nested metadata value contains malicious content")
        return v

    @model_validator(mode="before")
    def validate_type_consistency(self, values):
        """Validate consistency between content type and file type."""
        if isinstance(values, dict):
            content_type = values.get("content_type")
            file_type = values.get("file_type")
            if content_type and file_type:
                # Infer expected file type from content type
                main_type = content_type.split("/")[0].lower()
                type_mapping = {
                    "text": FileType.DOCUMENT,
                    "image": FileType.IMAGE,
                    "video": FileType.VIDEO,
                    "audio": FileType.AUDIO,
                    "application": FileType.DOCUMENT,  # Default for application types
                }
                expected_type = type_mapping.get(main_type, FileType.OTHER)
                if file_type != expected_type:
                    import logging

                    logging.warning(
                        f"File type '{file_type}' doesn't match content type '{content_type}'"
                    )
        return values


class FileChunkUploadRequest(BaseValidatedModel):
    """Chunked file upload request validation."""

    upload_id: StringConstraints.SAFE_TEXT = Field(..., description="Unique upload session ID")

    chunk_index: int = Field(..., ge=0, description="Chunk index (0-based)")

    total_chunks: int = Field(..., ge=1, le=1000, description="Total number of chunks")

    chunk_size: int = Field(
        ...,
        ge=1,
        le=NumericConstraints.MAX_CHUNK_SIZE,
        description="Size of this chunk in bytes",
    )

    file_size: int = Field(
        ...,
        ge=1,
        le=NumericConstraints.MAX_FILE_SIZE,
        description="Total file size in bytes",
    )

    filename: StringConstraints.FILENAME = Field(..., description="Original filename")

    content_type: str = Field(..., description="MIME content type")

    checksum: str | None = Field(
        None, description="MD5 hash of chunk data", pattern=r"^[a-fA-F0-9]{32}$"
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("chunk_index", "total_chunks")
    @classmethod
    def validate_chunk_numbers(cls, v, info):
        """Validate chunk numbers."""
        if "chunk_index" in info.data and "total_chunks" in info.data:
            chunk_index = info.data["chunk_index"]
            total_chunks = info.data["total_chunks"]
            if chunk_index >= total_chunks:
                raise ValueError("Chunk index must be less than total chunks")
        return v

    @field_validator("file_size")
    @classmethod
    def validate_total_size(cls, v, info):
        """Validate total file size."""
        chunk_size = info.data.get("chunk_size")
        total_chunks = info.data.get("total_chunks")
        if chunk_size and total_chunks:
            expected_size = chunk_size * total_chunks
            # Allow some variance for the last chunk
            if abs(v - expected_size) > NumericConstraints.MAX_CHUNK_SIZE:
                raise ValueError("File size doesn't match expected size from chunks")
        return v


class FileUpdateRequest(BaseValidatedModel):
    """File update request validation."""

    filename: StringConstraints.FILENAME | None = Field(None, description="New filename")

    purpose: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="New purpose")

    metadata: dict[str, Any] | None = Field(None, description="Updated metadata", max_length=50)

    tags: list[Annotated[str, PydanticStringConstraints(min_length=1, max_length=50)]] | None = (
        Field(None, description="File tags", max_length=10)
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        """Validate filename if provided."""
        if v is None:
            return v
        if not v.strip():
            raise ValueError("Filename cannot be empty")
        # Remove path traversal attempts
        v = os.path.basename(v)
        if ValidationPatterns.PATH_TRAVERSAL_PATTERN.search(v):
            raise ValueError("Filename contains path traversal patterns")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Filename contains potentially malicious content")
        return v

    @field_validator("purpose")
    @classmethod
    def validate_purpose(cls, v):
        """Validate purpose if provided."""
        if v is None:
            return v
        if not v.strip():
            raise ValueError("Purpose cannot be empty")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Purpose contains potentially malicious content")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata if provided."""
        if v is None:
            return v
        # Check metadata keys and values
        for key, value in v.items():
            if not key.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid metadata key: {key}")
            if isinstance(value, str):
                if len(value) > 1000:
                    raise ValueError(f"Metadata value too long for key '{key}'")
                if ValidationPatterns.SCRIPT_PATTERN.search(value):
                    raise ValueError(f"Metadata value contains malicious content for key '{key}'")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate file tags."""
        if v is None:
            return v
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate tags found")
        # Validate individual tags
        for tag in v:
            if ValidationPatterns.SCRIPT_PATTERN.search(tag):
                raise ValueError(f"Tag contains malicious content: {tag}")
        return v


class FileProcessingRequest(BaseValidatedModel):
    """File processing request validation."""

    operations: list[str] = Field(
        ..., description="Processing operations to perform", min_length=1, max_length=10
    )

    configuration: dict[str, Any] | None = Field(
        default_factory=dict, description="Processing configuration", max_length=20
    )

    priority: str = Field(
        default="normal",
        pattern=r"^(low|normal|high|urgent)$",
        description="Processing priority",
    )

    callback_url: str | None = Field(None, description="Callback URL for processing completion")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("operations")
    @classmethod
    def validate_operations(cls, v):
        """Validate processing operations."""
        valid_operations = {
            "extract_text",
            "ocr",
            "thumbnail",
            "metadata",
            "virus_scan",
            "convert_format",
            "compress",
            "resize",
            "crop",
            "rotate",
            "watermark",
            "encrypt",
            "decrypt",
            "sign",
            "validate",
        }
        # Check for valid operations
        invalid_ops = set(v) - valid_operations
        if invalid_ops:
            raise ValueError(f"Invalid operations: {invalid_ops}")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate operations found")
        return v

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, v):
        """Validate callback URL."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("Callback URL must use HTTP or HTTPS protocol")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Callback URL contains potentially malicious content")
        return v

    @field_validator("configuration")
    @classmethod
    def validate_configuration(cls, v):
        """Validate processing configuration."""
        if not v:
            return v
        # Validate configuration keys and values
        for key, value in v.items():
            if not key.replace("_", "-").replace(".", "").isalnum():
                raise ValueError(f"Invalid configuration key: {key}")
            if isinstance(value, str):
                if len(value) > 500:
                    raise ValueError(f"Configuration value too long for key '{key}'")
                if ValidationPatterns.SCRIPT_PATTERN.search(value):
                    raise ValueError(
                        f"Configuration value contains malicious content for key '{key}'"
                    )
        return v


class RAGUploadRequest(BaseValidatedModel):
    """RAG (Retrieval-Augmented Generation) file upload request."""

    files: list[FileUploadRequest] = Field(
        ..., description="Files to process for RAG", min_length=1, max_length=10
    )

    collection_name: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
    ] = Field(..., description="RAG collection name")

    chunk_size: int = Field(
        default=1000, ge=100, le=10000, description="Text chunk size for processing"
    )

    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")

    embedding_model: StringConstraints.MODEL_NAME | None = Field(
        None, description="Embedding model to use"
    )

    metadata_template: dict[str, str] | None = Field(
        default_factory=dict,
        description="Template for document metadata",
        max_length=20,
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("files")
    @classmethod
    def validate_files_list(cls, v):
        """Validate files list."""
        if len(v) > 10:
            raise ValueError("Too many files. Maximum 10 files allowed.")
        # Check for duplicate filenames
        filenames = [f.filename for f in v]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate filenames found")
        return v

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v):
        """Validate collection name."""
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Collection name contains potentially malicious content")
        return v

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_chunk_values(cls, v):
        """Validate chunk configuration."""
        if v <= 0:
            raise ValueError("Chunk size and overlap must be positive")
        return v

    @model_validator(mode="before")
    def validate_chunk_configuration(self, values):
        """Validate chunk size and overlap relationship."""
        if isinstance(values, dict):
            chunk_size = values.get("chunk_size", 1000)
            chunk_overlap = values.get("chunk_overlap", 200)
            if chunk_overlap >= chunk_size:
                raise ValueError("Chunk overlap must be less than chunk size")
        return values

    @field_validator("metadata_template")
    @classmethod
    def validate_metadata_template(cls, v):
        """Validate metadata template."""
        if not v:
            return v
        for key, template in v.items():
            if not key.replace("_", "").isalnum():
                raise ValueError(f"Invalid metadata template key: {key}")
            if ValidationPatterns.SCRIPT_PATTERN.search(template):
                raise ValueError(f"Metadata template contains malicious content for key '{key}'")
        return v


# File response models
class FileInfo(BaseValidatedModel):
    """File information response."""

    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    file_type: FileType = Field(..., description="Categorized file type")
    status: ProcessingStatus = Field(..., description="Processing status")
    upload_date: datetime = Field(..., description="Upload timestamp")
    checksum: str | None = Field(None, description="File checksum")
    metadata: dict[str, Any] = Field(default_factory=dict, description="File metadata")
    tags: list[str] = Field(default_factory=list, description="File tags")

    model_config = ConfigDict(
        extra="forbid",
    )


__all__ = [
    "FileUploadRequest",
    "FileChunkUploadRequest",
    "FileUpdateRequest",
    "FileProcessingRequest",
    "RAGUploadRequest",
    "FileInfo",
    "FileType",
    "ProcessingStatus",
]
