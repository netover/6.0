"""
Data validation models for the Resync application
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

class SyncItem(BaseModel):
    """
    Model for sync items with validation.
    """

    model_config = ConfigDict(
        validate_assignment=True,  # Revalidate on field assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
        arbitrary_types_allowed=True,
    )

    id: int
    content: str
    checksum: str
    created_at: datetime | None = None

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        """
        Validate that the checksum is a proper SHA-256 hash (64 hex characters).

        Args:
            v: The checksum string to validate

        Returns:
            The validated checksum string

        Raises:
            ValueError: If the checksum format is invalid
        """
        if not re.match(r"^[a-f0-9]{64}$", v):
            raise ValueError("Invalid SHA-256 checksum format")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """
        Validate content field.

        Args:
            v: The content string to validate

        Returns:
            The validated content string
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Content cannot be empty")
        if len(v) > 10000:  # Prevent overly large content
            raise ValueError("Content too long (max 10000 chars)")
        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: int) -> int:
        """
        Validate ID field.

        Args:
            v: The ID to validate

        Returns:
            The validated ID
        """
        if v <= 0:
            raise ValueError("ID must be a positive integer")
        return v

    @field_serializer("created_at")
    def serialize_datetime(self, dt: datetime) -> str:
        """
        Serialize datetime to ISO format.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.isoformat()

class AuditRecord(BaseModel):
    """
    Model for audit records with validation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    memory_id: str
    user_query: str
    agent_response: str
    user_id: str | None = None  # Add user context
    ia_audit_reason: str | None = None
    ia_audit_confidence: float | None = None
    status: str = "pending"
    timestamp: datetime | None = None  # Add timestamp

    @field_validator("memory_id")
    @classmethod
    def validate_memory_id(cls, v: str) -> str:
        """
        Validate memory ID format.

        Args:
            v: The memory ID to validate

        Returns:
            The validated memory ID
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Memory ID cannot be empty")
        if len(v) > 255:
            raise ValueError("Memory ID too long (max 255 chars)")
        if "\x00" in v:
            raise ValueError("Memory ID cannot contain null bytes")
        # Additional validation: alphanumeric with some special chars
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Memory ID contains invalid characters")
        return v

    @field_validator("user_query")
    @classmethod
    def validate_user_query(cls, v: str) -> str:
        """
        Validate user query.

        Args:
            v: The user query to validate

        Returns:
            The validated user query
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("User query cannot be empty")
        if len(v) > 10000:
            raise ValueError("User query too long (max 10000 chars)")
        if "\x00" in v:
            raise ValueError("User query cannot contain null bytes")
        return v

    @field_validator("agent_response")
    @classmethod
    def validate_agent_response(cls, v: str) -> str:
        """
        Validate agent response.

        Args:
            v: The agent response to validate

        Returns:
            The validated agent response
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Agent response cannot be empty")
        if len(v) > 50000:
            raise ValueError("Agent response too long (max 50000 chars)")
        if "\x00" in v:
            raise ValueError("Agent response cannot contain null bytes")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str | None) -> str | None:
        """
        Validate user ID if provided.

        Args:
            v: The user ID to validate

        Returns:
            The validated user ID
        """
        if v is None:
            return v  # Optional field
        if len(v) > 100:
            raise ValueError("User ID too long (max 100 chars)")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("User ID contains invalid characters")
        return v

    @field_validator("ia_audit_confidence")
    @classmethod
    def validate_ia_audit_confidence(cls, v: float | None) -> float | None:
        """
        Validate IA audit confidence.

        Args:
            v: The confidence value to validate

        Returns:
            The validated confidence value
        """
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("IA audit confidence must be between 0.0 and 1.0")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """
        Validate status value.

        Args:
            v: The status to validate

        Returns:
            The validated status
        """
        valid_statuses = {"pending", "approved", "rejected"}
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime) -> str:
        """
        Serialize timestamp to ISO format.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.isoformat()

class JobStatusUpdate(BaseModel):
    """
    Model for job status updates with validation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    job_id: str
    status: str
    reason: str | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """
        Validate job ID format.

        Args:
            v: The job ID to validate

        Returns:
            The validated job ID
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Job ID cannot be empty")
        # Allow alphanumeric, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid job ID format")
        return v

    @field_validator("status")
    @classmethod
    def validate_job_status(cls, v: str) -> str:
        """
        Validate job status value.

        Args:
            v: The status to validate

        Returns:
            The validated status
        """
        valid_statuses = {"running", "completed", "failed", "cancelled", "pending"}
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid job status: {v}. Must be one of {valid_statuses}"
            )
        return v.lower()

class DocumentUpload(BaseModel):
    """
    Model for document upload validation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    filename: str
    content_type: str
    size: int

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """
        Validate filename format.

        Args:
            v: The filename to validate

        Returns:
            The validated filename
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Filename cannot be empty")

        # Check for invalid characters in filenames
        invalid_chars = '<>:"/\\|?*'
        if any(char in v for char in invalid_chars):
            raise ValueError("Filename contains invalid characters")

        # Check file extension (only allow specific file types)
        allowed_extensions = {".pdf", ".docx", ".xlsx", ".txt", ".csv", ".json"}
        file_extension = ""
        if "." in v:
            file_extension = "." + v.lower().split(".")[-1]

        if file_extension not in allowed_extensions:
            raise ValueError(
                f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )

        return v

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """
        Validate content type.

        Args:
            v: The content type to validate

        Returns:
            The validated content type
        """
        if not v or len(v.strip()) == 0:
            raise ValueError("Content type cannot be empty")

        # Check if it's a valid content type
        valid_content_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/plain",
            "text/csv",
            "application/json",
        ]

        if not any(allowed_type in v.lower() for allowed_type in valid_content_types):
            raise ValueError("Unsupported content type")

        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        """
        Validate file size.

        Args:
            v: The file size to validate

        Returns:
            The validated file size
        """
        if v <= 0:
            raise ValueError("File size must be positive")

        # Use settings for dynamic limit, fallback to 10MB
        try:
            from resync.settings import get_settings

            max_size = get_settings().max_file_size
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
            max_size = 10 * 1024 * 1024

        if v > max_size:
            raise ValueError(
                f"File too large. Maximum size is {max_size / (1024 * 1024):.1f}MB."
            )

        return v

class CorsConfigResponse(BaseModel):
    """Response model for CORS configuration."""

    allow_origins: list[str]
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool
    expose_headers: list[str]
    max_age: int

class CorsTestParams(BaseModel):
    """Parameters for testing a CORS policy."""

    origin: str
    method: str
    path: str

class CorsTestResponse(BaseModel):
    """Response model for a CORS policy test."""

    is_allowed: bool
    origin: str
    method: str

class OriginValidationRequest(BaseModel):
    """Request model for validating a list of origins."""

    origins: list[str]

class OriginValidationResponse(BaseModel):
    """Response model for origin validation."""

    validated_origins: dict[str, str]
