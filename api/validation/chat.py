"""Chat and WebSocket message validation models."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator
from pydantic import StringConstraints as PydanticStringConstraints

from .common import BaseValidatedModel, NumericConstraints, StringConstraints, ValidationPatterns


class MessageType(str, Enum):
    """Valid message types."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    SYSTEM = "system"
    ERROR = "error"
    STREAM = "stream"
    INFO = "info"


class MessageStatus(str, Enum):
    """Message status values."""

    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    PENDING = "pending"


class ChatMessage(BaseValidatedModel):
    """Chat message validation model."""

    content: Annotated[
        str,
        PydanticStringConstraints(
            min_length=NumericConstraints.MIN_MESSAGE_LENGTH,
            max_length=NumericConstraints.MAX_MESSAGE_LENGTH,
            strip_whitespace=True,
        ),
    ] = Field(..., description="Message content", examples=["Hello, how can I help you today?"])

    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")

    sender: StringConstraints.SAFE_TEXT = Field(
        ..., description="Message sender identifier", examples=["user123"]
    )

    recipient: StringConstraints.SAFE_TEXT | None = Field(
        None, description="Message recipient identifier", examples=["agent_tws_specialist"]
    )

    session_id: StringConstraints.AGENT_ID | None = Field(
        None, description="Chat session identifier"
    )

    parent_message_id: str | None = Field(
        None, description="ID of the parent message (for threading)"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata", max_length=20
    )

    priority: int = Field(default=0, ge=0, le=10, description="Message priority (0-10)")

    status: MessageStatus = Field(default=MessageStatus.PENDING, description="Message status")

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @field_validator("content")
    @classmethod
    def validate_message_content(cls, v):
        """Validate message content for malicious patterns."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        # Check for script injection
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Message contains potentially malicious content")
        # Check for command injection
        if ValidationPatterns.COMMAND_INJECTION_PATTERN.search(v):
            raise ValueError("Message contains invalid characters")
        # Check for excessive length
        if len(v) > NumericConstraints.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message content exceeds maximum length of "
                f"{NumericConstraints.MAX_MESSAGE_LENGTH} characters"
            )
        return v

    @field_validator("sender", "recipient")
    @classmethod
    def validate_user_identifiers(cls, v):
        """Validate user identifiers."""
        if v and not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("User identifier contains invalid characters")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate message metadata."""
        if not v:
            return v
        # Check metadata size
        if len(str(v)) > 1000:  # Limit metadata size
            raise ValueError("Metadata is too large")
        # Sanitize metadata values
        sanitized_metadata = {}
        for key, value in v.items():
            if not key.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid metadata key: {key}")
            if isinstance(value, str):
                if ValidationPatterns.SCRIPT_PATTERN.search(value):
                    raise ValueError(f"Metadata value for '{key}' contains malicious content")
                sanitized_metadata[key] = value.strip()
            else:
                sanitized_metadata[key] = value
        return sanitized_metadata


class WebSocketMessage(BaseValidatedModel):
    """WebSocket message validation model."""

    type: str = Field(
        ...,
        pattern=r"^(message|stream|error|info|system)$",
        description="WebSocket message type",
    )

    sender: StringConstraints.SAFE_TEXT = Field(..., description="Message sender")

    message: (
        Annotated[
            str,
            PydanticStringConstraints(
                min_length=1,
                max_length=NumericConstraints.MAX_MESSAGE_LENGTH,
                strip_whitespace=True,
            ),
        ]
        | None
    ) = Field(None, description="Message content")

    agent_id: StringConstraints.AGENT_ID = Field(..., description="Target agent ID")

    session_id: StringConstraints.AGENT_ID | None = Field(None, description="WebSocket session ID")

    correlation_id: str | None = Field(None, description="Correlation ID for request tracking")

    is_final: bool = Field(
        default=False, description="Whether this is the final message in a stream"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata", max_length=10
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @field_validator("message")
    @classmethod
    def validate_message_content(cls, v, info):
        """Validate message content based on type."""
        if v is None:
            return v
        message_type = info.data.get("type")
        # Validate content based on message type
        if message_type == "error" and not v:
            raise ValueError("Error messages must have content")
        if message_type == "message" and not v:
            raise ValueError("Regular messages must have content")
        # Check for malicious content
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Message contains potentially malicious content")
        if ValidationPatterns.COMMAND_INJECTION_PATTERN.search(v):
            raise ValueError("Message contains invalid characters")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_websocket_metadata(cls, v):
        """Validate WebSocket message metadata."""
        if not v:
            return v
        # Limit metadata size for WebSocket messages
        if len(str(v)) > 500:
            raise ValueError("Metadata is too large for WebSocket message")
        return v


class ChatSession(BaseValidatedModel):
    """Chat session validation model."""

    session_id: StringConstraints.AGENT_ID = Field(..., description="Unique session identifier")

    user_id: StringConstraints.SAFE_TEXT = Field(..., description="User ID who owns the session")

    agent_id: StringConstraints.AGENT_ID = Field(
        ..., description="Agent ID participating in the session"
    )

    status: str = Field(
        default="active",
        pattern=r"^(active|paused|ended|expired)$",
        description="Session status",
    )

    context: dict[str, Any] = Field(
        default_factory=dict, description="Session context", max_length=10
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session metadata", max_length=10
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Session creation timestamp"
    )

    updated_at: datetime | None = Field(None, description="Last update timestamp")

    expires_at: datetime | None = Field(None, description="Session expiration timestamp")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class ChatHistoryRequest(BaseValidatedModel):
    """Chat history request validation model."""

    session_id: StringConstraints.AGENT_ID | None = Field(None, description="Filter by session ID")

    user_id: StringConstraints.SAFE_TEXT | None = Field(None, description="Filter by user ID")

    agent_id: StringConstraints.AGENT_ID | None = Field(None, description="Filter by agent ID")

    start_date: datetime | None = Field(None, description="Start date for history range")

    end_date: datetime | None = Field(None, description="End date for history range")

    message_types: list[MessageType] | None = Field(
        None, description="Filter by message types", max_length=5
    )

    search_query: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=100, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Search query for message content")

    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of messages to return")

    offset: int = Field(default=0, ge=0, le=10000, description="Number of messages to skip")

    sort_order: str = Field(
        default="desc", pattern=r"^(asc|desc)$", description="Sort order for messages"
    )

    include_metadata: bool = Field(default=False, description="Whether to include message metadata")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v, info):
        """Validate date range."""
        start_date = info.data.get("start_date")
        if start_date and v and v < start_date:
            raise ValueError("End date must be after start date")
        return v

    @field_validator("search_query")
    @classmethod
    def validate_search_query(cls, v):
        """Validate search query."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Search query contains potentially malicious content")
        return v


class MessageReaction(BaseValidatedModel):
    """Message reaction validation model."""

    message_id: str = Field(..., description="ID of the message being reacted to")

    reaction: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=10, strip_whitespace=True)
    ] = Field(..., description="Reaction emoji or text")

    user_id: StringConstraints.SAFE_TEXT = Field(..., description="User ID who added the reaction")

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Reaction timestamp")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("reaction")
    @classmethod
    def validate_reaction(cls, v):
        """Validate reaction format."""
        # Allow common emojis and simple text reactions
        allowed_reactions = {"👍", "👎", "❤️", "😊", "😢", "😡", "⭐", "🔥", "💯"}
        if v in allowed_reactions:
            return v
        # Allow simple text reactions (1-3 characters)
        if len(v) <= 3 and v.isalnum():
            return v
        raise ValueError("Invalid reaction format")


class ChatExportRequest(BaseValidatedModel):
    """Chat export request validation model."""

    session_id: StringConstraints.AGENT_ID = Field(..., description="Session ID to export")

    format: str = Field(
        default="json", pattern=r"^(json|csv|txt|pdf)$", description="Export format"
    )

    include_metadata: bool = Field(default=True, description="Whether to include message metadata")

    date_range: dict[str, datetime] | None = Field(None, description="Date range for export")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("date_range")
    @classmethod
    def validate_date_range(cls, v):
        """Validate date range."""
        if not v:
            return v
        if "start" not in v or "end" not in v:
            raise ValueError("Date range must include both 'start' and 'end' dates")
        if v["start"] >= v["end"]:
            raise ValueError("Start date must be before end date")
        return v
