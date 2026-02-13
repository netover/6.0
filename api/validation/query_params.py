"""Query parameter validation models for API endpoints."""

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator
from pydantic import StringConstraints as PydanticStringConstraints

from .common import BaseValidatedModel, NumericConstraints, StringConstraints, ValidationPatterns


class SortOrder(str, Enum):
    """Valid sort orders."""

    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    """Valid filter operators."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    NOT_LIKE = "not_like"


class PaginationParams(BaseValidatedModel):
    """Pagination query parameters."""

    page: int = Field(
        default=1,
        ge=NumericConstraints.MIN_PAGE,
        le=NumericConstraints.MAX_PAGE,
        description="Page number (1-based)",
    )

    page_size: int = Field(
        default=10,
        ge=NumericConstraints.MIN_PAGE_SIZE,
        le=NumericConstraints.MAX_PAGE_SIZE,
        description="Number of items per page",
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, v):
        """Validate page size is reasonable."""
        if v > 100 and v <= NumericConstraints.MAX_PAGE_SIZE:
            # Allow large page sizes but log a warning
            import logging

            logging.warning(f"Large page size requested: {v}")
        return v

    def get_offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size

    def get_limit(self) -> int:
        """Get limit for database queries."""
        return self.page_size


class SearchParams(BaseValidatedModel):
    """Search query parameters."""

    query: Annotated[
        str, PydanticStringConstraints(min_length=1, max_length=200, strip_whitespace=True)
    ] = Field(..., description="Search query string")

    search_fields: list[str] | None = Field(
        None, description="Specific fields to search in", max_length=10
    )

    fuzzy: bool = Field(default=False, description="Enable fuzzy search")

    case_sensitive: bool = Field(default=False, description="Case-sensitive search")

    whole_words: bool = Field(default=False, description="Match whole words only")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("query")
    @classmethod
    def validate_search_query(cls, v):
        """Validate search query for malicious content."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        # Check for script injection
        if ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Search query contains potentially malicious content")
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(\b(or|and)\b.*=.*)",
            r"('|\")(.*)(or|and)(.*)('|\")",
            r"(;|--|/\*|\*/|xp_)",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Search query contains invalid patterns")
        return v

    @field_validator("search_fields")
    @classmethod
    def validate_search_fields(cls, v):
        """Validate search fields."""
        if not v:
            return v
        # Validate field names
        for field in v:
            if not field.replace("_", "").replace(".", "").isalnum():
                raise ValueError(f"Invalid search field: {field}")
        # Check for duplicate fields
        if len(v) != len(set(v)):
            raise ValueError("Duplicate search fields found")
        return v


class FilterParams(BaseValidatedModel):
    """Filter query parameters."""

    filters: list[dict[str, Any]] | None = Field(
        default_factory=list, description="List of filter conditions", max_length=20
    )

    filter_logic: str = Field(
        default="and", pattern=r"^(and|or)$", description="Logic to combine filters"
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v):
        """Validate filter conditions."""
        if not v:
            return v
        for i, filter_condition in enumerate(v):
            # Validate filter structure
            required_keys = {"field", "operator", "value"}
            if not all(key in filter_condition for key in required_keys):
                raise ValueError(f"Filter at index {i} missing required keys: {required_keys}")
            # Validate field name
            field = filter_condition["field"]
            if not field.replace("_", "").replace(".", "").isalnum():
                raise ValueError(f"Invalid filter field: {field}")
            # Validate operator
            operator = filter_condition["operator"]
            valid_operators = {op.value for op in FilterOperator}
            if operator not in valid_operators:
                raise ValueError(f"Invalid filter operator: {operator}")
            # Validate value based on operator
            value = filter_condition["value"]
            if operator in {"in", "not_in"} and not isinstance(value, list):
                raise ValueError(f"Filter operator '{operator}' requires a list value")
            # Check for malicious content in string values
            if isinstance(value, str) and ValidationPatterns.SCRIPT_PATTERN.search(value):
                raise ValueError(f"Filter value contains malicious content: {value}")
            # Check for SQL injection in field names and values
            if isinstance(field, str):
                sql_patterns = [
                    r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                    r"(;|--|/\*|\*/)",
                ]
                for pattern in sql_patterns:
                    if re.search(pattern, field, re.IGNORECASE):
                        raise ValueError(f"Filter field contains invalid patterns: {field}")
        return v


class SortParams(BaseValidatedModel):
    """Sorting query parameters."""

    sort_by: list[str] | None = Field(
        default_factory=list, description="Fields to sort by", max_length=5
    )

    sort_order: list[SortOrder] | None = Field(
        default_factory=list, description="Sort order for each field", max_length=5
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("sort_by")
    @classmethod
    def validate_sort_fields(cls, v):
        """Validate sort fields."""
        if not v:
            return v
        # Validate field names
        for field in v:
            if not field.replace("_", "").replace(".", "").isalnum():
                raise ValueError(f"Invalid sort field: {field}")
        # Check for duplicate fields
        if len(v) != len(set(v)):
            raise ValueError("Duplicate sort fields found")
        return v

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v, info):
        """Validate sort order matches sort fields."""
        sort_by = info.data.get("sort_by", [])
        if v and sort_by and len(v) != len(sort_by):
            raise ValueError("Number of sort orders must match number of sort fields")
        return v


class DateRangeParams(BaseValidatedModel):
    """Date range query parameters."""

    start_date: datetime | None = Field(None, description="Start date (ISO 8601 format)")

    end_date: datetime | None = Field(None, description="End date (ISO 8601 format)")

    date_field: str = Field(
        default="created_at",
        pattern=r"^(created_at|updated_at|timestamp|date)$",
        description="Date field to filter on",
    )

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


class AgentQueryParams(BaseValidatedModel):
    """Agent-specific query parameters."""

    agent_id: StringConstraints.AGENT_ID | None = Field(None, description="Filter by agent ID")

    name: Annotated[str, PydanticStringConstraints(min_length=1, max_length=100)] | None = Field(
        None, description="Filter by agent name (partial match)"
    )

    type: str | None = Field(None, description="Filter by agent type")

    status: str | None = Field(None, description="Filter by agent status")

    tools: list[str] | None = Field(None, description="Filter by tools", max_length=10)

    model_name: StringConstraints.MODEL_NAME | None = Field(
        None, description="Filter by model name"
    )

    memory_enabled: bool | None = Field(None, description="Filter by memory setting")

    include_inactive: bool = Field(default=False, description="Include inactive agents")

    tags: list[Annotated[str, PydanticStringConstraints(min_length=1, max_length=50)]] | None = (
        Field(None, description="Filter by tags", max_length=5)
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("name", "type", "status", "model_name")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields for malicious content."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Field contains potentially malicious content")
        return v

    @field_validator("tools", "tags")
    @classmethod
    def validate_list_fields(cls, v):
        """Validate list fields."""
        if not v:
            return v
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate values found in list")
        # Validate individual items
        for item in v:
            if ValidationPatterns.SCRIPT_PATTERN.search(item):
                raise ValueError(f"List item contains malicious content: {item}")
        return v


class SystemQueryParams(BaseValidatedModel):
    """System monitoring query parameters."""

    metric_types: list[str] | None = Field(
        None, description="Types of metrics to retrieve", max_length=10
    )

    time_range: str | None = Field(
        None, pattern=r"^(1h|6h|24h|7d|30d|90d)$", description="Time range for metrics"
    )

    aggregation: str = Field(
        default="avg",
        pattern=r"^(avg|min|max|sum|count)$",
        description="Aggregation method",
    )

    include_alerts: bool = Field(default=True, description="Include system alerts")

    severity_filter: list[str] | None = Field(
        None, description="Filter by alert severity", max_length=3
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("metric_types", "severity_filter")
    @classmethod
    def validate_list_fields(cls, v):
        """Validate list fields."""
        if not v:
            return v
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate values found in list")
        return v


class AuditQueryParams(BaseValidatedModel):
    """Audit query parameters."""

    status: str = Field(
        default="pending",
        pattern=r"^(pending|approved|rejected|all)$",
        description="Audit status filter",
    )

    query: (
        Annotated[
            str, PydanticStringConstraints(min_length=1, max_length=200, strip_whitespace=True)
        ]
        | None
    ) = Field(None, description="Search query")

    user_id: StringConstraints.SAFE_TEXT | None = Field(None, description="Filter by user ID")

    agent_id: StringConstraints.AGENT_ID | None = Field(None, description="Filter by agent ID")

    severity: list[str] | None = Field(None, description="Filter by severity levels", max_length=3)

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("query")
    @classmethod
    def validate_search_query(cls, v):
        """Validate search query."""
        if v and ValidationPatterns.SCRIPT_PATTERN.search(v):
            raise ValueError("Search query contains potentially malicious content")
        return v


class FileQueryParams(BaseValidatedModel):
    """File-related query parameters."""

    file_types: list[str] | None = Field(None, description="Filter by file types", max_length=10)

    size_min: int | None = Field(None, ge=0, description="Minimum file size in bytes")

    size_max: int | None = Field(None, ge=0, description="Maximum file size in bytes")

    uploaded_by: StringConstraints.SAFE_TEXT | None = Field(None, description="Filter by uploader")

    status: str | None = Field(None, description="Filter by file processing status")

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("size_max")
    @classmethod
    def validate_size_range(cls, v, info):
        """Validate file size range."""
        size_min = info.data.get("size_min")
        if size_min and v and v < size_min:
            raise ValueError("Maximum size must be greater than minimum size")
        return v

    @field_validator("file_types", "status")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields."""
        if v:
            if isinstance(v, str) and ValidationPatterns.SCRIPT_PATTERN.search(v):
                raise ValueError("Field contains potentially malicious content")
            if isinstance(v, list):
                for item in v:
                    if ValidationPatterns.SCRIPT_PATTERN.search(item):
                        raise ValueError(f"List item contains malicious content: {item}")
        return v


# Combined query parameters for complex endpoints
class CombinedQueryParams(PaginationParams, SearchParams, SortParams, DateRangeParams):
    """Combined query parameters for endpoints that support multiple filtering options."""

    model_config = ConfigDict(
        extra="forbid",
    )


# Export individual parameter types for flexible usage
__all__ = [
    "PaginationParams",
    "SearchParams",
    "FilterParams",
    "SortParams",
    "DateRangeParams",
    "AgentQueryParams",
    "SystemQueryParams",
    "AuditQueryParams",
    "FileQueryParams",
    "CombinedQueryParams",
    "SortOrder",
    "FilterOperator",
]
