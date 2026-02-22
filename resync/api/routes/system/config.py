"""
System Configuration API - Comprehensive settings management.

Provides REST API endpoints for viewing and updating system configuration
through a web interface. Settings are organized into categories:

Categories:
- performance: Cache, memory, workers, timeouts
- tws_monitoring: TWS polling, alerts, patterns
- rate_limiting: API rate limits
- rag: RAG microservice settings
- llm: AI model configuration
- logging: Log levels and formats
- continual_learning: Feedback, active learning settings

Security:
- All endpoints require admin authentication
- Sensitive values (API keys, passwords) are never exposed
- Changes are validated before applying
- Some changes require restart (marked in response)
"""

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.structured_logger import get_logger
from resync.settings import get_settings

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/system-config",
    tags=["System Configuration"],
    dependencies=[Depends(verify_admin_credentials)],
)


# =============================================================================
# PYDANTIC MODELS - Request/Response
# =============================================================================


class ConfigField(BaseModel):
    """Metadata about a configuration field."""

    name: str
    value: Any
    type: str  # string, int, float, bool, list
    description: str
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    options: list[str] | None = None  # For select fields
    requires_restart: bool = False
    category: str
    sensitive: bool = False


class ConfigCategory(BaseModel):
    """A category of configuration fields."""

    name: str
    display_name: str
    description: str
    icon: str  # FontAwesome icon class
    fields: list[ConfigField]


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration values."""

    updates: dict[str, Any] = Field(..., description="Field name -> new value pairs")

    @field_validator("updates")
    @classmethod
    def validate_updates(cls, v):
        if not v:
            raise ValueError("At least one update is required")
        return v


class ConfigUpdateResponse(BaseModel):
    """Response after updating configuration."""

    success: bool
    updated_fields: list[str]
    requires_restart: bool
    errors: dict[str, str] = Field(default_factory=dict)
    message: str


class SystemResourcesResponse(BaseModel):
    """Current system resource usage."""

    memory_used_mb: float
    memory_percent: float
    cpu_percent: float
    disk_used_gb: float
    disk_percent: float
    open_connections: int
    active_workers: int
    uptime_seconds: float


# =============================================================================
# CONFIGURATION DEFINITIONS
# =============================================================================


def get_config_definitions() -> list[ConfigCategory]:
    """
    Define all configurable settings with metadata.

    This is the single source of truth for what can be configured
    via the web interface.
    """
    settings = get_settings()

    return [
        # =====================================================================
        # PERFORMANCE & CACHE
        # =====================================================================
        ConfigCategory(
            name="performance",
            display_name="Performance & Cache",
            description="Memory management, caching, and connection pools",
            icon="fa-tachometer-alt",
            fields=[
                ConfigField(
                    name="robust_cache_max_items",
                    value=settings.robust_cache_max_items,
                    type="int",
                    description="Maximum number of items in memory cache",
                    default=100000,
                    min_value=100,
                    max_value=1000000,
                    requires_restart=False,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="robust_cache_max_memory_mb",
                    value=settings.robust_cache_max_memory_mb,
                    type="int",
                    description="Maximum memory for cache (MB)",
                    default=100,
                    min_value=10,
                    max_value=4096,
                    requires_restart=False,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="robust_cache_eviction_batch_size",
                    value=settings.robust_cache_eviction_batch_size,
                    type="int",
                    description="Items to evict per batch when cache is full",
                    default=100,
                    min_value=1,
                    max_value=1000,
                    requires_restart=False,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="robust_cache_enable_weak_refs",
                    value=settings.robust_cache_enable_weak_refs,
                    type="bool",
                    description="Use weak references for large objects",
                    default=True,
                    requires_restart=False,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="db_pool_min_size",
                    value=settings.db_pool_min_size,
                    type="int",
                    description="Minimum database connections in pool",
                    default=20,
                    min_value=1,
                    max_value=100,
                    requires_restart=True,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="db_pool_max_size",
                    value=settings.db_pool_max_size,
                    type="int",
                    description="Maximum database connections in pool",
                    default=100,
                    min_value=1,
                    max_value=1000,
                    requires_restart=True,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="redis_pool_min_size",
                    value=settings.redis_pool_min_size,
                    type="int",
                    description="Minimum Redis connections in pool",
                    default=5,
                    min_value=1,
                    max_value=100,
                    requires_restart=True,
                    category="performance",
                    sensitive=False,
                ),
                ConfigField(
                    name="redis_pool_max_size",
                    value=settings.redis_pool_max_size,
                    type="int",
                    description="Maximum Redis connections in pool",
                    default=20,
                    min_value=1,
                    max_value=1000,
                    requires_restart=True,
                    category="performance",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # TWS MONITORING
        # =====================================================================
        ConfigCategory(
            name="tws_monitoring",
            display_name="TWS Monitoring",
            description="Polling intervals, alert thresholds, and pattern detection",
            icon="fa-chart-line",
            fields=[
                ConfigField(
                    name="tws_polling_enabled",
                    value=settings.tws_polling_enabled,
                    type="bool",
                    description="Enable automatic TWS polling",
                    default=True,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_polling_interval_seconds",
                    value=settings.tws_polling_interval_seconds,
                    type="int",
                    description="Polling interval (seconds)",
                    default=30,
                    min_value=5,
                    max_value=300,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_polling_mode",
                    value=settings.tws_polling_mode,
                    type="string",
                    description="Polling mode strategy",
                    default="fixed",
                    options=["fixed", "adaptive", "scheduled"],
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_job_stuck_threshold_minutes",
                    value=settings.tws_job_stuck_threshold_minutes,
                    type="int",
                    description="Minutes to consider a job as stuck",
                    default=60,
                    min_value=10,
                    max_value=1440,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_job_late_threshold_minutes",
                    value=settings.tws_job_late_threshold_minutes,
                    type="int",
                    description="Minutes to consider a job as late",
                    default=30,
                    min_value=5,
                    max_value=720,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_anomaly_failure_rate_threshold",
                    value=settings.tws_anomaly_failure_rate_threshold,
                    type="float",
                    description="Failure rate threshold for anomaly alerts (0.0-1.0)",
                    default=0.1,
                    min_value=0.01,
                    max_value=1.0,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_pattern_detection_enabled",
                    value=settings.tws_pattern_detection_enabled,
                    type="bool",
                    description="Enable automatic pattern detection",
                    default=True,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_pattern_detection_interval_minutes",
                    value=settings.tws_pattern_detection_interval_minutes,
                    type="int",
                    description="Pattern detection interval (minutes)",
                    default=60,
                    min_value=15,
                    max_value=1440,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_pattern_min_confidence",
                    value=settings.tws_pattern_min_confidence,
                    type="float",
                    description="Minimum confidence to report a pattern (0.0-1.0)",
                    default=0.5,
                    min_value=0.1,
                    max_value=1.0,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_solution_correlation_enabled",
                    value=settings.tws_solution_correlation_enabled,
                    type="bool",
                    description="Enable solution suggestions based on history",
                    default=True,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_solution_min_success_rate",
                    value=settings.tws_solution_min_success_rate,
                    type="float",
                    description="Minimum success rate to suggest solution (0.0-1.0)",
                    default=0.6,
                    min_value=0.0,
                    max_value=1.0,
                    requires_restart=False,
                    category="tws_monitoring",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # DATA RETENTION
        # =====================================================================
        ConfigCategory(
            name="data_retention",
            display_name="Data Retention",
            description="How long to keep various types of data",
            icon="fa-database",
            fields=[
                ConfigField(
                    name="tws_retention_days_full",
                    value=settings.tws_retention_days_full,
                    type="int",
                    description="Days to keep full job data",
                    default=7,
                    min_value=1,
                    max_value=30,
                    requires_restart=False,
                    category="data_retention",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_retention_days_summary",
                    value=settings.tws_retention_days_summary,
                    type="int",
                    description="Days to keep summaries and events",
                    default=30,
                    min_value=7,
                    max_value=90,
                    requires_restart=False,
                    category="data_retention",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_retention_days_patterns",
                    value=settings.tws_retention_days_patterns,
                    type="int",
                    description="Days to keep detected patterns",
                    default=90,
                    min_value=30,
                    max_value=365,
                    requires_restart=False,
                    category="data_retention",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # RATE LIMITING
        # =====================================================================
        ConfigCategory(
            name="rate_limiting",
            display_name="Rate Limiting",
            description="API request limits per minute",
            icon="fa-shield-alt",
            fields=[
                ConfigField(
                    name="rate_limit_public_per_minute",
                    value=settings.rate_limit_public_per_minute,
                    type="int",
                    description="Public endpoints (unauthenticated)",
                    default=100,
                    min_value=1,
                    max_value=10000,
                    requires_restart=False,
                    category="rate_limiting",
                    sensitive=False,
                ),
                ConfigField(
                    name="rate_limit_authenticated_per_minute",
                    value=settings.rate_limit_authenticated_per_minute,
                    type="int",
                    description="Authenticated endpoints",
                    default=1000,
                    min_value=1,
                    max_value=100000,
                    requires_restart=False,
                    category="rate_limiting",
                    sensitive=False,
                ),
                ConfigField(
                    name="rate_limit_critical_per_minute",
                    value=settings.rate_limit_critical_per_minute,
                    type="int",
                    description="Critical/admin endpoints",
                    default=50,
                    min_value=1,
                    max_value=1000,
                    requires_restart=False,
                    category="rate_limiting",
                    sensitive=False,
                ),
                ConfigField(
                    name="rate_limit_websocket_per_minute",
                    value=settings.rate_limit_websocket_per_minute,
                    type="int",
                    description="WebSocket connections",
                    default=30,
                    min_value=1,
                    max_value=1000,
                    requires_restart=False,
                    category="rate_limiting",
                    sensitive=False,
                ),
                ConfigField(
                    name="rate_limit_sliding_window",
                    value=settings.rate_limit_sliding_window,
                    type="bool",
                    description="Use sliding window (more accurate, higher overhead)",
                    default=True,
                    requires_restart=False,
                    category="rate_limiting",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # RAG SERVICE
        # =====================================================================
        ConfigCategory(
            name="rag_service",
            display_name="RAG Service",
            description="Retrieval-Augmented Generation microservice settings",
            icon="fa-brain",
            fields=[
                ConfigField(
                    name="rag_service_url",
                    value=settings.rag_service_url,
                    type="string",
                    description="RAG microservice URL",
                    default="http://localhost:8003",
                    requires_restart=False,
                    category="rag_service",
                    sensitive=False,
                ),
                ConfigField(
                    name="rag_service_timeout",
                    value=settings.rag_service_timeout,
                    type="int",
                    description="Request timeout (seconds)",
                    default=300,
                    min_value=10,
                    max_value=600,
                    requires_restart=False,
                    category="rag_service",
                    sensitive=False,
                ),
                ConfigField(
                    name="rag_service_max_retries",
                    value=settings.rag_service_max_retries,
                    type="int",
                    description="Maximum retry attempts",
                    default=3,
                    min_value=0,
                    max_value=10,
                    requires_restart=False,
                    category="rag_service",
                    sensitive=False,
                ),
                ConfigField(
                    name="rag_service_retry_backoff",
                    value=settings.rag_service_retry_backoff,
                    type="float",
                    description="Exponential backoff factor",
                    default=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    requires_restart=False,
                    category="rag_service",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # LITELLM CONFIGURATION
        # =====================================================================
        ConfigCategory(
            name="litellm",
            display_name="LiteLLM & AI Models",
            description="Multi-provider LLM configuration, model routing, and cost management",
            icon="fa-brain",
            fields=[
                # Basic LLM Settings
                ConfigField(
                    name="llm_endpoint",
                    value=settings.llm_endpoint,
                    type="string",
                    description="Primary LLM API endpoint URL",
                    default="https://integrate.api.nvidia.com/v1",
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                ConfigField(
                    name="llm_timeout",
                    value=settings.llm_timeout,
                    type="float",
                    description="LLM request timeout (seconds)",
                    default=60.0,
                    min_value=10.0,
                    max_value=300.0,
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                # Model Selection
                ConfigField(
                    name="auditor_model_name",
                    value=settings.auditor_model_name,
                    type="string",
                    description="Model for simple/auditor tasks (lower cost)",
                    default="gpt-3.5-turbo",
                    options=[
                        "gpt-3.5-turbo",
                        "gpt-4",
                        "gpt-4-turbo",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "claude-3-haiku-20240307",
                        "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229",
                        "claude-3-5-sonnet-20240620",
                        "ollama/llama3",
                        "ollama/mistral",
                        "ollama/codellama",
                        "together_ai/llama-3-70b",
                        "together_ai/mixtral-8x7b",
                    ],
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                ConfigField(
                    name="agent_model_name",
                    value=settings.agent_model_name,
                    type="string",
                    description="Model for complex/agent tasks (higher capability)",
                    default="gpt-4o",
                    options=[
                        "gpt-3.5-turbo",
                        "gpt-4",
                        "gpt-4-turbo",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "claude-3-haiku-20240307",
                        "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229",
                        "claude-3-5-sonnet-20240620",
                        "ollama/llama3",
                        "ollama/mistral",
                        "ollama/codellama",
                        "together_ai/llama-3-70b",
                        "together_ai/mixtral-8x7b",
                    ],
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                # LiteLLM Router Settings
                ConfigField(
                    name="LITELLM_PRE_CALL_CHECKS",
                    value=getattr(settings, "LITELLM_PRE_CALL_CHECKS", True),
                    type="bool",
                    description="Enable pre-call validation checks",
                    default=True,
                    requires_restart=True,
                    category="litellm",
                    sensitive=False,
                ),
                ConfigField(
                    name="LITELLM_NUM_RETRIES",
                    value=getattr(settings, "LITELLM_NUM_RETRIES", 3),
                    type="int",
                    description="Number of retry attempts on failure",
                    default=3,
                    min_value=0,
                    max_value=10,
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                ConfigField(
                    name="LITELLM_TIMEOUT",
                    value=getattr(settings, "LITELLM_TIMEOUT", 120),
                    type="int",
                    description="LiteLLM router timeout (seconds)",
                    default=120,
                    min_value=10,
                    max_value=600,
                    requires_restart=False,
                    category="litellm",
                    sensitive=False,
                ),
                ConfigField(
                    name="LITELLM_STRICT_INIT",
                    value=getattr(settings, "LITELLM_STRICT_INIT", False),
                    type="bool",
                    description="Fail startup if LiteLLM init fails",
                    default=False,
                    requires_restart=True,
                    category="litellm",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # LLM COST & BUDGET
        # =====================================================================
        ConfigCategory(
            name="llm_budget",
            display_name="LLM Cost & Budget",
            description="Cost monitoring, budget limits, and spending alerts",
            icon="fa-dollar-sign",
            fields=[
                ConfigField(
                    name="LLM_BUDGET_DAILY_USD",
                    value=getattr(settings, "LLM_BUDGET_DAILY_USD", 500.0),
                    type="float",
                    description="Daily budget limit (USD)",
                    default=500.0,
                    min_value=0.0,
                    max_value=10000.0,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_BUDGET_MONTHLY_USD",
                    value=getattr(settings, "LLM_BUDGET_MONTHLY_USD", 5000.0),
                    type="float",
                    description="Monthly budget limit (USD)",
                    default=5000.0,
                    min_value=0.0,
                    max_value=100000.0,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_BUDGET_ALERT_THRESHOLD",
                    value=getattr(settings, "LLM_BUDGET_ALERT_THRESHOLD", 0.8),
                    type="float",
                    description="Alert when budget reaches this % (0.0-1.0)",
                    default=0.8,
                    min_value=0.1,
                    max_value=1.0,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_COST_TRACKING_ENABLED",
                    value=getattr(settings, "LLM_COST_TRACKING_ENABLED", True),
                    type="bool",
                    description="Enable cost tracking and analytics",
                    default=True,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_USE_CACHE",
                    value=getattr(settings, "LLM_USE_CACHE", True),
                    type="bool",
                    description="Cache LLM responses to reduce costs",
                    default=True,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_CACHE_TTL_SECONDS",
                    value=getattr(settings, "LLM_CACHE_TTL_SECONDS", 3600),
                    type="int",
                    description="Cache TTL for LLM responses (seconds)",
                    default=3600,
                    min_value=60,
                    max_value=86400,
                    requires_restart=False,
                    category="llm_budget",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # MODEL ROUTING
        # =====================================================================
        ConfigCategory(
            name="model_routing",
            display_name="Smart Model Routing",
            description="Automatic model selection based on query complexity",
            icon="fa-route",
            fields=[
                ConfigField(
                    name="LLM_ROUTING_ENABLED",
                    value=getattr(settings, "LLM_ROUTING_ENABLED", True),
                    type="bool",
                    description="Enable automatic model routing",
                    default=True,
                    requires_restart=False,
                    category="model_routing",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_ROUTING_SIMPLE_MODEL",
                    value=getattr(
                        settings, "LLM_ROUTING_SIMPLE_MODEL", "gpt-3.5-turbo"
                    ),
                    type="string",
                    description="Model for simple queries (status checks, basic info)",
                    default="gpt-3.5-turbo",
                    options=[
                        "gpt-3.5-turbo",
                        "gpt-4o-mini",
                        "claude-3-haiku-20240307",
                        "ollama/llama3",
                        "ollama/mistral",
                    ],
                    requires_restart=False,
                    category="model_routing",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_ROUTING_COMPLEX_MODEL",
                    value=getattr(settings, "LLM_ROUTING_COMPLEX_MODEL", "gpt-4o"),
                    type="string",
                    description="Model for complex queries (analysis, troubleshooting)",
                    default="gpt-4o",
                    options=[
                        "gpt-4",
                        "gpt-4-turbo",
                        "gpt-4o",
                        "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229",
                        "together_ai/llama-3-70b",
                    ],
                    requires_restart=False,
                    category="model_routing",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_ROUTING_FALLBACK_MODEL",
                    value=getattr(
                        settings, "LLM_ROUTING_FALLBACK_MODEL", "gpt-3.5-turbo"
                    ),
                    type="string",
                    description="Fallback model when primary is unavailable",
                    default="gpt-3.5-turbo",
                    options=[
                        "gpt-3.5-turbo",
                        "gpt-4o-mini",
                        "claude-3-haiku-20240307",
                        "ollama/llama3",
                        "ollama/mistral",
                    ],
                    requires_restart=False,
                    category="model_routing",
                    sensitive=False,
                ),
                ConfigField(
                    name="LLM_PREFER_LOCAL_MODELS",
                    value=getattr(settings, "LLM_PREFER_LOCAL_MODELS", False),
                    type="bool",
                    description="Prefer local Ollama models when available",
                    default=False,
                    requires_restart=False,
                    category="model_routing",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # LOGGING
        # =====================================================================
        ConfigCategory(
            name="logging",
            display_name="Logging",
            description="Log levels and formats",
            icon="fa-file-alt",
            fields=[
                ConfigField(
                    name="log_level",
                    value=settings.log_level,
                    type="string",
                    description="Logging level",
                    default="INFO",
                    options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    requires_restart=False,
                    category="logging",
                    sensitive=False,
                ),
                ConfigField(
                    name="log_format",
                    value=settings.log_format,
                    type="string",
                    description="Log output format",
                    default="json",
                    options=["text", "json"],
                    requires_restart=True,
                    category="logging",
                    sensitive=False,
                ),
                ConfigField(
                    name="log_sensitive_data_redaction",
                    value=settings.log_sensitive_data_redaction,
                    type="bool",
                    description="Redact sensitive data in logs",
                    default=True,
                    requires_restart=False,
                    category="logging",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
        # NOTIFICATIONS
        # =====================================================================
        ConfigCategory(
            name="notifications",
            display_name="Notifications",
            description="Browser and Teams notification settings",
            icon="fa-bell",
            fields=[
                ConfigField(
                    name="tws_browser_notifications_enabled",
                    value=settings.tws_browser_notifications_enabled,
                    type="bool",
                    description="Enable browser notifications",
                    default=True,
                    requires_restart=False,
                    category="notifications",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_teams_notifications_enabled",
                    value=settings.tws_teams_notifications_enabled,
                    type="bool",
                    description="Enable Microsoft Teams notifications",
                    default=False,
                    requires_restart=False,
                    category="notifications",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_dashboard_refresh_seconds",
                    value=settings.tws_dashboard_refresh_seconds,
                    type="int",
                    description="Dashboard auto-refresh interval (seconds)",
                    default=5,
                    min_value=1,
                    max_value=60,
                    requires_restart=False,
                    category="notifications",
                    sensitive=False,
                ),
                ConfigField(
                    name="tws_dashboard_theme",
                    value=settings.tws_dashboard_theme,
                    type="string",
                    description="Dashboard theme",
                    default="auto",
                    options=["auto", "light", "dark"],
                    requires_restart=False,
                    category="notifications",
                    sensitive=False,
                ),
            ],
        ),
        # =====================================================================
    ]

    # =============================================================================
    # API ENDPOINTS
    # =============================================================================


@router.get("/categories", response_model=list[ConfigCategory])
async def get_config_categories():
    """
    Get all configuration categories with their fields.

    Returns the complete configuration schema organized by category,
    including current values, types, constraints, and metadata.
    """
    try:
        return get_config_definitions()
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("get_config_categories_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get configuration. Check server logs for details.",
        ) from e


@router.get("/category/{category_name}", response_model=ConfigCategory)
async def get_config_category(category_name: str):
    """
    Get a specific configuration category.
    """
    categories = get_config_definitions()

    for category in categories:
        if category.name == category_name:
            return category

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Category '{category_name}' not found",
    )


@router.post("/update", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    """
    Update configuration values.

    Validates all updates before applying. Returns which fields were
    updated and whether a restart is required.
    """
    try:
        settings = get_settings()
        categories = get_config_definitions()

        # Build field lookup
        field_lookup: dict[str, ConfigField] = {}
        for category in categories:
            for field in category.fields:
                field_lookup[field.name] = field

        updated_fields = []
        errors = {}
        requires_restart = False

        for field_name, new_value in request.updates.items():
            # Check if field exists and is configurable
            if field_name not in field_lookup:
                errors[field_name] = "Unknown or non-configurable field"
                continue

            field_def = field_lookup[field_name]

            # Check if sensitive
            if field_def.sensitive:
                errors[field_name] = "Cannot update sensitive fields via API"
                continue

            # Validate type and constraints
            try:
                validated_value = validate_field_value(field_def, new_value)
            except ValueError as e:
                errors[field_name] = str(e)
                continue

            # Apply update (to environment variable for persistence)
            env_name = f"APP_{field_name.upper()}"
            os.environ[env_name] = str(validated_value)

            # Update in-memory settings if possible
            try:
                setattr(settings, field_name, validated_value)
            except Exception as exc:
                logger.debug(
                    "suppressed_exception", error=str(exc), exc_info=True
                )  # was: pass

            updated_fields.append(field_name)

            if field_def.requires_restart:
                requires_restart = True

        # Determine overall success
        success = len(updated_fields) > 0 and len(errors) == 0

        if updated_fields:
            logger.info(
                "config_updated",
                updated_fields=updated_fields,
                requires_restart=requires_restart,
            )

        message = f"Updated {len(updated_fields)} field(s)"
        if requires_restart:
            message += ". Restart required for some changes to take effect."
        if errors:
            message += f" {len(errors)} field(s) had errors."

        return ConfigUpdateResponse(
            success=success,
            updated_fields=updated_fields,
            requires_restart=requires_restart,
            errors=errors,
            message=message,
        )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("update_config_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update configuration. Check server logs for details.",
        ) from e


@router.get("/resources", response_model=SystemResourcesResponse)
async def get_system_resources():
    """
    Get current system resource usage.
    """
    try:
        import psutil

        process = psutil.Process()
        disk = psutil.disk_usage("/")

        # Get worker count (approximate)
        parent = psutil.Process(os.getppid())
        try:
            workers = len(
                [c for c in parent.children() if "python" in c.name().lower()]
            )
        except Exception:
            workers = 1

        return SystemResourcesResponse(
            memory_used_mb=round(process.memory_info().rss / 1024 / 1024, 1),
            memory_percent=round(process.memory_percent(), 1),
            cpu_percent=round(process.cpu_percent(interval=0.1), 1),
            disk_used_gb=round(disk.used / 1024 / 1024 / 1024, 1),
            disk_percent=round(disk.percent, 1),
            open_connections=len(process.connections()),
            active_workers=workers,
            uptime_seconds=round(
                (
                    datetime.now(timezone.utc)
                    - datetime.fromtimestamp(process.create_time())
                ).total_seconds()
            ),
        )

    except ImportError:
        # psutil not available
        return SystemResourcesResponse(
            memory_used_mb=0,
            memory_percent=0,
            cpu_percent=0,
            disk_used_gb=0,
            disk_percent=0,
            open_connections=0,
            active_workers=1,
            uptime_seconds=0,
        )
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("get_resources_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get resources. Check server logs for details.",
        ) from e


@router.post("/gc")
async def trigger_garbage_collection():
    """
    Trigger garbage collection to free memory.
    """
    import gc

    before = 0
    try:
        import psutil

        before = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    collected = gc.collect()

    after = 0
    try:
        import psutil

        after = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    freed = before - after

    logger.info(
        "gc_triggered", objects_collected=collected, memory_freed_mb=round(freed, 1)
    )

    return {
        "success": True,
        "objects_collected": collected,
        "memory_before_mb": round(before, 1),
        "memory_after_mb": round(after, 1),
        "memory_freed_mb": round(freed, 1),
    }


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all caches.
    """
    try:
        # Clear settings cache
        from resync.settings import clear_settings_cache

        clear_settings_cache()

        # Clear metrics store cache if available
        try:
            from resync.core.metrics import get_metrics_store

            store = get_metrics_store()
            # Don't actually clear metrics, just flush
            await store._flush_buffer()
        except ImportError:
            pass

        logger.info("cache_cleared")

        return {"success": True, "message": "Caches cleared successfully"}

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("clear_cache_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache. Check server logs for details.",
        ) from e


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_field_value(field: ConfigField, value: Any) -> Any:
    """
    Validate and convert a field value.

    Args:
        field: Field definition with constraints
        value: Value to validate

    Returns:
        Validated and converted value

    Raises:
        ValueError: If validation fails
    """
    # Type conversion
    if field.type == "int":
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError("Must be an integer") from None

    elif field.type == "float":
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("Must be a number") from None

    elif field.type == "bool":
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes", "on")
        else:
            value = bool(value)

    elif field.type == "string":
        value = str(value)

    # Range validation for numbers
    if field.type in ("int", "float"):
        if field.min_value is not None and value < field.min_value:
            raise ValueError(f"Must be at least {field.min_value}")
        if field.max_value is not None and value > field.max_value:
            raise ValueError(f"Must be at most {field.max_value}")

    # Options validation
    if field.options and value not in field.options:
        raise ValueError(f"Must be one of: {', '.join(field.options)}")

    return value
