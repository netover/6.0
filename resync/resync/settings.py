"""Application settings and configuration management.

This module defines all application settings using Pydantic BaseSettings,
providing centralized configuration management with environment variable
support, validation, and type safety.

Settings are organized into logical groups:
- Database and Redis configuration
- TWS integration settings
- Security and authentication
- Logging and monitoring
- AI/ML model configurations

v5.4.9: Legacy properties integrated directly (settings_legacy.py removed)
"""

from __future__ import annotations

import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import shared types and validators from separate modules
from .settings_types import CacheHierarchyConfig, Environment
from .settings_validators import SettingsValidators


class Settings(BaseSettings, SettingsValidators):
    """
    Configurações da aplicação com validação type-safe.

    Todas as configurações podem ser sobrescritas via variáveis de ambiente
    com o prefixo APP_ (ex: APP_ENVIRONMENT=production).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="APP_",
        # v6.0.2: Enforce strict validation
        validate_default=True,
    )

    # ============================================================================
    # APLICAÇÃO
    # ============================================================================
    # v5.9.7: Accept both APP_ENVIRONMENT (preferred) and legacy ENVIRONMENT
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        validation_alias=AliasChoices("ENVIRONMENT", "APP_ENVIRONMENT"),
        description="Ambiente de execução",
    )

    project_name: str = Field(
        default="Resync",
        min_length=1,
        description="Nome do projeto",
    )

    project_version: str = Field(
        default="6.2.0",
        pattern=(
            r"^\d+\.\d+\.\d+(?:-(?:(?:0|[1-9]\d*|[a-zA-Z-][0-9a-zA-Z-]*)"
            r"(?:\.(?:0|[1-9]\d*|[a-zA-Z-][0-9a-zA-Z-]*))*))?"
            r"(?:\+(?:[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        ),
        description="Versão do projeto (semver com pre-release e build metadata)",
    )

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    log_format: Literal["text", "json"] = Field(
        default="json", description="Formato dos logs: text ou json"
    )

    service_name: str = Field(default="resync", description="Nome do serviço para logs")

    log_sensitive_data_redaction: bool = Field(
        default=True,
        description=("Enable redaction of sensitive data in logs (passwords, tokens, etc.)"),
    )

    description: str = Field(
        default="Real-time monitoring dashboard for HCL Workload Automation",
        description="Descrição do projeto",
    )

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[1],
        description="Diretório base da aplicação",
    )

    # v5.9.7: Accept both APP_LOG_LEVEL (preferred) and legacy LOG_LEVEL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "APP_LOG_LEVEL"),
        description="Nível de logging",
    )

    # ============================================================================
    # CONTEXT STORE (LEGACY - PostgreSQL now used)
    # ============================================================================
    # DEPRECATED: SQLite context store removed in v5.3
    # The system now uses PostgreSQL for all persistence
    # This field is kept for backward compatibility only
    context_db_path: str = Field(
        default="",
        description="DEPRECATED: SQLite removed - using PostgreSQL. Keep empty.",
        deprecated=True,
    )

    # ============================================================================
    # CONNECTION POOLS (PostgreSQL) - v5.3.22 adjusted for single VM
    # For Docker/K8s: increase via environment variables
    # ============================================================================
    db_pool_min_size: int = Field(default=5, ge=1, le=100)  # Reduced from 20
    db_pool_max_size: int = Field(default=20, ge=1, le=1000)  # Reduced from 100
    db_pool_idle_timeout: int = Field(default=600, ge=60)  # Reduced from 1200
    db_pool_connect_timeout: int = Field(default=30, ge=5)  # Reduced from 60
    db_pool_health_check_interval: int = Field(default=60, ge=10)
    db_pool_max_lifetime: int = Field(default=1800, ge=300)

    # ============================================================================
    # REDIS - v5.3.22 adjusted for single VM
    # ============================================================================
    # v5.9.7: Accept both APP_REDIS_URL (preferred) and legacy REDIS_URL
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "APP_REDIS_URL"),
        description="URL de conexão Redis",
    )

    redis_min_connections: int = Field(default=1, ge=1, le=100)
    redis_max_connections: int = Field(default=10, ge=1, le=1000)
    redis_timeout: float = Field(default=30.0, gt=0)

    # Connection Pool - Redis
    redis_pool_min_size: int = Field(default=2, ge=1, le=100)  # Reduced from 5
    redis_pool_max_size: int = Field(default=10, ge=1, le=1000)  # Reduced from 20
    redis_pool_idle_timeout: int = Field(default=300, ge=60)
    redis_pool_connect_timeout: int = Field(default=15, ge=5)  # Reduced from 30
    redis_pool_health_check_interval: int = Field(default=60, ge=10)
    redis_pool_max_lifetime: int = Field(default=1800, ge=300)

    # Redis Initialization
    redis_max_startup_retries: int = Field(default=3, ge=1, le=10)
    redis_startup_backoff_base: float = Field(default=0.1, gt=0)
    redis_startup_backoff_max: float = Field(default=10.0, gt=0)
    redis_startup_lock_timeout: int = Field(
        default=30,
        ge=5,
        description="Timeout for distributed Redis initialization lock",
    )

    redis_health_check_interval: int = Field(
        default=5, ge=1, description="Interval for Redis connection health checks"
    )

    # Robust Cache Configuration
    robust_cache_max_items: int = Field(
        default=100_000,
        ge=100,
        description="Maximum number of items in robust cache",
    )
    robust_cache_max_memory_mb: int = Field(
        default=100, ge=10, description="Maximum memory usage for robust cache"
    )
    robust_cache_eviction_batch_size: int = Field(
        default=100, ge=1, description="Number of items to evict in one batch"
    )
    robust_cache_enable_weak_refs: bool = Field(
        default=True, description="Enable weak references for large objects"
    )
    robust_cache_enable_wal: bool = Field(
        default=False, description="Enable Write-Ahead Logging for cache"
    )
    robust_cache_wal_path: str | None = Field(
        default=None, description="Path for cache Write-Ahead Log"
    )

    # ============================================================================
    # LLM
    # ============================================================================
    llm_endpoint: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Endpoint da API LLM (NVIDIA) - usado como fallback se Ollama falhar",
    )

    llm_api_key: SecretStr = Field(
        default="",
        min_length=0,
        description=(
            "Chave de API do LLM (NVIDIA). Deve ser configurada via variável de ambiente."
        ),
        validation_alias="LLM_API_KEY",
        exclude=True,
        repr=False,
    )

    llm_timeout: float = Field(
        default=8.0,
        gt=0,
        description="Timeout para chamadas LLM em segundos (8s para fallback rápido ao cloud)",
    )

    auditor_model_name: str = Field(default="gpt-3.5-turbo")
    agent_model_name: str = Field(default="gpt-4o")

    llm_model: str = Field(
        default="ollama/qwen2.5:3b",
        description="Modelo LLM padrão (Ollama local ou cloud)",
    )

    # ============================================================================
    # OLLAMA - LOCAL LLM (v5.2.3.21)
    # ============================================================================
    ollama_enabled: bool = Field(
        default=True,
        description="Habilitar Ollama como provider primário de LLM",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base do servidor Ollama",
    )

    ollama_model: str = Field(
        default="qwen2.5:3b",
        description="Modelo Ollama padrão (sem prefixo ollama/)",
    )

    ollama_num_ctx: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description="Tamanho da janela de contexto do Ollama",
    )

    ollama_num_thread: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Número de threads CPU para Ollama (usar = núcleos físicos)",
    )

    ollama_timeout: float = Field(
        default=8.0,
        gt=0,
        description="Timeout para Ollama em segundos (agressivo para fallback rápido)",
    )

    # Fallback cloud model quando Ollama falha
    llm_fallback_model: str = Field(
        default="gpt-4o-mini",
        description="Modelo de fallback na nuvem quando Ollama timeout/falha",
    )

    # ============================================================================
    # LANGFUSE - PROMPT MANAGEMENT & OBSERVABILITY
    # ============================================================================
    langfuse_enabled: bool = Field(
        default=False,
        description="Enable LangFuse integration for prompt management and tracing",
    )

    langfuse_public_key: str = Field(
        default="",
        description="LangFuse public key for API authentication",
    )

    langfuse_secret_key: SecretStr = Field(
        default="",
        description="LangFuse secret key for API authentication",
        validation_alias="LANGFUSE_SECRET_KEY",
        exclude=True,
        repr=False,
    )

    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="LangFuse host URL (cloud or self-hosted)",
    )

    langfuse_trace_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sample rate for LLM call tracing (1.0 = trace all)",
    )

    # ============================================================================
    # LANGGRAPH - AGENT ORCHESTRATION
    # ============================================================================
    langgraph_enabled: bool = Field(
        default=True,
        description="Enable LangGraph for state-based agent orchestration",
    )

    langgraph_checkpoint_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="Time-to-live for LangGraph checkpoints in hours",
    )

    langgraph_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed LLM/tool calls in LangGraph",
    )

    langgraph_require_approval: bool = Field(
        default=True,
        description="Require human approval for TWS action requests",
    )

    # ============================================================================
    # HYBRID RETRIEVER - BM25 + Vector Search (v5.2.3.22)
    # ============================================================================
    hybrid_vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso base da busca vetorial (semântica) no hybrid retriever",
    )

    hybrid_bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso base da busca BM25 (keywords) no hybrid retriever",
    )

    hybrid_auto_weight: bool = Field(
        default=True,
        description="Ajustar pesos automaticamente baseado no tipo de query (TWS patterns)",
    )

    # v5.2.3.23: Field boost weights for BM25 indexing
    hybrid_boost_job_name: float = Field(
        default=4.0,
        ge=0.0,
        le=10.0,
        description="Boost para nome de job no BM25 (mais alto = mais importante)",
    )

    hybrid_boost_error_code: float = Field(
        default=3.5,
        ge=0.0,
        le=10.0,
        description="Boost para códigos de erro (RC, ABEND) no BM25",
    )

    hybrid_boost_workstation: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Boost para nome de workstation no BM25",
    )

    hybrid_boost_job_stream: float = Field(
        default=2.5,
        ge=0.0,
        le=10.0,
        description="Boost para nome de job stream no BM25",
    )

    hybrid_boost_message_id: float = Field(
        default=2.5,
        ge=0.0,
        le=10.0,
        description="Boost para IDs de mensagem TWS (EQQQ...) no BM25",
    )

    hybrid_boost_resource: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Boost para nome de resource no BM25",
    )

    hybrid_boost_title: float = Field(
        default=1.5,
        ge=0.0,
        le=10.0,
        description="Boost para título do documento no BM25",
    )

    hybrid_boost_content: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Boost para conteúdo geral no BM25 (baseline)",
    )

    # ============================================================================
    # CACHE CONFIGURATION (v5.9.3 - TTL Diferenciado)
    # ============================================================================
    # Near Real-Time strategy: protect TWS API while providing fresh data

    # Dynamic status (jobs, workstations) - very short TTL
    cache_ttl_job_status: int = Field(
        default=10,
        ge=5,
        le=60,
        description="TTL in seconds for job status cache (near real-time)",
    )

    # Logs and output - short TTL
    cache_ttl_job_logs: int = Field(
        default=30,
        ge=10,
        le=120,
        description="TTL in seconds for job logs/stdlist cache",
    )

    # Static structure (dependencies, job definitions) - longer TTL
    cache_ttl_static_structure: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="TTL in seconds for static structure (dependencies, definitions)",
    )

    # Graph cache TTL
    cache_ttl_graph: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="TTL in seconds for dependency graph cache",
    )

    # ============================================================================
    # CACHE CONFIGURATION (Legacy)
    # ============================================================================
    # Cache Hierarchy Configuration
    cache_hierarchy_l1_max_size: int = Field(
        ge=1,
        default=5000, description="Maximum number of entries in L1 cache"
    )
    cache_hierarchy_l2_ttl: int = Field(
        gt=0,
        default=600, description="Time-To-Live for L2 cache entries in seconds"
    )
    # Cache Configuration
    enable_cache_swr: bool = Field(
        default=True, description="Enable cache stampede write protection"
    )
    cache_ttl_jitter_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Ratio of TTL to use as jitter to prevent thundering herd",
    )
    enable_cache_mutex: bool = Field(
        default=True, description="Enable cache mutex to prevent duplicate computations"
    )
    cache_hierarchy_l2_cleanup_interval: int = Field(
        gt=0,
        default=60, description="Cleanup interval for L2 cache in seconds"
    )
    cache_hierarchy_num_shards: int = Field(default=8, description="Number of shards for cache")
    cache_hierarchy_max_workers: int = Field(
        ge=1,
        default=4, description="Max workers for cache operations"
    )

    # ============================================================================
    # TWS (Workload Automation)
    # ============================================================================
    tws_mock_mode: bool = Field(
        default=True, description="Usar modo mock para TWS (desenvolvimento)"
    )

    tws_host: str | None = Field(default=None)
    tws_port: int | None = Field(default=None, ge=1, le=65535)
    tws_user: str | None = Field(
        default=None,
        validation_alias="TWS_USER",
        description="Usuário do TWS (obrigatório se não estiver em modo mock)",
    )
    tws_password: SecretStr | None = Field(
        default=None,
        validation_alias="TWS_PASSWORD",
        description="Senha do TWS (obrigatório se não estiver em modo mock)",
        exclude=True,
        repr=False,
    )
    tws_base_url: str = Field(default="http://localhost:31111")
    tws_request_timeout: float = Field(
        gt=0,
        default=30.0, description="Timeout for TWS requests in seconds"
    )

    # Resiliency tuning for TWS client (retries/backoff/timeouts)
    tws_joblog_timeout: float = Field(
        gt=0,
        default=60.0, description="Timeout for joblog endpoints in seconds"
    )
    
    # Fine-grained Timeouts (v6.0.2)
    tws_timeout_connect: float = Field(
        gt=0,
        default=5.0, description="Connect timeout for TWS requests"
    )
    tws_timeout_read: float = Field(
        gt=0,
        default=30.0, description="Read timeout for TWS requests"
    )
    tws_timeout_write: float = Field(
        gt=0,
        default=5.0, description="Write timeout for TWS requests"
    )
    tws_timeout_pool: float = Field(
        gt=0,
        default=5.0, description="Pool timeout for TWS requests"
    )

    tws_retry_total: int = Field(
        default=3, description="Total retry attempts for transient TWS failures"
    )
    tws_retry_backoff_base: float = Field(
        default=0.5, description="Base delay (seconds) for exponential backoff"
    )
    tws_retry_backoff_max: float = Field(
        gt=0,
        default=8.0, description="Maximum backoff delay (seconds)"
    )

    # ------------------------------------------------------------------
    # Predictive workflows
    # ------------------------------------------------------------------
    enable_predictive_workflows: bool = Field(
        default=False,
        description=(
            "Enable predictive maintenance/capacity workflows. When disabled, "
            "workflow nodes return safe defaults (no hidden 'predictions')."
        ),
    )
    tws_verify: bool | str = Field(
        default=True,
        description=(
            "TWS SSL verification (False/True/path to CA bundle). "
            "Set to False only in development with self-signed certs"
        ),
    )
    tws_ca_bundle: str | None = Field(
        default=None,
        description=("CA bundle for TWS TLS verification (ignored if tws_verify=False)"),
    )

    # Connection Pool - HTTP
    http_pool_min_size: int = Field(default=10, ge=1)
    http_pool_max_size: int = Field(default=100, ge=1)
    http_pool_idle_timeout: int = Field(default=300, ge=60)
    http_pool_connect_timeout: int = Field(default=10, ge=1)
    http_pool_health_check_interval: int = Field(default=60, ge=10)
    http_pool_max_lifetime: int = Field(default=1800, ge=300)

    # ============================================================================
    # MONITORAMENTO PROATIVO TWS
    # ============================================================================
    # Polling Configuration
    tws_polling_enabled: bool = Field(
        default=True,
        description="Habilita polling automático do TWS",
    )
    tws_polling_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Intervalo de polling em segundos (5s a 5min)",
    )
    tws_polling_mode: str = Field(
        default="fixed",
        description="Modo de polling: fixed, adaptive, scheduled",
    )

    # Alert Thresholds
    tws_job_stuck_threshold_minutes: int = Field(
        default=60,
        ge=10,
        description="Minutos para considerar um job stuck",
    )
    tws_job_late_threshold_minutes: int = Field(
        default=30,
        ge=5,
        description="Minutos para considerar um job atrasado",
    )
    tws_anomaly_failure_rate_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Taxa de falha para alerta de anomalia",
    )

    # Data Retention
    tws_retention_days_full: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Dias para reter dados completos",
    )
    tws_retention_days_summary: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Dias para reter sumários e eventos",
    )
    tws_retention_days_patterns: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Dias para reter padrões detectados",
    )

    # Pattern Detection
    tws_pattern_detection_enabled: bool = Field(
        default=True,
        description="Habilita detecção automática de padrões",
    )
    tws_pattern_detection_interval_minutes: int = Field(
        default=60,
        ge=15,
        description="Intervalo para rodar detecção de padrões",
    )
    tws_pattern_min_confidence: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Confiança mínima para reportar padrão",
    )

    # Solution Correlation
    tws_solution_correlation_enabled: bool = Field(
        default=True,
        description="Habilita sugestão de soluções baseada em histórico",
    )
    tws_solution_min_success_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Taxa de sucesso mínima para sugerir solução",
    )

    # Notifications
    tws_browser_notifications_enabled: bool = Field(
        default=True,
        description="Habilita notificações no browser",
    )
    tws_teams_notifications_enabled: bool = Field(
        default=False,
        description="Habilita notificações no Microsoft Teams",
    )
    tws_teams_webhook_url: str | None = Field(
        default=None,
        description="URL do webhook do Microsoft Teams",
    )

    # Dashboard
    tws_dashboard_theme: str = Field(
        default="auto",
        description="Tema do dashboard: auto, light, dark",
    )
    tws_dashboard_refresh_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Intervalo de refresh do dashboard",
    )

    # ============================================================================
    # SEGURANÇA
    # ============================================================================
    # JWT Configuration (v5.3.20 - consolidated from fastapi_app/core/config.py)
    secret_key: SecretStr = Field(
        default=SecretStr("CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"),
        validation_alias="SECRET_KEY",
        description="Secret key for JWT signing. MUST be set via SECRET_KEY env var in production.",
        exclude=True,
        repr=False,
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algorithm for JWT token signing",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Access token expiration time in minutes",
    )

    metrics_api_key_hash: str = Field(
        default="",
        description="SHA-256 hash of the API key for workstation metrics collection",
        exclude=True,
    )

    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug mode (never True in production)",
    )

    # Proxy settings for corporate environments
    use_system_proxy: bool = Field(
        default=False,
        description="Use system proxy settings for outbound connections",
    )

    # File upload settings
    upload_dir: Path = Field(
        default_factory=lambda: Path("uploads"),
        description="Directory for file uploads",
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum file size for uploads in bytes",
    )
    allowed_extensions: list[str] = Field(
        default=[".txt", ".pdf", ".docx", ".md", ".json"],
        description="Allowed file extensions for uploads",
    )

    # v5.9.7: Accept both APP_ADMIN_USERNAME (preferred) and legacy ADMIN_USERNAME
    admin_username: str = Field(
        default="admin",
        min_length=3,
        validation_alias=AliasChoices("ADMIN_USERNAME", "APP_ADMIN_USERNAME"),
        description="Nome de usuário do administrador",
    )
    admin_password: SecretStr | None = Field(
        default=None,
        # Reads from ADMIN_PASSWORD (without APP_ prefix)
        validation_alias="ADMIN_PASSWORD",
        description=("Senha do administrador. Deve ser configurada via variável de ambiente."),
        exclude=True,
        repr=False,
    )

    # CORS
    cors_allowed_origins: list[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=False)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    # Static Files
    static_cache_max_age: int = Field(default=3600, ge=0)

    # ============================================================================
    # SERVIDOR
    # ============================================================================
    server_host: str = Field(
        default="127.0.0.1", description="Host do servidor (padrão: localhost apenas)"
    )
    server_port: int = Field(default=8000, ge=1024, le=65535, description="Porta do servidor")

    # ============================================================================
    # RATE LIMITING (v5.3.22 - Production-hardened defaults)
    # ============================================================================
    rate_limit_public_per_minute: int = Field(default=60, ge=1)
    rate_limit_authenticated_per_minute: int = Field(default=300, ge=1)
    rate_limit_critical_per_minute: int = Field(default=10, ge=1)  # Reduced for security
    rate_limit_error_handler_per_minute: int = Field(default=10, ge=1)
    rate_limit_websocket_per_minute: int = Field(default=20, ge=1)
    rate_limit_dashboard_per_minute: int = Field(default=10, ge=1)
    rate_limit_storage_uri: str = Field(default="redis://localhost:6379/1")
    rate_limit_key_prefix: str = Field(default="resync:ratelimit:")
    rate_limit_sliding_window: bool = Field(default=True)

    # ============================================================================
    # COMPRESSION (v5.3.22 - Production optimization)
    # ============================================================================
    compression_enabled: bool = Field(
        default=True,
        description="Enable GZip compression for responses",
    )
    compression_minimum_size: int = Field(
        default=500,
        ge=0,
        description="Minimum response size in bytes to compress",
    )
    compression_level: int = Field(
        default=6,
        ge=1,
        le=9,
        description="GZip compression level (1=fastest, 9=best compression)",
    )

    # ============================================================================
    # HTTPS/TLS SECURITY (v5.3.22)
    # ============================================================================
    enforce_https: bool = Field(
        default=False,
        description="Enable HSTS and force HTTPS (set True in production behind TLS)",
    )
    ssl_redirect: bool = Field(
        default=False,
        description="Redirect HTTP to HTTPS (use when not behind reverse proxy)",
    )

    # ============================================================================
    # SESSION SECURITY (v5.3.22)
    # ============================================================================
    session_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=480,
        description="Session timeout in minutes (reduced for security)",
    )
    session_secure_cookie: bool = Field(
        default=True,
        description="Use secure cookies (HTTPS only)",
    )
    session_http_only: bool = Field(
        default=True,
        description="Prevent JavaScript access to session cookies",
    )
    session_same_site: str = Field(
        default="lax",
        description="SameSite cookie policy (strict, lax, none)",
    )

    # ============================================================================
    # WORKER CONFIGURATION (v5.3.22 - Docker/K8s compatible)
    # ============================================================================
    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes (set based on CPU cores)",
    )
    worker_class: str = Field(
        default="uvicorn.workers.UvicornWorker",
        description="Gunicorn worker class for async support",
    )
    worker_timeout: int = Field(
        default=120,
        ge=30,
        description="Worker timeout in seconds",
    )
    worker_keepalive: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Keep-alive timeout for worker connections",
    )
    graceful_timeout: int = Field(
        default=30,
        ge=5,
        description="Graceful shutdown timeout in seconds",
    )

    # ============================================================================
    # BACKUP CONFIGURATION (v5.3.22)
    # ============================================================================
    backup_enabled: bool = Field(
        default=True,
        description="Enable automatic backups",
    )
    backup_dir: Path = Field(
        default_factory=lambda: Path("backups"),
        description="Directory for backup files",
    )
    backup_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retain backups",
    )
    backup_schedule_cron: str = Field(
        default="0 2 * * *",
        description="Backup schedule in cron format (default: 2 AM daily)",
    )
    backup_include_database: bool = Field(
        default=True,
        description="Include database in backups",
    )
    backup_include_uploads: bool = Field(
        default=True,
        description="Include uploaded files in backups",
    )
    backup_include_config: bool = Field(
        default=True,
        description="Include configuration files in backups",
    )
    backup_compression: bool = Field(
        default=True,
        description="Compress backup files",
    )

    # ============================================================================
    # COMPUTED FIELDS
    # ============================================================================
    # File Ingestion Settings
    knowledge_base_dirs: list[Path] = Field(
        default_factory=lambda: [Path.cwd() / "resync/RAG"],
        description="Directories included in the knowledge base",
    )
    protected_directories: list[Path] = Field(
        default_factory=lambda: [Path.cwd() / "resync/RAG/BASE"],
        description="Protected directories that should not be modified",
    )

    # ============================================================================
    # RAG MICROSERVICE CONFIGURATION
    # ============================================================================
    rag_service_url: str = Field(
        default="http://localhost:8003",
        description="URL base do microserviço RAG (ex: http://rag-service:8000)",
    )
    rag_service_timeout: int = Field(
        gt=0,
        default=300,
        description="Timeout para requisições ao microserviço RAG (segundos)",
    )
    rag_service_max_retries: int = Field(
        ge=1,
        default=3,
        description="Número máximo de tentativas para requisições ao microserviço RAG",
    )
    rag_service_retry_backoff: float = Field(
        default=1.0,
        description=(
            "Fator de backoff exponencial para tentativas de requisição ao microserviço RAG"
        ),
    )

    # ============================================================================
    # ENTERPRISE MODULES (v5.5.0)
    # ============================================================================
    # Phase 1: Essential
    enterprise_enable_incident_response: bool = Field(
        default=True,
        description="Enable incident response module",
    )
    enterprise_enable_auto_recovery: bool = Field(
        default=True,
        description="Enable auto-recovery module",
    )
    enterprise_enable_runbooks: bool = Field(
        default=True,
        description="Enable runbooks automation",
    )

    # Phase 2: Compliance
    enterprise_enable_gdpr: bool = Field(
        default=False,
        description="Enable GDPR compliance (required for EU)",
    )
    enterprise_enable_encrypted_audit: bool = Field(
        default=True,
        description="Enable encrypted audit trail",
    )
    enterprise_enable_siem: bool = Field(
        default=False,
        description="Enable SIEM integration",
    )
    enterprise_siem_endpoint: str | None = Field(
        default=None,
        description="SIEM endpoint URL",
    )
    enterprise_siem_api_key: SecretStr | None = Field(
        default=None,
        description="SIEM API key",
    )

    # Phase 3: Observability
    enterprise_enable_log_aggregator: bool = Field(
        default=True,
        description="Enable log aggregation",
    )
    enterprise_enable_anomaly_detection: bool = Field(
        default=True,
        description="Enable ML anomaly detection",
    )
    enterprise_anomaly_sensitivity: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Anomaly detection sensitivity (0-1)",
    )

    # Phase 4: Resilience
    enterprise_enable_chaos_engineering: bool = Field(
        default=False,
        description="Enable chaos engineering (staging only!)",
    )
    enterprise_enable_service_discovery: bool = Field(
        default=False,
        description="Enable service discovery for microservices",
    )
    enterprise_service_discovery_backend: str = Field(
        default="consul",
        description="Service discovery backend (consul, kubernetes, etcd)",
    )

    # Auto-recovery settings
    enterprise_auto_recovery_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum auto-recovery retry attempts",
    )
    enterprise_auto_recovery_cooldown: int = Field(
        default=60,
        ge=10,
        description="Cooldown between recovery attempts (seconds)",
    )

    # Incident settings
    enterprise_incident_auto_escalate: bool = Field(
        default=True,
        description="Automatically escalate unresolved incidents",
    )
    enterprise_incident_escalation_timeout: int = Field(
        default=15,
        ge=5,
        description="Minutes before incident escalation",
    )

    # GDPR settings
    enterprise_gdpr_data_retention_days: int = Field(
        default=365,
        ge=30,
        description="Data retention period in days",
    )
    enterprise_gdpr_anonymization_enabled: bool = Field(
        default=True,
        description="Enable data anonymization for GDPR",
    )

    # ============================================================================
    # BACKWARD COMPATIBILITY PROPERTIES (UPPER_CASE aliases)
    # ============================================================================
    # v5.4.9: Integrated from settings_legacy.py

    # pylint: disable=invalid-name
    @property
    def RAG_SERVICE_URL(self) -> str:
        """Legacy alias for rag_service_url."""
        return self.rag_service_url

    @property
    def BASE_DIR(self) -> Path:
        """Legacy alias for base_dir."""
        return self.base_dir

    @property
    def PROJECT_NAME(self) -> str:
        """Legacy alias for project_name."""
        return self.project_name

    @property
    def PROJECT_VERSION(self) -> str:
        """Legacy alias for project_version."""
        return self.project_version

    @property
    def DESCRIPTION(self) -> str:
        """Legacy alias for description."""
        return self.description

    @property
    def LOG_LEVEL(self) -> str:
        """Legacy alias for log_level."""
        return self.log_level

    @property
    def ENVIRONMENT(self) -> str:
        """Legacy alias for environment."""
        env = self.environment
        return env.value if hasattr(env, "value") else str(env)

    @property
    def DEBUG(self) -> bool:
        """Legacy alias: True when environment == DEVELOPMENT."""
        env = self.environment
        return env == Environment.DEVELOPMENT

    @property
    def REDIS_URL(self) -> str:
        """Legacy alias for redis_url."""
        return self.redis_url

    @property
    def LLM_ENDPOINT(self) -> str | None:
        """Legacy alias for llm_endpoint."""
        return self.llm_endpoint

    @property
    def LLM_API_KEY(self) -> Any:
        """Legacy alias for llm_api_key."""
        return self.llm_api_key

    @property
    def LLM_TIMEOUT(self) -> float:
        """Legacy alias for llm_timeout."""
        return self.llm_timeout

    @property
    def ADMIN_USERNAME(self) -> str:
        """Legacy alias for admin_username."""
        return self.admin_username

    @property
    def ADMIN_PASSWORD(self) -> Any:
        """Legacy alias for admin_password."""
        return self.admin_password

    @property
    def TWS_MOCK_MODE(self) -> bool:
        """Legacy alias for tws_mock_mode."""
        return self.tws_mock_mode

    @property
    def TWS_HOST(self) -> str | None:
        """Legacy alias for tws_host."""
        return self.tws_host

    @property
    def TWS_PORT(self) -> int | None:
        """Legacy alias for tws_port."""
        return self.tws_port

    @property
    def TWS_USER(self) -> str | None:
        """Legacy alias for tws_user."""
        return self.tws_user

    @property
    def TWS_PASSWORD(self) -> Any:
        """Legacy alias for tws_password."""
        return self.tws_password

    @property
    def SERVER_HOST(self) -> str:
        """Legacy alias for server_host."""
        return self.server_host

    @property
    def SERVER_PORT(self) -> int:
        """Legacy alias for server_port."""
        return self.server_port

    @property
    def CORS_ALLOWED_ORIGINS(self) -> list[str]:
        """Legacy alias for cors_allowed_origins."""
        return self.cors_allowed_origins

    @property
    def CORS_ALLOW_CREDENTIALS(self) -> bool:
        """Legacy alias for cors_allow_credentials."""
        return self.cors_allow_credentials

    @property
    def CORS_ALLOW_METHODS(self) -> list[str]:
        """Legacy alias for cors_allow_methods."""
        return self.cors_allow_methods

    @property
    def CORS_ALLOW_HEADERS(self) -> list[str]:
        """Legacy alias for cors_allow_headers."""
        return self.cors_allow_headers

    @property
    def STATIC_CACHE_MAX_AGE(self) -> int:
        """Legacy alias for static_cache_max_age."""
        return self.static_cache_max_age

    @property
    def JINJA2_TEMPLATE_CACHE_SIZE(self) -> int:
        """Legacy alias derived from environment."""
        env = self.environment
        return 400 if env == Environment.PRODUCTION else 0

    @property
    def AGENT_CONFIG_PATH(self) -> Path:
        """Legacy alias computed from base_dir."""
        return self.base_dir / "config" / "agents.json"

    @property
    def MAX_CONCURRENT_AGENT_CREATIONS(self) -> int:
        """Legacy constant for compatibility."""
        return 5

    @property
    def TWS_ENGINE_NAME(self) -> str:
        """Legacy constant for compatibility."""
        return "TWS"

    @property
    def TWS_ENGINE_OWNER(self) -> str:
        """Legacy constant for compatibility."""
        return "twsuser"

    @property
    def TWS_REQUEST_TIMEOUT(self) -> float:
        """Legacy alias for tws_request_timeout."""
        return self.tws_request_timeout

    @property
    def AUDITOR_MODEL_NAME(self) -> str:
        """Legacy alias for auditor_model_name."""
        return self.auditor_model_name

    @property
    def AGENT_MODEL_NAME(self) -> str:
        """Legacy alias for agent_model_name."""
        return self.agent_model_name

    @cached_property
    def CACHE_HIERARCHY(self) -> CacheHierarchyConfig:
        """Legacy alias exposing cache hierarchy configuration object."""
        return CacheHierarchyConfig(
            l1_max_size=self.cache_hierarchy_l1_max_size,
            l2_ttl_seconds=self.cache_hierarchy_l2_ttl,
            l2_cleanup_interval=self.cache_hierarchy_l2_cleanup_interval,
            num_shards=self.cache_hierarchy_num_shards,
            max_workers=self.cache_hierarchy_max_workers,
        )

    @property
    def DB_POOL_MIN_SIZE(self) -> int:
        """Legacy alias for db_pool_min_size."""
        return self.db_pool_min_size

    @property
    def DB_POOL_MAX_SIZE(self) -> int:
        """Legacy alias for db_pool_max_size."""
        return self.db_pool_max_size

    @property
    def DB_POOL_IDLE_TIMEOUT(self) -> int:
        """Legacy alias for db_pool_idle_timeout."""
        return self.db_pool_idle_timeout

    @property
    def DB_POOL_CONNECT_TIMEOUT(self) -> int:
        """Legacy alias for db_pool_connect_timeout."""
        return self.db_pool_connect_timeout

    @property
    def DB_POOL_HEALTH_CHECK_INTERVAL(self) -> int:
        """Legacy alias for db_pool_health_check_interval."""
        return self.db_pool_health_check_interval

    @property
    def DB_POOL_MAX_LIFETIME(self) -> int:
        """Legacy alias for db_pool_max_lifetime."""
        return self.db_pool_max_lifetime

    @property
    def REDIS_POOL_MIN_SIZE(self) -> int:
        """Legacy alias for redis_pool_min_size."""
        return self.redis_pool_min_size

    @property
    def REDIS_POOL_MAX_SIZE(self) -> int:
        """Legacy alias for redis_pool_max_size."""
        return self.redis_pool_max_size

    @property
    def REDIS_POOL_IDLE_TIMEOUT(self) -> int:
        """Legacy alias for redis_pool_idle_timeout."""
        return self.redis_pool_idle_timeout

    @property
    def REDIS_POOL_CONNECT_TIMEOUT(self) -> int:
        """Legacy alias for redis_pool_connect_timeout."""
        return self.redis_pool_connect_timeout

    @property
    def REDIS_POOL_HEALTH_CHECK_INTERVAL(self) -> int:
        """Legacy alias for redis_pool_health_check_interval."""
        return self.redis_pool_health_check_interval

    @property
    def REDIS_POOL_MAX_LIFETIME(self) -> int:
        """Legacy alias for redis_pool_max_lifetime."""
        return self.redis_pool_max_lifetime

    @property
    def HTTP_POOL_MIN_SIZE(self) -> int:
        """Legacy alias for http_pool_min_size."""
        return self.http_pool_min_size

    @property
    def HTTP_POOL_MAX_SIZE(self) -> int:
        """Legacy alias for http_pool_max_size."""
        return self.http_pool_max_size

    @property
    def HTTP_POOL_IDLE_TIMEOUT(self) -> int:
        """Legacy alias for http_pool_idle_timeout."""
        return self.http_pool_idle_timeout

    @property
    def HTTP_POOL_CONNECT_TIMEOUT(self) -> int:
        """Legacy alias for http_pool_connect_timeout."""
        return self.http_pool_connect_timeout

    @property
    def HTTP_POOL_HEALTH_CHECK_INTERVAL(self) -> int:
        """Legacy alias for http_pool_health_check_interval."""
        return self.http_pool_health_check_interval

    @property
    def HTTP_POOL_MAX_LIFETIME(self) -> int:
        """Legacy alias for http_pool_max_lifetime."""
        return self.http_pool_max_lifetime

    @property
    def KNOWLEDGE_BASE_DIRS(self) -> list[Path]:
        """Legacy alias for knowledge_base_dirs."""
        return self.knowledge_base_dirs

    @property
    def PROTECTED_DIRECTORIES(self) -> list[Path]:
        """Legacy alias for protected_directories."""
        return self.protected_directories

    # pylint: enable=invalid-name

    # ============================================================================
    # ENVIRONMENT CHECKS (v5.9.3 FIX)
    # ============================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.environment == Environment.TEST

    # ============================================================================
    # MIGRATION GRADUAL - FEATURE FLAGS
    # ============================================================================
    # Controle de migração para novos componentes
    MIGRATION_USE_NEW_CACHE: bool = Field(
        default=False, description="Usar ImprovedAsyncCache ao invés de AsyncTTLCache"
    )
    MIGRATION_USE_NEW_TWS: bool = Field(
        default=False,
        description="Usar TWSClientFactory ao invés de implementação direta",
    )
    MIGRATION_USE_NEW_RATE_LIMIT: bool = Field(
        default=False,
        description="Usar RateLimiterManager ao invés de implementação básica",
    )
    MIGRATION_ENABLE_METRICS: bool = Field(
        default=True, description="Habilitar métricas de migração e monitoramento"
    )

    # ============================================================================
    # DOCUMENT KNOWLEDGE GRAPH (DKG)
    # ============================================================================
    KG_EXTRACTION_ENABLED: bool = Field(
        default=False,
        description="Habilitar extração de grafo de conhecimento na ingestão de documentos",
    )
    KG_EXTRACTION_MODEL: str = Field(
        default="",
        description="Modelo LLM para extração KG (vazio = usa modelo padrão do sistema)",
    )
    KG_EXTRACTION_MAX_CONCEPTS: int = Field(
        ge=1,
        default=10,
        description="Máximo de conceitos extraídos por chunk",
    )
    KG_EXTRACTION_MAX_EDGES: int = Field(
        ge=1,
        default=20,
        description="Máximo de relações extraídas por chunk",
    )
    KG_RETRIEVAL_ENABLED: bool = Field(
        default=False,
        description="Habilitar enriquecimento de contexto via Grafo de Conhecimento no agente",
    )
    KG_RETRIEVAL_DEPTH: int = Field(
        default=2,
        description="Profundidade máxima de traversal no grafo (CTE recursivo)",
    )
    KG_RETRIEVAL_MAX_EDGES: int = Field(
        ge=1,
        default=40,
        description="Maximum edges returned in the LLM context subgraph",
    )

    # ============================================================================
    # GRAPHRAG
    # ============================================================================
    GRAPHRAG_ENABLED: bool = Field(
        default=False,
        description="Habilitar inicialização do GraphRAG (subgraph retrieval + auto-discovery)",
    )

    # ============================================================================
    # STARTUP & HEALTH CHECKS
    # ============================================================================
    STARTUP_TCP_CHECK_TIMEOUT: float = Field(
        gt=0,
        default=3.0,
        description="Timeout in seconds for TCP reachability checks (TWS, etc.)",
    )
    STARTUP_LLM_HEALTH_TIMEOUT: float = Field(
        gt=0,
        default=5.0,
        description="Timeout in seconds for LLM service health check HTTP request",
    )
    STARTUP_REDIS_HEALTH_RETRIES: int = Field(
        ge=1,
        default=1,
        description="Max retry attempts for Redis connectivity check at startup",
    )
    STARTUP_REDIS_HEALTH_TIMEOUT: float = Field(
        gt=0,
        default=3.0,
        description="Timeout in seconds for Redis health check at startup",
    )
    require_llm_at_boot: bool = Field(
        default=False,
        description="If True, LLM service must be reachable for startup to succeed",
    )
    require_tws_at_boot: bool = Field(
        default=False,
        description="If True, TWS must be reachable for startup to succeed",
    )

    # ============================================================================
    # APP FACTORY & LIFESPAN
    # ============================================================================
    SHUTDOWN_TASK_CANCEL_TIMEOUT: float = Field(
        gt=0,
        default=5.0,
        description="Timeout in seconds for cancelling background tasks during shutdown",
    )
    ETAG_HASH_LENGTH: int = Field(
        default=16,
        description="Number of hex characters from SHA-256 used for static file ETag (16 = 64-bit)",
    )
    MIN_ADMIN_PASSWORD_LENGTH: int = Field(
        ge=1,
        default=8,
        description="Minimum length for admin password in production",
    )
    MIN_SECRET_KEY_LENGTH: int = Field(
        ge=1,
        default=32,
        description="Minimum length for SECRET_KEY in production",
    )

    # ============================================================================
    # VALIDADORES
    # ============================================================================
    # Validators are now imported from settings_validators.py

    # SSL/TLS (compat-shims; não usados diretamente por Pydantic)
    # >>> Enable certificate validation by default for security <<<
    # Set TWS_VERIFY=false in .env for development with self-signed certs
    TWS_VERIFY: bool | str = True  # SSL enabled by default
    TWS_CA_BUNDLE: str | None = None

    def __repr__(self) -> str:
        """Representation that excludes sensitive fields from the output."""
        fields: dict[str, Any] = {}
        for name, field_info in self.__class__.model_fields.items():
            if field_info.exclude:
                continue
            fields[name] = getattr(self, name, None)

        parts = [f"{name}={value!r}" for name, value in fields.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"


# -----------------------------------------------------------------------------
# Instância global (lazy) + helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Factory para obter settings (útil para dependency injection)."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()


class _SettingsProxy:
    """Proxy de conveniência para manter compatibilidade."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings = _SettingsProxy()


# -----------------------------------------------------------------------------
# Backward helper retained
# -----------------------------------------------------------------------------
def load_settings() -> Settings:
    """Load application settings (backward-compat shim)."""
    return settings  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# PEP 562 Lazy Imports (kept if other modules expect them from this namespace)
# -----------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}
_LOADED_IMPORTS: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """PEP 562 __getattr__ for lazy imports to avoid circular dependencies."""
    if name in _LAZY_IMPORTS:
        if name not in _LOADED_IMPORTS:
            try:
                module_name, attr = _LAZY_IMPORTS[name]
                module = __import__(module_name, fromlist=[attr])
                _LOADED_IMPORTS[name] = getattr(module, attr)
            except ImportError:
                _LOADED_IMPORTS[name] = None
        return _LOADED_IMPORTS[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


# =============================================================================
# TEAMS OUTGOING WEBHOOK CONFIGURATION
# =============================================================================

TEAMS_OUTGOING_WEBHOOK = {
    "enabled": os.getenv("TEAMS_OUTGOING_WEBHOOK_ENABLED", "false").lower() == "true",
    "security_token": os.getenv("TEAMS_OUTGOING_WEBHOOK_SECURITY_TOKEN", ""),
    "webhook_name": os.getenv("TEAMS_OUTGOING_WEBHOOK_NAME", "resync"),
    "callback_url": os.getenv("TEAMS_CALLBACK_URL", ""),
    "response_timeout": int(os.getenv("TEAMS_OUTGOING_WEBHOOK_TIMEOUT", "25")),
    "max_response_length": 28000,  # Teams limit
}
