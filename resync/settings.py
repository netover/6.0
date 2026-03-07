# ruff: noqa: E501
"""Application settings and configuration management.

This module defines all application settings using Pydantic BaseSettings,
providing centralized configuration management with environment variable
support, validation, and type safety.

Settings are organized into logical groups:
- Database and Valkey configuration
- TWS integration settings
- Security and authentication
- Logging and monitoring
- AI/ML model configurations

v5.4.9: Legacy properties integrated directly (settings_legacy.py removed)
"""
from __future__ import annotations

import threading  # [P1-07 FIX] For thread-safe singleton
from functools import cached_property
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode

# Import shared types and validators from separate modules
from .settings_types import CacheConfig, CacheHierarchyConfig, Environment
from .settings_validators import SettingsValidators

class Settings(BaseSettings, SettingsValidators):
    """
    Configurações da aplicação com validação type-safe.

    Todas as configurações podem ser sobrescritas via variáveis de ambiente
    com o prefixo APP_ (ex: APP_ENVIRONMENT=production).
    """
    valkey_health_timeout: float = 2.0
    rate_limit_fail_open_auth: bool = False
    rate_limit_fail_open_feedback: bool = True
    valkey_backend_name: str = "valkey"


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
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Ambiente de execução",
    )

    project_name: str = Field(
        default="Resync",
        min_length=1,
        description="Nome do projeto",
    )

    project_version: str = Field(
        default="6.2.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Versão do projeto (semantic versioning X.Y.Z)",
    )

    startup_timeout: int = Field(
        default=600,
        ge=10,
        description="Timeout global para o startup da aplicação em segundos",
    )

    # [FIX BUG #7] Enable optional services at startup - was missing from Settings
    startup_enable_optional_services: bool = Field(
        default=True,
        description="Enable optional service initialization at startup (metrics, cache warmup, etc.)",
    )

    # [P2 FIX] Metrics collector flag - legacy monitoring_dashboard module has no routes
    # Default is False until routes are explicitly wired to consume it
    startup_enable_metrics_collector: bool = Field(
        default=False,
        description="Enable legacy metrics_collector_loop (monitoring_dashboard) - requires routes to be wired",
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
        description=(
            "Enable redaction of sensitive data in logs (passwords, tokens, etc.)"
        ),
    )

    description: str = Field(
        default="Real-time monitoring dashboard for HCL Workload Automation",
        description="Descrição do projeto",
    )

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent,
        description="Diretório base da aplicação (resync package directory)",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nível de logging",
    )

    startup_log_path: Path | None = Field(
        default=None,
        description=(
            "Path opcional para persistir logs de startup (warnings/errors). "
            "Se None, apenas stdout/stderr. "
            "Evita crash em containers/non-root."
        ),
    )

    # When enabled, re-raise programming errors (TypeError/KeyError/AttributeError/IndexError)
    # that are caught by broad exception handlers, to fail fast in staging/canary.
    # Default is False for backward compatibility.
    strict_exception_handling: bool = Field(
        default=False,
        description="Re-levanta erros de programação capturados por handlers amplos (fail-fast)",
    )



    # When enabled, export a higher-cardinality metric label ("site" = file:line) for
    # programming errors caught by broad exception handlers. Useful to locate hotspots
    # quickly in staging/canary; keep disabled in production to avoid label explosion.
    programming_error_metrics_detailed_site: bool = Field(
        default=False,
        description="Métrica detalhada (site=file:linha) para erros de programação capturados (staging/canary)",
    )

    # ============================================================================
    # CONNECTION POOLS (PostgreSQL) - v6.0 adjusted for single VM
    # For Docker/K8s: increase via environment variables
    # ============================================================================
    db_pool_min_size: int = Field(default=2, ge=1, le=100)  # Reduced from 5
    db_pool_max_size: int = Field(default=10, ge=1, le=1000)  # Reduced from 20
    db_pool_idle_timeout: int = Field(default=600, ge=60)  # Reduced from 1200
    db_pool_connect_timeout: int = Field(default=30, ge=5)  # Reduced from 60
    db_pool_health_check_interval: int = Field(default=60, ge=10)
    db_pool_max_lifetime: int = Field(default=1800, ge=300)

    # ============================================================================
    # VALKEY (Valkey) - v6.3 unified
    # ============================================================================

    valkey_url: SecretStr = Field(
        default=SecretStr("valkey://localhost:6379/0"),
        description="URL de conexão Valkey",
        repr=False,
    )

    database_url: SecretStr = Field(
        default=SecretStr("postgresql+asyncpg://localhost:5432/resync"),
        description="URL de conexão com banco de dados PostgreSQL",
        repr=False,
    )

    # Legacy fields - use valkey_pool_* instead
    valkey_timeout: float = Field(default=30.0, gt=0)

    # Connection Pool - Valkey (Canonical)
    valkey_pool_min_size: int = Field(default=1, ge=1, le=100)
    valkey_pool_max_size: int = Field(default=5, ge=1, le=1000)
    valkey_pool_idle_timeout: int = Field(default=300, ge=60)
    valkey_pool_connect_timeout: int = Field(default=15, ge=5)
    valkey_pool_health_check_interval: int = Field(default=60, ge=10)
    valkey_pool_max_lifetime: int = Field(default=1800, ge=300)

    # Valkey Initialization
    valkey_max_startup_retries: int = Field(default=3, ge=1, le=10)
    valkey_startup_backoff_base: float = Field(default=0.1, gt=0)
    valkey_startup_backoff_max: float = Field(default=10.0, gt=0)
    valkey_startup_lock_timeout: int = Field(
        default=30,
        ge=5,
        description="Timeout for distributed Valkey initialization lock",
    )

    valkey_health_check_interval: int = Field(
        default=5, ge=1, description="Interval for Valkey connection health checks"
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
        default="https://openrouter.ai/api/v1",
        description="Endpoint da API LLM - usado como fallback se Ollama falhar",
    )

    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        min_length=0,
        description="Chave de API do LLM. Deve ser configurada via APP_LLM_API_KEY.",
        exclude=True,
        repr=False,
    )

    llm_timeout: float = Field(
        default=8.0,
        gt=0,
        description="Timeout para chamadas LLM em segundos (8s para fallback rápido ao cloud)",
    )

    
    llm_cache_enabled: bool = Field(
        default=True,
        description="Habilita semantic cache do LiteLLM via Valkey (reduz custo/latência)",
    )

    llm_cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        le=7 * 24 * 3600,
        description="TTL do semantic cache do LiteLLM (segundos). 0 desativa expiração.",
    )

    auditor_model_name: str = Field(default="liteLLM-default")
    agent_model_name: str = Field(default="liteLLM-default")

    llm_model: str = Field(
        default="liteLLM-default",
        description="Modelo/alias LLM padrão (resolvido via LiteLLM Router)",
    )

    # ============================================================================
    # OLLAMA - LOCAL LLM (v5.2.3.21)
    # ============================================================================
    ollama_enabled: bool = Field(
        default=False,
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

    # ---------------------------------------------------------------------
    # LLM Context Management (History Compaction)
    # ---------------------------------------------------------------------

    llm_compact_history_enabled: bool = Field(
        default=True,
        description="Se true, compacta/resume histórico quando próximo do limite de contexto (protege estabilidade/latência)",
    )

    llm_context_safety_margin_tokens: int = Field(
        default=512,
        ge=0,
        le=8192,
        description="Margem de segurança em tokens para evitar estourar a janela de contexto (inclui overhead e variação de tokenização)",
    )

    llm_compact_history_min_recent_messages: int = Field(
        default=6,
        ge=0,
        le=100,
        description="Quantidade mínima de mensagens mais recentes para manter sem compactação",
    )

    llm_compact_history_summary_max_tokens: int = Field(
        default=800,
        ge=0,
        le=8192,
        description="Tamanho máximo (em tokens estimados) do resumo do histórico compactado",
    )

    # Fallback cloud model quando Ollama falha
    llm_fallback_model: str = Field(
        default="openrouter-free",
        description="Modelo de fallback na nuvem quando Ollama timeout/falha",
    )

    llm_max_tool_calls_per_turn: int = Field(
        default=6,
        ge=0,
        le=50,
        description="Limite máximo de tool-calls por turno para evitar loops/runaway (0 desativa)",
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

    langgraph_memory_store_max_items: int = Field(
        default=5000,
        ge=100,
        le=200000,
        description="Limite de itens no InMemoryStore do LangGraph antes de rotacionar (evita crescimento infinito)",
    )

    langgraph_max_plan_iterations: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Limite de iterações do executor de plano no LangGraph (evita loops infinitos)",
    )

    langgraph_max_regenerations: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Limite de regenerações por checagem de alucinação no LangGraph (evita loops)",
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
    # A2A PROTOCOL (v6.2.0)
    # ============================================================================
    a2a_enabled: bool = Field(
        default=True,
        description="Habilitar protocolo A2A (Agent-to-Agent) para interoperabilidade",
    )

    # ============================================================================
    # ORCHESTRATION - MULTI-AGENT WORKFLOWS (v6.3.0)
    # ============================================================================
    orchestration_enabled: bool = Field(
        default=True,
        description="Habilitar o motor de orquestração multi-agente",
    )

    orchestration_execution_ttl_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Dias para reter históricos de execução de orquestração",
    )

    orchestration_default_strategy: str = Field(
        default="sequential",
        description="Estratégia de execução padrão: sequential, parallel, consensus, fallback",
    )

    orchestration_parallel_max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Máximo de workers para execução paralela de steps",
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
    # CACHE CONFIGURATION (v6.3.0 - Unified CacheConfig)
    # ============================================================================
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Unified cache configuration (TTL, hierarchy, stampede protection)",
    )

    # Backward compatibility properties
    @property
    def cache_ttl_job_status(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.ttl_job_status"""
        return self.cache.ttl_job_status

    @property
    def cache_ttl_job_logs(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.ttl_job_logs"""
        return self.cache.ttl_job_logs

    @property
    def cache_ttl_static_structure(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.ttl_static_structure"""
        return self.cache.ttl_static_structure

    @property
    def cache_ttl_graph(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.ttl_graph"""
        return self.cache.ttl_graph

    @property
    def cache_hierarchy_l1_max_size(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.hierarchy_l1_max_size"""
        return self.cache.hierarchy_l1_max_size

    @property
    def cache_hierarchy_l2_ttl(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.hierarchy_l2_ttl"""
        return self.cache.hierarchy_l2_ttl

    @property
    def enable_cache_swr(self) -> bool:
        """.. deprecated:: 6.3.0 Use settings.cache.enable_swr"""
        return self.cache.enable_swr

    @property
    def cache_ttl_jitter_ratio(self) -> float:
        """.. deprecated:: 6.3.0 Use settings.cache.ttl_jitter_ratio"""
        return self.cache.ttl_jitter_ratio

    @property
    def enable_cache_mutex(self) -> bool:
        """.. deprecated:: 6.3.0 Use settings.cache.enable_mutex"""
        return self.cache.enable_mutex

    @property
    def cache_hierarchy_l2_cleanup_interval(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.hierarchy_l2_cleanup_interval"""
        return self.cache.hierarchy_l2_cleanup_interval

    @property
    def cache_hierarchy_num_shards(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.hierarchy_num_shards"""
        return self.cache.hierarchy_num_shards

    @property
    def cache_hierarchy_max_workers(self) -> int:
        """.. deprecated:: 6.3.0 Use settings.cache.hierarchy_max_workers"""
        return self.cache.hierarchy_max_workers

    # ============================================================================
    # WEBSOCKET POOL
    # ============================================================================
    # FIX P0-02: These fields were missing — websocket_pool_manager accessed them
    # causing AttributeError on every WebSocket connection.
    ws_pool_max_size: int = Field(
        default=100, ge=1, description="Maximum simultaneous WebSocket connections"
    )
    ws_pool_cleanup_interval: float = Field(
        default=30.0, gt=0, description="WebSocket pool cleanup interval in seconds"
    )
    ws_connection_timeout: float = Field(
        default=300.0, gt=0, description="WebSocket inactivity timeout in seconds"
    )
    ws_max_connection_duration: float = Field(
        default=7200.0, gt=0, description="Maximum WebSocket connection duration in seconds"
    )

    @property
    def WS_MAX_CONNECTION_DURATION(self) -> float:
        """Uppercase alias for backward compat with websocket_pool_manager."""
        return self.ws_max_connection_duration

    # ============================================================================
    # TWS (Workload Automation)
    # ============================================================================
    tws_mock_mode: bool = Field(
        default=True,
        description="Usar modo mock para TWS (desenvolvimento)",
    )

    tws_host: str | None = Field(default=None, description="Host do TWS")
    tws_port: int | None = Field(default=None, ge=1, le=65535, description="Porta do TWS")
    tws_user: str | None = Field(
        default=None,
        description="Usuário do TWS (obrigatório se não estiver em modo mock)",
    )
    tws_password: SecretStr | None = Field(
        default=None,
        description="Senha do TWS (obrigatório se não estiver em modo mock)",
        exclude=True,
        repr=False,
    )
    tws_engine_name: str = Field(
        default="TWS",
        description="Nome do engine TWS",
    )
    tws_base_url: str = Field(default="http://localhost:31111")
    tws_request_timeout: float = Field(
        gt=0, default=30.0, description="Timeout for TWS requests in seconds"
    )

    # Resiliency tuning for TWS client (retries/backoff/timeouts)
    tws_joblog_timeout: float = Field(
        gt=0, default=60.0, description="Timeout for joblog endpoints in seconds"
    )

    # Fine-grained Timeouts (v6.0.2)
    tws_timeout_connect: float = Field(
        gt=0, default=5.0, description="Connect timeout for TWS requests"
    )
    tws_timeout_read: float = Field(
        gt=0, default=30.0, description="Read timeout for TWS requests"
    )
    tws_timeout_write: float = Field(
        gt=0, default=5.0, description="Write timeout for TWS requests"
    )
    tws_timeout_pool: float = Field(
        gt=0, default=5.0, description="Pool timeout for TWS requests"
    )

    tws_retry_total: int = Field(
        default=3, description="Total retry attempts for transient TWS failures"
    )
    tws_retry_backoff_base: float = Field(
        default=0.5, description="Base delay (seconds) for exponential backoff"
    )
    tws_retry_backoff_max: float = Field(
        gt=0, default=8.0, description="Maximum backoff delay (seconds)"
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
    enable_roma_orchestration: bool = Field(
        default=False,
        description="Enable ROMA orchestration graph endpoints",
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
        description=(
            "CA bundle for TWS TLS verification (ignored if tws_verify=False)"
        ),
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
    tws_teams_webhook_url: SecretStr | None = Field(
        default=None,
        description="URL do webhook do Microsoft Teams",
        repr=False,
    )

    # Teams Outgoing Webhook (Validation Logic)
    teams_outgoing_webhook_enabled: bool = Field(
        default=False,
        description="Enable Teams outgoing webhook",
    )
    teams_outgoing_webhook_security_token: SecretStr = Field(
        default=SecretStr(""),
        description="Security token for Teams webhook",
        exclude=True,
        repr=False,
    )
    teams_outgoing_webhook_name: str = Field(
        default="resync",
    )
    teams_callback_url: str = Field(
        default="",
    )
    teams_outgoing_webhook_timeout: int = Field(
        default=25,
        ge=1,
        le=60,
        description="Response timeout for Teams webhook",
    )
    teams_outgoing_webhook_max_response_length: int = Field(
        default=28000,
        description="Teams message length limit",
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
    secret_key: SecretStr | None = Field(
        default=None,
        description="Secret key for JWT signing. Set via APP_SECRET_KEY.",
        exclude=True,
        repr=False,
    )
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = Field(
        default="HS256",
        description=(
            "Algorithm for JWT token signing. HMAC-only by default; "
            "asymmetric algorithms require different key handling."
        ),
    )
    jwt_leeway_seconds: int = Field(
        default=60,
        ge=0,
        le=600,
        description="Allowed clock skew (seconds) when validating JWT exp/nbf claims",
    )

    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Access token expiration time in minutes",
    )

    metrics_api_key_hash: SecretStr = Field(
        default=SecretStr(""),
        description="SHA-256 hash of the API key for workstation metrics collection",
        exclude=True,
        repr=False,
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

    admin_username: str = Field(
        default="admin",
        min_length=3,
        description="Nome de usuário do administrador",
    )
    admin_password: SecretStr | None = Field(
        default=None,
        description="Senha do administrador. Set via APP_ADMIN_PASSWORD.",
        exclude=True,
        repr=False,
    )

    operator_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Chave de API para acesso limitado de operador",
        exclude=True,
        repr=False,
    )

    # CORS
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:3000"],
        description=(
            "Allowed CORS origins. MUST be overridden in production via "
            "APP_CORS_ALLOWED_ORIGINS env var (comma-separated). "
            "Wildcard '*' is rejected in production by _validate_critical_settings()."
        ),
    )
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
    server_port: int = Field(
        default=8000, ge=1024, le=65535, description="Porta do servidor"
    )

    # ============================================================================
    # OBSERVABILITY — Sentry SDK
    # Set SENTRY_DSN (or APP_SENTRY_DSN) to enable error tracking.
    # ============================================================================
    sentry_dsn: SecretStr | None = Field(
        default=None,
        description="Sentry DSN for error tracking (optional). Leave unset to disable.",
        repr=False,
    )
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of transactions to send to Sentry for performance monitoring.",
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of sampled transactions to profile (requires traces_sample_rate > 0).",
    )

    # ============================================================================
    # RATE LIMITING (v5.3.22 - Production-hardened defaults)
    # ============================================================================
    rate_limit_public_per_minute: int = Field(default=60, ge=1)
    rate_limit_authenticated_per_minute: int = Field(default=300, ge=1)
    rate_limit_critical_per_minute: int = Field(
        default=10, ge=1
    )  # Reduced for security
    rate_limit_error_handler_per_minute: int = Field(default=10, ge=1)
    rate_limit_websocket_per_minute: int = Field(default=20, ge=1)
    rate_limit_dashboard_per_minute: int = Field(default=10, ge=1)
    rate_limit_storage_uri: SecretStr = Field(
        default=SecretStr("valkey://localhost:6379/1"),
        repr=False,
    )
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
        default="uvicorn_worker.UvicornWorker",
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
        repr=False,
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

    # pylint
    @property
    def valkey_url_secret(self) -> SecretStr:
        """Safe access to the Valkey URL without automatic SecretStr unwrap.

        .. deprecated:: 6.3.0
            Use settings.valkey_url directly instead.
        """
        import warnings
        warnings.warn("Use settings.valkey_url directly", DeprecationWarning, stacklevel=2)
        return self.valkey_url

    # pylint

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

    # ============================================================================
    # DOCUMENT KNOWLEDGE GRAPH (DKG)
    # ============================================================================
    knowledge_docs_root: Path = Field(
        default_factory=lambda: Path("docs"),
        description=(
            "Allowed root directory for server-side batch document ingestion. "
            "All paths supplied to POST /knowledge/ingest/batch must resolve inside "
            "this directory. Set via APP_KNOWLEDGE_DOCS_ROOT. "
            "Default: ./docs relative to the working directory."
        ),
    )

    kg_extraction_enabled: bool = Field(
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
    STARTUP_VALKEY_HEALTH_RETRIES: int = Field(
        ge=1,
        default=1,
        description="Max retry attempts for Valkey connectivity check at startup",
    )
    STARTUP_VALKEY_HEALTH_TIMEOUT: float = Field(
        gt=0,
        default=3.0,
        description="Timeout in seconds for Valkey health check at startup",
    )
    require_llm_at_boot: bool = Field(
        default=False,
        description="If True, LLM service must be reachable for startup to succeed",
    )
    require_tws_at_boot: bool = Field(
        default=False,
        description="If True, TWS must be reachable for startup to succeed",
    )
    require_rag_at_boot: bool = Field(
        default=False,
        description="If True, RAG service must be reachable for startup to succeed",
    )

    # ============================================================================
    # APP FACTORY & LIFESPAN
    # ============================================================================
    SHUTDOWN_TASK_CANCEL_TIMEOUT: float = Field(
        gt=0,
        default=5.0,
        description="Timeout in seconds for cancelling background tasks during shutdown",
    )
    etag_hash_length: int = Field(
        default=16,
        description="Number of hex characters from SHA-256 used for static file ETag (16 = 64-bit)",
    )
    min_admin_password_length: int = Field(
        ge=1,
        default=8,
        description="Minimum length for admin password in production",
    )
    min_secret_key_length: int = Field(
        ge=1,
        default=32,
        description="Minimum length for SECRET_KEY in production",
    )

    smtp_enabled: bool = Field(
        default=False,
        description="Enable SMTP email notifications",
    )
    smtp_host: str = Field(
        default="localhost",
        description="SMTP server host",
    )
    smtp_port: int = Field(
        default=587,
        ge=1,
        le=65535,
        description="SMTP server port",
    )
    smtp_username: str | None = Field(
        default=None,
        description="SMTP username",
    )
    smtp_password: SecretStr | None = Field(
        default=None,
        description="SMTP password",
        exclude=True,
        repr=False,
    )
    smtp_from_email: str = Field(
        default="noreply@resync.local",
        description="Default sender email address",
    )
    smtp_use_tls: bool = Field(
        default=True,
        description="Use TLS for SMTP connection",
    )
    smtp_timeout: int = Field(
        default=30,
        ge=1,
        description="SMTP connection timeout in seconds",
    )
    # ============================================================================
    # VALIDADORES
    # ============================================================================
    # Validators are now imported from settings_validators.py

    def __repr__(self) -> str:
        """Representation that excludes sensitive fields from the output.
        
        [P0-08 FIX] Respects both field_info.exclude AND field_info.repr to prevent
        SecretStr leakage in logs/traces. SecretStr fields are masked as '**********'.
        """
        fields: dict[str, Any] = {}
        for name, field_info in self.__class__.model_fields.items():
            value = getattr(self, name, None)

            # Always mask secrets, even for repr=False/exclude fields.
            if isinstance(value, SecretStr):
                fields[name] = "SecretStr('**********')"
                continue

            # Non-secret hidden fields remain omitted.
            if field_info.exclude or (hasattr(field_info, 'repr') and not field_info.repr):
                continue

            fields[name] = value

        parts = [f"{name}={value!r}" for name, value in fields.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

# -----------------------------------------------------------------------------
# Instância global (lazy) + helpers
# -----------------------------------------------------------------------------
# [P1-07 FIX] Thread-safe singleton with explicit lock
_settings_instance: Settings | None = None
_settings_lock = threading.Lock()

def get_settings() -> Settings:
    """Factory para obter settings (útil para dependency injection).
    
    [P1-07 FIX] Thread-safe lazy singleton without lru_cache complexity.
    Uses double-checked locking pattern for optimal performance.
    
    [P0-07 FIX] SECRET_KEY auto-generation removed. Pydantic validators
    enforce SECRET_KEY presence in production. For dev/test, users MUST
    set SECRET_KEY explicitly via environment variable to ensure
    deterministic behavior.
    
    Returns:
        Settings: Immutable settings singleton instance
        
    Note:
        Use clear_settings_cache() to force reload (e.g., in tests)
    """
    global _settings_instance
    
    # Fast path: instance already created (99% of calls)
    if _settings_instance is not None:
        return _settings_instance
    
    # Slow path: acquire lock for thread-safe initialization
    with _settings_lock:
        # Double-check: another thread may have created it while we waited
        if _settings_instance is not None:
            return _settings_instance
        
        # [P0-07 FIX] No automatic SECRET_KEY generation
        # Pydantic validators handle production enforcement
        Settings.model_rebuild()
        _settings_instance = Settings()
        return _settings_instance

def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing).
    
    [P1-07 FIX] Thread-safe cache clearing.
    """
    global _settings_instance
    with _settings_lock:
        _settings_instance = None

class _SettingsProxy:
    """[P1-09 FIX] Read-only proxy to Settings singleton.

    Enforces immutability of settings after initialization. Tests that need to
    modify settings should use clear_settings_cache() + environment variables,
    not direct mutation.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying Settings instance.

        Args:
            name: Attribute name to access

        Returns:
            The attribute value from the Settings singleton

        Raises:
            AttributeError: If the attribute does not exist on Settings,
                with a helpful error message listing available attributes.
        """
        try:
            return getattr(get_settings(), name)
        except AttributeError:
            settings_obj = get_settings()
            available = [
                attr for attr in dir(settings_obj)
                if not attr.startswith('_')
            ]
            raise AttributeError(
                f"Settings has no attribute '{name}'. "
                f"Did you mean one of: {', '.join(available[:10])}...? "
                f"See dir(settings) for full list."
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """[P1-09 FIX] Settings are immutable after construction.
        
        Raises:
            AttributeError: Always (settings cannot be modified after init)
        """
        raise AttributeError(
            f"Cannot set attribute '{name}' on immutable Settings. "
            "To modify settings in tests, use clear_settings_cache() and set "
            "environment variables before calling get_settings()."
        )

    def __repr__(self) -> str:
        return repr(get_settings())

    def __dir__(self) -> list[str]:
        return dir(get_settings())

settings = _SettingsProxy()

# =============================================================================
# TEAMS OUTGOING WEBHOOK CONFIGURATION
# =============================================================================

# Define a proxy class to lazily access settings for the dictionary interface
class _TeamsConfigProxy(Mapping[str, Any]):
    """Read-only Mapping que delega ao Pydantic Settings."""

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "enabled": "teams_outgoing_webhook_enabled",
        "security_token": "teams_outgoing_webhook_security_token",
        "webhook_name": "teams_outgoing_webhook_name",
        "callback_url": "teams_callback_url",
        "response_timeout": "teams_outgoing_webhook_timeout",
        "max_response_length": "teams_outgoing_webhook_max_response_length",
    }

    # Keys whose values are secrets and must not be unwrapped implicitly.
    _SECRET_KEYS: ClassVar[frozenset[str]] = frozenset({"security_token"})

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEY_MAP:
            raise KeyError(key)
        s = get_settings()
        value = getattr(s, self._KEY_MAP[key])
        # IMPORTANT: do NOT unwrap SecretStr automatically.
        return value

    def get_secret(self, key: str) -> str:
        """Explicitly unwrap a secret key.

        This is the only supported way to access secret values.
        """
        if key not in self._SECRET_KEYS:
            raise KeyError(f"{key!r} is not a secret key")
        value = self[key]
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return str(value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEY_MAP)

    def __len__(self) -> int:
        return len(self._KEY_MAP)

    def __repr__(self) -> str:
        safe: dict[str, object] = {}
        for k in self:
            try:
                v = self[k]
                safe[k] = "**********" if isinstance(v, SecretStr) else v
            except Exception:  # noqa: BLE001
                safe[k] = "<unavailable>"
        return repr(safe)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

TEAMS_OUTGOING_WEBHOOK = _TeamsConfigProxy()

# ============================================================================
# NOTIFICATION (EMAIL/SMTP)
# ============================================================================
