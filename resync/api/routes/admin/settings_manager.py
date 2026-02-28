# ruff: noqa: E501
"""
Unified Settings Management API.

Provides comprehensive endpoints to view and modify ALL application settings
from the Settings class, organized by logical sections.

v5.9.9: Complete settings management for admin UI.
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials
from resync.settings import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/settings",
    tags=["Settings Management"],
    dependencies=[Depends(verify_admin_credentials)],
)

# Path for persisting settings overrides
SETTINGS_OVERRIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "config"
    / "settings_override.json"
)

# =============================================================================
# SETTINGS SCHEMA - Organized by Section
# =============================================================================

# =============================================================================
# HOT-RELOAD SUPPORT
# =============================================================================
# Settings marked with "hot_reload": True will take effect immediately after save.
# Settings marked with "hot_reload": False require application restart.
# Default is True (hot-reload supported) for most runtime settings.

SETTINGS_SCHEMA = {
    "application": {
        "title": "Aplicação",
        "icon": "fa-cog",
        "description": "Configurações gerais da aplicação",
        "fields": {
            "environment": {
                "type": "select",
                "options": ["development", "staging", "production"],
                "label": "Ambiente",
                "hot_reload": False,
                "restart_reason": "Ambiente afeta inicialização de middlewares e segurança",
            },
            "project_name": {
                "type": "text",
                "label": "Nome do Projeto",
                "hot_reload": True,
            },
            "project_version": {"type": "text", "label": "Versão", "readonly": True},
            "description": {"type": "text", "label": "Descrição", "hot_reload": True},
            "debug": {
                "type": "boolean",
                "label": "Modo Debug",
                "warning": "Nunca ativar em produção",
                "hot_reload": False,
                "restart_reason": "Debug mode afeta middlewares e error handlers",
            },
            "strict_exception_handling": {
                "type": "boolean",
                "label": "Fail-fast (Staging/Canary)",
                "description": "Re-levanta erros de programação capturados por handlers amplos. Ative apenas em staging/canary.",
                "warning": "Pode derrubar requests ao expor bugs mascarados. Use em canary primeiro.",
                "hot_reload": True,
            },
            "programming_error_metrics_detailed_site": {
                "type": "boolean",
                "label": "Métrica detalhada de erros (site=file:linha)",
                "description": "Inclui label site=file:linha na métrica de erros de programação capturados. Use apenas em staging/canary (cardinalidade alta).",
                "warning": "Pode aumentar muito a cardinalidade de séries no dashboard Prometheus-compatível. Mantenha OFF em produção.",
                "hot_reload": True,
            },
            "log_level": {
                "type": "select",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "label": "Nível de Log",
                "hot_reload": True,
            },
            "log_format": {
                "type": "select",
                "options": ["text", "json"],
                "label": "Formato de Log",
                "hot_reload": False,
                "restart_reason": "Formato de log é configurado na inicialização",
            },
            "service_name": {
                "type": "text",
                "label": "Nome do Serviço",
                "hot_reload": True,
            },
            "log_sensitive_data_redaction": {
                "type": "boolean",
                "label": "Redação de Dados Sensíveis",
                "hot_reload": True,
            },
        },
    },
    "server": {
        "title": "Servidor",
        "icon": "fa-server",
        "description": "Configurações do servidor web",
        "fields": {
            "server_host": {
                "type": "text",
                "label": "Host",
                "hot_reload": False,
                "restart_reason": "Requer reiniciar o servidor para bind em novo endereço",
            },
            "server_port": {
                "type": "number",
                "label": "Porta",
                "min": 1024,
                "max": 65535,
                "hot_reload": False,
                "restart_reason": "Requer reiniciar o servidor para bind em nova porta",
            },
            "workers": {
                "type": "number",
                "label": "Workers",
                "min": 1,
                "max": 32,
                "hot_reload": False,
                "restart_reason": "Número de workers é definido na inicialização do Gunicorn",
            },
            "worker_class": {
                "type": "text",
                "label": "Classe do Worker",
                "hot_reload": False,
                "restart_reason": "Worker class é definido na inicialização",
            },
            "worker_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 30,
                "hot_reload": False,
                "restart_reason": "Timeout do worker é definido na inicialização",
            },
            "worker_keepalive": {
                "type": "number",
                "label": "Keep-Alive (s)",
                "min": 1,
                "max": 30,
                "hot_reload": False,
                "restart_reason": "Keep-alive é configurado na inicialização",
            },
            "graceful_timeout": {
                "type": "number",
                "label": "Graceful Timeout (s)",
                "min": 5,
                "hot_reload": False,
                "restart_reason": "Graceful timeout é configurado na inicialização",
            },
        },
    },
    "database": {
        "title": "Database (PostgreSQL)",
        "icon": "fa-database",
        "description": "Configurações de conexão com PostgreSQL",
        "fields": {
            "db_pool_min_size": {
                "type": "number",
                "label": "Pool Min Size",
                "min": 1,
                "max": 100,
                "hot_reload": False,
                "restart_reason": "Pool de conexões é criado na inicialização",
            },
            "db_pool_max_size": {
                "type": "number",
                "label": "Pool Max Size",
                "min": 1,
                "max": 1000,
                "hot_reload": False,
                "restart_reason": "Pool de conexões é criado na inicialização",
            },
            "db_pool_idle_timeout": {
                "type": "number",
                "label": "Idle Timeout (s)",
                "min": 60,
                "hot_reload": True,
            },
            "db_pool_connect_timeout": {
                "type": "number",
                "label": "Connect Timeout (s)",
                "min": 5,
                "hot_reload": True,
            },
            "db_pool_health_check_interval": {
                "type": "number",
                "label": "Health Check (s)",
                "min": 10,
                "hot_reload": True,
            },
            "db_pool_max_lifetime": {
                "type": "number",
                "label": "Max Lifetime (s)",
                "min": 300,
                "hot_reload": True,
            },
        },
    },
    "redis": {
        "title": "Redis Cache",
        "icon": "fa-bolt",
        "description": "Configurações de cache Redis",
        "fields": {
            "redis_url": {
                "type": "text",
                "label": "URL de Conexão",
                "placeholder": "redis://localhost:6379/0",
                "hot_reload": False,
                "restart_reason": "URL do Redis requer reconexão",
            },
            "redis_min_connections": {
                "type": "number",
                "label": "Min Connections",
                "min": 1,
                "max": 100,
                "hot_reload": False,
                "restart_reason": "Pool de conexões é criado na inicialização",
            },
            "redis_max_connections": {
                "type": "number",
                "label": "Max Connections",
                "min": 1,
                "max": 1000,
                "hot_reload": False,
                "restart_reason": "Pool de conexões é criado na inicialização",
            },
            "redis_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 1,
                "hot_reload": True,
            },
            "redis_pool_min_size": {
                "type": "number",
                "label": "Pool Min",
                "min": 1,
                "hot_reload": False,
                "restart_reason": "Pool é criado na inicialização",
            },
            "redis_pool_max_size": {
                "type": "number",
                "label": "Pool Max",
                "min": 1,
                "hot_reload": False,
                "restart_reason": "Pool é criado na inicialização",
            },
            "redis_health_check_interval": {
                "type": "number",
                "label": "Health Check (s)",
                "min": 1,
                "hot_reload": True,
            },
        },
    },
    "llm": {
        "title": "LLM / AI",
        "icon": "fa-brain",
        "description": "Configurações de modelos de linguagem",
        "fields": {
            "llm_endpoint": {
                "type": "text",
                "label": "Endpoint LLM",
                "hot_reload": True,
            },
            "llm_model": {"type": "text", "label": "Modelo Padrão", "hot_reload": True},
            "llm_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 1,
                "step": 0.1,
                "hot_reload": True,
            },
            "llm_fallback_model": {
                "type": "text",
                "label": "Modelo Fallback",
                "hot_reload": True,
            },
            "auditor_model_name": {
                "type": "text",
                "label": "Modelo Auditor",
                "hot_reload": True,
            },
            "agent_model_name": {
                "type": "text",
                "label": "Modelo Agente",
                "hot_reload": True,
            },
        },
    },
    "ollama": {
        "title": "Ollama (Local LLM)",
        "icon": "fa-microchip",
        "description": "Configurações do Ollama para LLM local",
        "fields": {
            "ollama_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": True,
            },
            "ollama_base_url": {
                "type": "text",
                "label": "URL Base",
                "hot_reload": True,
            },
            "ollama_model": {"type": "text", "label": "Modelo", "hot_reload": True},
            "ollama_num_ctx": {
                "type": "number",
                "label": "Context Size",
                "min": 512,
                "max": 32768,
                "hot_reload": True,
            },
            "ollama_num_thread": {
                "type": "number",
                "label": "Threads CPU",
                "min": 1,
                "max": 32,
                "hot_reload": True,
            },
            "ollama_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 1,
                "step": 0.1,
                "hot_reload": True,
            },
        },
    },
    "langfuse": {
        "title": "LangFuse",
        "icon": "fa-chart-line",
        "description": "Observabilidade e gerenciamento de prompts",
        "fields": {
            "langfuse_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": False,
                "restart_reason": "Cliente LangFuse é inicializado uma vez",
            },
            "langfuse_host": {
                "type": "text",
                "label": "Host",
                "hot_reload": False,
                "restart_reason": "Host requer reconexão do cliente",
            },
            "langfuse_public_key": {
                "type": "text",
                "label": "Public Key",
                "hot_reload": False,
                "restart_reason": "Credenciais requerem reconexão",
            },
            "langfuse_trace_sample_rate": {
                "type": "number",
                "label": "Sample Rate",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "hot_reload": True,
            },
        },
    },
    "langgraph": {
        "title": "LangGraph",
        "icon": "fa-project-diagram",
        "description": "Orquestração de agentes",
        "fields": {
            "langgraph_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": False,
                "restart_reason": "Grafo de agentes é construído na inicialização",
            },
            "langgraph_checkpoint_ttl_hours": {
                "type": "number",
                "label": "Checkpoint TTL (h)",
                "min": 1,
                "hot_reload": True,
            },
            "langgraph_max_retries": {
                "type": "number",
                "label": "Max Retries",
                "min": 0,
                "max": 10,
                "hot_reload": True,
            },
            "langgraph_require_approval": {
                "type": "boolean",
                "label": "Requer Aprovação Humana",
                "hot_reload": True,
            },
        },
    },
    "orchestration": {
        "title": "Orquestração Multi-Agente",
        "icon": "fa-project-diagram",
        "description": "Configurações do novo motor de orquestração",
        "fields": {
            "orchestration_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": True,
            },
            "orchestration_execution_ttl_days": {
                "type": "number",
                "label": "Retenção de Execuções (dias)",
                "min": 1,
                "max": 365,
                "hot_reload": True,
            },
            "orchestration_default_strategy": {
                "type": "select",
                "options": ["sequential", "parallel", "consensus", "fallback"],
                "label": "Estratégia Padrão",
                "hot_reload": True,
            },
            "orchestration_parallel_max_workers": {
                "type": "number",
                "label": "Max Parallel Workers",
                "min": 1,
                "max": 32,
                "hot_reload": True,
            },
        },
    },
    "a2a": {
        "title": "A2A Protocol",
        "icon": "fa-exchange-alt",
        "description": "Interoperabilidade entre agentes (Agent-to-Agent)",
        "fields": {
            "a2a_enabled": {
                "type": "boolean",
                "label": "Habilitar Protocolo A2A",
                "hot_reload": True,
            },
        },
    },
    "tws": {
        "title": "TWS Connection",
        "icon": "fa-plug",
        "description": "Conexão com HCL Workload Automation",
        "fields": {
            "tws_mock_mode": {
                "type": "boolean",
                "label": "Modo Mock",
                "hot_reload": False,
                "restart_reason": "Mock mode afeta inicialização do cliente TWS",
            },
            "tws_host": {
                "type": "text",
                "label": "Host",
                "hot_reload": False,
                "restart_reason": "Requer reconexão ao TWS",
            },
            "tws_port": {
                "type": "number",
                "label": "Porta",
                "min": 1,
                "max": 65535,
                "hot_reload": False,
                "restart_reason": "Requer reconexão ao TWS",
            },
            "tws_user": {
                "type": "text",
                "label": "Usuário",
                "hot_reload": False,
                "restart_reason": "Credenciais requerem reautenticação",
            },
            "tws_base_url": {
                "type": "text",
                "label": "URL Base",
                "hot_reload": False,
                "restart_reason": "Requer reconexão ao TWS",
            },
            "tws_request_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 1,
                "hot_reload": True,
            },
            "tws_verify": {
                "type": "boolean",
                "label": "Verificar SSL",
                "hot_reload": False,
                "restart_reason": "SSL é configurado na criação do cliente HTTP",
            },
        },
    },
    "tws_monitoring": {
        "title": "TWS Monitoring",
        "icon": "fa-eye",
        "description": "Monitoramento proativo do TWS",
        "fields": {
            "tws_polling_enabled": {
                "type": "boolean",
                "label": "Polling Habilitado",
                "hot_reload": True,
            },
            "tws_polling_interval_seconds": {
                "type": "number",
                "label": "Intervalo (s)",
                "min": 5,
                "max": 300,
                "hot_reload": True,
            },
            "tws_polling_mode": {
                "type": "select",
                "options": ["fixed", "adaptive", "scheduled"],
                "label": "Modo",
                "hot_reload": True,
            },
            "tws_job_stuck_threshold_minutes": {
                "type": "number",
                "label": "Threshold Job Stuck (min)",
                "min": 10,
                "hot_reload": True,
            },
            "tws_job_late_threshold_minutes": {
                "type": "number",
                "label": "Threshold Job Atrasado (min)",
                "min": 5,
                "hot_reload": True,
            },
            "tws_anomaly_failure_rate_threshold": {
                "type": "number",
                "label": "Threshold Anomalia",
                "min": 0.01,
                "max": 1,
                "step": 0.01,
                "hot_reload": True,
            },
            "tws_pattern_detection_enabled": {
                "type": "boolean",
                "label": "Detecção de Padrões",
                "hot_reload": True,
            },
            "tws_pattern_min_confidence": {
                "type": "number",
                "label": "Confiança Mínima",
                "min": 0.1,
                "max": 1,
                "step": 0.1,
                "hot_reload": True,
            },
            "tws_solution_correlation_enabled": {
                "type": "boolean",
                "label": "Correlação de Soluções",
                "hot_reload": True,
            },
        },
    },
    "tws_retention": {
        "title": "TWS Data Retention",
        "icon": "fa-archive",
        "description": "Retenção de dados do TWS",
        "fields": {
            "tws_retention_days_full": {
                "type": "number",
                "label": "Dados Completos (dias)",
                "min": 1,
                "max": 30,
                "hot_reload": True,
            },
            "tws_retention_days_summary": {
                "type": "number",
                "label": "Sumários (dias)",
                "min": 7,
                "max": 90,
                "hot_reload": True,
            },
            "tws_retention_days_patterns": {
                "type": "number",
                "label": "Padrões (dias)",
                "min": 30,
                "max": 365,
                "hot_reload": True,
            },
        },
    },
    "tws_notifications": {
        "title": "TWS Notifications",
        "icon": "fa-bell",
        "description": "Notificações do TWS",
        "fields": {
            "tws_browser_notifications_enabled": {
                "type": "boolean",
                "label": "Notificações Browser",
                "hot_reload": True,
            },
            "tws_teams_notifications_enabled": {
                "type": "boolean",
                "label": "Notificações Teams",
                "hot_reload": True,
            },
            "tws_teams_webhook_url": {
                "type": "text",
                "label": "Teams Webhook URL",
                "hot_reload": True,
            },
        },
    },
    "tws_dashboard": {
        "title": "TWS Dashboard",
        "icon": "fa-tachometer-alt",
        "description": "Configurações do dashboard",
        "fields": {
            "tws_dashboard_theme": {
                "type": "select",
                "options": ["auto", "light", "dark"],
                "label": "Tema",
                "hot_reload": True,
            },
            "tws_dashboard_refresh_seconds": {
                "type": "number",
                "label": "Refresh (s)",
                "min": 1,
                "max": 60,
                "hot_reload": True,
            },
        },
    },
    "hybrid_retriever": {
        "title": "Hybrid Retriever",
        "icon": "fa-search",
        "description": "Configurações de busca híbrida (BM25 + Vector)",
        "fields": {
            "hybrid_vector_weight": {
                "type": "number",
                "label": "Peso Vetorial",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "hot_reload": True,
            },
            "hybrid_bm25_weight": {
                "type": "number",
                "label": "Peso BM25",
                "min": 0,
                "max": 1,
                "step": 0.1,
                "hot_reload": True,
            },
            "hybrid_auto_weight": {
                "type": "boolean",
                "label": "Auto-Peso",
                "hot_reload": True,
            },
            "hybrid_boost_job_name": {
                "type": "number",
                "label": "Boost Job Name",
                "min": 0,
                "max": 10,
                "step": 0.5,
                "hot_reload": True,
            },
            "hybrid_boost_error_code": {
                "type": "number",
                "label": "Boost Error Code",
                "min": 0,
                "max": 10,
                "step": 0.5,
                "hot_reload": True,
            },
            "hybrid_boost_workstation": {
                "type": "number",
                "label": "Boost Workstation",
                "min": 0,
                "max": 10,
                "step": 0.5,
                "hot_reload": True,
            },
        },
    },
    "cache": {
        "title": "Cache TTL",
        "icon": "fa-clock",
        "description": "Tempos de expiração de cache",
        "fields": {
            "cache_ttl_job_status": {
                "type": "number",
                "label": "Job Status (s)",
                "min": 5,
                "max": 60,
                "hot_reload": True,
            },
            "cache_ttl_job_logs": {
                "type": "number",
                "label": "Job Logs (s)",
                "min": 10,
                "max": 120,
                "hot_reload": True,
            },
            "cache_ttl_static_structure": {
                "type": "number",
                "label": "Estrutura Estática (s)",
                "min": 300,
                "max": 86400,
                "hot_reload": True,
            },
            "cache_ttl_graph": {
                "type": "number",
                "label": "Grafo (s)",
                "min": 60,
                "max": 3600,
                "hot_reload": True,
            },
            "cache_hierarchy_l1_max_size": {
                "type": "number",
                "label": "L1 Max Size",
                "min": 100,
                "hot_reload": False,
                "restart_reason": "Cache L1 é inicializado com tamanho fixo",
            },
            "cache_hierarchy_l2_ttl": {
                "type": "number",
                "label": "L2 TTL (s)",
                "min": 60,
                "hot_reload": True,
            },
            "enable_cache_swr": {
                "type": "boolean",
                "label": "Stale-While-Revalidate",
                "hot_reload": True,
            },
            "enable_cache_mutex": {
                "type": "boolean",
                "label": "Cache Mutex",
                "hot_reload": True,
            },
        },
    },
    "security": {
        "title": "Segurança",
        "icon": "fa-shield-alt",
        "description": "Configurações de segurança",
        "fields": {
            "jwt_algorithm": {
                "type": "text",
                "label": "Algoritmo JWT",
                "readonly": True,
            },
            "access_token_expire_minutes": {
                "type": "number",
                "label": "Token Expiration (min)",
                "min": 5,
                "max": 1440,
                "hot_reload": True,
            },
            "session_timeout_minutes": {
                "type": "number",
                "label": "Session Timeout (min)",
                "min": 5,
                "max": 480,
                "hot_reload": True,
            },
            "session_secure_cookie": {
                "type": "boolean",
                "label": "Secure Cookie",
                "hot_reload": False,
                "restart_reason": "Configuração de cookie é definida na inicialização do middleware",
            },
            "session_http_only": {
                "type": "boolean",
                "label": "HTTP Only Cookie",
                "hot_reload": False,
                "restart_reason": "Configuração de cookie é definida na inicialização",
            },
            "session_same_site": {
                "type": "select",
                "options": ["strict", "lax", "none"],
                "label": "SameSite Policy",
                "hot_reload": False,
                "restart_reason": "Política de cookie é definida na inicialização",
            },
            "enforce_https": {
                "type": "boolean",
                "label": "Forçar HTTPS",
                "hot_reload": False,
                "restart_reason": "HSTS é configurado na inicialização do middleware",
            },
            "ssl_redirect": {
                "type": "boolean",
                "label": "Redirect HTTP→HTTPS",
                "hot_reload": False,
                "restart_reason": "Redirect é configurado na inicialização",
            },
        },
    },
    "rate_limiting": {
        "title": "Rate Limiting",
        "icon": "fa-tachometer-alt",
        "description": "Limites de requisições",
        "fields": {
            "rate_limit_public_per_minute": {
                "type": "number",
                "label": "Público/min",
                "min": 1,
                "hot_reload": True,
            },
            "rate_limit_authenticated_per_minute": {
                "type": "number",
                "label": "Autenticado/min",
                "min": 1,
                "hot_reload": True,
            },
            "rate_limit_critical_per_minute": {
                "type": "number",
                "label": "Crítico/min",
                "min": 1,
                "hot_reload": True,
            },
            "rate_limit_websocket_per_minute": {
                "type": "number",
                "label": "WebSocket/min",
                "min": 1,
                "hot_reload": True,
            },
            "rate_limit_dashboard_per_minute": {
                "type": "number",
                "label": "Dashboard/min",
                "min": 1,
                "hot_reload": True,
            },
            "rate_limit_sliding_window": {
                "type": "boolean",
                "label": "Sliding Window",
                "hot_reload": False,
                "restart_reason": "Tipo de janela é configurado na inicialização do rate limiter",
            },
        },
    },
    "compression": {
        "title": "Compressão",
        "icon": "fa-compress",
        "description": "Configurações de compressão HTTP",
        "fields": {
            "compression_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": False,
                "restart_reason": "Middleware de compressão é adicionado na inicialização",
            },
            "compression_minimum_size": {
                "type": "number",
                "label": "Tamanho Mínimo (bytes)",
                "min": 0,
                "hot_reload": False,
                "restart_reason": "Configuração do middleware de compressão",
            },
            "compression_level": {
                "type": "number",
                "label": "Nível (1-9)",
                "min": 1,
                "max": 9,
                "hot_reload": False,
                "restart_reason": "Nível de compressão é definido na inicialização",
            },
        },
    },
    "cors": {
        "title": "CORS",
        "icon": "fa-globe",
        "description": "Cross-Origin Resource Sharing",
        "fields": {
            "cors_allow_credentials": {
                "type": "boolean",
                "label": "Allow Credentials",
                "hot_reload": False,
                "restart_reason": "CORS middleware é configurado na inicialização",
            },
        },
    },
    "uploads": {
        "title": "Uploads",
        "icon": "fa-upload",
        "description": "Configurações de upload de arquivos",
        "fields": {
            "max_file_size": {
                "type": "number",
                "label": "Tamanho Máximo (bytes)",
                "min": 1024,
                "hot_reload": True,
            },
        },
    },
    "backup": {
        "title": "Backup",
        "icon": "fa-hdd",
        "description": "Configurações de backup automático",
        "fields": {
            "backup_enabled": {
                "type": "boolean",
                "label": "Habilitado",
                "hot_reload": True,
            },
            "backup_retention_days": {
                "type": "number",
                "label": "Retenção (dias)",
                "min": 1,
                "max": 365,
                "hot_reload": True,
            },
            "backup_schedule_cron": {
                "type": "text",
                "label": "Schedule (cron)",
                "hot_reload": False,
                "restart_reason": "Schedule de backup requer reiniciar o scheduler",
            },
            "backup_include_database": {
                "type": "boolean",
                "label": "Incluir Database",
                "hot_reload": True,
            },
            "backup_include_uploads": {
                "type": "boolean",
                "label": "Incluir Uploads",
                "hot_reload": True,
            },
            "backup_include_config": {
                "type": "boolean",
                "label": "Incluir Config",
                "hot_reload": True,
            },
            "backup_compression": {
                "type": "boolean",
                "label": "Comprimir",
                "hot_reload": True,
            },
        },
    },
    "rag_service": {
        "title": "RAG Service",
        "icon": "fa-book",
        "description": "Microserviço RAG",
        "fields": {
            "rag_service_url": {"type": "text", "label": "URL", "hot_reload": True},
            "rag_service_timeout": {
                "type": "number",
                "label": "Timeout (s)",
                "min": 1,
                "hot_reload": True,
            },
            "rag_service_max_retries": {
                "type": "number",
                "label": "Max Retries",
                "min": 0,
                "hot_reload": True,
            },
            "rag_service_retry_backoff": {
                "type": "number",
                "label": "Retry Backoff",
                "min": 0,
                "step": 0.1,
                "hot_reload": True,
            },
        },
    },
    "enterprise": {
        "title": "Enterprise Features",
        "icon": "fa-building",
        "description": "Módulos enterprise",
        "fields": {
            "enterprise_enable_incident_response": {
                "type": "boolean",
                "label": "Incident Response",
                "hot_reload": False,
                "restart_reason": "Módulo de incidentes é inicializado no boot",
            },
            "enterprise_enable_auto_recovery": {
                "type": "boolean",
                "label": "Auto Recovery",
                "hot_reload": False,
                "restart_reason": "Auto-recovery requer inicialização de handlers",
            },
            "enterprise_enable_runbooks": {
                "type": "boolean",
                "label": "Runbooks",
                "hot_reload": True,
            },
            "enterprise_enable_gdpr": {
                "type": "boolean",
                "label": "GDPR Compliance",
                "hot_reload": False,
                "restart_reason": "GDPR requer inicialização de data handlers",
            },
            "enterprise_enable_encrypted_audit": {
                "type": "boolean",
                "label": "Encrypted Audit",
                "hot_reload": False,
                "restart_reason": "Criptografia de audit requer inicialização de chaves",
            },
            "enterprise_enable_siem": {
                "type": "boolean",
                "label": "SIEM Integration",
                "hot_reload": False,
                "restart_reason": "SIEM requer conexão inicial",
            },
            "enterprise_enable_log_aggregator": {
                "type": "boolean",
                "label": "Log Aggregator",
                "hot_reload": False,
                "restart_reason": "Log aggregator é inicializado no boot",
            },
            "enterprise_enable_anomaly_detection": {
                "type": "boolean",
                "label": "Anomaly Detection",
                "hot_reload": True,
            },
            "enterprise_anomaly_sensitivity": {
                "type": "number",
                "label": "Sensibilidade Anomalia",
                "min": 0,
                "max": 1,
                "step": 0.05,
                "hot_reload": True,
            },
            "enterprise_enable_chaos_engineering": {
                "type": "boolean",
                "label": "Chaos Engineering",
                "warning": "Apenas staging!",
                "hot_reload": False,
                "restart_reason": "Chaos engineering requer inicialização segura",
            },
            "enterprise_enable_service_discovery": {
                "type": "boolean",
                "label": "Service Discovery",
                "hot_reload": False,
                "restart_reason": "Service discovery requer registro inicial",
            },
        },
    },
}
# =============================================================================
# SCHEMA AUTO-EXTENSION
# =============================================================================

def _infer_field_schema(key: str, field: Any) -> dict[str, Any]:
    """Infer a UI schema entry from a Pydantic Settings field.

    We keep this conservative: default hot_reload=False to avoid runtime surprises.
    """
    from typing import get_origin, get_args, Literal
    ann = field.annotation
    origin = get_origin(ann)

    # Secret-like fields (SecretStr, SecretBytes) -> password input
    if hasattr(ann, "__name__") and ann.__name__ in {"SecretStr", "SecretBytes"}:
        return {"type": "password", "label": key.replace("_", " ").title(), "hot_reload": False}

    # Literal -> select
    if origin is Literal:
        options = [str(v) for v in get_args(ann)]
        return {"type": "select", "options": options, "label": key.replace("_", " ").title(), "hot_reload": False}

    # bool / int / float / list[str] / Path
    if ann is bool:
        return {"type": "boolean", "label": key.replace("_", " ").title(), "hot_reload": False}
    if ann in (int, float):
        return {"type": "number", "label": key.replace("_", " ").title(), "hot_reload": False}
    if origin in (list, tuple, set):
        return {"type": "textarea", "label": key.replace("_", " ").title(), "hot_reload": False}
    if getattr(ann, "__name__", "") == "Path":
        return {"type": "text", "label": key.replace("_", " ").title(), "hot_reload": False}

    # default
    return {"type": "text", "label": key.replace("_", " ").title(), "hot_reload": False}


def _select_group_for_setting_key(key: str) -> str:
    """Map a setting key to an existing UI group.

    Groups are aligned with the UI recommendation:
      - server: WebSocket + HTTP pool
      - cache: cache hierarchy / robust cache
      - tws: Teams/TWS
      - cors: Security/Network (CORS)
    Everything else goes to 'advanced'.
    """
    if key.startswith(("ws_", "http_pool_")):
        return "server"
    if key.startswith(("cache_", "robust_cache_")):
        return "cache"
    if key.startswith("tws_"):
        return "tws"
    if key.startswith("cors_"):
        return "cors"
    if key.startswith("redis_"):
        return "redis"
    if key.startswith(("db_", "database_")):
        return "database"
    if key.startswith("langfuse_"):
        return "langfuse"
    if key.startswith(("ollama_",)):
        return "ollama"
    if key.startswith(("llm_", "litellm_")):
        return "llm"
    if key.startswith("hybrid_"):
        return "hybrid_retriever"
    if key.startswith("rate_limit_") or key.startswith("rate_limiting_"):
        return "rate_limiting"
    if key.startswith("upload_"):
        return "uploads"
    if key.startswith("backup_"):
        return "backup"
    if key.startswith("enterprise_"):
        return "enterprise"
    if key.startswith("security_") or key.startswith("jwt_") or key.startswith("auth_"):
        return "security"
    return "advanced"


def _extend_settings_schema_with_all_settings() -> None:
    """Ensure the Admin UI exposes *all* Settings keys.

    Any Settings fields not present in SETTINGS_SCHEMA will be added to a
    conservative 'advanced' group or inferred group based on prefix.

    This is safe by default:
      - hot_reload defaults to False (restart required) unless already specified.
      - Secret values remain masked by existing API logic.
    """
    try:
        from resync.settings import Settings
    except Exception:
        return

    all_keys = list(getattr(Settings, "model_fields", {}).keys())
    if not all_keys:
        return

    # Ensure Advanced group exists
    if "advanced" not in SETTINGS_SCHEMA:
        SETTINGS_SCHEMA["advanced"] = {
            "title": "Advanced",
            "icon": "fa-sliders",
            "description": "Configurações avançadas (auto-geradas) para cobrir todos os campos do Settings.",
            "fields": {},
        }

    existing = set()
    for grp in SETTINGS_SCHEMA.values():
        existing.update(grp.get("fields", {}).keys())

    for key in all_keys:
        if key in existing:
            continue

        group_key = _select_group_for_setting_key(key)
        if group_key not in SETTINGS_SCHEMA:
            group_key = "advanced"

        field = Settings.model_fields.get(key)
        if field is None:
            continue

        entry = _infer_field_schema(key, field)

        # A few safe hot-reload knobs (read at runtime frequently)
        if key in {"ws_connection_timeout", "strict_exception_handling", "programming_error_metrics_detailed_site"}:
            entry["hot_reload"] = True

        # Provide minimal descriptions for some secret-ish keys
        if entry.get("type") == "password":
            entry.setdefault("description", "Campo sensível: será armazenado como override e exibido mascarado.")
            entry.setdefault("hot_reload", False)

        SETTINGS_SCHEMA[group_key]["fields"][key] = entry


_extend_settings_schema_with_all_settings()

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SettingUpdate(BaseModel):
    """Model for updating a single setting."""

    key: str = Field(..., description="Setting key (e.g., 'llm_timeout')")
    value: Any = Field(..., description="New value for the setting")

class BulkSettingsUpdate(BaseModel):
    """Model for updating multiple settings at once."""

    settings: dict[str, Any] = Field(
        ..., description="Dictionary of settings to update"
    )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_overrides_sync() -> dict[str, Any]:
    """Load settings overrides from JSON file (blocking)."""
    if SETTINGS_OVERRIDE_PATH.exists():
        try:
            with SETTINGS_OVERRIDE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            TimeoutError,
            ConnectionError,
        ) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error

            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("Failed to load settings overrides: %s", e)
    return {}


async def _load_overrides() -> dict[str, Any]:
    """Load settings overrides without blocking the event loop."""
    return await asyncio.to_thread(_load_overrides_sync)


def _save_overrides_sync(overrides: dict[str, Any]) -> None:
    """Save settings overrides to JSON file (blocking).

    Resilience/Security:
    - Atomic write via temp file + os.replace (prevents partial writes)
    - Restrictive permissions (0o600) for secrets-at-rest defense in depth
    """
    try:
        SETTINGS_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        overrides["_last_modified"] = datetime.now(timezone.utc).isoformat()

        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="settings_override_",
            suffix=".json",
            dir=str(SETTINGS_OVERRIDE_PATH.parent),
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, SETTINGS_OVERRIDE_PATH)
        finally:
            if os.path.exists(tmp_path) and tmp_path != str(SETTINGS_OVERRIDE_PATH):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Ensure target file permissions (in case it pre-existed)
        try:
            os.chmod(SETTINGS_OVERRIDE_PATH, 0o600)
        except OSError:
            pass

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        TimeoutError,
        ConnectionError,
    ) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error

        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Failed to save settings overrides: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save settings. Check server logs for details.",
        ) from None


async def _save_overrides(overrides: dict[str, Any]) -> None:
    """Save settings overrides without blocking the event loop."""
    await asyncio.to_thread(_save_overrides_sync, overrides)



def _get_current_value(settings: Settings, key: str, overrides: dict[str, Any]) -> Any:
    """Get current value for a setting, checking overrides first."""
    if key in overrides:
        return overrides[key]

    try:
        value = getattr(settings, key, None)
        # Handle SecretStr
        if hasattr(value, "get_secret_value"):
            return "********"  # Don't expose secrets
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
        return None

def _get_all_settings_values(settings: Settings, overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Get all current settings values organized by section."""
    result = {}

    for section_key, section in SETTINGS_SCHEMA.items():
        section_values = {}
        for field_key, field_config in section["fields"].items():  # type: ignore[attr-defined]
            # Check override first
            if field_key in overrides:
                section_values[field_key] = overrides[field_key]
            else:
                try:
                    value = getattr(settings, field_key, None)
                    # Handle SecretStr
                    if hasattr(value, "get_secret_value"):
                        section_values[field_key] = "********"
                    elif isinstance(value, Path):
                        section_values[field_key] = str(value)
                    else:
                        section_values[field_key] = value
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
                    section_values[field_key] = None
        result[section_key] = section_values

    return result

# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/schema")
async def get_settings_schema() -> dict[str, Any]:
    """Get the complete settings schema with field definitions.

    Returns the schema that describes all configurable settings,
    organized by section with field types, labels, and constraints.
    """
    return {
        "schema": SETTINGS_SCHEMA,
        "total_sections": len(SETTINGS_SCHEMA),
        "total_fields": sum(len(s["fields"]) for s in SETTINGS_SCHEMA.values()),
    }

@router.get("/all")
async def get_all_settings() -> dict[str, Any]:
    """Get all current settings values organized by section.

    Returns current values for all settings, with sensitive
    values (passwords, API keys) masked.
    """
    settings = get_settings()
    overrides = await _load_overrides()
    values = _get_all_settings_values(settings, overrides)

    return {
        "values": values,
        "schema": SETTINGS_SCHEMA,
        "overrides_count": len([k for k in overrides if not k.startswith("_")]),
        "last_modified": overrides.get("_last_modified"),
    }

@router.get("/section/{section_key}")
async def get_section_settings(section_key: str) -> dict[str, Any]:
    """Get settings for a specific section.

    Args:
        section_key: The section identifier (e.g., 'llm', 'tws', 'redis')

    Returns:
        Section schema and current values.
    """
    if section_key not in SETTINGS_SCHEMA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section '{section_key}' not found",
        )

    settings = get_settings()
    section = SETTINGS_SCHEMA[section_key]
    values = {}
    overrides = await _load_overrides()

    for field_key in section["fields"]:
        if field_key in overrides:
            values[field_key] = overrides[field_key]
        else:
            try:
                value = getattr(settings, field_key, None)
                if hasattr(value, "get_secret_value"):
                    values[field_key] = "********"
                elif isinstance(value, Path):
                    values[field_key] = str(value)
                else:
                    values[field_key] = value
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):
                values[field_key] = None

    return {
        "section": section_key,
        "title": section["title"],
        "icon": section["icon"],
        "description": section["description"],
        "fields": section["fields"],
        "values": values,
    }

@router.put("/update")
async def update_setting(update: SettingUpdate) -> dict[str, Any]:
    """Update a single setting.

    Settings marked with hot_reload=True will take effect immediately.
    Settings marked with hot_reload=False require application restart.

    Args:
        update: The setting key and new value.

    Returns:
        Confirmation with old and new values, and restart requirements.
    """
    from resync.settings import clear_settings_cache

    settings = get_settings()

    # Validate key exists in schema
    found = False
    hot_reload = True
    restart_reason = None

    for section in SETTINGS_SCHEMA.values():  # type: ignore[attr-defined]
        if update.key in section["fields"]:  # type: ignore[index]
            field_config = section["fields"][update.key]
            found = True
            hot_reload = field_config.get("hot_reload", True)
            restart_reason = field_config.get("restart_reason")
            break

    if not found:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown setting key: {update.key}",
        )

    # Check if readonly
    if field_config.get("readonly"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Setting '{update.key}' is read-only",
        )

    overrides = await _load_overrides()

    # Get old value
    old_value = _get_current_value(settings, update.key, overrides)

    # Save to overrides
    overrides[update.key] = update.value
    await _save_overrides(overrides)
    # Also update environment variable for immediate effect on some settings
    env_key = f"APP_{update.key.upper()}"
    os.environ[env_key] = str(update.value)

    # Clear settings cache if hot-reload is supported
    if hot_reload:
        try:
            clear_settings_cache()
            logger.info("Setting hot-reloaded: %s = %s", update.key, update.value)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.warning("Could not clear settings cache: %s", e)
    else:
        logger.info(
            "Setting updated (requires restart): %s = %s", update.key, update.value
        )

    return {
        "success": True,
        "key": update.key,
        "old_value": old_value,
        "new_value": update.value,
        "hot_reload": hot_reload,
        "requires_restart": not hot_reload,
        "restart_reason": restart_reason if not hot_reload else None,
        "message": "Aplicado imediatamente"
        if hot_reload
        else f"Requer restart: {restart_reason}",
    }

@router.put("/bulk-update")
async def bulk_update_settings(update: BulkSettingsUpdate) -> dict[str, Any]:
    """Update multiple settings at once.

    Settings marked with hot_reload=True will take effect immediately.
    Settings marked with hot_reload=False require application restart.

    Args:
        update: Dictionary of settings to update.

    Returns:
        Summary of updated settings with restart requirements.
    """
    from resync.settings import clear_settings_cache

    overrides = await _load_overrides()
    updated = []
    errors = []
    requires_restart = []
    hot_reloaded = []

    for key, value in update.settings.items():
        # Validate key exists
        found = False
        readonly = False
        hot_reload = True
        restart_reason = None

        for section in SETTINGS_SCHEMA.values():  # type: ignore[attr-defined]
            if key in section["fields"]:  # type: ignore[index]
                found = True
                field_config = section["fields"][key]
                readonly = field_config.get("readonly", False)
                hot_reload = field_config.get("hot_reload", True)
                restart_reason = field_config.get("restart_reason")
                break

        if not found:
            errors.append(f"Unknown key: {key}")
            continue

        if readonly:
            errors.append(f"Read-only: {key}")
            continue

        overrides[key] = value
        os.environ[f"APP_{key.upper()}"] = str(value)
        updated.append(key)

        if hot_reload:
            hot_reloaded.append(key)
        else:
            requires_restart.append(
                {"key": key, "reason": restart_reason or "Requer restart"}
            )

    await _save_overrides(overrides)
    # Clear settings cache to force reload on next access
    try:
        clear_settings_cache()
        logger.info("Settings cache cleared after updating %s settings", len(updated))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.warning("Could not clear settings cache: %s", e)

    return {
        "success": len(errors) == 0,
        "updated": updated,
        "updated_count": len(updated),
        "hot_reloaded": hot_reloaded,
        "hot_reloaded_count": len(hot_reloaded),
        "requires_restart": requires_restart,
        "requires_restart_count": len(requires_restart),
        "errors": errors,
        "message": _build_update_message(
            len(hot_reloaded), len(requires_restart), len(errors)
        ),
    }

def _build_update_message(hot_count: int, restart_count: int, error_count: int) -> str:
    """Build a human-readable update message."""
    parts = []
    if hot_count > 0:
        parts.append(f"{hot_count} aplicadas imediatamente")
    if restart_count > 0:
        parts.append(f"{restart_count} requerem restart")
    if error_count > 0:
        parts.append(f"{error_count} erros")
    return ", ".join(parts) if parts else "Nenhuma alteração"

@router.delete("/reset/{key}")
async def reset_setting(key: str) -> dict[str, Any]:
    """Reset a setting to its default value.

    Removes the override for the specified setting, allowing
    the default value from Settings class to be used.

    Args:
        key: The setting key to reset.
    """
    overrides = await _load_overrides()

    if key not in overrides:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No override found for '{key}'",
        )

    del overrides[key]
    await _save_overrides(overrides)
    # Remove from environment
    env_key = f"APP_{key.upper()}"
    if env_key in os.environ:
        del os.environ[env_key]

    return {
        "success": True,
        "key": key,
        "message": "Setting reset to default",
    }

@router.delete("/reset-all")
async def reset_all_settings() -> dict[str, Any]:
    """Reset all settings to their default values.

    Clears all overrides, reverting to defaults from Settings class.
    """
    overrides = await _load_overrides()
    count = len([k for k in overrides if not k.startswith("_")])

    # Clear overrides file
    await _save_overrides({"_last_modified": datetime.now(timezone.utc).isoformat()})
    return {
        "success": True,
        "reset_count": count,
        "message": f"Reset {count} settings to defaults",
    }

@router.get("/export")
async def export_settings() -> dict[str, Any]:
    """Export all current settings as JSON.

    Useful for backup or migration purposes.
    Sensitive values are masked.
    """
    settings = get_settings()
    overrides = await _load_overrides()
    values = _get_all_settings_values(settings, overrides)

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "version": settings.project_version,
        "environment": settings.environment.value
        if hasattr(settings.environment, "value")
        else str(settings.environment),
        "settings": values,
    }

@router.post("/import")
async def import_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Import settings from a JSON export.

    Validates and applies settings from a previous export.

    Args:
        data: The exported settings data.
    """
    if "settings" not in data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid import format: missing 'settings' key",
        )

    overrides = await _load_overrides()
    imported = 0

    for section_key, section_values in data["settings"].items():
        if section_key not in SETTINGS_SCHEMA:
            continue

        for key, value in section_values.items():
            if key in SETTINGS_SCHEMA[section_key]["fields"]:
                # Skip masked values
                if value == "********":
                    continue
                overrides[key] = value
                imported += 1

    await _save_overrides(overrides)
    return {
        "success": True,
        "imported_count": imported,
        "message": f"Imported {imported} settings",
    }