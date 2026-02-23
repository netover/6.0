"""
Configurações de Monitoramento Proativo

Este módulo define todas as configurações relacionadas ao
monitoramento proativo do TWS, incluindo polling, alertas e retenção.

Autor: Resync Team
Versão: 5.2
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PollingMode(str, Enum):
    """Modos de polling."""

    FIXED = "fixed"  # Intervalo fixo
    ADAPTIVE = "adaptive"  # Adapta baseado em atividade
    SCHEDULED = "scheduled"  # Horários específicos


class MonitoringConfig(BaseModel):
    """Configurações de monitoramento proativo."""

    # =========================================================================
    # POLLING
    # =========================================================================

    # Intervalo base de polling em segundos (5s a 300s)
    polling_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Intervalo de polling em segundos",
    )

    # Modo de polling
    polling_mode: PollingMode = Field(
        default=PollingMode.FIXED,
        description="Modo de polling: fixed, adaptive, scheduled",
    )

    # Polling adaptativo: intervalo mínimo durante alta atividade
    polling_min_interval: int = Field(
        default=10,
        ge=5,
        description="Intervalo mínimo durante alta atividade (modo adaptive)",
    )

    # Polling adaptativo: intervalo máximo durante baixa atividade
    polling_max_interval: int = Field(
        default=120,
        le=300,
        description="Intervalo máximo durante baixa atividade (modo adaptive)",
    )

    # Horários de polling agendado (modo scheduled)
    # Formato: "HH:MM" ou "HH:MM-HH:MM" para intervalos
    polling_schedule: list[str] = Field(
        default=["06:00-22:00"],
        description="Horários de polling ativo (modo scheduled)",
    )

    # Polling fora do horário agendado
    off_schedule_polling_interval: int = Field(
        default=300,
        description="Intervalo de polling fora do horário agendado",
    )

    # =========================================================================
    # ALERTAS
    # =========================================================================

    # Habilita alertas
    alerts_enabled: bool = Field(
        default=True,
        description="Habilita geração de alertas",
    )

    # Limiar para job stuck (minutos)
    job_stuck_threshold_minutes: int = Field(
        default=60,
        ge=10,
        description="Minutos para considerar job stuck",
    )

    # Limiar para job atrasado (minutos)
    job_late_threshold_minutes: int = Field(
        default=30,
        ge=5,
        description="Minutos para considerar job atrasado",
    )

    # Taxa de falha para alerta de anomalia (0.0 a 1.0)
    anomaly_failure_rate_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Taxa de falha para gerar alerta de anomalia",
    )

    # Mínimo de workstations offline para alerta crítico
    critical_ws_offline_threshold: int = Field(
        default=2,
        ge=1,
        description="Workstations offline para alerta crítico",
    )

    # =========================================================================
    # NOTIFICAÇÕES
    # =========================================================================

    # Habilita notificações browser (Web Push)
    browser_notifications_enabled: bool = Field(
        default=True,
        description="Habilita notificações no browser",
    )

    # Habilita notificações Teams
    teams_notifications_enabled: bool = Field(
        default=True,
        description="Habilita notificações no Microsoft Teams",
    )

    # Webhook URL para Teams
    teams_webhook_url: str | None = Field(
        default=None,
        description="URL do webhook do Microsoft Teams",
    )

    # Severidades que geram notificação
    notification_severities: list[str] = Field(
        default=["critical", "error"],
        description="Severidades que geram notificação push",
    )

    # Notificação para job ABEND
    notify_abend: bool = Field(
        default=True,
        description="Notificar quando job entrar em ABEND",
    )

    # Notificação para workstation offline
    notify_ws_offline: bool = Field(
        default=True,
        description="Notificar quando workstation ficar offline",
    )

    # Notificação para job travado
    notify_stuck: bool = Field(
        default=False,
        description="Notificar quando job ficar travado",
    )

    # Notificação para padrão detectado
    notify_pattern: bool = Field(
        default=False,
        description="Notificar quando padrão for detectado",
    )

    # Habilita alerta sonoro
    sound_enabled: bool = Field(
        default=False,
        description="Habilita alerta sonoro para eventos críticos",
    )

    # =========================================================================
    # WEBSOCKET
    # =========================================================================

    # Habilita WebSocket broadcast
    websocket_enabled: bool = Field(
        default=True,
        description="Habilita broadcast de eventos via WebSocket",
    )

    # Filtro: eventos de jobs
    ws_filter_jobs: bool = Field(
        default=True,
        description="Broadcast eventos de jobs",
    )

    # Filtro: eventos de workstations
    ws_filter_workstations: bool = Field(
        default=True,
        description="Broadcast eventos de workstations",
    )

    # Filtro: eventos de sistema
    ws_filter_system: bool = Field(
        default=True,
        description="Broadcast eventos de sistema",
    )

    # Filtro: apenas críticos
    ws_filter_critical_only: bool = Field(
        default=False,
        description="Broadcast apenas eventos críticos",
    )

    # Severidade mínima para broadcast
    ws_min_severity: str = Field(
        default="info",
        description="Severidade mínima para broadcast: info, warning, error, critical",
    )

    # Regex para filtrar jobs
    ws_job_filter_regex: str = Field(
        default="",
        description="Regex para filtrar jobs (vazio = todos)",
    )

    # =========================================================================
    # RETENÇÃO DE DADOS
    # =========================================================================

    # Dias para reter dados completos (jobs, workstations)
    retention_days_full: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Dias para reter dados completos",
    )

    # Dias para reter sumários e eventos
    retention_days_summary: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Dias para reter sumários e eventos",
    )

    # Dias para reter padrões detectados
    retention_days_patterns: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Dias para reter padrões",
    )

    # =========================================================================
    # DETECÇÃO DE PADRÕES
    # =========================================================================

    # Habilita detecção de padrões
    pattern_detection_enabled: bool = Field(
        default=True,
        description="Habilita detecção automática de padrões",
    )

    # Intervalo de execução da detecção de padrões (minutos)
    pattern_detection_interval_minutes: int = Field(
        default=60,
        ge=15,
        description="Intervalo para rodar detecção de padrões",
    )

    # Confiança mínima para reportar padrão
    pattern_min_confidence: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Confiança mínima para reportar padrão",
    )

    # =========================================================================
    # APRENDIZADO
    # =========================================================================

    # Habilita correlação problema-solução
    solution_correlation_enabled: bool = Field(
        default=True,
        description="Habilita sugestão de soluções baseada em histórico",
    )

    # Taxa de sucesso mínima para sugerir solução
    solution_min_success_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Taxa de sucesso mínima para sugerir solução",
    )

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    # Tema do dashboard
    dashboard_theme: str = Field(
        default="auto",  # auto, light, dark
        description="Tema do dashboard: auto, light, dark",
    )

    # Atualização automática do dashboard (segundos)
    dashboard_refresh_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Intervalo de refresh do dashboard",
    )

    # Quantidade de eventos a mostrar
    dashboard_events_limit: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Quantidade máxima de eventos no dashboard",
    )

    # =========================================================================
    # MÉTODOS
    # =========================================================================

    def to_frontend_config(self) -> dict[str, Any]:
        """Retorna configurações relevantes para o frontend."""
        return {
            "polling": {
                "enabled": True,
                "interval": self.polling_interval_seconds,
                "mode": self.polling_mode.value
                if isinstance(self.polling_mode, PollingMode)
                else self.polling_mode,
                "schedule": self.polling_schedule,
            },
            "alerts": {
                "enabled": self.alerts_enabled,
                "jobStuckMinutes": self.job_stuck_threshold_minutes,
                "jobLateMinutes": self.job_late_threshold_minutes,
                "anomalyThreshold": self.anomaly_failure_rate_threshold,
            },
            "notifications": {
                "browser": self.browser_notifications_enabled,
                "teams": self.teams_notifications_enabled,
                "severities": self.notification_severities,
                "notifyAbend": self.notify_abend,
                "notifyWsOffline": self.notify_ws_offline,
                "notifyStuck": self.notify_stuck,
                "notifyPattern": self.notify_pattern,
                "soundEnabled": self.sound_enabled,
            },
            "websocket": {
                "enabled": self.websocket_enabled,
                "filterJobs": self.ws_filter_jobs,
                "filterWorkstations": self.ws_filter_workstations,
                "filterSystem": self.ws_filter_system,
                "filterCriticalOnly": self.ws_filter_critical_only,
                "minSeverity": self.ws_min_severity,
                "jobFilterRegex": self.ws_job_filter_regex,
            },
            "retention": {
                "fullDays": self.retention_days_full,
                "summaryDays": self.retention_days_summary,
                "patternsDays": self.retention_days_patterns,
            },
            "dashboard": {
                "theme": self.dashboard_theme,
                "refreshSeconds": self.dashboard_refresh_seconds,
                "eventsLimit": self.dashboard_events_limit,
            },
            "patterns": {
                "enabled": self.pattern_detection_enabled,
                "intervalMinutes": self.pattern_detection_interval_minutes,
                "minConfidence": self.pattern_min_confidence,
            },
        }

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_MONITORING_CONFIG = MonitoringConfig()


def get_monitoring_config() -> MonitoringConfig:
    """Retorna configuração de monitoramento atual."""
    # Em produção, isso pode carregar de arquivo/env/banco
    return DEFAULT_MONITORING_CONFIG


def update_monitoring_config(updates: dict[str, Any]) -> MonitoringConfig:
    """Atualiza configuração de monitoramento."""
    global DEFAULT_MONITORING_CONFIG

    current_dict = DEFAULT_MONITORING_CONFIG.model_dump()
    current_dict.update(updates)

    DEFAULT_MONITORING_CONFIG = MonitoringConfig(**current_dict)

    return DEFAULT_MONITORING_CONFIG
