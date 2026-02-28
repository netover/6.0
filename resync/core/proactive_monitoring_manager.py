# pylint
"""
Inicializador do Sistema de Monitoramento Proativo

Este módulo coordena a inicialização de todos os componentes
de monitoramento em tempo real do TWS.

Componentes gerenciados:
- TWSBackgroundPoller: Coleta status do TWS
- EventBus: Broadcast de eventos
- TWSStatusStore: Persistência e aprendizado
- Pattern Detection: Detecção de padrões

Autor: Resync Team
Versão: 5.2
"""

import asyncio
import contextlib
from datetime import datetime, timezone, timedelta
from typing import Any

import structlog

from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)

class ProactiveMonitoringManager:
    """
    Gerenciador central do sistema de monitoramento proativo.

    Responsável por:
    - Inicializar todos os componentes
    - Coordenar comunicação entre componentes
    - Gerenciar ciclo de vida
    - Agendar tarefas periódicas
    """

    def __init__(self):
        self._initialized = False
        self._running = False

        # Componentes
        self._poller = None
        self._event_bus = None
        self._status_store = None

        # Tasks
        self._pattern_detection_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Config
        self._config = None

        logger.info("proactive_monitoring_manager_created")

    async def initialize(
        self,
        tws_client: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Inicializa todos os componentes de monitoramento.

        Args:
            tws_client: Cliente TWS para chamadas API
            config: Configurações opcionais
        """
        if self._initialized:
            logger.warning("Monitoring already initialized")
            return

        logger.info("initializing_proactive_monitoring")

        # Carrega configuração
        from resync.core.monitoring_config import (
            get_monitoring_config,
            update_monitoring_config,
        )

        if config:
            self._config = update_monitoring_config(config)
        else:
            self._config = get_monitoring_config()

        # 1. Inicializa Event Bus
        from resync.core.event_bus import init_event_bus

        self._event_bus = init_event_bus(
            history_size=1000,
            enable_persistence=True,
        )
        await self._event_bus.start()
        logger.info("event_bus_initialized")

        # 2. Inicializa Status Store
        from resync.core.tws_status_store import init_status_store

        self._status_store = await init_status_store(
            # db_path removed - using PostgreSQL
            retention_days_full=self._config.retention_days_full,
            retention_days_summary=self._config.retention_days_summary,
            retention_days_patterns=self._config.retention_days_patterns,
        )
        logger.info("status_store_initialized")

        # 3. Inicializa Background Poller
        from resync.core.tws_background_poller import init_tws_poller

        self._poller = init_tws_poller(
            tws_client=tws_client,
            polling_interval=self._config.polling_interval_seconds,
            status_store=self._status_store,
            event_bus=self._event_bus,
        )

        # Configura thresholds
        self._poller.job_stuck_threshold_minutes = (
            self._config.job_stuck_threshold_minutes
        )
        self._poller.job_late_threshold_minutes = (
            self._config.job_late_threshold_minutes
        )

        # Adiciona handler para persistir eventos
        self._poller.add_event_handler(self._on_event)

        logger.info("background_poller_initialized")

        self._initialized = True
        logger.info("proactive_monitoring_initialized")

    async def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Inicia o sistema de monitoramento.

        Args:
            tg: Optional TaskGroup to run the background tasks in
        """
        if not self._initialized:
            raise RuntimeError("Monitoring not initialized. Call initialize() first.")

        if self._running:
            logger.warning("Monitoring already running")
            return

        logger.info("starting_proactive_monitoring")

        # Inicia poller
        await self._poller.start(tg=tg)

        # Inicia detecção de padrões periódica
        if self._config.pattern_detection_enabled:
            if tg:
                self._pattern_detection_task = tg.create_task(
                    self._pattern_detection_loop(), name="pattern_detection_loop"
                )
            else:
                self._pattern_detection_task = create_tracked_task(
                    self._pattern_detection_loop(), name="pattern_detection_loop"
                )

        # Inicia limpeza periódica
        if tg:
            self._cleanup_task = tg.create_task(
                self._cleanup_loop(), name="cleanup_loop"
            )
        else:
            self._cleanup_task = create_tracked_task(
                self._cleanup_loop(), name="cleanup_loop"
            )

        self._running = True
        logger.info(
            "proactive_monitoring_started", method="task_group" if tg else "track_task"
        )

    async def stop(self) -> None:
        """Para o sistema de monitoramento."""
        if not self._running:
            return

        logger.info("stopping_proactive_monitoring")

        # Para poller
        if self._poller:
            await self._poller.stop()

        # Para event bus
        if self._event_bus:
            await self._event_bus.stop()

        # Cancela tasks
        if self._pattern_detection_task:
            self._pattern_detection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pattern_detection_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Fecha store
        if self._status_store:
            await self._status_store.close()

        self._running = False
        logger.info("proactive_monitoring_stopped")

    async def _on_event(self, event: Any) -> None:
        """Handler para eventos gerados pelo poller."""
        if self._status_store:
            try:
                await self._status_store.save_event(event)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("failed_to_save_event", error=str(e))

        # Se for um ABEND, notifica Teams e tenta encontrar solução
        if hasattr(event, "event_type") and event.event_type.value == "job_abend":
            # Notificação proativa no Teams
            await self._notify_teams_abend(event)
            # Busca solução
            await self._suggest_solution(event)

    async def _notify_teams_abend(self, event: Any) -> None:
        """Envia notificação proativa para Teams quando detecta ABEND."""
        try:
            from resync.core.database.session import SessionLocal
            from resync.core.teams_notifier import TeamsNotificationManager

            # Extrai dados do evento
            job_data = event.details.get("job", {})
            job_name = job_data.get("job_name", event.source)
            job_status = job_data.get("status", "ABEND")
            workstation = job_data.get("workstation", "")
            return_code = job_data.get("return_code")
            error_message = job_data.get("error_message") or event.message

            # Cria sessão do banco
            db = SessionLocal()
            try:
                manager = TeamsNotificationManager(db)

                # Verifica se notificações estão habilitadas
                if not manager.is_enabled():
                    logger.debug("teams_notifications_disabled_skipping")
                    return

                # Envia notificação
                success = await manager.send_job_notification(
                    job_name=job_name,
                    job_status=job_status,
                    instance_name=workstation,
                    return_code=return_code,
                    error_message=error_message,
                    timestamp=event.timestamp.isoformat()
                    if hasattr(event, "timestamp")
                    else None,
                )

                if success:
                    logger.info(
                        "teams_abend_notification_sent", job=job_name, status=job_status
                    )
                else:
                    logger.warning(
                        "teams_abend_notification_skipped",
                        job=job_name,
                        reason="notification_filter_or_no_channel",
                    )

            finally:
                db.close()

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "teams_abend_notification_error",
                error=str(e),
                job=getattr(event, "source", "unknown"),
            )

    async def _suggest_solution(self, event: Any) -> None:
        """Busca e sugere solução para um problema."""
        if not self._status_store or not self._config.solution_correlation_enabled:
            return

        try:
            error_msg = event.details.get("job", {}).get("error_message", "")
            if not error_msg:
                return

            solution = await self._status_store.find_solution(
                problem_type="job_abend",
                error_message=error_msg,
            )

            if (
                solution
                and solution.get("success_rate", 0)
                >= self._config.solution_min_success_rate
            ):
                # Publica sugestão de solução
                import time

                from resync.core.tws_background_poller import (
                    AlertSeverity,
                    EventType,
                    TWSEvent,
                )

                suggestion_event = TWSEvent(
                    event_id=f"suggestion_{int(time.time())}",
                    event_type=EventType.PATTERN_DETECTED,
                    severity=AlertSeverity.INFO,
                    timestamp=datetime.now(timezone.utc),
                    source=event.source,
                    message=f"Solução sugerida: {solution['solution']}",
                    details={
                        "solution": solution,
                        "original_event": event.to_dict()
                        if hasattr(event, "to_dict")
                        else str(event),
                    },
                )

                if self._event_bus:
                    await self._event_bus.publish(suggestion_event)

                logger.info(
                    "solution_suggested",
                    job=event.source,
                    solution=solution["solution"],
                    success_rate=solution["success_rate"],
                )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("solution_suggestion_error", error=str(e))

    async def _pattern_detection_loop(self) -> None:
        """Loop de detecção de padrões."""
        interval = self._config.pattern_detection_interval_minutes * 60

        while True:
            try:
                await asyncio.sleep(interval)

                if self._status_store:
                    patterns = await self._status_store.detect_patterns()

                    # Publica padrões significativos
                    for pattern in patterns:
                        if pattern.confidence >= self._config.pattern_min_confidence:
                            await self._publish_pattern(pattern)

                    logger.info(
                        "pattern_detection_completed",
                        patterns_found=len(patterns),
                    )

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("pattern_detection_error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry

    async def _publish_pattern(self, pattern: Any) -> None:
        """Publica um padrão detectado."""
        if not self._event_bus:
            return

        import time

        from resync.core.tws_background_poller import AlertSeverity, EventType, TWSEvent

        event = TWSEvent(
            event_id=f"pattern_{int(time.time())}",
            event_type=EventType.PATTERN_DETECTED,
            severity=AlertSeverity.WARNING
            if pattern.confidence > 0.7
            else AlertSeverity.INFO,
            timestamp=datetime.now(timezone.utc),
            source="PatternDetector",
            message=pattern.description,
            details=pattern.to_dict()
            if hasattr(pattern, "to_dict")
            else {"pattern": str(pattern)},
        )

        await self._event_bus.publish(event)

    async def _cleanup_loop(self) -> None:
        """Loop de limpeza de dados antigos."""
        # Executa às 03:00 todos os dias
        while True:
            try:
                # Calcula tempo até próxima execução (03:00)
                now = datetime.now(timezone.utc)
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    # Add one day safely (avoids end-of-month ValueError)
                    next_run = next_run + timedelta(days=1)

                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)

                if self._status_store:
                    deleted = await self._status_store.cleanup_old_data()
                    logger.info("scheduled_cleanup_completed", deleted=deleted)

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("cleanup_error", error=str(e))
                await asyncio.sleep(3600)  # Wait 1h before retry

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Retorna status do sistema de monitoramento."""
        status = {
            "initialized": self._initialized,
            "running": self._running,
            "components": {
                "poller": None,
                "event_bus": None,
                "status_store": None,
            },
        }

        if self._poller:
            status["components"]["poller"] = self._poller.get_metrics()

        if self._event_bus:
            status["components"]["event_bus"] = self._event_bus.get_metrics()

        return status

    def update_config(self, updates: dict[str, Any]) -> None:
        """Atualiza configurações em runtime."""
        from resync.core.monitoring_config import update_monitoring_config

        self._config = update_monitoring_config(updates)

        # Aplica mudanças ao poller
        if self._poller and "polling_interval_seconds" in updates:
            self._poller.set_polling_interval(updates["polling_interval_seconds"])

        if self._poller and "job_stuck_threshold_minutes" in updates:
            self._poller.job_stuck_threshold_minutes = updates[
                "job_stuck_threshold_minutes"
            ]

        logger.info("monitoring_config_updated", updates=updates)

    @property
    def poller(self):
        return self._poller

    @property
    def event_bus(self):
        return self._event_bus

    @property
    def status_store(self):
        return self._status_store

    @property
    def config(self):
        return self._config

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_manager_instance: ProactiveMonitoringManager | None = None

def get_monitoring_manager() -> ProactiveMonitoringManager | None:
    """Retorna instância singleton do manager."""
    return _manager_instance

def init_monitoring_manager() -> ProactiveMonitoringManager:
    """Inicializa o manager singleton."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = ProactiveMonitoringManager()

    return _manager_instance

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def setup_proactive_monitoring(
    tws_client: Any,
    config: dict[str, Any] | None = None,
    auto_start: bool = True,
    tg: asyncio.TaskGroup | None = None,
) -> ProactiveMonitoringManager:
    """
    Configura e inicia o sistema de monitoramento proativo.

    Args:
        tws_client: Cliente TWS
        config: Configurações opcionais
        auto_start: Se deve iniciar automaticamente
        tg: Optional TaskGroup to start the monitoring in

    Returns:
        Manager configurado
    """
    manager = init_monitoring_manager()

    await manager.initialize(tws_client, config)

    if auto_start:
        await manager.start(tg=tg)

    return manager

async def shutdown_proactive_monitoring() -> None:
    """Para e limpa o sistema de monitoramento."""
    manager = get_monitoring_manager()

    if manager:
        await manager.stop()
