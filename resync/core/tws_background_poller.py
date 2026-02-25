"""
TWS Background Poller - Monitoramento Proativo em Tempo Real

Este módulo implementa coleta assíncrona contínua de status do TWS,
detectando mudanças e gerando eventos para broadcast via WebSocket.

Funcionalidades:
- Polling configurável (intervalo em segundos)
- Detecção de mudanças de status (jobs, workstations)
- Geração de eventos para broadcast
- Armazenamento histórico para aprendizado
- Detecção de anomalias em tempo real

Autor: Resync Team
Versão: 5.2
"""

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from resync.core.task_tracker import track_task

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Tipos de eventos gerados pelo poller."""

    # Job Events
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_ABEND = "job_abend"
    JOB_LATE = "job_late"
    JOB_STUCK = "job_stuck"
    JOB_RERUN = "job_rerun"

    # Workstation Events
    WS_ONLINE = "workstation_online"
    WS_OFFLINE = "workstation_offline"
    WS_LINKED = "workstation_linked"
    WS_UNLINKED = "workstation_unlinked"

    # System Events
    SYSTEM_HEALTHY = "system_healthy"
    SYSTEM_DEGRADED = "system_degraded"
    SYSTEM_CRITICAL = "system_critical"

    # Anomaly Events
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_DETECTED = "pattern_detected"


class AlertSeverity(str, Enum):
    """Severidade dos alertas."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TWSEvent:
    """Representa um evento do TWS."""

    event_id: str
    event_type: EventType
    severity: AlertSeverity
    timestamp: datetime
    source: str  # job_name, workstation_name, etc.
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    previous_state: str | None = None
    current_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converte evento para dicionário (para JSON)."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "previous_state": self.previous_state,
            "current_state": self.current_state,
        }


@dataclass
class JobStatus:
    """Status de um job."""

    job_id: str
    job_name: str
    job_stream: str
    workstation: str
    status: str
    return_code: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "job_stream": self.job_stream,
            "workstation": self.workstation,
            "status": self.status,
            "return_code": self.return_code,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class WorkstationStatus:
    """Status de uma workstation."""

    name: str
    status: str  # ONLINE, OFFLINE, LINKED, UNLINKED
    agent_status: str
    jobs_running: int = 0
    jobs_pending: int = 0
    last_seen: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "agent_status": self.agent_status,
            "jobs_running": self.jobs_running,
            "jobs_pending": self.jobs_pending,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class SystemSnapshot:
    """Snapshot completo do estado do sistema."""

    timestamp: datetime
    workstations: list[WorkstationStatus]
    jobs: list[JobStatus]
    total_jobs_today: int = 0
    jobs_running: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_pending: int = 0
    system_health: str = "healthy"  # healthy, degraded, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "workstations": [ws.to_dict() for ws in self.workstations],
            "jobs": [job.to_dict() for job in self.jobs],
            "summary": {
                "total_jobs_today": self.total_jobs_today,
                "jobs_running": self.jobs_running,
                "jobs_completed": self.jobs_completed,
                "jobs_failed": self.jobs_failed,
                "jobs_pending": self.jobs_pending,
                "system_health": self.system_health,
            },
        }


class TWSBackgroundPoller:
    """
    Poller assíncrono para coleta de status do TWS.

    Características:
    - Polling configurável (intervalo em segundos)
    - Detecção automática de mudanças
    - Geração de eventos para broadcast
    - Cache de estado anterior para comparação
    - Suporte a múltiplos event handlers
    """

    def __init__(
        self,
        tws_client: Any,
        polling_interval: int = 30,
        status_store: Any | None = None,
        event_bus: Any | None = None,
    ):
        """
        Inicializa o poller.

        Args:
            tws_client: Cliente TWS para chamadas API
            polling_interval: Intervalo de polling em segundos (default: 30)
            status_store: Store para persistência de status
            event_bus: Bus para broadcast de eventos
        """
        self.tws_client = tws_client
        self.polling_interval = polling_interval
        self.status_store = status_store
        self.event_bus = event_bus

        # Estado interno
        self._is_running = False
        self._polling_task: asyncio.Task | None = None
        self._event_counter = 0

        # Cache de estado anterior (para detectar mudanças)
        self._previous_jobs: dict[str, JobStatus] = {}
        self._previous_workstations: dict[str, WorkstationStatus] = {}
        self._previous_snapshot: SystemSnapshot | None = None

        # Event handlers
        self._event_handlers: list[Callable[[TWSEvent], None]] = []
        self._snapshot_handlers: list[Callable[[SystemSnapshot], None]] = []

        # Métricas
        self._polls_count = 0
        self._errors_count = 0
        self._events_generated = 0
        self._start_time: datetime | None = None

        # Configurações de detecção
        # v5.3.20: Thresholds agora são configuráveis e baseados em porcentagem
        self.job_stuck_threshold_minutes = 60  # Job rodando há mais de 60 min
        self.job_late_threshold_minutes = 30  # Job atrasado mais de 30 min

        # Thresholds para health status (configurable via settings)
        # Podem ser absolutos ou percentuais dependendo do volume
        self.failure_threshold_min = 5  # Mínimo de falhas para considerar degraded
        self.failure_threshold_critical_min = 10  # Mínimo para critical
        self.failure_threshold_percentage = 0.05  # 5% de falhas = degraded
        self.failure_threshold_critical_percentage = 0.10  # 10% = critical
        self.workstation_offline_threshold = 1  # Qualquer WS offline = degraded
        self.workstation_offline_critical = 2  # 2+ WS offline = critical

        logger.info(
            "tws_background_poller_initialized",
            polling_interval=polling_interval,
            failure_thresholds={
                "min_degraded": self.failure_threshold_min,
                "min_critical": self.failure_threshold_critical_min,
                "pct_degraded": f"{self.failure_threshold_percentage * 100}%",
                "pct_critical": f"{self.failure_threshold_critical_percentage * 100}%",
            },
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Inicia o polling em background.

        Args:
            tg: Optional TaskGroup to run the polling loop in
        """
        if self._is_running:
            logger.warning("Poller already running")
            return

        self._is_running = True
        self._start_time = datetime.now(timezone.utc)
        if tg:
            self._polling_task = tg.create_task(
                self._polling_loop(), name="polling_loop"
            )
        else:
            self._polling_task = track_task(self._polling_loop(), name="polling_loop")

        logger.info(
            "tws_background_poller_started",
            polling_interval=self.polling_interval,
            method="task_group" if tg else "track_task",
        )

    async def stop(self) -> None:
        """Para o polling."""
        self._is_running = False

        if self._polling_task:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task
            self._polling_task = None

        logger.info(
            "tws_background_poller_stopped",
            polls_count=self._polls_count,
            events_generated=self._events_generated,
        )

    def set_polling_interval(self, interval: int) -> None:
        """Altera o intervalo de polling em runtime."""
        old_interval = self.polling_interval
        self.polling_interval = max(5, min(300, interval))  # 5s a 5min

        logger.info(
            "polling_interval_changed",
            old_interval=old_interval,
            new_interval=self.polling_interval,
        )

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def add_event_handler(self, handler: Callable[[TWSEvent], None]) -> None:
        """Adiciona um handler para eventos."""
        self._event_handlers.append(handler)

    def add_snapshot_handler(self, handler: Callable[[SystemSnapshot], None]) -> None:
        """Adiciona um handler para snapshots."""
        self._snapshot_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[TWSEvent], None]) -> None:
        """Remove um handler de eventos."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    # =========================================================================
    # POLLING LOOP
    # =========================================================================

    async def _polling_loop(self) -> None:
        """Loop principal de polling."""
        while self._is_running:
            try:
                # Coleta snapshot
                snapshot = await self._collect_snapshot()

                if snapshot:
                    # Detecta mudanças e gera eventos
                    events = self._detect_changes(snapshot)

                    # Persiste snapshot
                    if self.status_store:
                        await self._persist_snapshot(snapshot)

                    # Notifica handlers
                    await self._notify_snapshot_handlers(snapshot)

                    for event in events:
                        await self._notify_event_handlers(event)

                        # Publica no event bus
                        if self.event_bus:
                            await self.event_bus.publish(event)

                    # Atualiza cache
                    self._update_cache(snapshot)

                    self._polls_count += 1

                # Aguarda próximo ciclo
                await asyncio.sleep(self.polling_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                logger.error(
                    "polling_error",
                    error=str(e),
                    polls_count=self._polls_count,
                    errors_count=self._errors_count,
                )
                # Aguarda um pouco antes de tentar novamente
                await asyncio.sleep(min(self.polling_interval, 10))

    async def _collect_snapshot(self) -> SystemSnapshot | None:
        """Coleta snapshot atual do TWS."""
        try:
            # Coleta workstations
            ws_data = await self.tws_client.query_workstations(limit=100)
            workstations = self._parse_workstations(ws_data)

            # Coleta jobs ativos (running + recent)
            jobs_data = await self.tws_client.get_plan_jobs(
                status=["EXEC", "READY", "HOLD", "ABEND", "SUCC"], limit=500
            )
            jobs = self._parse_jobs(jobs_data)

            # Calcula métricas
            jobs_running = sum(1 for j in jobs if j.status == "EXEC")
            jobs_completed = sum(1 for j in jobs if j.status == "SUCC")
            jobs_failed = sum(1 for j in jobs if j.status == "ABEND")
            jobs_pending = sum(1 for j in jobs if j.status in ["READY", "HOLD"])

            # Determina saúde do sistema
            # v5.3.20: Lógica baseada em porcentagem + mínimos configuráveis
            # Isso evita alert fatigue em ambientes com alto volume de jobs
            ws_offline = sum(1 for ws in workstations if ws.status != "LINKED")
            total_jobs = len(jobs)

            # Calcula thresholds dinâmicos baseados no volume
            # Usa o MAIOR entre: threshold absoluto mínimo OU porcentagem do total
            failure_threshold_degraded = max(
                self.failure_threshold_min,
                int(total_jobs * self.failure_threshold_percentage),
            )
            failure_threshold_critical = max(
                self.failure_threshold_critical_min,
                int(total_jobs * self.failure_threshold_critical_percentage),
            )

            system_health = "healthy"
            if (
                jobs_failed >= failure_threshold_critical
                or ws_offline >= self.workstation_offline_critical
            ):
                system_health = "critical"
            elif (
                jobs_failed >= failure_threshold_degraded
                or ws_offline >= self.workstation_offline_threshold
            ):
                system_health = "degraded"

            return SystemSnapshot(
                timestamp=datetime.now(timezone.utc),
                workstations=workstations,
                jobs=jobs,
                total_jobs_today=len(jobs),
                jobs_running=jobs_running,
                jobs_completed=jobs_completed,
                jobs_failed=jobs_failed,
                jobs_pending=jobs_pending,
                system_health=system_health,
            )

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("snapshot_collection_failed", error=str(e))
            return None

    def _parse_workstations(self, data: Any) -> list[WorkstationStatus]:
        """Converte dados da API para WorkstationStatus."""
        workstations: list[WorkstationStatus] = []

        if not data:
            return workstations

        items = data if isinstance(data, list) else data.get("items", [])

        for item in items:
            ws = WorkstationStatus(
                name=item.get("name", item.get("workstation", "unknown")),
                status=item.get("status", "UNKNOWN"),
                agent_status=item.get("agentStatus", "UNKNOWN"),
                jobs_running=item.get("jobsRunning", 0),
                jobs_pending=item.get("jobsPending", 0),
                last_seen=datetime.now(timezone.utc),
            )
            workstations.append(ws)

        return workstations

    def _parse_jobs(self, data: Any) -> list[JobStatus]:
        """Converte dados da API para JobStatus."""
        jobs: list[JobStatus] = []

        if not data:
            return jobs

        items = data if isinstance(data, list) else data.get("items", [])

        for item in items:
            # Parse timestamps
            start_time = None
            end_time = None
            if item.get("startTime"):
                with contextlib.suppress(ValueError, TypeError):
                    start_time = datetime.fromisoformat(
                        item["startTime"].replace("Z", "+00:00")
                    )

            if item.get("endTime"):
                with contextlib.suppress(ValueError, TypeError):
                    end_time = datetime.fromisoformat(
                        item["endTime"].replace("Z", "+00:00")
                    )

            # Calcula duração
            duration = None
            if start_time and end_time:
                duration = (end_time - start_time).total_seconds()
            elif start_time:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            job = JobStatus(
                job_id=str(item.get("id", item.get("jobId", ""))),
                job_name=item.get("name", item.get("jobName", "unknown")),
                job_stream=item.get("jobStream", item.get("stream", "")),
                workstation=item.get("workstation", ""),
                status=item.get("status", "UNKNOWN"),
                return_code=item.get("returnCode"),
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                error_message=item.get("errorMessage", item.get("message")),
            )
            jobs.append(job)

        return jobs

    # =========================================================================
    # CHANGE DETECTION
    # =========================================================================

    def _detect_changes(self, snapshot: SystemSnapshot) -> list[TWSEvent]:
        """Detecta mudanças comparando com estado anterior."""
        events = []

        # Detecta mudanças em jobs
        events.extend(self._detect_job_changes(snapshot.jobs))

        # Detecta mudanças em workstations
        events.extend(self._detect_workstation_changes(snapshot.workstations))

        # Detecta mudanças de sistema
        events.extend(self._detect_system_changes(snapshot))

        # Detecta anomalias
        events.extend(self._detect_anomalies(snapshot))

        self._events_generated += len(events)

        return events

    def _detect_job_changes(self, jobs: list[JobStatus]) -> list[TWSEvent]:
        """Detecta mudanças em jobs."""
        events = []
        current_jobs = {j.job_id: j for j in jobs}

        for job_id, job in current_jobs.items():
            prev_job = self._previous_jobs.get(job_id)

            if not prev_job:
                # Novo job detectado
                if job.status == "EXEC":
                    events.append(
                        self._create_event(
                            EventType.JOB_STARTED,
                            AlertSeverity.INFO,
                            job.job_name,
                            (
                                f"Job {job.job_name} iniciado na "
                                f"workstation {job.workstation}"
                            ),
                            {"job": job.to_dict()},
                            current_state=job.status,
                        )
                    )
            else:
                # Job existente - verifica mudança de status
                if prev_job.status != job.status:
                    if job.status == "SUCC":
                        events.append(
                            self._create_event(
                                EventType.JOB_COMPLETED,
                                AlertSeverity.INFO,
                                job.job_name,
                                f"Job {job.job_name} completado com sucesso",
                                {
                                    "job": job.to_dict(),
                                    "duration": job.duration_seconds,
                                },
                                previous_state=prev_job.status,
                                current_state=job.status,
                            )
                        )
                    elif job.status == "ABEND":
                        events.append(
                            self._create_event(
                                EventType.JOB_ABEND,
                                AlertSeverity.ERROR,
                                job.job_name,
                                (
                                f"Job {job.job_name} falhou com erro: "
                                f"{job.error_message or 'N/A'}"
                            ),
                                {"job": job.to_dict(), "return_code": job.return_code},
                                previous_state=prev_job.status,
                                current_state=job.status,
                            )
                        )

            # Detecta job stuck (rodando há muito tempo)
            if (
                job.status == "EXEC"
                and job.duration_seconds is not None
                and job.duration_seconds > self.job_stuck_threshold_minutes * 60
            ):
                # Só gera evento se não foi gerado antes
                stuck_key = f"stuck_{job.job_id}"
                if stuck_key not in self._previous_jobs:
                    events.append(
                        self._create_event(
                            EventType.JOB_STUCK,
                            AlertSeverity.WARNING,
                            job.job_name,
                            (
                                f"Job {job.job_name} em execução há "
                                f"{job.duration_seconds / 60:.1f} minutos"
                            ),
                            {"job": job.to_dict()},
                            current_state=job.status,
                        )
                    )

        return events

    def _detect_workstation_changes(
        self, workstations: list[WorkstationStatus]
    ) -> list[TWSEvent]:
        """Detecta mudanças em workstations."""
        events = []
        current_ws = {ws.name: ws for ws in workstations}

        for ws_name, ws in current_ws.items():
            prev_ws = self._previous_workstations.get(ws_name)

            if prev_ws and prev_ws.status != ws.status:
                if ws.status == "LINKED":
                    events.append(
                        self._create_event(
                            EventType.WS_LINKED,
                            AlertSeverity.INFO,
                            ws_name,
                            f"Workstation {ws_name} reconectada",
                            {"workstation": ws.to_dict()},
                            previous_state=prev_ws.status,
                            current_state=ws.status,
                        )
                    )
                elif ws.status == "UNLINKED":
                    events.append(
                        self._create_event(
                            EventType.WS_UNLINKED,
                            AlertSeverity.WARNING,
                            ws_name,
                            f"Workstation {ws_name} desconectada",
                            {"workstation": ws.to_dict()},
                            previous_state=prev_ws.status,
                            current_state=ws.status,
                        )
                    )
                elif ws.status == "OFFLINE":
                    events.append(
                        self._create_event(
                            EventType.WS_OFFLINE,
                            AlertSeverity.ERROR,
                            ws_name,
                            f"Workstation {ws_name} offline!",
                            {"workstation": ws.to_dict()},
                            previous_state=prev_ws.status,
                            current_state=ws.status,
                        )
                    )

        return events

    def _detect_system_changes(self, snapshot: SystemSnapshot) -> list[TWSEvent]:
        """Detecta mudanças no estado geral do sistema."""
        events = []

        if self._previous_snapshot:
            prev_health = self._previous_snapshot.system_health
            curr_health = snapshot.system_health

            if prev_health != curr_health:
                severity = AlertSeverity.INFO
                event_type = EventType.SYSTEM_HEALTHY

                if curr_health == "degraded":
                    severity = AlertSeverity.WARNING
                    event_type = EventType.SYSTEM_DEGRADED
                elif curr_health == "critical":
                    severity = AlertSeverity.CRITICAL
                    event_type = EventType.SYSTEM_CRITICAL

                events.append(
                    self._create_event(
                        event_type,
                        severity,
                        "SYSTEM",
                        f"Estado do sistema mudou de {prev_health} para {curr_health}",
                        {
                            "jobs_failed": snapshot.jobs_failed,
                            "jobs_running": snapshot.jobs_running,
                            "workstations_offline": sum(
                                1
                                for ws in snapshot.workstations
                                if ws.status != "LINKED"
                            ),
                        },
                        previous_state=prev_health,
                        current_state=curr_health,
                    )
                )

        return events

    def _detect_anomalies(self, snapshot: SystemSnapshot) -> list[TWSEvent]:
        """Detecta anomalias no sistema."""
        events = []

        # Taxa de falha alta
        if snapshot.total_jobs_today > 0:
            failure_rate = snapshot.jobs_failed / snapshot.total_jobs_today
            if failure_rate > 0.1:  # Mais de 10% de falha
                events.append(
                    self._create_event(
                        EventType.ANOMALY_DETECTED,
                        AlertSeverity.WARNING,
                        "SYSTEM",
                        f"Taxa de falha alta detectada: {failure_rate:.1%}",
                        {
                            "failure_rate": failure_rate,
                            "jobs_failed": snapshot.jobs_failed,
                            "total_jobs": snapshot.total_jobs_today,
                        },
                    )
                )

        return events

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_event(
        self,
        event_type: EventType,
        severity: AlertSeverity,
        source: str,
        message: str,
        details: dict[str, Any],
        previous_state: str | None = None,
        current_state: str | None = None,
    ) -> TWSEvent:
        """Cria um novo evento."""
        self._event_counter += 1

        return TWSEvent(
            event_id=f"evt_{int(time.time())}_{self._event_counter}",
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            source=source,
            message=message,
            details=details,
            previous_state=previous_state,
            current_state=current_state,
        )

    def _update_cache(self, snapshot: SystemSnapshot) -> None:
        """Atualiza cache com estado atual."""
        self._previous_jobs = {j.job_id: j for j in snapshot.jobs}
        self._previous_workstations = {ws.name: ws for ws in snapshot.workstations}
        self._previous_snapshot = snapshot

    async def _notify_event_handlers(self, event: TWSEvent) -> None:
        """Notifica todos os handlers de evento."""
        for handler in self._event_handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # P1 fix: Run sync handler in thread pool and await the result
                    await asyncio.to_thread(handler, event)
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    error=str(e),
                    event_type=event.event_type.value,
                )

    async def _notify_snapshot_handlers(self, snapshot: SystemSnapshot) -> None:
        """Notifica todos os handlers de snapshot."""
        for handler in self._snapshot_handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(snapshot)
                else:
                    # P1 fix: Run sync handler in thread pool and await the result
                    await asyncio.to_thread(handler, snapshot)
            except Exception as e:
                logger.error("snapshot_handler_error", error=str(e))

    async def _persist_snapshot(self, snapshot: SystemSnapshot) -> None:
        """Persiste snapshot no store."""
        if self.status_store:
            try:
                await self.status_store.save_snapshot(snapshot)
            except Exception as e:
                logger.error("snapshot_persistence_error", error=str(e))

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_snapshot(self) -> SystemSnapshot | None:
        """Retorna o snapshot mais recente."""
        return self._previous_snapshot

    def get_metrics(self) -> dict[str, Any]:
        """Retorna métricas do poller."""
        uptime = None
        if self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return {
            "is_running": self._is_running,
            "polling_interval": self.polling_interval,
            "polls_count": self._polls_count,
            "errors_count": self._errors_count,
            "events_generated": self._events_generated,
            "uptime_seconds": uptime,
            "cached_jobs": len(self._previous_jobs),
            "cached_workstations": len(self._previous_workstations),
        }

    async def force_poll(self) -> SystemSnapshot | None:
        """Força uma coleta imediata."""
        return await self._collect_snapshot()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_poller_instance: TWSBackgroundPoller | None = None


def get_tws_poller() -> TWSBackgroundPoller | None:
    """Retorna instância singleton do poller."""
    return _poller_instance


def init_tws_poller(
    tws_client: Any,
    polling_interval: int = 30,
    status_store: Any = None,
    event_bus: Any = None,
) -> TWSBackgroundPoller:
    """Inicializa o poller singleton."""
    global _poller_instance

    _poller_instance = TWSBackgroundPoller(
        tws_client=tws_client,
        polling_interval=polling_interval,
        status_store=status_store,
        event_bus=event_bus,
    )

    return _poller_instance
