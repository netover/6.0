"""
API Routes para Monitoramento Proativo em Tempo Real

Este módulo define os endpoints REST e WebSocket para
monitoramento proativo do TWS.

Funcionalidades:
- GET/PUT configurações de monitoramento
- WebSocket para eventos em tempo real
- Endpoints para status, eventos, padrões
- Gerenciamento do poller

Autor: Resync Team
Versão: 5.2
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# Router para endpoints de monitoramento
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PollingConfigUpdate(BaseModel):
    """Atualização de configuração de polling."""

    polling_interval_seconds: int | None = Field(None, ge=5, le=300)
    polling_mode: str | None = None
    alerts_enabled: bool | None = None
    browser_notifications_enabled: bool | None = None
    dashboard_theme: str | None = None
    dashboard_refresh_seconds: int | None = Field(None, ge=1, le=60)


class SolutionInput(BaseModel):
    """Input para adicionar uma solução."""

    problem_type: str
    problem_pattern: str
    solution: str


class SolutionResultInput(BaseModel):
    """Input para registrar resultado de solução."""

    problem_id: str
    success: bool


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================


@monitoring_router.get("/config")
async def get_monitoring_config():
    """Obtém configuração atual de monitoramento."""
    from resync.core.monitoring_config import get_monitoring_config

    config = get_monitoring_config()
    return {
        "success": True,
        "config": config.model_dump(),
        "frontend_config": config.to_frontend_config(),
    }


@monitoring_router.put("/config")
async def update_monitoring_config_endpoint(updates: PollingConfigUpdate):
    """Atualiza configuração de monitoramento."""
    from resync.core.monitoring_config import update_monitoring_config
    from resync.core.tws_background_poller import get_tws_poller

    # Filtra campos não-nulos
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        raise HTTPException(status_code=400, detail="No updates provided")

    # Atualiza config
    new_config = update_monitoring_config(update_dict)

    # Se mudou intervalo de polling, atualiza o poller
    if "polling_interval_seconds" in update_dict:
        poller = get_tws_poller()
        if poller:
            poller.set_polling_interval(update_dict["polling_interval_seconds"])

    logger.info("monitoring_config_updated", updates=update_dict)

    return {
        "success": True,
        "message": "Configuration updated",
        "config": new_config.model_dump(),
    }


# =============================================================================
# STATUS ENDPOINTS
# =============================================================================


@monitoring_router.get("/status")
async def get_current_status():
    """Obtém status atual do sistema."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        return {
            "success": True,
            "status": "not_initialized",
            "message": "Monitoring not initialized",
        }

    snapshot = poller.get_current_snapshot()
    metrics = poller.get_metrics()

    return {
        "success": True,
        "status": "running" if metrics["is_running"] else "stopped",
        "metrics": metrics,
        "snapshot": snapshot.to_dict() if snapshot else None,
    }


@monitoring_router.get("/status/workstations")
async def get_workstations_status():
    """Obtém status das workstations."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        return {"success": True, "workstations": []}

    snapshot = poller.get_current_snapshot()

    if not snapshot:
        return {"success": True, "workstations": []}

    return {
        "success": True,
        "workstations": [ws.to_dict() for ws in snapshot.workstations],
        "summary": {
            "total": len(snapshot.workstations),
            "online": sum(1 for ws in snapshot.workstations if ws.status == "LINKED"),
            "offline": sum(1 for ws in snapshot.workstations if ws.status != "LINKED"),
        },
    }


@monitoring_router.get("/status/jobs")
async def get_jobs_status(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
):
    """Obtém status dos jobs."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        return {"success": True, "jobs": []}

    snapshot = poller.get_current_snapshot()

    if not snapshot:
        return {"success": True, "jobs": []}

    jobs = snapshot.jobs

    if status:
        jobs = [j for j in jobs if j.status == status.upper()]

    return {
        "success": True,
        "jobs": [j.to_dict() for j in jobs[:limit]],
        "summary": {
            "total": snapshot.total_jobs_today,
            "running": snapshot.jobs_running,
            "completed": snapshot.jobs_completed,
            "failed": snapshot.jobs_failed,
            "pending": snapshot.jobs_pending,
        },
    }


# =============================================================================
# EVENTS ENDPOINTS
# =============================================================================


@monitoring_router.get("/events")
async def get_events(
    hours: int = Query(24, ge=1, le=168),
    severity: str | None = Query(None),
    event_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Obtém eventos recentes."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        # Fallback para event bus
        from resync.core.event_bus import get_event_bus

        bus = get_event_bus()
        if bus:
            events = bus.get_recent_events(limit)
            return {"success": True, "events": events, "total": len(events)}
        return {"success": True, "events": [], "total": 0}

    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    end_time = datetime.now(timezone.utc)

    event_types = [event_type] if event_type else None

    events = await store.get_events_in_range(
        start_time=start_time,
        end_time=end_time,
        event_types=event_types,
        severity=severity,
        limit=limit,
    )

    return {
        "success": True,
        "events": events,
        "total": len(events),
        "period": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
    }


@monitoring_router.get("/events/search")
async def search_events(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(50, ge=1, le=200),
):
    """Busca eventos por texto."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    events = await store.search_events(q, limit)

    return {
        "success": True,
        "query": q,
        "events": events,
        "total": len(events),
    }


@monitoring_router.get("/events/critical")
async def get_critical_events(
    limit: int = Query(20, ge=1, le=100),
):
    """Obtém eventos críticos."""
    from resync.core.event_bus import get_event_bus

    bus = get_event_bus()

    if not bus:
        return {"success": True, "events": []}

    events = bus.get_critical_events(limit)

    return {
        "success": True,
        "events": events,
        "total": len(events),
    }


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================


@monitoring_router.get("/history/job/{job_name}")
async def get_job_history(
    job_name: str,
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(100, ge=1, le=500),
):
    """Obtém histórico de um job específico."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    history = await store.get_job_history(job_name, days, limit)

    return {
        "success": True,
        "job_name": job_name,
        "history": history,
        "total": len(history),
        "period_days": days,
    }


@monitoring_router.get("/history/daily/{date}")
async def get_daily_summary(date: str):
    """Obtém resumo de um dia específico (formato: YYYY-MM-DD)."""
    from resync.core.tws_status_store import get_status_store

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD") from None

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    summary = await store.get_daily_summary(target_date)

    return {
        "success": True,
        **summary,
    }


# =============================================================================
# PATTERNS ENDPOINTS
# =============================================================================


@monitoring_router.get("/patterns")
async def get_patterns(
    pattern_type: str | None = Query(None),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
):
    """Obtém padrões detectados."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    patterns = await store.get_patterns(pattern_type, min_confidence)

    return {
        "success": True,
        "patterns": patterns,
        "total": len(patterns),
    }


@monitoring_router.post("/patterns/detect")
async def trigger_pattern_detection():
    """Dispara detecção manual de padrões."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    patterns = await store.detect_patterns()

    return {
        "success": True,
        "patterns_detected": len(patterns),
        "patterns": [p.to_dict() for p in patterns],
    }


# =============================================================================
# SOLUTIONS ENDPOINTS
# =============================================================================


@monitoring_router.post("/solutions")
async def add_solution(input: SolutionInput):
    """Adiciona uma correlação problema-solução."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    problem_id = await store.add_solution(
        input.problem_type,
        input.problem_pattern,
        input.solution,
    )

    return {
        "success": True,
        "problem_id": problem_id,
        "message": "Solution added successfully",
    }


@monitoring_router.get("/solutions/find")
async def find_solution(
    problem_type: str = Query(...),
    error_message: str = Query(...),
):
    """Busca solução para um problema."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    solution = await store.find_solution(problem_type, error_message)

    if solution:
        return {
            "success": True,
            "found": True,
            **solution,
        }

    return {
        "success": True,
        "found": False,
        "message": "No matching solution found",
    }


@monitoring_router.post("/solutions/result")
async def record_solution_result(input: SolutionResultInput):
    """Registra resultado de aplicação de solução."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    await store.record_solution_result(input.problem_id, input.success)

    return {
        "success": True,
        "message": "Result recorded",
    }


# =============================================================================
# POLLER CONTROL ENDPOINTS
# =============================================================================


@monitoring_router.post("/poller/start")
async def start_poller():
    """Inicia o poller de background."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        raise HTTPException(status_code=503, detail="Poller not initialized")

    await poller.start()

    return {
        "success": True,
        "message": "Poller started",
        "metrics": poller.get_metrics(),
    }


@monitoring_router.post("/poller/stop")
async def stop_poller():
    """Para o poller de background."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        raise HTTPException(status_code=503, detail="Poller not initialized")

    await poller.stop()

    return {
        "success": True,
        "message": "Poller stopped",
        "metrics": poller.get_metrics(),
    }


@monitoring_router.post("/poller/poll")
async def force_poll():
    """Força uma coleta imediata."""
    from resync.core.tws_background_poller import get_tws_poller

    poller = get_tws_poller()

    if not poller:
        raise HTTPException(status_code=503, detail="Poller not initialized")

    snapshot = await poller.force_poll()

    return {
        "success": True,
        "message": "Poll completed",
        "snapshot": snapshot.to_dict() if snapshot else None,
    }


# =============================================================================
# STATS ENDPOINTS
# =============================================================================


@monitoring_router.get("/stats")
async def get_monitoring_stats():
    """Obtém estatísticas do sistema de monitoramento."""
    from resync.core.event_bus import get_event_bus
    from resync.core.tws_background_poller import get_tws_poller
    from resync.core.tws_status_store import get_status_store

    stats = {}

    # Poller stats
    poller = get_tws_poller()
    if poller:
        stats["poller"] = poller.get_metrics()

    # Event bus stats
    bus = get_event_bus()
    if bus:
        stats["event_bus"] = bus.get_metrics()

    # Store stats
    store = get_status_store()
    if store:
        stats["database"] = await store.get_database_stats()

    return {
        "success": True,
        "stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@monitoring_router.post("/cleanup")
async def cleanup_old_data():
    """Remove dados antigos do banco."""
    from resync.core.tws_status_store import get_status_store

    store = get_status_store()

    if not store:
        raise HTTPException(status_code=503, detail="Status store not available")

    deleted = await store.cleanup_old_data()

    return {
        "success": True,
        "message": "Cleanup completed",
        "deleted": deleted,
    }


# =============================================================================
# RAG QUERY ENDPOINTS
# =============================================================================


class RAGQueryInput(BaseModel):
    """Input para query RAG."""

    query: str = Field(..., min_length=3, max_length=500)


@monitoring_router.post("/query")
async def process_rag_query(input: RAGQueryInput):
    """
    Processa uma query em linguagem natural sobre o TWS.

    Exemplos:
    - "O que aconteceu ontem?"
    - "Quais jobs falharam hoje?"
    - "Tem algum padrão nas falhas?"
    """
    from resync.core.tws_rag_queries import process_tws_query

    result = await process_tws_query(input.query)

    return {
        "success": result.success,
        "query": input.query,
        "summary": result.summary,
        "details": result.details[:10],  # Limita detalhes
        "suggestions": result.suggestions,
        "metadata": result.metadata,
    }


@monitoring_router.get("/query/examples")
async def get_query_examples():
    """Retorna exemplos de queries RAG."""
    from resync.core.tws_rag_queries import EXAMPLE_QUERIES

    return {
        "success": True,
        "examples": EXAMPLE_QUERIES,
    }


# =============================================================================
# DASHBOARD ROUTE
# =============================================================================
@monitoring_router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve o dashboard de monitoramento em tempo real."""
    from pathlib import Path

    template_path = Path(__file__).parent.parent.parent / "templates" / "realtime_dashboard.html"

    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))

    return HTMLResponse(
        content="<h1>Dashboard not found</h1><p>Template file missing.</p>",
        status_code=404,
    )


# =============================================================================
# NOTIFICATION MANAGEMENT
# =============================================================================


@monitoring_router.post("/notification-dismissed")
async def track_notification_dismissed(data: dict):
    """Registra quando usuário dispensa uma notificação."""
    event_id = data.get("eventId")
    if event_id:
        logger.info("notification_dismissed", event_id=event_id)
    return {"success": True}


@monitoring_router.post("/subscribe-push")
async def subscribe_push_notifications(subscription: dict):
    """Registra subscription para push notifications."""
    # Em produção, salvaria no banco
    logger.info(
        "push_subscription_registered",
        endpoint=subscription.get("endpoint", "")[:50],
    )
    return {
        "success": True,
        "message": "Subscription registered",
    }


@monitoring_router.post("/test-notification")
async def send_test_notification():
    """Envia uma notificação de teste."""
    from resync.core.event_bus import get_event_bus

    bus = get_event_bus()

    if not bus:
        raise HTTPException(status_code=503, detail="Event bus not available")

    test_notification = {
        "type": "notification",
        "title": "Teste de Notificação",
        "message": "Esta é uma notificação de teste do Resync TWS Monitor",
        "severity": "info",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    count = await bus.broadcast_message(test_notification)

    return {
        "success": True,
        "message": f"Test notification sent to {count} clients",
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================


@monitoring_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para eventos em tempo real.

    Protocolo:
    - Cliente se conecta
    - Servidor envia eventos recentes
    - Servidor envia novos eventos conforme ocorrem
    - Cliente pode enviar mensagens de controle
    """
    from resync.core.event_bus import SubscriptionType, get_event_bus

    await websocket.accept()

    client_id = str(uuid.uuid4())
    bus = get_event_bus()

    if not bus:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Event bus not available",
            }
        )
        await websocket.close()
        return

    # Registra cliente
    await bus.register_websocket(
        client_id=client_id,
        websocket=websocket,
        subscription_types={SubscriptionType.ALL},
    )

    logger.info("websocket_client_connected", client_id=client_id)

    # Envia config inicial
    from resync.core.monitoring_config import get_monitoring_config

    config = get_monitoring_config()

    await websocket.send_json(
        {
            "type": "connected",
            "client_id": client_id,
            "config": config.to_frontend_config(),
        }
    )

    try:
        while True:
            # Aguarda mensagens do cliente
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "subscribe":
                    # Atualiza assinaturas
                    types = message.get("types", ["all"])
                    subscription_types = {
                        SubscriptionType(t)
                        for t in types
                        if t in [e.value for e in SubscriptionType]
                    }
                    await bus.update_websocket_subscriptions(client_id, subscription_types)
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "types": list(subscription_types),
                        }
                    )

                elif msg_type == "get_status":
                    # Envia status atual
                    from resync.core.tws_background_poller import get_tws_poller

                    poller = get_tws_poller()
                    if poller:
                        snapshot = poller.get_current_snapshot()
                        await websocket.send_json(
                            {
                                "type": "status",
                                "snapshot": snapshot.to_dict() if snapshot else None,
                            }
                        )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON",
                    }
                )

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected", client_id=client_id)
    except Exception as e:
        logger.error("websocket_error", client_id=client_id, error=str(e))
    finally:
        await bus.unregister_websocket(client_id)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def broadcast_notification(
    title: str,
    message: str,
    severity: str = "info",
    data: dict[str, Any] | None = None,
):
    """
    Envia notificação broadcast para todos os clientes.

    Args:
        title: Título da notificação
        message: Mensagem
        severity: Severidade (info, warning, error, critical)
        data: Dados adicionais
    """
    from resync.core.event_bus import get_event_bus

    bus = get_event_bus()

    if not bus:
        return

    notification = {
        "type": "notification",
        "title": title,
        "message": message,
        "severity": severity,
        "data": data or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await bus.broadcast_message(notification)
