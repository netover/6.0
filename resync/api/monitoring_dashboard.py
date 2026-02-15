# monitoring_dashboard.py — API endpoint para dashboard interno de monitoramento (Redis Version)
# Substitui necessidade de Prometheus/Grafana com solução distribuída e sincronizada.
#
# Características:
#   - Persistência Global em Redis (History List e Latest Hash)
#   - Broadcast sincronizado via Redis Pub/Sub (Sincronização entre workers)
#   - Liderança Distribuída: Apenas um worker coleta métricas por vez (Resolução de Duplicação)
#   - Alta Performance: Serialização otimizada com orjson

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Depends

from resync.core.redis_init import get_redis_client
from resync.api.security import decode_token, require_role
from resync.core.metrics import runtime_metrics

logger = logging.getLogger(__name__)

# ── Configurações e Keys ───────────────────────────────────────────────────

SAMPLE_INTERVAL_SECONDS = 5
MAX_SAMPLES = (2 * 60 * 60) // SAMPLE_INTERVAL_SECONDS  # 2 horas
MAX_WS_CONNECTIONS = 50
WS_SEND_TIMEOUT = 1.0  # Tempo máximo para envio em um WebSocket lento

# Redis Keys
REDIS_KEY_HISTORY = "resync:monitoring:history"
REDIS_KEY_ALERTS = "resync:monitoring:alerts"
REDIS_KEY_LATEST = "resync:monitoring:latest"
REDIS_KEY_START_TIME = "resync:monitoring:start_time"
REDIS_CH_BROADCAST = "resync:monitoring:broadcast"
REDIS_LOCK_COLLECTOR = "resync:monitoring:collector:lock"


# ── Helpers de Serialização ──────────────────────────────────────────────────

try:
    import orjson
    def json_dumps(data: Any) -> str:
        return orjson.dumps(data).decode()
    def json_loads(data: str | bytes) -> Any:
        return orjson.loads(data)
except ImportError:
    import json
    def json_dumps(data: Any) -> str:
        return json.dumps(data)
    def json_loads(data: str | bytes) -> Any:
        return json.loads(data)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try: return float(val) if val is not None else default
    except (TypeError, ValueError): return default


def _safe_int(val: Any, default: int = 0) -> int:
    try: return int(val) if val is not None else default
    except (TypeError, ValueError): return default


def _safe_json_loads(data: str | bytes, context: str) -> dict | list | None:
    """Parse JSON com tratamento de erro robusto."""
    if not data: return None
    try:
        if isinstance(data, bytes):
            data = data.decode()
        return json_loads(data)
    except Exception as e:
        logger.error("JSON corrompido (%s): %s", context, e)
        return None


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class MetricSample:
    """Uma amostra de métricas em um ponto no tempo."""
    timestamp: float
    datetime_str: str

    # API
    requests_total: int = 0
    requests_per_sec: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_avg: float = 0.0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    cache_size: int = 0
    cache_evictions: int = 0

    # Agent
    agents_active: int = 0
    agents_created: int = 0
    agents_failed: int = 0

    # LLM
    llm_requests: int = 0
    llm_tokens_used: int = 0
    llm_errors: int = 0

    # TWS
    tws_connected: bool = False
    tws_latency_ms: float = 0.0
    tws_errors: int = 0
    tws_requests_success: int = 0
    tws_requests_failed: int = 0

    # System
    system_uptime: float = 0.0
    system_availability: float = 100.0
    async_operations_active: int = 0
    correlation_ids_active: int = 0

    collection_error: str | None = None


# ── Dashboard Metrics Store ──────────────────────────────────────────────────

class DashboardMetricsStore:
    """Store de métricas persistido no Redis para consistência global."""

    def __init__(self):
        self._prev_requests = 0
        self._last_sample_mono = 0.0
        self._local_lock = asyncio.Lock()

    async def compute_rate_and_add_sample(self, requests_total: int, now_mono: float, sample_builder: Any) -> None:
        """Calcula RPS localmente e persiste no Redis."""
        async with self._local_lock:
            time_delta = now_mono - self._last_sample_mono if self._last_sample_mono > 0 else SAMPLE_INTERVAL_SECONDS
            req_delta = requests_total - self._prev_requests if self._prev_requests > 0 else 0
            rps = req_delta / time_delta if time_delta > 0 else 0.0

            self._prev_requests = requests_total
            self._last_sample_mono = now_mono

            sample = sample_builder(rps)
            await self.add_sample(sample)

    async def add_error_sample(self, error: Exception) -> None:
        """Persiste amostra indicando falha na coleta."""
        now_wall = time.time()
        dt_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        raw_msg = str(error)
        error_msg = raw_msg[:197] + "..." if len(raw_msg) > 200 else raw_msg
        global_uptime = await self.get_global_uptime()

        # Tenta determinar conexão real se possível, senão assume falso
        tws_status = False
        try:
            snapshot = runtime_metrics.get_snapshot()
            tws_status = snapshot.get("slo", {}).get("tws_connection_success_rate", 0) > 0.5
        except Exception:
            pass  # Silencioso em caso de erro na obtenção do status

        sample = MetricSample(
            timestamp=now_wall,
            datetime_str=dt_str,
            collection_error=error_msg,
            system_uptime=global_uptime,
            system_availability=0.0,
            tws_connected=tws_status
        )
        await self.add_sample(sample)

    async def add_sample(self, sample: MetricSample) -> None:
        """Persiste amostra no Redis e gera alertas."""
        redis = get_redis_client()
        data = json_dumps(asdict(sample))

        pipe = redis.pipeline()
        pipe.lpush(REDIS_KEY_HISTORY, data)
        pipe.ltrim(REDIS_KEY_HISTORY, 0, MAX_SAMPLES - 1)
        pipe.set(REDIS_KEY_LATEST, data)

        # Tratamento de erro para escrita Redis
        try:
            await pipe.execute()
        except Exception as e:
            logger.error("Falha ao persistir amostra no Redis: %s", e)

        await self._check_alerts_redis(sample)

    async def _check_alerts_redis(self, sample: MetricSample) -> None:
        """Verifica alertas e persiste no Redis."""
        new_alerts = []
        if sample.error_rate > 5.0:
            new_alerts.append({
                "type": "error_rate",
                "severity": "critical" if sample.error_rate >= 10 else "warning",
                "message": f"Error rate: {sample.error_rate:.1f}%",
                "timestamp": sample.datetime_str
            })
        if sample.collection_error is None and not sample.tws_connected:
            new_alerts.append({
                "type": "tws",
                "severity": "critical",
                "message": "TWS desconectado",
                "timestamp": sample.datetime_str
            })

        if new_alerts:
            redis = get_redis_client()
            pipe = redis.pipeline()
            for alert in new_alerts:
                pipe.lpush(REDIS_KEY_ALERTS, json_dumps(alert))
            pipe.ltrim(REDIS_KEY_ALERTS, 0, 19)  # Manter últimos 20
            try:
                await pipe.execute()
            except Exception as e:
                logger.error("Falha ao persistir alertas no Redis: %s", e)

    async def get_global_uptime(self) -> float:
        """Obtém o tempo de uptime global via Redis."""
        redis = get_redis_client()
        try:
            now = time.time()
            # Usar redis.lock para避免 race conditions
            lock = redis.lock(f"{REDIS_KEY_START_TIME}:lock", timeout=1)
            if await lock.acquire(blocking=False):
                try:
                    # Verificar se já existe um start time
                    existing = await redis.get(REDIS_KEY_START_TIME)
                    if existing is None:
                        await redis.set(REDIS_KEY_START_TIME, str(now))
                finally:
                    await lock.release()

            raw = await redis.get(REDIS_KEY_START_TIME)
            if raw is None:
                logger.warning("Não foi possível determinar o tempo de início global do Redis.")
                return 0.0
            return now - float(raw)
        except Exception as e:
            logger.error("Erro ao obter uptime global do Redis: %s", e)
            return 0.0

    async def get_current_metrics(self) -> dict[str, Any]:
        """Retorna métricas atuais."""
        redis = get_redis_client()
        try:
            raw = await redis.get(REDIS_KEY_LATEST)
            if not raw:
                return self._empty_response("initializing")

            data = _safe_json_loads(raw, "latest")
            if not data:
                return self._empty_response("data_error")

            alerts_raw = await redis.lrange(REDIS_KEY_ALERTS, 0, 4)
            alerts = [p for a in alerts_raw if (p := _safe_json_loads(a, "alert"))]

            return self._format_metrics_dict(data, alerts)
        except Exception as e:
            logger.error("Erro ao obter métricas: %s", e)
            return self._empty_response("error")

    async def get_history(self, minutes: int = 120) -> dict[str, Any]:
        """Retorna histórico de métricas."""
        redis = get_redis_client()
        try:
            needed = (minutes * 60) // SAMPLE_INTERVAL_SECONDS
            raw_list = await redis.lrange(REDIS_KEY_HISTORY, 0, needed - 1)
            if not raw_list:
                return self._empty_history()

            samples = [p for r in reversed(raw_list) if (p := _safe_json_loads(r, "history"))]
            if not samples:
                return self._empty_history()

            return {
                "timestamps": [s.get("datetime_str", "") for s in samples],
                "api": {
                    "requests_per_sec": [round(s.get("requests_per_sec", 0), 1) for s in samples],
                    "error_rate": [round(s.get("error_rate", 0), 2) for s in samples]
                },
                "cache": {"hit_ratio": [round(s.get("cache_hit_ratio", 0), 1) for s in samples]},
                "agents": {"active": [s.get("agents_active", 0) for s in samples]},
                "sample_count": len(samples)
            }
        except Exception as e:
            logger.error("Erro ao obter histórico: %s", e)
            return self._empty_history()

    def _format_metrics_dict(self, current: dict, alerts: list) -> dict:
        """Formata métricas para resposta API."""
        err_msg = current.get("collection_error")
        error_rate = current.get("error_rate", 0)
        if err_msg:
            status_val = "collection_error"
        elif error_rate >= 10:
            status_val = "critical"
        elif error_rate >= 5:
            status_val = "degraded"
        else:
            status_val = "ok"

        return {
            "status": status_val,
            "uptime_seconds": round(current.get("system_uptime", 0), 1),
            "last_update": current.get("datetime_str"),
            "api": {
                "requests_per_sec": round(current.get("requests_per_sec", 0), 1),
                "requests_total": current.get("requests_total", 0),
                "error_rate": round(current.get("error_rate", 0), 2)
            },
            "system": {"availability": round(current.get("system_availability", 100), 2)},
            "alerts": alerts,
            "collection_error": err_msg
        }

    def _empty_response(self, status: str) -> dict:
        """Retorna resposta vazia."""
        return {"status": status, "api": {"requests_per_sec": 0}, "alerts": []}

    def _empty_history(self) -> dict:
        """Retorna histórico vazio."""
        return {
            "timestamps": [],
            "api": {"requests_per_sec": [], "error_rate": []},
            "cache": {"hit_ratio": []},
            "agents": {"active": []},
            "sample_count": 0
        }


# ── WebSocket Manager ───────────────────────────────────────────────────────

class WebSocketManager:
    """Gerencia conexões WebSocket locais e sincroniza via Redis Pub/Sub."""

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._sync_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start_sync(self):
        """Inicia o listener de Pub/Sub."""
        if self._stop_event.is_set():
            return
        if self._sync_task is None or self._sync_task.done():
            self._stop_event.clear()
            self._sync_task = asyncio.create_task(self._pubsub_listener())

    async def stop(self) -> None:
        """Para o manager."""
        self._stop_event.set()
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._sync_task
        self._sync_task = None

    async def connect(self, websocket: WebSocket) -> bool:
        """Conecta um WebSocket."""
        async with self._lock:
            if len(self._clients) >= MAX_WS_CONNECTIONS:
                logger.warning("Limite de conexões WebSocket atingido: %d", MAX_WS_CONNECTIONS)
                return False
            self._clients.add(websocket)
            return True

    async def disconnect(self, websocket: WebSocket):
        """Desconecta um WebSocket."""
        async with self._lock:
            self._clients.discard(websocket)

    async def _pubsub_listener(self):
        """Listener do Redis Pub/Sub para broadcast."""
        while not self._stop_event.is_set():
            pubsub = None
            try:
                redis = get_redis_client()
                pubsub = redis.pubsub()
                await pubsub.subscribe(REDIS_CH_BROADCAST)
                logger.info("WebSocketManager sincronizado com Redis Pub/Sub")
                async for message in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode()
                        await self._local_broadcast(data)
            except asyncio.CancelledError:
                logger.info("WebSocketManager Pub/Sub listener cancelado")
                break
            except Exception:
                logger.exception("Pub/Sub desconectado; tentando reconectar em 5s")
                if not self._stop_event.is_set():
                    await asyncio.sleep(5)
            finally:
                if pubsub is not None:
                    with suppress(Exception):
                        await pubsub.unsubscribe(REDIS_CH_BROADCAST)
                    close_fn = getattr(pubsub, "close", None)
                    if close_fn is not None:
                        with suppress(Exception):
                            maybe_awaitable = close_fn()
                            if asyncio.iscoroutine(maybe_awaitable):
                                await maybe_awaitable

    async def broadcast(self, message_str: str) -> None:
        """Broadcast de mensagem para todos os clientes."""
        await self._local_broadcast(message_str)

    async def _local_broadcast(self, message_str: str):
        """Envia mensagem para todos os clientes locais."""
        async with self._lock:
            clients = list(self._clients)

        async def _safe_send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_text(message_str), timeout=WS_SEND_TIMEOUT)
            except Exception:
                logger.debug("Falha ao enviar mensagem WebSocket; desconectando cliente")
                await self.disconnect(ws)

        if clients:
            tasks = [asyncio.create_task(_safe_send(c)) for c in clients]
            await asyncio.gather(*tasks, return_exceptions=True)


# ── WebSocket Authentication ─────────────────────────────────────────────────

async def _verify_ws_admin(websocket: WebSocket) -> str | None:
    """Valida autenticação admin para WebSocket.
    
    Aceita token tanto do header Authorization quanto do query param access_token.
    """
    try:
        # Normalizar: aceitar ambos Authorization header e access_token query
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = websocket.query_params.get("access_token")

        if not token:
            return None

        payload = decode_token(token)
        username = payload.get("sub")

        # Verificar se é admin (suporta roles ou scopes)
        roles_claim = payload.get("roles")
        if roles_claim is None:
            legacy_role = payload.get("role")
            roles = [legacy_role] if legacy_role else []
        elif isinstance(roles_claim, list):
            roles = roles_claim
        else:
            roles = [roles_claim]

        if "admin" not in roles:
            return None

        return username
    except Exception as e:
        logger.warning("WebSocket authentication failed: %s", e, exc_info=True)
        return None


# ── Singletons ───────────────────────────────────────────────────────────────

_metrics_store = DashboardMetricsStore()
ws_manager = WebSocketManager()


# ── Collector Logic ──────────────────────────────────────────────────────────

async def collect_metrics_sample() -> None:
    """Apenas um worker coleta por vez (Liderança via Redis Lock)."""
    redis = get_redis_client()

    # Usar redis.lock consistentemente para evitar race conditions
    lock = redis.lock(REDIS_LOCK_COLLECTOR, timeout=15)
    if not await lock.acquire(blocking=False):
        return  # Outro worker já está coletando

    try:
        snapshot = runtime_metrics.get_snapshot()
        now_mono = time.monotonic()
        now_wall = time.time()
        dt_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        agent = snapshot.get("agent", {})
        slo = snapshot.get("slo", {})
        req_total = _safe_int(agent.get("initializations"))
        uptime = await _metrics_store.get_global_uptime()

        def build_sample(rps: float) -> MetricSample:
            return MetricSample(
                timestamp=now_wall,
                datetime_str=dt_str,
                requests_total=req_total,
                requests_per_sec=rps,
                error_rate=_safe_float(slo.get("api_error_rate")) * 100,
                tws_connected=_safe_float(slo.get("tws_connection_success_rate")) > 0.5,
                system_uptime=uptime,
                system_availability=_safe_float(slo.get("availability"), 1.0) * 100
            )

        await _metrics_store.compute_rate_and_add_sample(req_total, now_mono, build_sample)

        current = await _metrics_store.get_current_metrics()
        subscribers = await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        if subscribers == 0:
            logger.debug("Nenhum subscriber no canal de broadcast")

    except Exception as e:
        logger.error("Erro na coleta: %s", e)
        try:
            await _metrics_store.add_error_sample(e)
            current = await _metrics_store.get_current_metrics()
            await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        except Exception:
            logger.debug("Falha ao persistir/broadcast amostra de erro")
    finally:
        # Libera o lock explicitamente após a coleta
        await lock.release()


async def metrics_collector_loop() -> None:
    """Loop de coleta de métricas em background."""
    try:
        redis = get_redis_client()
        await redis.ping()
    except Exception as e:
        logger.error("Redis não disponível, encerrando collector: %s", e)
        return

    await ws_manager.start_sync()
    while True:
        try:
            await collect_metrics_sample()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Erro no collector loop ao executar collect_metrics_sample")
        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)


# ── FastAPI Router ───────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/current", dependencies=[Depends(require_role("admin"))])
async def get_current():
    """Retorna métricas atuais do sistema. Requer autenticação admin."""
    return await _metrics_store.get_current_metrics()


@router.get("/history", dependencies=[Depends(require_role("admin"))])
async def get_history(minutes: int = 120):
    """Retorna histórico de métricas para gráficos. Requer autenticação admin."""
    return await _metrics_store.get_history(min(max(1, minutes), 120))


@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket para métricas em tempo real com autenticação."""
    username = await _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # ÚNICA chamada a accept() - não duplicar
    await websocket.accept()

    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    try:
        initial = await _metrics_store.get_current_metrics()
        await websocket.send_json(initial)
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await ws_manager.disconnect(websocket)
