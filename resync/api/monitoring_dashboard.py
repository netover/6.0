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
import secrets
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Depends

from resync.core.redis_init import get_redis_client
from resync.api.security import decode_token, require_role
from resync.core.metrics import runtime_metrics

if TYPE_CHECKING:
    import redis.asyncio as redis_async

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
REDIS_KEY_PREV_REQUESTS = "resync:monitoring:prev_requests"
REDIS_KEY_PREV_WALLTIME = "resync:monitoring:prev_walltime"
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
    if isinstance(val, bytes):
        val = val.decode(errors="ignore")
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if isinstance(val, bytes):
        val = val.decode(errors="ignore")
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_json_loads(data: str | bytes, context: str) -> dict | list | None:
    """Parse JSON com tratamento de erro robusto."""
    if not data:
        return None
    try:
        if isinstance(data, bytes):
            data = data.decode()
        return json_loads(data)
    except Exception as e:
        logger.error("JSON corrompido (%s): %s", context, e)
        return None


def _classify_collection_error(error: Exception) -> str:
    """Classifica erro de coleta sem expor detalhes internos."""
    type_name = type(error).__name__
    mapping = {
        "ConnectionError": "redis_connection_failed",
        "TimeoutError": "collection_timeout",
        "ConnectionRefusedError": "redis_connection_refused",
    }
    return mapping.get(type_name, "collection_failed")


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_metric_sample(
    snapshot: dict[str, Any],
    now_wall: float,
    dt_str: str,
    req_total: int,
    rps: float,
    uptime: float,
) -> "MetricSample":
    """Constrói MetricSample completo a partir do snapshot de runtime."""
    agent = snapshot.get("agent", {})
    slo = snapshot.get("slo", {})
    cache = snapshot.get("cache", {})
    llm = snapshot.get("llm", {})
    tws = snapshot.get("tws", {})
    system = snapshot.get("system", {})
    router_cache = snapshot.get("router_cache", {})

    cache_hits = _safe_int(cache.get("hits"))
    cache_misses = _safe_int(cache.get("misses"))
    cache_total = cache_hits + cache_misses

    router_hits = _safe_int(router_cache.get("hits"))
    router_misses = _safe_int(router_cache.get("misses"))
    router_total = router_hits + router_misses

    return MetricSample(
        timestamp=now_wall,
        datetime_str=dt_str,
        requests_total=req_total,
        requests_per_sec=rps,
        error_count=_safe_int(agent.get("creation_failures")),
        error_rate=_safe_float(slo.get("api_error_rate")) * 100,
        response_time_p50=_safe_float(slo.get("api_response_time_p50")) * 1000,
        response_time_p95=_safe_float(slo.get("api_response_time_p95")) * 1000,
        response_time_avg=_safe_float(slo.get("api_response_time_avg")) * 1000,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_hit_ratio=((cache_hits / cache_total) * 100) if cache_total > 0 else 0.0,
        cache_size=_safe_int(cache.get("size")),
        cache_evictions=_safe_int(cache.get("evictions")),
        router_cache_hits=router_hits,
        router_cache_misses=router_misses,
        router_cache_hit_ratio=((router_hits / router_total) * 100)
        if router_total > 0
        else 0.0,
        agents_active=_safe_int(agent.get("active_count")),
        agents_created=_safe_int(agent.get("initializations")),
        agents_failed=_safe_int(agent.get("creation_failures")),
        llm_requests=_safe_int(llm.get("requests")),
        llm_tokens_used=_safe_int(llm.get("tokens_used")),
        llm_errors=_safe_int(llm.get("errors")),
        tws_connected=_safe_float(slo.get("tws_connection_success_rate")) > 0.5,
        tws_latency_ms=_safe_float(tws.get("latency_ms")),
        tws_errors=_safe_int(tws.get("errors")),
        tws_requests_success=_safe_int(tws.get("success")),
        tws_requests_failed=_safe_int(tws.get("failed")),
        system_uptime=uptime,
        system_availability=_safe_float(slo.get("availability"), 1.0) * 100,
        async_operations_active=_safe_int(system.get("async_operations_active")),
        correlation_ids_active=_safe_int(system.get("correlation_ids_active")),
    )


def _get_redis() -> "redis_async.Redis":
    """Obtém o cliente Redis canônico da aplicação."""
    client = get_redis_client()
    if client is None:
        raise ConnectionError("Redis client indisponível")
    return client


def _jitter_seconds(base: float) -> float:
    """Retorna jitter em [0, 10% de base] sem RNG pseudo-aleatório."""
    if base <= 0:
        return 0.0
    max_millis = int(base * 100)
    return secrets.randbelow(max_millis + 1) / 1000.0


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

    # Router Cache
    router_cache_hits: int = 0
    router_cache_misses: int = 0
    router_cache_hit_ratio: float = 0.0

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
        self._cached_start_time: float | None = None

    async def compute_rate_and_add_sample(
        self, requests_total: int, now_wall: float, sample_builder: Any
    ) -> None:
        """Calcula RPS com estado persistido no Redis (consistente entre workers)."""
        try:
            redis = _get_redis()
            prev_req_raw, prev_time_raw = await asyncio.gather(
                redis.get(REDIS_KEY_PREV_REQUESTS),
                redis.get(REDIS_KEY_PREV_WALLTIME),
            )
            prev_requests = _safe_int(prev_req_raw)
            prev_walltime = _safe_float(prev_time_raw)

            time_delta = (
                now_wall - prev_walltime
                if prev_walltime > 0
                else SAMPLE_INTERVAL_SECONDS
            )
            req_delta = requests_total - prev_requests if prev_requests > 0 else 0
            if req_delta < 0:
                req_delta = 0

            rps = req_delta / time_delta if time_delta > 0 else 0.0

            pipe = redis.pipeline()
            pipe.set(REDIS_KEY_PREV_REQUESTS, str(requests_total))
            pipe.set(REDIS_KEY_PREV_WALLTIME, str(now_wall))
            await pipe.execute()

            sample = sample_builder(max(0.0, rps))
            await self.add_sample(sample)
        except Exception as e:
            logger.error("Falha ao calcular RPS; usando 0 (%s)", type(e).__name__)
            sample = sample_builder(0.0)
            await self.add_sample(sample)

    async def add_error_sample(self, error: Exception) -> None:
        """Persiste amostra indicando falha na coleta."""
        now_wall = time.time()
        dt_str = _utc_iso_now()
        error_msg = _classify_collection_error(error)
        global_uptime = await self.get_global_uptime()

        # Tenta determinar conexão real se possível, senão assume falso
        tws_status = False
        try:
            snapshot = runtime_metrics.get_snapshot()
            tws_status = (
                snapshot.get("slo", {}).get("tws_connection_success_rate", 0) > 0.5
            )
        except Exception as e:
            logger.debug("Failed to get TWS status: %s", e)

        sample = MetricSample(
            timestamp=now_wall,
            datetime_str=dt_str,
            collection_error=error_msg,
            system_uptime=global_uptime,
            system_availability=0.0,
            tws_connected=tws_status,
        )
        await self.add_sample(sample)

    async def add_sample(self, sample: MetricSample) -> None:
        """Persiste amostra no Redis e gera alertas."""
        try:
            redis = _get_redis()
            data = json_dumps(asdict(sample))
            new_alerts = self._compute_alerts(sample)

            pipe = redis.pipeline()
            pipe.lpush(REDIS_KEY_HISTORY, data)
            pipe.ltrim(REDIS_KEY_HISTORY, 0, MAX_SAMPLES - 1)
            pipe.set(REDIS_KEY_LATEST, data)
            for alert in new_alerts:
                pipe.lpush(REDIS_KEY_ALERTS, json_dumps(alert))
            if new_alerts:
                pipe.ltrim(REDIS_KEY_ALERTS, 0, 19)
            await pipe.execute()
        except Exception as e:
            logger.error("Falha ao persistir amostra no Redis (%s)", type(e).__name__)

    def _compute_alerts(self, sample: MetricSample) -> list[dict[str, Any]]:
        """Computa alertas localmente."""
        new_alerts = []
        if sample.error_rate > 5.0:
            new_alerts.append(
                {
                    "type": "error_rate",
                    "severity": "critical" if sample.error_rate >= 10 else "warning",
                    "message": f"Error rate: {sample.error_rate:.1f}%",
                    "timestamp": sample.datetime_str,
                }
            )
        if sample.collection_error is None and not sample.tws_connected:
            new_alerts.append(
                {
                    "type": "tws",
                    "severity": "critical",
                    "message": "TWS desconectado",
                    "timestamp": sample.datetime_str,
                }
            )

        return new_alerts

    async def get_global_uptime(self) -> float:
        """Obtém o tempo de uptime global via Redis."""
        now = time.time()
        if self._cached_start_time is not None:
            return now - self._cached_start_time

        try:
            redis = _get_redis()
            await redis.set(REDIS_KEY_START_TIME, str(now), nx=True)
            raw = await redis.get(REDIS_KEY_START_TIME)
            if raw is None:
                logger.warning(
                    "Não foi possível determinar o tempo de início global do Redis."
                )
                return 0.0
            self._cached_start_time = float(raw)
            return now - self._cached_start_time
        except Exception as e:
            logger.exception(
                "Erro ao obter uptime global do Redis (%s)", type(e).__name__
            )
            return 0.0

    async def get_current_metrics(self) -> dict[str, Any]:
        """Retorna métricas atuais."""
        try:
            redis = _get_redis()
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
            logger.error("Erro ao obter métricas (%s)", type(e).__name__)
            return self._empty_response("error")

    async def get_history(self, minutes: int = 120) -> dict[str, Any]:
        """Retorna histórico de métricas."""
        try:
            redis = _get_redis()
            needed = (minutes * 60) // SAMPLE_INTERVAL_SECONDS
            raw_list = await redis.lrange(REDIS_KEY_HISTORY, 0, needed - 1)
            if not raw_list:
                return self._empty_history()

            raw_list.reverse()
            samples = [p for r in raw_list if (p := _safe_json_loads(r, "history"))]
            if not samples:
                return self._empty_history()

            return {
                "timestamps": [s.get("datetime_str", "") for s in samples],
                "api": {
                    "requests_per_sec": [
                        round(s.get("requests_per_sec", 0), 1) for s in samples
                    ],
                    "error_rate": [round(s.get("error_rate", 0), 2) for s in samples],
                },
                "cache": {
                    "hit_ratio": [
                        round(s.get("cache_hit_ratio", 0), 1) for s in samples
                    ]
                },
                "router_cache": {
                    "hit_ratio": [
                        round(s.get("router_cache_hit_ratio", 0), 1) for s in samples
                    ],
                    "operations": [
                        s.get("router_cache_hits", 0) + s.get("router_cache_misses", 0)
                        for s in samples
                    ],
                },
                "agents": {"active": [s.get("agents_active", 0) for s in samples]},
                "sample_count": len(samples),
            }
        except Exception as e:
            logger.error("Erro ao obter histórico (%s)", type(e).__name__)
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
                "error_rate": round(current.get("error_rate", 0), 2),
            },
            "system": {
                "availability": round(current.get("system_availability", 100), 2)
            },
            "router_cache": {
                "hits": current.get("router_cache_hits", 0),
                "misses": current.get("router_cache_misses", 0),
                "hit_ratio": round(current.get("router_cache_hit_ratio", 0), 1),
            },
            "alerts": alerts,
            "collection_error": err_msg,
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
            "router_cache": {"hit_ratio": [], "operations": []},
            "agents": {"active": []},
            "sample_count": 0,
        }


# ── WebSocket Manager ───────────────────────────────────────────────────────


class WebSocketManager:
    """Gerencia conexões WebSocket locais e sincroniza via Redis Pub/Sub."""

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._sync_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start_sync(self) -> bool:
        """Inicia o listener de Pub/Sub.

        Returns:
            True if started successfully, False if already stopped or already running.
        """
        if self._stop_event.is_set():
            logger.debug(
                "start_sync called but stop_event is set - call restart() to reset"
            )
            return False
        if self._sync_task is None or self._sync_task.done():
            self._stop_event.clear()
            self._sync_task = asyncio.create_task(self._pubsub_listener())
            return True
        return False

    async def restart(self) -> bool:
        """Reinicia o listener de Pub/Sub após stop().

        Returns:
            True if restarted successfully.
        """
        self._stop_event.clear()
        return await self.start_sync()

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
                logger.warning(
                    "Limite de conexões WebSocket atingido: %d", MAX_WS_CONNECTIONS
                )
                return False
            self._clients.add(websocket)
            return True

    async def disconnect(self, websocket: WebSocket) -> None:
        """Desconecta um WebSocket."""
        async with self._lock:
            self._clients.discard(websocket)

    async def _subscribe_to_broadcast(self):
        """Cria e subscreve um cliente pubsub para o canal de broadcast."""
        redis = _get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(REDIS_CH_BROADCAST)
        return pubsub

    async def _handle_pubsub_message(self, message: dict) -> None:
        """Processa uma mensagem recebida do pubsub."""
        if message.get("type") != "message":
            return

        data = message.get("data")
        if isinstance(data, bytes):
            data = data.decode()
        await self._local_broadcast(data)

    async def _close_pubsub(self, pubsub) -> None:
        """Fecha conexão pubsub com cleanup resiliente e logging explícito."""
        if pubsub is None:
            return

        try:
            await pubsub.unsubscribe(REDIS_CH_BROADCAST)
        except Exception as e:
            logger.warning("PubSub unsubscribe failed: %s", e)

        close_fn = getattr(pubsub, "close", None)
        if close_fn is not None:
            try:
                maybe_awaitable = close_fn()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            except Exception as e:
                logger.warning("PubSub close failed: %s", e)

    async def _pubsub_listener(self) -> None:
        """Listener do Redis Pub/Sub para broadcast."""
        backoff = 1.0
        max_backoff = 60.0
        while not self._stop_event.is_set():
            pubsub = None
            try:
                pubsub = await self._subscribe_to_broadcast()
                backoff = 1.0
                logger.info("WebSocketManager sincronizado com Redis Pub/Sub")
                async for message in pubsub.listen():
                    if self._stop_event.is_set():
                        break
                    await self._handle_pubsub_message(message)
            except asyncio.CancelledError:
                logger.info("WebSocketManager Pub/Sub listener cancelado")
                raise
            except Exception:
                logger.exception("Pub/Sub desconectado; tentando reconectar")
                if not self._stop_event.is_set():
                    jitter = _jitter_seconds(backoff)
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, max_backoff)
            finally:
                await self._close_pubsub(pubsub)

    async def broadcast(self, message_str: str) -> None:
        """Broadcast de mensagem para todos os clientes."""
        await self._local_broadcast(message_str)

    async def _local_broadcast(self, message_str: str) -> None:
        """Envia mensagem para todos os clientes locais."""
        async with self._lock:
            clients = list(self._clients)

        async def _safe_send(ws: WebSocket):
            try:
                await asyncio.wait_for(
                    ws.send_text(message_str), timeout=WS_SEND_TIMEOUT
                )
            except Exception:
                logger.debug(
                    "Falha ao enviar mensagem WebSocket; desconectando cliente"
                )
                await self.disconnect(ws)

        if clients:
            tasks = [asyncio.create_task(_safe_send(c)) for c in clients]
            await asyncio.gather(*tasks, return_exceptions=True)


# ── WebSocket Authentication ─────────────────────────────────────────────────


def _verify_ws_admin(websocket: WebSocket) -> str | None:
    """Valida autenticação admin para WebSocket.

    Aceita token apenas do header Authorization (não aceita query params por segurança).
    """
    try:
        # Security: Only accept token from Authorization header (not query params)
        auth_header = websocket.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]
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
        logger.debug("WebSocket authentication failed: %s", type(e).__name__)
        return None


# ── Singletons ───────────────────────────────────────────────────────────────

_metrics_store: DashboardMetricsStore | None = None
_ws_manager: WebSocketManager | None = None
_singleton_lock = threading.Lock()


def get_metrics_store() -> DashboardMetricsStore:
    global _metrics_store
    if _metrics_store is None:
        with _singleton_lock:
            if _metrics_store is None:
                _metrics_store = DashboardMetricsStore()
    return _metrics_store


def get_ws_manager() -> WebSocketManager:
    global _ws_manager
    if _ws_manager is None:
        with _singleton_lock:
            if _ws_manager is None:
                _ws_manager = WebSocketManager()
    return _ws_manager


# ── Collector Logic ──────────────────────────────────────────────────────────


async def collect_metrics_sample() -> None:
    """Apenas um worker coleta por vez (Liderança via Redis Lock)."""
    redis = _get_redis()

    lock = redis.lock(REDIS_LOCK_COLLECTOR, timeout=15)
    if not await lock.acquire(blocking=False):
        return  # Outro worker já está coletando

    try:
        snapshot = runtime_metrics.get_snapshot()
        now_wall = time.time()
        dt_str = _utc_iso_now()

        agent = snapshot.get("agent", {})
        req_total = _safe_int(agent.get("initializations"))
        store = get_metrics_store()
        uptime = await store.get_global_uptime()

        def build_sample(rps: float) -> MetricSample:
            return _build_metric_sample(
                snapshot, now_wall, dt_str, req_total, rps, uptime
            )

        await store.compute_rate_and_add_sample(req_total, now_wall, build_sample)

        current = await store.get_current_metrics()
        subscribers = await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        if subscribers == 0:
            logger.debug("Nenhum subscriber no canal de broadcast")

    except Exception as e:
        logger.error("Erro na coleta (%s)", type(e).__name__)
        try:
            store = get_metrics_store()
            await store.add_error_sample(e)
            current = await store.get_current_metrics()
            await redis.publish(REDIS_CH_BROADCAST, json_dumps(current))
        except Exception:
            logger.debug("Falha ao persistir/broadcast amostra de erro")
    finally:
        try:
            await lock.release()
        except Exception as lock_error:
            logger.debug("Lock release failed (possibly expired): %s", lock_error)


async def metrics_collector_loop() -> None:
    """Loop de coleta de métricas em background."""
    try:
        redis = _get_redis()
        await redis.ping()
    except Exception as e:
        logger.exception(
            "Redis não disponível, encerrando collector (%s)", type(e).__name__
        )
        return

    ws_manager = get_ws_manager()
    await ws_manager.start_sync()
    try:
        while True:
            try:
                await collect_metrics_sample()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Erro no collector loop ao executar collect_metrics_sample"
                )
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
    finally:
        await ws_manager.stop()


# ── FastAPI Router ───────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/current", dependencies=[Depends(require_role("admin"))])
async def get_current():
    """Retorna métricas atuais do sistema. Requer autenticação admin."""
    return await get_metrics_store().get_current_metrics()


@router.get("/history", dependencies=[Depends(require_role("admin"))])
async def get_history(minutes: int = 120):
    """Retorna histórico de métricas para gráficos. Requer autenticação admin."""
    return await get_metrics_store().get_history(min(max(1, minutes), 120))


@router.websocket("/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket para métricas em tempo real com autenticação."""
    username = _verify_ws_admin(websocket)
    if not username:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    ws_manager = get_ws_manager()

    if not await ws_manager.connect(websocket):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    try:
        initial = await get_metrics_store().get_current_metrics()
        await websocket.send_json(initial)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)
