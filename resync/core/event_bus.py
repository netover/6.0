"""
Event Bus - Sistema de Broadcast de Eventos em Tempo Real

Versão 5.3 — Hardened Hybrid

Combina as melhores decisões de três análises independentes:
  - Fila limitada com backpressure e métrica de descarte (Gemini)
  - Timeout em send_text para evitar head-of-line blocking (Gemini)
  - Pré-serialização JSON única por evento (Gemini)
  - EventBusConfig imutável via Pydantic v2 (Gemini)
  - WebSocketProtocol exportado para desacoplamento e testes (Code Review)
  - deque tipada como deque[dict[str, Any]] (Code Review)
  - client.client_id corrigido em broadcast_message (Code Review)
  - Snapshot em _notify_subscribers sem lock desnecessário (Adpta)
  - _should_deliver com mapeamento de prioridade explícita (Adpta)
  - enable_persistence com NotImplementedError explícito (Adpta)
  - start() com guard de event loop ativo (Adpta)
  - subscribe/unsubscribe síncronos preservados (retrocompatibilidade)
  - get_event_bus() com RuntimeError explícito (falha rápida)
  - Histórico apenas em eventos enfileirados com sucesso (consistência)

Autor: Resync Team
Versão: 5.3
"""

import asyncio
import contextlib
import inspect
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from resync.core.task_tracker import track_task

logger = structlog.get_logger(__name__)


# =============================================================================
# PROTOCOL — desacoplamento do FastAPI/Starlette para testes
# =============================================================================

@runtime_checkable
class WebSocketProtocol(Protocol):
    """
    Protocolo mínimo para objetos WebSocket.

    Exportado publicamente para facilitar mocks em testes unitários
    sem depender do Starlette diretamente.

    Exemplo de mock:
        class FakeWS:
            async def send_text(self, data: str) -> None: ...
            async def close(self, code: int = 1000) -> None: ...
    """

    async def send_text(self, data: str) -> None:
        ...

    async def close(self, code: int = 1000) -> None:
        ...


# =============================================================================
# CONFIGURAÇÃO — Pydantic v2, imutável e validada na instanciação
# =============================================================================

class EventBusConfig(BaseModel):
    """
    Configuração imutável do EventBus.

    Validada pelo Pydantic v2 na criação — parâmetros inválidos
    explodem na instanciação, não em runtime obscuro.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    history_size: int = Field(default=1000, gt=0)
    max_queue_size: int = Field(default=10_000, gt=0)
    websocket_send_timeout: float = Field(default=1.5, gt=0.0)
    enable_persistence: bool = False


# =============================================================================
# TIPOS DE ASSINATURA
# =============================================================================

class SubscriptionType(str, Enum):
    """Tipos de assinatura para filtro de eventos."""

    ALL = "all"
    JOBS = "jobs"
    WORKSTATIONS = "ws"
    SYSTEM = "system"
    CRITICAL = "critical"


# Mapeamento de prioridade explícita para _should_deliver.
# Ordem importa: "critical" é avaliado antes de "system", que é avaliado
# antes de "job" — corrige o bug onde "system_job_error" caía em JOBS.
_SUBSCRIPTION_PRIORITY: list[tuple[str, SubscriptionType]] = [
    ("critical",    SubscriptionType.CRITICAL),
    ("system",      SubscriptionType.SYSTEM),
    ("job",         SubscriptionType.JOBS),
    ("workstation", SubscriptionType.WORKSTATIONS),
    ("ws_",         SubscriptionType.WORKSTATIONS),
]


# =============================================================================
# DATACLASSES — slots=True para eficiência de memória por conexão
# =============================================================================

@dataclass(slots=True)
class Subscriber:
    """Representa um assinante interno com callback."""

    subscriber_id: str
    callback: Callable
    subscription_types: set[SubscriptionType]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events_received: int = 0


@dataclass(slots=True)
class WebSocketClient:
    """Representa um cliente WebSocket conectado."""

    client_id: str
    websocket: WebSocketProtocol  # tipado via Protocol, não Any
    subscription_types: set[SubscriptionType]
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages_sent: int = 0
    last_ping: datetime | None = None


# =============================================================================
# EVENT BUS
# =============================================================================

class EventBus:
    """
    Barramento de eventos assíncrono com suporte a WebSocket.

    Características desta versão:
    - Publicação síncrona e não-bloqueante (segura para rotas FastAPI)
    - Fila limitada com política de descarte explícita (sem OOM)
    - Timeout em todos os send_text (sem head-of-line blocking)
    - Pré-serialização JSON única por evento (eficiência de CPU)
    - Broadcast com remoção automática de clientes zumbis
    - subscribe/unsubscribe síncronos (retrocompatibilidade)
    - Histórico consistente com a fila (sem eventos fantasmas)
    """

    def __init__(self, config: EventBusConfig | None = None) -> None:
        self.config = config or EventBusConfig()

        if self.config.enable_persistence:
            raise NotImplementedError(
                "Persistência de eventos ainda não implementada. "
                "Use enable_persistence=False até a versão 6.0."
            )

        # Subscribers — acesso síncrono, snapshot antes de iterar
        self._subscribers: dict[str, Subscriber] = {}

        # WebSocket clients — acesso protegido por lock
        self._websocket_clients: dict[str, WebSocketClient] = {}

        # Histórico tipado — populado apenas após enfileiramento bem-sucedido
        self._event_history: deque[dict[str, Any]] = deque(
            maxlen=self.config.history_size
        )

        # Métricas
        self._events_published: int = 0
        self._events_delivered: int = 0
        self._delivery_errors: int = 0
        self._dropped_events: int = 0

        # Primitivas asyncio — inicializadas em start(), não aqui
        self._websocket_lock: asyncio.Lock | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._processor_task: asyncio.Task | None = None
        self._is_running: bool = False

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Inicia o processamento de eventos.

        Deve ser chamado dentro de um contexto assíncrono ativo.
        Todas as primitivas asyncio são inicializadas aqui para
        garantir associação correta ao event loop em execução.

        Args:
            tg: TaskGroup opcional. Se fornecido, a task de processamento
                roda sob supervisão do grupo. Caso contrário, usa track_task.

        Raises:
            RuntimeError: Se chamado fora de um event loop ativo.
        """
        if self._is_running:
            return

        # Guard obrigatório — asyncio.Lock/Queue exigem loop ativo no Python 3.10+
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "EventBus.start() deve ser chamado dentro de um contexto assíncrono ativo. "
                "Garanta que start() seja chamado em um lifespan ou handler async."
            )

        # Inicialização de primitivas no loop correto
        self._websocket_lock = asyncio.Lock()
        self._event_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._is_running = True

        if tg:
            self._processor_task = tg.create_task(
                self._process_events(), name="event_bus_worker"
            )
        else:
            self._processor_task = track_task(
                self._process_events(), name="event_bus_worker"
            )

        logger.info(
            "event_bus_started",
            max_queue_size=self.config.max_queue_size,
            history_size=self.config.history_size,
            method="task_group" if tg else "track_task",
        )

    async def stop(self) -> None:
        """
        Para o processamento de eventos graciosamente.

        Cancela a task de processamento e aguarda sua finalização.
        Loga métricas de uso antes de encerrar.
        """
        self._is_running = False

        if self._processor_task:
            self._processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processor_task
            self._processor_task = None

        logger.info(
            "event_bus_stopped",
            events_published=self._events_published,
            events_delivered=self._events_delivered,
            delivery_errors=self._delivery_errors,
            dropped_events=self._dropped_events,
        )

    # =========================================================================
    # SUBSCRIPTIONS — síncronos por retrocompatibilidade
    # =========================================================================

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable,
        subscription_types: set[SubscriptionType] | None = None,
    ) -> None:
        """
        Registra um subscriber síncrono.

        A segurança de iteração é garantida via snapshot em
        _notify_subscribers, não por lock aqui.

        Args:
            subscriber_id: ID único do subscriber.
            callback: Função síncrona ou coroutine a ser chamada com o evento.
            subscription_types: Tipos de evento a receber. None = todos.
        """
        self._subscribers[subscriber_id] = Subscriber(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_types=subscription_types or {SubscriptionType.ALL},
        )
        logger.info("subscriber_added", subscriber_id=subscriber_id)

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove um subscriber. Seguro mesmo durante iteração ativa."""
        if self._subscribers.pop(subscriber_id, None) is not None:
            logger.info("subscriber_removed", subscriber_id=subscriber_id)

    # =========================================================================
    # WEBSOCKET MANAGEMENT
    # =========================================================================

    async def register_websocket(
        self,
        client_id: str,
        websocket: WebSocketProtocol,
        subscription_types: set[SubscriptionType] | None = None,
    ) -> None:
        """
        Registra um cliente WebSocket e envia eventos recentes filtrados.

        Args:
            client_id: ID único do cliente.
            websocket: Objeto que implementa WebSocketProtocol.
            subscription_types: Filtros de evento. None = todos.
        """
        assert self._websocket_lock is not None, "EventBus não iniciado. Chame start() primeiro."

        async with self._websocket_lock:
            self._websocket_clients[client_id] = WebSocketClient(
                client_id=client_id,
                websocket=websocket,
                subscription_types=subscription_types or {SubscriptionType.ALL},
            )

        logger.info("websocket_registered", client_id=client_id)

        # Envia histórico recente fora do lock — operação de rede pode ser lenta
        await self._send_recent_events(client_id)

    async def unregister_websocket(self, client_id: str) -> None:
        """Remove um cliente WebSocket de forma segura."""
        assert self._websocket_lock is not None

        async with self._websocket_lock:
            if self._websocket_clients.pop(client_id, None) is not None:
                logger.info("websocket_unregistered", client_id=client_id)

    async def update_websocket_subscriptions(
        self,
        client_id: str,
        subscription_types: set[SubscriptionType],
    ) -> None:
        """Atualiza filtros de assinatura de um cliente conectado."""
        assert self._websocket_lock is not None

        async with self._websocket_lock:
            if client := self._websocket_clients.get(client_id):
                client.subscription_types = subscription_types

    async def _send_recent_events(self, client_id: str, count: int = 50) -> None:
        """
        Envia eventos recentes filtrados para um cliente recém-conectado.

        Adquire o lock apenas para copiar a referência ao cliente.
        O envio ocorre fora do lock para não bloquear outros registros.
        """
        assert self._websocket_lock is not None

        async with self._websocket_lock:
            client = self._websocket_clients.get(client_id)
            if not client:
                return
            # Copia referência — o envio ocorre fora do lock
            client_ref = client

        # Filtra histórico pelo tipo de assinatura do cliente
        recent = [
            e for e in list(self._event_history)[-count:]
            if self._should_deliver(e.get("event_type", ""), client_ref.subscription_types)
        ]

        if not recent:
            return

        message = {
            "type": "recent_events",
            "events": recent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            await asyncio.wait_for(
                client_ref.websocket.send_text(json.dumps(message, default=str)),
                timeout=self.config.websocket_send_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "recent_events_timeout",
                client_id=client_id,
                events_count=len(recent),
            )
        except Exception as e:
            logger.warning(
                "recent_events_failed",
                client_id=client_id,
                error=str(e),
            )

    # =========================================================================
    # PUBLISHING — síncrono e não-bloqueante para rotas FastAPI
    # =========================================================================

    def publish(self, event: Any) -> None:
        """
        Publica um evento de forma síncrona e não-bloqueante.

        Usa put_nowait para nunca travar rotas FastAPI. Em caso de fila
        cheia, o evento é descartado com log e métrica — sem OOM.

        O evento é adicionado ao histórico APENAS após enfileiramento
        bem-sucedido, garantindo consistência entre histórico e entrega.

        Args:
            event: Evento a publicar. Suporta Pydantic v2 BaseModel
                   (model_dump), objetos com to_dict(), dicts ou qualquer
                   objeto (convertido via str).
        """
        assert self._event_queue is not None, "EventBus não iniciado. Chame start() primeiro."

        # Serialização com suporte a Pydantic v2 como prioridade
        if hasattr(event, "model_dump"):
            event_data: dict[str, Any] = event.model_dump(mode="json")
        elif hasattr(event, "to_dict"):
            event_data = event.to_dict()
        elif isinstance(event, dict):
            event_data = event
        else:
            event_data = {"data": str(event)}

        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Tenta enfileirar — falha graciosamente sob pressão
        try:
            self._event_queue.put_nowait(event_data)
        except asyncio.QueueFull:
            self._dropped_events += 1
            logger.error(
                "event_queue_full_dropped",
                event_type=event_data.get("event_type", "unknown"),
                dropped_total=self._dropped_events,
                queue_maxsize=self.config.max_queue_size,
            )
            return  # Não adiciona ao histórico — mantém consistência

        # Histórico e métrica apenas após enfileiramento bem-sucedido
        self._event_history.append(event_data)
        self._events_published += 1

    def publish_batch(self, events: list[Any]) -> None:
        """Publica múltiplos eventos sequencialmente."""
        for event in events:
            self.publish(event)

    # =========================================================================
    # PROCESSAMENTO INTERNO
    # =========================================================================

    async def _process_events(self) -> None:
        """
        Worker assíncrono de processamento da fila.

        Serializa o JSON uma única vez por evento antes de distribuir
        para subscribers e clientes WebSocket em paralelo.
        """
        assert self._event_queue is not None

        while self._is_running:
            try:
                event_data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )

                # Pré-serialização única — economiza CPU com N clientes conectados
                json_message = json.dumps(
                    {"type": "event", "event": event_data},
                    default=str,
                )

                # Notificação interna e broadcast em paralelo
                await asyncio.gather(
                    self._notify_subscribers(event_data),
                    self._broadcast_to_websockets(event_data, json_message),
                    return_exceptions=True,
                )

                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "event_processing_fatal",
                    error=str(e),
                    exc_info=True,
                )

    async def _notify_subscribers(self, event_data: dict[str, Any]) -> None:
        """
        Notifica subscribers internos.

        Tira snapshot da lista ANTES de iterar para evitar
        RuntimeError em caso de subscribe/unsubscribe concorrente.
        Não usa lock — coroutines no mesmo loop não têm preempção
        entre si exceto nos pontos de await.
        """
        event_type = event_data.get("event_type", "")

        # Snapshot garante iteração segura sem lock
        subscribers = list(self._subscribers.values())

        for subscriber in subscribers:
            if not self._should_deliver(event_type, subscriber.subscription_types):
                continue

            try:
                if inspect.iscoroutinefunction(subscriber.callback):
                    await subscriber.callback(event_data)
                else:
                    await asyncio.to_thread(subscriber.callback, event_data)

                subscriber.events_received += 1
                self._events_delivered += 1

            except Exception as e:
                self._delivery_errors += 1
                logger.error(
                    "subscriber_notification_error",
                    subscriber_id=subscriber.subscriber_id,
                    error=str(e),
                )

    async def _broadcast_to_websockets(
        self,
        event_data: dict[str, Any],
        json_message: str,  # pré-serializado em _process_events
    ) -> None:
        """
        Broadcast para todos os clientes WebSocket conectados.

        Cada envio tem timeout individual — um cliente zumbi não
        bloqueia os demais (head-of-line blocking eliminado).
        Clientes com falha são removidos automaticamente.
        """
        assert self._websocket_lock is not None

        event_type = event_data.get("event_type", "")

        async with self._websocket_lock:
            clients = list(self._websocket_clients.values())

        if not clients:
            return

        async def _send(client: WebSocketClient) -> str | None:
            if not self._should_deliver(event_type, client.subscription_types):
                return None

            try:
                await asyncio.wait_for(
                    client.websocket.send_text(json_message),
                    timeout=self.config.websocket_send_timeout,
                )
                client.messages_sent += 1
                self._events_delivered += 1
                return None

            except asyncio.TimeoutError:
                logger.warning(
                    "websocket_send_timeout",
                    client_id=client.client_id,
                    timeout=self.config.websocket_send_timeout,
                )
                return client.client_id

            except Exception as e:
                logger.warning(
                    "websocket_send_error",
                    client_id=client.client_id,
                    error=str(e),
                )
                return client.client_id

        results = await asyncio.gather(
            *[_send(c) for c in clients],
            return_exceptions=True,
        )

        disconnected: list[str] = []
        for r in results:
            if isinstance(r, str):
                disconnected.append(r)
            elif isinstance(r, Exception):
                # Exceção grave que escapou do _send (ex: CancelledError durante shutdown)
                logger.debug("severe_broadcast_exception", error=str(r))

        for client_id in disconnected:
            await self.unregister_websocket(client_id)

    # =========================================================================
    # FILTRO DE ENTREGA — mapeamento de prioridade explícita
    # =========================================================================

    def _should_deliver(
        self,
        event_type: str,
        subscription_types: set[SubscriptionType],
    ) -> bool:
        """
        Verifica se um evento deve ser entregue ao subscriber/cliente.

        Usa mapeamento de prioridade explícita (_SUBSCRIPTION_PRIORITY)
        para evitar o bug onde "system_job_error" caía em JOBS antes
        de SYSTEM por causa da ordem dos if/elif.

        Tipos desconhecidos são entregues a todos por padrão.
        """
        if SubscriptionType.ALL in subscription_types:
            return True

        evt_lower = event_type.lower()

        for prefix, sub_type in _SUBSCRIPTION_PRIORITY:
            if prefix in evt_lower:
                # Encontrou a categoria do evento — entrega apenas se subscrito
                return sub_type in subscription_types

        # event_type não mapeado — entrega a todos
        return True

    # =========================================================================
    # API PÚBLICA
    # =========================================================================

    def get_recent_events(self, count: int = 100) -> list[dict[str, Any]]:
        """Retorna os N eventos mais recentes do histórico."""
        return list(self._event_history)[-count:]

    def get_events_by_type(
        self,
        event_type: str,
        count: int = 50,
    ) -> list[dict[str, Any]]:
        """Retorna eventos de um tipo específico."""
        return [
            e for e in self._event_history
            if e.get("event_type") == event_type
        ][-count:]

    def get_critical_events(self, count: int = 20) -> list[dict[str, Any]]:
        """Retorna eventos com severity critical ou error."""
        return [
            e for e in self._event_history
            if e.get("severity") in ("critical", "error")
        ][-count:]

    def get_metrics(self) -> dict[str, Any]:
        """Retorna métricas completas do event bus."""
        assert self._event_queue is not None
        return {
            "is_running": self._is_running,
            "subscribers_count": len(self._subscribers),
            "websocket_clients_count": len(self._websocket_clients),
            "events_published": self._events_published,
            "events_delivered": self._events_delivered,
            "delivery_errors": self._delivery_errors,
            "dropped_events": self._dropped_events,
            "history_size": len(self._event_history),
            "queue_size": self._event_queue.qsize(),
            "queue_maxsize": self.config.max_queue_size,
        }

    def get_connected_clients(self) -> list[dict[str, Any]]:
        """Retorna informações dos clientes WebSocket conectados."""
        return [
            {
                "client_id": client.client_id,
                "subscription_types": [t.value for t in client.subscription_types],
                "connected_at": client.connected_at.isoformat(),
                "messages_sent": client.messages_sent,
            }
            for client in self._websocket_clients.values()
        ]

    async def broadcast_message(self, message: dict[str, Any]) -> int:
        """
        Broadcast de mensagem arbitrária para todos os clientes conectados.

        Não aplica filtros de subscription_types — envia para todos.
        Usa timeout individual por cliente.

        Returns:
            Número de clientes que receberam a mensagem com sucesso.
        """
        assert self._websocket_lock is not None

        msg_json = json.dumps(message, default=str)

        async with self._websocket_lock:
            clients = list(self._websocket_clients.values())

        async def _send(client: WebSocketClient) -> bool:
            try:
                await asyncio.wait_for(
                    client.websocket.send_text(msg_json),
                    timeout=self.config.websocket_send_timeout,
                )
                return True
            except Exception as e:
                logger.debug(
                    "broadcast_message_failed",
                    client_id=client.client_id,  # corrigido: era getattr(client, "id")
                    error=str(e),
                )
                return False

        results = await asyncio.gather(
            *[_send(c) for c in clients],
            return_exceptions=True,
        )

        delivered = sum(1 for r in results if r is True)
        failed = len(results) - delivered

        if failed > 0:
            logger.debug(
                "broadcast_message_partial",
                delivered=delivered,
                failed=failed,
            )

        return delivered


# =============================================================================
# SINGLETON
# =============================================================================

_event_bus_instance: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Retorna o singleton do EventBus.

    Raises:
        RuntimeError: Se init_event_bus() não foi chamado antes.
                      Falha rápida e explícita — sem instância fantasma.
    """
    if _event_bus_instance is None:
        raise RuntimeError(
            "EventBus não inicializado. "
            "Chame init_event_bus() no lifespan da aplicação antes de usar get_event_bus()."
        )
    return _event_bus_instance


def init_event_bus(
    config: EventBusConfig | None = None,
    *,
    history_size: int = 1000,
    enable_persistence: bool = False,
) -> EventBus:
    """
    Inicializa o singleton do EventBus.

    Aceita um EventBusConfig completo ou parâmetros individuais
    para retrocompatibilidade com o código existente.

    Args:
        config: Configuração completa. Se fornecido, ignora os demais params.
        history_size: Tamanho do histórico em memória.
        enable_persistence: Reservado — levanta NotImplementedError se True.

    Returns:
        Instância configurada do EventBus (não iniciada — chame .start()).
    """
    global _event_bus_instance

    resolved_config = config or EventBusConfig(
        history_size=history_size,
        enable_persistence=enable_persistence,
    )

    _event_bus_instance = EventBus(config=resolved_config)
    return _event_bus_instance