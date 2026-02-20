"""
Event Bus - Sistema de Broadcast de Eventos em Tempo Real

Este módulo implementa um barramento de eventos para comunicação
assíncrona entre componentes e broadcast via WebSocket.

Funcionalidades:
- Pub/Sub assíncrono
- Broadcast para todos os clientes WebSocket conectados
- Filtros por tipo de evento
- Histórico de eventos recentes
- Métricas de publicação

Autor: Resync Team
Versão: 5.2
"""

import asyncio
import inspect
from resync.core.task_tracker import track_task
import contextlib
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SubscriptionType(str, Enum):
    """Tipos de assinatura."""

    ALL = "all"  # Recebe todos os eventos
    JOBS = "jobs"  # Só eventos de jobs
    WORKSTATIONS = "ws"  # Só eventos de workstations
    SYSTEM = "system"  # Só eventos de sistema
    CRITICAL = "critical"  # Só eventos críticos


@dataclass
class Subscriber:
    """Representa um assinante."""

    subscriber_id: str
    callback: Callable
    subscription_types: set[SubscriptionType]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events_received: int = 0


@dataclass
class WebSocketClient:
    """Representa um cliente WebSocket."""

    client_id: str
    websocket: Any  # WebSocket object
    subscription_types: set[SubscriptionType]
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages_sent: int = 0
    last_ping: datetime | None = None


class EventBus:
    """
    Barramento de eventos com suporte a WebSocket.

    Características:
    - Publicação assíncrona de eventos
    - Múltiplos subscribers com filtros
    - Broadcast para clientes WebSocket
    - Histórico de eventos recentes
    - Métricas de uso
    """

    def __init__(
        self,
        history_size: int = 1000,
        enable_persistence: bool = False,
    ):
        """
        Inicializa o event bus.

        Args:
            history_size: Quantidade de eventos a manter em memória
            enable_persistence: Se deve persistir eventos
        """
        self.history_size = history_size
        self.enable_persistence = enable_persistence

        # Subscribers
        self._subscribers: dict[str, Subscriber] = {}

        # WebSocket clients
        self._websocket_clients: dict[str, WebSocketClient] = {}
        self._websocket_lock: asyncio.Lock | None = None

        # Histórico de eventos
        self._event_history: deque = deque(maxlen=history_size)

        # Métricas
        self._events_published = 0
        self._events_delivered = 0
        self._delivery_errors = 0

        # Queue para processamento assíncrono (lazy-initialized)
        self._event_queue: asyncio.Queue | None = None
        self._processor_task: asyncio.Task | None = None
        self._is_running = False

    @property
    def websocket_lock(self) -> asyncio.Lock:
        """Lazy initialization of websocket lock."""
        if self._websocket_lock is None:
            self._websocket_lock = asyncio.Lock()
        return self._websocket_lock

    @property
    def event_queue(self) -> asyncio.Queue:
        """Lazy initialization of event queue."""
        if self._event_queue is None:
            self._event_queue = asyncio.Queue()
        return self._event_queue

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
        """Inicia o processamento de eventos."""
        if self._is_running:
            return

        self._is_running = True
        self._processor_task = track_task(self._process_events(), name="process_events")
        logger.info("event_bus_started")

    async def stop(self) -> None:
        """Para o processamento de eventos."""
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
        )

    # =========================================================================
    # SUBSCRIPTIONS
    # =========================================================================

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable,
        subscription_types: set[SubscriptionType] | None = None,
    ) -> None:
        """
        Registra um subscriber.

        Args:
            subscriber_id: ID único do subscriber
            callback: Função a ser chamada com o evento
            subscription_types: Tipos de eventos a receber (None = todos)
        """
        if subscription_types is None:
            subscription_types = {SubscriptionType.ALL}

        self._subscribers[subscriber_id] = Subscriber(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_types=subscription_types,
        )

        logger.info(
            "subscriber_added",
            subscriber_id=subscriber_id,
            types=list(subscription_types),
        )

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove um subscriber."""
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]
            logger.info("subscriber_removed", subscriber_id=subscriber_id)

    # =========================================================================
    # WEBSOCKET MANAGEMENT
    # =========================================================================

    async def register_websocket(
        self,
        client_id: str,
        websocket: Any,
        subscription_types: set[SubscriptionType] | None = None,
    ) -> None:
        """
        Registra um cliente WebSocket.

        Args:
            client_id: ID único do cliente
            websocket: Objeto WebSocket
            subscription_types: Tipos de eventos a receber
        """
        if subscription_types is None:
            subscription_types = {SubscriptionType.ALL}

        async with self.websocket_lock:
            self._websocket_clients[client_id] = WebSocketClient(
                client_id=client_id,
                websocket=websocket,
                subscription_types=subscription_types,
            )

        logger.info(
            "websocket_client_registered",
            client_id=client_id,
            types=list(subscription_types),
        )

        # Envia eventos recentes ao novo cliente
        await self._send_recent_events(client_id)

    async def unregister_websocket(self, client_id: str) -> None:
        """Remove um cliente WebSocket."""
        async with self.websocket_lock:
            if client_id in self._websocket_clients:
                del self._websocket_clients[client_id]
                logger.info("websocket_client_unregistered", client_id=client_id)

    async def update_websocket_subscriptions(
        self,
        client_id: str,
        subscription_types: set[SubscriptionType],
    ) -> None:
        """Atualiza tipos de assinatura de um cliente."""
        async with self.websocket_lock:
            if client_id in self._websocket_clients:
                self._websocket_clients[
                    client_id
                ].subscription_types = subscription_types

    async def _send_recent_events(self, client_id: str, count: int = 50) -> None:
        """Envia eventos recentes para um cliente."""
        client = self._websocket_clients.get(client_id)
        if not client:
            return

        # Pega últimos N eventos
        recent = list(self._event_history)[-count:]

        try:
            # Envia como batch
            message = {
                "type": "recent_events",
                "events": recent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await client.websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(
                "failed_to_send_recent_events",
                client_id=client_id,
                error=str(e),
            )

    # =========================================================================
    # EVENT PUBLISHING
    # =========================================================================

    async def publish(self, event: Any) -> None:
        """
        Publica um evento.

        Args:
            event: Evento a ser publicado (deve ter método to_dict())
        """
        # Converte para dict se necessário
        if hasattr(event, "to_dict"):
            event_data = event.to_dict()
        elif isinstance(event, dict):
            event_data = event
        else:
            event_data = {"data": str(event)}

        # Adiciona timestamp se não existir
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Adiciona ao histórico
        self._event_history.append(event_data)

        # Coloca na fila para processamento
        await self.event_queue.put(event_data)

        self._events_published += 1

    async def publish_batch(self, events: list[Any]) -> None:
        """Publica múltiplos eventos."""
        for event in events:
            await self.publish(event)

    async def _process_events(self) -> None:
        """Processa eventos da fila."""
        while self._is_running:
            try:
                # Aguarda próximo evento (com timeout)
                event_data = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0,
                )

                # Notifica subscribers
                await self._notify_subscribers(event_data)

                # Broadcast para WebSockets
                await self._broadcast_to_websockets(event_data)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    async def _notify_subscribers(self, event_data: dict[str, Any]) -> None:
        """Notifica subscribers sobre um evento."""
        event_type = event_data.get("event_type", "")

        for subscriber in self._subscribers.values():
            # Verifica se subscriber quer este tipo de evento
            if not self._should_deliver(subscriber.subscription_types, event_type):
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

    async def _broadcast_to_websockets(self, event_data: dict[str, Any]) -> None:
        """Broadcast evento para todos os clientes WebSocket."""
        event_type = event_data.get("event_type", "")

        # Prepara mensagem
        message = json.dumps(
            {
                "type": "event",
                "event": event_data,
            },
            default=str,
        )

        # Copia lista para evitar modificação durante iteração
        async with self.websocket_lock:
            clients = list(self._websocket_clients.values())

        async def _send_to_client(client) -> str | None:
            if not self._should_deliver(client.subscription_types, event_type):
                return None
            try:
                await client.websocket.send_text(message)
                client.messages_sent += 1
                self._events_delivered += 1
                return None
            except Exception as e:
                logger.warning(
                    "websocket_send_error",
                    client_id=client.client_id,
                    error=str(e),
                )
                return client.client_id

        results = await asyncio.gather(
            *[_send_to_client(c) for c in clients], return_exceptions=False
        )
        disconnected = [cid for cid in results if cid]

        # Remove clientes desconectados
        for client_id in disconnected:
            await self.unregister_websocket(client_id)

    def _should_deliver(
        self,
        subscription_types: set[SubscriptionType],
        event_type: str,
    ) -> bool:
        """Verifica se evento deve ser entregue baseado no tipo."""
        if SubscriptionType.ALL in subscription_types:
            return True

        # Mapeia event_type para subscription_type
        if "job" in event_type.lower():
            return SubscriptionType.JOBS in subscription_types
        if "workstation" in event_type.lower() or "ws_" in event_type.lower():
            return SubscriptionType.WORKSTATIONS in subscription_types
        if "system" in event_type.lower():
            return SubscriptionType.SYSTEM in subscription_types
        if "critical" in event_type.lower():
            return SubscriptionType.CRITICAL in subscription_types

        return True  # Default: entrega

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_recent_events(self, count: int = 100) -> list[dict[str, Any]]:
        """Retorna eventos recentes."""
        return list(self._event_history)[-count:]

    def get_events_by_type(
        self,
        event_type: str,
        count: int = 50,
    ) -> list[dict[str, Any]]:
        """Retorna eventos de um tipo específico."""
        events = [e for e in self._event_history if e.get("event_type") == event_type]
        return events[-count:]

    def get_critical_events(self, count: int = 20) -> list[dict[str, Any]]:
        """Retorna eventos críticos."""
        events = [
            e for e in self._event_history if e.get("severity") in ["critical", "error"]
        ]
        return events[-count:]

    def get_metrics(self) -> dict[str, Any]:
        """Retorna métricas do event bus."""
        return {
            "is_running": self._is_running,
            "subscribers_count": len(self._subscribers),
            "websocket_clients_count": len(self._websocket_clients),
            "events_published": self._events_published,
            "events_delivered": self._events_delivered,
            "delivery_errors": self._delivery_errors,
            "history_size": len(self._event_history),
            "queue_size": self.event_queue.qsize(),
        }

    def get_connected_clients(self) -> list[dict[str, Any]]:
        """Retorna informações dos clientes conectados."""
        return [
            {
                "client_id": client.client_id,
                "subscription_types": list(client.subscription_types),
                "connected_at": client.connected_at.isoformat(),
                "messages_sent": client.messages_sent,
            }
            for client in self._websocket_clients.values()
        ]

    async def broadcast_message(self, message: dict[str, Any]) -> int:
        """
        Broadcast mensagem arbitrária para todos os clientes.

        Returns:
            Número de clientes que receberam a mensagem
        """
        msg_json = json.dumps(message, default=str)

        async with self.websocket_lock:
            clients = list(self._websocket_clients.values())

        async def _send(client) -> bool:
            try:
                await client.websocket.send_text(msg_json)
                return True
            except Exception as e:
                logger.debug(
                    "websocket_send_failed",
                    client_id=getattr(client, "id", "unknown"),
                    error=str(e),
                )
                return False

        results = await asyncio.gather(
            *[_send(c) for c in clients], return_exceptions=False
        )
        delivered = sum(1 for r in results if r)
        failed = len(results) - delivered

        if failed > 0:
            logger.debug(
                "websocket_broadcast_partial", delivered=delivered, failed=failed
            )

        return delivered


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_event_bus_instance: EventBus | None = None


def get_event_bus() -> EventBus | None:
    """Retorna instância singleton do event bus."""
    return _event_bus_instance


def init_event_bus(
    history_size: int = 1000,
    enable_persistence: bool = False,
) -> EventBus:
    """Inicializa o event bus singleton."""
    global _event_bus_instance

    _event_bus_instance = EventBus(
        history_size=history_size,
        enable_persistence=enable_persistence,
    )

    return _event_bus_instance
