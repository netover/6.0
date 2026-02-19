"""
Orchestration Event Bus

Manages internal events for the orchestration engine.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import UUID

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of orchestration events."""

    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_PAUSED = "execution.paused"

    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"

    AGENT_THINKING = "agent.thinking"
    AGENT_ACTION = "agent.action"
    AGENT_OBSERVATION = "agent.observation"


class OrchestrationEvent(BaseModel):
    """Event payload structure."""

    type: EventType
    trace_id: str
    execution_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    step_id: str | None = None
    step_index: int | None = None

    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Subscription:
    """Represents a subscription with an ID for tracking."""
    subscription_id: str
    callback: Callable
    event_type: EventType | None = None  # None means global subscription


class EventBus:
    """
    Simple in-memory event bus for orchestration events.
    Allows decoupling of Runner from WebSocket/Persistence/Logging.
    
    Supports subscription tracking to prevent memory leaks.
    """

    def __init__(self):
        self._subscribers: dict[EventType, list[Subscription]] = {}
        self._global_subscribers: dict[str, Subscription] = {}

    def subscribe(self, event_type: EventType, callback: Callable) -> str:
        """
        Subscribe to a specific event type.
        
        Returns:
            subscription_id that can be used to unsubscribe
        """
        subscription_id = str(uuid.uuid4())
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        subscription = Subscription(
            subscription_id=subscription_id,
            callback=callback,
            event_type=event_type
        )
        self._subscribers[event_type].append(subscription)
        
        logger.debug("subscribed_to_event id=%s event_type=%s", subscription_id, event_type.value)
        
        return subscription_id

    def subscribe_all(self, callback: Callable) -> str:
        """
        Subscribe to all events.
        
        Returns:
            subscription_id that can be used to unsubscribe
        """
        subscription_id = str(uuid.uuid4())
        
        subscription = Subscription(
            subscription_id=subscription_id,
            callback=callback,
            event_type=None  # Global subscription
        )
        self._global_subscribers[subscription_id] = subscription
        
        logger.debug("subscribed_to_all_events id=%s", subscription_id)
        
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe using the subscription_id returned from subscribe or subscribe_all.
        
        Returns:
            True if subscription was found and removed, False otherwise
        """
        # Check global subscribers first
        if subscription_id in self._global_subscribers:
            del self._global_subscribers[subscription_id]
            logger.debug("unsubscribed_from_all_events id=%s", subscription_id)
            return True
        
        # Check event-specific subscribers
        for event_type, subscriptions in self._subscribers.items():
            initial_count = len(subscriptions)
            self._subscribers[event_type] = [
                s for s in subscriptions if s.subscription_id != subscription_id
            ]
            if len(self._subscribers[event_type]) < initial_count:
                logger.debug("unsubscribed_from_event id=%s event_type=%s", subscription_id, event_type.value)
                return True
        
        logger.warning("subscription_not_found id=%s", subscription_id)
        return False

    async def publish(self, event: OrchestrationEvent):
        """Publish an event to all subscribers."""
        # Notify specific subscribers
        if event.type in self._subscribers:
            for subscription in self._subscribers[event.type]:
                try:
                    await self._invoke_callback(subscription.callback, event)
                except Exception as e:
                    logger.error(f"Error in event subscriber {subscription.subscription_id}: {e}")

        # Notify global subscribers
        for subscription in self._global_subscribers.values():
            try:
                await self._invoke_callback(subscription.callback, event)
            except Exception as e:
                logger.error(f"Error in global event subscriber {subscription.subscription_id}: {e}")

    async def _invoke_callback(self, callback: Callable, event: OrchestrationEvent):
        """Helper to invoke sync or async callbacks."""
        if asyncio.iscoroutinefunction(callback):
            await callback(event)
        else:
            callback(event)
    
    def get_subscription_count(self) -> int:
        """Get total number of active subscriptions (for debugging/monitoring)."""
        return len(self._global_subscribers) + sum(
            len(subs) for subs in self._subscribers.values()
        )


# Global instance
event_bus = EventBus()
