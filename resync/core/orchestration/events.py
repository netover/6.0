"""
Orchestration Event Bus

Manages internal events for the orchestration engine.
"""
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


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
    timestamp: datetime = datetime.utcnow()
    
    step_id: Optional[str] = None
    step_index: Optional[int] = None
    
    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class EventBus:
    """
    Simple in-memory event bus for orchestration events.
    Allows decoupling of Runner from WebSocket/Persistence/Logging.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._global_subscribers: List[Callable] = []
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable):
        """Subscribe to all events."""
        self._global_subscribers.append(callback)
    
    async def publish(self, event: OrchestrationEvent):
        """Publish an event to all subscribers."""
        # Notify specific subscribers
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    await self._invoke_callback(callback, event)
                except Exception as e:
                    logger.error(f"Error in event subscriber {callback}: {e}")
        
        # Notify global subscribers
        for callback in self._global_subscribers:
            try:
                await self._invoke_callback(callback, event)
            except Exception as e:
                logger.error(f"Error in global event subscriber {callback}: {e}")
                
    async def _invoke_callback(self, callback: Callable, event: OrchestrationEvent):
        """Helper to invoke sync or async callbacks."""
        if asyncio.iscoroutinefunction(callback):
            await callback(event)
        else:
            callback(event)

# Global instance
event_bus = EventBus()
