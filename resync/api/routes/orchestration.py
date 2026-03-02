"""Orchestration API endpoints.

Provides REST and WebSocket endpoints for workflow orchestration,
including execution, status monitoring, and real-time event streaming.

Critical fixes applied:
- P0-15: WebSocket subscription cleanup to prevent memory leak
- P1-23: Validation of bg_tasks availability in runner initialization
- P2-36: Safe WebSocket close handling after errors
"""

import asyncio
import logging
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from resync.api.dependencies_v2 import get_database
from resync.core.database.engine import get_db_session
from resync.core.database.repositories.orchestration_config_repo import (
    OrchestrationConfigRepository,
)
from resync.core.database.repositories.orchestration_execution_repo import (
    OrchestrationExecutionRepository,
)
from resync.core.orchestration.runner import OrchestrationRunner
from resync.core.orchestration.events import event_bus, EventType, OrchestrationEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestration", tags=["Orchestration"])

# Dependency for Runner
def get_runner(request: Request):
    """Get OrchestrationRunner instance with structured concurrency.
    
    P1-23 fix: Validates bg_tasks availability to prevent silent failures.
    """
    bg_tasks = getattr(request.app.state, "bg_tasks", None)
    if not bg_tasks:
        # P1-23 fix: Fail loudly instead of silently returning broken runner
        logger.error(
            "bg_tasks_not_initialized",
            hint="Check lifespan initialization in main.py",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestration runner not available. Background task group not initialized.",
        )

    return OrchestrationRunner(session_factory=get_db_session, tg=bg_tasks)

# --- Config Endpoints ---

@router.post("/configs", status_code=status.HTTP_201_CREATED)
async def create_config(
    name: str,
    strategy: str,
    steps: dict,  # Accepts raw dict, validates partially?
    description: str | None = None,
    db: AsyncSession = Depends(get_database),
):
    """Create a new orchestration configuration."""
    repo = OrchestrationConfigRepository(db)

    # Basic validation
    # TODO: Validate 'steps' against WorkflowConfig schema?
    try:
        # Check if steps have "steps" key list
        steps_list = steps.get("steps") if isinstance(steps, dict) else steps
        if not isinstance(steps_list, list):
            raise ValueError("Steps must be a list or dict with 'steps' key")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        raise HTTPException(status_code=400, detail=str(e))

    config = await repo.create(
        name=name, strategy=strategy, steps=steps, description=description
    )
    return config

@router.get("/configs")
async def list_configs(
    limit: int = 100, offset: int = 0, db: AsyncSession = Depends(get_database)
):
    """List orchestration configurations."""
    repo = OrchestrationConfigRepository(db)
    return await repo.list_all(limit=limit, offset=offset)

@router.get("/configs/{config_id}")
async def get_config(config_id: UUID, db: AsyncSession = Depends(get_database)):
    """Get configuration by ID."""
    repo = OrchestrationConfigRepository(db)
    config = await repo.get_by_id(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    return config

# --- Execution Endpoints ---

@router.post("/execute/{config_id}", status_code=status.HTTP_202_ACCEPTED)
async def execute_workflow(
    config_id: UUID,
    input_data: dict,
    background_tasks: BackgroundTasks,
    runner: OrchestrationRunner = Depends(get_runner),
):
    """
    Start a workflow execution.
    Returns trace_id immediately.
    """
    try:
        trace_id = await runner.start_execution(
            config_id, input_data, user_id="api_user"
        )  # TODO: get user from auth
        return {"trace_id": trace_id, "status": "accepted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("Execution start failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/status/{trace_id}")
async def get_execution_status(trace_id: str, db: AsyncSession = Depends(get_database)):
    """Get execution status by trace ID."""
    repo = OrchestrationExecutionRepository(db)
    execution = await repo.get_by_trace_id(trace_id)

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Get steps
    steps = await repo.get_step_runs(execution.id)

    return {
        "execution_id": execution.id,
        "trace_id": execution.trace_id,
        "status": execution.status,
        "config_id": execution.config_id,
        "created_at": execution.created_at,
        "steps": [
            {
                "step_id": s.step_id,
                "status": s.status,
                "output": s.output,
                "error": s.error_message,
            }
            for s in steps
        ],
    }

# --- WebSocket ---

@router.websocket("/ws/execute/{config_id}")
async def websocket_execute(
    websocket: WebSocket,
    config_id: UUID,
    runner: OrchestrationRunner = Depends(get_runner),
):
    """
    WebSocket endpoint to start execution and stream events.
    Protocol:
    1. Client connects.
    2. Client sends JSON: {"input": {...}}
    3. Server starts execution and streams events.
    4. Server closes connection when done.
    
    P0-15 fix: Proper subscription cleanup to prevent memory leak.
    """
    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue()
    subscription_id: str | None = None  # Initialize for finally block
    trace_id: str | None = None  # Initialize for logging

    try:
        # Wait for input
        data = await websocket.receive_json()
        input_data = data.get("input", {})

        # Start execution
        trace_id = await runner.start_execution(
            config_id, input_data, user_id="ws_user"
        )

        # Subscribe to events for this trace_id
        async def filtered_handler(event: OrchestrationEvent):
            if event.trace_id == trace_id:
                await queue.put(event)

        # P0-15 fix: Move subscription inside try to ensure it's set before cleanup
        subscription_id = event_bus.subscribe_all(filtered_handler)
        logger.info(
            "websocket_subscription_created",
            subscription_id=subscription_id,
            trace_id=trace_id,
        )

        await websocket.send_json({"type": "started", "trace_id": trace_id})

        # Stream loop
        while True:
            event = await queue.get()

            # Send to client
            await websocket.send_json(
                {
                    "type": event.type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "step_id": event.step_id,
                    "data": event.data,
                }
            )

            if event.type in (
                EventType.EXECUTION_COMPLETED,
                EventType.EXECUTION_FAILED,
            ):
                # End of stream
                break

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", trace_id=trace_id)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        logger.error("websocket_error", error=str(e), trace_id=trace_id)
        # P2-36 fix: Safe close - wrap in try/except to prevent masking original error
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_exc:
            # Close can fail if connection already closed - expected in many scenarios
            logger.debug(
                "websocket_close_after_error_failed",
                error=str(close_exc),
                trace_id=trace_id,
            )
    finally:
        # P0-15 fix: Always cleanup subscription to prevent memory leak
        # Defensive check: only unsubscribe if subscription was created
        if subscription_id is not None:
            event_bus.unsubscribe(subscription_id)
            logger.info(
                "websocket_subscription_cleaned_up",
                subscription_id=subscription_id,
                trace_id=trace_id if trace_id else "unknown",
            )
        else:
            logger.debug(
                "websocket_cleanup_skipped_no_subscription",
                trace_id=trace_id if trace_id else "unknown",
            )
