import asyncio
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
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
    """Get OrchestrationRunner instance with structured concurrency."""
    bg_tasks = getattr(request.app.state, "bg_tasks", None)
    if not bg_tasks:
        # Fallback to local TaskGroup if not in lifespan (though less ideal for long-running)
        # However, for API requests, we should expect lifespan to be active.
        logger.warning("bg_tasks_not_found_in_app_state", hint="Check lifespan initialization")

    return OrchestrationRunner(
        session_factory=get_db_session,
        tg=bg_tasks
    )


# --- Config Endpoints ---


@router.post("/configs", status_code=status.HTTP_201_CREATED)
async def create_config(
    name: str,
    strategy: str,
    steps: dict,  # Accepts raw dict, validates partially?
    description: Optional[str] = None,
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
    except Exception as e:
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
    background_tasks: BackgroundTasks,  # Using FastAPI background tasks or Runner's internal?
    # Runner spawns its own asyncio task, but calling it from here relies on current loop.
    # It's better to await runner.start_execution which returns trace_id and spawns task.
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
    except Exception as e:
        logger.error(f"Execution start failed: {e}", exc_info=True)
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
    4. Server closes connection when done? Or keeps open?
    """
    await websocket.accept()

    queue = asyncio.Queue()
    subscription_id = None  # Initialize for finally block

    async def event_handler(event: OrchestrationEvent):
        await queue.put(event)

    try:
        # Wait for input
        data = await websocket.receive_json()
        input_data = data.get("input", {})

        # Start execution
        trace_id = await runner.start_execution(
            config_id, input_data, user_id="ws_user"
        )

        # Subscribe to events for this trace_id
        # We need a way to filter subscriptions in EventBus or filter here.
        # EventBus.subscribe accepts a callback.
        # We can wrap a filter.

        async def filtered_handler(event: OrchestrationEvent):
            if event.trace_id == trace_id:
                await queue.put(event)

        # Subscribe to all types we care about
        # Ideally EventBus supports wildcards or we verify in handler
        # Store the subscription_id so we can unsubscribe later
        subscription_id = event_bus.subscribe_all(filtered_handler)

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
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        # CRITICAL: Cleanup subscription to prevent memory leak
        # Unsubscribe when WebSocket disconnects
        if subscription_id:
            event_bus.unsubscribe(subscription_id)
            logger.info(
                "websocket_subscription_cleaned_up",
                subscription_id=subscription_id,
                trace_id=trace_id if "trace_id" in locals() else "unknown",
            )
