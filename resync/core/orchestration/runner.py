"""
Orchestration Runner

Core engine for executing orchestration workflows.
"""
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from resync.core.database.repositories.orchestration_config_repo import OrchestrationConfigRepository
from resync.core.database.repositories.orchestration_execution_repo import (
    OrchestrationExecutionRepository,
    OrchestrationStepRunRepository,
)
from resync.core.orchestration.agent_adapter import AgentAdapter
from resync.core.orchestration.events import EventBus, EventType, OrchestrationEvent, event_bus
from resync.core.orchestration.schemas import WorkflowConfig, StepConfig
from resync.core.orchestration.strategies import StrategyFactory

logger = logging.getLogger(__name__)


class OrchestrationRunner:
    """
    Executes orchestration workflows based on configuration and state strings.
    """
    
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.agent_adapter = AgentAdapter()
        self.event_bus = event_bus

    async def start_execution(self, config_id: UUID, input_data: dict, user_id: str | None = None) -> str:
        """
        Starts a new execution for a given config.
        Returns the trace_id.
        """
        async with self.session_factory() as session:
            config_repo = OrchestrationConfigRepository(session)
            exec_repo = OrchestrationExecutionRepository(session)
            
            config = await config_repo.get_by_id(config_id)
            if not config:
                raise ValueError(f"Configuration {config_id} not found")
            
            # Create execution record
            import uuid
            trace_id = str(uuid.uuid4())
            
            execution = await exec_repo.create(
                trace_id=trace_id,
                config_id=config_id,
                config_name=config.name,
                input_data=input_data,
                user_id=user_id,
            )
            
            # Fire event
            await self.event_bus.publish(OrchestrationEvent(
                type=EventType.EXECUTION_STARTED,
                trace_id=trace_id,
                execution_id=execution.id,
                data={"config_name": config.name, "input": input_data}
            ))

            # Start background processing
            # In a real system, this might be a Celery task or similar.
            # Here we run it as an asyncio task (fire and forget handled by caller or here?)
            # Ideally, we return trace_id immediately and run async.
            asyncio.create_task(self._run_loop(execution.id))
            
            return trace_id

    async def _run_loop(self, execution_id: UUID):
        """
        Main execution loop.
        """
        logger.info(f"Starting execution loop for {execution_id}")
        
        async with self.session_factory() as session:
            exec_repo = OrchestrationExecutionRepository(session)
            step_repo = OrchestrationStepRunRepository(session)
            config_repo = OrchestrationConfigRepository(session)
            
            execution = await exec_repo.get_by_id(execution_id)
            if not execution:
                logger.error(f"Execution {execution_id} not found in run loop")
                return

            try:
                # Update status to running
                await exec_repo.update_status(execution_id, "running")
                
                # Load config to get steps definition
                if not execution.config_id:
                     raise ValueError("Execution missing config_id")
                     
                db_config = await config_repo.get_by_id(execution.config_id)
                if not db_config:
                     raise ValueError("Config not found")
                
                # Parse config into Pydantic models
                # Ensure steps is a list of dicts first
                steps_data = db_config.steps
                if isinstance(steps_data, dict) and "steps" in steps_data:
                    steps_data = steps_data["steps"]
                
                workflow_config = WorkflowConfig(
                    version=str(db_config.version),
                    steps=[StepConfig(**s) for s in steps_data]
                )
                
                strategy = StrategyFactory.get_strategy(db_config.strategy)
                
                # Initialize execution context
                context = {
                    "execution_id": str(execution_id),
                    "trace_id": execution.trace_id,
                    "user_id": execution.user_id,
                    "global_input": execution.input_data,
                    "step_outputs": {}
                }
                
                # Loop until completion
                while True:
                    # Refresh state
                    # We need to know which steps are completed.
                    step_runs = await exec_repo.get_step_runs(execution_id)
                    completed_step_ids = {
                        run.step_id for run in step_runs 
                        if run.status == "completed"
                    }
                    
                    # Update context with outputs
                    for run in step_runs:
                        if run.status == "completed":
                            context["step_outputs"][run.step_id] = run.output
                    
                    # Plan next steps
                    batches = await strategy.plan_execution(workflow_config, completed_step_ids)
                    
                    if not batches:
                        # No more steps to run.
                        # Check if all steps in config are completed
                        all_config_ids = {s.id for s in workflow_config.steps}
                        if completed_step_ids >= all_config_ids:
                             logger.info("All steps completed.")
                             break
                        else:
                             # Stalled? Or waiting for external event?
                             # For now assume if plan returns empty and not all done, it's a deadlock or error?
                             # Or maybe some steps are just unreachable/skipped?
                             logger.warning(f"Plan returned empty but not all steps done. Completed: {completed_step_ids}, Total: {all_config_ids}")
                             break
                    
                    # Execute first batch (strategies might return multiple parallel batches sequence? 
                    # Usually just returns next runnable batch)
                    current_batch = batches[0]
                    
                    logger.info(f"Executing batch: {[s.id for s in current_batch]}")
                    
                    # Create step runs in DB
                    step_run_objects = []
                    for step in current_batch:
                         # Calculate index (simple append logic for now, or based on config index?)
                         # We'll use len(step_runs) + i
                         idx = len(step_runs) + len(step_run_objects)
                         s_run = await step_repo.create(
                             execution_id=execution.id,
                             step_index=idx,
                             step_id=step.id,
                             step_name=step.name,
                             dependencies_json=[d.dict() for d in step.dependencies]
                         )
                         step_run_objects.append((s_run, step))
                    
                    # Run in parallel
                    tasks = []
                    for s_run, step_config in step_run_objects:
                        tasks.append(
                            self._run_step(
                                step_repo, 
                                s_run.id, 
                                step_config, 
                                context
                            )
                        )
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results/errors
                    batch_failed = False
                    for res in results:
                        if isinstance(res, Exception):
                            logger.error(f"Step failed with exception: {res}")
                            batch_failed = True
                        elif res and res.get("status") == "failed":
                            batch_failed = True
                    
                    if batch_failed:
                        # By default, stop on failure?
                        # Or strategy determines?
                        # For now, break loop
                        logger.error("Batch execution failed. Stopping workflow.")
                        await exec_repo.update_status(execution_id, "failed")
                        
                        await self.event_bus.publish(OrchestrationEvent(
                            type=EventType.EXECUTION_FAILED,
                            trace_id=execution.trace_id,
                            execution_id=execution.id,
                            data={"reason": "Step validation failed"}
                        ))
                        return

                # If loop finishes successfully
                final_output = context["step_outputs"]
                # Maybe aggregate or pick last?
                
                await exec_repo.update_status(
                    execution_id, 
                    "completed", 
                    output=final_output,
                    completed_at=datetime.utcnow()
                )
                
                await self.event_bus.publish(OrchestrationEvent(
                    type=EventType.EXECUTION_COMPLETED,
                    trace_id=execution.trace_id,
                    execution_id=execution.id,
                    data={"output_keys": list(final_output.keys())}
                ))

            except Exception as e:
                logger.error(f"Execution loop error: {e}", exc_info=True)
                await exec_repo.update_status(execution_id, "failed")
                # TODO: Save error details to execution record if schema supported it (it does: error metadata?)

    async def _run_step(
        self,
        step_repo: OrchestrationStepRunRepository,
        step_run_id: UUID,
        step_config: StepConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single step wrapper.
        Updates DB status and calls adapter.
        """
        start_time = datetime.utcnow()
        await step_repo.update_status(step_run_id, "running")
        
        trace_id = context.get("trace_id", "unknown")
        
        await self.event_bus.publish(OrchestrationEvent(
            type=EventType.STEP_STARTED,
            trace_id=trace_id,
            execution_id=UUID(context["execution_id"]),
            step_id=step_config.id,
            data={"step_name": step_config.name}
        ))
        
        try:
            # Prepare input given dependencies
            # We merge global input with outputs from dependencies
            step_input = context["global_input"].copy()
            
            # Map outputs from dependencies if specified
            # For now, we just pass the whole context's step_outputs 
            # or maybe the step definition defines strict input mapping?
            # Simple approach: pass context + step_outputs
            input_data = {
                **step_input,
                "previous_outputs": context.get("step_outputs", {})
            }
            
            # Execute via adapter
            result = await self.agent_adapter.execute_step(
                step_config, 
                input_data, 
                context
            )
            
            end_time = datetime.utcnow()
            latency = int((end_time - start_time).total_seconds() * 1000)
            
            await step_repo.update_status(
                step_run_id, 
                "completed", 
                output=result.get("output"), # Assuming output is JSON serializable
                latency_ms=latency
            )
            
            await self.event_bus.publish(OrchestrationEvent(
                type=EventType.STEP_COMPLETED,
                trace_id=trace_id,
                execution_id=UUID(context["execution_id"]),
                step_id=step_config.id,
                data={"latency_ms": latency}
            ))
            
            return {"status": "completed", "output": result.get("output")}

        except Exception as e:
            logger.error(f"Step {step_config.id} execution failed: {e}")
            end_time = datetime.utcnow()
            latency = int((end_time - start_time).total_seconds() * 1000)
            
            state = "failed"
            # TODO: handle retries here or in runner loop?
            
            await step_repo.update_status(
                step_run_id, 
                state, 
                error_message=str(e),
                error_trace=traceback.format_exc(),
                latency_ms=latency
            )
            
            await self.event_bus.publish(OrchestrationEvent(
                type=EventType.STEP_FAILED,
                trace_id=trace_id,
                execution_id=UUID(context["execution_id"]),
                step_id=step_config.id,
                data={"error": str(e)}
            ))
            
            return {"status": "failed", "error": str(e)}
