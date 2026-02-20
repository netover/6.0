"""
Orchestration Strategies

Defines how steps are executed (Sequential, Parallel, etc.).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Set

from resync.core.orchestration.schemas import StepConfig, WorkflowConfig


logger = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""

    @abstractmethod
    async def plan_execution(
        self, config: WorkflowConfig, completed_steps: Set[str]
    ) -> List[List[StepConfig]]:
        """
        Determine which steps can run next.
        Returns a list of batches, where each batch is a list of steps that can run in parallel.
        """
        pass


class SequentialStrategy(ExecutionStrategy):
    """
    Executes steps one by one, respecting dependencies.
    """

    async def plan_execution(
        self, config: WorkflowConfig, completed_steps: Set[str]
    ) -> List[List[StepConfig]]:
        """
        Finds the first step that hasn't run and whose dependencies are met.
        Returns it as a single-item batch.
        """
        next_steps = []

        # Sort steps to ensure deterministic order if no explicit dependencies
        # This is simple sequential: list order matter unless dependencies exist

        # For true DAG topological sort, we'd need graph logic.
        # Here we implement a "greedy" check: first uncompleted step whose deps are met.

        for step in config.steps:
            if step.id in completed_steps:
                continue

            # Check dependencies
            deps_met = True
            for dep in step.dependencies:
                # TODO: Check 'condition' logic here if needed
                if dep.step_id not in completed_steps:
                    deps_met = False
                    break

            if deps_met:
                next_steps.append(step)
                # Sequential strategy only picks one at a time (the first one ready)
                # Unless we strictly follow list order as fallback?
                # Let's assume list order is the default sequence.
                break

        if next_steps:
            return [next_steps]
        return []


class ParallelStrategy(ExecutionStrategy):
    """
    Executes all ready steps in parallel.
    """

    async def plan_execution(
        self, config: WorkflowConfig, completed_steps: Set[str]
    ) -> List[List[StepConfig]]:
        """
        Finds ALL steps that haven't run and whose dependencies are met.
        Returns them as a single batch.
        """
        ready_steps = []

        for step in config.steps:
            if step.id in completed_steps:
                continue

            deps_met = True
            for dep in step.dependencies:
                if dep.step_id not in completed_steps:
                    deps_met = False
                    break

            if deps_met:
                ready_steps.append(step)

        if ready_steps:
            return [ready_steps]
        return []


class StrategyFactory:
    """Factory to get strategy by name."""

    @staticmethod
    def get_strategy(strategy_name: str) -> ExecutionStrategy:
        if strategy_name == "parallel":
            return ParallelStrategy()
        # Default to sequential
        return SequentialStrategy()
