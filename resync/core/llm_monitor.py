"""
LLM cost monitoring and streaming implementation for TWS optimization.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from resync.core.cache import AsyncTTLCache
from resync.core.utils.llm import call_llm

logger = logging.getLogger(__name__)


@dataclass
class LLMCost:
    """LLM cost tracking."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMUsageStats:
    """LLM usage statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    model_usage: dict[str, int] = field(default_factory=dict)
    daily_costs: dict[str, float] = field(default_factory=dict)


class LLMCostMonitor:
    """
    Monitor LLM usage and costs for TWS operations.

    Features:
    - Real-time cost tracking
    - Usage analytics
    - Budget alerts
    - Model performance metrics
    """

    def __init__(self):
        """Initialize LLM cost monitor."""
        self.usage_stats = LLMUsageStats()
        self.cost_history: list[LLMCost] = []
        self.budget_limit = 500.0  # USD per month for 4M jobs/month
        self.cache = None  # Lazy initialization

    def _ensure_cache(self) -> None:
        """Ensure cache is initialized."""
        if self.cache is None:
            self.cache = AsyncTTLCache(ttl_seconds=3600)

    async def track_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        response_time: float,
        success: bool = True,
    ) -> None:
        """
        Track an LLM request with cost calculation.

        Args:
            model: Model used
            input_tokens: Input tokens consumed
            output_tokens: Output tokens generated
            response_time: Response time in seconds
            success: Whether request was successful
        """
        # Cost calculation (approximate based on common rates)
        cost_per_1k = self._get_cost_per_1k_tokens(model)

        input_cost = (input_tokens / 1000) * cost_per_1k["input"]
        output_cost = (output_tokens / 1000) * cost_per_1k["output"]
        total_cost = input_cost + output_cost

        # Record cost
        cost_record = LLMCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
        )

        self.cost_history.append(cost_record)

        # Update statistics
        self.usage_stats.total_requests += 1
        self.usage_stats.total_input_tokens += input_tokens
        self.usage_stats.total_output_tokens += output_tokens
        self.usage_stats.total_cost_usd += total_cost
        self.usage_stats.avg_response_time = (
            self.usage_stats.avg_response_time * (self.usage_stats.total_requests - 1)
            + response_time
        ) / self.usage_stats.total_requests

        if not success:
            self.usage_stats.error_rate = (
                self.usage_stats.error_rate * (self.usage_stats.total_requests - 1) + 1
            ) / self.usage_stats.total_requests

        # Update model usage
        self.usage_stats.model_usage[model] = (
            self.usage_stats.model_usage.get(model, 0) + 1
        )

        # Check budget
        self._check_budget_alerts()

        logger.debug(f"LLM request tracked: {model}, cost: ${total_cost:.4f}")

    def _get_cost_per_1k_tokens(self, model: str) -> dict[str, float]:
        """Get cost per 1K tokens for different models."""
        # With LiteLLM integration, we can also use LiteLLM's cost calculation
        # but keeping this for fallback and custom models
        from litellm import get_model_info

        try:
            # Try to get cost info from LiteLLM if available
            model_info = get_model_info(model=model)
            if model_info and "input_cost_per_token" in model_info:
                input_cost = model_info["input_cost_per_token"]
                output_cost = model_info["output_cost_per_token"]
                # Convert to cost per 1K tokens
                return {"input": input_cost * 1000, "output": output_cost * 1000}
        except Exception as e:
            # Log pricing calculation error and fallback to hardcoded values
            logger.debug(
                "LLM pricing calculation failed, using hardcoded values: %s", e
            )

        # Approximate costs (update with real pricing)
        costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "llama3": {"input": 0.000, "output": 0.000},  # Free for local
            "ollama/*": {
                "input": 0.000,
                "output": 0.000,
            },  # Free for local Ollama models
        }

        # Handle Ollama models specifically
        if model.startswith("ollama/"):
            return {"input": 0.000, "output": 0.000}

        return costs.get(model, {"input": 0.001, "output": 0.004})

    def _check_budget_alerts(self) -> None:
        """Check if budget limits are being approached."""
        today = time.strftime("%Y-%m-%d")
        daily_cost = sum(
            cost.cost_usd
            for cost in self.cost_history
            if time.strftime("%Y-%m-%d", time.localtime(cost.timestamp)) == today
        )

        self.usage_stats.daily_costs[today] = daily_cost

        # Alert if approaching budget
        if daily_cost > self.budget_limit * 0.8:
            logger.warning(
                f"Daily LLM cost approaching budget: ${daily_cost:.2f}/${self.budget_limit:.2f}"
            )

        # Alert if over budget
        if daily_cost > self.budget_limit:
            logger.error(
                f"Daily LLM budget exceeded: ${daily_cost:.2f}/${self.budget_limit:.2f}"
            )

    def get_usage_report(self) -> dict[str, Any]:
        """Get comprehensive usage report."""
        return {
            "total_requests": self.usage_stats.total_requests,
            "total_input_tokens": self.usage_stats.total_input_tokens,
            "total_output_tokens": self.usage_stats.total_output_tokens,
            "total_cost_usd": self.usage_stats.total_cost_usd,
            "avg_response_time": self.usage_stats.avg_response_time,
            "error_rate": self.usage_stats.error_rate,
            "model_usage": self.usage_stats.model_usage,
            "daily_costs": self.usage_stats.daily_costs,
            "budget_limit": self.budget_limit,
            "budget_usage_percent": (
                self.usage_stats.total_cost_usd / self.budget_limit
            )
            * 100,
        }


class StreamingLLMResponse:
    """
    Streaming LLM response for long outputs (troubleshooting).
    """

    def __init__(self, prompt: str, model: str, max_tokens: int = 1000):
        """Initialize streaming response."""
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.response_chunks: list[str] = []
        self.is_complete = False

    async def generate_chunks(self) -> AsyncGenerator[str, None]:
        """
        Generate response in chunks for streaming.

        Note: This is a placeholder implementation.
        Real streaming would require LLM provider support.
        """
        try:
            # For now, get full response and split into chunks
            full_response = await call_llm(
                self.prompt, model=self.model, max_tokens=self.max_tokens
            )

            if full_response:
                # Split into chunks of ~100 characters
                chunk_size = 100
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i : i + chunk_size]
                    self.response_chunks.append(chunk)
                    yield chunk
                    await asyncio.sleep(0.1)  # Simulate streaming delay

            self.is_complete = True

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Error in streaming response: %s", e)
            yield f"Error: {str(e)}"
            self.is_complete = True

    def get_full_response(self) -> str:
        """Get complete response."""
        return "".join(self.response_chunks)


# Global instances
_llm_cost_monitor_instance: LLMCostMonitor | None = None


class _LazyLLMCostMonitor:
    """Lazy proxy to avoid import-time side effects (gunicorn --preload safe)."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        self._instance = None

    def get_instance(self) -> LLMCostMonitor:
        if self._instance is None:
            self._instance = LLMCostMonitor()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get_instance(), name)


llm_cost_monitor = _LazyLLMCostMonitor()


def get_llm_cost_monitor() -> LLMCostMonitor:
    """Return the singleton instance (preferred over using the proxy directly)."""
    return llm_cost_monitor.get_instance()
