"""
LiteLLM Configuration and Monitoring API.

Provides REST API endpoints for managing LiteLLM configuration,
monitoring costs, and viewing usage analytics.

Endpoints:
- GET  /api/v1/litellm/status        - Router status and health
- GET  /api/v1/litellm/models        - List available models
- GET  /api/v1/litellm/usage         - Usage statistics
- GET  /api/v1/litellm/costs         - Cost breakdown
- POST /api/v1/litellm/test          - Test model connectivity
- POST /api/v1/litellm/reset         - Reset router
- GET  /api/v1/litellm/providers     - List configured providers
"""

import asyncio
import os
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.auth import verify_admin_credentials
from resync.core.structured_logger import get_logger
from resync.settings import get_settings

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/litellm",
    tags=["LiteLLM Configuration"],
    dependencies=[Depends(verify_admin_credentials)],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class LiteLLMStatus(BaseModel):
    """LiteLLM router status."""

    initialized: bool
    router_available: bool
    init_success_count: int
    init_failures: dict[str, int]
    cost_calc_failures: int
    uptime_seconds: float | None = None


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    provider: str
    model_type: str  # chat, completion, embedding
    max_tokens: int | None = None
    input_cost_per_1k: float | None = None
    output_cost_per_1k: float | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    is_local: bool = False


class UsageStats(BaseModel):
    """LLM usage statistics."""

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_response_time_ms: float
    error_rate: float
    model_usage: dict[str, int]
    requests_today: int
    cost_today_usd: float


class CostBreakdown(BaseModel):
    """Cost breakdown by model and time."""

    daily_costs: dict[str, float]
    model_costs: dict[str, float]
    total_cost_usd: float
    budget_limit_usd: float
    budget_used_percent: float
    projected_monthly_cost: float


class ProviderStatus(BaseModel):
    """Status of an LLM provider."""

    name: str
    enabled: bool
    configured: bool
    models_available: list[str]
    last_success: str | None = None
    last_error: str | None = None
    avg_latency_ms: float | None = None


class ModelTestRequest(BaseModel):
    """Request to test a model."""

    model: str = Field(..., description="Model name to test")
    prompt: str = Field(
        default="Hello, respond with 'OK' if you're working.", description="Test prompt"
    )
    timeout: int = Field(default=30, ge=5, le=120, description="Timeout in seconds")


class ModelTestResponse(BaseModel):
    """Response from model test."""

    success: bool
    model: str
    response: str | None = None
    latency_ms: float
    error: str | None = None
    tokens_used: int | None = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_litellm_status() -> LiteLLMStatus:
    """Get current LiteLLM router status."""
    try:
        from resync.core.litellm_init import get_litellm_metrics, get_litellm_router

        metrics = get_litellm_metrics()
        router = get_litellm_router()

        return LiteLLMStatus(
            initialized=metrics.get("router_initialized", False),
            router_available=router is not None,
            init_success_count=metrics.get("init_success", 0),
            init_failures=metrics.get("init_fail_reason", {}),
            cost_calc_failures=metrics.get("cost_calc_fail", 0),
        )
    except ImportError:
        return LiteLLMStatus(
            initialized=False,
            router_available=False,
            init_success_count=0,
            init_failures={"ImportError": 1},
            cost_calc_failures=0,
        )


def get_available_models() -> list[ModelInfo]:
    """Get list of available models."""
    models = []

    # Define known models with their properties
    model_definitions = [
        # OpenAI
        {
            "name": "gpt-4o",
            "provider": "openai",
            "type": "chat",
            "max_tokens": 128000,
            "input_cost": 0.005,
            "output_cost": 0.015,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "gpt-4o-mini",
            "provider": "openai",
            "type": "chat",
            "max_tokens": 128000,
            "input_cost": 0.00015,
            "output_cost": 0.0006,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "gpt-4-turbo",
            "provider": "openai",
            "type": "chat",
            "max_tokens": 128000,
            "input_cost": 0.01,
            "output_cost": 0.03,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "gpt-4",
            "provider": "openai",
            "type": "chat",
            "max_tokens": 8192,
            "input_cost": 0.03,
            "output_cost": 0.06,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "gpt-3.5-turbo",
            "provider": "openai",
            "type": "chat",
            "max_tokens": 16385,
            "input_cost": 0.0005,
            "output_cost": 0.0015,
            "streaming": True,
            "functions": True,
        },
        # Anthropic
        {
            "name": "claude-3-opus-20240229",
            "provider": "anthropic",
            "type": "chat",
            "max_tokens": 200000,
            "input_cost": 0.015,
            "output_cost": 0.075,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "claude-3-sonnet-20240229",
            "provider": "anthropic",
            "type": "chat",
            "max_tokens": 200000,
            "input_cost": 0.003,
            "output_cost": 0.015,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "claude-3-5-sonnet-20240620",
            "provider": "anthropic",
            "type": "chat",
            "max_tokens": 200000,
            "input_cost": 0.003,
            "output_cost": 0.015,
            "streaming": True,
            "functions": True,
        },
        {
            "name": "claude-3-haiku-20240307",
            "provider": "anthropic",
            "type": "chat",
            "max_tokens": 200000,
            "input_cost": 0.00025,
            "output_cost": 0.00125,
            "streaming": True,
            "functions": True,
        },
        # Ollama (Local)
        {
            "name": "ollama/llama3",
            "provider": "ollama",
            "type": "chat",
            "max_tokens": 8192,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "streaming": True,
            "functions": False,
            "local": True,
        },
        {
            "name": "ollama/mistral",
            "provider": "ollama",
            "type": "chat",
            "max_tokens": 32768,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "streaming": True,
            "functions": False,
            "local": True,
        },
        {
            "name": "ollama/codellama",
            "provider": "ollama",
            "type": "chat",
            "max_tokens": 16384,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "streaming": True,
            "functions": False,
            "local": True,
        },
        # Together AI
        {
            "name": "together_ai/llama-3-70b",
            "provider": "together_ai",
            "type": "chat",
            "max_tokens": 8192,
            "input_cost": 0.0009,
            "output_cost": 0.0009,
            "streaming": True,
            "functions": False,
        },
        {
            "name": "together_ai/mixtral-8x7b",
            "provider": "together_ai",
            "type": "chat",
            "max_tokens": 32768,
            "input_cost": 0.0006,
            "output_cost": 0.0006,
            "streaming": True,
            "functions": False,
        },
    ]

    for m in model_definitions:
        models.append(
            ModelInfo(
                name=m["name"],
                provider=m["provider"],
                model_type=m["type"],
                max_tokens=m.get("max_tokens"),
                input_cost_per_1k=m.get("input_cost"),
                output_cost_per_1k=m.get("output_cost"),
                supports_streaming=m.get("streaming", True),
                supports_function_calling=m.get("functions", False),
                is_local=m.get("local", False),
            )
        )

    return models


def get_usage_stats() -> UsageStats:
    """Get usage statistics from LLM monitor."""
    try:
        from resync.core.llm_monitor import llm_cost_monitor

        report = llm_cost_monitor.get_usage_report()
        today = datetime.now().strftime("%Y-%m-%d")

        return UsageStats(
            total_requests=report.get("total_requests", 0),
            total_input_tokens=report.get("total_input_tokens", 0),
            total_output_tokens=report.get("total_output_tokens", 0),
            total_cost_usd=report.get("total_cost_usd", 0.0),
            avg_response_time_ms=report.get("avg_response_time", 0.0) * 1000,
            error_rate=report.get("error_rate", 0.0),
            model_usage=report.get("model_usage", {}),
            requests_today=0,  # Would need to filter by date
            cost_today_usd=report.get("daily_costs", {}).get(today, 0.0),
        )
    except ImportError:
        return UsageStats(
            total_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            avg_response_time_ms=0.0,
            error_rate=0.0,
            model_usage={},
            requests_today=0,
            cost_today_usd=0.0,
        )


def get_cost_breakdown() -> CostBreakdown:
    """Get cost breakdown."""
    try:
        from resync.core.llm_monitor import llm_cost_monitor

        report = llm_cost_monitor.get_usage_report()
        settings = get_settings()

        budget_limit = getattr(settings, "LLM_BUDGET_DAILY_USD", 500.0)
        total_cost = report.get("total_cost_usd", 0.0)
        daily_costs = report.get("daily_costs", {})

        # Calculate model costs from history
        model_costs = {}
        for cost in llm_cost_monitor.cost_history:
            model_costs[cost.model] = model_costs.get(cost.model, 0.0) + cost.cost_usd

        # Project monthly cost based on recent daily average
        recent_daily = list(daily_costs.values())[-7:] if daily_costs else [0]
        avg_daily = sum(recent_daily) / len(recent_daily) if recent_daily else 0
        projected_monthly = avg_daily * 30

        return CostBreakdown(
            daily_costs=daily_costs,
            model_costs=model_costs,
            total_cost_usd=total_cost,
            budget_limit_usd=budget_limit,
            budget_used_percent=(total_cost / budget_limit * 100) if budget_limit > 0 else 0,
            projected_monthly_cost=projected_monthly,
        )
    except ImportError:
        return CostBreakdown(
            daily_costs={},
            model_costs={},
            total_cost_usd=0.0,
            budget_limit_usd=500.0,
            budget_used_percent=0.0,
            projected_monthly_cost=0.0,
        )


def get_provider_status() -> list[ProviderStatus]:
    """Get status of configured providers."""
    settings = get_settings()
    providers = []

    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    providers.append(
        ProviderStatus(
            name="OpenAI",
            enabled=bool(openai_key),
            configured=bool(openai_key),
            models_available=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        )
    )

    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    providers.append(
        ProviderStatus(
            name="Anthropic",
            enabled=bool(anthropic_key),
            configured=bool(anthropic_key),
            models_available=[
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-3-haiku",
                "claude-3-5-sonnet",
            ],
        )
    )

    # Ollama (Local)
    providers.append(
        ProviderStatus(
            name="Ollama (Local)",
            enabled=True,  # Always available if installed
            configured=True,
            models_available=["llama3", "mistral", "codellama", "phi"],
        )
    )

    # Together AI
    together_key = os.environ.get("TOGETHER_API_KEY", "")
    providers.append(
        ProviderStatus(
            name="Together AI",
            enabled=bool(together_key),
            configured=bool(together_key),
            models_available=["llama-3-70b", "mixtral-8x7b"],
        )
    )

    # OpenRouter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    providers.append(
        ProviderStatus(
            name="OpenRouter",
            enabled=bool(openrouter_key),
            configured=bool(openrouter_key),
            models_available=["Various (100+ models)"],
        )
    )

    # NVIDIA NIM
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "") or getattr(settings, "llm_api_key", None)
    nvidia_configured = bool(nvidia_key) or "nvidia" in settings.llm_endpoint.lower()
    providers.append(
        ProviderStatus(
            name="NVIDIA NIM",
            enabled=nvidia_configured,
            configured=nvidia_configured,
            models_available=["llama3-70b-instruct", "mixtral-8x7b-instruct"],
        )
    )

    return providers


# =============================================================================
# API ENDPOINTS
# =============================================================================


@router.get("/status", response_model=LiteLLMStatus)
async def get_status():
    """
    Get LiteLLM router status and health.

    Returns initialization status, success/failure counts, and availability.
    """
    return get_litellm_status()


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    provider: str | None = None,
    local_only: bool = False,
):
    """
    List available models.

    Args:
        provider: Filter by provider (openai, anthropic, ollama, etc.)
        local_only: Show only local models (free)
    """
    models = get_available_models()

    if provider:
        models = [m for m in models if m.provider.lower() == provider.lower()]

    if local_only:
        models = [m for m in models if m.is_local]

    return models


@router.get("/usage", response_model=UsageStats)
async def get_usage():
    """
    Get LLM usage statistics.

    Returns request counts, token usage, costs, and error rates.
    """
    return get_usage_stats()


@router.get("/costs", response_model=CostBreakdown)
async def get_costs():
    """
    Get cost breakdown by model and time period.

    Returns daily costs, per-model costs, budget status, and projections.
    """
    return get_cost_breakdown()


@router.get("/providers", response_model=list[ProviderStatus])
async def list_providers():
    """
    List configured LLM providers and their status.
    """
    return get_provider_status()


@router.post("/test", response_model=ModelTestResponse)
async def test_model(request: ModelTestRequest):
    """
    Test connectivity to a specific model.

    Sends a simple prompt to verify the model is accessible and responding.
    """
    start_time = time.time()

    try:
        from resync.core.utils.llm import call_llm

        response = await asyncio.wait_for(
            call_llm(
                request.prompt,
                model=request.model,
                max_tokens=50,
            ),
            timeout=request.timeout,
        )

        latency_ms = (time.time() - start_time) * 1000

        return ModelTestResponse(
            success=True,
            model=request.model,
            response=response[:200] if response else None,
            latency_ms=latency_ms,
        )

    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        return ModelTestResponse(
            success=False,
            model=request.model,
            latency_ms=latency_ms,
            error=f"Timeout after {request.timeout}s",
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("model_test_failed", model=request.model, error=str(e))
        return ModelTestResponse(
            success=False,
            model=request.model,
            latency_ms=latency_ms,
            error=str(e),
        )


@router.post("/reset")
async def reset_router():
    """
    Reset the LiteLLM router.

    Forces re-initialization on next request. Useful after config changes.
    """
    try:
        from resync.core.litellm_init import reset_litellm_router

        reset_litellm_router()
        logger.info("litellm_router_reset")

        return {
            "success": True,
            "message": "LiteLLM router reset. Will reinitialize on next request.",
        }

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiteLLM not installed",
        ) from None
    except Exception as e:
        logger.error("litellm_reset_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset router. Check server logs for details.",
        ) from e


@router.get("/dashboard-data")
async def get_dashboard_data():
    """
    Get all data needed for the LiteLLM dashboard in a single call.

    Returns status, usage, costs, and provider information.
    """
    return {
        "status": get_litellm_status().model_dump(),
        "usage": get_usage_stats().model_dump(),
        "costs": get_cost_breakdown().model_dump(),
        "providers": [p.model_dump() for p in get_provider_status()],
        "models_count": len(get_available_models()),
        "timestamp": datetime.now().isoformat(),
    }
