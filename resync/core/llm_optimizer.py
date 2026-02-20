"""
TWS-optimized LLM integration with prompt caching and model routing.
"""

import hashlib
import logging
import time
from typing import Any

import pybreaker

from resync.core.cache import AsyncTTLCache
from resync.core.litellm_init import get_litellm_router
from resync.core.llm_monitor import llm_cost_monitor
from resync.core.utils.llm import call_llm
from resync.settings import settings

# Define the circuit breaker for LLM API calls
llm_api_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=(ValueError,),  # Don't count validation errors
)

logger = logging.getLogger(__name__)


class TWS_LLMOptimizer:
    """
    TWS-optimized LLM integration with caching and model routing.

    Features:
    - Prompt caching for TWS templates
    - Model selection based on query complexity
    - Response streaming for long outputs
    - TWS-specific template matching
    """

    def __init__(self):
        """Initialize the TWS LLM optimizer."""
        self.prompt_cache = AsyncTTLCache(ttl_seconds=3600)
        self.response_cache = AsyncTTLCache(ttl_seconds=300)

        # TWS-specific templates
        self.tws_templates = {
            "job_status": "Get status for TWS job {job_id}",
            "job_failure": "Analyze failure for job {job_id}: {error_msg}",
            "system_health": "Summarize TWS system health",
            "job_dependencies": "Show dependencies for job {job_id}",
            "troubleshooting": "Troubleshoot TWS issue: {description}",
        }

        # Model routing based on complexity
        self.model_routing = {
            "simple": getattr(
                settings, "AUDITOR_MODEL_NAME", "gpt-3.5-turbo"
            ),  # For basic queries
            "complex": getattr(
                settings, "AGENT_MODEL_NAME", "gpt-4o"
            ),  # For complex analysis
            "troubleshooting": getattr(
                settings, "AGENT_MODEL_NAME", "gpt-4o"
            ),  # For troubleshooting
        }

    def _match_template(self, query: str) -> str:
        """
        Match query to TWS template.

        Args:
            query: User query string

        Returns:
            Template key or 'custom'
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["status", "estado"]):
            return "job_status"
        if any(word in query_lower for word in ["failed", "error", "falhou", "erro"]):
            return "job_failure"
        if any(word in query_lower for word in ["health", "saúde", "sistema"]):
            return "system_health"
        if any(word in query_lower for word in ["dependenc", "depende"]):
            return "job_dependencies"
        if any(word in query_lower for word in ["troubleshoot", "problem", "issue"]):
            return "troubleshooting"

        return "custom"

    def _select_model(self, query: str, context: dict) -> str:
        """
        Select appropriate model based on query complexity and availability.

        Args:
            query: User query
            context: Additional context

        Returns:
            Model name to use
        """
        complexity_indicators = [
            "analyze",
            "explain",
            "why",
            "how",
            "complex",
            "detailed",
            "comprehensive",
            "troubleshoot",
        ]

        query_lower = query.lower()
        is_complex = any(
            indicator in query_lower for indicator in complexity_indicators
        )

        # Check if we have a LiteLLM router available for advanced model selection
        router = get_litellm_router()
        if router:
            # Use LiteLLM's model group feature if available
            if is_complex:
                # Try to use a TWS troubleshooting model group if defined
                try:
                    from litellm import get_available_models

                    available_models = get_available_models()
                    if any("gpt-4" in model for model in available_models):
                        return self.model_routing["troubleshooting"]
                    if any("gpt-4" in model for model in self.model_routing.values()):
                        return self.model_routing["complex"]
                except Exception as e:
                    # Log model selection error and fall back to normal selection
                    logger.debug("GPT-4 model selection failed, using fallback: %s", e)

            # Check for local model preference
            if not is_complex and "local" in settings.ENVIRONMENT.lower():
                # Prioritize local Ollama models for non-complex queries in local environments
                try:
                    # Check if Ollama is available
                    import httpx

                    with httpx.Client(timeout=5.0) as client:
                        response = client.get(
                            f"{settings.LLM_ENDPOINT}/api/tags"
                        )  # Assuming Ollama endpoint
                        if response.status_code == 200:
                            return (
                                "ollama/mistral"  # Use local model for simple queries
                            )
                except Exception as e:
                    # Log Ollama availability check error and continue with normal selection
                    logger.debug(
                        "Ollama availability check failed, using fallback: %s", e
                    )

        if is_complex:
            return self.model_routing["complex"]
        return self.model_routing["simple"]

    async def get_optimized_response(
        self,
        query: str,
        context: dict | None = None,
        use_cache: bool = True,
        stream: bool = False,
    ) -> Any:
        """
        Get optimized LLM response with LiteLLM integration, caching and model routing.

        Args:
            query: User query
            context: Additional context
            use_cache: Whether to use response caching
            stream: Whether to stream the response

        Returns:
            LLM response
        """
        if context is None:
            context = {}

        # Generate cache key
        context_str = str(sorted(context.items()))
        cache_key = hashlib.sha256(f"{query}:{context_str}".encode()).hexdigest()

        # Check response cache first
        if use_cache:
            cached_response = await self.response_cache.get(cache_key)
            if cached_response:
                logger.debug("Using cached LLM response")
                return cached_response

        # Template matching for common TWS queries
        template_key = self._match_template(query)

        # Check prompt cache
        prompt_hash = hash(f"{template_key}:{context_str}")
        cached_prompt = await self.prompt_cache.get(str(prompt_hash))

        if cached_prompt:
            prompt = cached_prompt
            logger.debug("Using cached prompt")
        else:
            # Generate prompt based on template
            if template_key in self.tws_templates:
                prompt = self.tws_templates[template_key].format(**context)
                await self.prompt_cache.set(str(prompt_hash), prompt)
            else:
                prompt = query

        # Select appropriate model
        model = self._select_model(query, context)

        # Get response with circuit breaker protection
        start_time = time.time()

        try:
            if stream and "troubleshoot" in template_key:
                # Use streaming for troubleshooting
                response = await self.stream_llm_response(prompt, model)
            else:
                response = await llm_api_breaker.async_call(
                    call_llm,
                    prompt,
                    model=model,
                    max_tokens=500 if template_key != "troubleshooting" else 1000,
                    # Pass settings-based configuration to take advantage of LiteLLM features
                    api_base=getattr(settings, "LLM_ENDPOINT", None),
                    api_key=(
                        settings.LLM_API_KEY
                        if settings.LLM_API_KEY != "your_default_api_key_here"
                        else None
                    ),
                )

            response_time = time.time() - start_time

            # Track costs and performance
            # Try to get actual token counts from LiteLLM if available
            input_tokens = len(prompt.split()) * 1.3  # Fallback estimate
            output_tokens = len(str(response).split()) * 1.3  # Fallback estimate

            await llm_cost_monitor.track_request(
                model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                response_time=response_time,
                success=True,
            )

        except Exception as e:
            response_time = time.time() - start_time
            # Track failed request
            await llm_cost_monitor.track_request(
                model=model,
                input_tokens=len(prompt.split()) * 1.3,
                output_tokens=0,
                response_time=response_time,
                success=False,
            )
            logger.error("LLM request failed: %s", e)
            raise

        # Cache response
        if use_cache and response:
            await self.response_cache.set(cache_key, response)

        return response

    async def stream_llm_response(self, prompt: str, model: str = "gpt-4") -> str:
        """
        Streams response from LLM with caching.

        Args:
            prompt: The input prompt
            model: The LLM model to use

        Returns:
            Streamed response
        """
        # Check cache first
        cache_key = f"stream_{hash(prompt)}_{model}"
        cached = await self.response_cache.get(cache_key)

        if cached:
            logger.info("LLM stream cache hit for key: %s", cache_key)
            return cached

        # Use litellm for streaming capability
        try:
            from litellm import acompletion

            # Prepare the message in the required format
            messages = [{"content": prompt, "role": "user"}]

            # Create async generator for streaming
            response = await acompletion(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=1000,
                temperature=0.7,
                api_base=getattr(settings, "LLM_ENDPOINT", None),
                api_key=(
                    settings.LLM_API_KEY
                    if settings.LLM_API_KEY != "your_default_api_key_here"
                    else None
                ),
            )

            full_response = ""
            async for chunk in response:
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    content = chunk.choices[0].get("delta", {}).get("content", "")
                    if content:
                        full_response += content

            # Cache the result
            await self.response_cache.set(cache_key, full_response)

            return full_response

        except Exception:
            logger.error("Error in LLM streaming", exc_info=True)
            # Fallback to original method (which now also uses LiteLLM)
            result = await call_llm(prompt, model=model, max_tokens=1000)
            await self.response_cache.set(cache_key, result)
            return result

    async def clear_caches(self) -> None:
        """Clear both prompt and response caches."""
        await self.prompt_cache.clear()
        await self.response_cache.clear()
        logger.info("LLM caches cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "prompt_cache": self.prompt_cache.get_metrics(),
            "response_cache": self.response_cache.get_metrics(),
        }


# Global instance (lazy loaded)
_tws_llm_optimizer: TWS_LLMOptimizer | None = None


def get_llm_optimizer() -> TWS_LLMOptimizer:
    """Get the singleton instance of TWS_LLMOptimizer."""
    global _tws_llm_optimizer
    if _tws_llm_optimizer is None:
        _tws_llm_optimizer = TWS_LLMOptimizer()
    return _tws_llm_optimizer


def __getattr__(name: str) -> Any:
    """Module-level getattr to support lazy loading of tws_llm_optimizer."""
    if name == "tws_llm_optimizer":
        return get_llm_optimizer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
