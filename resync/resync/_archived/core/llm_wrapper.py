"""
LLM wrapper module that integrates the TWS-optimized LLM functionality
with the agent system using the LLM optimizer.
"""

from typing import Any

from resync.core.llm_optimizer import TWS_LLMOptimizer
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# Global instance of the LLM optimizer
llm_optimizer = TWS_LLMOptimizer()


class OptimizedLLMWrapper:
    """
    Wrapper class that provides optimized LLM capabilities
    for use with the agent system.
    """

    def __init__(self):
        self.optimizer = llm_optimizer

    async def get_response(
        self,
        query: str,
        context: dict[str, Any] = None,
        use_cache: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Get optimized LLM response using the TWS optimizer.

        Args:
            query: User query or prompt
            context: Additional context for the query
            use_cache: Whether to use response caching
            stream: Whether to stream the response

        Returns:
            Optimized LLM response
        """
        try:
            return await self.optimizer.get_optimized_response(
                query=query, context=context or {}, use_cache=use_cache, stream=stream
            )
        except Exception as e:
            logger.error("llm_optimization_failed", error=str(e), exc_info=True)
            # Fallback to direct LLM call if optimization fails
            from resync.core.utils.llm import call_llm

            return await call_llm(query, model="gpt-3.5-turbo")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics from the optimizer."""
        return self.optimizer.get_cache_stats()

    async def clear_caches(self) -> None:
        """Clear optimizer caches."""
        await self.optimizer.clear_caches()


# Global instance of the wrapper
optimized_llm = OptimizedLLMWrapper()
