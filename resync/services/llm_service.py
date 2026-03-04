
from __future__ import annotations
import logging
from typing import Any, AsyncGenerator

import litellm

from resync.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ALIAS = getattr(settings, "llm_model", "liteLLM-default")


class LLMService:
    """LiteLLM façade used by the entire application."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL_ALIAS

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call LiteLLM router."""
        if stream:
            return await litellm.acompletion(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response

    async def health_check(self) -> bool:
        try:
            await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("llm_healthcheck_failed: %s", exc)
            return False


_llm_service: LLMService | None = None


async def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
