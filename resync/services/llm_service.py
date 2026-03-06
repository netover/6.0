
from __future__ import annotations
import logging
import asyncio
from typing import Any, AsyncGenerator

from resync.core.litellm_init import get_litellm_router, maybe_reload_litellm_router
from resync.settings import settings
from resync.core.litellm_hooks import record_llm_fallback

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ALIAS = getattr(settings, "llm_model", "liteLLM-default")


def _require_router():
    # Opportunistic hot-reload (cheap, rate-limited internally)
    router = maybe_reload_litellm_router() or get_litellm_router()
    if router is None:
        raise RuntimeError("LiteLLM router not initialized (check llm_endpoint/llm_api_key and litellm_config.yaml)")
    return router


def _is_retryable_llm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    # Common provider/router failure patterns
    if any(s in msg for s in [
        "429", "rate limit", "too many requests",
        "timeout", "timed out", "deadline exceeded",
        "temporarily unavailable", "service unavailable", "502", "503", "504",
        "connection reset", "connection error", "dns", "network",
    ]):
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return False



def _fallback_reason(exc: Exception) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg or "deadline exceeded" in msg:
        return "timeout"
    if any(s in msg for s in ["502", "503", "504", "service unavailable", "bad gateway", "gateway timeout"]):
        return "provider_5xx"
    if any(s in msg for s in ["connection", "dns", "network", "reset"]):
        return "network"
    return "other"

def _fallback_candidates(primary: str) -> list[str]:
    """Ordered model aliases to try on retryable errors."""
    cands: list[str] = [primary]
    for cand in [
        "tws-fallback",
        "openrouter-free",
        getattr(settings, "llm_fallback_model", None),
        "gpt-fallback",
        "gpt-cheap",
    ]:
        if not cand:
            continue
        cand_s = str(cand)
        if cand_s not in cands:
            cands.append(cand_s)
    return cands


async def _acompletion_with_fallback(
    *,
    router: Any,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    max_attempts: int = 3,
    base_backoff_s: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Attempt completion with model fallbacks on 429/timeouts/transient errors."""
    last_exc: Exception | None = None
    attempts = 0

    candidates = _fallback_candidates(model)

    for idx, cand_model in enumerate(candidates):
        last_exc_for_model: Exception | None = None
        last_reason: str | None = None
        per_model_tries = 2 if max_attempts >= 2 else 1
        for _ in range(per_model_tries):
            attempts += 1
            try:
                return await router.acompletion(
                    model=cand_model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                last_exc_for_model = exc
                last_reason = _fallback_reason(exc)
                if (not _is_retryable_llm_error(exc)) or attempts >= max_attempts:
                    raise
                backoff = base_backoff_s * (2 ** min(attempts - 1, 4))
                logger.warning(
                    "llm_retryable_error_fallback",
                    model=cand_model,
                    attempt=attempts,
                    wait_s=backoff,
                    error=str(exc)[:200],
                )
                await asyncio.sleep(backoff)

        # If we exhausted tries for this candidate and will proceed to the next candidate,
        # emit a fallback telemetry event (best-effort).
        if last_exc_for_model is not None:
            next_model = candidates[idx + 1] if (idx + 1) < len(candidates) else None
            if next_model and _is_retryable_llm_error(last_exc_for_model) and attempts < max_attempts:
                try:
                    record_llm_fallback(
                        from_model=str(cand_model),
                        to_model=str(next_model),
                        reason=str(last_reason or "unknown"),
                        error_type=type(last_exc_for_model).__name__,
                        error=str(last_exc_for_model),
                        attempt=attempts,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:

                    import logging
                    logging.getLogger(__name__).debug("Ignored exception in /mnt/data/proj_v5/resync/services/llm_service.py", exc_info=True)


    if last_exc:
        raise last_exc
    raise RuntimeError("LLM call failed without exception")



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
        """Call LiteLLM router (non-streaming)."""
        if stream:
            # For streaming, use `stream_response()`.
            raise ValueError("Use stream_response() for stream=True")

        router = _require_router()
        response = await _acompletion_with_fallback(
            router=router,
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return response

    async def stream_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream deltas from LiteLLM when supported.

        Notes:
        - Some providers return different chunk schemas; we normalize to text deltas.
        - If the provider/router cannot stream, we fall back to a single chunk.
        """
        router = _require_router()
        result = _acompletion_with_fallback(router=router, model=self.model, messages=messages, stream=True, **kwargs)

        # `acompletion(stream=True)` is implementation-dependent:
        # - may return an awaitable that yields an async iterator
        # - may directly return an async iterator
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[assignment]

        # If it isn't an async iterator, just fall back to non-streaming.
        if not hasattr(result, "__aiter__"):
            full = await _acompletion_with_fallback(router=router, model=self.model, messages=messages, stream=False, **kwargs)
            yield _extract_text(full)
            return

        async for chunk in result:  # type: ignore[operator]
            delta = _extract_delta_text(chunk)
            if delta:
                yield delta

    async def health_check(self) -> bool:
        try:
            router = _require_router()
            await _acompletion_with_fallback(
                router=router,
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=False,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("llm_healthcheck_failed: %s", exc)
            return False


def _extract_text(response: Any) -> str:
    """Best-effort extraction of a final text response."""
    try:
        # OpenAI-like
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
            if msg is not None:
                return str(getattr(msg, "content", "") or "")
    except asyncio.CancelledError:
        raise
    except Exception:

        import logging
        logging.getLogger(__name__).debug("Ignored exception in /mnt/data/proj_v5/resync/services/llm_service.py", exc_info=True)
    return str(response)


def _extract_delta_text(chunk: Any) -> str:
    """Best-effort extraction of streaming delta text."""
    try:
        # dict schema
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if isinstance(delta, dict):
                    return str(delta.get("content") or "")
                # some providers
                return str(choices[0].get("text") or "")
        # object schema
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                return str(getattr(delta, "content", "") or "")
            return str(getattr(choices[0], "text", "") or "")
    except asyncio.CancelledError:
        raise
    except Exception:

        return ""
    return ""


_llm_service: LLMService | None = None


async def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
