from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from resync.core.metrics import business_metrics
from resync.core.metrics.runtime_metrics import runtime_metrics
from resync.core.valkey_init import get_valkey_client, is_valkey_available
from resync.core.litellm_init import calculate_completion_cost

logger = logging.getLogger(__name__)

# Valkey keys (keep compact; used by Admin UI)
VALKEY_LLM_METRICS_HASH = "resync:llm:metrics:v1"   # HSET: requests, errors, tokens_total, latency_ms_total, fallbacks_total, fallbacks_*
VALKEY_LLM_RECENT_LIST = "resync:llm:recent:v1"     # LPUSH JSON, LTRIM 0..199


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_provider_model(model: str | None) -> tuple[str, str]:
    if not model:
        return ("unknown", "unknown")
    if "/" in model:
        provider, rest = model.split("/", 1)
        return (provider, rest)
    return ("unknown", model)


async def _write_valkey_event(event: dict[str, Any]) -> None:
    if not is_valkey_available():
        return
    client = get_valkey_client()
    if client is None:
        return

    try:
        pipe = client.pipeline()
        pipe.hincrby(VALKEY_LLM_METRICS_HASH, "requests", 1)
        if event.get("status") != "success":
            pipe.hincrby(VALKEY_LLM_METRICS_HASH, "errors", 1)
        if (t := event.get("tokens_total")) is not None:
            pipe.hincrby(VALKEY_LLM_METRICS_HASH, "tokens_total", int(t))
        if (lat := event.get("latency_ms")) is not None:
            pipe.hincrby(VALKEY_LLM_METRICS_HASH, "latency_ms_total", int(lat))

        pipe.lpush(VALKEY_LLM_RECENT_LIST, json_dumps(event))
        pipe.ltrim(VALKEY_LLM_RECENT_LIST, 0, 199)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm_valkey_metrics_write_failed: %s", exc, exc_info=False)



async def _write_valkey_fallback_event(event: dict[str, Any]) -> None:
    """Write a fallback telemetry event without counting as a new request."""
    if not is_valkey_available():
        return
    client = get_valkey_client()
    if client is None:
        return
    reason = str(event.get("reason") or "unknown")
    try:
        pipe = client.pipeline()
        pipe.hincrby(VALKEY_LLM_METRICS_HASH, "fallbacks_total", 1)
        pipe.hincrby(VALKEY_LLM_METRICS_HASH, f"fallbacks_{reason}", 1)
        pipe.lpush(VALKEY_LLM_RECENT_LIST, json_dumps(event))
        pipe.ltrim(VALKEY_LLM_RECENT_LIST, 0, 199)
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm_valkey_fallback_write_failed: %s", exc, exc_info=False)


def record_llm_fallback(
    *,
    from_model: str,
    to_model: str,
    reason: str,
    error_type: str | None = None,
    error: str | None = None,
    attempt: int | None = None,
) -> None:
    """Record a model fallback event (best-effort)."""
    provider_from, model_from = _extract_provider_model(from_model)
    provider_to, model_to = _extract_provider_model(to_model)

    # Business metrics (Prometheus-like)
    try:
        business_metrics.llm_fallbacks_total.labels(
            reason=str(reason),
            from_model=f"{provider_from}/{model_from}",
            to_model=f"{provider_to}/{model_to}",
        ).inc()
    except asyncio.CancelledError:
        raise
    except Exception:

        import logging
        logging.getLogger(__name__).debug("Ignored exception in /mnt/data/proj_v5/resync/core/litellm_hooks.py", exc_info=True)

    event: dict[str, Any] = {
        "ts_ms": _now_ms(),
        "status": "fallback",
        "reason": str(reason),
        "from_provider": provider_from,
        "from_model": model_from,
        "to_provider": provider_to,
        "to_model": model_to,
    }
    if attempt is not None:
        event["attempt"] = int(attempt)
    if error_type:
        event["error_type"] = str(error_type)
    if error:
        event["error"] = str(error)[:300]

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_write_valkey_fallback_event(event))


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def on_litellm_success(kwargs: dict[str, Any], response: Any, start_time: float, end_time: float) -> None:
    """LiteLLM success callback (sync). Schedules async writes without blocking."""
    try:
        latency_s = max(0.0, float(end_time) - float(start_time))
    except Exception:  # noqa: BLE001
        latency_s = 0.0

    model = kwargs.get("model")
    provider, model_name = _extract_provider_model(model)

    usage = {}
    try:
        usage = getattr(response, "usage", None) or response.get("usage", {})  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        usage = {}

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    # Internal metrics (Prometheus-like + dashboard snapshot)
    business_metrics.llm_requests_total.labels(provider=provider, model=model_name, status="success").inc()
    business_metrics.llm_latency_seconds.observe(latency_s, labels={"provider": provider, "model": model_name})
    business_metrics.llm_tokens_consumed.observe(prompt_tokens, labels={"provider": provider, "model": model_name, "type": "prompt"})
    business_metrics.llm_tokens_consumed.observe(completion_tokens, labels={"provider": provider, "model": model_name, "type": "completion"})
    business_metrics.llm_tokens_consumed.observe(total_tokens, labels={"provider": provider, "model": model_name, "type": "total"})

    runtime_metrics.llm_requests_total.inc()
    runtime_metrics.llm_tokens_used_total.inc(total_tokens)

    # Cache info (best-effort; LiteLLM responses vary by provider)
    cache_hit = False
    try:
        meta = response.get("litellm_metadata", {})  # type: ignore[union-attr]
        cache_hit = bool(meta.get("cache_hit") or meta.get("semantic_cache_hit"))
    except Exception:  # noqa: BLE001
        cache_hit = False

    event = {
        "ts_ms": _now_ms(),
        "status": "success",
        "provider": provider,
        "model": model_name,
        "latency_ms": int(latency_s * 1000),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_total": total_tokens,
        "cache_hit": cache_hit,
    }

    # Best-effort cost calculation (USD)
    try:
        cost_usd = float(calculate_completion_cost(response))
    except Exception:  # noqa: BLE001
        cost_usd = 0.0
    event["cost_usd"] = cost_usd
    if cost_usd > 0:
        try:
            business_metrics.llm_cost_usd_total.labels(provider=provider, model=model_name).inc(cost_usd)
        except asyncio.CancelledError:
            raise
        except Exception:

            import logging
            logging.getLogger(__name__).debug("Ignored exception in /mnt/data/proj_v5/resync/core/litellm_hooks.py", exc_info=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_write_valkey_event(event))


def on_litellm_failure(kwargs: dict[str, Any], exception: Exception, start_time: float, end_time: float) -> None:
    """LiteLLM failure callback (sync)."""
    try:
        latency_s = max(0.0, float(end_time) - float(start_time))
    except Exception:  # noqa: BLE001
        latency_s = 0.0

    model = kwargs.get("model")
    provider, model_name = _extract_provider_model(model)

    business_metrics.llm_requests_total.labels(provider=provider, model=model_name, status="error").inc()
    business_metrics.llm_latency_seconds.observe(latency_s, labels={"provider": provider, "model": model_name})

    runtime_metrics.llm_requests_total.inc()
    runtime_metrics.llm_errors_total.inc()

    event = {
        "ts_ms": _now_ms(),
        "status": "error",
        "provider": provider,
        "model": model_name,
        "latency_ms": int(latency_s * 1000),
        "error_type": type(exception).__name__,
        "error": str(exception)[:300],
    }

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_write_valkey_event(event))
