from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.litellm_config_store import load_litellm_config_async
from resync.core.litellm_init import get_litellm_router

router = APIRouter(prefix="/llm", tags=["Admin - LiteLLM Health"])


class ModelStatus(BaseModel):
    model_name: str
    ok: bool
    latency_ms: int | None = None
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    statuses: list[ModelStatus] = Field(default_factory=list)


@router.get(
    "/health/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def list_models() -> ModelsResponse:
    cfg = await load_litellm_config_async()
    model_list = cfg.raw.get("model_list") or []
    names: list[str] = []
    for item in model_list:
        if isinstance(item, dict) and isinstance(item.get("model_name"), str):
            names.append(item["model_name"])
    # also include common aliases
    aliases = cfg.raw.get("model_aliases") or {}
    if isinstance(aliases, dict):
        for a in aliases.keys():
            if isinstance(a, str) and a not in names:
                names.append(a)
    return ModelsResponse(models=names[:50])


async def _ping_one(model_name: str) -> ModelStatus:
    router = get_litellm_router()
    t0 = time.time()
    try:
        # completion ping
        await router.acompletion(model=model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        latency_ms = int((time.time() - t0) * 1000)
        return ModelStatus(model_name=model_name, ok=True, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.time() - t0) * 1000)
        return ModelStatus(model_name=model_name, ok=False, latency_ms=latency_ms, error=str(exc)[:300])


@router.post(
    "/health/ping",
    response_model=HealthResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def ping_models(models: list[str] | None = None) -> HealthResponse:
    if not models:
        cfg = await load_litellm_config_async()
        model_list = cfg.raw.get("model_list") or []
        models = [i.get("model_name") for i in model_list if isinstance(i, dict) and isinstance(i.get("model_name"), str)]
        models = models[:10]
    statuses = await asyncio.gather(*[_ping_one(m) for m in models[:20]])
    return HealthResponse(statuses=statuses)


class EmbeddingPingRequest(BaseModel):
    model: str | None = None
    inputs: list[str] = Field(default_factory=lambda: ["ping"])


class EmbeddingPingResponse(BaseModel):
    model: str
    ok: bool
    latency_ms: int | None = None
    error: str | None = None


@router.post(
    "/health/ping-embedding",
    response_model=EmbeddingPingResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def ping_embedding_model(req: EmbeddingPingRequest) -> EmbeddingPingResponse:
    """Ping embedding model (defaults to canonical embedding model from config)."""
    cfg = await load_litellm_config_async()
    default_model: str | None = None
    if isinstance(cfg.raw.get("embedding_model"), str):
        default_model = cfg.raw["embedding_model"]
    if default_model is None:
        for item in cfg.raw.get("model_list") or []:
            if isinstance(item, dict):
                params = item.get("litellm_params") or {}
                if isinstance(params, dict):
                    m = params.get("model")
                    if isinstance(m, str) and "embed" in m.lower():
                        mname = item.get("model_name")
                        if isinstance(mname, str):
                            default_model = mname
                            break
    model_name = req.model or default_model or "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
    router = get_litellm_router()
    t0 = time.time()
    try:
        fn = getattr(router, "aembedding", None) or getattr(router, "embedding", None)
        if fn is None:
            raise RuntimeError("router_embedding_not_supported")
        if asyncio.iscoroutinefunction(fn):
            await fn(model=model_name, input=req.inputs)
        else:
            await asyncio.to_thread(fn, model=model_name, input=req.inputs)
        latency_ms = int((time.time() - t0) * 1000)
        return EmbeddingPingResponse(model=model_name, ok=True, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.time() - t0) * 1000)
        return EmbeddingPingResponse(model=model_name, ok=False, latency_ms=latency_ms, error=str(exc)[:300])