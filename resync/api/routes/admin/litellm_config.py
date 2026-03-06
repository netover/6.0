from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from resync.api.routes.core.auth import verify_admin_credentials
from resync.core.litellm_config_store import load_litellm_config_async, save_litellm_config_async, get_history
from resync.core.litellm_init import reload_litellm_router


router = APIRouter(prefix="/llm", tags=["Admin - LiteLLM Config"])


class LiteLLMConfigResponse(BaseModel):
    yaml: str
    mtime_ns: int
    warnings: list[str] = Field(default_factory=list)


class LiteLLMConfigUpdate(BaseModel):
    yaml: str
    run_smoke: bool = False



class LiteLLMValidateResponse(BaseModel):
    ok: bool
    error: str | None = None


def _warnings_from_text(text: str) -> list[str]:
    warnings: list[str] = []
    if "api_key:" in text and "os.environ/" not in text:
        warnings.append("Inline api_key detected. Prefer os.environ/OPENROUTER_API_KEY etc.")
    return warnings


@router.get(
    "/config",
    response_model=LiteLLMConfigResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_config() -> LiteLLMConfigResponse:
    cfg = await load_litellm_config_async()
    return LiteLLMConfigResponse(yaml=cfg.text, mtime_ns=cfg.mtime_ns, warnings=_warnings_from_text(cfg.text))


@router.put(
    "/config",
    response_model=LiteLLMConfigResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def put_config(update: LiteLLMConfigUpdate) -> LiteLLMConfigResponse:
    cfg = await save_litellm_config_async(update.yaml)
    # Apply immediately without restart.
    reload_litellm_router()
    return LiteLLMConfigResponse(yaml=cfg.text, mtime_ns=cfg.mtime_ns, warnings=_warnings_from_text(cfg.text))


@router.post(
    "/reload",
    response_model=dict[str, Any],
    dependencies=[Depends(verify_admin_credentials)],
)
async def reload_config() -> dict[str, Any]:
    reload_litellm_router()
    return {"status": "ok", "message": "LiteLLM router reloaded"}


@router.post(
    "/validate",
    response_model=LiteLLMValidateResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def validate_config(update: LiteLLMConfigUpdate) -> LiteLLMValidateResponse:
    """Dry-run validate YAML by attempting to construct a LiteLLM Router."""
    try:
        import yaml
        data = yaml.safe_load(update.yaml)
        if not isinstance(data, dict):
            return LiteLLMValidateResponse(ok=False, error="root must be a mapping")

        model_list = data.get("model_list")
        if not isinstance(model_list, list):
            return LiteLLMValidateResponse(ok=False, error="model_list must be a list")

        kwargs: dict[str, Any] = {"model_list": model_list}

        model_aliases = data.get("model_aliases")
        if isinstance(model_aliases, dict):
            kwargs["model_aliases"] = model_aliases

        router_settings = data.get("router_settings")
        if isinstance(router_settings, dict):
            kwargs["router_settings"] = router_settings

        from litellm import Router as LiteLLMRouter  # type: ignore
        router = LiteLLMRouter(**kwargs)

        if update.run_smoke:
            # best-effort smoke: 1 completion + 1 embedding
            try:
                await router.acompletion(model=(list((data.get('model_aliases') or {}).keys())[:1] or ['liteLLM-default'])[0], messages=[{'role':'user','content':'ping'}], max_tokens=1)
            except Exception as exc:
                return LiteLLMValidateResponse(ok=False, error=f'completion_smoke_failed: {exc}')
            try:
                await router.aembedding(model=(data.get('embedding_model') or 'openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free'), input=['ping'])
            except Exception as exc:
                return LiteLLMValidateResponse(ok=False, error=f'embedding_smoke_failed: {exc}')

        return LiteLLMValidateResponse(ok=True)
    except Exception as exc:  # noqa: BLE001
        return LiteLLMValidateResponse(ok=False, error=str(exc))


class LiteLLMHistoryResponse(BaseModel):
    entries: list[dict[str, Any]] = Field(default_factory=list)

@router.get(
    "/config/history",
    response_model=LiteLLMHistoryResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def get_config_history(limit: int = 20) -> LiteLLMHistoryResponse:
    entries = await get_history(limit=limit)
    return LiteLLMHistoryResponse(entries=entries)

@router.post(
    "/config/rollback",
    response_model=LiteLLMConfigResponse,
    dependencies=[Depends(verify_admin_credentials)],
)
async def rollback_config(index: int = 0) -> LiteLLMConfigResponse:
    """Rollback to a previous YAML from history (index 0 = most recent)."""
    entries = await get_history(limit=max(1, index + 1))
    if len(entries) <= index:
        raise ValueError("rollback_index_out_of_range")
    yaml_text = entries[index].get("yaml") or ""
    cfg = await save_litellm_config_async(yaml_text)
    reload_litellm_router()
    return LiteLLMConfigResponse(yaml=cfg.text, mtime_ns=cfg.mtime_ns, warnings=_warnings_from_text(cfg.text))
