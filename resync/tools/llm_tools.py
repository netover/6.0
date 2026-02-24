# pylint
"""
LLM Tools - Ferramentas integradas com LLM Service.

Este módulo define ferramentas que podem ser chamadas pelo LLM através de
function calling, com validação automática via Pydantic e registro no ToolCatalog.

Cada tool é automaticamente registrado e exposto via schemas OpenAI compatíveis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from resync.tools.registry import ToolPermission, UserRole, get_tool_catalog, tool

logger = logging.getLogger(__name__)

PROGRAMMING_ERRORS = (TypeError, KeyError, AttributeError, IndexError)


# =============================================================================
# SCHEMAS PYDANTIC PARA TOOLS
# =============================================================================


class GetJobStatusInput(BaseModel):
    """Input para consulta de status de job."""

    job_name: str = Field(..., description="Nome do job TWS/HWA")
    workspace: str = Field(default="PROD", description="Workspace (PROD/TEST/DEV)")


class GetJobStatusOutput(BaseModel):
    """Output de status de job."""

    job_name: str
    status: str
    last_run: str | None = None
    next_run: str | None = None
    workspace: str


class GetJobLogsInput(BaseModel):
    """Input para obter logs de job."""

    job_name: str = Field(..., description="Nome do job")
    lines: int = Field(default=100, description="Número de linhas", ge=1, le=1000)


# =============================================================================
# TOOLS DECORADAS (Auto-registro + Validação)
# =============================================================================


@tool(
    permission=ToolPermission.READ_ONLY,
    input_schema=GetJobStatusInput,
    output_schema=GetJobStatusOutput,
    tags=["tws", "status", "monitoring"],
)
async def get_job_status(job_name: str, workspace: str = "PROD") -> dict[str, Any]:
    """
    Obtém o status atual de um job TWS/HWA.

    Retorna informações sobre estado de execução, última execução e próxima execução
    programada. Use esta tool para verificar se um job está rodando, falhou ou
    foi executado com sucesso.

    Args:
        job_name: Nome completo do job
        workspace: Workspace onde buscar (padrão: PROD)

    Returns:
        Dict com status atual do job
    """
    from resync.services.tws_service import get_tws_client

    try:
        client: Any = get_tws_client()
        status_data = await client.get_job_status(job_name)

        return {
            "job_name": job_name,
            "status": status_data.get("state", status_data.get("status", "UNKNOWN")),
            "return_code": status_data.get(
                "returnCode", status_data.get("return_code")
            ),
            "last_run": status_data.get("last_execution"),
            "next_run": status_data.get("next_execution"),
            "workspace": workspace,
        }
    except Exception as e:
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error getting job status: %s", e, exc_info=True)
        return {
            "job_name": job_name,
            "status": "ERROR",
            "return_code": None,
            "last_run": None,
            "next_run": None,
            "workspace": workspace,
            "error": str(e),
        }


@tool(
    permission=ToolPermission.READ_ONLY,
    tags=["tws", "monitoring", "troubleshoot"],
)
async def get_failed_jobs(hours: int = 24) -> dict[str, Any]:
    """
    Lista todos os jobs que falharam nas últimas N horas.

    Esta tool é útil para troubleshooting rápido e identificação de padrões
    de falha. Retorna nome do job e mensagem de erro quando disponível.

    Args:
        hours: Janela de tempo em horas (padrão: 24)

    Returns:
        Dict com contagem e lista de jobs falhados
    """
    from resync.services.tws_service import get_tws_client

    try:
        client: Any = get_tws_client()
        jobs = await client.get_jobs(status="ABEND", hours=hours)

        return {
            "count": len(jobs),
            "hours": hours,
            "jobs": [
                {
                    "name": j.get("name", j.get("jobName", "UNKNOWN")),
                    "status": j.get("status", "UNKNOWN"),
                    "return_code": j.get("returnCode", j.get("return_code")),
                    "error": j.get(
                        "errorMessage", j.get("error_msg", "No error message")
                    ),
                    "failed_at": j.get("failed_at", j.get("endTime")),
                }
                for j in jobs[:50]  # Limitar a 50 jobs
            ],
        }
    except Exception as e:
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error getting failed jobs: %s", e, exc_info=True)
        return {"count": 0, "hours": hours, "jobs": [], "error": str(e)}


@tool(
    permission=ToolPermission.READ_ONLY,
    input_schema=GetJobLogsInput,
    tags=["tws", "logs", "troubleshoot"],
)
async def get_job_logs(job_name: str, lines: int = 100) -> dict[str, Any]:
    """
    Recupera os últimos logs de execução de um job.

    Use esta tool para investigar falhas ou verificar o que um job está fazendo.
    Os logs mais recentes aparecem primeiro.

    Args:
        job_name: Nome do job
        lines: Número de linhas a retornar (padrão: 100, máximo: 1000)

    Returns:
        Dict com logs do job
    """
    from resync.services.tws_service import get_tws_client

    try:
        client: Any = get_tws_client()
        jobs = await client.get_jobs(q=job_name, limit=1)
        first_job = jobs[0] if jobs else {}
        log_content = first_job.get("logs") if isinstance(first_job, dict) else None

        return {"job_name": job_name, "lines": lines, "content": log_content or ""}
    except Exception as e:
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error getting job logs: %s", e, exc_info=True)
        return {"job_name": job_name, "lines": 0, "content": "", "error": str(e)}


@tool(
    permission=ToolPermission.READ_ONLY,
    tags=["tws", "monitoring"],
)
async def get_system_health() -> dict[str, Any]:
    """
    Verifica a saúde geral do sistema TWS/HWA.

    Retorna informações sobre o engine, jobs críticos e estatísticas gerais.
    Use para verificar se o sistema está operacional.

    Returns:
        Dict com informações de saúde do sistema
    """
    from resync.services.tws_service import get_tws_client

    try:
        client: Any = get_tws_client()

        # Coletar informações em paralelo
        import asyncio

        gathered: tuple[Any, Any] = await asyncio.gather(
            client.get_engine_info(),
            client.get_jobs(status="ABEND", hours=24),
            return_exceptions=True,
        )
        engine_info, failed_jobs = gathered

        if isinstance(engine_info, Exception):
            engine_status = "ERROR"
            engine_details: dict[str, Any] | str = str(engine_info)
        else:
            engine_status = "HEALTHY"
            engine_details = (
                engine_info if isinstance(engine_info, dict) else {"value": engine_info}
            )

        failed_count = len(failed_jobs) if isinstance(failed_jobs, list) else 0

        return {
            "status": "HEALTHY" if failed_count < 10 else "DEGRADED",
            "engine": {"status": engine_status, "details": engine_details},
            "jobs": {
                "failed_last_24h": failed_count,
                "threshold": 10,
            },
        }
    except Exception as e:
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error getting system health: %s", e, exc_info=True)
        return {"status": "ERROR", "error": str(e)}


@tool(
    permission=ToolPermission.READ_ONLY,
    tags=["tws", "dependencies"],
)
async def get_job_dependencies(job_name: str) -> dict[str, Any]:
    """
    Obtém as dependências de um job (predecessores e sucessores).

    Use para entender o fluxo de trabalho e diagnosticar problemas de dependência.

    Args:
        job_name: Nome do job

    Returns:
        Dict com dependências do job
    """
    from resync.services.tws_service import get_tws_client

    try:
        client: Any = get_tws_client()
        jobs = await client.get_jobs(q=job_name, limit=1)
        first_job = jobs[0] if jobs else {}
        predecessors = (
            first_job.get("predecessors", []) if isinstance(first_job, dict) else []
        )
        successors = (
            first_job.get("successors", []) if isinstance(first_job, dict) else []
        )

        return {
            "job_name": job_name,
            "predecessors": predecessors if isinstance(predecessors, list) else [],
            "successors": successors if isinstance(successors, list) else [],
        }
    except Exception as e:
        if isinstance(e, PROGRAMMING_ERRORS):
            raise
        logger.error("Error getting job dependencies: %s", e, exc_info=True)
        return {
            "job_name": job_name,
            "predecessors": [],
            "successors": [],
            "error": str(e),
        }


# =============================================================================
# CONVERSOR: Pydantic → OpenAI Function Schema
# =============================================================================


def pydantic_to_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Converte Pydantic BaseModel para OpenAI Function JSON Schema.

    Args:
        model: Pydantic BaseModel class

    Returns:
        Dict no formato OpenAI function schema
    """
    schema = model.model_json_schema()

    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


# =============================================================================
# GERADOR AUTOMÁTICO DE TOOLS PARA LLM
# =============================================================================


def get_llm_tools(
    user_role: UserRole | None = None, tags: list[str] | None = None
) -> list[dict]:
    """
    Gera automaticamente lista de tools para LLM baseado no registry.

    Args:
        user_role: Filtrar por role (opcional)
        tags: Filtrar por tags (opcional)

    Returns:
        Lista de tools no formato OpenAI
    """
    catalog = get_tool_catalog()
    tool_defs = catalog.list_tools(user_role=user_role, tags=tags)

    llm_tools = []

    for tool_def in tool_defs:
        # Schema dos parâmetros
        if tool_def.input_schema:
            parameters = pydantic_to_openai_schema(tool_def.input_schema)
        else:
            # Tool sem input - inferir do signature da função
            parameters = {"type": "object", "properties": {}}

        llm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description.strip(),
                    "parameters": parameters,
                },
            }
        )

    logger.info(
        "Generated %s LLM tools for role=%s, tags=%s", len(llm_tools), user_role, tags
    )
    return llm_tools


# =============================================================================
# EXECUTOR CENTRALIZADO (Type-Safe)
# =============================================================================


async def execute_tool_call(
    tool_call,
    user_id: str | None = None,
    user_role: UserRole = UserRole.OPERATOR,
    session_id: str | None = None,
) -> dict:
    """
    Executa uma tool call de forma type-safe.

    Fluxo:
    1. Valida permissão via ToolRegistry
    2. Valida input via Pydantic
    3. Executa tool
    4. Valida output via Pydantic
    5. Retorna resultado

    Args:
        tool_call: Tool call do LLM
        user_id: ID do usuário (para audit)
        user_role: Role do usuário (para permissões)
        session_id: ID da sessão

    Returns:
        Dict com resultado ou erro
    """
    catalog = get_tool_catalog()
    tool_name = tool_call.function.name

    # 1. Get tool definition
    tool_def = catalog.get(tool_name)
    if not tool_def:
        return {"error": f"Tool '{tool_name}' not found"}

    # 2. Check permission
    can_exec, reason = catalog.can_execute(tool_name, user_role)
    if not can_exec:
        logger.warning(
            "Permission denied: %s", reason, extra={"user_role": user_role.value}
        )
        return {"error": reason}

    # 3. Parse arguments
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON arguments: %s", e)
        return {"error": f"Invalid JSON arguments: {e}"}

    # 4. Inject context for audit
    args["_user_id"] = user_id
    args["_user_role"] = user_role.value
    args["_session_id"] = session_id

    # 5. Execute (validation happens inside @tool decorator)
    try:
        result = await tool_def.function(**args)

        logger.info(
            f"Tool executed successfully: {tool_name}",
            extra={"user_id": user_id, "session_id": session_id},
        )

        return {"success": True, "result": result}
    except Exception as e:
        logger.error("Tool execution failed: %s", e, exc_info=True)
        return {"error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tools
    "get_job_status",
    "get_failed_jobs",
    "get_job_logs",
    "get_system_health",
    "get_job_dependencies",
    # Utilities
    "get_llm_tools",
    "execute_tool_call",
    "pydantic_to_openai_schema",
]
