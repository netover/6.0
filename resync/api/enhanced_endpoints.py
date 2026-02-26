"""
Enhanced API Endpoints - Endpoints otimizados com orchestrator.

Este módulo fornece endpoints aprimorados que usam o Service Orchestrator
para chamadas paralelas e melhor performance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from resync.core.orchestrator import ServiceOrchestrator
from resync.knowledge.retrieval.graph import get_knowledge_graph
from resync.services.llm_service import get_llm_service
from resync.services.tws_service import OptimizedTWSClient

logger = logging.getLogger(__name__)

# Router
enhanced_router = APIRouter(prefix="/api/v2", tags=["enhanced"])
INTERNAL_SERVER_ERROR_DETAIL = "Internal server error. Check server logs for details."

# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_tws_client() -> OptimizedTWSClient:
    """Get TWS client singleton.

    Uses the canonical factory singleton instead of creating a new client
    on every call (which leaked connections and ignored pool settings).
    """
    from resync.core.factories.tws_factory import get_tws_client_singleton

    return get_tws_client_singleton()

_orchestrator_instance = None

async def get_orchestrator(
    tws_client: Annotated[OptimizedTWSClient, Depends(get_tws_client)],
    knowledge_graph: Annotated[Any, Depends(get_knowledge_graph)],
) -> ServiceOrchestrator:
    """Get Service Orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ServiceOrchestrator(
            tws_client=tws_client,  # type: ignore[arg-type]
            knowledge_graph=knowledge_graph,
            max_retries=2,
            timeout_seconds=10,
        )
    return _orchestrator_instance

# =============================================================================
# ENHANCED ENDPOINTS
# =============================================================================

JOB_NAME_PATTERN = r"^[A-Z0-9_\-]{1,64}$"

@enhanced_router.get(
    "/jobs/{job_name}/investigate",
    responses={500: {"description": "Internal Server Error"}},
)
async def investigate_job(
    job_name: Annotated[
        str,
        Path(
            min_length=1,
            max_length=64,
            pattern=JOB_NAME_PATTERN,
            description="TWS job name (uppercase alphanumeric + _ -)",
        ),
    ],
    orchestrator: Annotated[ServiceOrchestrator, Depends(get_orchestrator)],
    include_logs: bool = Query(default=True, description="Include job logs"),
    include_deps: bool = Query(default=True, description="Include dependencies"),
) -> Any:
    """
    Investiga job de forma completa e paralela.

    Busca simultânea de:
    - Status do job
    - Contexto histórico (Knowledge Graph)
    - Logs (opcional)
    - Dependências (opcional)
    - Falhas históricas similares

    Args:
        job_name: Nome do job
        include_logs: Incluir logs?
        include_deps: Incluir dependências?
        orchestrator: Orchestrator (injetado)

    Returns:
        Resultado completo da investigação
    """
    try:
        result = await orchestrator.investigate_job_failure(
            job_name=job_name,
            include_logs=include_logs,
            include_dependencies=include_deps,
        )

        # Retornar com status apropriado
        if result.has_errors:
            return JSONResponse(
                status_code=207,  # Multi-Status (sucesso parcial)
                content={
                    "status": "partial",
                    "success_rate": result.success_rate,
                    "data": {
                        "tws_status": result.tws_status,
                        "kg_context": result.kg_context,
                        "logs": result.tws_logs,
                        "dependencies": result.job_dependencies,
                        "historical_failures": result.historical_failures,
                    },
                    "errors": result.errors,
                },
            )

        return {
            "status": "success",
            "data": {
                "tws_status": result.tws_status,
                "kg_context": result.kg_context,
                "logs": result.tws_logs,
                "dependencies": result.job_dependencies,
                "historical_failures": result.historical_failures,
            },
        }

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error investigating job %s: %s", job_name, e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL
        ) from None

@enhanced_router.get(
    "/system/health",
    responses={503: {"description": "Service Unavailable"}},
)
async def system_health(
    orchestrator: Annotated[ServiceOrchestrator, Depends(get_orchestrator)],
) -> Any:
    """
    Health check completo do sistema em paralelo.

    Verifica:
    - Engine TWS
    - Jobs críticos
    - Jobs falhados recentemente

    Returns:
        Status de saúde do sistema
    """
    try:
        health = await orchestrator.get_system_health()

        status_code = 200 if health["status"] == "HEALTHY" else 503

        return JSONResponse(status_code=status_code, content=health)

    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error checking system health: %s", e, exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "ERROR", "error": "Service temporarily unavailable"},
        )

@enhanced_router.get(
    "/jobs/failed",
    responses={500: {"description": "Internal Server Error"}},
)
async def get_failed_jobs_endpoint(
    tws_client: Annotated[OptimizedTWSClient, Depends(get_tws_client)],
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
) -> Any:
    """
    Lista jobs que falharam nas últimas N horas.

    Args:
        hours: Janela de tempo (padrão: 24h, máximo: 7 dias)
        tws_client: Cliente TWS (injetado)

    Returns:
        Lista de jobs falhados
    """
    try:
        query_jobs = getattr(tws_client, "query_jobs", None)
        query_jobstreams = getattr(tws_client, "query_jobstreams", None)
        if callable(query_jobs):
            jobs = await query_jobs(status="ABEND", hours=hours)
        elif callable(query_jobstreams):
            jobs = await query_jobstreams(status="ABEND", hours=hours)
        else:
            raise HTTPException(
                status_code=500, detail="TWS client missing query methods"
            )

        return {
            "count": len(jobs),
            "hours": hours,
            "jobs": jobs,
        }

    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("Error getting failed jobs: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL
        ) from None

@enhanced_router.get(
    "/jobs/{job_name}/summary",
    responses={
        404: {"description": "Job Not Found"},
        500: {"description": "Internal Server Error"},
    },
)
async def get_job_summary(
    job_name: str,
    tws_client: Annotated[OptimizedTWSClient, Depends(get_tws_client)],
    llm_service: Annotated[Any, Depends(get_llm_service)],
    knowledge_graph: Annotated[Any, Depends(get_knowledge_graph)],
) -> Any:
    """
    Gera sumário inteligente de um job usando LLM + RAG.

    Combina:
    - Status atual (TWS)
    - Contexto histórico (KG)
    - Análise via LLM

    Args:
        job_name: Nome do job
        tws_client: Cliente TWS (injetado)
        llm_service: Serviço LLM (injetado)
        knowledge_graph: Knowledge Graph (injetado)

    Returns:
        Sumário inteligente do job
    """
    try:
        # Buscar informações em paralelo
        status_task = None
        context_task = None

        try:
            async with asyncio.TaskGroup() as tg:
                status_task = tg.create_task(
                    tws_client.get_job_status(job_name)
                    if hasattr(tws_client, "get_job_status")
                    else tws_client.get_job_status_cached(job_name)
                )
                context_task = tg.create_task(
                    knowledge_graph.get_relevant_context(
                        f"informações sobre job {job_name}"
                    )
                )
        except* asyncio.CancelledError:
            logger.warning("Task cancellation during job summary for %s", job_name)
            raise
        except* TimeoutError as exc_group:
            logger.error(
                "Timeout fetching data for job %s: %d operations timed out",
                job_name,
                len(exc_group.exceptions),
                exc_info=True,
            )
        except* Exception as exc_group:
            logger.error(
                "Unexpected errors in job summary for %s: %s",
                job_name,
                [type(e).__name__ for e in exc_group.exceptions],
                exc_info=exc_group,
            )

        # Extração segura sem "dummy tasks"
        status = None
        if status_task and status_task.done() and not status_task.cancelled():
            try:
                status = status_task.result()
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Error extracting job status: %s", e)

        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_name} not found")

        # Contexto é opcional (fail-safe)
        context = None
        if context_task and context_task.done() and not context_task.cancelled():
            try:
                context = context_task.result()
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Error extracting relevant context: %s", e)
                context = f"Erro ao recuperar contexto: {e}"

        # Gerar sumário com LLM
        prompt = f"""Analise o seguinte job TWS e forneça um sumário conciso:

**Job:** {job_name}
**Status Atual:** {status}

**Contexto Histórico:**
{context if not isinstance(context, Exception) else "Nenhum contexto disponível"}

Forneça um sumário executivo em 3-4 sentenças sobre:
1. Estado atual
2. Padrões históricos (se houver)
3. Recomendações (se houver problemas)
"""

        summary = await llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        return {
            "job_name": job_name,
            "status": status,
            "summary": summary,
            "has_context": not isinstance(context, Exception),
        }

    except HTTPException:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error(
            "Error generating job summary for job '%s': %s",
            job_name,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL
        ) from None

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "enhanced_router",
]
