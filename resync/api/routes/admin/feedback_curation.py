# mypy: ignore-errors
# pylint: disable=not-callable
"""
Admin Feedback Curation API - Endpoints para curadoria de feedback.

v5.2.3.20: Implementa o fluxo "Golden Record" para transformar
feedback aprovado em conhecimento pesquisável.

Endpoints:
- GET /pending - Lista feedbacks pendentes de aprovação
- GET /stats - Estatísticas de curadoria
- POST /{feedback_id}/approve - Aprova feedback e incorpora como conhecimento
- POST /{feedback_id}/reject - Rejeita feedback
- DELETE /{feedback_id}/rollback - Remove conhecimento incorporado

Author: Resync Team
Version: 5.2.3.20
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from resync.api.routes.admin.main import verify_admin_credentials
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

# v5.9.5: Added authentication
router = APIRouter(
    prefix="/api/v1/admin/feedback",
    tags=["Admin - Feedback Curation"],
    dependencies=[Depends(verify_admin_credentials)],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ApprovalRequest(BaseModel):
    """Request para aprovar feedback e incorporar conhecimento."""

    reviewer_id: str = Field(..., description="ID do revisor")
    user_correction: str = Field(..., description="Resposta correta (do especialista)")
    incorporate_to_kb: bool = Field(
        True, description="Se deve incorporar ao Knowledge Base"
    )
    notes: str | None = Field(None, description="Notas do revisor")


class RejectionRequest(BaseModel):
    """Request para rejeitar feedback."""

    reviewer_id: str = Field(..., description="ID do revisor")
    reason: str = Field(..., description="Motivo da rejeição")


class FeedbackListItem(BaseModel):
    """Item da lista de feedback."""

    id: int
    session_id: str
    query_text: str | None = None
    response_text: str | None = None
    rating: int | None = None
    feedback_text: str | None = None
    curation_status: str
    created_at: str
    has_correction: bool


class FeedbackDetail(BaseModel):
    """Detalhes completos de um feedback."""

    id: int
    session_id: str
    query_id: str | None = None
    query_text: str | None = None
    response_text: str | None = None
    rating: int | None = None
    feedback_type: str
    feedback_text: str | None = None
    is_positive: bool | None = None
    created_at: str
    curation_status: str
    user_correction: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    incorporated_doc_id: str | None = None


class CurationStats(BaseModel):
    """Estatísticas de curadoria."""

    total: int
    pending: int
    approved: int
    rejected: int
    incorporated: int
    avg_rating: float | None = None
    pending_with_correction: int


class ApprovalResponse(BaseModel):
    """Resposta de aprovação."""

    message: str
    feedback_id: int
    incorporated: bool
    doc_id: str | None = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _get_feedback_by_id(feedback_id: int):
    """Busca feedback por ID usando o repositório."""
    from sqlalchemy import select

    from resync.core.database import Feedback, get_session

    async with get_session() as session:
        result = await session.execute(
            select(Feedback).where(Feedback.id == feedback_id)
        )
        return result.scalar_one_or_none()


async def _update_feedback(feedback_id: int, **updates):
    """Atualiza feedback no banco."""
    from sqlalchemy import update

    from resync.core.database import Feedback, get_session

    async with get_session() as session:
        await session.execute(
            update(Feedback).where(Feedback.id == feedback_id).values(**updates)
        )
        await session.commit()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/pending", response_model=list[FeedbackListItem])
async def list_pending_feedback(
    limit: int = Query(50, ge=1, le=200, description="Máximo de itens"),
    has_negative_rating: bool | None = Query(
        None, description="Filtrar por rating negativo"
    ),
    has_correction: bool | None = Query(
        None, description="Filtrar por feedbacks com correção sugerida"
    ),
):
    """
    Lista feedbacks pendentes de aprovação.

    Prioriza feedbacks com rating negativo e aqueles que já têm
    uma correção sugerida pelo usuário.
    """
    try:
        from sqlalchemy import select

        from resync.core.database import Feedback, get_session

        async with get_session() as session:
            query = select(Feedback).where(Feedback.curation_status == "pending")

            if has_negative_rating is True:
                query = query.where(Feedback.rating <= 2)
            elif has_negative_rating is False:
                query = query.where(Feedback.rating > 2)

            if has_correction is True:
                query = query.where(Feedback.feedback_text.isnot(None))
            elif has_correction is False:
                query = query.where(Feedback.feedback_text.is_(None))

            # Ordenar por rating (mais negativos primeiro) e data
            query = query.order_by(Feedback.rating.asc(), Feedback.created_at.desc())
            query = query.limit(limit)

            result = await session.execute(query)
            feedbacks = result.scalars().all()

            return [
                FeedbackListItem(
                    id=fb.id,
                    session_id=fb.session_id,
                    query_text=fb.query_text,
                    response_text=fb.response_text[:200] + "..."
                    if fb.response_text and len(fb.response_text) > 200
                    else fb.response_text,
                    rating=fb.rating,
                    feedback_text=fb.feedback_text,
                    curation_status=fb.curation_status,
                    created_at=fb.created_at.isoformat() if fb.created_at else "",
                    has_correction=bool(fb.feedback_text),
                )
                for fb in feedbacks
            ]

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("list_pending_feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/{feedback_id}", response_model=FeedbackDetail)
async def get_feedback_detail(feedback_id: int):
    """Obtém detalhes completos de um feedback."""
    try:
        feedback = await _get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback não encontrado")

        return FeedbackDetail(
            id=feedback.id,
            session_id=feedback.session_id,
            query_id=feedback.query_id,
            query_text=feedback.query_text,
            response_text=feedback.response_text,
            rating=feedback.rating,
            feedback_type=feedback.feedback_type,
            feedback_text=feedback.feedback_text,
            is_positive=feedback.is_positive,
            created_at=feedback.created_at.isoformat() if feedback.created_at else "",
            curation_status=feedback.curation_status,
            user_correction=feedback.user_correction,
            approved_by=feedback.approved_by,
            approved_at=feedback.approved_at.isoformat()
            if feedback.approved_at
            else None,
            incorporated_doc_id=feedback.incorporated_doc_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("get_feedback_detail_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.get("/stats", response_model=CurationStats)
async def get_curation_stats():
    """Obtém estatísticas de curadoria."""
    try:
        from sqlalchemy import and_, func, select

        from resync.core.database import Feedback, get_session

        async with get_session() as session:
            # Total
            total_result = await session.execute(select(func.count(Feedback.id)))
            total = total_result.scalar() or 0

            # Por status
            status_query = select(
                Feedback.curation_status, func.count(Feedback.id)
            ).group_by(Feedback.curation_status)
            status_result = await session.execute(status_query)
            status_counts = dict(status_result.all())

            # Rating médio
            avg_result = await session.execute(select(func.avg(Feedback.rating)))
            avg_rating = avg_result.scalar()

            # Pendentes com correção
            pending_with_correction = await session.execute(
                select(func.count(Feedback.id)).where(
                    and_(
                        Feedback.curation_status == "pending",
                        Feedback.feedback_text.isnot(None),
                    )
                )
            )
            pending_correction_count = pending_with_correction.scalar() or 0

            return CurationStats(
                total=total,
                pending=status_counts.get("pending", 0),
                approved=status_counts.get("approved", 0),
                rejected=status_counts.get("rejected", 0),
                incorporated=status_counts.get("incorporated", 0),
                avg_rating=round(avg_rating, 2) if avg_rating else None,
                pending_with_correction=pending_correction_count,
            )

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("get_curation_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.post("/{feedback_id}/approve", response_model=ApprovalResponse)
async def approve_and_incorporate(
    feedback_id: int,
    request: ApprovalRequest,
):
    """
    Aprova feedback e incorpora como conhecimento.

    Este é o endpoint principal do fluxo "Golden Record":

    1. Atualiza o feedback com a correção do especialista
    2. Marca como aprovado
    3. Se incorporate_to_kb=True, cria documento no vector store
    4. Retorna confirmação com doc_id criado

    O documento criado tem máxima prioridade no retrieval porque:
    - source_tier: "verified"
    - authority_tier: 1
    - doc_type: "golden_record"
    """
    try:
        # 1. Buscar feedback
        feedback = await _get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback não encontrado")

        if feedback.curation_status == "incorporated":
            raise HTTPException(
                status_code=400, detail="Feedback já foi incorporado ao Knowledge Base"
            )

        # 2. Verificar se tem query_text (necessário para incorporação)
        if not feedback.query_text:
            raise HTTPException(
                status_code=400,
                detail="Feedback não possui query_text, não é possível incorporar",
            )

        # 3. Atualizar com correção e aprovação
        now = datetime.now(timezone.utc)
        await _update_feedback(
            feedback_id=feedback_id,
            user_correction=request.user_correction,
            curation_status="approved",
            approved_by=request.reviewer_id,
            approved_at=now,
        )

        logger.info(
            "feedback_approved",
            feedback_id=feedback_id,
            reviewer_id=request.reviewer_id,
        )

        # 4. Incorporar ao Knowledge Base se solicitado
        doc_id = None
        if request.incorporate_to_kb:
            from resync.core.continual_learning import get_knowledge_incorporator

            incorporator = get_knowledge_incorporator()
            doc_id = await incorporator.incorporate_feedback(
                feedback_id=feedback_id,
                original_question=feedback.query_text,
                user_correction=request.user_correction,
                reviewer_id=request.reviewer_id,
                metadata={"notes": request.notes} if request.notes else None,
            )

            # Marcar como incorporado
            await _update_feedback(
                feedback_id=feedback_id,
                curation_status="incorporated",
                incorporated_doc_id=doc_id,
            )

            logger.info(
                "feedback_incorporated",
                feedback_id=feedback_id,
                doc_id=doc_id,
            )

        return ApprovalResponse(
            message="O Resync aprendeu com sucesso!"
            if doc_id
            else "Feedback aprovado com sucesso!",
            feedback_id=feedback_id,
            incorporated=doc_id is not None,
            doc_id=doc_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("approve_feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.post("/{feedback_id}/reject")
async def reject_feedback(
    feedback_id: int,
    request: RejectionRequest,
):
    """
    Rejeita um feedback.

    Feedbacks rejeitados não são incorporados ao knowledge base,
    mas ficam registrados para análise posterior.
    """
    try:
        # 1. Buscar feedback
        feedback = await _get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback não encontrado")

        if feedback.curation_status == "incorporated":
            raise HTTPException(
                status_code=400,
                detail="Feedback já foi incorporado, use rollback primeiro",
            )

        # 2. Atualizar status
        now = datetime.now(timezone.utc)
        await _update_feedback(
            feedback_id=feedback_id,
            curation_status="rejected",
            approved_by=request.reviewer_id,
            approved_at=now,
            # Salvar motivo da rejeição no campo de correção
            user_correction=f"[REJEITADO] {request.reason}",
        )

        logger.info(
            "feedback_rejected",
            feedback_id=feedback_id,
            reviewer_id=request.reviewer_id,
            reason=request.reason,
        )

        return {
            "message": "Feedback rejeitado",
            "feedback_id": feedback_id,
            "reason": request.reason,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("reject_feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.delete("/{feedback_id}/rollback")
async def rollback_incorporation(
    feedback_id: int,
    reviewer_id: str = Query(..., description="ID do revisor"),
):
    """
    Remove um documento incorporado (rollback).

    Use caso o conhecimento incorporado esteja incorreto.
    O feedback volta para status "approved" (não deleta o feedback).
    """
    try:
        # 1. Buscar feedback
        feedback = await _get_feedback_by_id(feedback_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback não encontrado")

        if feedback.curation_status != "incorporated":
            raise HTTPException(
                status_code=400, detail="Feedback não está incorporado"
            )

        if not feedback.incorporated_doc_id:
            raise HTTPException(
                status_code=400, detail="Feedback não possui doc_id registrado"
            )

        # 2. Remover do vector store
        from resync.core.continual_learning import get_knowledge_incorporator

        incorporator = get_knowledge_incorporator()
        removed = await incorporator.remove_incorporated(feedback_id)

        if not removed:
            logger.warning(
                "rollback_remove_failed",
                feedback_id=feedback_id,
                doc_id=feedback.incorporated_doc_id,
            )

        # 3. Atualizar status
        await _update_feedback(
            feedback_id=feedback_id,
            curation_status="approved",  # Volta para aprovado
            incorporated_doc_id=None,
        )

        logger.info(
            "feedback_rollback",
            feedback_id=feedback_id,
            reviewer_id=reviewer_id,
            doc_id=feedback.incorporated_doc_id,
        )

        return {
            "message": "Incorporação revertida com sucesso",
            "feedback_id": feedback_id,
            "removed_doc_id": feedback.incorporated_doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("rollback_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e


@router.post("/bulk-approve")
async def bulk_approve(
    feedback_ids: list[int],
    reviewer_id: str = Query(..., description="ID do revisor"),
    incorporate: bool = Query(True, description="Incorporar ao KB"),
):
    """
    Aprova múltiplos feedbacks em lote.

    Nota: Os feedbacks precisam ter feedback_text preenchido
    (que será usado como user_correction).
    """
    try:
        results = {
            "total": len(feedback_ids),
            "approved": 0,
            "incorporated": 0,
            "skipped": 0,
            "errors": [],
        }

        for fid in feedback_ids:
            try:
                feedback = await _get_feedback_by_id(fid)
                if not feedback:
                    results["skipped"] += 1
                    results["errors"].append({"id": fid, "error": "Não encontrado"})
                    continue

                if not feedback.feedback_text:
                    results["skipped"] += 1
                    results["errors"].append({"id": fid, "error": "Sem correção"})
                    continue

                # Usar feedback_text como correção
                request = ApprovalRequest(
                    reviewer_id=reviewer_id,
                    user_correction=feedback.feedback_text,
                    incorporate_to_kb=incorporate,
                )

                response = await approve_and_incorporate(fid, request)
                results["approved"] += 1
                if response.incorporated:
                    results["incorporated"] += 1

            except HTTPException as he:
                results["errors"].append({"id": fid, "error": he.detail})
            except Exception as e:
                results["errors"].append({"id": fid, "error": str(e)})

        return results

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("bulk_approve_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.") from e
