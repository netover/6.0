"""
Teams Outgoing Webhook Public Endpoint.

Este endpoint recebe mensagens do Microsoft Teams quando usuários mencionam o bot.
Implementa validação HMAC, verificação de permissões e processamento de queries.
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from resync.core.database.session import get_db
from resync.core.teams_permissions import TeamsPermissionsManager
from resync.core.teams_webhook_handler import TeamsWebhookHandler
from resync.core.teams_webhook_security import extract_bearer_token, verify_teams_hmac_signature
from resync.settings import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/teams", tags=["Teams Webhook"])


# =============================================================================
# MODELS
# =============================================================================

class TeamsMessageFrom(BaseModel):
    """Informações do remetente."""
    name: str
    id: str
    aadObjectId: str | None = None
    userPrincipalName: str | None = None


class TeamsConversation(BaseModel):
    """Informações da conversa."""
    id: str
    name: str | None = None


class TeamsMessage(BaseModel):
    """Mensagem recebida do Teams."""
    type: str = "message"
    text: str
    channelId: str
    channelName: str | None = None
    from_: TeamsMessageFrom = Field(alias="from")
    conversation: TeamsConversation
    serviceUrl: str

    model_config = ConfigDict(populate_by_name=True)


class TeamsResponse(BaseModel):
    """Resposta para o Teams."""
    type: str = "message"
    text: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_user_email(message: TeamsMessage) -> str:
    """
    Extrai email do usuário da mensagem.

    Tenta obter userPrincipalName, caso contrário usa fallback.
    """
    # Tenta obter email do Teams
    if hasattr(message.from_, "userPrincipalName") and message.from_.userPrincipalName:
        return message.from_.userPrincipalName.lower()

    # Fallback: cria email baseado no nome
    name = message.from_.name.lower().replace(" ", ".")
    return f"{name}@teams.unknown"


async def get_agent_manager():
    """Dependency: retorna Agent Manager."""
    from resync.core.agent_manager import get_agent_manager as get_am
    return get_am()


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/webhook", response_model=TeamsResponse)
async def teams_outgoing_webhook_endpoint(
    request: Request,
    authorization: str | None = Header(None),
    db: Session = Depends(get_db)
):
    """
    **Endpoint principal do Teams Outgoing Webhook.**

    Fluxo:
    1. Valida assinatura HMAC
    2. Parse da mensagem
    3. Extrai informações do usuário
    4. Verifica permissões
    5. Processa query via Agent Manager
    6. Retorna resposta formatada

    **Segurança:**
    - Validação HMAC obrigatória
    - Sistema de permissões baseado em email
    - Rate limiting automático
    - Logging de todas as interações
    """

    # Verifica se webhook está habilitado
    config = settings.TEAMS_OUTGOING_WEBHOOK

    if not config.get("enabled", False):
        logger.warning("teams_webhook_disabled_attempted_access")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Teams webhook is currently disabled"
        )

    # Lê corpo da requisição
    body_bytes = await request.body()

    # Valida assinatura HMAC
    security_token = config.get("security_token")

    if not security_token:
        logger.error("teams_webhook_no_security_token_configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security token not configured"
        )

    if not authorization:
        logger.warning("teams_webhook_missing_authorization_header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )

    # Extrai signature do header
    signature = extract_bearer_token(authorization)
    if not signature:
        logger.warning("teams_webhook_invalid_authorization_format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format"
        )

    # Verifica HMAC
    if not verify_teams_hmac_signature(body_bytes, signature, security_token):
        logger.warning(
            "teams_webhook_invalid_hmac_signature",
            signature_preview=signature[:20] if len(signature) > 20 else signature
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid HMAC signature"
        )

    logger.debug("teams_webhook_hmac_validated_successfully")

    # Parse mensagem
    try:
        message = TeamsMessage.model_validate_json(body_bytes)
    except Exception as e:
        logger.error("teams_webhook_invalid_message_format", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid message format. Check server logs for details."
        ) from None

    # Extrai informações do usuário
    user_name = message.from_.name
    user_email = extract_user_email(message)
    channel_name = message.channelName or message.conversation.name or "Unknown Channel"

    logger.info(
        "teams_webhook_message_received",
        user_name=user_name,
        user_email=user_email,
        channel=channel_name,
        message_preview=message.text[:100]
    )

    # Inicializa componentes
    try:
        agent_manager = await get_agent_manager()
        permissions_manager = TeamsPermissionsManager(db)
        handler = TeamsWebhookHandler(agent_manager, permissions_manager)

        # Processa mensagem
        answer = await handler.process_message(
            text=message.text,
            user_email=user_email,
            user_name=user_name,
            channel_id=message.channelId,
            channel_name=channel_name,
            webhook_name=config.get("webhook_name", "resync")
        )

        logger.info(
            "teams_webhook_response_sent",
            user_email=user_email,
            response_length=len(answer)
        )

        return TeamsResponse(text=answer)

    except Exception as e:
        logger.error(
            "teams_webhook_processing_error",
            error=str(e),
            user_email=user_email,
            exc_info=True
        )

        # Retorna mensagem de erro amigável
        error_msg = f"""❌ **Erro ao processar solicitação**

Desculpe {user_name}, ocorreu um erro ao processar sua mensagem.

**Erro:** {str(e)[:200]}

Por favor, tente novamente ou entre em contato com o suporte."""

        return TeamsResponse(text=error_msg)


@router.get("/webhook/health")
async def teams_webhook_health():
    """
    Health check do webhook.

    Retorna status de configuração e disponibilidade.
    """
    config = settings.TEAMS_OUTGOING_WEBHOOK

    return {
        "status": "healthy" if config.get("enabled") else "disabled",
        "webhook_enabled": config.get("enabled", False),
        "webhook_configured": bool(config.get("security_token")),
        "webhook_name": config.get("webhook_name", "resync"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
