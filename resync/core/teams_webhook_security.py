"""Teams Webhook Security - HMAC validation."""

import base64
import hashlib
import hmac

import structlog

logger = structlog.get_logger(__name__)


def verify_teams_hmac_signature(
    request_body: bytes, signature: str, security_token: str
) -> bool:
    """
    Verifica assinatura HMAC-SHA256 do Teams.

    Teams envia: Authorization: HMAC <base64_signature>

    Args:
        request_body: Corpo da requisição em bytes
        signature: Assinatura Base64 do header Authorization
        security_token: Token de segurança configurado

    Returns:
        True se assinatura válida, False caso contrário
    """
    try:
        # Decodifica assinatura recebida
        expected_signature = base64.b64decode(signature)

        # Calcula HMAC-SHA256
        calculated_hmac = hmac.new(
            security_token.encode("utf-8"), request_body, hashlib.sha256
        ).digest()

        # Compara de forma segura (evita timing attacks)
        is_valid = hmac.compare_digest(expected_signature, calculated_hmac)

        if not is_valid:
            logger.warning(
                "hmac_signature_mismatch",
                expected_len=len(expected_signature),
                calculated_len=len(calculated_hmac),
            )

        return is_valid

    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("hmac_verification_error", error=str(e))
        return False


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """
    Extrai token do header Authorization.

    Formato: "HMAC <base64_token>" ou "Bearer <token>"
    """
    if not authorization_header:
        return None

    parts = authorization_header.split(" ", 1)
    if len(parts) != 2:
        return None

    prefix, token = parts
    if prefix.upper() not in ["HMAC", "BEARER"]:
        return None

    return token
