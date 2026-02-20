"""Teams Webhook Permissions System."""

from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from resync.core.database.models.teams import TeamsWebhookAudit, TeamsWebhookUser

logger = structlog.get_logger(__name__)


class TeamsPermissionsManager:
    """Gerencia permissões de usuários do Teams webhook."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def check_user_permission(
        self, email: str, command_type: str = "query"
    ) -> dict[str, Any]:
        """
        Verifica permissão do usuário.

        Args:
            email: Email do usuário
            command_type: "query" ou "execute"

        Returns:
            Dict com: has_permission, role, reason
        """
        # Busca usuário no banco
        stmt = select(TeamsWebhookUser).where(
            TeamsWebhookUser.email == email, TeamsWebhookUser.is_active
        )
        result = self.db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            logger.info("user_not_found_defaulting_to_readonly", email=email)
            return {
                "has_permission": command_type == "query",
                "role": "viewer",
                "reason": "User not registered - read-only mode",
            }

        # Atualiza last_activity
        user.last_activity = datetime.now(timezone.utc)
        self.db.commit()

        # Verifica permissão baseada no tipo de comando
        if command_type == "execute":
            has_permission = user.can_execute_commands
            reason = "authorized" if has_permission else "execute_permission_required"
        else:
            has_permission = True  # Todos podem fazer queries
            reason = "query_allowed"

        logger.info(
            "permission_check",
            email=email,
            role=user.role,
            command_type=command_type,
            has_permission=has_permission,
        )

        return {
            "has_permission": has_permission,
            "role": user.role,
            "reason": reason,
            "user_id": user.id,
        }

    def create_user(
        self,
        email: str,
        name: str,
        role: str = "viewer",
        can_execute: bool = False,
        aad_object_id: str | None = None,
    ) -> TeamsWebhookUser:
        """Cria novo usuário com permissões."""
        user = TeamsWebhookUser(
            email=email,
            name=name,
            role=role,
            can_execute_commands=can_execute,
            aad_object_id=aad_object_id,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        logger.info("user_created", email=email, role=role)
        return user

    def update_user_permissions(
        self, email: str, role: str | None = None, can_execute: bool | None = None
    ) -> TeamsWebhookUser | None:
        """Atualiza permissões do usuário."""
        stmt = select(TeamsWebhookUser).where(TeamsWebhookUser.email == email)
        result = self.db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            return None

        if role is not None:
            user.role = role
        if can_execute is not None:
            user.can_execute_commands = can_execute

        user.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user)

        logger.info("user_permissions_updated", email=email, role=user.role)
        return user

    def log_interaction(
        self,
        user_email: str,
        user_name: str,
        channel_id: str,
        channel_name: str,
        message_text: str,
        command_type: str,
        was_authorized: bool,
        response_sent: bool,
        error_message: str | None = None,
    ) -> None:
        """Registra interação no audit log."""
        audit = TeamsWebhookAudit(
            user_email=user_email,
            user_name=user_name,
            channel_id=channel_id,
            channel_name=channel_name,
            message_text=message_text[:1000],  # Limita tamanho
            command_type=command_type,
            was_authorized=was_authorized,
            response_sent=response_sent,
            error_message=error_message,
        )
        self.db.add(audit)
        self.db.commit()
