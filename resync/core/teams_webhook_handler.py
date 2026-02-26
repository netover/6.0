"""Teams Webhook Message Handler."""

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

class TeamsWebhookHandler:
    """Processa mensagens recebidas do Teams."""

    def __init__(self, agent_manager, permissions_manager):
        self.agent_manager = agent_manager
        self.permissions_manager = permissions_manager

    def extract_query_from_message(self, text: str, webhook_name: str) -> str:
        """Remove menção do webhook e limpa texto."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove menção
        text = re.sub(f"@{webhook_name}", "", text, flags=re.IGNORECASE)
        # Remove espaços extras
        text = " ".join(text.split())
        return text.strip()

    def detect_command_type(self, query: str) -> str:
        """Detecta se é query (consulta) ou execute (ação)."""
        execute_keywords = [
            "rerun",
            "restart",
            "hold",
            "release",
            "cancel",
            "stop",
            "kill",
            "force",
            "submit",
            "execute",
        ]

        query_lower = query.lower()
        for keyword in execute_keywords:
            if keyword in query_lower:
                return "execute"

        return "query"

    async def process_message(
        self,
        text: str,
        user_email: str,
        user_name: str,
        channel_id: str,
        channel_name: str,
        webhook_name: str,
    ) -> str:
        """Processa mensagem e retorna resposta."""

        # Extrai query limpa
        query = self.extract_query_from_message(text, webhook_name)

        if not query:
            return "Por favor, envie uma mensagem após mencionar o bot."

        # Detecta tipo de comando
        command_type = self.detect_command_type(query)

        # Verifica permissão
        permission = await self.permissions_manager.check_user_permission(
            email=user_email, command_type=command_type
        )

        # Se não tem permissão e é comando de execução
        if not permission["has_permission"] and command_type == "execute":
            logger.warning("unauthorized_execute_attempt", user=user_email, query=query)
            return self._format_unauthorized_response(user_name, query)

        # Processa query via Agent Manager
        try:
            context = {
                "source": "microsoft_teams",
                "user": user_name,
                "user_email": user_email,
                "channel": channel_name,
                "command_type": command_type,
                "has_execute_permission": permission["has_permission"],
            }

            response = await self.agent_manager.process_query(
                query=query, context=context, stream=False
            )

            # Log interação
            await self.permissions_manager.log_interaction(
                user_email=user_email,
                user_name=user_name,
                channel_id=channel_id,
                channel_name=channel_name,
                message_text=query,
                command_type=command_type,
                was_authorized=permission["has_permission"],
                response_sent=True,
            )

            answer = self._extract_answer(response)
            return self._format_response(answer, command_type, permission["role"])

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("query_processing_error", error=str(e))
            return f"❌ Erro ao processar solicitação: {str(e)[:200]}"

    def _extract_answer(self, response: Any) -> str:
        """Extrai resposta do Agent Manager."""
        if isinstance(response, dict):
            return str(response.get("response", response.get("answer", str(response))))
        return str(response)

    def _format_response(self, answer: str, command_type: str, role: str) -> str:
        """Formata resposta para o Teams."""
        # Limita tamanho (Teams tem limite de ~28KB)
        if len(answer) > 3000:
            answer = answer[:3000] + "\n\n... _(resposta truncada)_"

        return answer

    def _format_unauthorized_response(self, user_name: str, query: str) -> str:
        """Resposta para comandos não autorizados."""
        return f"""❌ **Permissão Negada**

Olá {user_name}, você não tem permissão para executar comandos.

**Sua solicitação:** {query[:100]}

**Permissões atuais:**
- ✅ Consultas (status, histórico, etc.)
- ❌ Execução de comandos (rerun, hold, etc.)

Para obter permissões de execução, entre em contato com o administrador do sistema.

_Você ainda pode fazer consultas normalmente!_"""

async def send_teams_message(message: str, webhook_url: str = "", **kwargs) -> bool:
    """Convenience wrapper: send a plain-text Teams message."""
    handler = TeamsWebhookHandler(webhook_url=webhook_url or "")
    return await handler.send_message(message, **kwargs)
