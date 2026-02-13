"""Teams Notifications - Core Logic."""

import fnmatch
import re
from datetime import datetime, timezone, timedelta

import aiohttp
import structlog
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from resync.core.database.models.teams_notifications import (
    TeamsChannel,
    TeamsJobMapping,
    TeamsNotificationConfig,
    TeamsNotificationLog,
    TeamsPatternRule,
)

logger = structlog.get_logger(__name__)


class TeamsNotificationManager:
    """Gerencia notificações proativas para o Teams."""

    def __init__(self, db_session: Session):
        self.db = db_session
        self._rate_limit_cache: dict[str, list[datetime]] = {}

    def get_channel_for_job(self, job_name: str) -> TeamsChannel | None:
        """
        Determina qual canal notificar para um job.

        Ordem de prioridade:
        1. Mapeamento específico (teams_job_mappings)
        2. Regras de padrões (teams_pattern_rules) por prioridade
        3. Canal padrão (fallback)

        Args:
            job_name: Nome do job

        Returns:
            TeamsChannel ou None se não encontrar
        """

        # 1. Verificar mapeamento específico
        stmt = select(TeamsJobMapping).where(
            and_(
                TeamsJobMapping.job_name == job_name,
                TeamsJobMapping.is_active
            )
        )
        mapping = self.db.execute(stmt).scalar_one_or_none()

        if mapping:
            channel = self.db.get(TeamsChannel, mapping.channel_id)
            if channel and channel.is_active:
                logger.debug(
                    "channel_matched_by_mapping",
                    job=job_name,
                    channel=channel.name
                )
                return channel

        # 2. Verificar regras de padrões (ordenadas por prioridade)
        stmt = select(TeamsPatternRule).where(
            TeamsPatternRule.is_active
        ).order_by(TeamsPatternRule.priority.desc())

        rules = self.db.execute(stmt).scalars().all()

        for rule in rules:
            if self._matches_pattern(job_name, rule.pattern, rule.pattern_type):
                channel = self.db.get(TeamsChannel, rule.channel_id)
                if channel and channel.is_active:
                    # Incrementa contador de uso da regra
                    rule.match_count += 1
                    self.db.commit()

                    logger.debug(
                        "channel_matched_by_pattern",
                        job=job_name,
                        pattern=rule.pattern,
                        channel=channel.name
                    )
                    return channel

        # 3. Canal padrão (fallback)
        config = self._get_config()
        if config and config.default_channel_id:
            channel = self.db.get(TeamsChannel, config.default_channel_id)
            if channel and channel.is_active:
                logger.debug(
                    "channel_matched_by_default",
                    job=job_name,
                    channel=channel.name
                )
                return channel

        logger.warning("no_channel_found_for_job", job=job_name)
        return None

    def _matches_pattern(self, job_name: str, pattern: str, pattern_type: str) -> bool:
        """Verifica se job_name corresponde ao padrão."""

        try:
            if pattern_type == "glob":
                return fnmatch.fnmatch(job_name, pattern)

            if pattern_type == "regex":
                return bool(re.match(pattern, job_name))

            if pattern_type == "prefix":
                return job_name.startswith(pattern)

            if pattern_type == "suffix":
                return job_name.endswith(pattern)

            if pattern_type == "contains":
                return pattern in job_name

            logger.warning("unknown_pattern_type", type=pattern_type)
            return False

        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error(
                "pattern_match_error",
                pattern=pattern,
                type=pattern_type,
                error=str(e)
            )
            return False

    def should_notify(self, job_name: str, job_status: str) -> bool:
        """
        Verifica se deve notificar baseado em filtros e rate limiting.

        Args:
            job_name: Nome do job
            job_status: Status do job (ABEND, ERROR, etc.)

        Returns:
            True se deve notificar, False caso contrário
        """

        config = self._get_config()

        # Verificar se status está na lista de notificação
        if config and job_status not in config.notify_on_status:
            logger.debug(
                "status_not_in_notify_list",
                job=job_name,
                status=job_status
            )
            return False

        # Verificar horários silenciosos
        if config and config.quiet_hours_enabled:
            if self._is_quiet_hours(config.quiet_hours_start, config.quiet_hours_end):
                logger.debug(
                    "quiet_hours_active",
                    job=job_name
                )
                return False

        # Verificar rate limiting
        if config and config.rate_limit_enabled and not self._check_rate_limit(
            job_name,
            config.max_notifications_per_job,
            config.rate_limit_window_minutes
        ):
            logger.warning(
                "rate_limit_exceeded",
                job=job_name
            )
            return False

        return True

    def _is_quiet_hours(self, start: str, end: str) -> bool:
        """Verifica se está em horário silencioso."""

        if not start or not end:
            return False

        now = datetime.now(timezone.utc).time()
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()

        if start_time <= end_time:
            return start_time <= now <= end_time
        # Atravessa meia-noite
        return now >= start_time or now <= end_time

    def _check_rate_limit(
        self,
        job_name: str,
        max_count: int,
        window_minutes: int
    ) -> bool:
        """
        Verifica rate limit para um job.

        Returns:
            True se pode enviar, False se excedeu limite
        """

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=window_minutes)

        # Limpa cache antigo
        if job_name in self._rate_limit_cache:
            self._rate_limit_cache[job_name] = [
                ts for ts in self._rate_limit_cache[job_name]
                if ts > window_start
            ]
        else:
            self._rate_limit_cache[job_name] = []

        # Verifica limite
        if len(self._rate_limit_cache[job_name]) >= max_count:
            return False

        # Adiciona timestamp atual
        self._rate_limit_cache[job_name].append(now)
        return True

    async def send_job_notification(
        self,
        job_name: str,
        job_status: str,
        instance_name: str,
        return_code: int | None = None,
        error_message: str | None = None,
        timestamp: str | None = None
    ) -> bool:
        """
        Envia notificação de job para o canal apropriado.

        Args:
            job_name: Nome do job
            job_status: Status (ABEND, ERROR, FAILED, etc.)
            instance_name: Nome da instância TWS
            return_code: Código de retorno
            error_message: Mensagem de erro
            timestamp: Timestamp do evento

        Returns:
            True se enviou com sucesso, False caso contrário
        """

        # Verificar se deve notificar
        if not self.should_notify(job_name, job_status):
            logger.debug(
                "notification_skipped",
                job=job_name,
                status=job_status
            )
            return False

        # Obter canal
        channel = self.get_channel_for_job(job_name)
        if not channel:
            logger.warning(
                "no_channel_for_job",
                job=job_name
            )
            return False

        # Criar mensagem
        card = self._create_job_notification_card(
            job_name=job_name,
            job_status=job_status,
            instance_name=instance_name,
            return_code=return_code,
            error_message=error_message,
            timestamp=timestamp,
            channel=channel
        )

        # Enviar para Teams
        success, response_status, error = await self._send_to_teams(
            channel.webhook_url,
            card
        )

        # Registrar log
        log_entry = TeamsNotificationLog(
            channel_id=channel.id,
            channel_name=channel.name,
            job_name=job_name,
            job_status=job_status,
            instance_name=instance_name,
            return_code=return_code,
            error_message=error_message,
            notification_sent=success,
            response_status=response_status,
            error=error
        )
        self.db.add(log_entry)

        # Atualizar canal
        if success:
            channel.last_notification_sent = datetime.now(timezone.utc)
            channel.notification_count += 1

        self.db.commit()

        logger.info(
            "job_notification_sent" if success else "job_notification_failed",
            job=job_name,
            channel=channel.name,
            status=job_status,
            success=success
        )

        return success

    def _create_job_notification_card(
        self,
        job_name: str,
        job_status: str,
        instance_name: str,
        return_code: int | None,
        error_message: str | None,
        timestamp: str | None,
        channel: TeamsChannel
    ) -> dict:
        """Cria Adaptive Card para notificação de job."""

        # Determinar cor e emoji baseado no status
        status_config = {
            "ABEND": {"color": "attention", "emoji": "🚨", "title": "JOB ABEND"},
            "ERROR": {"color": "attention", "emoji": "❌", "title": "JOB ERROR"},
            "FAILED": {"color": "warning", "emoji": "⚠️", "title": "JOB FAILED"},
            "WARNING": {"color": "warning", "emoji": "⚠️", "title": "JOB WARNING"}
        }

        config = status_config.get(job_status, {
            "color": "default",
            "emoji": "ℹ️",
            "title": f"JOB {job_status}"
        })

        # Adicionar ícone do canal
        title = f"{channel.icon} {config['emoji']} {config['title']}"

        facts = [
            {"title": "Job:", "value": job_name},
            {"title": "Canal:", "value": channel.name},
            {"title": "Status:", "value": job_status},
            {"title": "Instância:", "value": instance_name}
        ]

        if return_code is not None:
            facts.append({"title": "Return Code:", "value": str(return_code)})

        if timestamp:
            facts.append({"title": "Timestamp:", "value": timestamp})

        card = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "bolder",
                            "size": "large",
                            "color": config["color"]
                        },
                        {
                            "type": "FactSet",
                            "facts": facts
                        }
                    ],
                    "actions": [
                        {
                            "type": "Action.OpenUrl",
                            "title": "🔍 Ver no Resync",
                            "url": f"https://resync.company.com/jobs/{job_name}"
                        }
                    ]
                }
            }]
        }

        # Adicionar mensagem de erro se houver
        if error_message:
            card["attachments"][0]["content"]["body"].insert(2, {
                "type": "TextBlock",
                "text": f"**Erro:** {error_message[:500]}",
                "wrap": True,
                "color": "attention"
            })

        # Adicionar mention se configurado
        config_obj = self._get_config()
        if config_obj and config_obj.include_mention_on_critical:
            if job_status in ["ABEND", "ERROR"]:
                card["attachments"][0]["content"]["body"].insert(0, {
                    "type": "TextBlock",
                    "text": config_obj.mention_text,
                    "weight": "bolder"
                })

        return card

    async def _send_to_teams(
        self,
        webhook_url: str,
        card: dict
    ) -> tuple[bool, int | None, str | None]:
        """
        Envia mensagem para Teams via webhook.

        Returns:
            (success, response_status, error_message)
        """

        try:
            async with aiohttp.ClientSession() as session, session.post(
                webhook_url,
                json=card,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                success = response.status == 200

                if not success:
                    error = await response.text()
                    return False, response.status, error[:500]

                return True, response.status, None

        except aiohttp.ClientTimeout:
            return False, None, "Timeout"
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            return False, None, str(e)[:500]

    def _get_config(self) -> TeamsNotificationConfig | None:
        """Obtém configuração global."""
        return self.db.query(TeamsNotificationConfig).first()

    def is_enabled(self) -> bool:
        """Verifica se notificações do Teams estão habilitadas."""
        config = self._get_config()
        if not config:
            return False
        return config.enabled

    async def test_channel(self, channel_id: int) -> bool:
        """
        Envia notificação de teste para um canal.

        Args:
            channel_id: ID do canal

        Returns:
            True se enviou com sucesso
        """

        channel = self.db.get(TeamsChannel, channel_id)
        if not channel:
            return False

        card = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"{channel.icon} 🧪 Teste de Notificação",
                            "weight": "bolder",
                            "size": "large"
                        },
                        {
                            "type": "TextBlock",
                            "text": f"Este é um teste de notificação para o canal **{channel.name}**.",
                            "wrap": True
                        },
                        {
                            "type": "TextBlock",
                            "text": f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
                            "size": "small",
                            "isSubtle": True
                        }
                    ]
                }
            }]
        }

        success, _, _ = await self._send_to_teams(channel.webhook_url, card)
        return success
