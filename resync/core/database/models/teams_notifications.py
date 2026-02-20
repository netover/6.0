"""Teams Notifications - Database Models."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TeamsChannel(Base):
    """Canais do Teams configurados para notificações."""

    __tablename__ = "teams_channels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(500))
    webhook_url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    color = Column(String(20), default="#0078D4")  # Cor no frontend
    icon = Column(String(20), default="📢")  # Emoji/ícone
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_notification_sent = Column(DateTime(timezone=True))
    notification_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<TeamsChannel(name={self.name}, active={self.is_active})>"


class TeamsJobMapping(Base):
    """Mapeamento de jobs específicos para canais."""

    __tablename__ = "teams_job_mappings"

    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String(255), unique=True, nullable=False, index=True)
    channel_id = Column(Integer, nullable=False)  # FK to teams_channels
    priority = Column(Integer, default=0)  # Para ordenação
    is_active = Column(Boolean, default=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<TeamsJobMapping(job={self.job_name}, channel_id={self.channel_id})>"


class TeamsPatternRule(Base):
    """Regras de padrões (regex) para mapeamento automático."""

    __tablename__ = "teams_pattern_rules"

    id = Column(Integer, primary_key=True, index=True)
    pattern = Column(String(255), nullable=False)  # Regex ou glob pattern
    channel_id = Column(Integer, nullable=False)  # FK to teams_channels
    description = Column(String(500))
    priority = Column(Integer, default=0)  # Ordem de avaliação (maior = primeiro)
    pattern_type = Column(String(20), default="glob")  # glob, regex, prefix, suffix
    is_active = Column(Boolean, default=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    match_count = Column(Integer, default=0)

    def __repr__(self):
        return (
            f"<TeamsPatternRule(pattern={self.pattern}, channel_id={self.channel_id})>"
        )


class TeamsNotificationConfig(Base):
    """Configuração global de notificações."""

    __tablename__ = "teams_notification_config"

    id = Column(Integer, primary_key=True)
    # Status que devem gerar notificações
    notify_on_status = Column(JSON, default=["ABEND", "ERROR", "FAILED"])
    # Horários silenciosos
    quiet_hours_enabled = Column(Boolean, default=False)
    quiet_hours_start = Column(String(5))  # "22:00"
    quiet_hours_end = Column(String(5))  # "07:00"
    # Rate limiting
    rate_limit_enabled = Column(Boolean, default=True)
    max_notifications_per_job = Column(Integer, default=5)
    rate_limit_window_minutes = Column(Integer, default=60)
    # Canal padrão (fallback)
    default_channel_id = Column(Integer)
    # Outros
    include_mention_on_critical = Column(Boolean, default=False)
    mention_text = Column(String(100), default="@Operations")
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<TeamsNotificationConfig(id={self.id})>"


class TeamsNotificationLog(Base):
    """Log de notificações enviadas."""

    __tablename__ = "teams_notification_log"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, index=True)
    channel_name = Column(String(100))
    job_name = Column(String(255), index=True)
    job_status = Column(String(50))
    instance_name = Column(String(100))
    return_code = Column(Integer)
    error_message = Column(Text)
    notification_sent = Column(Boolean, default=False)
    response_status = Column(Integer)
    error = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self):
        return (
            f"<TeamsNotificationLog(job={self.job_name}, channel={self.channel_name})>"
        )
