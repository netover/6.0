"""Teams Webhook database models."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TeamsWebhookUser(Base):
    """Teams webhook user permissions model."""

    __tablename__ = "teams_webhook_users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    aad_object_id = Column(String(255), index=True)
    role = Column(String(50), default="viewer")  # viewer, operator, admin
    can_execute_commands = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_activity = Column(DateTime)

    def __repr__(self):
        return f"<TeamsWebhookUser(email={self.email}, role={self.role})>"


class TeamsWebhookAudit(Base):
    """Audit log for Teams webhook interactions."""

    __tablename__ = "teams_webhook_audit"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String(255), index=True)
    user_name = Column(String(255))
    channel_id = Column(String(255))
    channel_name = Column(String(255))
    message_text = Column(Text)
    command_type = Column(String(50))  # query, execute
    was_authorized = Column(Boolean)
    response_sent = Column(Boolean)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self):
        return f"<TeamsWebhookAudit(user={self.user_email}, type={self.command_type})>"
