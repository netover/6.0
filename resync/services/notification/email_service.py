"""Email service for sending notifications and reports.

This module provides the EmailService class to handle SMTP communication,
template rendering, and attachment management.
"""

from __future__ import annotations

import asyncio
import smtplib
from email import encoders
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from resync.core.structured_logger import get_logger
from resync.settings import get_settings

logger = get_logger(__name__)

class EmailService:
    """Service to send emails using SMTP with template support."""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.smtp_enabled
        self.template_env = self._init_template_env()

    def _init_template_env(self) -> Environment:
        """Initialize Jinja2 environment."""
        template_dir = self.settings.base_dir / "templates" / "email"
        return Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )

    async def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Render an email template asynchronously."""
        try:
            template = self.template_env.get_template(template_name)
            return await template.render_async(**context)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error(
                "email_template_render_error", template=template_name, error=str(e)
            )
            raise RuntimeError(
                f"Failed to render email template {template_name}"
            ) from e

    async def send_email(
        self,
        to_email: str | list[str],
        subject: str,
        body: str,
        is_html: bool = True,
        attachments: list[Path | str] | None = None,
    ) -> bool:
        """
        Send an email asynchronously (runs blocking SMTP in a thread).

        Args:
            to_email: Recipient email(s)
            subject: Email subject
            body: Email body content
            is_html: True if body is HTML, False for plain text
            attachments: List of file paths to attach

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            logger.warning(
                "email_service_disabled", action="send_email", recipient=to_email
            )
            return False

        if isinstance(to_email, str):
            recipients = [to_email]
        else:
            recipients = to_email

        # Prepare message
        msg = MIMEMultipart() if attachments else EmailMessage()
        msg["From"] = self.settings.smtp_from_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        if attachments:
            # Multipart message
            msg.attach(MIMEText(body, "html" if is_html else "plain"))
            for file_path in attachments:
                path = Path(file_path)
                if not path.exists():
                    logger.warning("email_attachment_missing", path=str(path))
                    continue

                try:
                    part = MIMEBase("application", "octet-stream")
                    payload = await asyncio.to_thread(path.read_bytes)
                    part.set_payload(payload)
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {path.name}",
                    )
                    msg.attach(part)
                except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                    import sys as _sys
                    from resync.core.exception_guard import maybe_reraise_programming_error
                    _exc_type, _exc, _tb = _sys.exc_info()
                    maybe_reraise_programming_error(_exc, _tb)

                    logger.error("email_attachment_error", path=str(path), error=str(e))
        else:
            # Simple message
            if isinstance(msg, EmailMessage):
                msg.set_content(body, subtype="html" if is_html else "plain")

        # Send in thread
        try:
            await asyncio.to_thread(self._send_smtp, msg)
            logger.info("email_sent", recipient=to_email, subject=subject)
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            logger.error("email_send_failed", recipient=to_email, error=str(e))
            return False

    def _send_smtp(self, msg: EmailMessage | MIMEMultipart) -> None:
        """Blocking SMTP send logic."""
        try:
            with smtplib.SMTP(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=self.settings.smtp_timeout,
            ) as server:
                if self.settings.smtp_use_tls:
                    server.starttls()

                if self.settings.smtp_username and self.settings.smtp_password:
                    server.login(
                        self.settings.smtp_username,
                        self.settings.smtp_password.get_secret_value(),
                    )

                server.send_message(msg)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise to be caught by the async wrapper
            raise e

# Singleton factory
_email_service_instance: EmailService | None = None

def get_email_service() -> EmailService:
    global _email_service_instance
    if _email_service_instance is None:
        _email_service_instance = EmailService()
    return _email_service_instance
