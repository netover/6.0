# ruff: noqa: E501
from __future__ import annotations

from typing import Any

from resync.core.specialists.tools import ToolPermission, tool
from resync.services.notification.email_service import get_email_service


@tool(
    name="send_email_report",
    description="Send a formatted report via email to a user. Supports optional attachment.",
    permission=ToolPermission.READ_WRITE,  # Has side effects (sending email)
    requires_approval=False,  # Could be True in prod
    tags=["email", "notification", "report"],
)
async def send_email_report(
    recipient: str,
    subject: str,
    message: str,
    data: dict[str, Any] | None = None,
    attachment_path: str | None = None,
) -> str:
    """
    Send an email report.

    Args:
        recipient: Email address of the recipient
        subject: Subject line
        message: Main body message
        data: Optional dictionary of key-value pairs to display as a table
        attachment_path: Optional path to a file to attach (e.g. PDF report)

    Returns:
        Status message string
    """
    service = get_email_service()

    if not service.enabled:
        return "Email service is disabled in configuration."

    # Render body
    context = {"title": subject, "message": message, "data": data or {}}

    body = await service.render_template("default_report.html", context)

    attachments = [attachment_path] if attachment_path else None

    success = await service.send_email(
        to_email=recipient,
        subject=subject,
        body=body,
        is_html=True,
        attachments=attachments,
    )

    if success:
        return f"Email sent successfully to {recipient}"
    else:
        return f"Failed to send email to {recipient}. Check logs."
