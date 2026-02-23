from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from resync.core.security.auth import get_current_admin
from resync.services.notification.email_service import get_email_service
from resync.settings import get_settings

router = APIRouter(prefix="/admin/notifications", tags=["admin", "notifications"])


class SMTPConfigUpdate(BaseModel):
    enabled: bool
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    from_email: EmailStr
    use_tls: bool


class SMTPTestRequest(BaseModel):
    recipient: EmailStr


@router.get("/smtp", response_model=SMTPConfigUpdate)
async def get_smtp_config(_admin: Annotated[dict, Depends(get_current_admin)]):
    """Get current SMTP configuration."""
    settings = get_settings()
    return SMTPConfigUpdate(
        enabled=settings.smtp_enabled,
        host=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_username,
        # Do not return password
        password=None,
        from_email=settings.smtp_from_email,
        use_tls=settings.smtp_use_tls,
    )


@router.put("/smtp")
async def update_smtp_config(
    config: SMTPConfigUpdate, _admin: Annotated[dict, Depends(get_current_admin)]
):
    """Update SMTP configuration (persists to .env not implemented in this demo)."""
    # In a real app, you would write these to .env file or a database settings table.
    # For now, we update the runtime settings singleton.
    settings = get_settings()

    settings.smtp_enabled = config.enabled
    settings.smtp_host = config.host
    settings.smtp_port = config.port
    settings.smtp_username = config.username
    if config.password:
        from pydantic import SecretStr

        settings.smtp_password = SecretStr(config.password)
    settings.smtp_from_email = config.from_email
    settings.smtp_use_tls = config.use_tls

    # Reload service to pick up changes (re-init template env if needed)
    # But EmailService reads settings on init. We might need to refresh it.
    get_email_service().settings = settings
    get_email_service().enabled = settings.smtp_enabled

    return {"message": "SMTP configuration updated successfully (runtime only)"}


@router.post("/smtp/test")
async def test_smtp_config(
    request: SMTPTestRequest, _admin: Annotated[dict, Depends(get_current_admin)]
):
    """Send a test email to verify SMTP configuration."""
    service = get_email_service()

    if not service.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SMTP is disabled in configuration",
        )

    success = await service.send_email(
        to_email=request.recipient,
        subject="Resync SMTP Test",
        body="This is a test email from your Resync monitoring system. If you see this, SMTP is working correctly.",
        is_html=False,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send test email. Check server logs.",
        )

    return {"message": f"Test email sent to {request.recipient}"}
