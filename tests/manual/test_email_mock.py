import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from resync.services.notification.email_service import get_email_service
from resync.settings import get_settings


async def test_email_service():
    print("Testing EmailService...")

    # Mock settings to enable SMTP
    settings = get_settings()
    settings.smtp_enabled = True
    settings.smtp_host = "mock.smtp"
    settings.smtp_port = 587

    service = get_email_service()
    # Force re-read enabled status
    service.enabled = True

    # Mock SMTP
    with patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Test 1: Simple Email
        print("\nTest 1: Simple Email")
        success = await service.send_email(
            to_email="test@example.com",
            subject="Test Subject",
            body="Test Body",
            is_html=False,
        )

        if success:
            print("✅ Simple email sent")
        else:
            print("❌ Simple email failed")

        # Verify calls
        mock_server.send_message.assert_called_once()
        msg = mock_server.send_message.call_args[0][0]
        print(f"   Subject: {msg['Subject']}")
        print(f"   To: {msg['To']}")

        # Reset mock
        mock_server.reset_mock()

        # Test 2: HTML Email with Template
        print("\nTest 2: Template Rendering")
        context = {"title": "Report", "message": "Hello", "data": {"CPU": "90%"}}
        html_body = await service.render_template("default_report.html", context)

        if "<table>" in html_body and "90%" in html_body:
            print("✅ Template rendered correctly")
        else:
            print("❌ Template rendering failed")
            print(html_body)

        success = await service.send_email(
            to_email="admin@example.com", subject="Report", body=html_body, is_html=True
        )

        if success:
            print("✅ HTML email sent")
        else:
            print("❌ HTML email failed")


if __name__ == "__main__":
    asyncio.run(test_email_service())
