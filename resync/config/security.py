"""
Additional security headers and configuration for Resync application.

UPDATED: 2024-12-23 - Implementação completa conforme OWASP 2024
"""

import structlog

from fastapi import FastAPI

from .enhanced_security import configure_enhanced_security

logger = structlog.get_logger(__name__)


def add_additional_security_headers(app: FastAPI) -> None:
    """
    Add security headers middleware to the application.

    Implementa headers de segurança obrigatórios conforme OWASP:
    - HSTS (Strict-Transport-Security)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    - Cross-Origin Policies

    Args:
        app: FastAPI application instance
    """
    from resync.api.middleware.security_headers import SecurityHeadersMiddleware
    from resync.settings import settings

    # Configure enhanced security features
    configure_enhanced_security(app)

    # Add comprehensive security headers middleware
    # HSTS é ativado apenas em produção (requer HTTPS)
    app.add_middleware(
        SecurityHeadersMiddleware,
        enforce_https=settings.is_production,
        hsts_max_age=63072000,  # 2 anos
        hsts_include_subdomains=True,
        hsts_preload=False,  # Ative manualmente se registrar no HSTS preload list
    )

    logger.info(
        "security_headers_configured",
        enforce_https=settings.is_production,
        environment=settings.environment.value,
    )
