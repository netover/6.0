"""Custom template response with CSP nonce support."""

from typing import Any

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


class CSPTemplateResponse(HTMLResponse):
    """
    Custom HTML response that includes CSP nonce in the template context.

    This response class automatically adds the CSP nonce to the template
    context, making it available for use in script tags and other elements
    that require nonce attributes.
    """

    def __init__(
        self,
        template_name: str,
        context: dict[str, Any],
        templates: Jinja2Templates,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background=None,
    ):
        """
        Initialize CSP template response.

        Args:
            template_name: Name of the template to render
            context: Template context dictionary
            templates: Jinja2Templates instance
            status_code: HTTP status code
            headers: Additional headers
            media_type: Media type
            background: Background task
        """
        self.template_name = template_name
        self.context = context
        self.templates = templates

        # Get the request from context
        request = context.get("request")

        # Add CSP nonce to context if available
        if request and hasattr(request.state, "csp_nonce"):
            context["csp_nonce"] = request.state.csp_nonce

        # Render the template
        content = templates.get_template(template_name).render(context)

        super().__init__(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )


def create_csp_template_response(
    template_name: str,
    context: dict[str, Any],
    templates: Jinja2Templates,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    media_type: str | None = None,
    background=None,
) -> CSPTemplateResponse:
    """
    Create a CSP-aware template response.

    This function creates a template response that automatically includes
    the CSP nonce in the template context, making it available for use
    in script tags and other elements.

    Args:
        template_name: Name of the template to render
        context: Template context dictionary (must include 'request')
        templates: Jinja2Templates instance
        status_code: HTTP status code
        headers: Additional headers
        media_type: Media type
        background: Background task

    Returns:
        CSPTemplateResponse instance
    """
    return CSPTemplateResponse(
        template_name=template_name,
        context=context,
        templates=templates,
        status_code=status_code,
        headers=headers,
        media_type=media_type,
        background=background,
    )
