"""Utilitários para construção de links HATEOAS (RFC 8288).

Este módulo implementa helpers para criar links seguindo o padrão RFC 8288
(Web Linking), facilitando a navegação e descoberta de recursos na API.

Referência: https://tools.ietf.org/html/rfc8288
"""

from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# MODELS
# ============================================================================


class Link(BaseModel):
    """Modelo de link HATEOAS.

    Attributes:
        href: URI do recurso
        rel: Relação do link (self, next, prev, etc.)
        method: Método HTTP (GET, POST, etc.)
        title: Título descritivo do link
        type: Tipo de mídia esperado
    """

    href: str = Field(
        ...,
        description="URI do recurso",
        json_schema_extra={"example": "/api/v1/resources/123"},
    )

    rel: str = Field(
        ..., description="Relação do link", json_schema_extra={"example": "self"}
    )

    method: str = Field(
        default="GET", description="Método HTTP", json_schema_extra={"example": "GET"}
    )

    title: str | None = Field(
        None,
        description="Título descritivo",
        json_schema_extra={"example": "Get resource details"},
    )

    type: str | None = Field(
        None, description="Tipo de mídia", examples=["application/json"]
    )


class HATEOASResponse(BaseModel):
    """Response com suporte a HATEOAS.

    Attributes:
        data: Dados do recurso
        _links: Links relacionados
    """

    data: Any = Field(..., description="Dados do recurso")

    links: dict[str, Link] = Field(
        default_factory=dict,
        alias="_links",
        description="Links relacionados ao recurso",
    )

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# LINK BUILDERS
# ============================================================================


class LinkBuilder:
    """Construtor de links HATEOAS."""

    def __init__(self, base_url: str = ""):
        """Inicializa o builder.

        Args:
            base_url: URL base da API
        """
        self.base_url = base_url.rstrip("/")

    def build_link(
        self,
        path: str,
        rel: str,
        method: str = "GET",
        title: str | None = None,
        type: str = "application/json",
        query_params: dict[str, Any] | None = None,
    ) -> Link:
        """Constrói um link.

        Args:
            path: Caminho do recurso
            rel: Relação do link
            method: Método HTTP
            title: Título descritivo
            type: Tipo de mídia
            query_params: Parâmetros de query string

        Returns:
            Link configurado
        """
        # Construir URL completa
        href = f"{self.base_url}{path}"

        # Adicionar query params se fornecidos
        if query_params:
            query_string = urlencode(query_params)
            href = f"{href}?{query_string}"

        return Link(href=href, rel=rel, method=method, title=title, type=type)

    def build_self_link(self, path: str, title: str | None = None) -> Link:
        """Constrói link 'self'.

        Args:
            path: Caminho do recurso
            title: Título descritivo

        Returns:
            Link self
        """
        return self.build_link(
            path=path, rel="self", method="GET", title=title or "Current resource"
        )

    def build_collection_link(self, path: str, title: str | None = None) -> Link:
        """Constrói link para coleção.

        Args:
            path: Caminho da coleção
            title: Título descritivo

        Returns:
            Link collection
        """
        return self.build_link(
            path=path,
            rel="collection",
            method="GET",
            title=title or "Resource collection",
        )

    def build_pagination_links(
        self,
        base_path: str,
        page: int,
        page_size: int,
        total_pages: int,
        query_params: dict[str, Any] | None = None,
    ) -> dict[str, Link]:
        """Constrói links de paginação.

        Args:
            base_path: Caminho base
            page: Página atual
            page_size: Tamanho da página
            total_pages: Total de páginas
            query_params: Parâmetros adicionais

        Returns:
            Dicionário com links de paginação
        """
        links = {}
        base_params = query_params or {}

        # Self
        self_params = {**base_params, "page": page, "page_size": page_size}
        links["self"] = self.build_link(
            path=base_path, rel="self", query_params=self_params, title="Current page"
        )

        # First
        first_params = {**base_params, "page": 1, "page_size": page_size}
        links["first"] = self.build_link(
            path=base_path, rel="first", query_params=first_params, title="First page"
        )

        # Last
        last_params = {**base_params, "page": total_pages, "page_size": page_size}
        links["last"] = self.build_link(
            path=base_path, rel="last", query_params=last_params, title="Last page"
        )

        # Previous
        if page > 1:
            prev_params = {**base_params, "page": page - 1, "page_size": page_size}
            links["prev"] = self.build_link(
                path=base_path,
                rel="prev",
                query_params=prev_params,
                title="Previous page",
            )

        # Next
        if page < total_pages:
            next_params = {**base_params, "page": page + 1, "page_size": page_size}
            links["next"] = self.build_link(
                path=base_path, rel="next", query_params=next_params, title="Next page"
            )

        return links

    def build_crud_links(
        self,
        resource_path: str,
        resource_id: str | None = None,
        collection_path: str | None = None,
    ) -> dict[str, Link]:
        """Constrói links CRUD para um recurso.

        Args:
            resource_path: Caminho do recurso
            resource_id: ID do recurso (se existir)
            collection_path: Caminho da coleção

        Returns:
            Dicionário com links CRUD
        """
        links = {}

        if resource_id:
            # Links para recurso existente
            item_path = f"{resource_path}/{resource_id}"

            links["self"] = self.build_link(
                path=item_path, rel="self", method="GET", title="Get resource"
            )

            links["update"] = self.build_link(
                path=item_path, rel="update", method="PUT", title="Update resource"
            )

            links["patch"] = self.build_link(
                path=item_path,
                rel="patch",
                method="PATCH",
                title="Partially update resource",
            )

            links["delete"] = self.build_link(
                path=item_path, rel="delete", method="DELETE", title="Delete resource"
            )

        # Link para coleção
        if collection_path:
            links["collection"] = self.build_link(
                path=collection_path,
                rel="collection",
                method="GET",
                title="Resource collection",
            )

            links["create"] = self.build_link(
                path=collection_path,
                rel="create",
                method="POST",
                title="Create new resource",
            )

        return links


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def add_hateoas_links(data: Any, links: dict[str, Link]) -> HATEOASResponse:
    """Adiciona links HATEOAS a uma resposta.

    Args:
        data: Dados da resposta
        links: Links a serem adicionados

    Returns:
        Response com links HATEOAS
    """
    return HATEOASResponse(data=data, links=links)


def build_link_header(links: dict[str, Link]) -> str:
    """Constrói header Link (RFC 8288).

    Args:
        links: Dicionário de links

    Returns:
        String formatada para header Link

    Example:
        >>> links = {"next": Link(href="/page/2", rel="next")}
        >>> build_link_header(links)
        '</page/2>; rel="next"'
    """
    link_parts = []

    for link in links.values():
        parts = [f"<{link.href}>"]
        parts.append(f'rel="{link.rel}"')

        if link.title:
            parts.append(f'title="{link.title}"')

        if link.type:
            parts.append(f'type="{link.type}"')

        link_parts.append("; ".join(parts))

    return ", ".join(link_parts)


__all__ = [
    "Link",
    "HATEOASResponse",
    "LinkBuilder",
    "add_hateoas_links",
    "build_link_header",
]
