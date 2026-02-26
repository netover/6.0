"""Endpoints de exemplo para demonstrar RFC 7807 e HATEOAS.

Este módulo demonstra o uso completo de:
- RFC 7807 (Problem Details for HTTP APIs)
- RFC 8288 (Web Linking / HATEOAS)
- Paginação com links de navegação
- Respostas padronizadas
"""

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from resync.api.models.links import LinkBuilder
from resync.api.models.responses import create_paginated_response
from resync.core.exceptions import ResourceNotFoundError, ValidationError
from resync.core.structured_logger import get_logger

__all__ = [
    "logger",
    "router",
    "Book",
    "BookOut",
    "ISBN",
    "BookCreate",
    "list_books",
    "get_book",
    "create_book",
    "delete_book",
    "get_rfc_examples",
]

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/examples", tags=["RFC Examples"])

# ============================================================================
# MODELS
# ============================================================================

class Book(BaseModel):
    """Modelo de exemplo: Livro."""

    id: str = Field(..., description="ID único do livro")
    title: str = Field(..., description="Título do livro")
    author: str = Field(..., description="Autor do livro")
    isbn: str | None = Field(None, description="ISBN")
    published_year: int | None = Field(None, description="Ano de publicação")
    created_at: str = Field(..., description="Data de criação")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Effective Java",
                "author": "Joshua Bloch",
                "isbn": "978-0132350884",
                "published_year": 2008,
                "created_at": "2024-01-15T10:30:00Z",
            }
        }
    )

class BookOut(Book):
    """Book out."""

    _links: dict[str, Any]

ISBN = Annotated[str, StringConstraints(pattern=r"^[\d-]+$")]

class BookCreate(BaseModel):
    """Request para criar livro."""

    title: str = Field(..., description="Título do livro", min_length=1, max_length=200)
    author: str = Field(..., description="Autor do livro", min_length=1, max_length=100)
    isbn: ISBN | None = Field(None, description="ISBN")
    published_year: int | None = Field(
        None, description="Ano de publicação", ge=1000, le=9999
    )

# Thread-safe lock para operações em _books_db
_books_lock = asyncio.Lock()

# Simulação de banco de dados em memória
_books_db: list[Book] = [
    Book(
        id=str(uuid4()),
        title="Clean Code",
        author="Robert C. Martin",
        isbn="978-0132350884",
        published_year=2008,
        created_at=datetime.now(timezone.utc).isoformat() + "Z",
    ),
    Book(
        id=str(uuid4()),
        title="The Pragmatic Programmer",
        author="Andrew Hunt",
        isbn="978-0201616224",
        published_year=1999,
        created_at=datetime.now(timezone.utc).isoformat() + "Z",
    ),
]

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "/books",
    summary="List books with pagination and HATEOAS",
    description="""
    Lista livros com paginação e links HATEOAS.

    **Características**:
    - Paginação com links de navegação (first, last, prev, next)
    - Links HATEOAS para cada recurso
    - Respostas padronizadas

    **Exemplo de Response**:
    ```json
    {
      "items": [...],
      "total": 100,
      "page": 1,
      "page_size": 10,
      "total_pages": 10,
      "has_next": true,
      "has_previous": false,
      "_links": {
        "self": {"href": "/api/v1/examples/books?page=1", "rel": "self"},
        "next": {"href": "/api/v1/examples/books?page=2", "rel": "next"},
        "first": {"href": "/api/v1/examples/books?page=1", "rel": "first"},
        "last": {"href": "/api/v1/examples/books?page=10", "rel": "last"}
      }
    }
    ```
    """,
)
async def list_books(
    request: Request,
    page: int = Query(1, ge=1, description="Número da página"),
    page_size: int = Query(10, ge=1, le=100, description="Tamanho da página"),
    author: str | None = Query(None, description="Filtrar por autor"),
):
    """Lista livros com paginação e HATEOAS."""

    # Cópia defensiva thread-safe
    async with _books_lock:
        books = list(_books_db)
    
    # Filtrar livros (fora do lock para não bloquear operações)
    if author:
        books = [b for b in books if author.lower() in b.author.lower()]

    # Calcular paginação
    total = len(books)
    start = (page - 1) * page_size
    end = start + page_size
    items = books[start:end]

    # Adicionar links HATEOAS a cada item
    builder = LinkBuilder()
    items_with_links = []

    for book in items:
        book_dict = book.model_dump()
        book_dict["_links"] = {
            "self": builder.build_self_link(
                path=f"/api/v1/examples/books/{book.id}"
            ).model_dump(),
            "update": builder.build_link(
                path=f"/api/v1/examples/books/{book.id}",
                rel="update",
                method="PUT",
                title="Update book",
            ).model_dump(),
            "delete": builder.build_link(
                path=f"/api/v1/examples/books/{book.id}",
                rel="delete",
                method="DELETE",
                title="Delete book",
            ).model_dump(),
        }
        items_with_links.append(book_dict)

    # Criar resposta paginada com links
    query_params = {}
    if author:
        query_params["author"] = author

    return create_paginated_response(
        items=items_with_links,
        total=total,
        page=page,
        page_size=page_size,
        base_path="/api/v1/examples/books",
        query_params=query_params,
    )

@router.get(
    "/books/{book_id}",
    response_model=BookOut,
    summary="Get book by ID with HATEOAS",
    description="""
    Obtém um livro específico com links HATEOAS.

    **Características**:
    - Links para operações relacionadas (update, delete, collection)
    - Tratamento de erro RFC 7807 se não encontrado

    **Exemplo de Response**:
    ```json
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Clean Code",
      "author": "Robert C. Martin",
      "_links": {
        "self": {"href": "/api/v1/examples/books/550e8400...", "rel": "self"},
        "update": {"href": "/api/v1/examples/books/550e8400...", "rel": "update", "method": "PUT"},
        "delete": {"href": "/api/v1/examples/books/550e8400...", "rel": "delete", "method": "DELETE"},
        "collection": {"href": "/api/v1/examples/books", "rel": "collection"}
      }
    }
    ```

    **Exemplo de Erro (RFC 7807)**:
    ```json
    {
      "type": "https://api.resync.com/errors/resource-not-found",
      "title": "Resource Not Found",
      "status": 404,
      "detail": "Book with ID '123' not found",
      "instance": "/api/v1/examples/books/123",
      "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """,
)
async def get_book(book_id: str):
    """Obtém livro por ID com links HATEOAS."""

    # Buscar livro (cópia defensiva thread-safe)
    async with _books_lock:
        book = next((b for b in _books_db if b.id == book_id), None)

    if not book:
        raise ResourceNotFoundError(
            message=f"Book with ID '{book_id}' not found", details={"book_id": book_id}
        )

    # Adicionar links HATEOAS
    builder = LinkBuilder()
    book_dict = book.model_dump()
    book_dict["_links"] = {
        "self": builder.build_self_link(
            path=f"/api/v1/examples/books/{book_id}"
        ).model_dump(),
        "update": builder.build_link(
            path=f"/api/v1/examples/books/{book_id}",
            rel="update",
            method="PUT",
            title="Update this book",
        ).model_dump(),
        "delete": builder.build_link(
            path=f"/api/v1/examples/books/{book_id}",
            rel="delete",
            method="DELETE",
            title="Delete this book",
        ).model_dump(),
        "collection": builder.build_collection_link(
            path="/api/v1/examples/books"
        ).model_dump(),
    }

    return book_dict

@router.post(
    "/books",
    response_model=BookOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new book",
    description="""
    Cria um novo livro.

    **Características**:
    - Validação automática com erros RFC 7807
    - Response com links HATEOAS
    - Status 201 Created

    **Exemplo de Erro de Validação (RFC 7807)**:
    ```json
    {
      "type": "https://api.resync.com/errors/validation-error",
      "title": "Validation Error",
      "status": 400,
      "detail": "Validation failed with 2 error(s)",
      "instance": "/api/v1/examples/books",
      "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2024-01-15T10:30:00Z",
      "errors": [
        {
          "field": "title",
          "message": "Field required",
          "code": "missing"
        },
        {
          "field": "published_year",
          "message": "Input should be less than or equal to 9999",
          "code": "less_than_equal"
        }
      ]
    }
    ```
    """,
)
async def create_book(book_data: BookCreate):
    """Cria um novo livro."""

    # Validação customizada
    if (
        book_data.published_year
        and book_data.published_year > datetime.now(timezone.utc).year
    ):
        raise ValidationError(
            message="Published year cannot be in the future",
            details={
                "field": "published_year",
                "value": book_data.published_year,
                "max_allowed": datetime.now(timezone.utc).year,
            },
        )

    # Criar livro
    book = Book(
        id=str(uuid4()),
        title=book_data.title,
        author=book_data.author,
        isbn=book_data.isbn,
        published_year=book_data.published_year,
        created_at=datetime.now(timezone.utc).isoformat() + "Z",
    )

    # Adicionar ao banco thread-safe
    async with _books_lock:
        _books_db.append(book)

    # Adicionar links HATEOAS
    builder = LinkBuilder()
    book_dict = book.model_dump()
    book_dict["_links"] = {
        "self": builder.build_self_link(
            path=f"/api/v1/examples/books/{book.id}"
        ).model_dump(),
        "update": builder.build_link(
            path=f"/api/v1/examples/books/{book.id}", rel="update", method="PUT"
        ).model_dump(),
        "delete": builder.build_link(
            path=f"/api/v1/examples/books/{book.id}", rel="delete", method="DELETE"
        ).model_dump(),
        "collection": builder.build_collection_link(
            path="/api/v1/examples/books"
        ).model_dump(),
    }

    logger.info("Book created", book_id=book.id, title=book.title)

    return book_dict

@router.delete(
    "/books/{book_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a book",
    description="""
    Deleta um livro.

    **Características**:
    - Status 204 No Content em sucesso
    - Erro RFC 7807 se não encontrado
    """,
)
async def delete_book(book_id: str):
    """Deleta um livro."""

    # Buscar e remover livro thread-safe
    async with _books_lock:
        index = next((i for i, b in enumerate(_books_db) if b.id == book_id), None)
        
        if index is None:
            raise ResourceNotFoundError(
                message=f"Book with ID '{book_id}' not found", details={"book_id": book_id}
            )
        
        deleted_book = _books_db.pop(index)

    logger.info("Book deleted", book_id=book_id, title=deleted_book.title)

@router.get(
    "/rfc-examples",
    summary="Get RFC implementation examples",
    description="Returns documentation about RFC 7807 and RFC 8288 implementation",
)
async def get_rfc_examples():
    """Retorna exemplos de implementação RFC."""
    return {
        "rfc_7807": {
            "name": "Problem Details for HTTP APIs",
            "url": "https://tools.ietf.org/html/rfc7807",
            "description": "Padronização de respostas de erro HTTP",
            "implementation": {
                "error_format": {
                    "type": "URI identifying the problem type",
                    "title": "Short, human-readable summary",
                    "status": "HTTP status code",
                    "detail": "Human-readable explanation",
                    "instance": "URI identifying the specific occurrence",
                    "correlation_id": "For distributed tracing",
                    "timestamp": "ISO 8601 timestamp",
                    "errors": "Array of validation errors (optional)",
                },
                "examples": {
                    "validation_error": "/api/v1/examples/books (POST with invalid data)",
                    "not_found": "/api/v1/examples/books/invalid-id",
                    "business_error": "/api/v1/examples/books (POST with future year)",
                },
            },
        },
        "rfc_8288": {
            "name": "Web Linking (HATEOAS)",
            "url": "https://tools.ietf.org/html/rfc8288",
            "description": "Links para navegação e descoberta de recursos",
            "implementation": {
                "link_format": {
                    "href": "URI of the resource",
                    "rel": "Relation type (self, next, prev, etc.)",
                    "method": "HTTP method",
                    "title": "Human-readable description",
                    "type": "Media type",
                },
                "examples": {
                    "pagination": "/api/v1/examples/books?page=1",
                    "resource_links": "/api/v1/examples/books/{id}",
                    "crud_operations": "All endpoints include CRUD links",
                },
            },
        },
        "testing": {
            "list_books": "GET /api/v1/examples/books?page=1&page_size=5",
            "get_book": "GET /api/v1/examples/books/{id}",
            "create_book": "POST /api/v1/examples/books",
            "delete_book": "DELETE /api/v1/examples/books/{id}",
            "trigger_404": "GET /api/v1/examples/books/invalid-id",
            "trigger_validation": "POST /api/v1/examples/books with empty body",
        },
    }
