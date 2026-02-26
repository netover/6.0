"""
User models for authentication.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

class UserRole(str, Enum):
    """User roles in the system."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"

class UserBase(BaseModel):
    """Base user model with common fields."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str | None = None
    full_name: str | None = None
    role: UserRole = UserRole.USER
    is_active: bool = True

class UserCreate(UserBase):
    """Model for creating a new user."""

    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """Model for updating a user."""

    email: str | None = None
    full_name: str | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    password: str | None = None

class UserInDB(UserBase):
    """User model as stored in database."""

    id: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None
    permissions: list[str] = []

    model_config = ConfigDict(from_attributes=True)

class User(UserBase):
    """User model returned to clients (no password)."""

    id: str
    created_at: datetime
    permissions: list[str] = []

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    """JWT token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # User ID
    username: str
    role: str
    permissions: list[str] = []
    exp: datetime
