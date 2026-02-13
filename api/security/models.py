from datetime import datetime

from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator

# --- Password Validation Context ---
password_hasher = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    """Request model for login operations."""

    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v


class UserCreate(BaseModel):
    """User create."""

    username: str = Field(
        ..., min_length=3, max_length=32, json_schema_extra={"example": "johndoe"}
    )
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=8, json_schema_extra={"example": "securepassword123!"})


class UserResponse(BaseModel):
    """Response model for user operations."""

    id: str
    username: str
    email: EmailStr
    created_at: datetime


class TokenRequest(BaseModel):
    """Request model for token operations."""

    refresh_token: str = Field(...)


class OAuthToken(BaseModel):
    """Token model for authentication."""

    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    expires_in: int
