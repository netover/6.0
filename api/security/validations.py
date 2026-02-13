from re import match

from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator

__all__ = ['password_hasher', 'SensitiveFieldValidator', 'EnhancedLoginRequest', 'UserCreateWithValidation', 'TokenRequestWithValidation']


# --- Enhanced Validation Rules ---
password_hasher = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SensitiveFieldValidator:
    """Sensitive field validator."""

    @staticmethod
    def validate_password(password: str) -> str:
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        if match(r"^\d+", password):
            raise ValueError("Password must not start with a digit")
        return password

    @staticmethod
    def validate_email(email: str) -> str:
        # Basic email format validation
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(email):
            raise ValueError("Invalid email format")
        if "@" not in email:
            raise ValueError("Email must contain '@'")
        local, domain = email.split("@")
        if len(local) < 1 or len(local) > 64:
            raise ValueError("Local part must be 1-64 characters")
        if len(domain) < 1 or len(domain) > 255:
            raise ValueError("Domain part must be 1-255 characters")
        return email


class EnhancedLoginRequest(BaseModel):
    """Request model for enhanced login operations."""

    username: str = Field(
        ..., min_length=3, max_length=32, json_schema_extra={"example": "johndoe"}
    )
    password: str = Field(..., min_length=8, json_schema_extra={"example": "SecureP@ss123!"})

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        return SensitiveFieldValidator.validate_password(v)


class UserCreateWithValidation(BaseModel):
    """User create with validation."""

    username: str = Field(
        ..., min_length=3, max_length=32, json_schema_extra={"example": "johndoe"}
    )
    email: EmailStr = Field(..., json_schema_extra={"example": "user@example.com"})
    password: str = Field(..., min_length=8, json_schema_extra={"example": "SecureP@ss123!"})

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        return SensitiveFieldValidator.validate_email(v)


class TokenRequestWithValidation(BaseModel):
    """Request model for token with validation operations."""

    refresh_token: str = Field(
        min_length=512,
        max_length=2048,
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )

    @field_validator("refresh_token")
    @classmethod
    def validate_refresh_token(cls, v):
        if len(v) < 512:
            raise ValueError("Refresh token must be at least 512 characters")
        if not v.startswith("eyJ"):
            raise ValueError("Token must start with 'eyJ' (JWT header)")
        return v
