"""Authentication and user management models."""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    """User login request model."""

    username: str
    password: str
    # Security fields
    captcha_token: str | None = None
    client_fingerprint: str | None = None
    session_token: str | None = None


class UserRegistrationRequest(BaseModel):
    """User registration request model."""

    username: str
    email: str
    password: str
    # Security fields
    captcha_token: str | None = None
    terms_accepted: bool = False
    client_fingerprint: str | None = None


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str
    new_password: str
    confirm_password: str
    # Security fields
    session_token: str | None = None
    client_fingerprint: str | None = None


class Token(BaseModel):
    """OAuth2 token response model."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """JWT token payload data."""

    username: str | None = None
