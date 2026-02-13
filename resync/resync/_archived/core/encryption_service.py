"""resync.core.encryption_service

Encryption utilities intended for *explicit* use.

Production notes:
- Do not auto-generate encryption keys at runtime. If encryption is enabled, a stable
  key must be provided via configuration (e.g., ENCRYPTION_KEY).
- This module does **not** install global logging filters at import time.
  Resync already performs sensitive-data redaction in structured logging.

The original implementation generated a new key when ENCRYPTION_KEY was not set
and installed an invalid logging filter (a function) on the root logger. Both
behaviors are unsafe/unreliable for production.

This refactor keeps backward compatibility for decrypting historical tokens that
were "double base64" encoded by the legacy implementation.
"""

from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class EncryptionServiceError(RuntimeError):
    """Raised when encryption/decryption cannot be performed safely."""


class EncryptionService:
    """Simple Fernet-based encryption service for sensitive string data."""

    def __init__(self, key: bytes):
        # Fernet validates key format on construction.
        try:
            self._cipher = Fernet(key)
        except Exception as e:  # pragma: no cover
            raise EncryptionServiceError(
                "Invalid Fernet key. ENCRYPTION_KEY must be a valid Fernet key generated via "
                "cryptography.fernet.Fernet.generate_key()."
            ) from e
        self.key = key

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new Fernet key."""
        return Fernet.generate_key()

    def encrypt(self, data: str) -> str:
        """Encrypt a string and return a URL-safe token."""
        if data is None:
            raise ValueError("Cannot encrypt None")
        if not isinstance(data, str):
            data = str(data)
        token = self._cipher.encrypt(data.encode("utf-8"))
        # Fernet tokens are already urlsafe base64 bytes.
        return token.decode("utf-8")

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt a token produced by :meth:`encrypt`.

        Backward compatibility:
        - Also supports legacy tokens that were additionally base64-encoded.
        """
        if encrypted_data is None:
            raise ValueError("Cannot decrypt None")

        if not isinstance(encrypted_data, str):
            encrypted_data = str(encrypted_data)

        # Fast path: token is a normal Fernet token string.
        try:
            clear = self._cipher.decrypt(encrypted_data.encode("utf-8"))
            return clear.decode("utf-8")
        except InvalidToken as exc:
            logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

        # Legacy path: token was base64(token_bytes).
        try:
            raw = base64.b64decode(encrypted_data.encode("utf-8"))
            clear = self._cipher.decrypt(raw)
            return clear.decode("utf-8")
        except Exception as e:
            logger.error(
                "decryption_failed",
                extra={"encrypted_data_preview": encrypted_data[:50], "error": str(e)},
            )
            raise EncryptionServiceError("Decryption failed") from e


def _load_key_from_env() -> bytes:
    env_key = os.getenv("ENCRYPTION_KEY")
    if not env_key:
        raise EncryptionServiceError(
            "ENCRYPTION_KEY is not set. Provide a stable Fernet key in production. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return env_key.encode("utf-8")


@lru_cache(maxsize=1)
def get_encryption_service() -> EncryptionService:
    """Return a singleton EncryptionService initialized from ENCRYPTION_KEY."""
    return EncryptionService(_load_key_from_env())
