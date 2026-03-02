"""
Encoding utilities for Windows portability and emoji-safe logging.

Provides robust detection of stream encoding capabilities and safe symbol rendering.
Uses only standard library, avoiding external dependencies.
"""

import logging
from typing import TextIO

logger = logging.getLogger(__name__)

def _get_encoding(stream: TextIO | None) -> str | None:
    """Safely retrieve encoding from a text stream, handling edge cases."""
    try:
        return getattr(stream, "encoding", None) if stream else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
        import sys as _sys
        from resync.core.exception_guard import maybe_reraise_programming_error
        _exc_type, _exc, _tb = _sys.exc_info()
        maybe_reraise_programming_error(_exc, _tb)

        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("exception_caught", exc_info=True)
        return None

def can_encode(
    text: str, stream: TextIO | None = None, encoding: str | None = None
) -> bool:
    """
    Check if text can be encoded with the given encoding/stream.

    Uses strict encoding to detect incompatibility, preventing UnicodeEncodeError.
    Defaults to stream's encoding or utf-8 as fallback.
    """
    enc = encoding or _get_encoding(stream) or "utf-8"
    try:
        text.encode(enc, errors="strict")
        return True
    except (LookupError, UnicodeEncodeError):
        return False

def symbol(ok: bool, stream: TextIO | None = None, encoding: str | None = None) -> str:
    """
    Return appropriate status symbol based on stream encoding support.

    Uses emoji if supported, falls back to ASCII otherwise.
    Decision based on can_encode() to ensure no UnicodeEncodeError.
    """
    emoji_good = "✅"
    emoji_bad = "❌"
    ascii_good = "[OK]"
    ascii_bad = "[ERR]"

    if can_encode(emoji_good + emoji_bad, stream=stream, encoding=encoding):
        return emoji_good if ok else emoji_bad
    return ascii_good if ok else ascii_bad
