"""
Optimized String Processing Utilities

This module provides high-performance string processing functions
that avoid O(n²) complexity and reduce memory allocations.
"""

import re
from re import Pattern
from typing import Any

# Pre-compiled regex patterns for reuse
COMMON_PATTERNS: dict[str, Pattern] = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
    "safe_chars": re.compile(r"^[a-zA-Z0-9_.-]+$"),
    "url_safe": re.compile(r"^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9._-]*)*$"),
    "api_endpoint": re.compile(r"^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/[a-zA-Z0-9._-]*)*$"),
}


class StringProcessor:
    """
    High-performance string processor with optimized algorithms.

    Features:
    - Pre-compiled regex patterns
    - Efficient string joining
    - Optimized character replacement
    - Memory-efficient operations
    """

    @staticmethod
    def replace_multiple(text: str, replacements: dict[str, str]) -> str:
        """
        Replace multiple substrings efficiently using single pass.

        Args:
            text: Input string
            replacements: Dictionary of replacements {old: new}

        Returns:
            String with all replacements applied
        """
        if not replacements:
            return text

        # Create regex pattern for all replacements
        pattern = re.compile("|".join(map(re.escape, replacements.keys())))

        # Single pass replacement
        def _replacer(match):
            return replacements[match.group(0)]

        return pattern.sub(_replacer, text)

    @staticmethod
    def join_efficient(parts: list[str], separator: str = " ") -> str:
        """
        Efficient string joining that handles edge cases.

        Args:
            parts: List of string parts
            separator: Separator string

        Returns:
            Joined string
        """
        # Filter out empty strings and None values
        filtered_parts = [part for part in parts if part]
        return separator.join(filtered_parts)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename using pre-compiled regex.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        basename = filename.split("/")[-1].split("\\")[-1]
        # Use pre-compiled pattern for alphanumeric check
        return re.sub(r"[^\w\-_.]", "", basename)

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email using pre-compiled regex.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        return bool(COMMON_PATTERNS["email"].match(email))

    @staticmethod
    def validate_alphanumeric(text: str) -> bool:
        """
        Validate alphanumeric string using pre-compiled regex.

        Args:
            text: Text to validate

        Returns:
            True if alphanumeric, False otherwise
        """
        return bool(COMMON_PATTERNS["alphanumeric"].match(text))

    @staticmethod
    def validate_safe_chars(text: str) -> bool:
        """
        Validate string contains only safe characters.

        Args:
            text: Text to validate

        Returns:
            True if safe, False otherwise
        """
        return bool(COMMON_PATTERNS["safe_chars"].match(text))

    @staticmethod
    def extract_field_names(text: str) -> list[str]:
        """
        Extract field names from error messages efficiently.

        Args:
            text: Error message text

        Returns:
            List of field names
        """
        # Use regex to find quoted field paths
        pattern = re.compile(r'"([^"]+)"')
        return pattern.findall(text)

    @staticmethod
    def format_error_field(field_path: list[str]) -> str:
        """
        Format field path efficiently.

        Args:
            field_path: List of field path components

        Returns:
            Formatted field path string
        """
        return ".".join(str(loc) for loc in field_path)

    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = "*") -> str:
        """
        Mask sensitive data efficiently.

        Args:
            data: Data to mask
            mask_char: Character to use for masking

        Returns:
            Masked data
        """
        if not data or len(data) <= 4:
            return data
        return data[:2] + mask_char * (len(data) - 4) + data[-2:]


class TextChunker:
    """
    Optimized text chunking for large documents.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk text efficiently using pre-calculated positions.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            start += self.step

        return chunks


class StringBuilder:
    """
    Memory-efficient string builder for multiple concatenations.
    """

    def __init__(self):
        self.parts = []

    def append(self, text: str) -> None:
        """Add text part."""
        self.parts.append(text)

    def extend(self, parts: list[str]) -> None:
        """Add multiple text parts."""
        self.parts.extend(parts)

    def build(self, separator: str = "") -> str:
        """Build final string."""
        return separator.join(self.parts)

    def clear(self) -> None:
        """Clear all parts."""
        self.parts.clear()

    def __len__(self) -> int:
        """Get current length in characters."""
        return sum(len(part) for part in self.parts)


def optimize_json_output(data: dict | list | Any) -> str:
    """
    Convert JSON data to readable string efficiently.

    Args:
        data: JSON data structure

    Returns:
        Readable string representation
    """
    import json

    if isinstance(data, dict):
        # Use list comprehension for better performance
        parts = [
            f"{key}: {json.dumps(value, indent=2) if isinstance(value, (dict, list)) else value}"
            for key, value in data.items()
        ]
        return "\n".join(parts)
    if isinstance(data, list):
        return json.dumps(data, indent=2)
    return str(data)


def compile_validation_patterns() -> None:
    """
    Pre-compile all validation patterns for reuse.
    Call this during application startup.
    """
    # Patterns are already compiled at module level
