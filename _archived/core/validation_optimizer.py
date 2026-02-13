"""
Optimized Validation Patterns

This module provides high-performance validation functions
that eliminate O(n²) complexity through pre-compilation and caching.
"""

import re
from functools import lru_cache
from typing import Any


class ValidationCache:
    """
    Cache for validation results to avoid repeated computations.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: dict[str, bool] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> bool | None:
        """Get cached validation result."""
        return self._cache.get(key)

    def set(self, key: str, result: bool) -> None:
        """Set validation result in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

        self._cache[key] = result
        self._access_order.append(key)


class OptimizedValidator:
    """
    High-performance validator with pre-compiled patterns and caching.
    """

    def __init__(self):
        self.cache = ValidationCache()
        self._compiled_patterns: dict[str, re.Pattern] = {}

        # Pre-compile additional patterns
        self._compiled_patterns.update(
            {
                "email_extended": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
                "username": re.compile(r"^[a-zA-Z0-9_.-]+$"),
                "api_key_name": re.compile(r"^[a-zA-Z0-9_-]+$"),
                "tool_name": re.compile(r"^[a-zA-Z0-9_-]+$"),
                "tag_name": re.compile(r"^[a-zA-Z0-9_-]+$"),
                "mime_type": re.compile(r"^[a-zA-Z0-9\-]+\/[a-zA-Z0-9\-\+]+(;.*)?$"),
                "metadata_key": re.compile(r"^[a-zA-Z0-9_.-]+$"),
                "config_key": re.compile(r"^[a-zA-Z0-9_.-]+$"),
                "json_web_key": re.compile(r"^[a-zA-Z0-9_.-]+$"),
            }
        )

    @lru_cache(maxsize=128)
    def validate_email(self, email: str) -> bool:
        """
        Validate email with caching.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        # Check cache first
        cache_key = f"email:{email}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Use pre-compiled pattern
        result = bool(self._compiled_patterns["email_extended"].match(email))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_username(self, username: str) -> bool:
        """
        Validate username efficiently.
        """
        cache_key = f"username:{username}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["username"].match(username))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_api_key_name(self, name: str) -> bool:
        """
        Validate API key name efficiently.
        """
        cache_key = f"api_key_name:{name}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["api_key_name"].match(name))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_tool_name(self, tool: str) -> bool:
        """
        Validate tool name efficiently.
        """
        cache_key = f"tool_name:{tool}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["tool_name"].match(tool))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_tag_name(self, tag: str) -> bool:
        """
        Validate tag name efficiently.
        """
        cache_key = f"tag_name:{tag}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["tag_name"].match(tag))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_mime_type(self, mime_type: str) -> bool:
        """
        Validate MIME type efficiently.
        """
        cache_key = f"mime_type:{mime_type}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["mime_type"].match(mime_type))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_metadata_key(self, key: str) -> bool:
        """
        Validate metadata key efficiently.
        """
        cache_key = f"metadata_key:{key}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["metadata_key"].match(key))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_config_key(self, key: str) -> bool:
        """
        Validate configuration key efficiently.
        """
        cache_key = f"config_key:{key}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["config_key"].match(key))
        self.cache.set(cache_key, result)
        return result

    @lru_cache(maxsize=128)
    def validate_json_web_key(self, key: str) -> bool:
        """
        Validate JSON web key efficiently.
        """
        cache_key = f"json_web_key:{key}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = bool(self._compiled_patterns["json_web_key"].match(key))
        self.cache.set(cache_key, result)
        return result

    def validate_field_names_batch(self, fields: list[str]) -> dict[str, bool]:
        """
        Validate multiple field names efficiently.

        Args:
            fields: List of field names to validate

        Returns:
            Dictionary mapping field names to validation results
        """
        results = {}
        for field in fields:
            results[field] = self.validate_metadata_key(field)
        return results

    def extract_field_paths_optimized(self, error_messages: list[str]) -> list[str]:
        """
        Extract field paths from error messages efficiently.

        Args:
            error_messages: List of error messages

        Returns:
            List of unique field paths
        """
        field_paths = set()
        pattern = re.compile(r'"([^"]+)"')

        for message in error_messages:
            matches = pattern.findall(message)
            field_paths.update(matches)

        return list(field_paths)


class DataStructureValidator:
    """
    Validator for complex data structures with optimized algorithms.
    """

    def __init__(self):
        self.validator = OptimizedValidator()

    def validate_dict_structure(
        self,
        data: dict[str, Any],
        required_keys: set[str] | None = None,
        key_validator: str | None = None,
    ) -> list[str]:
        """
        Validate dictionary structure efficiently.

        Args:
            data: Dictionary to validate
            required_keys: Set of required keys
            key_validator: Type of validation for keys

        Returns:
            List of validation errors
        """
        errors = []

        # Check required keys using set operations (O(1) instead of O(n))
        if required_keys:
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                errors.append(f"Missing required keys: {', '.join(missing_keys)}")

        # Validate key types efficiently
        if key_validator:
            invalid_keys = []
            for key in data:
                if (
                    key_validator == "metadata"
                    and not self.validator.validate_metadata_key(key)
                    or key_validator == "config"
                    and not self.validator.validate_config_key(key)
                    or key_validator == "json_web"
                    and not self.validator.validate_json_web_key(key)
                ):
                    invalid_keys.append(key)

            if invalid_keys:
                errors.append(f"Invalid keys: {', '.join(invalid_keys)}")

        return errors

    def validate_list_structure(
        self, data: list[Any], item_validator: str | None = None, max_length: int | None = None
    ) -> list[str]:
        """
        Validate list structure efficiently.

        Args:
            data: List to validate
            item_validator: Type of validation for items
            max_length: Maximum allowed length

        Returns:
            List of validation errors
        """
        errors = []

        # Check length
        if max_length and len(data) > max_length:
            errors.append(f"List too long: {len(data)} > {max_length}")

        # Validate items efficiently
        if item_validator:
            invalid_items = []
            for i, item in enumerate(data):
                if item_validator == "tool_name" and not self.validator.validate_tool_name(
                    str(item)
                ):
                    invalid_items.append(f"item[{i}]")

            if invalid_items:
                errors.append(f"Invalid items: {', '.join(invalid_items)}")

        return errors


# Global validator instance
_global_validator: OptimizedValidator | None = None


def get_global_validator() -> OptimizedValidator:
    """Get or create global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = OptimizedValidator()
    return _global_validator


def validate_email_cached(email: str) -> bool:
    """Convenience function for cached email validation."""
    return get_global_validator().validate_email(email)


def validate_username_cached(username: str) -> bool:
    """Convenience function for cached username validation."""
    return get_global_validator().validate_username(username)


def validate_tool_names_batch(tools: list[str]) -> dict[str, bool]:
    """Convenience function for batch tool name validation."""
    return get_global_validator().validate_field_names_batch(
        tools, lambda x: get_global_validator().validate_tool_name(x)
    )


def validate_tag_names_batch(tags: list[str]) -> dict[str, bool]:
    """Convenience function for batch tag name validation."""
    return get_global_validator().validate_field_names_batch(
        tags, lambda x: get_global_validator().validate_tag_name(x)
    )
