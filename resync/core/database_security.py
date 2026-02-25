# ruff: noqa: E501
"""
Centralized database security utilities for SQL injection prevention and input validation.

This module provides comprehensive security controls for all database operations:
- SQL injection prevention through input validation
- Query parameter sanitization
- Database connection security
- Audit logging for database operations
"""

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseSecurityError(Exception):
    """Security-related database operation error."""


class DatabaseInputValidator:
    """
    Centralized input validation for database operations.

    Provides comprehensive validation to prevent SQL injection and other attacks.
    """

    # Safe SQL patterns and whitelists
    SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    SAFE_TABLE_NAMES = {
        "audit_log",
        "log",
        "logs",
        "events",
        "audit_queue",
        "users",
        "configurations",
        "settings",
        "conversations",
        "content",
        "memory",
    }
    SAFE_COLUMN_NAMES = {
        "id",
        "user_query",
        "agent_response",
        "agent_id",
        "model_used",
        "timestamp",
        "created_at",
        "updated_at",
        "status",
        "feedback",
        "rating",
        "content",
        "metadata",
        "observations",
        "is_flagged",
        "is_approved",
        "processed",
        "flag_reason",
        "flag_confidence",
        "memory_id",
    }

    # Input length limits
    MAX_IDENTIFIER_LENGTH = 64
    MAX_STRING_INPUT_LENGTH = 10000
    MAX_QUERY_LENGTH = 1000

    @classmethod
    def validate_table_name(cls, table_name: str) -> str:
        """
        Validates table name against whitelist to prevent SQL injection.

        Args:
            table_name: Table name to validate

        Returns:
            Validated table name

        Raises:
            DatabaseSecurityError: If table name is invalid
        """
        if not table_name:
            raise DatabaseSecurityError("Table name cannot be empty")

        if len(table_name) > cls.MAX_IDENTIFIER_LENGTH:
            raise DatabaseSecurityError(
                f"Table name too long: {len(table_name)} > {cls.MAX_IDENTIFIER_LENGTH}"
            )

        if table_name not in cls.SAFE_TABLE_NAMES:
            raise DatabaseSecurityError(f"Table name not in whitelist: {table_name}")

        return table_name

    @classmethod
    def validate_column_name(cls, column_name: str) -> str:
        """
        Validates column name against whitelist to prevent SQL injection.

        Args:
            column_name: Column name to validate

        Returns:
            Validated column name

        Raises:
            DatabaseSecurityError: If column name is invalid
        """
        if not column_name:
            raise DatabaseSecurityError("Column name cannot be empty")

        if len(column_name) > cls.MAX_IDENTIFIER_LENGTH:
            raise DatabaseSecurityError(
                f"Column name too long: {len(column_name)} > {cls.MAX_IDENTIFIER_LENGTH}"
            )

        if column_name not in cls.SAFE_COLUMN_NAMES:
            raise DatabaseSecurityError(f"Column name not in whitelist: {column_name}")

        return column_name

    @classmethod
    def validate_sql_identifier(cls, identifier: str) -> str:
        """
        Validates generic SQL identifiers (table names, column names, etc.).

        Args:
            identifier: SQL identifier to validate

        Returns:
            Validated identifier

        Raises:
            DatabaseSecurityError: If identifier is invalid
        """
        if not identifier:
            raise DatabaseSecurityError("SQL identifier cannot be empty")

        if len(identifier) > cls.MAX_IDENTIFIER_LENGTH:
            raise DatabaseSecurityError(
                f"Identifier too long: {len(identifier)} > {cls.MAX_IDENTIFIER_LENGTH}"
            )

        if not cls.SAFE_IDENTIFIER_PATTERN.match(identifier):
            raise DatabaseSecurityError(f"Invalid SQL identifier: {identifier}")

        # Check for dangerous SQL keywords
        dangerous_keywords = {
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "CREATE",
            "ALTER",
            "EXEC",
            "EXECUTE",
            "UNION",
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER",
            "OUTER",
            "LEFT",
            "RIGHT",
            "GROUP",
            "ORDER",
            "HAVING",
            "LIMIT",
            "OFFSET",
        }

        if identifier.upper() in dangerous_keywords:
            raise DatabaseSecurityError(
                f"SQL keyword not allowed as identifier: {identifier}"
            )

        return identifier

    @classmethod
    def validate_string_input(
        cls, input_value: str, max_length: int | None = None
    ) -> str:
        """
        Validates string input for database operations.

        Args:
            input_value: String to validate
            max_length: Maximum allowed length (uses default if None)

        Returns:
            Validated string

        Raises:
            DatabaseSecurityError: If input is invalid
        """
        if input_value is None:
            raise DatabaseSecurityError("String input cannot be None")

        if not isinstance(input_value, str):
            raise DatabaseSecurityError(
                f"Input must be string, got {type(input_value)}"
            )

        max_len = max_length or cls.MAX_STRING_INPUT_LENGTH
        if len(input_value) > max_len:
            raise DatabaseSecurityError(
                f"String input too long: {len(input_value)} > {max_len}"
            )

        # Check for null bytes
        if "\x00" in input_value:
            raise DatabaseSecurityError("String input cannot contain null bytes")

        # Check for dangerous patterns (SQL injection indicators)
        # Note: Single quotes, double quotes, and semicolons are allowed
        # since parameterized queries handle escaping safely.
        dangerous_patterns = [
            r"--",  # SQL comment
            r"/\*",
            r"\*/",  # Multi-line comments
            r"xp_",  # Extended procedures
            r"sp_",  # Stored procedures
            r"(?i)\bunion\b\s+(?:all\s+)?\bselect\b",
            r"(?i)(?:\bor\b|\band\b)\s+(?:'\w+'|\d+)\s*=\s*(?:'\w+'|\d+)",
            r"(?i)(?:\bor\b|\band\b)\s+(?:'\w+'\s*=\s*'\w+'|\d+\s*=\s*\d+)",  # Fix: support quoted strings like '1'='1'
            r"(?i)(?:\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"(?i);\s*(?:drop|alter|create|truncate|exec|execute)\b",
            r"(?i)'\s*(?:or|and)\b.*'.*'=",  # Detect ' OR '1'='1 pattern
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                raise DatabaseSecurityError(
                    f"Dangerous pattern detected in input: {pattern}"
                )

        return input_value

    @classmethod
    def validate_numeric_input(
        cls,
        input_value: int | float,
        min_val: int | float | None = None,
        max_val: int | float | None = None,
    ) -> int | float:
        """
        Validates numeric input for database operations.

        Args:
            input_value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated numeric value

        Raises:
            DatabaseSecurityError: If input is invalid
        """
        if input_value is None:
            raise DatabaseSecurityError("Numeric input cannot be None")

        if not isinstance(input_value, (int, float)):
            raise DatabaseSecurityError(
                f"Input must be numeric, got {type(input_value)}"
            )

        if min_val is not None and input_value < min_val:
            raise DatabaseSecurityError(
                f"Value below minimum: {input_value} < {min_val}"
            )

        if max_val is not None and input_value > max_val:
            raise DatabaseSecurityError(
                f"Value above maximum: {input_value} > {max_val}"
            )

        return input_value

    @classmethod
    def validate_limit(cls, limit: int | str) -> int:
        """
        Validates LIMIT clause parameters to prevent injection.

        Args:
            limit: Limit value to validate

        Returns:
            Validated integer limit

        Raises:
            DatabaseSecurityError: If limit is invalid
        """
        try:
            int_limit = int(limit)
        except (ValueError, TypeError):
            raise DatabaseSecurityError(f"Invalid limit value: {limit}") from None

        if int_limit < 1:
            raise DatabaseSecurityError(f"Limit must be positive: {int_limit}")

        if int_limit > 10000:  # Reasonable upper bound
            raise DatabaseSecurityError(f"Limit too large: {int_limit}")

        return int_limit

    @classmethod
    def sanitize_query_string(cls, query: str) -> str:
        """
        Basic sanitization for query strings (for search operations).

        Args:
            query: Query string to sanitize

        Returns:
            Sanitized query string

        Raises:
            DatabaseSecurityError: If query contains dangerous content
        """
        if not query:
            raise DatabaseSecurityError("Query string cannot be empty")

        if len(query) > cls.MAX_QUERY_LENGTH:
            raise DatabaseSecurityError(
                f"Query too long: {len(query)} > {cls.MAX_QUERY_LENGTH}"
            )

        # Remove or escape dangerous characters
        sanitized = query.replace("'", "''")  # Escape single quotes
        sanitized = sanitized.replace('"', '""')  # Escape double quotes
        sanitized = sanitized.replace(";", "")  # Remove statement separators
        sanitized = sanitized.replace("--", "")  # Remove comments
        sanitized = re.sub(
            r"/\*.*?\*/", "", sanitized, flags=re.DOTALL
        )  # Remove block comments

        return sanitized.strip()


class DatabaseAuditor:
    """
    Audit logging for database operations.

    Provides comprehensive logging of database access for security monitoring.
    """

    @staticmethod
    def log_database_operation(
        operation: str,
        table: str,
        user_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Logs a database operation for audit purposes.

        Args:
            operation: Type of operation (SELECT, INSERT, UPDATE, DELETE, etc.)
            table: Table name being accessed
            user_id: User performing the operation (if available)
            details: Additional operation details
        """
        log_entry = {
            "operation": operation,
            "table": table,
            "user_id": user_id,
            "details": details or {},
            "timestamp": time.time(),
        }

        logger.info("database_operation_audited", extra=log_entry)

    @staticmethod
    def log_security_violation(
        violation_type: str, input_value: str, user_id: str | None = None
    ) -> None:
        """
        Logs a security violation for monitoring and alerting.

        Args:
            violation_type: Type of security violation
            input_value: The problematic input
            user_id: User who provided the input (if available)
        """
        log_entry = {
            "violation_type": violation_type,
            "input_value": input_value[:100] + "..."
            if len(input_value) > 100
            else input_value,
            "user_id": user_id,
            "timestamp": time.time(),
        }

        logger.warning("database_security_violation", extra=log_entry)


class SecureQueryBuilder:
    """
    Secure query builder with built-in injection prevention.

    Provides methods to build SQL queries safely with validation.
    """

    @staticmethod
    def build_select_query(
        table: str,
        columns: list[str] | None = None,
        where_clause: str | None = None,
        order_by: str | None = None,
        limit: int | str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Builds a secure SELECT query with validation.

        Args:
            table: Table name (validated against whitelist)
            columns: List of columns to select (validated against whitelist)
            where_clause: WHERE clause (must be parameterized)
            order_by: ORDER BY clause (validated)
            limit: LIMIT value (validated)

        Returns:
            Tuple of (query, parameters)

        Raises:
            DatabaseSecurityError: If any component is invalid
        """
        # Validate table name
        validated_table = DatabaseInputValidator.validate_table_name(table)

        # Validate columns
        if columns:
            validated_columns = []
            for col in columns:
                validated_columns.append(
                    DatabaseInputValidator.validate_column_name(col)
                )
            columns_str = ", ".join(validated_columns)
        else:
            columns_str = "*"

        # Build base query
        query = f"SELECT {columns_str} FROM {validated_table}"
        params = {}

        # Add WHERE clause if provided
        if where_clause:
            # Security: WHERE clause MUST use parameterized queries (:param_name).
            # Reject anything that contains subqueries, string literals, or
            # dangerous SQL keywords.  This is defence-in-depth on top of the
            # ORM layer — callers should ALWAYS pass params separately.
            _SAFE_WHERE = re.compile(r"^[\w.:=<>!%\s,\(\)]+$", re.ASCII)
            if not _SAFE_WHERE.match(where_clause):
                raise DatabaseSecurityError(
                    "WHERE clause contains disallowed characters. "
                    "Use parameterized queries (:param_name syntax) only."
                )

            dangerous_patterns = [
                "--",
                ";",
                "/*",
                "*/",
                "DROP",
                "DELETE",
                "INSERT",
                "UPDATE",
                "EXEC",
                "UNION",
                "INTO",
                "ALTER",
                "CREATE",
                "TRUNCATE",
                "GRANT",
                "REVOKE",
                "SLEEP",
                "BENCHMARK",
                "LOAD_FILE",
                "OUTFILE",
            ]
            where_upper = where_clause.upper()
            for pattern in dangerous_patterns:
                if pattern in where_upper:
                    raise DatabaseSecurityError(
                        f"Potentially unsafe WHERE clause detected: contains '{pattern}'"
                    )
            query += f" WHERE {where_clause}"

        # Add ORDER BY if provided
        if order_by:
            # Validate ORDER BY clause
            if DatabaseInputValidator.SAFE_IDENTIFIER_PATTERN.match(
                order_by.replace(" DESC", "").replace(" ASC", "")
            ):
                query += f" ORDER BY {order_by}"
            else:
                raise DatabaseSecurityError(f"Invalid ORDER BY clause: {order_by}")

        # Add LIMIT if provided
        if limit:
            validated_limit = DatabaseInputValidator.validate_limit(limit)
            query += " LIMIT ?"
            params["limit"] = validated_limit

        return query, params


# Convenience functions for common operations
def validate_database_inputs(
    table_name: str, limit: int | str | None = None, columns: list[str] | None = None
) -> dict[str, Any]:
    """
    Validates common database inputs and returns validated parameters.

    Args:
        table_name: Table name to validate
        limit: Limit value to validate
        columns: Columns to validate

    Returns:
        Dictionary of validated parameters

    Raises:
        DatabaseSecurityError: If any input is invalid
    """
    validated: dict[str, Any] = {
        "table": DatabaseInputValidator.validate_table_name(table_name)
    }

    if limit is not None:
        validated["limit"] = DatabaseInputValidator.validate_limit(limit)

    if columns:
        validated["columns"] = [
            DatabaseInputValidator.validate_column_name(col) for col in columns
        ]

    return validated


def log_database_access(
    operation: str,
    table: str,
    success: bool,
    user_id: str | None = None,
    error: str | None = None,
) -> None:
    """
    Logs database access for security monitoring.

    Args:
        operation: Database operation type
        table: Table accessed
        success: Whether the operation succeeded
        user_id: User performing operation
        error: Error message if operation failed
    """
    details: dict[str, bool | str] = {"success": success}
    if error:
        details["error"] = error

    if success:
        DatabaseAuditor.log_database_operation(operation, table, user_id, details)
    else:
        DatabaseAuditor.log_security_violation(
            "database_access_failed", f"{operation} on {table}", user_id
        )
