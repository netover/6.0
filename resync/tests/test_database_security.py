"""
Comprehensive SQL Injection Security Test Suite

This module provides automated testing for SQL injection vulnerabilities:
- Parameterized query validation
- Input sanitization testing
- Security middleware verification
- Database operation testing

Tests cover common attack vectors and edge cases.
"""

import asyncio
import importlib.util

import pytest

if importlib.util.find_spec("sqlalchemy") is None:
    pytest.skip("sqlalchemy not installed in this environment", allow_module_level=True)
if importlib.util.find_spec("structlog") is None:
    pytest.skip(
        "Optional dependency 'structlog' not installed", allow_module_level=True
    )

from resync.api.middleware.database_security_middleware import (
    DatabaseSecurityMiddleware,
)
from resync.core.audit_db import _validate_audit_record
from resync.core.database_security import (
    DatabaseInputValidator,
    DatabaseSecurityError,
    SecureQueryBuilder,
    validate_database_inputs,
)


class TestDatabaseInputValidator:
    """Test cases for DatabaseInputValidator class."""

    def test_validate_table_name_valid_cases(self):
        """Test valid table names."""
        valid_tables = ["audit_log", "log", "events", "audit_queue"]

        for table in valid_tables:
            result = DatabaseInputValidator.validate_table_name(table)
            assert result == table, f"Valid table name rejected: {table}"

    def test_validate_table_name_invalid_cases(self):
        """Test invalid table names."""
        invalid_cases = [
            ("", "Table name cannot be empty"),
            ("invalid_table", "Table name not in whitelist"),
            ("audit_log; DROP TABLE users; --", "Table name not in whitelist"),
            ("a" * 65, "Table name too long"),
            ("audit' OR 1=1 --", "Table name not in whitelist"),
        ]

        for table, expected_error in invalid_cases:
            with pytest.raises(DatabaseSecurityError, match=expected_error):
                DatabaseInputValidator.validate_table_name(table)

    def test_validate_column_name_valid_cases(self):
        """Test valid column names."""
        valid_columns = ["id", "user_query", "agent_response", "timestamp"]

        for column in valid_columns:
            result = DatabaseInputValidator.validate_column_name(column)
            assert result == column, f"Valid column name rejected: {column}"

    def test_validate_column_name_invalid_cases(self):
        """Test invalid column names."""
        invalid_cases = [
            ("", "Column name cannot be empty"),
            ("invalid_column", "Column name not in whitelist"),
            ("id; DROP TABLE users; --", "Column name not in whitelist"),
            ("a" * 65, "Column name too long"),
        ]

        for column, expected_error in invalid_cases:
            with pytest.raises(DatabaseSecurityError, match=expected_error):
                DatabaseInputValidator.validate_column_name(column)

    def test_validate_string_input_valid_cases(self):
        """Test valid string inputs."""
        valid_inputs = [
            "This is a valid string",
            "User query with normal text",
            "Agent response 123",
            "Special chars: !@#$%^&*()",
        ]

        for input_text in valid_inputs:
            result = DatabaseInputValidator.validate_string_input(input_text)
            assert result == input_text, f"Valid string input rejected: {input_text}"

    def test_validate_string_input_invalid_cases(self):
        """Test invalid string inputs."""
        invalid_cases = [
            (None, "String input cannot be None"),
            (123, "Input must be string"),
            ("a" * 10001, "String input too long"),
            ("text with \x00 null byte", "String input cannot contain null bytes"),
            ("'; DROP TABLE users; --", "Dangerous pattern detected"),
            ("' OR '1'='1", "Dangerous pattern detected"),
        ]

        for input_text, expected_error in invalid_cases:
            with pytest.raises(DatabaseSecurityError, match=expected_error):
                DatabaseInputValidator.validate_string_input(input_text)

    def test_validate_numeric_input_valid_cases(self):
        """Test valid numeric inputs."""
        valid_inputs = [
            (123, None, None),
            (0, 0, None),
            (999999, None, 999999),
            (-100, -100, None),
            (3.14, 0, 10),
        ]

        for value, min_val, max_val in valid_inputs:
            result = DatabaseInputValidator.validate_numeric_input(
                value, min_val, max_val
            )
            assert result == value, f"Valid numeric input rejected: {value}"

    def test_validate_numeric_input_invalid_cases(self):
        """Test invalid numeric inputs."""
        invalid_cases = [
            (None, "Numeric input cannot be None"),
            ("not_a_number", "Input must be numeric"),
            (50, 100, "Value below minimum"),
            (200, None, 150, "Value above maximum"),
        ]

        args_list = []
        for case in invalid_cases:
            args = [case[0]]
            if len(case) > 1:
                args.append(case[1])
            if len(case) > 2:
                args.append(case[2])
            args_list.append((args, case[-1]))  # Last element is expected error

        for args, expected_error in args_list:
            with pytest.raises(DatabaseSecurityError, match=expected_error):
                DatabaseInputValidator.validate_numeric_input(*args)

    def test_validate_limit_valid_cases(self):
        """Test valid limit values."""
        valid_limits = [1, 10, 100, 1000, 9999]

        for limit in valid_limits:
            result = DatabaseInputValidator.validate_limit(limit)
            assert result == int(limit), f"Valid limit rejected: {limit}"

    def test_validate_limit_invalid_cases(self):
        """Test invalid limit values."""
        invalid_cases = [
            (0, "Limit must be positive"),
            (-1, "Limit must be positive"),
            (10001, "Limit too large"),
            ("not_a_number", "Invalid limit value"),
            (None, "Invalid limit value"),
            ("5; DROP TABLE users; --", "Invalid limit value"),
        ]

        for limit, expected_error in invalid_cases:
            with pytest.raises(DatabaseSecurityError, match=expected_error):
                DatabaseInputValidator.validate_limit(limit)

    def test_sanitize_query_string(self):
        """Test query string sanitization."""
        test_cases = [
            ("normal query", "normal query"),
            ("query with 'quotes'", "query with ''quotes''"),
            ('query with "quotes"', 'query with ""quotes""'),
            ("query; DROP TABLE users; --", "query DROP TABLE users"),
            ("query with /* comment */ text", "query with  text"),
        ]

        for input_query, expected_output in test_cases:
            result = DatabaseInputValidator.sanitize_query_string(input_query)
            assert result == expected_output, (
                f"Query sanitization failed: {input_query} -> {result}"
            )


class TestSecureQueryBuilder:
    """Test cases for SecureQueryBuilder class."""

    def test_build_select_query_basic(self):
        """Test basic SELECT query building."""
        query, params = SecureQueryBuilder.build_select_query(
            table="audit_log", columns=["id", "user_query"], limit=10
        )

        expected_query = "SELECT id, user_query FROM audit_log LIMIT ?"
        assert query == expected_query
        assert params == {"limit": 10}

    def test_build_select_query_with_where(self):
        """Test SELECT query with WHERE clause."""
        query, params = SecureQueryBuilder.build_select_query(
            table="audit_log", where_clause="status = :status", limit=50
        )

        expected_query = "SELECT * FROM audit_log WHERE status = :status LIMIT ?"
        assert query == expected_query
        assert params == {"limit": 50}

    def test_build_select_query_with_order(self):
        """Test SELECT query with ORDER BY clause."""
        query, params = SecureQueryBuilder.build_select_query(
            table="audit_log", order_by="created_at DESC", limit=25
        )

        expected_query = "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?"
        assert query == expected_query
        assert params == {"limit": 25}

    def test_build_select_query_invalid_table(self):
        """Test SELECT query with invalid table name."""
        with pytest.raises(DatabaseSecurityError, match="Table name not in whitelist"):
            SecureQueryBuilder.build_select_query(table="invalid_table", limit=10)

    def test_build_select_query_invalid_column(self):
        """Test SELECT query with invalid column name."""
        with pytest.raises(DatabaseSecurityError, match="Column name not in whitelist"):
            SecureQueryBuilder.build_select_query(
                table="audit_log", columns=["id", "invalid_column"], limit=10
            )


class TestSQLInjectionMiddleware:
    """Test cases for SQL injection middleware."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request for testing."""

        class MockRequest:
            def __init__(
                self, query_params=None, path_params=None, headers=None, method="GET"
            ):
                self.query_params = query_params or {}
                self.path_params = path_params or {}
                self.headers = headers or {}
                self.method = method
                self.url = MockUrl("/test")
                self.client = MockClient()
                self.state = {}

        class MockUrl:
            def __init__(self, path):
                self.path = path

        class MockClient:
            def __init__(self):
                self.host = "test-client"

        return MockRequest

    @pytest.fixture
    def mock_app(self):
        """Create mock app for testing."""

        async def app(scope, receive, send):
            pass

        return app

    def test_sql_injection_detection_patterns(self, mock_request):
        """Test SQL injection pattern detection."""
        injection_patterns = [
            ("'; DROP TABLE users; --", True),
            ("' OR '1'='1", True),
            ("' UNION SELECT * FROM users --", True),
            ("; EXEC xp_cmdshell('dir') --", True),
            ("'; WAITFOR DELAY '00:00:05' --", True),
            ("' AND 1=CONVERT(int, (SELECT @@version)) --", True),
        ]

        middleware = DatabaseSecurityMiddleware(None, enabled=True)

        for injection_attempt, should_detect in injection_patterns:
            request = mock_request(query_params={"id": injection_attempt})

            if should_detect:
                with pytest.raises(Exception):  # HTTPException
                    asyncio.run(middleware._analyze_request_for_sql_injection(request))
            else:
                # Should not raise exception
                asyncio.run(middleware._analyze_request_for_sql_injection(request))

    def test_safe_requests_pass_through(self, mock_request):
        """Test that safe requests pass through middleware."""
        safe_inputs = [
            "normal_user_query",
            "agent_response_123",
            "search term with spaces",
            "valid-id-123",
        ]

        middleware = DatabaseSecurityMiddleware(None, enabled=True)

        for safe_input in safe_inputs:
            request = mock_request(query_params={"query": safe_input})

            # Should not raise exception
            try:
                asyncio.run(middleware._analyze_request_for_sql_injection(request))
            except Exception as e:
                pytest.fail(f"Safe input was blocked: {safe_input} - {e}")

    def test_request_data_extraction(self, mock_request):
        """Test request data extraction for analysis."""
        request = mock_request(
            query_params={"id": "123"},
            path_params={"user_id": "456"},
            headers={"user-agent": "Test Browser"},
            method="POST",
        )

        middleware = DatabaseSecurityMiddleware(None, enabled=True)
        data = asyncio.run(middleware._extract_request_data(request))

        assert "query.id" in data
        assert "path.user_id" in data
        assert "header.user-agent" in data
        assert data["query.id"] == "123"
        assert data["path.user_id"] == "456"
        assert data["header.user-agent"] == "Test Browser"

    def test_security_stats(self):
        """Test security statistics tracking."""
        middleware = DatabaseSecurityMiddleware(None, enabled=True)

        # Initial stats
        stats = middleware.get_security_stats()
        assert stats["total_requests"] == 0
        assert stats["blocked_requests"] == 0
        assert stats["block_rate_percent"] == 0
        assert stats["middleware_enabled"] is True
        assert stats["patterns_monitored"] > 0


class TestAuditRecordValidation:
    """Test cases for audit record validation."""

    def test_validate_audit_record_valid_cases(self):
        """Test valid audit record inputs."""
        valid_records = [
            {
                "id": "test_id_123",
                "user_query": "What is the weather today?",
                "agent_response": "The weather is sunny with a high of 75°F.",
                "ia_audit_reason": None,
                "ia_audit_confidence": None,
                "action": "QUERY",
            },
            {
                "id": "test_id_456",
                "user_query": "How do I reset my password?",
                "agent_response": "You can reset your password by clicking the forgot password link.",
                "ia_audit_reason": "Suspicious query pattern",
                "ia_audit_confidence": 0.85,
                "action": "RESET_PASSWORD",
            },
        ]

        for record in valid_records:
            result = _validate_audit_record(record)
            assert result == record
            assert result["user_query"] == record["user_query"]
            assert result["agent_response"] == record["agent_response"]

    def test_validate_audit_record_invalid_cases(self):
        """Test invalid audit record inputs."""
        invalid_cases = [
            # Missing required fields
            ({"user_query": "test", "action": "TEST"}, "Memory ID is required"),
            ({"id": "test", "action": "TEST"}, "User query is required"),
            (
                {"id": "test", "user_query": "test", "action": "TEST"},
                "Agent response is required",
            ),
            # Invalid data types
            (
                {
                    "id": 123,
                    "user_query": "test",
                    "agent_response": "response",
                    "action": "TEST",
                },
                "Memory ID must be string",
            ),
            (
                {
                    "id": "test",
                    "user_query": None,
                    "agent_response": "response",
                    "action": "TEST",
                },
                "User query is required",
            ),
            # Length validation
            (
                {
                    "id": "x" * 256,
                    "user_query": "test",
                    "agent_response": "response",
                    "action": "TEST",
                },
                "Memory ID too long",
            ),
            (
                {
                    "id": "test",
                    "user_query": "x" * 10001,
                    "agent_response": "response",
                    "action": "TEST",
                },
                "User query too long",
            ),
            # Dangerous content
            (
                {
                    "id": "test",
                    "user_query": "'; DROP TABLE users; --",
                    "agent_response": "response",
                    "action": "TEST",
                },
                "Dangerous pattern detected",
            ),
        ]

        for record, expected_error in invalid_cases:
            with pytest.raises(
                (ValueError, TypeError, DatabaseSecurityError), match=expected_error
            ):
                _validate_audit_record(record)


class TestDatabaseSecurityIntegration:
    """Integration tests for database security components."""

    def test_validate_database_inputs_convenience_function(self):
        """Test the convenience validation function."""
        # Valid inputs
        result = validate_database_inputs(
            "audit_log", limit=50, columns=["id", "status"]
        )
        assert result["table"] == "audit_log"
        assert result["limit"] == 50
        assert "id" in result["columns"]
        assert "status" in result["columns"]

        # Invalid table
        with pytest.raises(DatabaseSecurityError):
            validate_database_inputs("invalid_table")

        # Invalid limit
        with pytest.raises(DatabaseSecurityError):
            validate_database_inputs("audit_log", limit=0)

    def test_middleware_factory_functions(self):
        """Test middleware factory functions."""
        from resync.api.middleware.database_security_middleware import (
            create_database_connection_security_middleware,
            create_database_security_middleware,
        )

        # Test security middleware factory
        def app(scope, receive, send):
            return None

        security_middleware = create_database_security_middleware(app, enabled=True)
        assert security_middleware.enabled is True

        # Test connection security middleware factory
        connection_middleware = create_database_connection_security_middleware(
            app, enabled=False
        )
        assert connection_middleware.enabled is False


class TestSQLInjectionAttackVectors:
    """Test comprehensive SQL injection attack vectors."""

    def test_time_based_attacks(self):
        """Test time-based SQL injection attacks."""
        time_attacks = [
            "'; WAITFOR DELAY '00:00:05' --",
            "'; SELECT pg_sleep(5) --",
            "'; SELECT SLEEP(5) --",
            "' AND 1=1 AND SLEEP(5) --",
        ]

        for attack in time_attacks:
            with pytest.raises(DatabaseSecurityError):
                DatabaseInputValidator.validate_string_input(attack)

    def test_boolean_based_attacks(self):
        """Test boolean-based SQL injection attacks."""
        boolean_attacks = [
            "' OR '1'='1",
            "' OR 'x'='x",
            "') OR ('1'='1' AND ''='",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
        ]

        for attack in boolean_attacks:
            with pytest.raises(DatabaseSecurityError):
                DatabaseInputValidator.validate_string_input(attack)

    def test_union_based_attacks(self):
        """Test UNION-based SQL injection attacks."""
        union_attacks = [
            "' UNION SELECT username, password FROM users --",
            "' UNION SELECT NULL, table_name FROM information_schema.tables --",
            "') UNION SELECT id, name FROM products --",
        ]

        for attack in union_attacks:
            with pytest.raises(DatabaseSecurityError):
                DatabaseInputValidator.validate_string_input(attack)

    def test_error_based_attacks(self):
        """Test error-based SQL injection attacks."""
        error_attacks = [
            "' AND 1=CONVERT(int, (SELECT @@version)) --",
            "' AND 1=CONVERT(int, (SELECT @@servername)) --",
            "') AND (SELECT * FROM (SELECT COUNT(*),CONCAT(username,':',password) FROM users) WHERE username LIKE 'admin' --",
        ]

        for attack in error_attacks:
            with pytest.raises(DatabaseSecurityError):
                DatabaseInputValidator.validate_string_input(attack)

    def test_stored_procedure_attacks(self):
        """Test stored procedure SQL injection attacks."""
        sp_attacks = [
            "'; EXEC xp_cmdshell('dir') --",
            "'; EXEC sp_configure 'show advanced options', 1 --",
            "'; CALL sa_utility --",
        ]

        for attack in sp_attacks:
            with pytest.raises(DatabaseSecurityError):
                DatabaseInputValidator.validate_string_input(attack)


# Performance and load testing
class TestDatabaseSecurityPerformance:
    """Performance tests for database security components."""

    def test_validation_performance(self):
        """Test that validation doesn't impact performance significantly."""
        import time

        # Test large number of validations
        start_time = time.time()

        for i in range(10000):
            DatabaseInputValidator.validate_table_name("audit_log")
            DatabaseInputValidator.validate_string_input(f"valid_string_{i}")
            DatabaseInputValidator.validate_limit(i % 100 + 1)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 30,000 validations in under 1 second
        assert duration < 1.0, f"Validation too slow: {duration}s for 30,000 operations"

    def test_middleware_performance(self):
        """Test that middleware doesn't impact performance significantly."""
        import time

        middleware = DatabaseSecurityMiddleware(None, enabled=True)

        # Test large number of requests
        start_time = time.time()

        for i in range(1000):
            request_data = {
                "query.id": f"valid_query_{i}",
                "search.term": f"normal_search_term_{i}",
            }

            # Simulate the extraction and analysis (without actually processing)
            data = {}
            for key, value in request_data.items():
                data[key] = value

            # Check for injection (should be False for all)
            contains_injection = any(
                pattern.search(str(value)) for pattern in middleware._compiled_patterns
            )
            assert not contains_injection, (
                f"Valid input flagged as injection: {request_data}"
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should process 1000 requests in under 1 second
        assert duration < 1.0, f"Middleware too slow: {duration}s for 1000 requests"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__])
