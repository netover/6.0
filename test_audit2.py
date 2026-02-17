from resync.core.audit_db import _validate_audit_record

# All test cases
invalid_cases = [
    # Missing required fields
    ({"user_query": "test", "action": "TEST"}, "Memory ID is required"),
    ({"id": "test", "action": "TEST"}, "User query is required"),
    ({"id": "test", "user_query": "test", "action": "TEST"}, "Agent response is required"),
    # Invalid data types
    (
        {"id": 123, "user_query": "test", "agent_response": "response", "action": "TEST"},
        "Memory ID must be string",
    ),
    (
        {"id": "test", "user_query": None, "agent_response": "response", "action": "TEST"},
        "User query is required",
    ),
    # Length validation
    (
        {"id": "x" * 256, "user_query": "test", "agent_response": "response", "action": "TEST"},
        "Memory ID too long",
    ),
    (
        {"id": "test", "user_query": "x" * 10001, "agent_response": "response", "action": "TEST"},
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

for record, expected in invalid_cases:
    try:
        _validate_audit_record(record)
        print(f"FAIL: No exception for case: {expected}")
    except Exception as e:
        print(f"Case '{expected}': {type(e).__name__}: {e}")
