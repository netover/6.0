from resync.core.database_security import DatabaseInputValidator
from resync.core.audit_db import _validate_audit_record

# Test cases
test_cases = [
    ("'; DROP TABLE users; --", "Dangerous pattern"),
    ("' OR '1'='1", "OR pattern"),
]

for query, desc in test_cases:
    try:
        DatabaseInputValidator.validate_string_input(query)
        print(f"{desc}: PASS (no exception)")
    except Exception as e:
        print(f"{desc}: FAIL - {e}")

# Test audit record
record = {
    "id": "test",
    "user_query": "'; DROP TABLE users; --",
    "agent_response": "response",
    "action": "TEST"
}
try:
    _validate_audit_record(record)
    print("Audit record: PASS (no exception)")
except Exception as e:
    print(f"Audit record: FAIL - {e}")
