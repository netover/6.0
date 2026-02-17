from resync.core.audit_db import _validate_audit_record

# Test cases
cases = [
    ({"user_query": "test", "action": "TEST"}, "Memory ID is required"),
    ({"id": "test", "action": "TEST"}, "User query is required"),
]

for record, expected in cases:
    try:
        _validate_audit_record(record)
        print(f"FAIL: No exception for {record}")
    except Exception as e:
        print(f"OK: {type(e).__name__}: {e}")
