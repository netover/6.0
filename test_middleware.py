from resync.api.middleware.database_security_middleware import DatabaseSecurityMiddleware

middleware = DatabaseSecurityMiddleware(None, enabled=True)

# Test patterns
patterns = [
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "' UNION SELECT * FROM users --",
    "; EXEC xp_cmdshell('dir') --",
    "'; WAITFOR DELAY '00:00:05' --",
    "' AND 1=CONVERT(int, (SELECT @@version)) --",
]

for p in patterns:
    result = middleware._contains_sql_injection(p)
    print(f"Pattern: {p[:30]}... -> Detected: {result}")
