# Database Security Guide

This guide provides comprehensive database security practices for the Resync project.

## Overview

The Resync project implements multiple database technologies:
- **SQLite** - Primary database for audit logs and queue management
- **Redis** - Used for caching and stream processing  
- **Neo4j** - Knowledge graph operations with Cypher queries
- **Connection Pooling** - Advanced pooling with SQLAlchemy 2.x

## Security Measures Implemented

### 1. SQL Injection Prevention

#### Parameterized Queries
All database operations use parameterized queries to prevent SQL injection:

```python
# ✅ SAFE - Parameterized query
cursor.execute("SELECT * FROM audit_log WHERE status = ? LIMIT ?", (status, limit))

# ❌ UNSAFE - String interpolation  
cursor.execute(f"SELECT * FROM audit_log WHERE status = '{status}' LIMIT {limit}")
```

#### Input Validation
Comprehensive input validation is implemented in `resync/core/database_security.py`:

- **Table name whitelist** - Only allowed table names
- **Column name whitelist** - Only allowed column names  
- **String length limits** - Prevent buffer overflow attacks
- **Pattern detection** - Detect dangerous SQL patterns
- **Type validation** - Ensure correct data types

#### Security Middleware
SQL injection detection middleware (`resync/api/middleware/database_security_middleware.py`):

- **Pattern matching** - Detects 15+ injection patterns
- **Request analysis** - Examines query params, path params, headers, and body
- **Automatic blocking** - Blocks suspicious requests with 400 status
- **Audit logging** - Logs all security violations

### 2. Database Connection Security

#### Secure Connection Management
- **Connection pooling** - SQLAlchemy with proper connection management
- **Timeout handling** - Prevents connection exhaustion
- **Secret redaction** - Removes credentials from logs
- **Health checks** - Validates connection status

#### Environment Separation
- **Development vs Production** - Separate database configurations
- **Access controls** - Different users for different operations
- **Connection string security** - Encrypted credentials where possible

### 3. Audit and Monitoring

#### Comprehensive Logging
- **Operation logging** - All database operations are logged
- **Security violation logging** - All attacks are logged with details
- **Performance monitoring** - Track query performance and bottlenecks
- **Error tracking** - Detailed error information for debugging

## Security Checklist

### Development Phase

#### Code Review Checklist
- [ ] All SQL queries use parameterized queries?
- [ ] No string interpolation in SQL queries?
- [ ] Input validation implemented for all user inputs?
- [ ] Table/column names validated against whitelists?
- [ ] Length limits enforced on all inputs?
- [ ] Dangerous patterns detected and rejected?

#### Testing Checklist  
- [ ] SQL injection tests passing?
- [ ] Boundary condition tests passing?
- [ ] Performance tests passing?
- [ ] Integration tests passing?
- [ ] Fuzzing tests completed?
- [ ] Load tests passing?

### Deployment Phase

#### Configuration Checklist
- [ ] Database credentials stored securely?
- [ ] Connection strings use encrypted credentials?
- [ ] Principle of least privilege applied?
- [ ] Database access logs enabled?
- [ ] Security monitoring enabled?
- [ ] Backup and recovery procedures in place?

#### Runtime Checklist
- [ ] Security middleware enabled?
- [ ] Input validation active?
- [ ] Audit logging functioning?
- [ ] Error monitoring active?
- [ ] Performance monitoring active?
- [ ] Security alerts configured?

## Secure Coding Practices

### 1. Query Construction

#### ✅ Recommended Practices
```python
# Use parameterized queries
async with get_db_connection_sqlite() as conn:
    cursor = await conn.execute(
        "SELECT * FROM audit_queue WHERE status = ? ORDER BY created_at DESC LIMIT ?",
        (status, limit)
    )
    return [dict(row) for row in await cursor.fetchall()]

# Use secure query builders
query, params = SecureQueryBuilder.build_select_query(
    table='audit_log',
    columns=['id', 'user_query', 'status'],
    where_clause='created_at >= ?',
    limit=100
)
```

#### ❌ Forbidden Practices
```python
# NEVER do this - SQL injection vulnerability
query = f"SELECT * FROM audit_queue WHERE status = '{status}' LIMIT {limit}"
cursor.execute(query)

# NEVER do this - Table name injection
query = f"SELECT * FROM {table_name} WHERE id = ?"
cursor.execute(query, (id,))

# NEVER do this - Unsafe string formatting
query = "SELECT * FROM audit_log WHERE user_query LIKE '%s'" % user_input
cursor.execute(query)
```

### 2. Input Validation

#### ✅ Recommended Practices
```python
# Use centralized validation
from resync.core.database_security import validate_database_inputs

# Validate all inputs
validated = validate_database_inputs(
    table_name='audit_log',
    limit=user_limit,
    columns=['id', 'status', 'created_at']
)

# Use whitelist validation
table = DatabaseInputValidator.validate_table_name(user_input_table)
columns = [DatabaseInputValidator.validate_column_name(col) for col in user_columns]
```

#### ❌ Forbidden Practices
```python
# NEVER trust user input directly
table_name = request.query_params['table']  # User could inject anything
query = f"SELECT * FROM {table_name}"  # Direct injection risk

# NEVER skip validation
user_query = request.query_params['search']  # No validation
cursor.execute(f"SELECT * FROM logs WHERE message LIKE '%{user_query}%'")
```

### 3. Error Handling

#### ✅ Recommended Practices
```python
# Secure error handling
try:
    result = await database_operation(params)
    log_database_access('SELECT', 'audit_log', True, user_id='user123')
    return result
except DatabaseSecurityError as e:
    log_database_access('SELECT', 'audit_log', False, user_id='user123', error=str(e))
    raise HTTPException(
        status_code=400,
        detail="Invalid input provided"
    )
except Exception as e:
    log_database_access('SELECT', 'audit_log', False, user_id='user123', error=str(e))
    raise HTTPException(
        status_code=500,
        detail="Database operation failed"
    )
```

#### ❌ Forbidden Practices
```python
# NEVER expose database errors to users
try:
    result = cursor.execute(query)
except Exception as e:
    return {"error": str(e), "query": query}  # Exposes internal details

# NEVER fail silently
try:
    result = cursor.execute(query)
except:
    pass  # Security violations go undetected
```

## Attack Prevention

### Common Attack Vectors

#### 1. SQL Injection Patterns
- **Union-based**: `' UNION SELECT * FROM users --`
- **Boolean-based**: `' OR '1'='1`
- **Time-based**: `'; WAITFOR DELAY '00:00:05' --`
- **Error-based**: `' AND 1=CONVERT(int, (SELECT @@version)) --`
- **Stored procedures**: `'; EXEC xp_cmdshell('dir') --`

#### 2. Prevention Measures
- **Input validation** - Reject dangerous patterns
- **Parameterized queries** - Use prepared statements
- **Least privilege** - Minimal database permissions
- **Error handling** - Don't expose database details
- **Monitoring** - Detect and alert on attacks

### 3. Defense in Depth

#### Layer 1: Input Validation
```python
# First line of defense
try:
    table = DatabaseInputValidator.validate_table_name(user_table)
    limit = DatabaseInputValidator.validate_limit(user_limit)
except DatabaseSecurityError:
    raise HTTPException(status_code=400, detail="Invalid input")
```

#### Layer 2: Parameterized Queries
```python
# Second line of defense
query = "SELECT * FROM ? WHERE id = ? LIMIT ?"
params = (table, record_id, limit)
cursor.execute(query, params)
```

#### Layer 3: Security Middleware
```python
# Third line of defense
app.add_middleware(
    create_database_security_middleware(app, enabled=True)
)
```

#### Layer 4: Database Permissions
```python
# Fourth line of defense
# Use read-only user for select operations
# Use separate user for write operations
# Use admin user only for administrative tasks
```

## Monitoring and Alerting

### 1. Security Metrics

#### Key Indicators
- **Blocked requests** - Number of SQL injection attempts blocked
- **Block rate** - Percentage of requests blocked
- **Attack patterns** - Types of attacks detected
- **Response times** - Database operation performance
- **Error rates** - Database operation failure rates

#### 2. Alert Configuration
```python
# Configure alerts for security events
ALERT_THRESHOLDS = {
    'block_rate_percent': 5.0,  # Alert if >5% requests blocked
    'error_rate_percent': 2.0,  # Alert if >2% operations fail
    'response_time_ms': 5000,    # Alert if >5s response time
}
```

### 3. Log Analysis
```bash
# Monitor security logs
grep "sql_injection_blocked" /var/log/app.log | tail -f

# Monitor database access logs  
grep "database_operation" /var/log/app.log | grep "FAILED" | tail -f

# Check for attack patterns
grep "Dangerous pattern detected" /var/log/app.log | wc -l
```

## Testing Procedures

### 1. Automated Testing

#### Unit Tests
```bash
# Run security test suite
python -m pytest resync/tests/test_database_security.py -v

# Run with coverage
python -m pytest resync/tests/test_database_security.py --cov=resync.core.database_security
```

#### Integration Tests
```bash
# Test with real database
python -m pytest resync/tests/test_database_integration.py -v

# Test middleware functionality
python -m pytest resync/tests/test_middleware_security.py -v
```

### 2. Fuzzing Tests

#### SQLMap Integration
```bash
# Automated SQL injection testing
sqlmap -u "http://localhost:8000/api/v1/admin/audit" \
       --data="limit=1*" \
       --level=5 \
       --risk=3 \
       --tamper=between
```

#### Custom Fuzzing
```python
# Custom fuzzing script
import asyncio
import aiohttp

async def fuzz_endpoint(url, payload_generator):
    async with aiohttp.ClientSession() as session:
        for payload in payload_generator():
            try:
                async with session.post(url, json={'query': payload}) as resp:
                    if resp.status == 400:
                        print(f"Blocked: {payload}")
                    else:
                        print(f"Passed: {payload}")
            except Exception as e:
                print(f"Error: {payload} - {e}")
```

### 3. Load Testing

#### Performance Under Attack
```bash
# Load test with attack traffic
locust -f resync/tests/locustfile.py \
       -H "Content-Type: application/json" \
       --html \
       http://localhost:8000/api/v1/admin/audit
```

## Incident Response

### 1. Detection
#### Immediate Actions
1. **Block IP address** - Temporary block of attacking IP
2. **Increase logging** - Enhanced logging for attack source
3. **Alert team** - Immediate notification of security team
4. **Preserve evidence** - Save attack logs for analysis

### 2. Containment
#### Isolation Measures
1. **Rate limiting** - Reduce request rate from attacking IP
2. **Input filtering** - Enhanced validation for attack patterns
3. **Service protection** - Temporary read-only mode if needed
4. **Database isolation** - Separate database if compromise suspected

### 3. Recovery
#### Restoration Steps
1. **Analyze logs** - Review attack vectors and impact
2. **Patch vulnerabilities** - Apply security fixes
3. **Reset credentials** - Change database passwords if needed
4. **Monitor closely** - Enhanced monitoring after incident

## Compliance Requirements

### 1. OWASP Top 10 Compliance

#### A03:2021 - Injection
- ✅ Parameterized queries implemented
- ✅ Input validation implemented  
- ✅ Security middleware active
- ✅ Comprehensive testing completed

#### A01:2021 - Broken Access Control
- ✅ Database user privilege separation
- ✅ Role-based access controls
- ✅ Authentication required for sensitive operations

### 2. Industry Standards

#### PCI DSS Compliance
- ✅ Cardholder data protection measures
- ✅ Secure database connections
- ✅ Access logging and monitoring
- ✅ Regular security testing

#### GDPR Compliance
- ✅ Data protection by design
- ✅ Audit trail implementation
- ✅ Data minimization principles
- ✅ Right to be forgotten mechanisms

## Maintenance Procedures

### 1. Regular Tasks

#### Daily
- [ ] Review security logs for anomalies
- [ ] Monitor database performance metrics
- [ ] Check for blocked request patterns
- [ ] Verify backup completion

#### Weekly  
- [ ] Update security rule sets
- [ ] Review user access patterns
- [ ] Test security controls effectiveness
- [ ] Update security documentation

#### Monthly
- [ ] Comprehensive security assessment
- [ ] Penetration testing engagement
- [ ] Security training refresh
- [ ] Update incident response procedures

### 2. Update Procedures

#### Security Updates
```bash
# Update security dependencies
pip install --upgrade sqlalchemy aiosqlite redis neo4j

# Review and apply security patches
git fetch origin security
git cherry-pick <security-fix-commit>

# Test security changes
python -m pytest resync/tests/test_database_security.py -v
```

#### Configuration Updates
```python
# Update security configurations
# Rotate database credentials
# Update whitelist values
# Modify security middleware settings
# Adjust alert thresholds
```

## References

### Security Standards
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework/)
- [CWE Mitigation](https://cwe.mitre.org/)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)

### Tools and Resources
- [SQLMap](http://sqlmap.org/) - SQL injection testing
- [OWASP ZAP](https://www.zaproxy.org/) - Security scanning
- [Burp Suite](https://portswigger.net/burp) - Application testing
- [Metasploit](https://www.metasploit.com/) - Penetration testing

---

**Last Updated**: 2025-11-19  
**Version**: 1.0  
**Maintainer**: Resync Security Team  
**Review Schedule**: Monthly
