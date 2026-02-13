
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

async def verify_system():
    print("Starting Phase 2 & 3 Verification...")
    
    # 1. Verify Security & Auth
    try:
        from resync.api.core import security
        from resync.api.routes.core import auth
        print("✓ Security and Auth modules imported successfully.")
        
        # Test password hashing (bcrypt)
        pwd = "test_password"
        hashed = security.get_password_hash(pwd)
        assert security.verify_password(pwd, hashed) is True
        print("✓ Password hashing (bcrypt) verified.")
        
        # Test JWT (jose)
        token = security.create_access_token("test_user")
        payload = security.decode_access_token(token)
        assert payload["sub"] == "test_user"
        print("✓ JWT creation and decoding (jose) verified.")
        
    except Exception as e:
        print(f"✗ Security/Auth verification failed: {e}")
        return False

    # 2. Verify Health System
    try:
        from resync.core.health import get_unified_health_service
        # We don't need a running loop here if we just check the service instantiation
        # (Though we are in an async function)
        svc = await get_unified_health_service()
        print("✓ UnifiedHealthService instantiated successfully.")
    except Exception as e:
        print(f"✗ Health system verification failed: {e}")
        return False

    # 3. Verify Circuit Breakers
    try:
        from resync.core.circuit_breaker_registry import get_circuit_breaker, CircuitBreakers
        cb = get_circuit_breaker(CircuitBreakers.REDIS)
        assert cb is not None
        print(f"✓ Circuit Breaker registry verified (Redis breaker: {cb.state.value}).")
    except Exception as e:
        print(f"✗ Circuit breaker verification failed: {e}")
        return False

    # 4. Verify Idempotency Middleware
    try:
        from resync.api.middleware.idempotency import IdempotencyMiddleware
        print("✓ IdempotencyMiddleware imported successfully.")
    except Exception as e:
        print(f"✗ Idempotency verification failed: {e}")
        return False

    # 5. Application Factory Snapshot
    try:
        from resync.app_factory import ApplicationFactory
        factory = ApplicationFactory()
        # We won't call create_application() fully to avoid side effects (DB/Redis connections),
        # but importing it and instantiating the factory is a good sanity check.
        print("✓ ApplicationFactory instantiated successfully.")
    except Exception as e:
        print(f"✗ ApplicationFactory verification failed: {e}")
        return False

    print("\nAll Core Foundation and Security Hardening verifications passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_system())
    sys.exit(0 if success else 1)
