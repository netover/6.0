
import asyncio
import sys
import os
from pydantic import SecretStr

# Add current directory to path
sys.path.append(os.getcwd())

async def verify_app_factory_hardening():
    print("Starting Phase 5: App Factory Hardening Verification...")
    
    from resync.app_factory import ApplicationFactory
    from resync.settings import settings
    
    factory = ApplicationFactory()
    
    # 1. Test Production Validation (Failing closed)
    print("\n1. Testing Production failing closed with insecure settings...")
    
    # Mock production environment
    os.environ["APP_ENVIRONMENT"] = "production"
    # Using alias "SECRET_KEY" directly as defined in settings.py
    os.environ["SECRET_KEY"] = "CHANGE_ME" 
    os.environ["ADMIN_PASSWORD"] = "password"
    
    try:
        # We call the internal validation method directly
        # This will trigger Pydantic validators on the singleton 'settings' if it's re-evaluated,
        # but factory._validate_critical_settings() also does manual checks.
        factory._validate_critical_settings()
        print("✗ Validation should have failed for insecure production settings.")
        return False
    except (ValueError, Exception) as e:
        error_str = str(e)
        print(f"✓ Correctly blocked insecure production settings: {error_str[:200]}...")
        
        # Check for at least one of the expected security violations
        security_keywords = ["SECRET_KEY", "ADMIN_PASSWORD", "CORS_ALLOW_ORIGINS", "localhost"]
        found_keywords = [k for k in security_keywords if k in error_str]
        
        if found_keywords:
            print(f"✓ Caught security violations: {', '.join(found_keywords)}")
        else:
            print(f"✗ Unexpected error message: {error_str}")
            return False

    # 2. Test LLM Key validation
    print("\n2. Testing LLM Key validation in production...")
    os.environ["APP_SECRET_KEY"] = "a" * 32 # valid length
    os.environ["ADMIN_PASSWORD"] = "secure12345" # valid
    os.environ["LLM_API_KEY"] = "dummy_key_for_development"
    
    try:
        factory._validate_critical_settings()
        print("✗ Validation should have failed for dummy LLM key in production.")
        return False
    except ValueError as e:
        if "Invalid LLM API key" in str(e):
            print("✓ Correctly blocked dummy LLM key in production.")
        else:
            print(f"✗ Unexpected error message for LLM key: {e}")
            return False

    # 3. Test Graceful Shutdown Timeout (Sanity check)
    print("\n3. Testing Graceful Shutdown timeout logic...")
    # This is harder to test without a full app, but we can verify the method exists
    assert hasattr(ApplicationFactory, "_shutdown_services")
    print("✓ _shutdown_services method verified.")

    print("\nPhase 5: Production Hardening verifications passed!")
    return True

if __name__ == "__main__":
    # Clean up env before starting
    for key in ["APP_ENVIRONMENT", "APP_SECRET_KEY", "ADMIN_PASSWORD", "LLM_API_KEY"]:
        if key in os.environ: del os.environ[key]
        
    success = asyncio.run(verify_app_factory_hardening())
    sys.exit(0 if success else 1)
