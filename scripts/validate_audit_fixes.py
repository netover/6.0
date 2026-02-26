#!/usr/bin/env python3
"""Quick validation script for settings audit fixes.

Runs smoke tests to verify critical fixes are working.
No external dependencies required (only stdlib + resync).

Usage:
    python scripts/validate_audit_fixes.py

Exit codes:
    0: All checks passed
    1: At least one check failed
"""

import os
import secrets
import sys
import threading
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resync.settings import Settings, clear_settings_cache, get_settings, settings
from resync.settings_types import Environment
from pydantic import SecretStr, ValidationError


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_header(msg: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")


def print_test(name: str, passed: bool, details: str = "") -> bool:
    """Print test result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} | {name}")
    if details:
        print(f"         └─ {details}")
    return passed


def validate_p0_07_no_auto_secret_key() -> bool:
    """P0-07: SECRET_KEY not auto-generated."""
    print_header("P0-07: SECRET_KEY Auto-Generation Removed")
    
    # Test 1: Development allows None
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "development"
    clear_settings_cache()
    
    try:
        s = get_settings()
        test1 = print_test(
            "Development allows None SECRET_KEY",
            s.secret_key is None,
            f"secret_key={s.secret_key}"
        )
    except Exception as e:
        test1 = print_test("Development allows None SECRET_KEY", False, str(e))
    
    # Test 2: Production requires SECRET_KEY
    os.environ["APP_ENVIRONMENT"] = "production"
    clear_settings_cache()
    
    try:
        Settings()
        test2 = print_test("Production rejects None SECRET_KEY", False, "Should have raised ValidationError")
    except ValidationError:
        test2 = print_test("Production rejects None SECRET_KEY", True, "ValidationError raised as expected")
    except Exception as e:
        test2 = print_test("Production rejects None SECRET_KEY", False, str(e))
    
    return test1 and test2


def validate_p0_08_secretstr_masking() -> bool:
    """P0-08: SecretStr masked in repr."""
    print_header("P0-08: SecretStr Masking in repr()")
    
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "development"
    os.environ["SECRET_KEY"] = "my_super_secret_key_12345"
    os.environ["ADMIN_PASSWORD"] = "TestPassword123!"
    clear_settings_cache()
    
    try:
        s = get_settings()
        repr_output = repr(s)
        
        test1 = print_test(
            "SECRET_KEY not in repr",
            "my_super_secret_key" not in repr_output,
            f"Found in repr: {'my_super_secret_key' in repr_output}"
        )
        
        test2 = print_test(
            "ADMIN_PASSWORD not in repr",
            "TestPassword123" not in repr_output,
            f"Found in repr: {'TestPassword123' in repr_output}"
        )
        
        test3 = print_test(
            "Masked representation present",
            "**********" in repr_output or "SecretStr" in repr_output,
            "Found masking"
        )
        
        return test1 and test2 and test3
    except Exception as e:
        print_test("SecretStr masking", False, str(e))
        return False


def validate_p0_09_atomic_directory() -> bool:
    """P0-09: Atomic directory operations."""
    print_header("P0-09: TOCTOU Fix in Directory Validation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c" / "uploads"
        
        os.environ.clear()
        os.environ["APP_ENVIRONMENT"] = "development"
        os.environ["UPLOAD_DIR"] = str(nested)
        clear_settings_cache()
        
        try:
            s = Settings()
            
            test1 = print_test(
                "Nested directory created",
                s.upload_dir.exists() and s.upload_dir.is_dir(),
                f"Path: {s.upload_dir}"
            )
            
            # Test write permission
            test_file = s.upload_dir / ".test"
            test_file.touch()
            test2 = print_test(
                "Directory writable",
                test_file.exists(),
                "Test file created successfully"
            )
            test_file.unlink()
            
            return test1 and test2
        except Exception as e:
            print_test("Atomic directory creation", False, str(e))
            return False


def validate_p1_07_thread_safe_singleton() -> bool:
    """P1-07: Thread-safe singleton."""
    print_header("P1-07: Thread-Safe Settings Singleton")
    
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "development"
    clear_settings_cache()
    
    results = []
    
    def get_id():
        s = get_settings()
        results.append(id(s))
    
    try:
        threads = [threading.Thread(target=get_id) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        unique_ids = len(set(results))
        test1 = print_test(
            "Single instance across threads",
            unique_ids == 1,
            f"Unique instances: {unique_ids} (expected: 1)"
        )
        
        return test1
    except Exception as e:
        print_test("Thread safety", False, str(e))
        return False


def validate_p1_08_11_localhost_blocked() -> bool:
    """P1-08/P1-11: Localhost blocked in production."""
    print_header("P1-08/P1-11: Localhost Blocked in Production CORS")
    
    # Test 1: Localhost rejected in production
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "production"
    os.environ["SECRET_KEY"] = secrets.token_urlsafe(32)
    os.environ["ADMIN_PASSWORD"] = "SecureP@ss123!"
    os.environ["APP_CORS_ALLOWED_ORIGINS"] = "http://localhost:3000"
    clear_settings_cache()
    
    try:
        Settings()
        test1 = print_test("Production rejects localhost", False, "Should have raised ValidationError")
    except ValidationError as e:
        error_msg = str(e).lower()
        test1 = print_test(
            "Production rejects localhost",
            "localhost" in error_msg,
            "ValidationError raised with 'localhost'"
        )
    except Exception as e:
        test1 = print_test("Production rejects localhost", False, str(e))
    
    # Test 2: Localhost allowed in development
    os.environ["APP_ENVIRONMENT"] = "development"
    clear_settings_cache()
    
    try:
        s = Settings()
        test2 = print_test(
            "Development allows localhost",
            any("localhost" in origin.lower() for origin in s.cors_allowed_origins),
            f"CORS origins: {s.cors_allowed_origins}"
        )
    except Exception as e:
        test2 = print_test("Development allows localhost", False, str(e))
    
    # Test 3: Wildcard rejected in production
    os.environ["APP_ENVIRONMENT"] = "production"
    os.environ["APP_CORS_ALLOWED_ORIGINS"] = "*"
    clear_settings_cache()
    
    try:
        Settings()
        test3 = print_test("Production rejects wildcard", False, "Should have raised ValidationError")
    except ValidationError:
        test3 = print_test("Production rejects wildcard", True, "ValidationError raised")
    except Exception as e:
        test3 = print_test("Production rejects wildcard", False, str(e))
    
    return test1 and test2 and test3


def validate_p1_09_immutable_proxy() -> bool:
    """P1-09: Settings proxy is immutable."""
    print_header("P1-09: Immutable Settings Proxy")
    
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "development"
    clear_settings_cache()
    
    try:
        # Read should work
        _ = settings.project_name
        test1 = print_test("Read access works", True, "settings.project_name accessible")
    except Exception as e:
        test1 = print_test("Read access works", False, str(e))
    
    try:
        # Write should fail
        settings.project_name = "NewName"
        test2 = print_test("Write access blocked", False, "Should have raised AttributeError")
    except AttributeError:
        test2 = print_test("Write access blocked", True, "AttributeError raised")
    except Exception as e:
        test2 = print_test("Write access blocked", False, str(e))
    
    return test1 and test2


def validate_p1_10_env_comparison() -> bool:
    """P1-10: Environment enum comparison."""
    print_header("P1-10: Environment Enum Comparison Fix")
    
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "production"
    os.environ["SECRET_KEY"] = secrets.token_urlsafe(32)
    os.environ["ADMIN_PASSWORD"] = "SecureP@ss123!"
    os.environ["APP_CORS_ALLOWED_ORIGINS"] = "https://example.com"
    clear_settings_cache()
    
    try:
        s = Settings()
        
        test1 = print_test(
            "Environment is enum",
            isinstance(s.environment, Environment),
            f"Type: {type(s.environment)}"
        )
        
        test2 = print_test(
            "Environment equals PRODUCTION",
            s.environment == Environment.PRODUCTION,
            f"Value: {s.environment}"
        )
        
        # Verify validators actually ran (SECRET_KEY required)
        test3 = print_test(
            "Production validators enforced",
            s.secret_key is not None and len(s.secret_key.get_secret_value()) >= 32,
            f"SECRET_KEY length: {len(s.secret_key.get_secret_value())}"
        )
        
        return test1 and test2 and test3
    except Exception as e:
        print_test("Environment comparison", False, str(e))
        return False


def validate_p2_07_cached_properties() -> bool:
    """P2-07: Cached properties performance."""
    print_header("P2-07: Cached Properties Performance")
    
    os.environ.clear()
    os.environ["APP_ENVIRONMENT"] = "development"
    clear_settings_cache()
    
    try:
        s = get_settings()
        
        # CACHE_HIERARCHY should return same object
        obj1 = s.CACHE_HIERARCHY
        obj2 = s.CACHE_HIERARCHY
        test1 = print_test(
            "CACHE_HIERARCHY cached",
            obj1 is obj2,
            f"Same object: {obj1 is obj2}"
        )
        
        # AGENT_CONFIG_PATH should return same object
        path1 = s.AGENT_CONFIG_PATH
        path2 = s.AGENT_CONFIG_PATH
        test2 = print_test(
            "AGENT_CONFIG_PATH cached",
            path1 is path2,
            f"Same object: {path1 is path2}"
        )
        
        return test1 and test2
    except Exception as e:
        print_test("Cached properties", False, str(e))
        return False


def main() -> int:
    """Run all validation tests."""
    print(f"\n{Colors.BOLD}Settings Audit Fixes Validation{Colors.RESET}")
    print(f"{Colors.BOLD}Validating commits: f2669cb, 6275a00, 93d7501{Colors.RESET}")
    
    results = [
        validate_p0_07_no_auto_secret_key(),
        validate_p0_08_secretstr_masking(),
        validate_p0_09_atomic_directory(),
        validate_p1_07_thread_safe_singleton(),
        validate_p1_08_11_localhost_blocked(),
        validate_p1_09_immutable_proxy(),
        validate_p1_10_env_comparison(),
        validate_p2_07_cached_properties(),
    ]
    
    # Summary
    print_header("Validation Summary")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED ({passed}/{total}){Colors.RESET}")
        return 0
    else:
        print(f"  {Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED ({passed}/{total}){Colors.RESET}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}FATAL ERROR: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
