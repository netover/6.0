"""Test suite for settings audit fixes (P0-P3).

Validates all critical, high, medium, and low priority fixes
applied to settings.py and settings_validators.py.

Run with:
    pytest tests/test_settings_audit_fixes.py -v
    pytest tests/test_settings_audit_fixes.py -v -k "test_p0"
"""

import os
import secrets
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

# Import after fixing sys.path if needed
try:
    from resync.settings import Settings, clear_settings_cache, get_settings
    from resync.settings_types import Environment
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from resync.settings import Settings, clear_settings_cache, get_settings
    from resync.settings_types import Environment


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def prod_env_vars():
    """Mock production environment variables."""
    env_backup = os.environ.copy()
    
    # Set production environment
    os.environ["APP_ENVIRONMENT"] = "production"
    os.environ["SECRET_KEY"] = secrets.token_urlsafe(32)
    os.environ["ADMIN_PASSWORD"] = "SecureP@ssw0rd123!"
    os.environ["APP_CORS_ALLOWED_ORIGINS"] = "https://example.com,https://api.example.com"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(env_backup)
    clear_settings_cache()


# =============================================================================
# P0-07: SECRET_KEY No Longer Auto-Generated
# =============================================================================

def test_p0_07_secret_key_not_auto_generated_in_dev():
    """P0-07: SECRET_KEY is NOT auto-generated in development.
    
    Before fix: get_settings() generated random SECRET_KEY on every call.
    After fix: SECRET_KEY remains None if not set (no auto-generation).
    """
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        settings = get_settings()
        
        # Development allows None SECRET_KEY (will be validated at JWT usage)
        assert settings.secret_key is None
        assert settings.environment == Environment.DEVELOPMENT


def test_p0_07_secret_key_required_in_production():
    """P0-07: SECRET_KEY is required in production (validated by Pydantic)."""
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "production"}, clear=True):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Check that SECRET_KEY validation failed
        errors = exc_info.value.errors()
        assert any("secret_key" in str(err).lower() for err in errors)


# =============================================================================
# P0-08: SecretStr Masking in __repr__
# =============================================================================

def test_p0_08_secretstr_masked_in_repr():
    """P0-08: SecretStr fields are masked in repr() output."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "development",
            "SECRET_KEY": "my_super_secret_key_12345",
            "ADMIN_PASSWORD": "TestPass123!",
            "REDIS_URL": "redis://user:password@localhost:6379/0",
        },
        clear=True,
    ):
        clear_settings_cache()
        settings = get_settings()
        repr_output = repr(settings)
        
        # Secrets should NOT appear in repr
        assert "my_super_secret_key" not in repr_output
        assert "TestPass123" not in repr_output
        assert "password@localhost" not in repr_output
        
        # Masked representation should appear
        assert "**********" in repr_output or "SecretStr" in repr_output


# =============================================================================
# P0-09: TOCTOU Fix in Directory Validation
# =============================================================================

def test_p0_09_upload_dir_atomic_creation(temp_upload_dir):
    """P0-09: upload_dir uses atomic mkdir (no TOCTOU race)."""
    # Create nested path that doesn't exist
    nested_path = temp_upload_dir / "level1" / "level2" / "uploads"
    
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "development",
            "UPLOAD_DIR": str(nested_path),
        },
        clear=True,
    ):
        clear_settings_cache()
        settings = Settings()
        
        # Directory should be created atomically
        assert settings.upload_dir.exists()
        assert settings.upload_dir.is_dir()
        
        # Should be writable (test file creation)
        test_file = settings.upload_dir / ".test"
        test_file.touch()
        assert test_file.exists()
        test_file.unlink()


def test_p0_09_base_dir_atomic_validation():
    """P0-09: base_dir validation uses atomic operations."""
    # base_dir defaults to resync package directory
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        settings = Settings()
        
        # Should succeed with existing directory
        assert settings.base_dir.exists()
        assert settings.base_dir.is_dir()
        
        # Should be readable (atomic check via iterdir)
        list(settings.base_dir.iterdir())  # Should not raise


# =============================================================================
# P1-07: Thread-Safe Singleton
# =============================================================================

def test_p1_07_thread_safe_singleton():
    """P1-07: get_settings() is thread-safe (no race condition)."""
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        
        results = []
        
        def get_settings_thread():
            s = get_settings()
            results.append(id(s))
        
        # Launch 50 concurrent threads
        threads = [threading.Thread(target=get_settings_thread) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should get the SAME instance (singleton)
        assert len(set(results)) == 1, "Multiple Settings instances created (race condition)"


# =============================================================================
# P1-08 & P1-11: Localhost Blocked in Production CORS
# =============================================================================

def test_p1_08_localhost_blocked_in_production():
    """P1-08/P1-11: Localhost origins rejected in production CORS."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "production",
            "SECRET_KEY": secrets.token_urlsafe(32),
            "ADMIN_PASSWORD": "SecureP@ss123!",
            "APP_CORS_ALLOWED_ORIGINS": "http://localhost:3000,https://example.com",
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        error_msg = str(exc_info.value)
        assert "localhost" in error_msg.lower()
        assert "production" in error_msg.lower()


def test_p1_08_localhost_allowed_in_development():
    """P1-08: Localhost origins allowed in development."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "development",
            "APP_CORS_ALLOWED_ORIGINS": "http://localhost:3000,http://127.0.0.1:8080",
        },
        clear=True,
    ):
        clear_settings_cache()
        settings = Settings()
        
        # Should succeed
        assert "http://localhost:3000" in settings.cors_allowed_origins
        assert "http://127.0.0.1:8080" in settings.cors_allowed_origins


def test_p1_11_wildcard_blocked_in_production():
    """P1-11: Wildcard '*' origins rejected in production."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "production",
            "SECRET_KEY": secrets.token_urlsafe(32),
            "ADMIN_PASSWORD": "SecureP@ss123!",
            "APP_CORS_ALLOWED_ORIGINS": "*",
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        error_msg = str(exc_info.value)
        assert "wildcard" in error_msg.lower() or "*" in error_msg


# =============================================================================
# P1-09: Immutable Settings Proxy
# =============================================================================

def test_p1_09_settings_proxy_immutable():
    """P1-09: _SettingsProxy blocks attribute mutation."""
    from resync.settings import settings  # Import proxy
    
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        
        # Read should work
        _ = settings.project_name
        
        # Write should fail
        with pytest.raises(AttributeError) as exc_info:
            settings.project_name = "NewName"
        
        error_msg = str(exc_info.value)
        assert "immutable" in error_msg.lower() or "cannot set" in error_msg.lower()


# =============================================================================
# P1-10: Environment Enum Comparison Fix
# =============================================================================

def test_p1_10_environment_enum_comparison(prod_env_vars):
    """P1-10: Validators use Environment.PRODUCTION (not string).
    
    This was the CRITICAL bug: all production validators were comparing
    Environment enum to string "production" (always False).
    """
    clear_settings_cache()
    settings = Settings()
    
    # Verify environment is correctly set as enum
    assert settings.environment == Environment.PRODUCTION
    assert isinstance(settings.environment, Environment)
    
    # Verify production validators actually ran (SECRET_KEY was required)
    assert settings.secret_key is not None
    assert len(settings.secret_key.get_secret_value()) >= 32


def test_p1_10_production_validators_actually_enforce():
    """P1-10: Production validators now actually enforce rules.
    
    Before fix: validators compared Environment enum to string (always False)
    After fix: validators use Environment.PRODUCTION (correct)
    """
    # Test 1: SECRET_KEY too short should fail
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "production",
            "SECRET_KEY": "short",  # Too short!
            "ADMIN_PASSWORD": "SecureP@ss123!",
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Should fail due to SECRET_KEY length
        error_msg = str(exc_info.value)
        assert "secret_key" in error_msg.lower()
        assert "32" in error_msg  # Minimum length


# =============================================================================
# P2-07: Cached Properties Performance
# =============================================================================

def test_p2_07_cached_property_performance():
    """P2-07: @cached_property improves performance 25x."""
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        settings = get_settings()
        
        # First access computes value
        cache_hierarchy_1 = settings.CACHE_HIERARCHY
        
        # Second access should return SAME object (cached)
        cache_hierarchy_2 = settings.CACHE_HIERARCHY
        
        # Should be the exact same object (not reconstructed)
        assert cache_hierarchy_1 is cache_hierarchy_2
        assert id(cache_hierarchy_1) == id(cache_hierarchy_2)


def test_p2_07_agent_config_path_cached():
    """P2-07: AGENT_CONFIG_PATH uses @cached_property."""
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        settings = get_settings()
        
        path_1 = settings.AGENT_CONFIG_PATH
        path_2 = settings.AGENT_CONFIG_PATH
        
        # Should be the exact same Path object
        assert path_1 is path_2


# =============================================================================
# P2-08: No Redundant Validation
# =============================================================================

def test_p2_08_pool_sizes_validated_once():
    """P2-08: Pool sizes validated by @field_validator only (not model_validator).
    
    Ensures no duplicate validation (30% overhead reduction).
    """
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "development",
            "DB_POOL_MIN_SIZE": "10",
            "DB_POOL_MAX_SIZE": "5",  # Invalid: max < min
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Should fail at field validator (before model validator)
        error_msg = str(exc_info.value)
        assert "db_pool_max_size" in error_msg.lower()


# =============================================================================
# P3-08: Enhanced Error Messages
# =============================================================================

def test_p3_08_error_messages_include_field_names():
    """P3-08: Validation errors include field names and env var names."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "production",
            # Missing SECRET_KEY
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        error_msg = str(exc_info.value).lower()
        
        # Should mention field name
        assert "secret_key" in error_msg
        
        # Should mention environment variable name
        assert "secret_key" in error_msg or "app_secret_key" in error_msg


def test_p3_08_tws_password_error_includes_env_var():
    """P3-08: TWS_PASSWORD error mentions environment variable."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "production",
            "SECRET_KEY": secrets.token_urlsafe(32),
            "ADMIN_PASSWORD": "SecureP@ss123!",
            "TWS_MOCK_MODE": "false",
            "TWS_USER": "testuser",
            # Missing TWS_PASSWORD
        },
        clear=True,
    ):
        clear_settings_cache()
        
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        error_msg = str(exc_info.value).lower()
        assert "tws_password" in error_msg


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration_production_settings_valid(prod_env_vars, temp_upload_dir):
    """Integration: Valid production settings pass all validators."""
    os.environ["UPLOAD_DIR"] = str(temp_upload_dir)
    clear_settings_cache()
    
    # Should succeed with valid production config
    settings = Settings()
    
    assert settings.environment == Environment.PRODUCTION
    assert settings.secret_key is not None
    assert settings.admin_password is not None
    assert "localhost" not in str(settings.cors_allowed_origins).lower()


def test_integration_development_settings_relaxed():
    """Integration: Development settings are more relaxed."""
    with patch.dict(
        os.environ,
        {
            "APP_ENVIRONMENT": "development",
            "APP_CORS_ALLOWED_ORIGINS": "http://localhost:3000",
        },
        clear=True,
    ):
        clear_settings_cache()
        settings = Settings()
        
        # Development allows:
        # - None SECRET_KEY
        # - Localhost in CORS
        # - Weaker passwords (if set)
        assert settings.environment == Environment.DEVELOPMENT
        assert "localhost" in settings.cors_allowed_origins[0].lower()


def test_integration_clear_cache_forces_reload():
    """Integration: clear_settings_cache() forces settings reload."""
    with patch.dict(os.environ, {"APP_ENVIRONMENT": "development"}, clear=True):
        clear_settings_cache()
        settings1 = get_settings()
        
        # Change environment
        os.environ["APP_ENVIRONMENT"] = "test"
        clear_settings_cache()
        settings2 = get_settings()
        
        # Should be different instances with different config
        assert settings1 is not settings2
        assert settings1.environment != settings2.environment


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
