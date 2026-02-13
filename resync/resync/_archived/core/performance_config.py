"""
Performance optimization configuration for Resync application.

This module provides centralized configuration for performance optimizations
including connection pooling, caching strategies, and resource limits.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceConfig:
    """Centralized performance configuration management."""

    # Connection Pool Settings
    MAX_DB_CONNECTIONS = 20
    MIN_DB_CONNECTIONS = 5
    DB_CONNECTION_TIMEOUT = 30.0
    DB_COMMAND_TIMEOUT = 60.0

    # Redis Connection Pool
    REDIS_MAX_CONNECTIONS = 50
    REDIS_CONNECTION_TIMEOUT = 10.0
    REDIS_RETRY_ON_TIMEOUT = True

    # HTTP Client Optimizations
    HTTP_CONNECTION_POOL_SIZE = 100
    HTTP_MAX_KEEPALIVE = 20
    HTTP_TIMEOUT_CONNECT = 10.0
    HTTP_TIMEOUT_READ = 30.0
    HTTP_TIMEOUT_WRITE = 30.0

    # Async Task Management
    MAX_WORKER_TASKS = 100
    TASK_QUEUE_SIZE = 1000
    TASK_TIMEOUT_DEFAULT = 300.0  # 5 minutes

    # Caching Optimizations
    CACHE_MEMORY_LIMIT_MB = 512
    CACHE_COMPRESSION_THRESHOLD = 1024  # bytes
    CACHE_EVICTION_BATCH_SIZE = 100

    # LLM Performance
    LLM_CONCURRENT_REQUESTS_MAX = 10
    LLM_REQUEST_TIMEOUT = 30.0  # 30s realistic timeout for production
    LLM_RATE_LIMIT_REQUESTS = 60  # per minute
    LLM_RATE_LIMIT_TOKENS = 100000  # per minute

    @classmethod
    def get_database_config(cls) -> dict[str, Any]:
        """Get optimized database connection configuration."""
        return {
            "max_connections": cls.MAX_DB_CONNECTIONS,
            "min_connections": cls.MIN_DB_CONNECTIONS,
            "connection_timeout": cls.DB_CONNECTION_TIMEOUT,
            "command_timeout": cls.DB_COMMAND_TIMEOUT,
            "application_name": "resync_app",
        }

    @classmethod
    def get_redis_config(cls) -> dict[str, Any]:
        """Get optimized Redis connection configuration."""
        return {
            "max_connections": cls.REDIS_MAX_CONNECTIONS,
            "socket_timeout": cls.REDIS_CONNECTION_TIMEOUT,
            "socket_connect_timeout": cls.REDIS_CONNECTION_TIMEOUT,
            "retry_on_timeout": cls.REDIS_RETRY_ON_TIMEOUT,
            "health_check_interval": 30,
        }

    @classmethod
    def get_http_client_config(cls) -> dict[str, Any]:
        """Get optimized HTTP client configuration."""
        return {
            "limits": {
                "max_connections": cls.HTTP_CONNECTION_POOL_SIZE,
                "max_keepalive_connections": cls.HTTP_MAX_KEEPALIVE,
            },
            "timeout": {
                "connect": cls.HTTP_TIMEOUT_CONNECT,
                "read": cls.HTTP_TIMEOUT_READ,
                "write": cls.HTTP_TIMEOUT_WRITE,
                "pool": cls.HTTP_TIMEOUT_CONNECT,
            },
        }

    @classmethod
    def get_task_manager_config(cls) -> dict[str, Any]:
        """Get optimized task manager configuration."""
        return {
            "max_workers": min(cls.MAX_WORKER_TASKS, 10),  # Conservative default
            "queue_size": cls.TASK_QUEUE_SIZE,
            "default_timeout": cls.TASK_TIMEOUT_DEFAULT,
        }

    @classmethod
    def get_cache_config(cls) -> dict[str, Any]:
        """Get optimized cache configuration."""
        return {
            "memory_limit_mb": cls.CACHE_MEMORY_LIMIT_MB,
            "compression_threshold": cls.CACHE_COMPRESSION_THRESHOLD,
            "eviction_batch_size": cls.CACHE_EVICTION_BATCH_SIZE,
        }

    @classmethod
    def get_llm_config(cls) -> dict[str, Any]:
        """Get optimized LLM configuration."""
        return {
            "max_concurrent_requests": cls.LLM_CONCURRENT_REQUESTS_MAX,
            "request_timeout": cls.LLM_REQUEST_TIMEOUT,
            "rate_limit_requests": cls.LLM_RATE_LIMIT_REQUESTS,
            "rate_limit_tokens": cls.LLM_RATE_LIMIT_TOKENS,
        }

    @classmethod
    def validate_performance_settings(cls) -> bool:
        """Validate that performance settings are reasonable."""
        issues = []

        # Check connection pool sizes
        if cls.MAX_DB_CONNECTIONS < cls.MIN_DB_CONNECTIONS:
            issues.append("MAX_DB_CONNECTIONS must be >= MIN_DB_CONNECTIONS")

        if cls.HTTP_CONNECTION_POOL_SIZE < cls.HTTP_MAX_KEEPALIVE:
            issues.append("HTTP_CONNECTION_POOL_SIZE must be >= HTTP_MAX_KEEPALIVE")

        # Check timeouts
        if cls.HTTP_TIMEOUT_READ < cls.HTTP_TIMEOUT_CONNECT:
            issues.append("HTTP read timeout should be >= connect timeout")

        if issues:
            for issue in issues:
                logger.warning(f"Performance configuration issue: {issue}")
            return False

        logger.info("Performance configuration validated successfully")
        return True


# Validate configuration on import
if not PerformanceConfig.validate_performance_settings():
    logger.warning("Performance configuration has validation issues")

# Export configurations for easy access
database_config = PerformanceConfig.get_database_config()
redis_config = PerformanceConfig.get_redis_config()
http_config = PerformanceConfig.get_http_client_config()
task_config = PerformanceConfig.get_task_manager_config()
cache_config = PerformanceConfig.get_cache_config()
llm_config = PerformanceConfig.get_llm_config()
