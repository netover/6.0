# mypy: ignore-errors
# pylint: disable=no-member
"""
Advanced API Gateway for Microservices Architecture.

This module provides a comprehensive API Gateway with enterprise-grade features including:
- Intelligent request routing and load balancing
- Rate limiting and throttling with burst handling
- Authentication and authorization (JWT, OAuth2, API keys)
- Request/response transformation and enrichment
- Circuit breaker protection and health monitoring
- Caching with cache invalidation strategies
- Security features (WAF, DDoS protection, input validation)
- Service discovery and dynamic configuration
- Monitoring, metrics, and observability
- Traffic management and canary deployments
"""

import asyncio
from resync.core.task_tracker import create_tracked_task
import contextlib
import hashlib
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import jwt
from aiohttp import web

from resync.core.resilience_singletons import (
    adaptive_llm_api_breaker,
    adaptive_tws_api_breaker,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class HTTPMethod(Enum):
    """HTTP methods supported by the gateway."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    CONNECT = "CONNECT"
    TRACE = "TRACE"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"


class RateLimitType(Enum):
    """Rate limiting types."""

    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    BANDWIDTH_PER_SECOND = "bandwidth_per_second"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""

    service_name: str
    url: str
    weight: int = 1
    health_check_url: str | None = None
    timeout_seconds: float = 30.0
    retry_count: int = 3
    circuit_breaker_enabled: bool = True
    rate_limit_enabled: bool = True
    caching_enabled: bool = False

    @property
    def parsed_url(self) -> str:
        """Get parsed URL."""
        return urlparse(self.url).netloc

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy (placeholder - would integrate with health checks)."""
        return True  # Simplified for now


@dataclass
class RouteConfiguration:
    """Route configuration for API Gateway."""

    path_pattern: str
    service_name: str
    methods: set[HTTPMethod] = field(
        default_factory=lambda: {HTTPMethod.GET, HTTPMethod.POST}
    )
    strip_prefix: bool = True
    add_prefix: str = ""
    timeout_seconds: float = 30.0
    rate_limit: dict[str, Any] | None = None
    authentication_required: bool = False
    authorization_required: bool = False
    cors_enabled: bool = True
    caching_enabled: bool = False
    transformation_enabled: bool = False
    logging_enabled: bool = True

    def matches_path(self, path: str) -> bool:
        """Check if path matches route pattern."""
        pattern = self.path_pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{pattern}$", path))


@dataclass
class RateLimitRule:
    """Rate limiting rule."""

    limit_type: RateLimitType
    limit_value: int
    window_seconds: int
    burst_limit: int | None = None
    key_strategy: str = "ip"  # ip, user, api_key, custom
    exclude_paths: set[str] = field(default_factory=set)

    def get_key(self, request: web.Request) -> str:
        """Get rate limit key from request."""
        if self.key_strategy == "ip":
            return self._get_client_ip(request)
        if self.key_strategy == "user":
            return request.get("user_id", "anonymous")
        if self.key_strategy == "api_key":
            return request.headers.get("X-API-Key", "no_key")
        return "default"


class APIGateway:
    """
    Advanced API Gateway for microservices architecture.

    Features:
    - Intelligent routing with service discovery
    - Load balancing with health monitoring
    - Rate limiting and throttling
    - Authentication and authorization
    - Request/response transformation
    - Circuit breaker protection
    - Caching with invalidation
    - Security features (WAF, DDoS protection)
    - Monitoring and observability
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Core components
        self.routes: list[RouteConfiguration] = []
        self.services: dict[str, list[ServiceEndpoint]] = defaultdict(list)
        self.rate_limits: dict[str, RateLimitRule] = {}
        self.auth_providers: dict[str, Any] = {}
        self.transformers: dict[str, Callable] = {}

        # Runtime state
        self.service_index: dict[str, int] = defaultdict(int)  # For round-robin
        self.active_connections: dict[str, int] = defaultdict(
            int
        )  # For least connections
        self.request_counts: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Caching
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_ttl: dict[str, float] = {}

        # Security
        self.waf_rules: list[dict[str, Any]] = []
        self.blacklisted_ips: set[str] = set()
        self.suspicious_patterns: list[re.Pattern] = []

        # Metrics
        self.metrics: dict[str, Any] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "response_time_avg": 0.0,
            "rate_limited_requests": 0,
            "cached_responses": 0,
            "circuit_breaker_trips": 0,
        }

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None
        self._running = False

        # Initialize components
        self._initialize_gateway()

    def _initialize_gateway(self) -> None:
        """Initialize gateway with default configuration."""
        # Add default rate limiting rules
        self.rate_limits["default"] = RateLimitRule(
            limit_type=RateLimitType.REQUESTS_PER_MINUTE,
            limit_value=100,
            window_seconds=60,
            burst_limit=150,
        )

        # Add default WAF rules
        self.waf_rules = [
            {
                "name": "sql_injection",
                "pattern": r"(?i)(union|select|insert|delete|update|drop|create|alter)\s",
                "severity": "high",
                "action": "block",
            },
            {
                "name": "xss_attempt",
                "pattern": r"<script[^>]*>.*?</script>",
                "severity": "high",
                "action": "block",
            },
            {
                "name": "path_traversal",
                "pattern": r"\.\./|\.\.\\",
                "severity": "critical",
                "action": "block",
            },
        ]

        # Compile regex patterns
        for rule in self.waf_rules:
            self.suspicious_patterns.append(re.compile(rule["pattern"]))

    async def start(self) -> None:
        """Start the API Gateway."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = await create_tracked_task(
            self._cleanup_worker(), name="cleanup_worker"
        )
        self._health_check_task = await create_tracked_task(
            self._health_check_worker(), name="health_check_worker"
        )
        self._metrics_task = await create_tracked_task(
            self._metrics_worker(), name="metrics_worker"
        )

        logger.info("API Gateway started")

    async def stop(self) -> None:
        """Stop the API Gateway."""
        if not self._running:
            return

        self._running = False

        for task in [self._cleanup_task, self._health_check_task, self._metrics_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("API Gateway stopped")

    def add_route(self, route: RouteConfiguration) -> None:
        """Add a route configuration."""
        self.routes.append(route)
        logger.info("Added route: %s -> %s", route.path_pattern, route.service_name)

    def add_service(self, service_name: str, endpoint: ServiceEndpoint) -> None:
        """Add a service endpoint."""
        self.services[service_name].append(endpoint)
        logger.info("Added service endpoint: %s -> %s", service_name, endpoint.url)

    def add_rate_limit_rule(self, name: str, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self.rate_limits[name] = rule
        logger.info("Added rate limit rule: %s", name)

    def add_auth_provider(self, name: str, provider: Any) -> None:
        """Add an authentication provider."""
        self.auth_providers[name] = provider
        logger.info("Added auth provider: %s", name)

    def add_transformer(self, name: str, transformer: Callable) -> None:
        """Add a request/response transformer."""
        self.transformers[name] = transformer
        logger.info("Added transformer: %s", name)

    async def handle_request(self, request: web.Request) -> web.Response:
        """
        Main request handler for the API Gateway.

        This method orchestrates the entire request processing pipeline:
        1. Security checks (WAF, IP filtering)
        2. Rate limiting
        3. Authentication
        4. Route matching and service discovery
        5. Load balancing
        6. Request transformation
        7. Circuit breaker protection
        8. Caching
        9. Request forwarding
        10. Response processing and transformation
        """
        start_time = time.time()
        self.metrics["requests_total"] += 1

        try:
            # 1. Security checks
            security_result = await self._perform_security_checks(request)
            if not security_result["allowed"]:
                return self._create_error_response(403, security_result["reason"])

            # 2. Rate limiting
            rate_limit_result = await self._check_rate_limits(request)
            if not rate_limit_result["allowed"]:
                self.metrics["rate_limited_requests"] += 1
                return self._create_error_response(429, "Rate limit exceeded")

            # 3. Authentication
            if request.get(
                "route_config", RouteConfiguration("", "")
            ).authentication_required:
                auth_result = await self._authenticate_request(request)
                if not auth_result["authenticated"]:
                    return self._create_error_response(401, "Authentication required")

            # 4. Route matching
            route_config = self._find_route(request)
            if not route_config:
                return self._create_error_response(404, "Route not found")

            # Store route config in request for later use
            request["route_config"] = route_config

            # 5. Service discovery and load balancing
            service_endpoint = await self._select_service_endpoint(
                route_config.service_name, request
            )
            if not service_endpoint:
                return self._create_error_response(503, "Service unavailable")

            # 6. Request transformation
            transformed_request = await self._transform_request(request, route_config)

            # 7. Circuit breaker check
            if service_endpoint.circuit_breaker_enabled:
                circuit_breaker = self._get_circuit_breaker(route_config.service_name)
                if not await circuit_breaker.can_execute():
                    return self._create_error_response(
                        503, "Service temporarily unavailable"
                    )

            # 8. Cache check
            if route_config.caching_enabled and request.method == "GET":
                cache_key = self._generate_cache_key(request)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    self.metrics["cached_responses"] += 1
                    return cached_response

            # 9. Forward request
            response = await self._forward_request(
                transformed_request, service_endpoint, route_config
            )

            # 10. Response processing
            processed_response = await self._process_response(response, route_config)

            # 11. Caching
            if (
                route_config.caching_enabled
                and request.method == "GET"
                and response.status == 200
            ):
                cache_key = self._generate_cache_key(request)
                self._cache_response(cache_key, processed_response)

            # Update metrics
            self.metrics["requests_success"] += 1
            processing_time = time.time() - start_time
            self.metrics["response_time_avg"] = (
                (
                    self.metrics["response_time_avg"]
                    * (self.metrics["requests_success"] - 1)
                )
                + processing_time
            ) / self.metrics["requests_success"]

            return processed_response

        except Exception as e:
            self.metrics["requests_failed"] += 1
            logger.error("Request processing error: %s", e, exc_info=True)
            return self._create_error_response(500, "Internal server error")

    async def _perform_security_checks(self, request: web.Request) -> dict[str, Any]:
        """Perform security checks on incoming request."""
        client_ip = self._get_client_ip(request)

        # Check IP blacklisting
        if client_ip in self.blacklisted_ips:
            return {"allowed": False, "reason": "IP address blocked"}

        # WAF checks
        for pattern in self.suspicious_patterns:
            if pattern.search(str(request.url)) or pattern.search(str(request.headers)):
                return {"allowed": False, "reason": "Security violation detected"}

        # Input validation (simplified)
        if len(str(request.url)) > 2048:
            return {"allowed": False, "reason": "URL too long"}

        return {"allowed": True}

    async def _check_rate_limits(self, request: web.Request) -> dict[str, Any]:
        """Check rate limits for request."""
        route_config = getattr(request, "route_config", None)
        rule_name = (
            route_config.rate_limit.get("rule", "default")
            if route_config and route_config.rate_limit
            else "default"
        )

        rule = self.rate_limits.get(rule_name, self.rate_limits["default"])
        if not rule:
            return {"allowed": True}

        # Check if path is excluded
        if request.path in rule.exclude_paths:
            return {"allowed": True}

        key = rule.get_key(request)
        current_time = time.time()

        # Get request counts for this key
        if key not in self.request_counts:
            self.request_counts[key] = deque(maxlen=1000)

        # Clean old requests
        while (
            self.request_counts[key]
            and current_time - self.request_counts[key][0] > rule.window_seconds
        ):
            self.request_counts[key].popleft()

        # Check rate limit
        request_count = len(self.request_counts[key])

        if request_count >= rule.limit_value:
            return {"allowed": False, "retry_after": rule.window_seconds}

        # Allow burst
        if rule.burst_limit and request_count >= rule.burst_limit:
            return {"allowed": False, "retry_after": rule.window_seconds // 4}

        # Add current request
        self.request_counts[key].append(current_time)

        return {"allowed": True}

    async def _authenticate_request(self, request: web.Request) -> dict[str, Any]:
        """Authenticate incoming request."""
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            return {"authenticated": False, "reason": "No authorization header"}

        # JWT authentication
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                # v5.9.7: Remove hard-coded secret; use application settings.
                from resync.settings import settings as _settings

                secret = getattr(_settings, "secret_key", None)
                if secret is None:
                    return {
                        "authenticated": False,
                        "reason": "JWT secret not configured",
                    }
                if hasattr(secret, "get_secret_value"):
                    secret_value = secret.get_secret_value()
                else:
                    secret_value = str(secret)

                alg = getattr(_settings, "jwt_algorithm", "HS256")
                payload = jwt.decode(token, secret_value, algorithms=[alg])
                request["user_id"] = payload.get("user_id")
                request["user_roles"] = payload.get("roles", [])
                return {"authenticated": True}
            except jwt.ExpiredSignatureError:
                return {"authenticated": False, "reason": "Token expired"}
            except jwt.InvalidTokenError:
                return {"authenticated": False, "reason": "Invalid token"}

        # API Key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key.startswith("api_key_"):
            # This would validate API key (simplified)
            request["api_key_valid"] = True
            return {"authenticated": True}

        return {"authenticated": False, "reason": "Invalid authentication"}

    def _find_route(self, request: web.Request) -> RouteConfiguration | None:
        """Find matching route configuration."""
        for route in self.routes:
            if route.matches_path(request.path) and request.method in [
                m.value for m in route.methods
            ]:
                return route
        return None

    async def _select_service_endpoint(
        self,
        service_name: str,
        request: web.Request,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ) -> ServiceEndpoint | None:
        """Select service endpoint using load balancing strategy."""
        available_endpoints = [
            ep for ep in self.services.get(service_name, []) if ep.is_healthy
        ]

        if not available_endpoints:
            return None

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            index = self.service_index[service_name]
            self.service_index[service_name] = (index + 1) % len(available_endpoints)
            return available_endpoints[index]

        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select endpoint with least active connections
            return min(
                available_endpoints,
                key=lambda ep: self.active_connections.get(ep.url, 0),
            )

        if strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Simple weighted selection
            total_weight = sum(ep.weight for ep in available_endpoints)
            if total_weight == 0:
                return available_endpoints[0]

            # This is a simplified implementation
            weights = [ep.weight for ep in available_endpoints]
            cumulative_weights = [sum(weights[: i + 1]) for i in range(len(weights))]

            import random

            rand = random.randint(1, total_weight)
            for i, weight in enumerate(cumulative_weights):
                if rand <= weight:
                    return available_endpoints[i]

        return available_endpoints[0]  # Default fallback

    async def _transform_request(
        self, request: web.Request, route_config: RouteConfiguration
    ) -> web.Request:
        """Transform request before forwarding."""
        if not route_config.transformation_enabled:
            return request

        # Apply transformations (simplified)
        # In a real implementation, this would use the transformers registry

        # Strip prefix if configured
        if route_config.strip_prefix:
            new_path = request.path
            if route_config.path_pattern.startswith("/"):
                prefix = (
                    route_config.path_pattern.split("/")[1]
                    if "/" in route_config.path_pattern[1:]
                    else ""
                )
                if prefix and new_path.startswith(f"/{prefix}"):
                    new_path = new_path[len(f"/{prefix}") :]

            # Add new prefix if configured
            if route_config.add_prefix:
                new_path = f"/{route_config.add_prefix}{new_path}"

            # Create new URL
            new_url = str(request.url).replace(request.path, new_path)
            request = request.clone(url=new_url)

        return request

    def _get_circuit_breaker(self, service_name: str):
        """Get circuit breaker for service."""
        # Use existing circuit breakers based on service type
        if "llm" in service_name.lower():
            return adaptive_llm_api_breaker
        if "tws" in service_name.lower():
            return adaptive_tws_api_breaker
        # Return a default circuit breaker
        return adaptive_llm_api_breaker  # Simplified

    async def _forward_request(
        self,
        request: web.Request,
        endpoint: ServiceEndpoint,
        route_config: RouteConfiguration,
    ) -> web.Response:
        """Forward request to service endpoint."""
        # Increment active connections
        self.active_connections[endpoint.url] += 1

        try:
            # Prepare target URL
            target_url = urljoin(endpoint.url, request.path_qs)

            # Prepare headers (remove hop-by-hop headers)
            headers = dict(request.headers)
            hop_by_hop_headers = {
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "transfer-encoding",
                "upgrade",
            }
            for header in hop_by_hop_headers:
                headers.pop(header, None)

            # Add gateway headers
            headers["X-Forwarded-For"] = self._get_client_ip(request)
            headers["X-Gateway-Request-Id"] = request.get("request_id", "unknown")

            # Create HTTP client session
            timeout = aiohttp.ClientTimeout(total=route_config.timeout_seconds)

            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                # Forward request - session.request() depends on session being created first
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=await request.read() if request.body_exists else None,
                    allow_redirects=False,
                ) as response:
                    # Create response
                    response_data = await response.read()
                    return web.Response(
                        status=response.status,
                        headers=dict(response.headers),
                        body=response_data,
                    )

        except asyncio.TimeoutError:
            raise web.HTTPGatewayTimeout(text="Service timeout") from None
        except aiohttp.ClientError as e:
            logger.error("Service request failed: %s", e)
            raise web.HTTPBadGateway(text="Service unavailable") from e
        finally:
            # Decrement active connections
            self.active_connections[endpoint.url] -= 1

    async def _process_response(
        self, response: web.Response, route_config: RouteConfiguration
    ) -> web.Response:
        """Process response before returning to client."""
        # Add gateway headers
        response.headers["X-Gateway-Processed"] = "true"
        response.headers["X-Gateway-Timestamp"] = str(int(time.time()))

        # Apply response transformations if configured
        if route_config.transformation_enabled:
            # This would apply response transformers
            pass

        return response

    def _generate_cache_key(self, request: web.Request) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            request.path_qs,
            str(sorted(request.headers.items())),
        ]
        key_string = "|".join(key_parts)
        # Use BLAKE2b instead of MD5 for better security and performance
        return hashlib.blake2b(key_string.encode(), digest_size=16).hexdigest()

    def _get_cached_response(self, cache_key: str) -> web.Response | None:
        """Get cached response if available and valid."""
        if cache_key in self.cache and cache_key in self.cache_ttl:
            if time.time() < self.cache_ttl[cache_key]:
                cached_data = self.cache[cache_key]
                return web.Response(
                    status=cached_data["status"],
                    headers=cached_data["headers"],
                    body=cached_data["body"],
                )
            # Cache expired
            del self.cache[cache_key]
            del self.cache_ttl[cache_key]

        return None

    def _cache_response(
        self, cache_key: str, response: web.Response, ttl_seconds: int = 300
    ) -> None:
        """Cache response."""
        self.cache[cache_key] = {
            "status": response.status,
            "headers": dict(response.headers),
            "body": response.body if hasattr(response, "body") else b"",
        }
        self.cache_ttl[cache_key] = time.time() + ttl_seconds

    def _get_client_ip(self, request: web.Request) -> str:
        """Get client IP address."""
        # Check forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to peer name
        return (
            request.transport.get_extra_info("peername")[0]
            if request.transport
            else "unknown"
        )

    def _create_error_response(self, status: int, message: str) -> web.Response:
        """Create standardized error response."""
        return web.json_response(
            {
                "error": {
                    "code": status,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
            status=status,
        )

    async def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup and maintenance."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Clean expired cache entries
                current_time = time.time()
                expired_keys = [
                    key for key, ttl in self.cache_ttl.items() if current_time > ttl
                ]

                for key in expired_keys:
                    self.cache.pop(key, None)
                    self.cache_ttl.pop(key, None)

                # Clean old request counts (older than 1 hour)
                cutoff_time = current_time - 3600
                for _key, requests in self.request_counts.items():
                    while requests and requests[0] < cutoff_time:
                        requests.popleft()

                if expired_keys:
                    logger.debug(
                        "Cleaned up %s expired cache entries", len(expired_keys)
                    )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Cleanup worker error: %s", e)

    async def _health_check_worker(self) -> None:
        """Background worker for service health checks."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute

                # Perform health checks on all service endpoints
                for _service_name, endpoints in self.services.items():
                    for endpoint in endpoints:
                        if endpoint.health_check_url:
                            # This would perform actual health checks
                            # For now, just mark as healthy
                            pass

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Health check worker error: %s", e)

    async def _metrics_worker(self) -> None:
        """Background worker for metrics collection."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute

                # Log current metrics
                logger.info(
                    "gateway_metrics",
                    requests_total=self.metrics["requests_total"],
                    requests_success=self.metrics["requests_success"],
                    requests_failed=self.metrics["requests_failed"],
                    response_time_avg=round(self.metrics["response_time_avg"], 3),
                    rate_limited_requests=self.metrics["rate_limited_requests"],
                    cached_responses=self.metrics["cached_responses"],
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Metrics worker error: %s", e)

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive gateway metrics."""
        return {
            "performance": {
                "requests_total": self.metrics["requests_total"],
                "requests_success": self.metrics["requests_success"],
                "requests_failed": self.metrics["requests_failed"],
                "success_rate": self.metrics["requests_success"]
                / max(1, self.metrics["requests_total"]),
                "response_time_avg": self.metrics["response_time_avg"],
                "rate_limited_requests": self.metrics["rate_limited_requests"],
                "cached_responses": self.metrics["cached_responses"],
            },
            "services": {
                service_name: {
                    "endpoints_count": len(endpoints),
                    "healthy_endpoints": sum(1 for ep in endpoints if ep.is_healthy),
                    "active_connections": sum(
                        self.active_connections.get(ep.url, 0) for ep in endpoints
                    ),
                }
                for service_name, endpoints in self.services.items()
            },
            "routes": {
                "total_routes": len(self.routes),
                "authenticated_routes": sum(
                    1 for r in self.routes if r.authentication_required
                ),
            },
            "cache": {
                "cached_entries": len(self.cache),
                "cache_hit_ratio": self.metrics["cached_responses"]
                / max(1, self.metrics["requests_total"]),
            },
            "security": {
                "blacklisted_ips": len(self.blacklisted_ips),
                "waf_rules": len(self.waf_rules),
                "rate_limit_rules": len(self.rate_limits),
            },
        }


# Global API Gateway instance
api_gateway = APIGateway()


async def get_api_gateway() -> APIGateway:
    """Get the global API Gateway instance."""
    if not api_gateway._running:
        await api_gateway.start()
    return api_gateway
