# pylint: disable=all
"""
Circuit Breaker Manager

This module provides comprehensive management functionality for multiple
circuit breakers across different system components.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different system components.

    This class provides functionality for:
    - Managing circuit breakers for critical components
    - Monitoring circuit breaker health and status
    - Coordinating circuit breaker operations
    - Providing circuit breaker statistics and insights
    """

    def __init__(self):
        """Initialize the circuit breaker manager."""
        self._circuit_breakers: dict[str, Any] = {}
        self._breaker_stats: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def register_circuit_breaker(self, name: str, circuit_breaker: Any) -> None:
        """
        Register a circuit breaker for management.

        Args:
            name: Name of the circuit breaker
            circuit_breaker: Circuit breaker instance
        """
        self._circuit_breakers[name] = circuit_breaker
        logger.info("circuit_breaker_registered", name=name)

    def unregister_circuit_breaker(self, name: str) -> bool:
        """
        Unregister a circuit breaker.

        Args:
            name: Name of the circuit breaker to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._circuit_breakers:
            del self._circuit_breakers[name]
            logger.info("circuit_breaker_unregistered", name=name)
            return True
        return False

    async def execute_with_circuit_breaker(
        self, breaker_name: str, func, *args, **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            breaker_name: Name of the circuit breaker to use
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result if successful

        Raises:
            Exception: If circuit breaker is open or execution fails
        """
        circuit_breaker = self._circuit_breakers.get(breaker_name)
        if not circuit_breaker:
            logger.warning("circuit_breaker_not_found", breaker_name=breaker_name)
            # Execute without circuit breaker protection
            return await func(*args, **kwargs)

        try:
            result = await circuit_breaker.call(func, *args, **kwargs)

            # Update statistics
            await self._update_breaker_stats(breaker_name, success=True)

            return result

        except Exception as e:
            # Update statistics
            await self._update_breaker_stats(breaker_name, success=False)

            logger.error(
                "circuit_breaker_execution_failed",
                breaker_name=breaker_name,
                error=str(e),
            )
            raise

    def get_circuit_breaker_status(self, breaker_name: str) -> dict[str, Any] | None:
        """
        Get the status of a specific circuit breaker.

        Args:
            breaker_name: Name of the circuit breaker

        Returns:
            Circuit breaker status or None if not found
        """
        circuit_breaker = self._circuit_breakers.get(breaker_name)
        if not circuit_breaker:
            return None

        try:
            # Get stats from the circuit breaker
            if hasattr(circuit_breaker, "get_stats"):
                stats = circuit_breaker.get_stats()
            elif hasattr(circuit_breaker, "get_enhanced_stats"):
                stats = circuit_breaker.get_enhanced_stats()
            else:
                # Fallback to basic attributes
                stats = {
                    "state": getattr(circuit_breaker, "state", "unknown"),
                    "failures": getattr(circuit_breaker, "failure_count", 0),
                }

            # Add our own statistics
            our_stats = self._breaker_stats.get(breaker_name, {})

            return {
                "name": breaker_name,
                "state": stats.get("state", "unknown"),
                "failures": stats.get("failures", 0),
                "successes": stats.get("successes", 0),
                "error_rate": stats.get("failure_rate", 0),
                "last_failure": stats.get("last_failure_time"),
                "latency_p95": stats.get("latency_percentiles", {}).get("p95", 0),
                "our_stats": our_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(
                "failed_to_get_circuit_breaker_status",
                breaker_name=breaker_name,
                error=str(e),
            )
            return {
                "name": breaker_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_all_circuit_breaker_statuses(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all registered circuit breakers.

        Returns:
            Dictionary mapping breaker names to their status
        """
        statuses = {}

        for breaker_name in self._circuit_breakers:
            status = self.get_circuit_breaker_status(breaker_name)
            if status:
                statuses[breaker_name] = status

        return statuses

    async def reset_circuit_breaker(self, breaker_name: str) -> bool:
        """
        Reset a circuit breaker to closed state.

        Args:
            breaker_name: Name of the circuit breaker to reset

        Returns:
            True if reset, False if not found or reset failed
        """
        circuit_breaker = self._circuit_breakers.get(breaker_name)
        if not circuit_breaker:
            logger.warning(
                "circuit_breaker_not_found_for_reset", breaker_name=breaker_name
            )
            return False

        try:
            # Reset the circuit breaker
            if hasattr(circuit_breaker, "reset"):
                circuit_breaker.reset()
            else:
                # Manual reset for basic circuit breakers
                circuit_breaker.state = "closed"
                circuit_breaker.failure_count = 0

            logger.info("circuit_breaker_reset", breaker_name=breaker_name)

            # Update statistics
            await self._update_breaker_stats(breaker_name, reset=True)

            return True

        except Exception as e:
            logger.error(
                "circuit_breaker_reset_failed", breaker_name=breaker_name, error=str(e)
            )
            return False

    async def reset_all_circuit_breakers(self) -> dict[str, bool]:
        """
        Reset all circuit breakers to closed state.

        Returns:
            Dictionary mapping breaker names to reset success status
        """
        results = {}

        for breaker_name in self._circuit_breakers:
            results[breaker_name] = await self.reset_circuit_breaker(breaker_name)

        logger.info("all_circuit_breakers_reset", results=results)
        return results

    def get_open_circuit_breakers(self) -> list[str]:
        """
        Get list of circuit breakers that are currently open.

        Returns:
            List of open circuit breaker names
        """
        open_breakers = []

        for breaker_name, circuit_breaker in self._circuit_breakers.items():
            try:
                state = getattr(circuit_breaker, "state", "unknown")
                if state == "open":
                    open_breakers.append(breaker_name)
            except Exception as e:
                logger.error(
                    "failed_to_check_circuit_breaker_state",
                    breaker_name=breaker_name,
                    error=str(e),
                )

        return open_breakers

    async def get_circuit_breaker_summary(self) -> dict[str, Any]:
        """
        Get a summary of all circuit breaker states.

        Returns:
            Summary dictionary with circuit breaker statistics
        """
        all_statuses = await self.get_all_circuit_breaker_statuses()

        summary = {
            "total_breakers": len(all_statuses),
            "open_breakers": 0,
            "closed_breakers": 0,
            "half_open_breakers": 0,
            "unknown_breakers": 0,
            "breaker_names": list(all_statuses.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for status in all_statuses.values():
            state = status.get("state", "unknown")
            if state == "open":
                summary["open_breakers"] += 1
            elif state == "closed":
                summary["closed_breakers"] += 1
            elif state == "half-open":
                summary["half_open_breakers"] += 1
            else:
                summary["unknown_breakers"] += 1

        return summary

    async def _update_breaker_stats(
        self, breaker_name: str, success: bool = False, reset: bool = False
    ) -> None:
        """Update statistics for a circuit breaker."""
        async with self._lock:
            if breaker_name not in self._breaker_stats:
                self._breaker_stats[breaker_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "resets": 0,
                    "last_call": None,
                    "last_success": None,
                    "last_failure": None,
                }

            stats = self._breaker_stats[breaker_name]
            stats["total_calls"] += 1
            stats["last_call"] = datetime.now(timezone.utc)

            if reset:
                stats["resets"] += 1
            elif success:
                stats["successful_calls"] += 1
                stats["last_success"] = datetime.now(timezone.utc)
            else:
                stats["failed_calls"] += 1
                stats["last_failure"] = datetime.now(timezone.utc)

    def get_registered_breakers(self) -> list[str]:
        """Get list of registered circuit breaker names."""
        return list(self._circuit_breakers.keys())

    def is_circuit_breaker_registered(self, breaker_name: str) -> bool:
        """Check if a circuit breaker is registered."""
        return breaker_name in self._circuit_breakers

    async def cleanup_stale_stats(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale circuit breaker statistics.

        Args:
            max_age_hours: Maximum age in hours for statistics to keep

        Returns:
            Number of statistics entries cleaned up
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cleaned_count = 0

        async with self._lock:
            stale_breakers = []

            for breaker_name, stats in self._breaker_stats.items():
                # Check if all timestamps are stale
                timestamps = [
                    stats.get("last_call"),
                    stats.get("last_success"),
                    stats.get("last_failure"),
                ]

                if all(ts is None or ts < cutoff_time for ts in timestamps):
                    stale_breakers.append(breaker_name)

            # Remove stale statistics
            for breaker_name in stale_breakers:
                del self._breaker_stats[breaker_name]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                "cleaned_stale_circuit_breaker_stats", cleaned_count=cleaned_count
            )

        return cleaned_count

    def get_manager_stats(self) -> dict[str, Any]:
        """Get statistics about the circuit breaker manager itself."""
        return {
            "registered_breakers": len(self._circuit_breakers),
            "tracked_stats": len(self._breaker_stats),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
