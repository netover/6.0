"""
Self-Healing Server System for Resync

Automatically monitors and repairs Resync services:
- API server health checks
- Database connection monitoring
- Redis connection monitoring
- Process crash detection & restart
- Resource usage monitoring
- Alert notifications
- Auto-recovery procedures

Critical for TWS/HWA mission-critical operations!

Usage:
    python -m resync.tools.self_healing

    # Or with custom config:
    API_URL=http://localhost:8000/health \\
    CHECK_INTERVAL=30 \\
    ALERT_EMAIL=admin@company.com \\
    python -m resync.tools.self_healing

Author: Resync Team
Version: 5.9.9
"""

import asyncio
import os
import smtplib
import subprocess
from datetime import datetime
from email.mime.text import MIMEText

import httpx
import psutil
import structlog
from pydantic import BaseModel
import aiofiles

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class HealthCheckConfig(BaseModel):
    """Health check configuration."""
    api_url: str = "http://localhost:8000/health"
    database_check: bool = True
    redis_check: bool = True
    check_interval: int = 30  # seconds
    restart_threshold: int = 3  # failures before restart
    notification_email: str | None = None


class ProcessConfig(BaseModel):
    """Process monitoring configuration."""
    name: str
    command: list[str]
    working_dir: str = "."
    required: bool = True
    restart_delay: int = 5  # seconds before restart


# =============================================================================
# Health Checkers
# =============================================================================

class HealthChecker:
    """Base health checker."""

    def __init__(self, name: str):
        self.name = name
        self.failure_count = 0
        self.last_check = None
        self.last_status = "unknown"

    async def check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if healthy, False otherwise
        """
        raise NotImplementedError


class APIHealthChecker(HealthChecker):
    """Checks API endpoint health."""

    def __init__(self, url: str):
        super().__init__("API")
        self.url = url

    async def check(self) -> bool:
        """Check API health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.url)

                if response.status_code == 200:
                    data = response.json()

                    # Check all components
                    healthy = (
                        data.get("status") == "healthy" and
                        data.get("database") == "connected" and
                        data.get("redis") == "connected"
                    )

                    if healthy:
                        self.failure_count = 0
                        self.last_status = "healthy"
                        return True
                    self.failure_count += 1
                    self.last_status = "unhealthy"
                    logger.warning(
                        "API unhealthy",
                        data=data
                    )
                    return False
                self.failure_count += 1
                self.last_status = f"http_{response.status_code}"
                logger.error(
                    "API health check failed",
                    status_code=response.status_code
                )
                return False

        except Exception as e:
            self.failure_count += 1
            self.last_status = "unreachable"
            logger.error(f"API health check error: {e}")
            return False


class ProcessHealthChecker(HealthChecker):
    """Checks if process is running."""

    def __init__(self, process_name: str, pid_file: str | None = None):
        super().__init__(f"Process:{process_name}")
        self.process_name = process_name
        self.pid_file = pid_file

    async def check(self) -> bool:
        """Check if process is running."""
        try:
            # Check by PID file if provided
            if self.pid_file and os.path.exists(self.pid_file):
                async with aiofiles.open(self.pid_file) as f:
                    pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            self.failure_count = 0
                            self.last_status = "running"
                            return True

            # Check by process name
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    if self.process_name in ' '.join(proc.cmdline() or []):
                        self.failure_count = 0
                        self.last_status = "running"
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass

            # Process not found
            self.failure_count += 1
            self.last_status = "not_running"
            logger.warning(f"Process {self.process_name} not running")
            return False

        except Exception as e:
            self.failure_count += 1
            self.last_status = "error"
            logger.error(f"Process check error: {e}")
            return False


class ResourceMonitor(HealthChecker):
    """Monitors system resource usage."""

    def __init__(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0
    ):
        super().__init__("Resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check(self) -> bool:
        """Check system resources."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Check thresholds
            issues = []

            if cpu_percent > self.cpu_threshold:
                issues.append(f"CPU: {cpu_percent}%")

            if memory_percent > self.memory_threshold:
                issues.append(f"Memory: {memory_percent}%")

            if disk_percent > self.disk_threshold:
                issues.append(f"Disk: {disk_percent}%")

            if issues:
                self.failure_count += 1
                self.last_status = f"High usage: {', '.join(issues)}"
                logger.warning(
                    "Resource usage high",
                    cpu=cpu_percent,
                    memory=memory_percent,
                    disk=disk_percent
                )
                return False

            self.failure_count = 0
            self.last_status = "normal"
            return True

        except Exception as e:
            logger.error(f"Resource check error: {e}")
            return True  # Don't fail on monitoring errors


# =============================================================================
# Recovery Actions
# =============================================================================

class RecoveryAction:
    """Base recovery action."""

    def __init__(self, name: str):
        self.name = name

    async def execute(self) -> bool:
        """
        Execute recovery action.

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError


class RestartServiceAction(RecoveryAction):
    """Restarts a service."""

    def __init__(self, service_name: str, command: list[str], working_dir: str = "."):
        super().__init__(f"Restart:{service_name}")
        self.service_name = service_name
        self.command = command
        self.working_dir = working_dir

    async def execute(self) -> bool:
        """Restart the service."""
        try:
            logger.info(
                f"🔄 Restarting {self.service_name}...",
                command=' '.join(self.command)
            )

            # Kill existing process
            await self._kill_existing()

            # Wait a bit
            await asyncio.sleep(2)

            # Start new process
            process = subprocess.Popen(
                self.command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )

            # Save PID
            pid_file = f"/tmp/resync_{self.service_name}.pid"
            async with aiofiles.open(pid_file, 'w') as f:
                f.write(str(process.pid))

            logger.info(
                f"✅ {self.service_name} restarted",
                pid=process.pid
            )

            return True

        except Exception as e:
            logger.error(f"Failed to restart {self.service_name}: {e}")
            return False

    async def _kill_existing(self):
        """Kill existing process."""
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.cmdline() or [])
                if self.service_name in cmdline:
                    logger.info(f"Killing existing process {proc.pid}")
                    proc.terminate()

                    # Wait for graceful shutdown
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.debug("suppressed_exception", error=str(exc), exc_info=True)  # was: pass


class ClearCacheAction(RecoveryAction):
    """Clears application cache."""

    def __init__(self):
        super().__init__("ClearCache")

    async def execute(self) -> bool:
        """Clear Redis cache."""
        try:
            import redis

            r = redis.Redis(host='localhost', port=6379, db=0)
            r.flushdb()

            logger.info("🗑️ Cache cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# =============================================================================
# Notification System
# =============================================================================

class Notifier:
    """Sends notifications about issues and recoveries."""

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str | None = None,
        sender_password: str | None = None,
        recipient_email: str | None = None
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email or os.getenv("ALERT_EMAIL")
        self.sender_password = sender_password or os.getenv("ALERT_PASSWORD")
        self.recipient_email = recipient_email or self.sender_email

    async def send(self, subject: str, body: str):
        """Send email notification."""
        if not self.sender_email or not self.sender_password:
            logger.warning("Email not configured, skipping notification")
            return

        try:
            msg = MIMEText(body)
            msg['Subject'] = f"[Resync] {subject}"
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"📧 Notification sent: {subject}")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")


# =============================================================================
# Self-Healing Orchestrator
# =============================================================================

class SelfHealingOrchestrator:
    """
    Main orchestrator for self-healing system.

    Monitors health, detects issues, and executes recovery actions.
    """

    def __init__(self, config: HealthCheckConfig):
        self.config = config

        # Health checkers
        self.checkers: list[HealthChecker] = [
            APIHealthChecker(config.api_url),
            ProcessHealthChecker("uvicorn", "/tmp/resync_api.pid"),
            ResourceMonitor(cpu_threshold=90, memory_threshold=90, disk_threshold=90),
        ]

        # Recovery actions
        _api_host = os.environ.get("API_HOST", "0.0.0.0")  # noqa: S104
        _api_port = os.environ.get("API_PORT", "8000")
        self.recovery_actions = {
            "restart_api": RestartServiceAction(
                "api",
                ["uv", "run", "uvicorn", "resync.main:app", "--host", _api_host, "--port", _api_port],
                working_dir="."
            ),
            "clear_cache": ClearCacheAction(),
        }

        # Notifier
        self.notifier = Notifier(recipient_email=config.notification_email)

        # State
        self.consecutive_failures = 0
        self.last_recovery_time = None

    async def monitor(self):
        """
        Main monitoring loop.

        Runs forever, checking health and recovering as needed.
        """
        logger.info("🏥 Self-Healing System started")
        logger.info(f"API endpoint: {self.config.api_url}")
        logger.info(f"Check interval: {self.config.check_interval}s")

        try:
            while True:
                await self._check_and_heal()
                await asyncio.sleep(self.config.check_interval)

        except KeyboardInterrupt:
            logger.info("Self-healing system stopped")

    async def _check_and_heal(self):
        """Check health and heal if needed."""
        # Run all health checks
        results = {}

        for checker in self.checkers:
            healthy = await checker.check()
            results[checker.name] = {
                "healthy": healthy,
                "failures": checker.failure_count,
                "status": checker.last_status
            }

        # Check if any critical failures
        critical_failures = [
            name for name, result in results.items()
            if not result["healthy"] and name.startswith(("API", "Process"))
        ]

        if critical_failures:
            self.consecutive_failures += 1

            logger.warning(
                f"⚠️ Health check failures ({self.consecutive_failures}/{self.config.restart_threshold})",
                failures=critical_failures
            )

            # Trigger recovery if threshold reached
            if self.consecutive_failures >= self.config.restart_threshold:
                await self._recover(critical_failures, results)
        else:
            # All healthy
            if self.consecutive_failures > 0:
                logger.info("✅ System recovered and healthy")
                await self.notifier.send(
                    "System Recovered",
                    "All health checks passing. System is stable."
                )

            self.consecutive_failures = 0

    async def _recover(self, failures: list[str], results: dict):
        """Execute recovery actions."""
        logger.error(
            "🚨 CRITICAL: Initiating recovery",
            failures=failures
        )

        # Notify about issue
        await self.notifier.send(
            "System Issue Detected",
            "Critical failures detected:\n\n" +
            "\n".join(f"- {f}: {results[f]['status']}" for f in failures) +
            "\n\nAttempting automatic recovery..."
        )

        # Execute recovery actions
        recovery_success = False

        # 1. Try restarting API
        if "API" in str(failures) or "Process" in str(failures):
            success = await self.recovery_actions["restart_api"].execute()
            if success:
                # Wait for startup
                await asyncio.sleep(10)

                # Verify
                api_checker = next(c for c in self.checkers if c.name == "API")
                if await api_checker.check():
                    recovery_success = True
                    logger.info("✅ Recovery successful via API restart")

        # 2. Try clearing cache if that didn't work
        if not recovery_success:
            await self.recovery_actions["clear_cache"].execute()
            await asyncio.sleep(5)

        # Notify about recovery
        if recovery_success:
            await self.notifier.send(
                "Recovery Successful",
                "System automatically recovered and is now healthy."
            )
            self.consecutive_failures = 0
        else:
            await self.notifier.send(
                "Recovery Failed - Manual Intervention Required",
                "Automatic recovery attempts failed. Please investigate immediately."
            )

        self.last_recovery_time = datetime.now()


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """Run self-healing system."""
    config = HealthCheckConfig(
        api_url=os.getenv("API_URL", "http://localhost:8000/health"),
        check_interval=int(os.getenv("CHECK_INTERVAL", "30")),
        notification_email=os.getenv("ALERT_EMAIL"),
    )

    orchestrator = SelfHealingOrchestrator(config)
    await orchestrator.monitor()


if __name__ == "__main__":
    asyncio.run(main())
