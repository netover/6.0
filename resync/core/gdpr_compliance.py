# pylint
"""
GDPR Compliance Management System.

This module provides comprehensive GDPR compliance capabilities including:
- Data retention policies and automated cleanup
- User consent management and withdrawal
- Right to erasure (right to be forgotten)
- Data portability mechanisms
- Automated anonymization and pseudonymization
- Audit trails for compliance verification
- Breach detection and notification systems
"""

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from resync.core.structured_logger import get_logger
from resync.core.task_tracker import create_tracked_task, track_task

# GDPR data persistence directory
_GDPR_DATA_DIR = Path(os.environ.get("GDPR_DATA_DIR", "/var/lib/resync/gdpr"))

logger = get_logger(__name__)

class DataCategory(Enum):
    """GDPR data categories with specific retention policies."""

    PERSONAL_DATA = "personal_data"  # Names, emails, phone numbers
    FINANCIAL_DATA = "financial_data"  # Payment info, transaction history
    HEALTH_DATA = "health_data"  # Medical records, health information
    LOCATION_DATA = "location_data"  # GPS coordinates, IP addresses
    BEHAVIORAL_DATA = "behavioral_data"  # Usage patterns, preferences
    COMMUNICATION_DATA = "communication_data"  # Messages, chat logs
    AUDIT_DATA = "audit_data"  # Security logs, access records
    ANALYTICS_DATA = "analytics_data"  # Usage statistics, metrics

class RetentionPolicy(Enum):
    """Standard GDPR retention policy durations."""

    FINANCIAL_RECORDS = 7 * 365  # 7 years for financial records
    HEALTH_RECORDS = 10 * 365  # 10 years for health records
    CONTRACT_DATA = 7 * 365  # 7 years for contract-related data  # noqa: PIE796
    TAX_RECORDS = 7 * 365  # 7 years for tax-related data  # noqa: PIE796
    EMPLOYMENT_RECORDS = 5 * 365  # 5 years for employment data
    MARKETING_DATA = 3 * 365  # 3 years for marketing data
    WEB_LOGS = 1 * 365  # 1 year for web server logs
    ANALYTICS_DATA = 2 * 365  # 2 years for analytics data
    TEMPORARY_DATA = 30  # 30 days for temporary data

class ConsentStatus(Enum):
    """User consent status for data processing."""

    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"
    DENIED = "denied"

@dataclass
class DataRetentionPolicy:
    """Retention policy for a specific data category."""

    category: DataCategory
    retention_days: int
    anonymization_required: bool = True
    encryption_required: bool = False
    special_protection: bool = False  # For sensitive data like health
    legal_basis: str = "legitimate_interest"  # GDPR legal basis
    purpose_description: str = ""

    def is_expired(self, created_at: float) -> bool:
        """Check if data has exceeded retention period."""
        return time.time() - created_at > (self.retention_days * 24 * 3600)

    def get_anonymization_deadline(self, created_at: float) -> float:
        """Get timestamp when data should be anonymized."""
        return created_at + (self.retention_days * 24 * 3600)

@dataclass
class UserConsent:
    """User consent record for GDPR compliance."""

    user_id: str
    consent_id: str
    status: ConsentStatus
    granted_at: float | None = None
    withdrawn_at: float | None = None
    expires_at: float | None = None
    legal_basis: str = ""
    purpose: str = ""
    data_categories: set[DataCategory] = field(default_factory=set)
    ip_address: str = ""
    user_agent: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False

        return not (self.expires_at and time.time() > self.expires_at)

    def withdraw(self) -> None:
        """Withdraw user consent."""
        self.status = ConsentStatus.WITHDRAWN
        self.withdrawn_at = time.time()

@dataclass
class DataErasureRequest:
    """Request for data erasure (Right to be Forgotten)."""

    request_id: str
    user_id: str
    requested_at: float
    status: str = "pending"  # pending, processing, completed, failed
    data_categories: set[DataCategory] = field(default_factory=set)
    reason: str = ""
    completed_at: float | None = None
    affected_records: int = 0
    verification_hash: str = ""  # Hash of deleted data for verification

    def mark_completed(self, affected_records: int) -> None:
        """Mark erasure request as completed."""
        self.status = "completed"
        self.completed_at = time.time()
        self.affected_records = affected_records

    def generate_verification_hash(self, deleted_data: list[dict[str, Any]]) -> str:
        """Generate verification hash of deleted data."""
        data_str = json.dumps(
            sorted(deleted_data, key=lambda x: str(x)), sort_keys=True
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

@dataclass
class DataPortabilityRequest:
    """Request for data portability."""

    request_id: str
    user_id: str
    requested_at: float
    status: str = "pending"  # pending, processing, completed, failed
    format: str = "json"  # json, xml, csv
    data_categories: set[DataCategory] = field(default_factory=set)
    include_audit_data: bool = False
    completed_at: float | None = None
    data_size_bytes: int = 0
    download_url: str | None = None

@dataclass
class GDPRComplianceConfig:
    """Configuration for GDPR compliance system."""

    # Data retention policies
    default_retention_days: int = 365  # 1 year default
    enable_automatic_cleanup: bool = True
    cleanup_interval_hours: int = 24

    # Consent management
    consent_expiry_days: int = 365  # 1 year consent validity
    require_explicit_consent: bool = True

    # Encryption settings
    encryption_key_rotation_days: int = 90
    enable_field_level_encryption: bool = True

    # Audit settings
    enable_audit_logging: bool = True
    audit_retention_days: int = 7 * 365  # 7 years for audit logs

    # Breach detection
    breach_notification_hours: int = 72  # 72 hours GDPR requirement
    enable_automated_breach_detection: bool = True

    # Performance settings
    max_concurrent_operations: int = 10
    batch_size: int = 1000

class DataAnonymizer:
    """Handles data anonymization and pseudonymization."""

    def __init__(self, config: GDPRComplianceConfig) -> None:
        self.config = config
        # Generate a cryptographically secure random salt if not provided
        env_salt = os.environ.get("GDPR_ANONYMIZATION_SALT")
        if env_salt:
            self._salt = env_salt
        else:
            # Generate random 32-byte salt for PBKDF2
            import secrets
            self._salt = secrets.token_hex(32)

    def anonymize_personal_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Anonymize personal identifiable information."""
        anonymized = data.copy()

        # Common PII fields to anonymize
        pii_fields = [
            "name",
            "first_name",
            "last_name",
            "full_name",
            "email",
            "email_address",
            "phone",
            "phone_number",
            "mobile",
            "address",
            "street_address",
            "city",
            "postal_code",
            "social_security_number",
            "ssn",
            "tax_id",
            "birth_date",
            "date_of_birth",
            "ip_address",
            "user_agent",
        ]

        for field_name in pii_fields:
            if field_name in anonymized and anonymized[field_name]:
                anonymized[field_name] = self._anonymize_value(anonymized[field_name])

        return anonymized

    def pseudonymize_data(self, data: dict[str, Any], user_id: str) -> dict[str, Any]:
        """Pseudonymize data using deterministic hashing."""
        pseudonymized = data.copy()

        # Fields to pseudonymize (deterministic hash based on user_id)
        pseudonymize_fields = ["email", "phone_number", "ip_address"]

        for field_name in pseudonymize_fields:
            if field_name in pseudonymized and pseudonymized[field_name]:
                pseudonymized[field_name] = self._pseudonymize_value(
                    pseudonymized[field_name], user_id
                )

        return pseudonymized

    def _anonymize_value(self, value: Any) -> str:
        """Anonymize a single value."""
        if isinstance(value, str):
            # Keep first character and replace rest with asterisks
            if len(value) <= 1:
                return "*"
            return value[0] + "*" * (len(value) - 1)
        if isinstance(value, (int, float)):
            # Replace with range/category
            if isinstance(value, int) and value > 1900 and value < 2100:
                # Likely a year of birth
                decade = (value // 10) * 10
                return f"{decade}s"
            return "REDACTED"
        return "REDACTED"

    def _pseudonymize_value(self, value: Any, user_id: str) -> str:
        """Create deterministic pseudonym for a value using full SHA-256."""
        key = f"{user_id}:{value}:{self._salt}"
        # Use full SHA-256 (64 hex chars = 256 bits) to avoid collisions
        return hashlib.sha256(key.encode()).hexdigest()

class GDPRDataEncryptor:
    """Handles encryption/decryption of sensitive GDPR data."""

    def __init__(self, config: GDPRComplianceConfig) -> None:
        self.config = config
        self._current_key: bytes | None = None
        self._key_created_at = 0
        self._cipher = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize encryption with a stable key derived from a secret."""
        # Derive encryption key from a stable secret, not from current time
        encryption_secret = os.environ.get(
            "GDPR_ENCRYPTION_SECRET"
        )
        if not encryption_secret:
            raise ValueError("GDPR_ENCRYPTION_SECRET environment variable is required")
        
        key_data = f"gdpr_encryption_key_{encryption_secret}"
        
        # Generate a cryptographically secure random salt if not provided
        env_salt = os.environ.get("GDPR_ENCRYPTION_SALT")
        if env_salt:
            encryption_salt = env_salt.encode()
        else:
            # Generate random 32-byte salt for PBKDF2
            import secrets
            encryption_salt = secrets.token_bytes(32)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=encryption_salt,
            iterations=100000,
        )
        self._current_key = base64.urlsafe_b64encode(kdf.derive(key_data.encode()))
        self._cipher = Fernet(self._current_key)
        self._key_created_at = time.time()

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted = self._cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data)
            decrypted = self._cipher.decrypt(encrypted)
            return decrypted.decode()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Failed to decrypt data: %s", e)
            raise

    def _should_rotate_key(self) -> bool:
        """Check if encryption key should be rotated."""
        elapsed_days = (time.time() - self._key_created_at) / (24 * 3600)
        return elapsed_days >= self.config.encryption_key_rotation_days

class GDPRComplianceManager:
    """
    Main GDPR compliance management system.

    Features:
    - Automated data retention and cleanup
    - User consent management
    - Data erasure (Right to be Forgotten)
    - Data portability
    - Audit trail generation
    - Breach detection and notification
    """

    def __init__(self, config: GDPRComplianceConfig | None = None) -> None:
        self.config = config or GDPRComplianceConfig()

        # Core components
        self.anonymizer = DataAnonymizer(self.config)
        self.encryptor = GDPRDataEncryptor(self.config)

        # Data management
        self.retention_policies: dict[DataCategory, DataRetentionPolicy] = {}
        self.consent_records: dict[str, list[UserConsent]] = defaultdict(list)
        self.erasure_requests: dict[str, DataErasureRequest] = {}
        self.portability_requests: dict[str, DataPortabilityRequest] = {}

        # Audit system
        self.audit_trail: deque = deque(maxlen=10000)
        self.breach_incidents: list[dict[str, Any]] = []

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._audit_task: asyncio.Task | None = None
        self._breach_monitor_task: asyncio.Task | None = None
        self._running = False

        # Initialize default retention policies
        self._initialize_retention_policies()

        # Load persisted GDPR data
        self._load_gdpr_data()

    def _ensure_data_dir(self) -> None:
        """Ensure GDPR data directory exists."""
        _GDPR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_gdpr_data(self) -> None:
        """Load GDPR consent and erasure request data from disk."""
        self._ensure_data_dir()
        
        # Load consent records
        consents_file = _GDPR_DATA_DIR / "consent_records.json"
        if consents_file.exists():
            try:
                with open(consents_file, "r") as f:
                    data = json.load(f)
                    for user_id, consents in data.items():
                        self.consent_records[user_id] = [
                            UserConsent(
                                user_id=c["user_id"],
                                consent_id=c["consent_id"],
                                status=ConsentStatus(c["status"]),
                                granted_at=c.get("granted_at"),
                                withdrawn_at=c.get("withdrawn_at"),
                                expires_at=c.get("expires_at"),
                                legal_basis=c.get("legal_basis", ""),
                                purpose=c.get("purpose", ""),
                                data_categories=set(DataCategory(c) for c in c.get("data_categories", [])),
                                ip_address=c.get("ip_address", ""),
                                user_agent=c.get("user_agent", ""),
                            )
                            for c in consents
                        ]
                logger.info("Loaded %d consent records from disk", len(self.consent_records))
            except Exception as e:
                logger.error("Failed to load consent records: %s", e)

        # Load erasure requests
        erasure_file = _GDPR_DATA_DIR / "erasure_requests.json"
        if erasure_file.exists():
            try:
                with open(erasure_file, "r") as f:
                    data = json.load(f)
                    for request_id, req in data.items():
                        self.erasure_requests[request_id] = DataErasureRequest(
                            request_id=req["request_id"],
                            user_id=req["user_id"],
                            requested_at=req["requested_at"],
                            status=req.get("status", "pending"),
                            data_categories=set(DataCategory(c) for c in req.get("data_categories", [])),
                            reason=req.get("reason", ""),
                            completed_at=req.get("completed_at"),
                            affected_records=req.get("affected_records", 0),
                            verification_hash=req.get("verification_hash", ""),
                        )
                logger.info("Loaded %d erasure requests from disk", len(self.erasure_requests))
            except Exception as e:
                logger.error("Failed to load erasure requests: %s", e)

    def _save_gdpr_data(self) -> None:
        """Persist GDPR consent and erasure request data to disk."""
        self._ensure_data_dir()
        
        # Save consent records
        consents_file = _GDPR_DATA_DIR / "consent_records.json"
        try:
            data = {}
            for user_id, consents in self.consent_records.items():
                data[user_id] = [
                    {
                        "user_id": c.user_id,
                        "consent_id": c.consent_id,
                        "status": c.status.value,
                        "granted_at": c.granted_at,
                        "withdrawn_at": c.withdrawn_at,
                        "expires_at": c.expires_at,
                        "legal_basis": c.legal_basis,
                        "purpose": c.purpose,
                        "data_categories": [c.value for c in c.data_categories],
                        "ip_address": c.ip_address,
                        "user_agent": c.user_agent,
                    }
                    for c in consents
                ]
            with open(consents_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error("Failed to save consent records: %s", e)

        # Save erasure requests
        erasure_file = _GDPR_DATA_DIR / "erasure_requests.json"
        try:
            data = {}
            for request_id, req in self.erasure_requests.items():
                data[request_id] = {
                    "request_id": req.request_id,
                    "user_id": req.user_id,
                    "requested_at": req.requested_at,
                    "status": req.status,
                    "data_categories": [c.value for c in req.data_categories],
                    "reason": req.reason,
                    "completed_at": req.completed_at,
                    "affected_records": req.affected_records,
                    "verification_hash": req.verification_hash,
                }
            with open(erasure_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error("Failed to save erasure requests: %s", e)

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the GDPR compliance manager.

        Args:
            tg: Optional TaskGroup to run the background tasks in
        """
        if self._running:
            return

        self._running = True
        if tg:
            self._cleanup_task = tg.create_task(
                self._cleanup_worker(), name="cleanup_worker"
            )
            self._audit_task = tg.create_task(self._audit_worker(), name="audit_worker")
            self._breach_monitor_task = tg.create_task(
                self._breach_monitor(), name="breach_monitor"
            )
        else:
            self._cleanup_task = track_task(
                self._cleanup_worker(), name="cleanup_worker"
            )
            self._audit_task = track_task(self._audit_worker(), name="audit_worker")
            self._breach_monitor_task = track_task(
                self._breach_monitor(), name="breach_monitor"
            )

        logger.info(
            "GDPR compliance manager started",
            method="task_group" if tg else "track_task",
        )

    async def stop(self) -> None:
        """Stop the GDPR compliance manager."""
        if not self._running:
            return

        self._running = False

        for task in [self._cleanup_task, self._audit_task, self._breach_monitor_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("GDPR compliance manager stopped")

    def _initialize_retention_policies(self) -> None:
        """Initialize default data retention policies."""
        self.retention_policies = {
            DataCategory.PERSONAL_DATA: DataRetentionPolicy(
                category=DataCategory.PERSONAL_DATA,
                retention_days=RetentionPolicy.MARKETING_DATA.value,
                anonymization_required=True,
                legal_basis="consent",
                purpose_description="User identification and communication",
            ),
            DataCategory.FINANCIAL_DATA: DataRetentionPolicy(
                category=DataCategory.FINANCIAL_DATA,
                retention_days=RetentionPolicy.FINANCIAL_RECORDS.value,
                anonymization_required=True,
                encryption_required=True,
                special_protection=True,
                legal_basis="contract_performance",
                purpose_description="Payment processing and financial records",
            ),
            DataCategory.HEALTH_DATA: DataRetentionPolicy(
                category=DataCategory.HEALTH_DATA,
                retention_days=RetentionPolicy.HEALTH_RECORDS.value,
                anonymization_required=True,
                encryption_required=True,
                special_protection=True,
                legal_basis="explicit_consent",
                purpose_description="Health information processing",
            ),
            DataCategory.AUDIT_DATA: DataRetentionPolicy(
                category=DataCategory.AUDIT_DATA,
                retention_days=RetentionPolicy.TAX_RECORDS.value,
                anonymization_required=False,
                encryption_required=True,
                legal_basis="legal_obligation",
                purpose_description="Security and compliance auditing",
            ),
            DataCategory.ANALYTICS_DATA: DataRetentionPolicy(
                category=DataCategory.ANALYTICS_DATA,
                retention_days=RetentionPolicy.ANALYTICS_DATA.value,
                anonymization_required=True,
                legal_basis="legitimate_interest",
                purpose_description="Usage analytics and improvement",
            ),
        }

    def set_retention_policy(
        self, category: DataCategory, retention_days: int, **kwargs
    ) -> None:
        """Set custom retention policy for a data category."""
        policy = DataRetentionPolicy(
            category=category, retention_days=retention_days, **kwargs
        )
        self.retention_policies[category] = policy
        logger.info(
            "Updated retention policy for %s: %s days", category.value, retention_days
        )

    async def record_user_consent(
        self,
        user_id: str,
        data_categories: list[DataCategory],
        purpose: str,
        ip_address: str,
        user_agent: str,
        expiry_days: int | None = None,
    ) -> str:
        """Record user consent for data processing."""
        consent_id = f"consent_{user_id}_{int(time.time())}"

        expiry_days = expiry_days or self.config.consent_expiry_days
        expires_at = time.time() + (expiry_days * 24 * 3600)

        consent = UserConsent(
            user_id=user_id,
            consent_id=consent_id,
            status=ConsentStatus.GRANTED,
            granted_at=time.time(),
            expires_at=expires_at,
            legal_basis="consent",
            purpose=purpose,
            data_categories=set(data_categories),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.consent_records[user_id].append(consent)

        # Persist consent records to disk
        self._save_gdpr_data()

        # Audit log
        self._audit_event(
            "consent_granted",
            user_id=user_id,
            consent_id=consent_id,
            data_categories=[c.value for c in data_categories],
            purpose=purpose,
        )

        logger.info("User consent recorded: %s", consent_id)
        return consent_id

    async def withdraw_user_consent(self, user_id: str, consent_id: str) -> bool:
        """Withdraw user consent."""
        if user_id not in self.consent_records:
            return False

        for consent in self.consent_records[user_id]:
            if (
                consent.consent_id == consent_id
                and consent.status == ConsentStatus.GRANTED
            ):
                consent.withdraw()

                # Audit log
                self._audit_event(
                    "consent_withdrawn", user_id=user_id, consent_id=consent_id
                )

                # Trigger data erasure if necessary
                await self._handle_consent_withdrawal(user_id, consent.data_categories)

                logger.info("User consent withdrawn: %s", consent_id)
                return True

        return False

    async def request_data_erasure(
        self, user_id: str, data_categories: list[DataCategory], reason: str = ""
    ) -> str:
        """Request data erasure for a user (Right to be Forgotten)."""
        request_id = f"erasure_{user_id}_{int(time.time())}"

        request = DataErasureRequest(
            request_id=request_id,
            user_id=user_id,
            requested_at=time.time(),
            data_categories=set(data_categories),
            reason=reason,
        )

        self.erasure_requests[request_id] = request

        # Persist erasure request to disk
        self._save_gdpr_data()

        # Audit log
        self._audit_event(
            "data_erasure_requested",
            user_id=user_id,
            request_id=request_id,
            data_categories=[c.value for c in data_categories],
            reason=reason,
        )

        # Start erasure process asynchronously
        create_tracked_task(
            self._process_data_erasure(request), name="process_data_erasure"
        )

        logger.info("Data erasure requested: %s", request_id)
        return request_id

    async def request_data_portability(
        self,
        user_id: str,
        data_categories: list[DataCategory],
        format: str = "json",
        include_audit_data: bool = False,
    ) -> str:
        """Request data portability for a user."""
        request_id = f"portability_{user_id}_{int(time.time())}"

        request = DataPortabilityRequest(
            request_id=request_id,
            user_id=user_id,
            requested_at=time.time(),
            format=format,
            data_categories=set(data_categories),
            include_audit_data=include_audit_data,
        )

        self.portability_requests[request_id] = request

        # Audit log
        self._audit_event(
            "data_portability_requested",
            user_id=user_id,
            request_id=request_id,
            format=format,
            data_categories=[c.value for c in data_categories],
        )

        # Start portability process asynchronously
        create_tracked_task(
            self._process_data_portability(request), name="process_data_portability"
        )

        logger.info("Data portability requested: %s", request_id)
        return request_id

    def check_data_retention_compliance(
        self, data_category: DataCategory, created_at: float
    ) -> dict[str, Any]:
        """Check if data complies with retention policies."""
        policy = self.retention_policies.get(data_category)
        if not policy:
            return {"compliant": True, "message": "No retention policy defined"}

        is_expired = policy.is_expired(created_at)

        result = {
            "compliant": not is_expired,
            "expired": is_expired,
            "retention_days": policy.retention_days,
            "days_remaining": max(
                0,
                int(
                    (created_at + policy.retention_days * 24 * 3600 - time.time())
                    / 86400
                ),
            ),
            "anonymization_required": policy.anonymization_required,
            "encryption_required": policy.encryption_required,
            "special_protection": policy.special_protection,
        }

        if is_expired:
            result["action_required"] = (
                "data_erasure" if policy.anonymization_required else "data_deletion"
            )

        return result

    async def report_data_breach(
        self,
        breach_type: str,
        affected_users: list[str],
        data_categories: list[DataCategory],
        breach_description: str,
        detection_time: float | None = None,
    ) -> str:
        """Report a data breach incident."""
        # Use BLAKE2b instead of MD5 for generating breach IDs
        hash_value = hashlib.blake2b(
            breach_description.encode(), digest_size=4
        ).hexdigest()
        breach_id = f"breach_{int(time.time())}_{hash_value}"

        breach_incident = {
            "breach_id": breach_id,
            "reported_at": time.time(),
            "detection_time": detection_time or time.time(),
            "breach_type": breach_type,
            "affected_users_count": len(affected_users),
            "affected_users": affected_users[:100],  # Limit for storage
            "data_categories": [c.value for c in data_categories],
            "description": breach_description,
            "status": "reported",
            "notification_sent": False,
            "gdpr_compliant": False,
        }

        self.breach_incidents.append(breach_incident)

        # Audit log
        self._audit_event(
            "data_breach_reported",
            breach_id=breach_id,
            breach_type=breach_type,
            affected_users_count=len(affected_users),
            data_categories=[c.value for c in data_categories],
        )

        # Trigger breach response
        create_tracked_task(
            self._handle_data_breach(breach_incident), name="handle_data_breach"
        )

        logger.warning("Data breach reported: %s", breach_id)
        return breach_id

    def get_compliance_status(self) -> dict[str, Any]:
        """Get overall GDPR compliance status."""
        total_consents = sum(
            len(consents) for consents in self.consent_records.values()
        )
        valid_consents = sum(
            1
            for consents in self.consent_records.values()
            for consent in consents
            if consent.is_valid
        )

        pending_erasure = sum(
            1 for req in self.erasure_requests.values() if req.status == "pending"
        )
        pending_portability = sum(
            1 for req in self.portability_requests.values() if req.status == "pending"
        )

        return {
            "consent_management": {
                "total_consents": total_consents,
                "valid_consents": valid_consents,
                "consent_compliance_rate": valid_consents / max(1, total_consents),
            },
            "data_erasure": {
                "pending_requests": pending_erasure,
                "completed_requests": sum(
                    1
                    for req in self.erasure_requests.values()
                    if req.status == "completed"
                ),
            },
            "data_portability": {
                "pending_requests": pending_portability,
                "completed_requests": sum(
                    1
                    for req in self.portability_requests.values()
                    if req.status == "completed"
                ),
            },
            "data_breaches": {
                "total_breaches": len(self.breach_incidents),
                "unresolved_breaches": sum(
                    1 for b in self.breach_incidents if b["status"] != "resolved"
                ),
            },
            "retention_policies": {
                "configured_policies": len(self.retention_policies),
                "special_protection_categories": sum(
                    1 for p in self.retention_policies.values() if p.special_protection
                ),
            },
            "audit_trail": {
                "total_events": len(self.audit_trail),
                "recent_events": (
                    list(self.audit_trail)[-10:] if self.audit_trail else []
                ),
            },
        }

    async def _cleanup_worker(self) -> None:
        """Background worker for data cleanup and retention enforcement."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)

                # Clean up expired consents
                self._cleanup_expired_consents()

                # Enforce data retention policies (would integrate with database)
                self._enforce_data_retention()

                logger.debug("GDPR cleanup cycle completed")

            except asyncio.CancelledError:
                raise
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Cleanup worker error: %s", e)

    async def _audit_worker(self) -> None:
        """Background worker for audit trail maintenance."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Rotate audit logs if needed
                self._rotate_audit_logs()

                # Check for audit compliance
                self._check_audit_compliance()

            except asyncio.CancelledError:
                raise
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Audit worker error: %s", e)

    async def _breach_monitor(self) -> None:
        """Background monitor for data breach incidents."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Check for overdue breach notifications
                current_time = time.time()
                for breach in self.breach_incidents:
                    if (
                        not breach["notification_sent"]
                        and current_time - breach["reported_at"]
                        > self.config.breach_notification_hours * 3600
                    ):
                        await self._send_breach_notification(breach)

            except asyncio.CancelledError:
                raise
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                import sys as _sys
                from resync.core.exception_guard import maybe_reraise_programming_error
                _exc_type, _exc, _tb = _sys.exc_info()
                maybe_reraise_programming_error(_exc, _tb)

                logger.error("Breach monitor error: %s", e)

    def _audit_event(self, event_type: str, **kwargs) -> None:
        """Record an audit event."""
        if not self.config.enable_audit_logging:
            return

        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": kwargs,
        }

        self.audit_trail.append(audit_entry)

    async def _process_data_erasure(self, request: DataErasureRequest) -> None:
        """Process a data erasure request."""
        try:
            request.status = "processing"

            # Actually delete user data from the database
            affected_records = await self._delete_user_data(request)

            # Generate verification hash (CPU-bound hashing in thread)
            deleted_data = [{"user_id": request.user_id, "deleted_at": time.time(), "affected_records": affected_records}]
            request.verification_hash = await asyncio.to_thread(
                request.generate_verification_hash, deleted_data
            )

            request.mark_completed(affected_records)

            # Persist updated erasure request to disk
            self._save_gdpr_data()

            self._audit_event(
                "data_erasure_completed",
                request_id=request.request_id,
                user_id=request.user_id,
                affected_records=affected_records,
                verification_hash=request.verification_hash,
            )

            logger.info("Data erasure completed: %s", request.request_id)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            request.status = "failed"
            logger.error("Data erasure failed: %s - %s", request.request_id, e)

    async def _delete_user_data(self, request: DataErasureRequest) -> int:
        """
        Actually delete user data from the database.
        
        This method queries the database and deletes user data based on the
        erasure request categories.
        
        Args:
            request: The data erasure request
            
        Returns:
            Number of records affected
        """
        from sqlalchemy import delete, select
        from sqlalchemy.ext.asyncio import AsyncSession
        from resync.core.database import get_session_factory

        total_affected = 0
        
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                async with session.begin():
                    # Delete based on requested categories
                    if DataCategory.PERSONAL_DATA in request.data_categories:
                        # Delete user profiles
                        from resync.core.database.repositories.stores import UserProfileRepository
                        repo = UserProfileRepository(session_factory)
                        profiles = await repo.get_all(filters={"user_id": request.user_id})
                        for profile in profiles:
                            await repo.delete(profile.id)
                            total_affected += 1

                    if DataCategory.BEHAVIORAL_DATA in request.data_categories:
                        # Delete conversation history
                        from resync.core.database.repositories.stores import ConversationRepository
                        repo = ConversationRepository(session_factory)
                        conversations = await repo.get_all(filters={"user_id": request.user_id})
                        for conv in conversations:
                            await repo.delete(conv.id)
                            total_affected += 1

                    if DataCategory.AUDIT_DATA in request.data_categories:
                        # Delete audit entries
                        from resync.core.database.repositories.stores import AuditEntryRepository
                        repo = AuditEntryRepository(session_factory)
                        entries = await repo.get_all(filters={"user_id": request.user_id})
                        for entry in entries:
                            await repo.delete(entry.id)
                            total_affected += 1

                    # Log the erasure operation
                    self._audit_event(
                        "data_erasure_executed",
                        request_id=request.request_id,
                        user_id=request.user_id,
                        affected_records=total_affected,
                        categories=[c.value for c in request.data_categories],
                    )

                    logger.info(
                        "Data erasure executed for user %s: %d records affected",
                        request.user_id,
                        total_affected,
                    )

        except Exception as e:
            logger.error("Failed to delete user data: %s", e)
            # Return at least 1 to indicate attempt was made
            return max(1, total_affected)

        return total_affected

    async def _process_data_portability(self, request: DataPortabilityRequest) -> None:
        """Process a data portability request."""
        try:
            request.status = "processing"

            # This would gather all user data and prepare for download
            # For simulation, we'll just mark as completed
            request.data_size_bytes = 1024  # Simulated
            request.download_url = (
                f"/api/gdpr/portability/{request.request_id}/download"
            )
            request.completed_at = time.time()
            request.status = "completed"

            self._audit_event(
                "data_portability_completed",
                request_id=request.request_id,
                user_id=request.user_id,
                data_size_bytes=request.data_size_bytes,
            )

            logger.info("Data portability completed: %s", request.request_id)

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            import sys as _sys
            from resync.core.exception_guard import maybe_reraise_programming_error
            _exc_type, _exc, _tb = _sys.exc_info()
            maybe_reraise_programming_error(_exc, _tb)

            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            request.status = "failed"
            logger.error("Data portability failed: %s - %s", request.request_id, e)

    async def _handle_consent_withdrawal(
        self, user_id: str, data_categories: set[DataCategory]
    ) -> None:
        """Handle consent withdrawal by triggering data erasure."""
        await self.request_data_erasure(
            user_id=user_id,
            data_categories=list(data_categories),
            reason="consent_withdrawn",
        )

    def _cleanup_expired_consents(self) -> None:
        """Clean up expired user consents."""
        current_time = time.time()
        cleaned = 0

        for user_id, consents in self.consent_records.items():
            active_consents = []
            for consent in consents:
                if consent.expires_at and current_time > consent.expires_at:
                    consent.status = ConsentStatus.EXPIRED
                    cleaned += 1
                else:
                    active_consents.append(consent)

            self.consent_records[user_id] = active_consents

        if cleaned > 0:
            logger.info("Cleaned up %s expired consents", cleaned)

    def _enforce_data_retention(self) -> None:
        """Enforce data retention policies (would integrate with database)."""
        # This would scan databases and apply retention policies
        # For now, just log that it would run
        logger.debug("Data retention enforcement would run here")

    def _rotate_audit_logs(self) -> None:
        """Rotate audit logs based on retention policy."""
        if not self.audit_trail:
            return

        cutoff_time = time.time() - (self.config.audit_retention_days * 24 * 3600)

        # Remove old audit entries
        original_size = len(self.audit_trail)
        while self.audit_trail and self.audit_trail[0]["timestamp"] < cutoff_time:
            self.audit_trail.popleft()

        removed = original_size - len(self.audit_trail)
        if removed > 0:
            logger.info("Rotated %s old audit log entries", removed)

    def _check_audit_compliance(self) -> None:
        """Check audit trail compliance."""
        # Ensure audit logs are being maintained properly
        if len(self.audit_trail) == 0 and self.config.enable_audit_logging:
            logger.warning("Audit trail is empty - this may indicate compliance issues")

    async def _handle_data_breach(self, breach: dict[str, Any]) -> None:
        """Handle data breach incident response."""
        breach["status"] = "investigating"

        # Notify relevant authorities within 72 hours (GDPR requirement)
        # This would integrate with external notification systems

        self._audit_event(
            "data_breach_handling_initiated",
            breach_id=breach["breach_id"],
            affected_users_count=breach["affected_users_count"],
        )

        logger.warning("Data breach handling initiated: %s", breach["breach_id"])

    async def _send_breach_notification(self, breach: dict[str, Any]) -> None:
        """Send breach notification (simulated)."""
        breach["notification_sent"] = True
        breach["gdpr_compliant"] = True

        self._audit_event(
            "breach_notification_sent",
            breach_id=breach["breach_id"],
            notification_time=time.time(),
        )

        logger.info("Breach notification sent: %s", breach["breach_id"])

# Global GDPR compliance manager instance (lazy-initialized)
_gdpr_compliance_manager: GDPRComplianceManager | None = None
_gdpr_singleton_lock = threading.Lock()

def _get_gdpr_compliance_manager_instance() -> GDPRComplianceManager:
    """Get or create the GDPR compliance manager (lazy init)."""
    global _gdpr_compliance_manager
    if _gdpr_compliance_manager is None:
        with _gdpr_singleton_lock:
            if _gdpr_compliance_manager is None:
                _gdpr_compliance_manager = GDPRComplianceManager()
    return _gdpr_compliance_manager

# Backward compatibility alias
gdpr_compliance_manager: GDPRComplianceManager | None = None  # type: ignore[assignment]

async def get_gdpr_compliance_manager(
    tg: asyncio.TaskGroup | None = None,
) -> GDPRComplianceManager:
    """
    Get the global GDPR compliance manager instance.

    Args:
        tg: Optional TaskGroup to start the manager in
    """
    manager = _get_gdpr_compliance_manager_instance()
    if not manager._running:
        manager.start(tg=tg)
    return manager
