"""
Helper utilities for FastAPI application
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_uuid() -> str:
    """Generate a new UUID4 string"""
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def safe_json_loads(data: str) -> dict[str, Any] | None:
    """Safely parse JSON string"""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_json_dumps(data: Any) -> str:
    """Safely serialize to JSON string"""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return "{}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for _unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return ".1f"
        size_bytes /= 1024.0
    return ".1f"


def extract_text_from_filename(filename: str) -> str:
    """Extract text content indicator based on file type."""
    path = Path(filename)
    extension = path.suffix.lower()
    stem = path.stem

    # Map extensions to content types
    text_extensions = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".yml", ".yaml"}
    doc_extensions = {".pdf", ".doc", ".docx", ".rtf", ".odt"}
    code_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}

    if extension in text_extensions:
        return f"text:{stem}"
    if extension in doc_extensions:
        return f"document:{stem}"
    if extension in code_extensions:
        return f"code:{stem}"
    return stem


def create_pagination_info(
    total_items: int, limit: int, offset: int, current_page: int | None = None
) -> dict[str, Any]:
    """Create pagination information"""
    total_pages = (total_items + limit - 1) // limit if limit > 0 else 1
    current_page = current_page or (offset // limit) + 1

    return {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": current_page,
        "items_per_page": limit,
        "has_next": current_page < total_pages,
        "has_previous": current_page > 1,
    }


def mask_sensitive_data(data: str, mask_char: str = "*") -> str:
    """Mask sensitive data in strings"""
    # Simple masking for demo - replace with more sophisticated logic
    if len(data) <= 4:
        return mask_char * len(data)
    return data[:2] + mask_char * (len(data) - 4) + data[-2:]


def validate_redis_connection(redis_client) -> bool:
    """Validate Redis connection"""
    try:
        redis_client.ping()
        return True
    except Exception as e:
        # Re-raise programming errors — these are bugs, not runtime failures
        if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
            raise
        logger.error("exception_caught", error=str(e), exc_info=True)
        return False


def create_error_response(
    error_type: str, message: str, status_code: int, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create standardized error response"""
    response = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
            "timestamp": get_current_timestamp(),
        }
    }

    if details:
        response["error"]["details"] = details

    return response


def create_success_response(
    data: Any, message: str | None = None, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create standardized success response"""
    response = {"success": True, "data": data, "timestamp": get_current_timestamp()}

    if message:
        response["message"] = message

    if metadata:
        response["metadata"] = metadata

    return response


def normalize_agent_name(name: str) -> str:
    """Normalize agent name for consistent storage/retrieval"""
    return name.lower().strip().replace(" ", "_")


def generate_correlation_id() -> str:
    """Generate correlation ID for request tracing"""
    return f"req_{uuid.uuid4().hex[:8]}"


def parse_user_agent(user_agent: str) -> dict[str, str]:
    """Parse user agent string to extract browser, OS, and device info."""
    ua_lower = user_agent.lower()

    # Detect browser
    browser = "unknown"
    if "chrome" in ua_lower and "edg" not in ua_lower:
        browser = "Chrome"
    elif "firefox" in ua_lower:
        browser = "Firefox"
    elif "safari" in ua_lower and "chrome" not in ua_lower:
        browser = "Safari"
    elif "edg" in ua_lower:
        browser = "Edge"
    elif "opera" in ua_lower or "opr" in ua_lower:
        browser = "Opera"
    elif "msie" in ua_lower or "trident" in ua_lower:
        browser = "Internet Explorer"

    # Detect OS
    os_name = "unknown"
    if "windows" in ua_lower:
        os_name = "Windows"
    elif "mac os" in ua_lower or "macos" in ua_lower:
        os_name = "macOS"
    elif "linux" in ua_lower:
        os_name = "Linux"
    elif "android" in ua_lower:
        os_name = "Android"
    elif "iphone" in ua_lower or "ipad" in ua_lower:
        os_name = "iOS"

    # Detect device type
    device = "desktop"
    if "mobile" in ua_lower or "android" in ua_lower:
        device = "mobile"
    elif "tablet" in ua_lower or "ipad" in ua_lower:
        device = "tablet"
    elif "bot" in ua_lower or "crawler" in ua_lower or "spider" in ua_lower:
        device = "bot"

    return {
        "raw": user_agent[:200],  # Truncate for safety
        "browser": browser,
        "os": os_name,
        "device": device,
    }
