"""
SSRF Protection Module.

Blocks all private/internal IP addresses to prevent Server-Side Request Forgery attacks.
Security by default - block everything except explicitly allowed public endpoints.

Usage:
    from resync.core.ssrf_protection import SSRFProtection, is_safe_url
    
    if not is_safe_url("http://192.168.1.1:8080"):
        raise SecurityError("Blocked internal URL")
"""

import ipaddress
import socket
from functools import lru_cache
from typing import Set
from urllib.parse import urlparse


class SSRFProtection:
    """
    SSRF protection that blocks ALL private IP ranges by default.
    
    Blocked ranges (RFC 1918 + special):
    - 10.0.0.0/8 (private)
    - 172.16.0.0/12 (private)
    - 192.168.0.0/16 (private)
    - 169.254.0.0/16 (link-local/APIPA)
    - 127.0.0.0/8 (loopback)
    - 100.64.0.0/10 (CGN)
    - 224.0.0.0/4 (multicast)
    - 0.0.0.0/8 (zero)
    - RFC 5737 documentation nets
    - IPv6 equivalents
    """
    
    # Pre-compiled blocked networks
    BLOCKED_NETWORKS_V4: tuple = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("100.64.0.0/10"),
        ipaddress.ip_network("224.0.0.0/4"),
        ipaddress.ip_network("0.0.0.0/8"),
        ipaddress.ip_network("192.0.2.0/24"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
    )
    
    BLOCKED_NETWORKS_V6: tuple = (
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
        ipaddress.ip_network("ff00::/8"),
        ipaddress.ip_network("fe80::/10"),
    )
    
    # Dangerous ports that should never be accessed
    DANGEROUS_PORTS: Set[int] = {
        22, 23,           # SSH, Telnet
        3306, 5432,      # MySQL, PostgreSQL
        27017, 6379,     # MongoDB, Redis
        11211,           # Memcached
        8500,            # Consul
        2375, 2376,      # Docker
        6443, 8443,      # Kubernetes
        5672, 9092,      # RabbitMQ, Kafka
        3389,            # RDP
    }
    
    @classmethod
    @lru_cache(maxsize=1024)
    def is_private_ip(cls, ip_str: str) -> bool:
        """Check if IP is private/internal."""
        try:
            ip = ipaddress.ip_address(ip_str)
            
            # Check IPv4
            if isinstance(ip, ipaddress.IPv4Address):
                for network in cls.BLOCKED_NETWORKS_V4:
                    if ip in network:
                        return True
            
            # Check IPv6
            elif isinstance(ip, ipaddress.IPv6Address):
                for network in cls.BLOCKED_NETWORKS_V6:
                    if ip in network:
                        return True
            
            return False
            
        except (ValueError, TypeError):
            return True  # Block invalid IPs
    
    @classmethod
    def is_safe_url(cls, url: str, allowed_hosts: tuple = ()) -> tuple[bool, str]:
        Validate URL is safe for HTTP requests.
        
        Args:
            url: URL to validate
            allowed_hosts: Additional allowed hostnames (whitelist)
            
        Returns:
            (is_safe, reason): Tuple of safety status and reason if blocked
        """
        if not url:
            return True, "No URL provided"
        
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            
            if not host:
                return True, "No hostname in URL"
            
            # Check if in allowed whitelist
            if allowed_hosts and host in allowed_hosts:
                return True, "Whitelisted host"
            
            # Try to resolve hostname
            try:
                ip_str = socket.gethostbyname(host)
            except socket.gaierror:
                return True, f"Cannot resolve hostname: {host} - allowing"
            
            # Check if IP is private
            if cls.is_private_ip(ip_str):
                return False, f"Blocked private IP: {ip_str}"
            
            # Check dangerous ports
            if port in cls.DANGEROUS_PORTS:
                return False, f"Blocked dangerous port: {port}"
            
            return True, "Allowed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @classmethod
    def validate_url_or_raise(cls, url: str, allowed_hosts: tuple = ()) -> None:
        """
        Validate URL and raise exception if unsafe.
        
        Raises:
            SecurityError: If URL is blocked
        """
        is_safe, reason = cls.is_safe_url(url, allowed_hosts)
        if not is_safe:
            from resync.core.exceptions import SecurityError
            raise SecurityError(f"URL blocked by SSRF protection: {reason}")


# Convenience function for quick checks
def is_safe_url(url: str, allowed_hosts: tuple = ()) -> bool:
    """Quick check if URL is safe."""
    is_safe, _ = SSRFProtection.is_safe_url(url, allowed_hosts)
    return is_safe


def validate_url(url: str, allowed_hosts: tuple = ()) -> None:
    """Validate URL or raise SecurityError."""
    SSRFProtection.validate_url_or_raise(url, allowed_hosts)
