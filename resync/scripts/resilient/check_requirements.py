#!/usr/bin/env python3
"""
Requirements validation script.

This script checks if the installed packages match the requirements
and identifies potential security issues.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_installed_packages() -> dict[str, str]:
    """Get dict of installed packages {name: version}."""
    # Comando específico e seguro - elimina risco de command injection
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error getting installed packages: {result.stderr}")
        return {}

    try:
        packages = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except json.JSONDecodeError:
        print("Error parsing pip list output")
        return {}


def parse_requirements_file(filepath: str) -> dict[str, str]:
    """Parse requirements file and return {name: version_spec}."""
    requirements = {}
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-r"):
                    # Extract package name and version spec
                    parts = line.split("==")
                    if len(parts) == 2:
                        name = parts[0].lower().split("[")[0]  # Remove extras like [bcrypt]
                        version = parts[1]
                        requirements[name] = version
    except FileNotFoundError:
        print(f"Requirements file not found: {filepath}")
    return requirements


def check_requirements_match(installed: dict[str, str], required: dict[str, str]) -> list[str]:
    """Check if installed versions match requirements."""
    issues = []

    for name, required_version in required.items():
        if name not in installed:
            issues.append(f"[X] MISSING: {name}=={required_version}")
        elif installed[name] != required_version:
            issues.append(
                f"[!] VERSION MISMATCH: {name} (installed: {installed[name]}, required: {required_version})"
            )

    return issues


def check_security_issues() -> list[str]:
    """Check for security issues using safety."""
    issues = []

    # Comando específico e seguro - elimina risco de command injection
    result = subprocess.run(
        [sys.executable, "-m", "safety", "check", "--json"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        try:
            results = json.loads(result.stdout)
            if results:
                issues.append("[!] SECURITY ISSUES FOUND:")
                for issue in results:
                    issues.append(
                        f"   {issue['package']} {issue['vulnerable_spec']} - {issue['advisory']}"
                    )
        except json.JSONDecodeError:
            issues.append("Error parsing safety output")
    else:
        issues.append(f"Safety check failed: {result.stderr}")

    return issues


def main():
    """Main validation function."""
    print("[*] Validating Requirements Installation")
    print("=" * 50)

    # Get installed packages
    installed = get_installed_packages()
    if not installed:
        print("[X] Could not get installed packages")
        return 1

    print(f"[+] Found {len(installed)} installed packages")

    # Check different requirement files
    requirement_files = [
        ("requirements/base.txt", "Base Requirements"),
        ("requirements/dev.txt", "Development Requirements"),
        ("requirements/prod.txt", "Production Requirements"),
    ]

    all_issues = []

    for req_file, description in requirement_files:
        if not Path(req_file).exists():
            print(f"[!] {description} file missing: {req_file}")
            continue

        print(f"\n[*] Checking {description} ({req_file})")

        required = parse_requirements_file(req_file)
        if not required:
            print(f"[!] No requirements found in {req_file}")
            continue

        issues = check_requirements_match(installed, required)
        if issues:
            all_issues.extend(issues)
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"[OK] All {len(required)} {description.lower()} satisfied")

    # Security check
    print("\n[*] Checking Security Issues")
    security_issues = check_security_issues()
    if security_issues:
        all_issues.extend(security_issues)
        for issue in security_issues:
            print(issue)
    else:
        print("[OK] No security issues found")

    # Summary
    print("\n" + "=" * 50)
    if all_issues:
        print(f"[X] Found {len(all_issues)} issues")
        return 1
    print("[OK] All requirements validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
