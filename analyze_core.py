from collections import defaultdict
from pathlib import Path
from typing import Dict

def parse_core_report(report_path: Path) -> Dict[str, int]:
    """Parse mypy report and count errors by core domain."""
    domain_counts = defaultdict(int)
    
    if not report_path.exists():
        return domain_counts

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if "error:" not in line:
                    continue
                
                parts = line.split(":")
                if len(parts) < 4:
                    continue

                filepath = parts[0].replace('\\', '/')
                if not filepath.startswith("resync/core/"):
                    continue

                path_parts = filepath.split('/')
                if len(path_parts) >= 3:
                    # Group by: resync/core/domain
                    domain = f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}"
                    domain_counts[domain] += 1
                else:
                    domain_counts["resync/core (root files)"] += 1
    except Exception as e:
        print(f"Error reading report: {e}")
        
    return domain_counts

def format_core_plan(domain_counts: Dict[str, int]) -> str:
    """Format domain counts into a markdown plan."""
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    
    lines = [
        "# Mypy Core Remediation Task",
        "",
        "## Sub-tasks for `resync/core/`",
        ""
    ]
    
    if not sorted_domains:
        lines.append("No errors found! Core is 100% compliant.")
    else:
        for i, (domain, count) in enumerate(sorted_domains, 1):
            lines.append(f"- [ ] `mypy` for `{domain}/` ({count} errors)")
            
    return "\n".join(lines)

def generate_core_plan():
    """Main entry for generating core remediation plan."""
    report_file = Path("mypy_core_report.txt")
    counts = parse_core_report(report_file)
    
    if not counts and not report_file.exists():
        print("Report not found.")
        return

    plan = format_core_plan(counts)
    print(plan)

if __name__ == "__main__":
    generate_core_plan()
