from collections import defaultdict
from pathlib import Path

def generate_core_plan():
    report_file = Path("mypy_core_report.txt")
    if not report_file.exists():
        print("Report not found.")
        return

    domain_counts = defaultdict(int)
    
    with open(report_file, "r", encoding="utf-8") as f:
        for line in f:
            if "error:" in line:
                # Format: resync/core/something/file.py:line:col: error: ...
                parts = line.split(":")
                if len(parts) > 0:
                    filepath = parts[0].replace('\\', '/')
                    if filepath.startswith("resync/core/"):
                        # Group by top 3 directories, e.g., resync/core/health, resync/core/cache
                        path_parts = filepath.split('/')
                        if len(path_parts) >= 3:
                            domain = f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}"
                            domain_counts[domain] += 1
                        else:
                            domain_counts["resync/core (root files)"] += 1
    
    # Sort domains by error count (descending)
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    
    plan_content = [
        "# Mypy Core Remediation Task",
        "",
        "## Sub-tasks for `resync/core/`",
        ""
    ]
    
    for i, (domain, count) in enumerate(sorted_domains, 1):
        plan_content.append(f"- [ ] `mypy` for `{domain}/` ({count} errors)")
    
    if not sorted_domains:
        plan_content.append("No errors found! Core is 100% compliant.")
        
    print("\n".join(plan_content))

if __name__ == "__main__":
    generate_core_plan()
