from collections import defaultdict
from pathlib import Path

def generate_plan():
    report_file = Path("mypy_global_report.txt")
    if not report_file.exists():
        print("Report not found.")
        return

    domain_counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    with open(report_file, "r", encoding="utf-8") as f:
        for line in f:
            if "error:" in line:
                # Format: resync/something/file.py:line:col: error: ...
                parts = line.split(":")
                if len(parts) >= 4:
                    filepath = parts[0].replace('\\', '/')
                    if filepath.startswith("resync/"):
                        # Group by top 2 directories, e.g., resync/core, resync/api/routes
                        path_parts = filepath.split('/')
                        if len(path_parts) >= 2:
                            # Let's say resync/core, resync/api, resync/knowledge, resync/services, resync/models
                            domain = f"{path_parts[0]}/{path_parts[1]}"
                            domain_counts[domain] += 1
                            file_counts[filepath] += 1
                else:
                    print(f"DEBUG: Malformed mypy line: {line.strip()}")
    
    # Sort domains by error count (descending)
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    
    plan_content = [
        "# MYPY REMEDIATION PLAN",
        "",
        "This is the strict tracking file for achieving 100% mypy compliance.",
        "",
        "## Domain Groups",
        ""
    ]
    
    for i, (domain, count) in enumerate(sorted_domains, 1):
        plan_content.append(f"- [ ] STEP {i}: Fix `{domain}/` ({count} errors)")
    
    if not sorted_domains:
        plan_content.append("No errors found! We are 100% compliant.")
        
    plan_file = Path("MYPY_REMEDIATION_PLAN.md")
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write("\n".join(plan_content))
        
    print(f"Plan generated with {len(sorted_domains)} steps.")

if __name__ == "__main__":
    generate_plan()
