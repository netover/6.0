from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def parse_global_report(report_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Parse global mypy report into domain and file counts."""
    domain_counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    if not report_path.exists():
        return domain_counts, file_counts

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if "error:" not in line:
                    continue
                
                parts = line.split(":")
                if len(parts) >= 4:
                    filepath = parts[0].replace('\\', '/')
                    if filepath.startswith("resync/"):
                        path_parts = filepath.split('/')
                        if len(path_parts) >= 2:
                            domain = f"{path_parts[0]}/{path_parts[1]}"
                            domain_counts[domain] += 1
                            file_counts[filepath] += 1
                else:
                    print(f"DEBUG: Malformed mypy line: {line.strip()}")
    except Exception as e:
        print(f"Error parsing global report: {e}")
        
    return domain_counts, file_counts

def write_remediation_plan(domain_counts: Dict[str, int], output_path: Path):
    """Generate and write the remediation markdown file."""
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    
    content = [
        "# MYPY REMEDIATION PLAN",
        "",
        "This is the strict tracking file for achieving 100% mypy compliance.",
        "",
        "## Domain Groups",
        ""
    ]
    
    if not sorted_domains:
        content.append("No errors found! We are 100% compliant.")
    else:
        for i, (domain, count) in enumerate(sorted_domains, 1):
            content.append(f"- [ ] STEP {i}: Fix `{domain}/` ({count} errors)")
            
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def generate_plan():
    """Main execution flow for generating global plan."""
    report_file = Path("mypy_global_report.txt")
    if not report_file.exists():
        print("Report not found.")
        return

    domain_counts, _ = parse_global_report(report_file)
    write_remediation_plan(domain_counts, Path("MYPY_REMEDIATION_PLAN.md"))
    print(f"Plan generated with {len(domain_counts)} steps.")

if __name__ == "__main__":
    generate_plan()
