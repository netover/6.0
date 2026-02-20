import os
import subprocess
import zipfile


def run_tool(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr


def run_ruff_fix(target_dir):
    print("Running Ruff safe fixes...")
    subprocess.run(
        ["python", "-m", "ruff", "check", "--fix", target_dir],
        capture_output=True,
        text=True,
    )
    print("Running Ruff format...")
    subprocess.run(
        ["python", "-m", "ruff", "format", target_dir], capture_output=True, text=True
    )


def create_zip(source_dir, zip_filename):
    print(f"Creating zip file {zip_filename}...")
    allowed_dirs = ["resync", "tests"]
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root .py files
        for f in os.listdir(source_dir):
            if os.path.isfile(f) and f.endswith(".py"):
                zipf.write(f, f)

        # Add allowed directories
        for d in allowed_dirs:
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    # exclude __pycache__
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith(".pyc"):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=source_dir)
                        zipf.write(file_path, arcname)


def main():
    target_dir = "resync"
    report_file = "analysis_report.txt"
    zip_file = "resync_fixed.zip"

    # 1. Run tools analysis and save to txt
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE ANÁLISE DO CÓDIGO\n")
        f.write("=" * 50 + "\n\n")

        f.write("=== 1. RUFF (Linting & Bugs) ===\n")
        f.write(run_tool(["python", "-m", "ruff", "check", target_dir]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 2. MYPY (Typing) ===\n")
        f.write(run_tool(["python", "-m", "mypy", target_dir]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 3. PYLINT (Code Smells) ===\n")
        f.write(run_tool(["python", "-m", "pylint", target_dir, "--exit-zero"]))
        f.write("\n" + "=" * 50 + "\n\n")

        f.write("=== 4. RADON (Cyclomatic Complexity - Grades C or worse) ===\n")
        f.write(run_tool(["python", "-m", "radon", "cc", target_dir, "-nc"]))
        f.write("\n" + "=" * 50 + "\n\n")

    print(f"Report saved to {report_file}")

    # 2. Run autofixes and format (Ruff is the only reliable auto-fixer here)
    run_ruff_fix(target_dir)

    # 3. Create zip of the project
    create_zip(".", zip_file)
    print(f"Project zipped to {zip_file}")
    print("Done!")


if __name__ == "__main__":
    main()
