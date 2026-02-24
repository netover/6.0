import os
import re
import argparse
from pathlib import Path

def clean_comment_line(line: str) -> tuple[str, bool]:
    """
    Remove everything after 'pylint' or 'mypy' on the same line if they appear in a comment.
    
    Examples:
    '# pylint
    '# mypy
    'x = 1 # some comment' -> 'x = 1 # some comment' (no change)
    '# pylint
    """
    # Regex to match a comment start followed by optional spaces, 
    # then 'pylint' or 'mypy', followed by anything else on the line.
    # We want to keep up to 'pylint' or 'mypy'.
    
    # This pattern captures:
    # 1. Everything before the lint comment start (group 1)
    # 2. The lint keyword itself (group 2)
    # 3. Everything after it (which we will discard)
    pattern = re.compile(r'(#\s*(?:pylint|mypy)).*')
    
    new_line = pattern.sub(r'\1', line)
    
    return new_line, new_line != line

def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return False

    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        # We strip the newline to process the content, then re-add it or use the original line ending pattern
        # Actually, regex with . usually doesn't match \n, so sub works fine if we keep line endings.
        
        # Check if line contains pylint or mypy in a comment
        # Note: If we just want to remove content AFTER the word, we use the regex.
        processed_line, changed = clean_comment_line(line)
        
        if changed:
            # Re-add newline if it was stripped by regex (it usually isn't if we don't use DOTALL)
            # But let's be safe and ensure the line ending is preserved if it existed.
            if line.endswith('\n') and not processed_line.endswith('\n'):
                processed_line += '\n'
            elif line.endswith('\r\n') and not processed_line.endswith('\r\n'):
                processed_line += '\r\n'
                
            new_lines.append(processed_line)
            modified = True
            if verbose:
                print(f"  [{file_path.name}:{i+1}] {line.strip()} -> {processed_line.strip()}")
        else:
            new_lines.append(line)

    if modified:
        if dry_run:
            print(f"Would modify: {file_path}")
        else:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                if verbose:
                    print(f"Modified: {file_path}")
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False
    return modified

def main():
    parser = argparse.ArgumentParser(description="Remove content after 'pylint' and 'mypy' in comments.")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    root = Path(args.path).resolve()
    
    exclude_dirs = {'.git', 'venv', '.venv', '__pycache__', '.mypy_cache', '.pytest_cache'}
    
    count = 0
    modified_count = 0
    
    if root.is_file():
        files = [root]
    else:
        files = root.rglob("*.py")
        
    for py_file in files:
        # Skip excluded directories
        if any(part in exclude_dirs for part in py_file.parts):
            continue
            
        count += 1
        if process_file(py_file, args.dry_run, args.verbose):
            modified_count += 1
            
    print(f"\nFinished processing {count} files.")
    print(f"Modified {modified_count} files.")

if __name__ == "__main__":
    main()
