import sys

filename = 'resync/core/specialists/tools.py'
with open(filename, 'r') as f:
    lines = f.readlines()

future_import = [lines[0]] if lines[0].startswith('from __future__') else []
start_idx = 1 if future_import else 0

docstring = []
imports = []
code = []

in_docstring = False
for i in range(start_idx, len(lines)):
    line = lines[i]
    stripped = line.strip()

    if stripped.startswith('"""') or stripped.startswith("'''"):
        if in_docstring:
            in_docstring = False
            docstring.append(line)
        else:
            in_docstring = True
            docstring.append(line)
    elif in_docstring:
        docstring.append(line)
    elif stripped.startswith('import ') or stripped.startswith('from '):
        imports.append(line)
    else:
        code.append(line)

# Remove unused imports from the list
filtered_imports = []
for imp in imports:
    if 'concurrent.futures' in imp: continue
    if 'import time' in imp: continue
    if 'import uuid' in imp: continue
    if 'datetime.timedelta' in imp: continue
    if 'datetime.timezone' in imp: continue
    if 'pydantic.Field' in imp: continue
    if 'pydantic.ValidationError' in imp: continue
    if 'resync.core.utils.async_bridge' in imp: continue
    filtered_imports.append(imp)

with open(filename, 'w') as f:
    f.writelines(future_import)
    f.writelines(docstring)
    f.writelines(filtered_imports)
    f.writelines(code)
