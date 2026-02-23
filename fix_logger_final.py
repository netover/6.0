filename = "resync/core/structured_logger.py"
with open(filename, "r") as f:
    lines = f.readlines()

future_import = [lines[0]] if lines[0].startswith("from __future__") else []
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
    elif stripped.startswith("import ") or stripped.startswith("from "):
        imports.append(line)
    else:
        code.append(line)

with open(filename, "w") as f:
    f.writelines(future_import)
    f.writelines(docstring)
    f.writelines(imports)
    f.writelines(code)
