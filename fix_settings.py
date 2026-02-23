import sys

with open('resync/settings.py', 'r') as f:
    lines = f.readlines()

# Remove the appended SMTP config from the end (if it exists outside class)
# The  lines I appended are likely at the very end.
# I'll remove them first.

clean_lines = []
smtp_lines = []
capture_smtp = False

for line in lines:
    if "smtp_enabled: bool = Field" in line:
        capture_smtp = True

    if capture_smtp:
        smtp_lines.append(line)
        # Assuming the block ends at EOF for now as per my previous command
    else:
        clean_lines.append(line)

# Now I need to insert  INSIDE the Settings class.
# I will look for
#  section seems to be near the end of the class.
# Or I can look for  and indentation.

insert_idx = -1
for i, line in enumerate(clean_lines):
    if "# VALIDADORES" in line:
        insert_idx = i - 1 # Insert before Validators section
        break

if insert_idx != -1:
    # Ensure indentation
    indented_smtp = []
    for sline in smtp_lines:
        if not sline.strip():
            indented_smtp.append(sline)
        else:
            # Check if already indented
            if sline.startswith("    "):
                indented_smtp.append(sline)
            else:
                indented_smtp.append("    " + sline)

    final_lines = clean_lines[:insert_idx] + indented_smtp + clean_lines[insert_idx:]

    with open('resync/settings.py', 'w') as f:
        f.writelines(final_lines)
    print("Settings updated successfully.")
else:
    print("Could not find insertion point in Settings class.")
    # Fallback: Just read the file and append properly if class is huge
