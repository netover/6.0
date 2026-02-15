import re

with open('workflows/nodes_verbose.py', 'r') as f:
    content = f.read()

# Update fetch_job_history
content = content.replace(
    '                    ORDER BY timestamp DESC\n                    LIMIT 1000',
    f'                    ORDER BY timestamp DESC\n                    LIMIT {_JOB_EXECUTION_HISTORY_LIMIT}', # Wait, I need the value
)
# Ah, I should use the variable name if I'm doing it in python, but I need to make sure the variable is available or just use the literal in the script.
