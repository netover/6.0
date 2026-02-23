import sys

filename = 'resync/core/wiring.py'
with open(filename, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix type hint for tws variable to allow both Mock and Real
    if 'tws = MockTWSClient()' in line:
        new_lines.append('        tws: ITWSClient = MockTWSClient()\n')
    elif 'tws = get_tws_client_singleton()' in line:
        new_lines.append('            tws = get_tws_client_singleton()\n')
    else:
        new_lines.append(line)

with open(filename, 'w') as f:
    f.writelines(new_lines)
