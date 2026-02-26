#!/usr/bin/env bash
set -euo pipefail

# Generate a pinned, hash-optional lock file using pip-tools.
# Requires: pip install pip-tools
#
# Output: requirements.txt (pinned) from requirements.in

pip-compile --resolver=backtracking --output-file requirements.txt requirements.in
