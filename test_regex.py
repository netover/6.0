import re

# Test various patterns
patterns = [
    (r"(?i)'\s*(?:or|and)\b.*'.*'='", "' OR '1'='1"),
    (r"(?i)(?:\bor\b|\band\b)\s+'[^']+'", "' OR '1'='1"),
    (r"(?i)(?:\bor\b|\band\b)\s+\S+", "' OR '1'='1"),  # Simple: OR followed by any non-space
]

t = "' OR '1'='1"
for p, _ in patterns:
    print(f'Pattern: {p[:50]}... -> Match: {bool(re.search(p, t))}')
