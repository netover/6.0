from __future__ import annotations

import re
from pathlib import Path

BAD = [re.compile(r"\btime\.sleep\(")]

def main() -> int:
    bad = []
    for p in Path("resync").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if "async def" not in txt:
            continue
        for rx in BAD:
            if rx.search(txt):
                bad.append((p, rx.pattern))
    if bad:
        for p, pat in bad:
            print(f"Async blocking pattern {pat} in {p}")
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
