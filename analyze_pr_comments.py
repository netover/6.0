import json
from pathlib import Path
import sys
from typing import Any

COMMENTS_FILE = Path("pr_comments.json")


def _safe_stdout_utf8() -> None:
    """Configure stdout to UTF-8 when the runtime supports reconfigure."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _author_login(payload: dict[str, Any]) -> str:
    author = payload.get("author") or {}
    if isinstance(author, dict):
        return str(author.get("login") or "unknown")
    return "unknown"


def _timestamp(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "unknown-date")


def _body(payload: dict[str, Any]) -> str:
    return str(payload.get("body") or "")


def main() -> int:
    _safe_stdout_utf8()

    if not COMMENTS_FILE.exists():
        print(f"Error: file not found: {COMMENTS_FILE}", file=sys.stderr)
        return 1

    try:
        data = json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: expected top-level JSON object", file=sys.stderr)
        return 1

    comments = data.get("comments") or []
    reviews = data.get("reviews") or []

    if not isinstance(comments, list) or not isinstance(reviews, list):
        print("Error: expected 'comments' and 'reviews' to be lists", file=sys.stderr)
        return 1

    print(f"Total Comments: {len(comments)}")
    print(f"Total Reviews: {len(reviews)}")
    print("-" * 50)

    for index, comment in enumerate(comments, start=1):
        if not isinstance(comment, dict):
            continue
        print(
            f"COMMENT #{index} by {_author_login(comment)} "
            f"({_timestamp(comment, 'createdAt')})"
        )
        print(_body(comment))
        print("-" * 50)

    for index, review in enumerate(reviews, start=1):
        if not isinstance(review, dict):
            continue
        print(
            f"REVIEW #{index} by {_author_login(review)} "
            f"({_timestamp(review, 'submittedAt')})"
        )
        print(_body(review))
        print("-" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
