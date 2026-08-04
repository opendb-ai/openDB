"""Grep service: regex search across files on the filesystem."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path


async def grep_files(
    query: str,
    path: str,
    glob: str | None = None,
    case_insensitive: bool = False,
    context: int = 0,
    max_results: int = 100,
    per_file_timeout: float = 5.0,
) -> dict:
    """Regex search across files in a directory.

    Returns matching lines with file paths, line numbers, and optional context.
    ``per_file_timeout`` caps how long (seconds) a single file may be read.
    """
    return await asyncio.to_thread(
        _grep_files_sync,
        query, path, glob, case_insensitive, context, max_results,
        per_file_timeout,
    )


_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB — skip files larger than this

# Longest subject string handed to the regex engine, per line.
_MAX_LINE_CHARS = 20_000

# A quantified group that is itself quantified — (a+)+, (a*)*, (a|b)+* — is the
# classic exponential-backtracking shape. Python's `re` has no step limit and
# no timeout, so a single crafted pattern from an agent (the pattern is
# caller-supplied) could pin a CPU indefinitely.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")


def _has_catastrophic_backtracking(pattern: str) -> bool:
    """Cheap structural check for the exponential-backtracking shape."""
    return bool(_NESTED_QUANTIFIER_RE.search(pattern))


def _grep_files_sync(
    query: str,
    path: str,
    glob: str | None,
    case_insensitive: bool,
    context: int,
    max_results: int,
    per_file_timeout: float = 5.0,
) -> dict:
    """Synchronous grep implementation, run in a thread."""
    import time

    root = Path(path)
    if not root.is_dir():
        return {"total": 0, "results": [], "error": f"Directory not found: {path}"}

    flags = re.IGNORECASE if case_insensitive else 0
    if _has_catastrophic_backtracking(query):
        return {
            "total": 0,
            "results": [],
            "error": (
                "Rejected regex: nested quantifiers such as (a+)+ or (a*)* can "
                "take exponential time on non-matching input. Rewrite the "
                "pattern without a quantified group inside a quantifier."
            ),
        }
    try:
        pattern = re.compile(query, flags)
    except re.error as e:
        return {"total": 0, "results": [], "error": f"Invalid regex: {e}"}

    file_pattern = glob or "**/*"
    results: list[dict] = []
    total = 0
    timed_out_files: list[str] = []

    for file_path in root.glob(file_pattern):
        if not file_path.is_file():
            continue

        # Skip binary files and common non-text directories
        rel = str(file_path.relative_to(root)).replace(os.sep, "/")
        if _should_skip(rel):
            continue

        # Skip files larger than threshold
        try:
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        try:
            deadline = time.monotonic() + per_file_timeout
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue

        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Check the deadline every line. It used to be checked once every
            # 5000 lines, which is useless against catastrophic backtracking:
            # the blow-up happens *inside* a single search() call, so the
            # process could sit in line 3 of 4 for minutes without ever
            # reaching the next check.
            if time.monotonic() > deadline:
                timed_out_files.append(rel)
                break

            # A pathological pattern needs a long subject to blow up. Bounding
            # what reaches the engine bounds the worst case per line.
            if not pattern.search(line[:_MAX_LINE_CHARS]):
                continue

            total += 1
            if len(results) >= max_results:
                return {
                    "total": total,
                    "results": results,
                    "truncated": True,
                }

            ctx_before = []
            ctx_after = []
            if context > 0:
                for c in range(max(0, i - context), i):
                    ctx_before.append(lines[c])
                for c in range(i + 1, min(len(lines), i + context + 1)):
                    ctx_after.append(lines[c])

            results.append({
                "file": rel,
                "line": i + 1,  # 1-indexed
                "text": line,
                "context_before": ctx_before,
                "context_after": ctx_after,
            })

    return {
        "total": total,
        "results": results,
        "truncated": False,
        **({"timed_out_files": timed_out_files} if timed_out_files else {}),
    }


# Directories and file patterns to skip
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", ".svelte-kit",
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".pptx",
    ".db", ".sqlite", ".sqlite3",
}


def _should_skip(rel_path: str) -> bool:
    """Check if a file should be skipped based on path patterns."""
    parts = rel_path.split("/")
    for part in parts:
        if part in _SKIP_DIRS:
            return True
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in _SKIP_EXTENSIONS:
        return True
    return False
