"""Executable invariant checks for a workspace.

The project had no operability surface at all: no metrics, no slow-query log,
no way to ask "is this index consistent" or "which release wrote this file".
When recall got slow or started missing, there was nothing to run.

Each check is cheap enough to run on a large workspace and reports an
actionable remedy rather than a bare pass/fail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def worst(self) -> Status:
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "warn" for c in self.checks):
            return "warn"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "status": self.worst,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "remedy": c.remedy}
                for c in self.checks
            ],
        }


async def _sqlite_checks(backend) -> list[Check]:
    from opendb_core.storage._sqlite_txn import BUSY_TIMEOUT_MS
    from opendb_core.storage.sqlite import _MIGRATIONS

    checks: list[Check] = []
    db = backend._db

    async def scalar(sql: str, params=()):
        async with db.execute(sql, params) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    # --- schema version ---
    version = await scalar("PRAGMA user_version")
    target = _MIGRATIONS[-1][0]
    checks.append(Check(
        "schema_version",
        "ok" if version == target else "fail",
        f"user_version={version}, expected {target}",
        "" if version == target else "Reopen the workspace to apply migrations.",
    ))

    # --- durability pragmas ---
    journal = await scalar("PRAGMA journal_mode")
    busy = await scalar("PRAGMA busy_timeout")
    ok = str(journal).lower() == "wal" and int(busy) >= BUSY_TIMEOUT_MS
    checks.append(Check(
        "durability_pragmas",
        "ok" if ok else "warn",
        f"journal_mode={journal}, busy_timeout={busy}ms",
        "" if ok else "Expected WAL and a busy_timeout of at least "
                      f"{BUSY_TIMEOUT_MS}ms.",
    ))

    # --- FTS/base-table agreement: the invariant most likely to rot ---
    for base, fts in (("memories", "memories_fts"), ("pages", "pages_fts")):
        n_base = await scalar(f"SELECT COUNT(*) FROM {base}")
        n_fts = await scalar(f"SELECT COUNT(*) FROM {fts}")
        checks.append(Check(
            f"fts_consistency:{base}",
            "ok" if n_base == n_fts else "fail",
            f"{base}={n_base}, {fts}={n_fts}",
            "" if n_base == n_fts else
            f"{fts} has drifted from {base}. Reindex the workspace to rebuild it.",
        ))

    # --- orphans: rows whose parent file is gone ---
    orphan_pages = await scalar(
        "SELECT COUNT(*) FROM pages p "
        "WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.id = p.file_id)"
    )
    checks.append(Check(
        "orphan_pages",
        "ok" if not orphan_pages else "fail",
        f"{orphan_pages} pages with no parent file",
        "" if not orphan_pages else "Run `opendb index --force` to rebuild.",
    ))

    # --- files stuck mid-ingest ---
    stuck = await scalar("SELECT COUNT(*) FROM files WHERE status = 'processing'")
    checks.append(Check(
        "stuck_ingestions",
        "ok" if not stuck else "warn",
        f"{stuck} files still in 'processing'",
        "" if not stuck else "Re-index them; a crash can leave this state behind.",
    ))

    # --- tokenizer skew ---
    from opendb_core.utils.tokenizer import tokenizer_fingerprint
    current = tokenizer_fingerprint()
    stored = await scalar(
        "SELECT value FROM opendb_meta WHERE key = 'tokenizer_fingerprint'"
    ) if await scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='opendb_meta'"
    ) else None
    if stored is None:
        checks.append(Check(
            "tokenizer_fingerprint", "warn",
            f"not recorded (current: {current})",
            "Reindex once to record it; skew cannot be detected until then.",
        ))
    else:
        checks.append(Check(
            "tokenizer_fingerprint",
            "ok" if stored == current else "fail",
            f"indexed with {stored}, running {current}",
            "" if stored == current else
            "Tokenization changed since indexing; queries will silently miss "
            "documents. Reindex the workspace.",
        ))

    # --- recall latency, measured not guessed ---
    n_mem = await scalar("SELECT COUNT(*) FROM memories")
    if n_mem:
        t0 = time.perf_counter()
        await backend.recall_memories("deployment configuration service",
                                      None, None, 10, 0)
        ms = (time.perf_counter() - t0) * 1000
        checks.append(Check(
            "recall_latency",
            "ok" if ms < 50 else ("warn" if ms < 250 else "fail"),
            f"{ms:.1f}ms over {n_mem} memories",
            "" if ms < 50 else
            "Recall is slow for this corpus size. Check index bloat with "
            "`VACUUM` and consider pruning faded memories.",
        ))

    # --- storage amplification ---
    import os
    try:
        size = os.path.getsize(backend._db_path)
        per = size / n_mem if n_mem else 0
        checks.append(Check(
            "storage", "ok",
            f"{size / 1e6:.1f} MB on disk"
            + (f", {per:.0f} bytes/memory" if n_mem else ""),
        ))
    except OSError:
        pass

    return checks


async def _postgres_checks(backend) -> list[Check]:
    from opendb_core.database import get_pool
    from opendb_core.storage._pg_migrations import TARGET_VERSION

    checks: list[Check] = []
    pool = await get_pool()
    async with pool.acquire() as conn:
        version = await conn.fetchval(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        )
        checks.append(Check(
            "schema_version",
            "ok" if version == TARGET_VERSION else "fail",
            f"schema_version={version}, expected {TARGET_VERSION}",
            "" if version == TARGET_VERSION else
            "Restart the server to apply outstanding migrations.",
        ))

        orphan_pages = await conn.fetchval(
            "SELECT COUNT(*) FROM pages p "
            "WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.id = p.file_id)"
        )
        checks.append(Check(
            "orphan_pages",
            "ok" if not orphan_pages else "fail",
            f"{orphan_pages} pages with no parent file",
        ))

        stuck = await conn.fetchval(
            "SELECT COUNT(*) FROM files WHERE status = 'processing'"
        )
        checks.append(Check(
            "stuck_ingestions",
            "ok" if not stuck else "warn",
            f"{stuck} files still in 'processing'",
        ))

        # Index bloat is the operator's usual PostgreSQL question.
        rows = await conn.fetch(
            "SELECT indexrelname AS name, pg_relation_size(indexrelid) AS bytes, "
            "idx_scan FROM pg_stat_user_indexes ORDER BY bytes DESC LIMIT 5"
        )
        if rows:
            summary = ", ".join(
                f"{r['name']}={r['bytes'] / 1e6:.1f}MB/{r['idx_scan']} scans"
                for r in rows
            )
            unused = [r["name"] for r in rows if r["idx_scan"] == 0 and r["bytes"] > 1e7]
            checks.append(Check(
                "index_usage",
                "warn" if unused else "ok",
                summary,
                f"Never-scanned indexes over 10MB: {', '.join(unused)}" if unused else "",
            ))

    return checks


async def run_diagnostics(backend) -> Report:
    """Check every invariant this workspace is supposed to hold."""
    from opendb_core.storage.sqlite import SQLiteBackend

    report = Report()
    if isinstance(backend, SQLiteBackend):
        report.checks.extend(await _sqlite_checks(backend))
    else:
        report.checks.extend(await _postgres_checks(backend))
    return report


def format_report(report: Report) -> str:
    """Render a report for a terminal."""
    glyph = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    width = max((len(c.name) for c in report.checks), default=10)
    lines = []
    for c in report.checks:
        lines.append(f"[{glyph[c.status]}] {c.name.ljust(width)}  {c.detail}")
        if c.remedy:
            lines.append(f"{' ' * (width + 9)}-> {c.remedy}")
    lines.append("")
    lines.append(f"overall: {report.worst}")
    return "\n".join(lines)
