"""Commit-anchored memories that notice when the code moves under them.

A memory about code has an expiry date nobody records. "Auth lives in
pkg/gateway/middleware/auth.go" is true until someone moves it, and then it is
worse than useless: the agent reads it, believes it, and goes to the wrong file.
Time-decay cannot help — the fact did not get gradually less true, it became
false at a specific commit. Neither can the LLM, which has no way to know the
memory is describing a file that changed last Tuesday.

An anchor records what the memory was true *of*:

* the repository commit at the time it was written,
* the file it describes,
* optionally the symbol, and the hashes of that symbol's signature and body.

On reindex the anchors for the touched files are revalidated, and each one lands
in a state that says something useful:

``current``      nothing changed.
``body_changed`` the symbol still exists with the same signature, but its body
                 was edited. The memory is probably still true; flag it.
``signature_changed`` the declaration changed. A memory about arguments or
                 return type is likely wrong now.
``moved``        the symbol exists, under the same name, in a different file.
                 The memory is true but its location is not; the anchor is
                 repointed and the agent is told where it went.
``missing``      the symbol is gone. The memory is probably stale.

Nothing is deleted. A stale anchor is surfaced to the agent alongside the
memory, because "this may be out of date, the symbol moved to X" is far more
useful than silence — and because a heuristic that silently dropped memories
would be the same class of mistake as the destructive supersede this replaces.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Anchor states, worst-last so `max()` gives the most severe.
STATES = ("current", "body_changed", "signature_changed", "moved", "missing")


@dataclass(frozen=True)
class AnchorCheck:
    memory_id: str
    state: str
    detail: str
    new_path: str | None = None
    new_start_line: int | None = None


async def current_commit(repo_root: Path) -> str | None:
    """HEAD of the repository containing *repo_root*, or None if not a repo.

    One subprocess call, run off the event loop. Deliberately not a full git
    corpus: commits, blame and co-change are a much larger ingestion problem,
    and the anchor only needs to know *which* revision a memory was true at.
    """
    def _run() -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None

    return await asyncio.to_thread(_run)


def classify(
    anchor: dict,
    symbol: dict | None,
    *,
    found_elsewhere: dict | None = None,
) -> tuple[str, str]:
    """Compare a stored anchor against the symbol as it exists now."""
    if symbol is None:
        if found_elsewhere is not None:
            return "moved", (
                f"{anchor['symbol']} is no longer in {anchor['file_path']}; "
                f"it is now in {found_elsewhere['source_path']} "
                f"at line {found_elsewhere['start_line']}"
            )
        return "missing", (
            f"{anchor['symbol']} no longer exists in {anchor['file_path']}"
        )

    if anchor.get("sig_hash") and symbol.get("sig_hash") != anchor["sig_hash"]:
        return "signature_changed", (
            f"the declaration of {anchor['symbol']} changed since this was "
            f"recorded: now `{symbol.get('signature', '')[:160]}`"
        )
    if anchor.get("span_hash") and symbol.get("span_hash") != anchor["span_hash"]:
        return "body_changed", (
            f"the body of {anchor['symbol']} changed since this was recorded; "
            f"the signature is unchanged"
        )
    return "current", ""


class AnchorMixin:
    """Anchor storage and revalidation. Mixed into SQLiteBackend."""

    async def anchor_memory(
        self,
        *,
        memory_id: str,
        file_path: str,
        symbol: str | None = None,
        commit_sha: str | None = None,
        sig_hash: str | None = None,
        span_hash: str | None = None,
    ) -> dict:
        """Record what *memory_id* was true of."""
        async with self.write_txn() as conn:
            await conn.execute(
                """
                INSERT INTO memory_anchors
                    (memory_id, file_path, symbol, commit_sha, sig_hash, span_hash,
                     state, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, 'current',
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                ON CONFLICT(memory_id, file_path, COALESCE(symbol, ''))
                DO UPDATE SET commit_sha = excluded.commit_sha,
                              sig_hash = excluded.sig_hash,
                              span_hash = excluded.span_hash,
                              state = 'current', detail = '',
                              checked_at = excluded.checked_at
                """,
                (memory_id, file_path, symbol, commit_sha, sig_hash, span_hash),
            )
        return {"memory_id": memory_id, "file_path": file_path, "symbol": symbol,
                "commit_sha": commit_sha, "state": "current"}

    async def get_anchors(self, memory_id: str) -> list[dict]:
        async with self._db.execute(
            "SELECT memory_id, file_path, symbol, commit_sha, sig_hash, span_hash, "
            "state, detail, new_path, checked_at "
            "FROM memory_anchors WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def stale_anchors(self, limit: int = 100) -> list[dict]:
        """Anchors that no longer match the code, worst first."""
        async with self._db.execute(
            "SELECT memory_id, file_path, symbol, state, detail, new_path "
            "FROM memory_anchors WHERE state != 'current' "
            "ORDER BY CASE state WHEN 'missing' THEN 0 WHEN 'moved' THEN 1 "
            "WHEN 'signature_changed' THEN 2 ELSE 3 END, id LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def _symbol_now(self, conn, file_path: str, symbol: str) -> dict | None:
        async with conn.execute(
            """
            SELECT s.name, s.qualified_name, s.signature, s.start_line,
                   s.sig_hash, s.span_hash
            FROM code_symbols s
            JOIN files f ON f.id = s.file_id
            WHERE (json_extract(f.metadata, '$.source_path') = ? OR f.filename = ?)
              AND (s.qualified_name = ? OR s.name = ?)
            LIMIT 1
            """,
            (file_path, file_path, symbol, symbol),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def _symbol_anywhere(self, conn, symbol: str) -> dict | None:
        async with conn.execute(
            """
            SELECT s.name, s.qualified_name, s.start_line,
                   COALESCE(json_extract(f.metadata, '$.source_path'), f.filename)
                       AS source_path
            FROM code_symbols s
            JOIN files f ON f.id = s.file_id
            WHERE s.qualified_name = ? OR s.name = ?
            LIMIT 1
            """,
            (symbol, symbol),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def revalidate_anchors(
        self, file_paths: list[str] | None = None
    ) -> list[AnchorCheck]:
        """Re-check anchors against the current index.

        Called after reindexing. Restricting to *file_paths* keeps the cost
        proportional to what actually changed rather than to the store's size.
        """
        params: list = []
        where = ""
        if file_paths:
            where = f"WHERE file_path IN ({','.join('?' for _ in file_paths)})"
            params = list(file_paths)

        async with self._db.execute(
            f"SELECT id, memory_id, file_path, symbol, sig_hash, span_hash "
            f"FROM memory_anchors {where}",
            params,
        ) as cur:
            anchors = [dict(r) for r in await cur.fetchall()]
        if not anchors:
            return []

        checks: list[AnchorCheck] = []
        updates: list[tuple] = []
        for a in anchors:
            if not a["symbol"]:
                # File-level anchor: it is stale only if the file left the index.
                async with self._db.execute(
                    "SELECT 1 FROM files WHERE status = 'ready' "
                    "AND (json_extract(metadata, '$.source_path') = ? OR filename = ?)",
                    (a["file_path"], a["file_path"]),
                ) as cur:
                    present = await cur.fetchone()
                state, detail = (
                    ("current", "") if present
                    else ("missing", f"{a['file_path']} is no longer indexed")
                )
                new_path = None
            else:
                symbol = await self._symbol_now(self._db, a["file_path"], a["symbol"])
                elsewhere = None
                if symbol is None:
                    elsewhere = await self._symbol_anywhere(self._db, a["symbol"])
                state, detail = classify(a, symbol, found_elsewhere=elsewhere)
                new_path = elsewhere["source_path"] if state == "moved" else None

            checks.append(AnchorCheck(a["memory_id"], state, detail, new_path))
            updates.append((state, detail, new_path, a["id"]))

        async with self.write_txn() as conn:
            await conn.executemany(
                "UPDATE memory_anchors SET state = ?, detail = ?, new_path = ?, "
                "checked_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                updates,
            )
        return checks
