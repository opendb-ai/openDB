"""SQLite + FTS5 storage backend — zero-dependency embedded mode.

Uses aiosqlite for async access. Requires ``pip install opendb[embedded]``.

Schema notes:
- UUIDs stored as TEXT
- tags stored as JSON text (e.g. '["tag1","tag2"]')
- metadata stored as JSON text
- line_index stored as JSON text (list of int byte offsets)
- pages_fts is a standalone FTS5 virtual table (jieba-tokenized text)
"""

from __future__ import annotations

import aiosqlite
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from opendb_core.storage.shared import (
    build_highlight,
    escape_fts5,
    add_sqlite_filters,
    sqlite_file_row,
)
from opendb_core.storage._sqlite_memory import SQLiteMemoryMixin
from opendb_core.storage._sqlite_txn import SQLiteTxnMixin, apply_connection_pragmas

logger = logging.getLogger(__name__)

# Weight of the derived `expansion` FTS column relative to the literal body.
# Low enough that a document matching only through an identifier split always
# ranks below one that contains the query terms verbatim.
_EXPANSION_WEIGHT = 0.35
_PAGES_BM25 = f"bm25(pages_fts, 1.0, {_EXPANSION_WEIGHT})"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    mime_type   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    checksum    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'processing',
    error_message TEXT,
    tags        TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_files_checksum_ready
    ON files(checksum) WHERE status = 'ready';
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_created ON files(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files(filename);
CREATE INDEX IF NOT EXISTS idx_files_source_path
    ON files(json_extract(metadata, '$.source_path')) WHERE status = 'ready';

CREATE TABLE IF NOT EXISTS file_text (
    file_id     TEXT PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    full_text   TEXT NOT NULL,
    total_lines INTEGER NOT NULL,
    line_index  TEXT NOT NULL DEFAULT '[]',
    toc         TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,
    section_title TEXT,
    content_type  TEXT NOT NULL DEFAULT 'text',
    text          TEXT NOT NULL,
    line_start    INTEGER NOT NULL,
    line_end      INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_pages_file ON pages(file_id, page_number);

-- Two columns: `text` is the literal (tokenized) body, `expansion` holds
-- derived forms — today the space-separated split of camelCase identifiers.
-- Queries weight `expansion` down (see _FTS_WEIGHTS) so a derived match can
-- never outrank a literal one.
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(text, expansion);

CREATE TRIGGER IF NOT EXISTS files_updated AFTER UPDATE ON files
    WHEN old.updated_at = new.updated_at
BEGIN
    UPDATE files SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = new.id;
END;

-- -----------------------------------------------------------------
-- Agent Memory
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL UNIQUE,
    content       TEXT NOT NULL,
    memory_type   TEXT NOT NULL DEFAULT 'semantic'
                  CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    pinned        INTEGER NOT NULL DEFAULT 0,
    source        TEXT NOT NULL DEFAULT 'unknown',
    superseded_id TEXT DEFAULT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    last_accessed TEXT DEFAULT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    tags          TEXT NOT NULL DEFAULT '[]',
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, expansion);

-- Append-only history of superseded memory content.
--
-- Supersede used to be an in-place `UPDATE memories SET content = ?`, which
-- destroyed the prior fact outright and left `superseded_id` pointing at a row
-- that no longer existed. Every version is now retained here, so
-- `memory_history()` can answer "what did this say before?" and a wrong
-- supersede is recoverable instead of terminal.
CREATE TABLE IF NOT EXISTS memory_revisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id      TEXT NOT NULL,
    content        TEXT NOT NULL,
    memory_type    TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '[]',
    metadata       TEXT NOT NULL DEFAULT '{}',
    source         TEXT NOT NULL DEFAULT 'unknown',
    superseded_by  TEXT,
    valid_from     TEXT NOT NULL,
    valid_to       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    recorded_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_revisions_mid
    ON memory_revisions(memory_id, valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_memory_revisions_by
    ON memory_revisions(superseded_by);

-- NOTE: there is deliberately no `memories_updated` trigger.
--
-- One used to exist with `WHEN old.updated_at = new.updated_at`, which meant
-- *any* UPDATE that did not name `updated_at` silently reset it to now(). The
-- recall path's reinforcement UPDATE did exactly that, and `updated_at` is the
-- column the time-decay ranking reads — so reading a memory reset its age to
-- zero and a stale fact would outrank a fresh one. `updated_at` is now set
-- explicitly by the code paths that genuinely modify a fact.

-- Small key/value table for durable facts about the index itself.
-- `tokenizer_fingerprint` is what lets `opendb doctor` detect that text was
-- indexed under different tokenization rules than the ones now in effect —
-- previously that skew was undetectable and presented only as silently
-- degraded recall.
CREATE TABLE IF NOT EXISTS opendb_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- -----------------------------------------------------------------
-- Eval capture (opt-in)
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_captures (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name     TEXT NOT NULL CHECK (tool_name IN ('search', 'memory_recall')),
    query         TEXT NOT NULL CHECK (length(query) <= 51200),
    result_ids    TEXT NOT NULL DEFAULT '[]',
    result_count  INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_captures_created
    ON eval_captures(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_captures_tool
    ON eval_captures(tool_name);

-- -----------------------------------------------------------------
-- Lightweight file links
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS file_links (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    from_file_id  TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_file_id    TEXT REFERENCES files(id) ON DELETE SET NULL,
    target        TEXT NOT NULL,
    link_type     TEXT NOT NULL DEFAULT 'reference',
    context       TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(from_file_id, target, link_type)
);

CREATE INDEX IF NOT EXISTS idx_file_links_from ON file_links(from_file_id);
CREATE INDEX IF NOT EXISTS idx_file_links_to ON file_links(to_file_id);
CREATE INDEX IF NOT EXISTS idx_file_links_target ON file_links(target);

-- -----------------------------------------------------------------
-- Lightweight code symbols
-- -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS code_symbols (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id        TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    start_line     INTEGER NOT NULL,
    end_line       INTEGER NOT NULL,
    signature      TEXT NOT NULL DEFAULT '',
    docstring      TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_file ON code_symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(name);
CREATE INDEX IF NOT EXISTS idx_code_symbols_kind ON code_symbols(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS code_symbols_fts USING fts5(
    name,
    qualified_name,
    signature,
    docstring
);

CREATE TRIGGER IF NOT EXISTS code_symbols_ai AFTER INSERT ON code_symbols BEGIN
    INSERT INTO code_symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (NEW.id, NEW.name, NEW.qualified_name, NEW.signature, NEW.docstring);
END;

CREATE TRIGGER IF NOT EXISTS code_symbols_ad AFTER DELETE ON code_symbols BEGIN
    DELETE FROM code_symbols_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS code_symbols_au AFTER UPDATE ON code_symbols BEGIN
    DELETE FROM code_symbols_fts WHERE rowid = OLD.id;
    INSERT INTO code_symbols_fts(rowid, name, qualified_name, signature, docstring)
    VALUES (NEW.id, NEW.name, NEW.qualified_name, NEW.signature, NEW.docstring);
END;
"""


class SQLiteBackend(SQLiteTxnMixin, SQLiteMemoryMixin):
    """SQLite + FTS5 implementation of StorageBackend.

    Usage::

        backend = SQLiteBackend(db_path=".opendb/metadata.db")
        await backend.init()
        ...
        await backend.close()
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db = None   # reader connection (aiosqlite.Connection)
        self._wdb = None  # writer connection — only ever used inside write_txn()
        self._write_lock = asyncio.Lock()

    async def init(self) -> None:
        try:
            import aiosqlite
        except ImportError:
            raise RuntimeError(
                "aiosqlite is required for embedded mode. "
                "Install it with: pip install opendb[embedded]"
            )

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Two connections. Readers get a stable WAL snapshot and can therefore
        # never observe a writer's uncommitted rows — with a single shared
        # connection, a SELECT issued while another coroutine held an open
        # transaction ran *inside* that transaction and could return values
        # that were later rolled back.
        self._db = await aiosqlite.connect(str(self._db_path), isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        self._wdb = await aiosqlite.connect(str(self._db_path), isolation_level=None)
        self._wdb.row_factory = aiosqlite.Row

        await apply_connection_pragmas(self._wdb)
        await apply_connection_pragmas(self._db)

        # DDL is idempotent (IF NOT EXISTS) and executescript() implies COMMIT,
        # so it runs outside write_txn().
        await self._wdb.executescript(_SCHEMA)
        await self._run_migrations()

        logger.info("SQLite backend initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # Schema versioning
    # ------------------------------------------------------------------

    async def _user_version(self) -> int:
        async with self._wdb.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def _run_migrations(self) -> None:
        """Apply forward migrations, each atomically with its version bump.

        ``PRAGMA user_version`` is the single source of truth. Previously,
        migrations were ``PRAGMA table_info`` sniffing wrapped in
        ``except (OperationalError, DatabaseError): pass`` — a failed migration
        was indistinguishable from a completed one, and expensive backfills
        re-ran on every single ``init()``.
        """
        current = await self._user_version()
        for version, name, step in _MIGRATIONS:
            if current >= version:
                continue
            logger.info("Applying SQLite migration %d (%s)", version, name)
            async with self.write_txn() as conn:
                await step(self, conn)
                # Same transaction as the step: user_version is never left
                # between two states, even if the process is killed mid-way.
                await conn.execute(f"PRAGMA user_version={version:d}")
            current = version

    async def _backfill_code_symbols_if_needed(self) -> None:
        """Populate code_symbols for ready code files indexed before this table existed.

        Runs once, as migration step 3 — it used to run on every ``init()``,
        re-executing an unbounded ``files ⋈ pages`` join and re-parsing every
        code page on a workspace that had nothing to backfill.
        """
        try:
            async with self._wdb.execute(
                """
                SELECT
                    f.id AS file_id,
                    f.filename,
                    f.metadata,
                    p.page_number,
                    p.text,
                    p.line_start,
                    p.line_end
                FROM files f
                JOIN pages p ON p.file_id = f.id
                WHERE f.status = 'ready'
                  AND NOT EXISTS (
                    SELECT 1 FROM code_symbols s WHERE s.file_id = f.id
                  )
                  AND (
                    lower(f.filename) GLOB '*.py'
                    OR lower(f.filename) GLOB '*.js'
                    OR lower(f.filename) GLOB '*.jsx'
                    OR lower(f.filename) GLOB '*.ts'
                    OR lower(f.filename) GLOB '*.tsx'
                    OR lower(json_extract(f.metadata, '$.source_path')) GLOB '*.py'
                    OR lower(json_extract(f.metadata, '$.source_path')) GLOB '*.js'
                    OR lower(json_extract(f.metadata, '$.source_path')) GLOB '*.jsx'
                    OR lower(json_extract(f.metadata, '$.source_path')) GLOB '*.ts'
                    OR lower(json_extract(f.metadata, '$.source_path')) GLOB '*.tsx'
                  )
                ORDER BY f.id, p.page_number
                """
            ) as cur:
                rows = await cur.fetchall()
        except (aiosqlite.OperationalError, aiosqlite.DatabaseError):
            return
        if not rows:
            return

        from opendb_core.config import settings
        from opendb_core.utils.code_intel import extract_code_intel_from_pages

        current_file_id = None
        group: list[aiosqlite.Row] = []
        backfilled = 0

        async def flush_group(items: list[aiosqlite.Row]) -> None:
            nonlocal backfilled
            if not items:
                return
            first = items[0]
            metadata = json.loads(first["metadata"]) if first["metadata"] else {}
            source_path = metadata.get("source_path") or first["filename"]
            pages = [
                SimpleNamespace(page_number=row["page_number"], text=row["text"])
                for row in items
            ]
            line_ranges = [(row["line_start"], row["line_end"]) for row in items]
            symbols, code_links = extract_code_intel_from_pages(
                pages,
                line_ranges,
                filename=first["filename"],
                source_path=source_path,
            )
            await self._replace_code_symbols_unlocked(first["file_id"], symbols)
            if settings.link_extraction_enabled and code_links:
                await self._insert_file_links_unlocked(first["file_id"], code_links)
            backfilled += 1

        for row in rows:
            if current_file_id is None:
                current_file_id = row["file_id"]
            if row["file_id"] != current_file_id:
                await flush_group(group)
                group = []
                current_file_id = row["file_id"]
            group.append(row)
        await flush_group(group)
        if backfilled:
            logger.info("Backfilled code symbols for %d existing code files.", backfilled)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        if self._wdb:
            await self._wdb.close()
            self._wdb = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def check_duplicate(self, checksum: str) -> dict | None:
        async with self._db.execute(
            "SELECT id, filename FROM files WHERE checksum = ? AND status = 'ready'",
            (checksum,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return {
                "id": row["id"],
                "filename": row["filename"],
                "status": "duplicate",
                "detail": "File with identical content already exists",
            }
        return None

    async def persist_ingestion(
        self,
        *,
        file_id: str,
        file_path: str,
        original_filename: str,
        mime_type: str,
        file_size: int,
        checksum: str,
        tags: list[str],
        merged_metadata: dict,
        parse_result,
        full_text: str,
        total_lines: int,
        line_index: list[int],
        toc: str,
        page_line_ranges: list[tuple[int, int]],
    ) -> dict:
        from opendb_core.config import settings
        from opendb_core.utils.tokenizer import expand_identifiers, tokenize_for_fts
        from opendb_core.utils.code_intel import extract_code_intel_from_pages
        from opendb_core.utils.link_extractor import extract_file_links

        # Pure-CPU work first, outside the transaction: parsing, tokenizing and
        # symbol extraction are the slowest and most failure-prone part of
        # ingestion, and holding the write lock across them starved every other
        # writer for the duration.
        source_path = merged_metadata.get("source_path") or original_filename
        symbols, code_links = extract_code_intel_from_pages(
            parse_result.pages,
            page_line_ranges,
            filename=original_filename,
            source_path=source_path,
        )
        links: list[dict] = []
        if settings.link_extraction_enabled:
            links = extract_file_links(full_text, source_path=source_path)
            links.extend(code_links)

        page_rows = [
            (
                file_id,
                page.page_number,
                page.section_title,
                page.content_type,
                page.text,
                page_line_ranges[i][0],
                page_line_ranges[i][1],
            )
            for i, page in enumerate(parse_result.pages)
        ]

        try:
            # write_txn() rolls back on *any* BaseException. The previous
            # `except aiosqlite.IntegrityError` left the transaction open for
            # every other error class — a parse or tokenizer failure poisoned
            # the shared connection and every later write in the process failed.
            async with self.write_txn() as conn:
                await conn.execute(
                    """
                    INSERT INTO files
                        (id, filename, mime_type, file_size, file_path,
                         checksum, status, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, 'processing', ?, ?)
                    """,
                    (
                        file_id,
                        original_filename,
                        mime_type,
                        file_size,
                        file_path,
                        checksum,
                        json.dumps(tags),
                        json.dumps(merged_metadata),
                    ),
                )

                if page_rows:
                    await conn.executemany(
                        """
                        INSERT INTO pages
                            (file_id, page_number, section_title,
                             content_type, text, line_start, line_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        page_rows,
                    )
                    async with conn.execute(
                        "SELECT id, text FROM pages WHERE file_id = ? ORDER BY page_number",
                        (file_id,),
                    ) as cur:
                        page_id_rows = await cur.fetchall()
                    fts_rows = [
                        (r["id"], tokenize_for_fts(r["text"]), expand_identifiers(r["text"]))
                        for r in page_id_rows
                    ]
                    await conn.executemany(
                        "INSERT INTO pages_fts(rowid, text, expansion) VALUES (?, ?, ?)",
                        fts_rows,
                    )

                await conn.execute(
                    """
                    INSERT INTO file_text
                        (file_id, full_text, total_lines, line_index, toc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        full_text,
                        total_lines,
                        json.dumps(line_index),
                        toc,
                    ),
                )

                await self._replace_code_symbols_unlocked(file_id, symbols)
                if settings.link_extraction_enabled:
                    await self._replace_file_links_unlocked(file_id, links)

                await conn.execute(
                    "UPDATE files SET status = 'ready', metadata = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                    (json.dumps(merged_metadata), file_id),
                )
                if settings.link_extraction_enabled:
                    await self._reconcile_links_to_file_unlocked(file_id)

        except aiosqlite.IntegrityError as exc:
            # Duplicate checksum: another worker won the race for this content.
            if "UNIQUE constraint failed" in str(exc):
                dup = await self.check_duplicate(checksum)
                if dup:
                    return dup
            raise

        return {
            "id": file_id,
            "filename": original_filename,
            "mime_type": mime_type,
            "file_size": file_size,
            "status": "ready",
            "total_pages": len(parse_result.pages),
            "total_lines": total_lines,
            "metadata": merged_metadata,
        }

    async def mark_file_failed(self, file_id: str, error: str) -> None:
        # This is called from a *sibling* ingest worker's exception handler.
        # It used to run unlocked and call commit() on the shared connection,
        # which committed whatever transaction another worker had open.
        async with self.write_txn() as conn:
            await conn.execute(
                "UPDATE files SET status = 'failed', error_message = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (error, file_id),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_file_text(self, file_id: str) -> dict:
        async with self._db.execute(
            "SELECT full_text, total_lines, line_index, toc "
            "FROM file_text WHERE file_id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            from opendb_core.services.read_service import FileNotFoundError
            raise FileNotFoundError(f"No text found for file {file_id}")
        return {
            "full_text": row["full_text"],
            "total_lines": row["total_lines"],
            "line_index": json.loads(row["line_index"]),
            "toc": row["toc"],
        }

    async def get_total_pages(self, file_id: str) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM pages WHERE file_id = ?", (file_id,)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def get_page_line_ranges(
        self, file_id: str, page_numbers: list[int]
    ) -> list[tuple[int, int]]:
        placeholders = ",".join("?" * len(page_numbers))
        async with self._db.execute(
            f"SELECT line_start, line_end FROM pages "
            f"WHERE file_id = ? AND page_number IN ({placeholders}) "
            f"ORDER BY page_number",
            (file_id, *page_numbers),
        ) as cur:
            rows = await cur.fetchall()
        return [(r["line_start"], r["line_end"]) for r in rows]

    async def get_page_by_section_title(
        self, file_id: str, title: str
    ) -> list[int]:
        async with self._db.execute(
            "SELECT page_number FROM pages "
            "WHERE file_id = ? AND section_title LIKE ? "
            "ORDER BY page_number",
            (file_id, f"%{title}%"),
        ) as cur:
            rows = await cur.fetchall()
        return [r["page_number"] for r in rows]

    async def get_file_info(self, file_id: str) -> dict:
        async with self._db.execute(
            "SELECT file_path, mime_type, filename FROM files WHERE id = ?",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            from opendb_core.services.read_service import FileNotFoundError
            raise FileNotFoundError(f"File {file_id} not found")
        return {"file_path": row["file_path"], "mime_type": row["mime_type"], "filename": row["filename"]}

    async def get_sheet_names_for_pages(
        self, file_id: str, page_nums: list[int]
    ) -> list[str]:
        placeholders = ",".join("?" * len(page_nums))
        async with self._db.execute(
            f"SELECT DISTINCT section_title FROM pages "
            f"WHERE file_id = ? AND page_number IN ({placeholders}) "
            f"AND section_title IS NOT NULL",
            (file_id, *page_nums),
        ) as cur:
            rows = await cur.fetchall()
        names: set[str] = set()
        for r in rows:
            title = r["section_title"]
            base = title.split(" - rows ")[0] if " - rows " in title else title
            names.add(base)
        return list(names)

    # ------------------------------------------------------------------
    # Filename resolution
    # ------------------------------------------------------------------

    async def find_file_exact(self, filename: str) -> str | None:
        async with self._db.execute(
            "SELECT id FROM files WHERE filename = ? AND status = 'ready'",
            (filename,),
        ) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None

    async def find_by_source_path(self, source_path: str) -> str | None:
        """Find a file by its original source path (stored in metadata)."""
        async with self._db.execute(
            "SELECT id FROM files "
            "WHERE json_extract(metadata, '$.source_path') = ? AND status = 'ready'",
            (source_path,),
        ) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None

    async def find_file_by_uuid(self, file_id_str: str) -> str | None:
        import uuid as _uuid
        try:
            _uuid.UUID(file_id_str)  # validate format
        except ValueError:
            return None
        async with self._db.execute(
            "SELECT id FROM files WHERE id = ? AND status = 'ready'",
            (file_id_str,),
        ) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None

    async def find_files_fuzzy(self, filename: str) -> list[dict]:
        """Python-side fuzzy matching (no pg_trgm available in SQLite)."""
        from difflib import SequenceMatcher

        async with self._db.execute(
            "SELECT id, filename FROM files WHERE status = 'ready'"
        ) as cur:
            rows = await cur.fetchall()

        scored: list[dict] = []
        fn_lower = filename.lower()
        for r in rows:
            sim = SequenceMatcher(None, fn_lower, r["filename"].lower()).ratio()
            if sim > 0.3:
                scored.append({"id": r["id"], "filename": r["filename"], "sim": sim})

        scored.sort(key=lambda x: x["sim"], reverse=True)
        return scored[:5]

    async def find_files_ilike(self, pattern: str) -> list[dict]:
        async with self._db.execute(
            "SELECT id, filename FROM files "
            "WHERE filename LIKE ? AND status = 'ready'",
            (f"%{pattern}%",),
        ) as cur:
            rows = await cur.fetchall()
        return [{"id": r["id"], "filename": r["filename"]} for r in rows]

    async def find_by_source_path_suffix(self, suffix: str) -> list[dict]:
        norm = suffix.replace("\\", "/").lstrip("/")
        pattern = "%/" + norm
        async with self._db.execute(
            "SELECT id, filename FROM files "
            "WHERE json_extract(metadata, '$.source_path') LIKE ? "
            "AND status = 'ready'",
            (pattern,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"id": r["id"], "filename": r["filename"]} for r in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_fts(
        self, query: str, filters: dict, limit: int, offset: int
    ) -> dict:
        from opendb_core.utils.tokenizer import tokenize_for_fts

        # Tokenize query (handles CJK via jieba) then escape for FTS5
        fts_query = escape_fts5(tokenize_for_fts(query))

        conditions = ["f.status = 'ready'"]
        params: list = []
        add_sqlite_filters(conditions, params, filters)
        filter_clause = (" AND " + " AND ".join(conditions[1:])) if len(conditions) > 1 else ""

        # Join pages table to get original text for Python-side highlighting.
        # bm25() weights the derived `expansion` column at _EXPANSION_WEIGHT so
        # a camelCase-split match can surface a document that would otherwise be
        # unreachable, without ever outranking a literal hit in the body.
        search_sql = f"""
            SELECT f.filename, f.id AS file_id, p.page_number, p.section_title,
                   p.text, {_PAGES_BM25} AS rank, f.updated_at
            FROM pages_fts
            JOIN pages p ON pages_fts.rowid = p.id
            JOIN files f ON p.file_id = f.id
            WHERE pages_fts MATCH ? AND f.status = 'ready'{filter_clause}
            ORDER BY {_PAGES_BM25}
            LIMIT ? OFFSET ?
        """
        count_sql = f"""
            SELECT COUNT(*)
            FROM pages_fts
            JOIN pages p ON pages_fts.rowid = p.id
            JOIN files f ON p.file_id = f.id
            WHERE pages_fts MATCH ? AND f.status = 'ready'{filter_clause}
        """

        search_params = [fts_query, *params, limit, offset]
        count_params = [fts_query, *params]

        async with self._db.execute(search_sql, search_params) as cur:
            rows = await cur.fetchall()
        async with self._db.execute(count_sql, count_params) as cur:
            total_row = await cur.fetchone()
        total = total_row[0] if total_row else 0

        results = []
        for r in rows:
            base_score = abs(float(r["rank"]))
            results.append(
                {
                    "filename": r["filename"],
                    "file_id": r["file_id"],
                    "page_number": r["page_number"],
                    "section_title": r["section_title"],
                    "highlight": build_highlight(r["text"], query),
                    "relevance_score": base_score,
                    "updated_at": r["updated_at"],
                }
            )
        from opendb_core.config import settings
        if settings.backlink_boost_enabled and results:
            counts = await self.get_backlink_counts([r["file_id"] for r in results])
            weight = max(0.0, settings.backlink_boost_weight)
            for r in results:
                count = counts.get(r["file_id"], 0)
                if count:
                    r["relevance_score"] *= 1.0 + weight * count
            results.sort(key=lambda r: r["relevance_score"], reverse=True)
        for r in results:
            r["relevance_score"] = round(float(r["relevance_score"]), 3)
        return {"total": total, "results": results}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def batch_check_duplicates(self, checksums: list[str]) -> set[str]:
        if not checksums:
            return set()
        placeholders = ",".join("?" * len(checksums))
        async with self._db.execute(
            f"SELECT DISTINCT checksum FROM files "
            f"WHERE checksum IN ({placeholders}) AND status = 'ready'",
            checksums,
        ) as cur:
            rows = await cur.fetchall()
        return {r["checksum"] for r in rows}

    # ------------------------------------------------------------------
    # Files CRUD
    # ------------------------------------------------------------------

    async def list_files(
        self,
        filters: dict,
        sort_field: str,
        sort_dir: str,
        limit: int,
        offset: int,
    ) -> dict:
        allowed_sorts = {"created_at", "filename", "file_size"}
        sf = sort_field if sort_field in allowed_sorts else "created_at"
        sd = "ASC" if sort_dir.upper() == "ASC" else "DESC"

        conditions = ["f.status = 'ready'"]
        params: list = []

        if filters.get("tags"):
            # JSON contains — use LIKE for simplicity
            tag = filters["tags"] if isinstance(filters["tags"], str) else filters["tags"][0]
            params.append(f"%{tag}%")
            conditions.append("f.tags LIKE ?")

        if filters.get("mime_type"):
            params.append(filters["mime_type"])
            conditions.append("f.mime_type = ?")

        if filters.get("filename"):
            params.append(f"%{filters['filename']}%")
            conditions.append("f.filename LIKE ?")

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT f.id, f.filename, f.mime_type, f.file_size,
                   f.tags, f.metadata, f.created_at, f.updated_at, f.status,
                   ft.total_lines,
                   (SELECT COUNT(*) FROM pages p WHERE p.file_id = f.id) AS total_pages
            FROM files f
            LEFT JOIN file_text ft ON ft.file_id = f.id
            WHERE {where_clause}
            ORDER BY f.{sf} {sd}
            LIMIT ? OFFSET ?
        """
        count_query = f"SELECT COUNT(*) FROM files f WHERE {where_clause}"

        async with self._db.execute(query, [*params, limit, offset]) as cur:
            rows = await cur.fetchall()
        async with self._db.execute(count_query, params) as cur:
            total_row = await cur.fetchone()
        total = total_row[0] if total_row else 0

        return {"total": total, "files": [sqlite_file_row(r) for r in rows]}

    async def get_file_by_id(self, file_id: str) -> dict | None:
        async with self._db.execute(
            """
            SELECT f.id, f.filename, f.mime_type, f.file_size,
                   f.tags, f.metadata, f.created_at, f.updated_at, f.status,
                   ft.total_lines,
                   (SELECT COUNT(*) FROM pages p WHERE p.file_id = f.id) AS total_pages
            FROM files f
            LEFT JOIN file_text ft ON ft.file_id = f.id
            WHERE f.id = ?
            """,
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        return sqlite_file_row(row) if row else None

    async def delete_file(self, file_id: str) -> str | None:
        async with self._db.execute(
            "SELECT file_path FROM files WHERE id = ?", (file_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        file_path = row["file_path"]
        # Both statements in one transaction: pages_fts is a standalone FTS5
        # table with no foreign key to pages, so a crash between the two used
        # to leave orphaned FTS rows that still matched searches while the
        # underlying file was gone.
        async with self.write_txn() as conn:
            await conn.execute(
                "DELETE FROM pages_fts WHERE rowid IN "
                "(SELECT id FROM pages WHERE file_id = ?)",
                (file_id,),
            )
            await conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return file_path

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_workspace_stats(self) -> dict:
        # File counts by status
        async with self._db.execute(
            "SELECT status, COUNT(*) AS cnt FROM files GROUP BY status"
        ) as cur:
            status_rows = await cur.fetchall()
        by_status = {r["status"]: r["cnt"] for r in status_rows}

        # File counts by MIME type (ready only)
        async with self._db.execute(
            "SELECT mime_type, COUNT(*) AS cnt FROM files "
            "WHERE status = 'ready' GROUP BY mime_type ORDER BY cnt DESC"
        ) as cur:
            type_rows = await cur.fetchall()
        by_type = [(r["mime_type"], r["cnt"]) for r in type_rows]

        # Recently modified files (top 5)
        async with self._db.execute(
            "SELECT filename, updated_at FROM files "
            "WHERE status = 'ready' ORDER BY updated_at DESC LIMIT 5"
        ) as cur:
            recent_rows = await cur.fetchall()
        recent = [
            {"filename": r["filename"], "updated_at": r["updated_at"]}
            for r in recent_rows
        ]

        # Memory stats
        async with self._db.execute(
            "SELECT memory_type, COUNT(*) AS cnt FROM memories GROUP BY memory_type"
        ) as cur:
            mem_rows = await cur.fetchall()
        memory_by_type = {r["memory_type"]: r["cnt"] for r in mem_rows}
        memory_total = sum(memory_by_type.values())

        return {
            "by_status": by_status,
            "by_type": by_type,
            "recent": recent,
            "memory": {
                "total": memory_total,
                "by_type": memory_by_type,
            },
        }

    # ------------------------------------------------------------------
    # Agent Memory — see _sqlite_memory.py (SQLiteMemoryMixin)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Eval capture
    # ------------------------------------------------------------------

    async def log_eval_capture(
        self,
        *,
        tool_name: str,
        query: str,
        result_ids: list[str],
        result_count: int,
        latency_ms: int,
        metadata: dict,
    ) -> None:
        async with self.write_txn() as conn:
            await conn.execute(
                """
                INSERT INTO eval_captures
                    (tool_name, query, result_ids, result_count, latency_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    query[:51200],
                    json.dumps(result_ids),
                    result_count,
                    latency_ms,
                    json.dumps(metadata),
                ),
            )

    async def export_eval_captures(
        self,
        *,
        limit: int = 1000,
        tool_name: str | None = None,
    ) -> list[dict]:
        conditions = []
        params: list = []
        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        async with self._db.execute(
            f"""
            SELECT id, tool_name, query, result_ids, result_count,
                   latency_ms, metadata, created_at
            FROM eval_captures
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "tool_name": r["tool_name"],
                "query": r["query"],
                "result_ids": json.loads(r["result_ids"]) if r["result_ids"] else [],
                "result_count": r["result_count"],
                "latency_ms": r["latency_ms"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # File links
    # ------------------------------------------------------------------

    async def _resolve_link_target(self, target: str) -> str | None:
        async with self._wdb.execute(
            "SELECT id FROM files WHERE json_extract(metadata, '$.source_path') = ? "
            "AND status = 'ready'",
            (target,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row["id"]

        suffix = target.replace("\\", "/").lstrip("/")
        async with self._wdb.execute(
            "SELECT id FROM files WHERE json_extract(metadata, '$.source_path') LIKE ? "
            "AND status = 'ready' ORDER BY length(json_extract(metadata, '$.source_path')) LIMIT 1",
            (f"%/{suffix}",),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row["id"]

        basename = Path(target).name
        if basename:
            async with self._wdb.execute(
                "SELECT id FROM files WHERE filename = ? AND status = 'ready' LIMIT 1",
                (basename,),
            ) as cur:
                row = await cur.fetchone()
            if row:
                return row["id"]
        return None

    async def _replace_file_links_unlocked(self, file_id: str, links: list[dict]) -> int:
        await self._wdb.execute("DELETE FROM file_links WHERE from_file_id = ?", (file_id,))
        return await self._insert_file_links_unlocked(file_id, links)

    async def _insert_file_links_unlocked(self, file_id: str, links: list[dict]) -> int:
        rows = []
        for link in links:
            target = str(link.get("target", "")).strip()
            if not target:
                continue
            rows.append((
                file_id,
                await self._resolve_link_target(target),
                target,
                str(link.get("link_type") or "reference"),
                str(link.get("context") or "")[:500],
            ))
        if rows:
            await self._wdb.executemany(
                """
                INSERT OR IGNORE INTO file_links
                    (from_file_id, to_file_id, target, link_type, context)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    async def _reconcile_links_to_file_unlocked(self, file_id: str) -> None:
        async with self._wdb.execute(
            "SELECT filename, metadata FROM files WHERE id = ? AND status = 'ready'",
            (file_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        source_path = str(metadata.get("source_path") or "").replace("\\", "/")
        filename = row["filename"]
        if not source_path and not filename:
            return
        await self._wdb.execute(
            """
            UPDATE file_links
            SET to_file_id = ?
            WHERE to_file_id IS NULL
              AND (
                target = ?
                OR (? != '' AND ? LIKE '%' || CASE
                    WHEN substr(target, 1, 1) = '/' THEN target
                    ELSE '/' || target
                  END)
                OR target = ?
              )
            """,
            (file_id, source_path, source_path, source_path, filename),
        )

    async def replace_file_links(
        self,
        *,
        file_id: str,
        links: list[dict],
    ) -> int:
        async with self.write_txn():
            count = await self._replace_file_links_unlocked(file_id, links)
        return count

    async def get_backlink_counts(self, file_ids: list[str]) -> dict[str, int]:
        ids = sorted({fid for fid in file_ids if fid})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        async with self._db.execute(
            f"""
            SELECT to_file_id, COUNT(*) AS cnt
            FROM file_links
            WHERE to_file_id IN ({placeholders})
            GROUP BY to_file_id
            """,
            ids,
        ) as cur:
            rows = await cur.fetchall()
        return {r["to_file_id"]: int(r["cnt"]) for r in rows}

    async def _replace_code_symbols_unlocked(
        self,
        file_id: str,
        symbols: list[dict],
    ) -> int:
        await self._wdb.execute("DELETE FROM code_symbols WHERE file_id = ?", (file_id,))

        rows = []
        for symbol in symbols:
            name = str(symbol.get("name") or "").strip()
            if not name:
                continue
            rows.append((
                file_id,
                name,
                str(symbol.get("kind") or "symbol"),
                str(symbol.get("qualified_name") or name),
                int(symbol.get("start_line") or 1),
                int(symbol.get("end_line") or symbol.get("start_line") or 1),
                str(symbol.get("signature") or "")[:1000],
                str(symbol.get("docstring") or "")[:2000],
            ))
        if not rows:
            return 0
        await self._wdb.executemany(
            """
            INSERT INTO code_symbols
                (file_id, name, kind, qualified_name, start_line, end_line, signature, docstring)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    async def search_code_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        kinds: list[str] | None = None,
    ) -> list[dict]:
        from opendb_core.utils.tokenizer import tokenize_for_fts

        match_query = tokenize_for_fts(query)
        params: list[object] = [match_query]
        kind_clause = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_clause = f"AND s.kind IN ({placeholders})"
            params.extend(kinds)
        params.append(limit)
        try:
            async with self._db.execute(
                f"""
                SELECT
                    s.file_id,
                    f.filename,
                    json_extract(f.metadata, '$.source_path') AS source_path,
                    s.name,
                    s.kind,
                    s.qualified_name,
                    s.start_line,
                    s.end_line,
                    s.signature,
                    s.docstring,
                    bm25(code_symbols_fts) AS rank
                FROM code_symbols_fts
                JOIN code_symbols s ON s.id = code_symbols_fts.rowid
                JOIN files f ON f.id = s.file_id
                WHERE code_symbols_fts MATCH ?
                  AND f.status = 'ready'
                  {kind_clause}
                ORDER BY rank
                LIMIT ?
                """,
                params,
            ) as cur:
                rows = await cur.fetchall()
        except aiosqlite.OperationalError:
            rows = []
        if rows:
            return [dict(r) for r in rows]
        else:
            like = f"%{query}%"
            params = [like, like]
            kind_clause = ""
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                kind_clause = f"AND s.kind IN ({placeholders})"
                params.extend(kinds)
            params.append(limit)
            async with self._db.execute(
                f"""
                SELECT
                    s.file_id,
                    f.filename,
                    json_extract(f.metadata, '$.source_path') AS source_path,
                    s.name,
                    s.kind,
                    s.qualified_name,
                    s.start_line,
                    s.end_line,
                    s.signature,
                    s.docstring,
                    0.0 AS rank
                FROM code_symbols s
                JOIN files f ON f.id = s.file_id
                WHERE (s.name LIKE ? OR s.qualified_name LIKE ?)
                  AND f.status = 'ready'
                  {kind_clause}
                ORDER BY length(s.qualified_name), s.start_line
                LIMIT ?
                """,
                params,
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]


# ======================================================================
# Schema migrations
# ======================================================================
#
# Each step is applied inside its own write_txn() together with the
# user_version bump, so the version on disk is never left between two states.
# Steps must be idempotent-safe to *skip*, not idempotent to re-run: the
# version gate guarantees each runs at most once.


async def _m1_memory_columns(backend: "SQLiteBackend", conn) -> None:
    """Provenance / confidence columns on `memories` (was v1.1 + v1.6)."""
    async with conn.execute("PRAGMA table_info(memories)") as cur:
        cols = {row[1] for row in await cur.fetchall()}
    if "memory_id" not in cols:
        return  # fresh DB — _SCHEMA already created the current shape
    for col, ddl in [
        ("pinned", "ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"),
        ("source", "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'"),
        ("superseded_id", "ALTER TABLE memories ADD COLUMN superseded_id TEXT DEFAULT NULL"),
        ("confidence", "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"),
        ("last_accessed", "ALTER TABLE memories ADD COLUMN last_accessed TEXT DEFAULT NULL"),
        ("access_count", "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"),
    ]:
        if col not in cols:
            await conn.execute(ddl)


async def _fts_column_count(conn, table: str) -> int:
    """Number of user columns in an existing FTS5 table, or 0 if absent."""
    async with conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ? AND type = 'table'", (table,)
    ) as cur:
        row = await cur.fetchone()
    if not row or not row[0]:
        return 0
    inner = row[0][row[0].find("(") + 1 : row[0].rfind(")")]
    return len([c for c in inner.split(",") if c.strip()])


async def _m2_two_column_fts(backend: "SQLiteBackend", conn) -> None:
    """Rebuild both FTS tables with the derived `expansion` column.

    Also subsumes the old content-table -> standalone FTS migration: whatever
    shape `pages_fts` had, it is dropped and rebuilt from `pages`, which is the
    authority. FTS content is fully derived, so a rebuild loses nothing.
    """
    from opendb_core.utils.tokenizer import expand_identifiers, tokenize_for_fts

    # Legacy triggers from the content-table era would fire against a table
    # that no longer exists.
    for trig in ("pages_ai", "pages_ad", "pages_au"):
        await conn.execute(f"DROP TRIGGER IF EXISTS {trig}")

    if await _fts_column_count(conn, "pages_fts") != 2:
        await conn.execute("DROP TABLE IF EXISTS pages_fts")
        await conn.execute("CREATE VIRTUAL TABLE pages_fts USING fts5(text, expansion)")
        async with conn.execute("SELECT id, text FROM pages ORDER BY id") as cur:
            rows = await cur.fetchall()
        if rows:
            await conn.executemany(
                "INSERT INTO pages_fts(rowid, text, expansion) VALUES (?, ?, ?)",
                [
                    (r["id"], tokenize_for_fts(r["text"]), expand_identifiers(r["text"]))
                    for r in rows
                ],
            )
        logger.info("Rebuilt pages_fts with identifier expansion (%d pages)", len(rows))

    if await _fts_column_count(conn, "memories_fts") != 2:
        await conn.execute("DROP TABLE IF EXISTS memories_fts")
        await conn.execute(
            "CREATE VIRTUAL TABLE memories_fts USING fts5(content, expansion)"
        )
        async with conn.execute("SELECT id, content FROM memories ORDER BY id") as cur:
            rows = await cur.fetchall()
        if rows:
            await conn.executemany(
                "INSERT INTO memories_fts(rowid, content, expansion) VALUES (?, ?, ?)",
                [
                    (
                        r["id"],
                        tokenize_for_fts(r["content"]),
                        expand_identifiers(r["content"]),
                    )
                    for r in rows
                ],
            )
        logger.info("Rebuilt memories_fts with identifier expansion (%d memories)", len(rows))


async def _m3_backfill_code_symbols(backend: "SQLiteBackend", conn) -> None:
    """One-time code-symbol backfill (used to re-run on every init())."""
    await backend._backfill_code_symbols_if_needed()


async def _m5_record_tokenizer_fingerprint(backend: "SQLiteBackend", conn) -> None:
    """Stamp the tokenization the index was built with."""
    from opendb_core.utils.tokenizer import tokenizer_fingerprint

    await conn.execute(
        "INSERT INTO opendb_meta (key, value) VALUES ('tokenizer_fingerprint', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
        (tokenizer_fingerprint(),),
    )


async def _m4_drop_memories_updated_trigger(backend: "SQLiteBackend", conn) -> None:
    """Remove the trigger that reset the time-decay clock on every read.

    `memories_updated` fired `WHEN old.updated_at = new.updated_at`, so any
    UPDATE that did not name the column silently set it to now(). The recall
    path's reinforcement UPDATE did exactly that, and `updated_at` is what the
    decay ranking reads.
    """
    await conn.execute("DROP TRIGGER IF EXISTS memories_updated")


# (version, name, coroutine). Append only; never renumber.
_MIGRATIONS = [
    (1, "memory_provenance_columns", _m1_memory_columns),
    (2, "two_column_fts_with_identifier_expansion", _m2_two_column_fts),
    (3, "backfill_code_symbols", _m3_backfill_code_symbols),
    (4, "drop_memories_updated_trigger", _m4_drop_memories_updated_trigger),
    (5, "record_tokenizer_fingerprint", _m5_record_tokenizer_fingerprint),
]
