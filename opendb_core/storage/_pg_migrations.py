"""PostgreSQL schema bootstrap and versioned forward migrations.

Before this module the PostgreSQL backend had no bootstrap at all: `sql/schema.sql`
had to be applied by hand with psql, and `init()` ran a handful of ad-hoc
`information_schema` probes that were indistinguishable from "already migrated"
when they failed. There was no schema version anywhere, so an operator could not
tell which release a database had been written by.

`schema_version` is now the single source of truth, mirroring what
`PRAGMA user_version` does on the SQLite side. Each step runs inside its own
transaction together with its version row, so the recorded version is never left
between two states.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration 1 — base schema (idempotent; supersedes hand-applied sql/schema.sql)
# ---------------------------------------------------------------------------

_BASE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS files (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    file_size       BIGINT NOT NULL,
    file_path       TEXT NOT NULL,
    checksum        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'processing',
    error_message   TEXT,
    tags            TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_tags ON files USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_files_metadata ON files USING GIN(metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_files_filename ON files USING GIN(filename gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_files_created ON files(created_at DESC);

CREATE TABLE IF NOT EXISTS file_text (
    file_id         UUID PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    full_text       TEXT NOT NULL,
    total_lines     INT NOT NULL,
    line_index      INT[] NOT NULL DEFAULT '{}',
    toc             TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id         UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_number     INT NOT NULL,
    section_title   TEXT,
    content_type    TEXT DEFAULT 'text',
    text            TEXT NOT NULL,
    line_start      INT NOT NULL,
    line_end        INT NOT NULL,
    tsv             TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', text)
                    ) STORED,
    text_jieba      TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pages_file ON pages(file_id, page_number);
CREATE INDEX IF NOT EXISTS idx_pages_tsv ON pages USING GIN(tsv);

CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'semantic',
    pinned          BOOLEAN NOT NULL DEFAULT false,
    source          TEXT NOT NULL DEFAULT 'unknown',
    superseded_id   UUID DEFAULT NULL,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    last_accessed   TIMESTAMPTZ DEFAULT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    workspace_id    TEXT NOT NULL DEFAULT '_default',
    content_jieba   TEXT,
    tags            TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);

CREATE TABLE IF NOT EXISTS eval_captures (
    id            BIGSERIAL PRIMARY KEY,
    tool_name     TEXT NOT NULL CHECK (tool_name IN ('search', 'memory_recall')),
    query         TEXT NOT NULL CHECK (length(query) <= 51200),
    result_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
    result_count  INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_captures_created ON eval_captures(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_captures_tool ON eval_captures(tool_name);

CREATE TABLE IF NOT EXISTS file_links (
    id            BIGSERIAL PRIMARY KEY,
    from_file_id  UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    to_file_id    UUID REFERENCES files(id) ON DELETE SET NULL,
    target        TEXT NOT NULL,
    link_type     TEXT NOT NULL DEFAULT 'reference',
    context       TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE(from_file_id, target, link_type)
);

CREATE INDEX IF NOT EXISTS idx_file_links_from ON file_links(from_file_id);
CREATE INDEX IF NOT EXISTS idx_file_links_to ON file_links(to_file_id);
CREATE INDEX IF NOT EXISTS idx_file_links_target ON file_links(target);

CREATE TABLE IF NOT EXISTS code_symbols (
    id              BIGSERIAL PRIMARY KEY,
    file_id         UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,
    qualified_name  TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    signature       TEXT NOT NULL DEFAULT '',
    docstring       TEXT NOT NULL DEFAULT '',
    tsv             TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', name || ' ' || qualified_name || ' ' || signature || ' ' || docstring)
                    ) STORED,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_file ON code_symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(name);
CREATE INDEX IF NOT EXISTS idx_code_symbols_kind ON code_symbols(kind);
CREATE INDEX IF NOT EXISTS idx_code_symbols_tsv ON code_symbols USING GIN(tsv);
"""


async def _m1_base_schema(conn) -> None:
    await conn.execute(_BASE_SCHEMA)


async def _m2_cjk_and_provenance(conn) -> None:
    """Columns added by the old ad-hoc `_migrate_cjk_columns`."""
    stmts = [
        "ALTER TABLE pages ADD COLUMN IF NOT EXISTS text_jieba TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT false",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_jieba TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown'",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_id UUID DEFAULT NULL",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ DEFAULT NULL",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT '_default'",
        "CREATE INDEX IF NOT EXISTS idx_pages_jieba ON pages USING GIN("
        "to_tsvector('simple', COALESCE(text_jieba, '')))",
        "CREATE INDEX IF NOT EXISTS idx_memories_jieba ON memories USING GIN("
        "to_tsvector('simple', COALESCE(content_jieba, '')))",
        "CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(workspace_id)",
        "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence) "
        "WHERE confidence >= 0.3",
    ]
    for s in stmts:
        await conn.execute(s)


async def _m3_no_unconditional_timestamp_trigger(conn) -> None:
    """Drop the trigger that reset the time-decay clock on every UPDATE.

    `memories_updated BEFORE UPDATE ... NEW.updated_at = now()` fired
    unconditionally — worse than the SQLite equivalent, which at least had a
    `WHEN old.updated_at = new.updated_at` guard. `updated_at` is the column the
    time-decay ranking reads, so the recall path's reinforcement UPDATE made a
    stale fact look brand new. `updated_at` is now set explicitly by the code
    paths that genuinely modify a fact.

    The `files` trigger is left in place: nothing ranks files by decay.
    """
    await conn.execute("DROP TRIGGER IF EXISTS memories_updated ON memories")


async def _m4_memory_revisions(conn) -> None:
    """Append-only history so supersede stops destroying the prior fact."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_revisions (
            id             BIGSERIAL PRIMARY KEY,
            memory_id      UUID NOT NULL,
            content        TEXT NOT NULL,
            memory_type    TEXT NOT NULL,
            tags           TEXT[] DEFAULT '{}',
            metadata       JSONB DEFAULT '{}',
            source         TEXT NOT NULL DEFAULT 'unknown',
            workspace_id   TEXT NOT NULL DEFAULT '_default',
            superseded_by  UUID,
            valid_from     TIMESTAMPTZ NOT NULL,
            valid_to       TIMESTAMPTZ NOT NULL DEFAULT now(),
            recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_revisions_mid "
        "ON memory_revisions(memory_id, valid_to DESC)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_revisions_by "
        "ON memory_revisions(superseded_by)"
    )


async def _m5_workspace_scoped_files(conn) -> None:
    """Scope files to a workspace.

    `workspace_id` existed only on `memories`; `files`, `pages`, `file_text`,
    `code_symbols` and `file_links` had none, so two workspaces sharing one
    PostgreSQL database saw each other's documents. Everything else reaches a
    file through `files.id`, so one column plus a join predicate is enough.
    """
    await conn.execute(
        "ALTER TABLE files ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL "
        "DEFAULT '_default'"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_files_workspace ON files(workspace_id)"
    )
    # The uniqueness of a checksum is per workspace, not global: the same file
    # indexed in two workspaces is two rows.
    await conn.execute("DROP INDEX IF EXISTS idx_files_checksum_ready")
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_files_checksum_ready "
        "ON files(workspace_id, checksum) WHERE status = 'ready'"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_files_source_path "
        "ON files((metadata->>'source_path')) WHERE status = 'ready'"
    )


async def _m6_identifier_expansion(conn) -> None:
    """Derived camelCase splits, mirroring the SQLite `expansion` FTS column.

    Weighted 'D' in the combined tsvector — the lowest ts_rank weight — so a
    derived match can surface an otherwise unreachable row without outranking a
    literal one.
    """
    for table, col in (("pages", "text"), ("memories", "content")):
        await conn.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS expansion TEXT"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_expansion ON {table} "
            f"USING GIN(to_tsvector('simple', COALESCE(expansion, '')))"
        )


async def _m7_memory_type_check(conn) -> None:
    """Constrain memory_type — it silently flips storage semantics."""
    await conn.execute(
        "UPDATE memories SET memory_type = 'semantic' "
        "WHERE memory_type NOT IN ('episodic', 'semantic', 'procedural')"
    )
    await conn.execute("ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_type_check")
    await conn.execute(
        "ALTER TABLE memories ADD CONSTRAINT memories_type_check "
        "CHECK (memory_type IN ('episodic', 'semantic', 'procedural'))"
    )


async def _m8_drop_unused_trgm_index(conn) -> None:
    """Drop the GIN trigram index over full page text.

    `idx_pages_trgm` is read by no query in the backend, while a GIN trigram
    index over document bodies is one of the most expensive things you can
    maintain — it commonly exceeds the size of the data it indexes and is paid
    on every insert.
    """
    await conn.execute("DROP INDEX IF EXISTS idx_pages_trgm")


# (version, name, coroutine). Append only; never renumber.
MIGRATIONS = [
    (1, "base_schema", _m1_base_schema),
    (2, "cjk_and_provenance_columns", _m2_cjk_and_provenance),
    (3, "drop_unconditional_timestamp_trigger", _m3_no_unconditional_timestamp_trigger),
    (4, "memory_revisions", _m4_memory_revisions),
    (5, "workspace_scoped_files", _m5_workspace_scoped_files),
    (6, "identifier_expansion", _m6_identifier_expansion),
    (7, "memory_type_check", _m7_memory_type_check),
    (8, "drop_unused_trgm_index", _m8_drop_unused_trgm_index),
]

TARGET_VERSION = MIGRATIONS[-1][0]


async def run_migrations(pool) -> int:
    """Apply every outstanding migration. Returns the resulting version."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        current = await conn.fetchval("SELECT COALESCE(MAX(version), 0) FROM schema_version")

        for version, name, step in MIGRATIONS:
            if current >= version:
                continue
            logger.info("Applying PostgreSQL migration %d (%s)", version, name)
            # Step and version row in one transaction: the recorded version can
            # never disagree with what is actually on disk.
            async with conn.transaction():
                await step(conn)
                await conn.execute(
                    "INSERT INTO schema_version (version, name) VALUES ($1, $2)",
                    version, name,
                )
            current = version

    return current
