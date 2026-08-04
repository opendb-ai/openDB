"""Cross-backend conformance: SQLite and PostgreSQL must agree on semantics.

The project ships two storage backends and benchmarks only one. Before this
suite they disagreed on things as basic as whether a multi-term query means AND
or OR, and every memory-model fix had landed on SQLite alone — so a PostgreSQL
deployment still destroyed data on supersede, reset the time-decay clock on
read, and could not paginate past offset 60.

Every test here runs against **both** backends. PostgreSQL tests are skipped
unless a server is reachable:

    docker run -d --name opendb-pg-test \\
        -e POSTGRES_USER=opendb -e POSTGRES_PASSWORD=opendb \\
        -e POSTGRES_DB=opendb -p 55432:5432 postgres:16
    OPENDB_TEST_PG_DSN=postgresql://opendb:opendb@localhost:55432/opendb pytest
"""

from __future__ import annotations

import os
import uuid

import pytest

from opendb_core.storage.sqlite import SQLiteBackend

PG_DSN = os.environ.get("OPENDB_TEST_PG_DSN")
_pg_reason = "set OPENDB_TEST_PG_DSN to run PostgreSQL conformance tests"


# ----------------------------------------------------------------------
# Backend fixtures — parametrised so each test body runs on both engines
# ----------------------------------------------------------------------

async def _make_sqlite(tmp_path):
    b = SQLiteBackend(db_path=tmp_path / f"conf-{uuid.uuid4().hex}.db")
    await b.init()
    return b, b.close


async def _make_postgres(_tmp_path):
    import asyncpg
    from opendb_core import database as db_mod
    from opendb_core.storage.postgres import PostgresBackend

    # Every test gets its own schema so they cannot see each other's rows.
    schema = f"conf_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(PG_DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()

    pool = await asyncpg.create_pool(
        PG_DSN, min_size=1, max_size=4,
        server_settings={"search_path": f"{schema},public"},
    )
    db_mod.pool = pool

    backend = PostgresBackend(workspace_id="_default")
    await backend.init()

    async def close():
        await pool.close()
        db_mod.pool = None
        admin2 = await asyncpg.connect(PG_DSN)
        await admin2.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin2.close()

    return backend, close


BACKENDS = [
    pytest.param(_make_sqlite, id="sqlite"),
    pytest.param(
        _make_postgres, id="postgres",
        marks=pytest.mark.skipif(not PG_DSN, reason=_pg_reason),
    ),
]


@pytest.fixture(params=BACKENDS)
async def backend(request, tmp_path):
    b, close = await request.param(tmp_path)
    yield b
    await close()


async def _store(b, content, memory_type="semantic", memory_id=None, **kw):
    mid = memory_id or str(uuid.uuid4())
    await b.store_memory(
        memory_id=mid, content=content, memory_type=memory_type,
        tags=kw.pop("tags", []), metadata=kw.pop("metadata", {}),
        pinned=kw.pop("pinned", False), source=kw.pop("source", "user_explicit"),
    )
    return mid


def _ids(result):
    return [r["memory_id"] for r in result["results"]]


# ----------------------------------------------------------------------
# Query semantics
# ----------------------------------------------------------------------

class TestQuerySemantics:
    @pytest.mark.asyncio
    async def test_multi_term_query_is_or_not_and(self, backend) -> None:
        """SQLite ORs its terms; PostgreSQL used plainto_tsquery, which ANDs
        every lexeme. The same query against the same data returned different
        result sets per backend."""
        await _store(backend, "The billing service exposes a webhook endpoint.",
                     memory_type="episodic", memory_id=str(uuid.uuid4()))
        res = await backend.recall_memories(
            "billing kubernetes elasticsearch", None, None, 10, 0
        )
        # "billing" matches; the other two do not exist anywhere. Under AND
        # semantics this returns nothing.
        assert len(res["results"]) == 1

    @pytest.mark.asyncio
    async def test_camel_case_identifier_reachable_by_parts(self, backend) -> None:
        await _store(backend, "CreateInvoice handles PDF rendering.",
                     memory_type="episodic", memory_id="11111111-1111-4111-8111-111111111111")
        for q in ("CreateInvoice", "invoice", "create invoice"):
            res = await backend.recall_memories(q, None, None, 5, 0)
            assert res["results"], f"{q!r} found nothing"

    @pytest.mark.asyncio
    async def test_query_syntax_cannot_reach_the_parser(self, backend) -> None:
        """Operator characters in a caller's query must be data, not syntax."""
        await _store(backend, "A note about deployment topology.",
                     memory_type="episodic")
        for q in ("deployment & topology", "deployment | !topology",
                  "deployment <-> topology", "(deployment", "topology:*)"):
            res = await backend.recall_memories(q, None, None, 5, 0)
            assert isinstance(res["results"], list)


# ----------------------------------------------------------------------
# Supersede semantics
# ----------------------------------------------------------------------

class TestSupersedeConformance:
    @pytest.mark.asyncio
    async def test_distinct_dated_events_are_not_collapsed(self, backend) -> None:
        events = [
            "On 2026-01-10 we fixed a deadlock in the connection pool.",
            "On 2026-02-14 we fixed a memory leak in the connection pool.",
            "On 2026-03-02 we fixed a race condition in the connection pool.",
        ]
        for e in events:
            await _store(backend, e)
        listed = await backend.list_memories(None, None, 100, 0)
        assert listed["total"] == 3

    @pytest.mark.asyncio
    async def test_explicit_correction_supersedes(self, backend) -> None:
        await _store(backend, "The API rate limit is 100 requests per minute.")
        await _store(backend, "The API rate limit is now 200 requests per minute. "
                              "We changed it.")
        listed = await backend.list_memories(None, None, 100, 0)
        assert listed["total"] == 1
        assert "200" in listed["memories"][0]["content"]

    @pytest.mark.asyncio
    async def test_superseded_content_is_recoverable(self, backend) -> None:
        """Supersede was a destructive in-place UPDATE on both backends."""
        new_id = str(uuid.uuid4())
        await _store(backend, "The API rate limit is 100 requests per minute.")
        await _store(backend, "The API rate limit is now 200 requests per minute. "
                              "We changed it.", memory_id=new_id)
        history = await backend.memory_history(new_id)
        assert len(history) == 1
        assert history[0]["content"] == "The API rate limit is 100 requests per minute."

    @pytest.mark.asyncio
    async def test_unrelated_facts_survive_an_update_phrase(self, backend) -> None:
        base = [
            "The payment service listens on port 9090 for internal gRPC traffic.",
            "Our deployment pipeline runs golangci-lint before every merge.",
            "Database migrations live in db/migrations and are applied by Flyway.",
        ]
        for b in base:
            await _store(backend, b)
        await _store(backend, "We switched to Prometheus for metrics collection.")
        listed = await backend.list_memories(None, None, 100, 0)
        assert listed["total"] == 4

    @pytest.mark.asyncio
    async def test_invalid_memory_type_rejected(self, backend) -> None:
        with pytest.raises(ValueError, match="memory_type"):
            await _store(backend, "some fact", memory_type="epsiodic")


# ----------------------------------------------------------------------
# Decay clock and reinforcement
# ----------------------------------------------------------------------

class TestDecayConformance:
    @pytest.mark.asyncio
    async def test_recall_does_not_reset_confidence(self, backend) -> None:
        mid = await _store(backend, "The API endpoint for user auth is /api/v2/auth.")
        await _set_confidence(backend, mid, 0.7)

        await backend.recall_memories("API endpoint user auth", None, None, 5, 0)

        assert await _get_confidence(backend, mid) == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_reinforcement_preserves_the_decay_clock(self, backend) -> None:
        """PostgreSQL's memories_updated trigger fired unconditionally, so any
        UPDATE reset `updated_at` — the column the decay ranking reads."""
        mid = await _store(backend, "The deployment schedule is every Tuesday at 3pm.")
        before = await _get_updated_at(backend, mid)

        await backend.reinforce_memories([mid])

        assert await _get_updated_at(backend, mid) == before


# ----------------------------------------------------------------------
# Counts and pagination
# ----------------------------------------------------------------------

class TestPaginationConformance:
    @pytest.mark.asyncio
    async def test_total_is_the_match_count(self, backend) -> None:
        for i in range(80):
            await _store(backend, f"deployment note {i} about kubernetes cluster a{i}",
                         memory_type="episodic")
        res = await backend.recall_memories("kubernetes cluster deployment",
                                            None, None, 10, 0)
        assert res["total"] == 80
        assert "ranked" in res and "truncated" in res

    @pytest.mark.asyncio
    async def test_pagination_reaches_past_the_first_window(self, backend) -> None:
        for i in range(120):
            await _store(backend, f"deployment note {i} about kubernetes cluster a{i}",
                         memory_type="episodic")
        for offset in (0, 50, 60, 100):
            res = await backend.recall_memories("kubernetes cluster deployment",
                                                None, None, 10, offset)
            assert len(res["results"]) == 10, f"offset {offset} returned nothing"

    @pytest.mark.asyncio
    async def test_pages_do_not_overlap(self, backend) -> None:
        for i in range(120):
            await _store(backend, f"release note {i} for the billing rollout",
                         memory_type="episodic")
        seen = []
        for offset in range(0, 60, 10):
            res = await backend.recall_memories("billing rollout release",
                                                None, None, 10, offset)
            seen.extend(_ids(res))
        assert len(seen) == len(set(seen))


# ----------------------------------------------------------------------
# Workspace isolation
# ----------------------------------------------------------------------

class TestWorkspaceIsolation:
    @pytest.mark.asyncio
    @pytest.mark.skipif(not PG_DSN, reason=_pg_reason)
    async def test_memories_do_not_leak_between_workspaces(self, tmp_path) -> None:
        import asyncpg
        from opendb_core import database as db_mod
        from opendb_core.storage.postgres import PostgresBackend

        schema = f"conf_{uuid.uuid4().hex[:12]}"
        admin = await asyncpg.connect(PG_DSN)
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.close()
        pool = await asyncpg.create_pool(
            PG_DSN, min_size=1, max_size=4,
            server_settings={"search_path": f"{schema},public"},
        )
        db_mod.pool = pool
        try:
            a = PostgresBackend(workspace_id="ws-a")
            await a.init()
            b = PostgresBackend(workspace_id="ws-b")

            await _store(a, "Workspace A holds the billing rotation schedule.")
            res_b = await b.recall_memories("billing rotation schedule",
                                            None, None, 10, 0)
            assert res_b["results"] == [], "workspace A leaked into workspace B"

            res_a = await a.recall_memories("billing rotation schedule",
                                            None, None, 10, 0)
            assert len(res_a["results"]) == 1
        finally:
            await pool.close()
            db_mod.pool = None
            admin2 = await asyncpg.connect(PG_DSN)
            await admin2.execute(f'DROP SCHEMA "{schema}" CASCADE')
            await admin2.close()


# ----------------------------------------------------------------------
# Backend-specific plumbing for the assertions above
# ----------------------------------------------------------------------

def _is_sqlite(b) -> bool:
    return isinstance(b, SQLiteBackend)


async def _set_confidence(b, mid: str, value: float) -> None:
    if _is_sqlite(b):
        await b._wdb.execute(
            "UPDATE memories SET confidence = ? WHERE memory_id = ?", (value, mid)
        )
        await b._wdb.commit()
    else:
        import uuid as _uuid
        from opendb_core.database import get_pool
        pool = await get_pool()
        async with pool.acquire() as c:
            await c.execute("UPDATE memories SET confidence = $1 WHERE id = $2",
                            value, _uuid.UUID(mid))


async def _get_confidence(b, mid: str) -> float:
    if _is_sqlite(b):
        async with b._db.execute(
            "SELECT confidence FROM memories WHERE memory_id = ?", (mid,)
        ) as cur:
            return float((await cur.fetchone())[0])
    import uuid as _uuid
    from opendb_core.database import get_pool
    pool = await get_pool()
    async with pool.acquire() as c:
        return float(await c.fetchval(
            "SELECT confidence FROM memories WHERE id = $1", _uuid.UUID(mid)
        ))


async def _get_updated_at(b, mid: str):
    if _is_sqlite(b):
        async with b._db.execute(
            "SELECT updated_at FROM memories WHERE memory_id = ?", (mid,)
        ) as cur:
            return (await cur.fetchone())[0]
    import uuid as _uuid
    from opendb_core.database import get_pool
    pool = await get_pool()
    async with pool.acquire() as c:
        return await c.fetchval(
            "SELECT updated_at FROM memories WHERE id = $1", _uuid.UUID(mid)
        )
