"""Hybrid recall (FTS + local dense vectors) tests.

Skipped automatically when the optional `[hybrid]` deps (sqlite-vec, model2vec)
are not installed, so the default test run is unaffected.
"""
from __future__ import annotations

import uuid

import pytest

from opendb_core import embedding

pytestmark = pytest.mark.skipif(
    not embedding.hybrid_available(),
    reason="hybrid deps (sqlite-vec, model2vec) not installed",
)


@pytest.fixture(autouse=True)
def _restore_retrieval_mode():
    """Keep the global retrieval mode from leaking into other tests."""
    from opendb_core.config import settings
    prev = settings.memory_retrieval_mode
    yield
    settings.memory_retrieval_mode = prev


async def _fresh_backend(tmp_path, mode: str):
    from opendb_core.config import settings
    from opendb_core.storage.sqlite import SQLiteBackend

    settings.memory_retrieval_mode = mode
    backend = SQLiteBackend(db_path=tmp_path / f"{mode}.db")
    await backend.init()
    return backend


SEMANTIC_MEM = "We migrated the frontend from React to Svelte."
PARAPHRASE_Q = "which UI framework does the team use now"  # zero lexical overlap
DISTRACTORS = [
    "The billing service stores invoices in PostgreSQL.",
    "Integration tests run with testcontainers.",
    "Quarterly revenue grew to fifteen million dollars.",
]


@pytest.mark.asyncio
async def test_pure_fts_misses_paraphrase(tmp_path) -> None:
    """Baseline: pure FTS returns nothing for a zero-overlap paraphrase."""
    backend = await _fresh_backend(tmp_path, "fts")
    try:
        assert backend._hybrid is False
        await backend.store_memory(memory_id=str(uuid.uuid4()), content=SEMANTIC_MEM,
                                   memory_type="semantic", tags=[], metadata={})
        res = await backend.recall_memories(query=PARAPHRASE_Q, memory_type=None,
                                            tags=None, limit=5, offset=0)
        assert res["total"] == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hybrid_recovers_paraphrase(tmp_path) -> None:
    """Hybrid surfaces the semantically-correct memory FTS cannot find."""
    backend = await _fresh_backend(tmp_path, "hybrid")
    try:
        assert backend._hybrid is True
        target = str(uuid.uuid4())
        await backend.store_memory(memory_id=target, content=SEMANTIC_MEM,
                                   memory_type="semantic", tags=[], metadata={})
        for d in DISTRACTORS:
            await backend.store_memory(memory_id=str(uuid.uuid4()), content=d,
                                       memory_type="semantic", tags=[], metadata={})
        res = await backend.recall_memories(query=PARAPHRASE_Q, memory_type=None,
                                            tags=None, limit=5, offset=0)
        ids = [r["memory_id"] for r in res["results"]]
        assert target in ids, "hybrid failed to recover the paraphrased memory"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hybrid_preserves_exact_match_top_rank(tmp_path) -> None:
    """FTS-first: an exact keyword hit must stay rank #1 under hybrid."""
    backend = await _fresh_backend(tmp_path, "hybrid")
    try:
        target = str(uuid.uuid4())
        await backend.store_memory(memory_id=target,
                                   content="The deploy uses ArgoCD and cosign signing.",
                                   memory_type="semantic", tags=[], metadata={})
        for d in DISTRACTORS:
            await backend.store_memory(memory_id=str(uuid.uuid4()), content=d,
                                       memory_type="semantic", tags=[], metadata={})
        res = await backend.recall_memories(query="ArgoCD cosign deploy", memory_type=None,
                                            tags=None, limit=5, offset=0)
        assert res["results"], "expected at least the exact match"
        assert res["results"][0]["memory_id"] == target
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_hybrid_delete_removes_vector(tmp_path) -> None:
    """Deleting a memory also drops its vector row (no stale recall)."""
    backend = await _fresh_backend(tmp_path, "hybrid")
    try:
        target = str(uuid.uuid4())
        await backend.store_memory(memory_id=target, content=SEMANTIC_MEM,
                                   memory_type="semantic", tags=[], metadata={})
        assert await backend.delete_memory(target) is True
        res = await backend.recall_memories(query=PARAPHRASE_Q, memory_type=None,
                                            tags=None, limit=5, offset=0)
        assert all(r["memory_id"] != target for r in res["results"])
    finally:
        await backend.close()
