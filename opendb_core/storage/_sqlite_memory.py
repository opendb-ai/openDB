"""SQLite memory operations mixin.

Extracted from sqlite.py to keep file sizes manageable.
"""

from __future__ import annotations

import aiosqlite
import json
import logging
import re

from opendb_core.storage.shared import (
    build_highlight,
    compute_confidence,
    compute_temporal_score,
    content_token_set,
    escape_fts5,
    has_recency_intent,
    jaccard_similarity,
    passes_memory_query_gate,
)

logger = logging.getLogger(__name__)


class SQLiteMemoryMixin:
    """Agent Memory operations for SQLiteBackend.

    Expects ``self._db`` (aiosqlite connection) and ``self._write_lock``
    (asyncio.Lock) to be set by the host class.
    """

    # ------------------------------------------------------------------
    # Conflict detection for knowledge updates
    # ------------------------------------------------------------------

    # Phrases that signal the content is an update to a previous fact.
    _UPDATE_SIGNALS = re.compile(
        r"\b(moved to|changed|switched|updated|no longer|"
        r"instead of|replaced|now (?:use|live|work|prefer)|"
        r"grew to|new role|started a new|"
        r"changed it from|switched to|migrated)\b",
        re.IGNORECASE,
    )

    async def _find_conflicting_memory(
        self,
        content: str,
        memory_type: str,
        threshold: float = 0.4,
    ) -> tuple[int | None, str | None]:
        """Find an existing memory that overlaps significantly with *content*.

        Returns (integer rowid, memory_id string) of the best match whose
        Jaccard token similarity >= *threshold*, or (None, None).

        Pinned memories and low-confidence memories are excluded.
        """
        from opendb_core.utils.tokenizer import tokenize_for_fts

        new_tokens = content_token_set(content)
        if len(new_tokens) < 2:
            return None, None

        # Lower threshold when the content explicitly signals an update
        effective_threshold = threshold
        if self._UPDATE_SIGNALS.search(content):
            effective_threshold = min(threshold, 0.15)

        fts_query = escape_fts5(tokenize_for_fts(content), use_or=True)
        if not fts_query.strip():
            return None, None

        sql = """
            SELECT m.id, m.memory_id, m.content
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND m.memory_type = ?
                  AND m.pinned = 0 AND m.confidence >= 0.3
            LIMIT 10
        """
        try:
            async with self._db.execute(sql, (fts_query, memory_type)) as cur:
                rows = await cur.fetchall()
        except aiosqlite.DatabaseError:
            rows = []

        best_id: int | None = None
        best_mid: str | None = None
        best_sim = 0.0
        for r in rows:
            old_tokens = content_token_set(r["content"])
            sim = jaccard_similarity(new_tokens, old_tokens)
            if sim >= effective_threshold and sim > best_sim:
                best_sim = sim
                best_id = r["id"]
                best_mid = r["memory_id"]

        # Fallback for update-signal content with zero FTS overlap:
        # find the most recent same-type memory and check if the new
        # content looks like a replacement (e.g. address change).
        if best_id is None and self._UPDATE_SIGNALS.search(content):
            fallback_sql = """
                SELECT m.id, m.memory_id, m.content
                FROM memories m
                WHERE m.memory_type = ? AND m.pinned = 0 AND m.confidence >= 0.3
                ORDER BY m.updated_at DESC
                LIMIT 5
            """
            try:
                async with self._db.execute(fallback_sql, (memory_type,)) as cur:
                    fallback_rows = await cur.fetchall()
            except aiosqlite.DatabaseError:
                fallback_rows = []

            for r in fallback_rows:
                old_tokens = content_token_set(r["content"])
                sim = jaccard_similarity(new_tokens, old_tokens)
                # Even very low overlap counts when update signal is present
                if sim >= 0.05 and sim > best_sim:
                    best_sim = sim
                    best_id = r["id"]
                    best_mid = r["memory_id"]

        return best_id, best_mid

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store_memory(
        self,
        *,
        memory_id: str,
        content: str,
        memory_type: str,
        tags: list[str],
        metadata: dict,
        pinned: bool = False,
        source: str = "unknown",
    ) -> dict:
        from opendb_core.utils.tokenizer import tokenize_for_fts

        async with self._write_lock:
            # Check for conflicting existing memory (knowledge-update detection).
            # Skip for episodic memories — they are event records that should
            # never overwrite each other.
            conflict_id = None
            superseded_mid = None
            if memory_type != "episodic":
                conflict_id, superseded_mid = await self._find_conflicting_memory(
                    content, memory_type, threshold=0.3,
                )

            vec_rowid: int | None = None
            await self._db.execute("BEGIN")
            try:
                if conflict_id is not None:
                    # Supersede: update existing memory instead of inserting duplicate.
                    # Record the old memory_id as superseded_id for provenance.
                    await self._db.execute(
                        "UPDATE memories SET content = ?, memory_id = ?, tags = ?, "
                        "metadata = ?, pinned = ?, source = ?, superseded_id = ?, "
                        "confidence = 1.0, last_accessed = NULL, access_count = 0, "
                        "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                        "WHERE id = ?",
                        (content, memory_id, json.dumps(tags), json.dumps(metadata),
                         int(pinned), source, superseded_mid, conflict_id),
                    )
                    await self._db.execute(
                        "UPDATE memories_fts SET content = ? WHERE rowid = ?",
                        (tokenize_for_fts(content), conflict_id),
                    )
                    vec_rowid = conflict_id
                else:
                    # Normal insert
                    await self._db.execute(
                        """
                        INSERT INTO memories (memory_id, content, memory_type, pinned,
                                              source, tags, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (memory_id, content, memory_type, int(pinned),
                         source, json.dumps(tags), json.dumps(metadata)),
                    )
                    async with self._db.execute(
                        "SELECT id FROM memories WHERE memory_id = ?", (memory_id,)
                    ) as cur:
                        row = await cur.fetchone()
                    rowid = row["id"]
                    await self._db.execute(
                        "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                        (rowid, tokenize_for_fts(content)),
                    )
                    vec_rowid = rowid
                await self._db.execute("COMMIT")
            except aiosqlite.DatabaseError:
                await self._db.execute("ROLLBACK")
                raise

            # Hybrid: keep the dense vector index in sync with this memory.
            if getattr(self, "_hybrid", False) and vec_rowid is not None:
                await self._write_memory_vector(vec_rowid, content)

        # Return the stored record
        return await self.get_memory(memory_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Recall reinforcement (fire-and-forget update after recall)
    # ------------------------------------------------------------------

    async def _reinforce_memories(self, memory_ids: list[str]) -> None:
        """Bump confidence/access_count for recalled memories."""
        if not memory_ids:
            return
        try:
            async with self._write_lock:
                placeholders = ",".join("?" for _ in memory_ids)
                await self._db.execute(
                    f"UPDATE memories SET confidence = 1.0, "
                    f"last_accessed = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
                    f"access_count = access_count + 1 "
                    f"WHERE memory_id IN ({placeholders})",
                    memory_ids,
                )
                await self._db.commit()
        except aiosqlite.DatabaseError:
            pass  # Best-effort, don't break recall

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall_memories(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
        pinned_only: bool = False,
    ) -> dict:
        # Fast path: return all pinned memories without FTS search
        if pinned_only:
            return await self._list_pinned(memory_type, tags, limit, offset)

        # Hybrid path: fuse FTS with a dense vector leg via RRF.
        if getattr(self, "_hybrid", False) and query.strip():
            return await self._recall_hybrid(query, memory_type, tags, limit, offset)

        scored = await self._fts_scored(query, memory_type, tags, pool=max(limit * 3, 60))
        return await self._finalize_recall(scored, limit, offset)

    async def _fts_scored(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        pool: int = 60,
    ) -> list[dict]:
        """Pure-FTS scored candidates: sorted, rank-aware-gated, time-decayed.

        Internal fields ``_rid`` / ``_fts`` / ``_age_days`` are retained for
        hybrid fusion and stripped by :meth:`_finalize_recall`.
        """
        from opendb_core.utils.tokenizer import tokenize_for_fts

        fts_query = escape_fts5(tokenize_for_fts(query), use_or=True)

        conditions: list[str] = []
        params: list = []
        if memory_type:
            conditions.append("m.memory_type = ?")
            params.append(memory_type)
        if tags:
            for tag in tags:
                conditions.append("m.tags LIKE ?")
                params.append(f'%"{tag}"%')
        filter_clause = (" AND " + " AND ".join(conditions)) if conditions else ""

        search_sql = f"""
            SELECT m.id AS _rid, m.memory_id, m.content, m.memory_type, m.pinned,
                   m.source, m.superseded_id, m.confidence,
                   m.last_accessed, m.access_count,
                   m.tags, m.metadata, m.created_at, m.updated_at,
                   memories_fts.rank AS fts_rank,
                   julianday('now') - julianday(m.updated_at) AS age_days,
                   julianday('now') - julianday(COALESCE(m.last_accessed, m.created_at)) AS days_since_access
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ?{filter_clause}
            ORDER BY memories_fts.rank
            LIMIT ?
        """
        async with self._db.execute(search_sql, [fts_query, *params, pool]) as cur:
            rows = await cur.fetchall()

        from opendb_core.config import settings
        halflife = settings.memory_decay_halflife_days
        stability = settings.memory_stability_days
        threshold = settings.memory_confidence_threshold
        recency = has_recency_intent(query)

        scored = []
        for r in rows:
            fts_score = abs(float(r["fts_rank"]))
            db_age = float(r["age_days"]) if r["age_days"] else 0.0
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            days_since = float(r["days_since_access"]) if r["days_since_access"] else 0.0

            live_conf = compute_confidence(
                base_confidence=float(r["confidence"]),
                days_since_last_access=days_since,
                access_count=int(r["access_count"]),
                pinned=bool(r["pinned"]),
                stability=stability,
            )
            if live_conf < threshold:
                continue

            score, eff_age = compute_temporal_score(
                fts_score, db_age, meta, halflife,
                pinned=bool(r["pinned"]), confidence=live_conf,
                recency_intent=recency,
            )
            scored.append({
                "memory_id": r["memory_id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "pinned": bool(r["pinned"]),
                "source": r["source"] if r["source"] else "unknown",
                "superseded_id": r["superseded_id"],
                "confidence": round(live_conf, 4),
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": meta,
                "highlight": build_highlight(r["content"], query),
                "score": score,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "_age_days": eff_age,
                "_fts": fts_score,
                "_rid": r["_rid"],
            })

        # Rank-aware query gate: never discard a strong lexical hit, gate only
        # the weak tail.
        if scored:
            best_fts = max(s["_fts"] for s in scored)
            keep_floor = best_fts * 0.15
            scored = [
                s for s in scored
                if s["_fts"] >= keep_floor
                or passes_memory_query_gate(query, str(s["content"]))
            ]

        # Recency tiebreaker: when FTS scores cluster, boost newer memories.
        if len(scored) >= 2:
            max_score = max(s["score"] for s in scored)
            if max_score > 0:
                for s in scored:
                    if s["score"] / max_score > 0.7:
                        age = s["_age_days"]
                        s["score"] = s["score"] * (1.0 + 0.3 * (0.5 ** (age / 1.0)))

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _finalize_recall(
        self, scored: list[dict], limit: int, offset: int
    ) -> dict:
        """Strip internal fields, round scores, slice, and reinforce."""
        total = len(scored)
        for s in scored:
            s.pop("_age_days", None)
            s.pop("_fts", None)
            s.pop("_rid", None)
            s["score"] = float(f"{s['score']:.6g}")
        results = scored[offset : offset + limit]
        await self._reinforce_memories([r["memory_id"] for r in results])
        return {"total": total, "results": results}

    # ------------------------------------------------------------------
    # Hybrid recall (FTS + dense vectors, fused with RRF)
    # ------------------------------------------------------------------

    async def _write_memory_vector(self, rowid: int, content: str) -> None:
        """(Re)write the dense-vector row for a memory. Caller orders locking;
        the connection autocommits (isolation_level=None)."""
        import sqlite_vec
        from opendb_core import embedding

        try:
            vec = embedding.embed_one(content)
        except Exception:
            logger.exception("hybrid: embedding failed for rowid=%s", rowid)
            return
        await self._db.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
        await self._db.execute(
            "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(vec)),
        )

    async def _recall_hybrid(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict:
        """FTS-first hybrid recall.

        The pure-FTS result and its ranking are preserved **exactly** (so
        exact-identifier / keyword queries never regress). The dense vector
        leg only *appends* semantically-related memories that FTS missed
        entirely — recovering paraphrased queries that share no keywords with
        the stored answer. This avoids the classic failure mode where naive
        rank fusion lets vector noise outrank a strong lexical match.
        """
        import sqlite_vec
        from opendb_core import embedding
        from opendb_core.config import settings

        pool = max(limit * 3, 60)
        # 1. FTS leg — untouched ordering/scoring.
        fts_scored = await self._fts_scored(query, memory_type, tags, pool=pool)
        seen = {s["_rid"] for s in fts_scored}

        # 2. Vector leg — semantic candidates FTS did not already return.
        vec_order: list[int] = []
        try:
            qvec = embedding.embed_one(query)
            async with self._db.execute(
                "SELECT rowid AS rid FROM memories_vec "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (sqlite_vec.serialize_float32(qvec), settings.memory_vector_candidates),
            ) as cur:
                for r in await cur.fetchall():
                    if r["rid"] not in seen:
                        vec_order.append(r["rid"])
                        seen.add(r["rid"])
        except Exception:
            logger.exception("hybrid: vector search failed; FTS only this query")

        vec_scored = await self._score_vector_only(vec_order, memory_type, tags, query)

        # 3. FTS hits first, semantic-only additions after (in vector order).
        return await self._finalize_recall(fts_scored + vec_scored, limit, offset)

    async def _score_vector_only(
        self,
        rowids: list[int],
        memory_type: str | None,
        tags: list[str] | None,
        query: str,
    ) -> list[dict]:
        """Build result dicts for vector-only candidates, preserving the input
        (vector-distance) order and applying the same filters/decay as FTS, but
        scored strictly below any FTS hit (they are appended, not re-sorted)."""
        if not rowids:
            return []
        from opendb_core.config import settings

        conditions = ["m.id IN (%s)" % ",".join("?" for _ in rowids)]
        params: list = list(rowids)
        if memory_type:
            conditions.append("m.memory_type = ?")
            params.append(memory_type)
        if tags:
            for tag in tags:
                conditions.append("m.tags LIKE ?")
                params.append(f'%"{tag}"%')
        sql = (
            "SELECT m.id AS _rid, m.memory_id, m.content, m.memory_type, m.pinned, "
            "m.source, m.superseded_id, m.confidence, m.last_accessed, m.access_count, "
            "m.tags, m.metadata, m.created_at, m.updated_at, "
            "julianday('now') - julianday(COALESCE(m.last_accessed, m.created_at)) AS days_since_access "
            f"FROM memories m WHERE {' AND '.join(conditions)}"
        )
        async with self._db.execute(sql, params) as cur:
            rows = {r["_rid"]: r for r in await cur.fetchall()}

        stability = settings.memory_stability_days
        threshold = settings.memory_confidence_threshold
        out = []
        for pos, rid in enumerate(rowids):  # preserve vector-distance order
            r = rows.get(rid)
            if r is None:  # filtered out by type/tag
                continue
            days_since = float(r["days_since_access"]) if r["days_since_access"] else 0.0
            live_conf = compute_confidence(
                base_confidence=float(r["confidence"]),
                days_since_last_access=days_since,
                access_count=int(r["access_count"]),
                pinned=bool(r["pinned"]),
                stability=stability,
            )
            if live_conf < threshold:
                continue
            out.append({
                "memory_id": r["memory_id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "pinned": bool(r["pinned"]),
                "source": r["source"] if r["source"] else "unknown",
                "superseded_id": r["superseded_id"],
                "confidence": round(live_conf, 4),
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "highlight": build_highlight(r["content"], query),
                # cosmetic, strictly-decreasing score below FTS hits; order is
                # already fixed by list position (vector distance).
                "score": round(0.01 * (0.99 ** pos), 8),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return out

    async def _list_pinned(
        self,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict:
        """Return all pinned memories (no FTS search needed)."""
        conditions = ["m.pinned = 1"]
        params: list = []
        if memory_type:
            conditions.append("m.memory_type = ?")
            params.append(memory_type)
        if tags:
            for tag in tags:
                conditions.append("m.tags LIKE ?")
                params.append(f'%"{tag}"%')

        where = " AND ".join(conditions)
        sql = f"""
            SELECT m.memory_id, m.content, m.memory_type, m.pinned,
                   m.source, m.superseded_id, m.confidence,
                   m.tags, m.metadata, m.created_at, m.updated_at
            FROM memories m
            WHERE {where}
            ORDER BY m.created_at DESC
            LIMIT ? OFFSET ?
        """
        count_sql = f"SELECT COUNT(*) FROM memories m WHERE {where}"

        async with self._db.execute(sql, [*params, limit, offset]) as cur:
            rows = await cur.fetchall()
        async with self._db.execute(count_sql, params) as cur:
            total = (await cur.fetchone())[0]

        results = []
        for r in rows:
            results.append({
                "memory_id": r["memory_id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "pinned": True,
                "source": r["source"] if r["source"] else "unknown",
                "superseded_id": r["superseded_id"],
                "confidence": float(r["confidence"]),
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "score": 1.0,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return {"total": total, "results": results}

    async def get_memory(self, memory_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT memory_id, content, memory_type, pinned, source, superseded_id, "
            "confidence, last_accessed, access_count, "
            "tags, metadata, created_at, updated_at "
            "FROM memories WHERE memory_id = ?",
            (memory_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {
            "memory_id": row["memory_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "pinned": bool(row["pinned"]),
            "source": row["source"] if row["source"] else "unknown",
            "superseded_id": row["superseded_id"],
            "confidence": float(row["confidence"]),
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_memory(self, memory_id: str) -> bool:
        async with self._write_lock:
            async with self._db.execute(
                "SELECT id FROM memories WHERE memory_id = ?", (memory_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return False
            rowid = row["id"]
            await self._db.execute(
                "DELETE FROM memories_fts WHERE rowid = ?", (rowid,)
            )
            await self._db.execute(
                "DELETE FROM memories WHERE id = ?", (rowid,)
            )
            if getattr(self, "_hybrid", False):
                await self._db.execute(
                    "DELETE FROM memories_vec WHERE rowid = ?", (rowid,)
                )
            await self._db.commit()
            return True

    async def list_memories(
        self,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict:
        conditions: list[str] = []
        params: list = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        query = f"""
            SELECT memory_id, content, memory_type, pinned, source, superseded_id,
                   confidence, tags, metadata, created_at, updated_at
            FROM memories
            {where_clause}
            ORDER BY pinned DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        count_query = f"SELECT COUNT(*) FROM memories {where_clause}"

        async with self._db.execute(query, [*params, limit, offset]) as cur:
            rows = await cur.fetchall()
        async with self._db.execute(count_query, params) as cur:
            total_row = await cur.fetchone()
        total = total_row[0] if total_row else 0

        memories = []
        for r in rows:
            memories.append({
                "memory_id": r["memory_id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "pinned": bool(r["pinned"]),
                "source": r["source"] if r["source"] else "unknown",
                "superseded_id": r["superseded_id"],
                "confidence": float(r["confidence"]),
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return {"total": total, "memories": memories}
