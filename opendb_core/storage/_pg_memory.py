"""PostgreSQL memory operations mixin.

Extracted from postgres.py to keep file sizes manageable.
"""

from __future__ import annotations

import json
import logging

from opendb_core.storage.shared import (
    VALID_MEMORY_TYPES,
    build_pg_or_tsquery,
    compute_confidence,
    compute_temporal_score,
    content_token_set,
    describes_distinct_events,
    has_recency_intent,
    jaccard_similarity,
    passes_memory_query_gate,
    supersede_threshold,
)

logger = logging.getLogger(__name__)

# Bounds on the re-ranking candidate window. Mirrors the SQLite backend.
_MIN_CANDIDATES = 60
_MAX_CANDIDATES = 2000


class PgMemoryMixin:
    """Agent Memory operations for PostgresBackend.

    All methods acquire their own connection from the pool via
    ``opendb_core.database.get_pool()``.

    Expects ``self.workspace_id`` (str) to be set by the host class.
    """

    async def _find_conflicting_memory_pg(
        self,
        conn,
        content: str,
        memory_type: str,
        threshold: float = 0.3,
    ) -> tuple[str | None, str | None]:
        """Find an existing PG memory that overlaps significantly.

        Returns (UUID string, UUID string) of the best match, or (None, None).
        Pinned and low-confidence memories are excluded from conflict candidates.

        Candidates are selected *lexically*. This used to take the 20 most
        recently updated same-type memories with no relevance filter at all, so
        on any real store the true best candidate was almost never considered —
        and the 20 arbitrary rows it did consider were all eligible to be
        destroyed.
        """
        new_tokens = content_token_set(content)
        if len(new_tokens) < 2:
            return None, None

        effective_threshold = supersede_threshold(content, threshold)
        tsquery = build_pg_or_tsquery(content)
        if not tsquery:
            return None, None

        rows = await conn.fetch(
            """
            SELECT id, content FROM memories m
            WHERE memory_type = $1 AND workspace_id = $2
              AND pinned = false AND confidence >= 0.3
              AND to_tsvector('english', m.content) @@ to_tsquery('english', $3)
            ORDER BY ts_rank_cd(to_tsvector('english', m.content),
                                to_tsquery('english', $3)) DESC
            LIMIT 50
            """,
            memory_type, self.workspace_id, tsquery,
        )

        best_id: str | None = None
        best_sim = 0.0
        for r in rows:
            # Dated records of different events are never versions of one fact.
            if describes_distinct_events(content, r["content"]):
                continue
            old_tokens = content_token_set(r["content"])
            sim = jaccard_similarity(new_tokens, old_tokens)
            if sim >= effective_threshold and sim > best_sim:
                best_sim = sim
                best_id = str(r["id"])

        return best_id, best_id

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
        import uuid as _uuid
        from opendb_core.database import get_pool
        from opendb_core.utils.tokenizer import expand_identifiers, tokenize_for_fts
        from opendb_core.storage.shared import pg_memory_row as _pg_memory_row

        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {sorted(VALID_MEMORY_TYPES)}, got {memory_type!r}"
            )

        pool = await get_pool()
        async with pool.acquire() as conn:
            # Conflict detection and the write that acts on it run in one
            # transaction. They used to be separate statements, so two
            # concurrent stores of the same fact could both select the same
            # supersede target and one update was lost.
            async with conn.transaction():
                conflict_id = None
                if memory_type != "episodic":
                    conflict_id, _ = await self._find_conflicting_memory_pg(
                        conn, content, memory_type,
                    )

                if conflict_id:
                    # Archive the outgoing version first. Supersede used to be a
                    # bare UPDATE that destroyed the prior content and set
                    # superseded_id = id — a self-reference, not provenance.
                    await conn.execute(
                        """
                        INSERT INTO memory_revisions
                            (memory_id, content, memory_type, tags, metadata,
                             source, workspace_id, superseded_by, valid_from)
                        SELECT id, content, memory_type, tags, metadata,
                               source, workspace_id, $2, updated_at
                        FROM memories WHERE id = $1
                        """,
                        _uuid.UUID(conflict_id), _uuid.UUID(memory_id),
                    )
                    row = await conn.fetchrow(
                        """
                        UPDATE memories
                        SET content = $1, pinned = $2, tags = $3, metadata = $4::jsonb,
                            content_jieba = $5, expansion = $6, source = $7,
                            superseded_id = id, id = $8,
                            confidence = 1.0, last_accessed = NULL, access_count = 0,
                            updated_at = now()
                        WHERE id = $9
                        RETURNING id, content, memory_type, pinned, source,
                                  superseded_id, confidence, tags, metadata,
                                  created_at, updated_at
                        """,
                        content, pinned, tags, json.dumps(metadata),
                        tokenize_for_fts(content), expand_identifiers(content),
                        source, _uuid.UUID(memory_id), _uuid.UUID(conflict_id),
                    )
                else:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO memories (id, content, memory_type, pinned, source,
                                              tags, metadata, content_jieba, expansion,
                                              workspace_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
                        RETURNING id, content, memory_type, pinned, source,
                                  superseded_id, confidence, tags, metadata,
                                  created_at, updated_at
                        """,
                        _uuid.UUID(memory_id), content, memory_type, pinned, source,
                        tags, json.dumps(metadata), tokenize_for_fts(content),
                        expand_identifiers(content), self.workspace_id,
                    )
        return _pg_memory_row(row)

    async def memory_history(self, memory_id: str) -> list[dict]:
        """Every archived prior version of *memory_id*, newest first."""
        import uuid as _uuid
        from opendb_core.database import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id, content, memory_type, tags, metadata, source,
                       superseded_by, valid_from, valid_to
                FROM memory_revisions
                WHERE (memory_id = $1 OR superseded_by = $1) AND workspace_id = $2
                ORDER BY valid_to DESC, id DESC
                """,
                _uuid.UUID(memory_id), self.workspace_id,
            )
        return [
            {
                "memory_id": str(r["memory_id"]),
                "content": r["content"],
                "memory_type": r["memory_type"],
                "tags": list(r["tags"] or []),
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "source": r["source"],
                "superseded_by": str(r["superseded_by"]) if r["superseded_by"] else None,
                "valid_from": r["valid_from"].isoformat() if r["valid_from"] else None,
                "valid_to": r["valid_to"].isoformat() if r["valid_to"] else None,
            }
            for r in rows
        ]

    async def reinforce_memories(self, memory_ids: list[str]) -> None:
        """Record an explicit review of *memory_ids*.

        No longer called from ``recall_memories``. It used to set
        ``confidence = 1.0`` on every hit — making the decay model a no-op for
        anything ever retrieved — and, because the ``memories_updated`` trigger
        fired unconditionally on PostgreSQL, it also reset ``updated_at``, the
        column the time-decay ranking reads. Reading a stale memory made it look
        brand new.
        """
        if not memory_ids:
            return
        import uuid as _uuid
        from opendb_core.database import get_pool

        uuids = [_uuid.UUID(mid) for mid in memory_ids]
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memories SET last_accessed = now(), "
                "access_count = access_count + 1 "
                "WHERE id = ANY($1::uuid[]) AND workspace_id = $2",
                uuids, self.workspace_id,
            )

    async def recall_memories(
        self,
        query: str,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
        pinned_only: bool = False,
    ) -> dict:
        from opendb_core.database import get_pool
        from opendb_core.storage.shared import pg_memory_row as _pg_memory_row

        pool = await get_pool()

        # Fast path: pinned-only retrieval without FTS
        if pinned_only:
            async with pool.acquire() as conn:
                conditions = [
                    "m.pinned = true",
                    "m.workspace_id = $1",
                ]
                params: list = [self.workspace_id]
                idx = 1
                if memory_type:
                    idx += 1
                    conditions.append(f"m.memory_type = ${idx}")
                    params.append(memory_type)
                if tags:
                    idx += 1
                    conditions.append(f"m.tags @> ${idx}::text[]")
                    params.append(tags)
                where = " AND ".join(conditions)
                n = len(params)
                rows = await conn.fetch(
                    f"SELECT m.* FROM memories m WHERE {where} "
                    f"ORDER BY m.created_at DESC LIMIT ${n+1} OFFSET ${n+2}",
                    *params, limit, offset,
                )
                total = await conn.fetchval(
                    f"SELECT COUNT(*) FROM memories m WHERE {where}", *params
                )
            return {"total": total or 0, "results": [
                {**_pg_memory_row(r), "score": 1.0} for r in rows
            ]}

        from opendb_core.utils.tokenizer import _CJK_RE, tokenize_for_fts
        from opendb_core.config import settings

        has_cjk = bool(_CJK_RE.search(query))
        halflife = settings.memory_decay_halflife_days
        stability = settings.memory_stability_days
        threshold = settings.memory_confidence_threshold

        async with pool.acquire() as conn:
            # OR semantics, matching SQLite. `plainto_tsquery` ANDs every lexeme,
            # so the same query against the same data used to return a different
            # result set per backend — and only SQLite was ever benchmarked.
            #
            # The document side is `content` (or the jieba column for CJK)
            # weighted 'A', unioned with the derived `expansion` column weighted
            # 'D'. ts_rank's default weights make D worth 0.1 of A, so a
            # camelCase-split match can surface a memory that would otherwise be
            # unreachable without ever outranking a literal hit.
            if has_cjk:
                cfg = "simple"
                doc_expr = (
                    "setweight(to_tsvector('simple', COALESCE(m.content_jieba, '')), 'A') "
                    "|| setweight(to_tsvector('simple', COALESCE(m.expansion, '')), 'D')"
                )
                headline_src = "COALESCE(m.content_jieba, '')"
                query_param = build_pg_or_tsquery(tokenize_for_fts(query))
            else:
                cfg = "english"
                doc_expr = (
                    "setweight(to_tsvector('english', m.content), 'A') "
                    "|| setweight(to_tsvector('simple', COALESCE(m.expansion, '')), 'D')"
                )
                headline_src = "m.content"
                query_param = build_pg_or_tsquery(query)

            if not query_param:
                return {"total": 0, "ranked": 0, "truncated": False, "results": []}

            fts_cond = f"({doc_expr}) @@ to_tsquery('{cfg}', $1)"
            rank_expr = f"ts_rank_cd({doc_expr}, to_tsquery('{cfg}', $1))"
            headline_expr = (
                f"ts_headline('{cfg}', {headline_src}, "
                f"to_tsquery('{cfg}', $1), 'MaxWords=30, MinWords=10')"
            )

            conditions = [fts_cond, "m.workspace_id = $2"]
            params: list = [query_param, self.workspace_id]
            idx = 2

            if memory_type:
                idx += 1
                conditions.append(f"m.memory_type = ${idx}")
                params.append(memory_type)
            if tags:
                idx += 1
                conditions.append(f"m.tags @> ${idx}::text[]")
                params.append(tags)

            where_clause = " AND ".join(conditions)
            n = len(params)

            # Two fixed window sizes rather than one that grows with `offset`.
            # It used to be max(limit * 3, 60) regardless of offset, so any
            # offset at or beyond 60 returned nothing however many memories
            # matched. Re-ranking happens in Python over the window, so a window
            # that grew per page would also reorder results between pages.
            fetch_limit = _MAX_CANDIDATES if offset + limit > _MIN_CANDIDATES else _MIN_CANDIDATES

            search_sql = f"""
                SELECT m.id, m.content, m.memory_type, m.pinned, m.source,
                       m.superseded_id, m.confidence, m.last_accessed, m.access_count,
                       m.tags, m.metadata, m.created_at, m.updated_at,
                       {rank_expr} AS fts_score,
                       EXTRACT(EPOCH FROM (now() - m.updated_at)) / 86400.0 AS age_days,
                       EXTRACT(EPOCH FROM (now() - COALESCE(m.last_accessed, m.created_at))) / 86400.0 AS days_since_access,
                       {headline_expr} AS highlight
                FROM memories m
                WHERE {where_clause}
                ORDER BY {rank_expr} DESC
                LIMIT ${n + 1}
            """
            count_sql = f"""
                SELECT COUNT(*) FROM memories m WHERE {where_clause}
            """

            rows = await conn.fetch(search_sql, *params, fetch_limit)
            # How many memories actually match, independent of the window.
            matched_total = await conn.fetchval(count_sql, *params) or 0

        recency = has_recency_intent(query)
        scored = []
        for r in rows:
            fts_score = float(r["fts_score"]) if r["fts_score"] else 0.0
            db_age = float(r["age_days"]) if r["age_days"] else 0.0
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            days_since = float(r["days_since_access"]) if r["days_since_access"] else 0.0

            live_conf = compute_confidence(
                base_confidence=float(r["confidence"]),
                days_since_last_access=days_since,
                access_count=int(r["access_count"]),
                pinned=bool(r.get("pinned", False)),
                stability=stability,
            )

            if live_conf < threshold:
                continue

            score, eff_age = compute_temporal_score(
                fts_score, db_age, meta, halflife,
                pinned=bool(r.get("pinned", False)), confidence=live_conf,
                recency_intent=recency,
            )
            scored.append({
                "memory_id": str(r["id"]),
                "content": r["content"],
                "memory_type": r["memory_type"],
                "pinned": bool(r.get("pinned", False)),
                "source": r.get("source", "unknown"),
                "superseded_id": str(r["superseded_id"]) if r.get("superseded_id") else None,
                "confidence": round(live_conf, 4),
                "tags": r["tags"],
                "metadata": meta,
                "highlight": r["highlight"] or "",
                "score": score,
                "created_at": r["created_at"].isoformat() + "Z" if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() + "Z" if r["updated_at"] else None,
                "_age_days": eff_age,
                "_fts": fts_score,
            })

        # Rank-aware query gate: never discard a strong lexical hit just
        # because the question and the stored answer share <2 tokens. Keep
        # results close to the best FTS score; gate only the weaker tail.
        # (Mirrors the SQLite backend.)
        if scored:
            best_fts = max(s["_fts"] for s in scored)
            keep_floor = best_fts * 0.15
            scored = [
                s for s in scored
                if s["_fts"] >= keep_floor
                or passes_memory_query_gate(query, str(s["content"]))
            ]

        # Recency tiebreaker
        if len(scored) >= 2:
            max_score = max(s["score"] for s in scored)
            if max_score > 0:
                for s in scored:
                    if s["score"] / max_score > 0.7:
                        recency_bonus = 1.0 + 0.3 * (0.5 ** (s["_age_days"] / 1.0))
                        s["score"] = s["score"] * recency_bonus

        scored.sort(key=lambda x: x["score"], reverse=True)
        for s in scored:
            s.pop("_age_days", None)
            s.pop("_fts", None)
            s["score"] = float(f"{s['score']:.6g}")
        results = scored[offset : offset + limit]

        # `total` used to be overwritten with len(scored) — the candidate window
        # size after gating, not the match count. Recall is also read-only now;
        # reinforcement is the explicit `reinforce_memories()`.
        return {
            "total": matched_total,
            "ranked": len(scored),
            "truncated": matched_total > len(rows),
            "results": results,
        }

    async def get_memory(self, memory_id: str) -> dict | None:
        import uuid as _uuid
        from opendb_core.database import get_pool
        from opendb_core.storage.shared import pg_memory_row as _pg_memory_row

        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, content, memory_type, pinned, source, superseded_id, "
                "confidence, tags, metadata, created_at, updated_at "
                "FROM memories WHERE id = $1 AND workspace_id = $2",
                _uuid.UUID(memory_id), self.workspace_id,
            )
        return _pg_memory_row(row) if row else None

    async def delete_memory(self, memory_id: str) -> bool:
        import uuid as _uuid
        from opendb_core.database import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1 AND workspace_id = $2",
                _uuid.UUID(memory_id), self.workspace_id,
            )
        return result == "DELETE 1"

    async def list_memories(
        self,
        memory_type: str | None,
        tags: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict:
        from opendb_core.database import get_pool
        from opendb_core.storage.shared import pg_memory_row as _pg_memory_row

        pool = await get_pool()
        async with pool.acquire() as conn:
            conditions: list[str] = ["workspace_id = $1"]
            params: list = [self.workspace_id]
            idx = 1

            if memory_type:
                idx += 1
                conditions.append(f"memory_type = ${idx}")
                params.append(memory_type)
            if tags:
                idx += 1
                conditions.append(f"tags @> ${idx}::text[]")
                params.append(tags)

            where_clause = "WHERE " + " AND ".join(conditions)
            n = len(params)

            query = f"""
                SELECT id, content, memory_type, pinned, source, superseded_id,
                       confidence, tags, metadata, created_at, updated_at
                FROM memories
                {where_clause}
                ORDER BY pinned DESC, created_at DESC
                LIMIT ${n + 1} OFFSET ${n + 2}
            """
            count_query = f"SELECT COUNT(*) FROM memories {where_clause}"

            rows = await conn.fetch(query, *params, limit, offset)
            total = await conn.fetchval(count_query, *params)

        return {
            "total": total or 0,
            "memories": [_pg_memory_row(r) for r in rows],
        }
