"""SQLite memory operations mixin.

Extracted from sqlite.py to keep file sizes manageable.
"""

from __future__ import annotations

import aiosqlite
import json
import logging
import re

from opendb_core.storage.ranking import RankingWeights, fuse
from opendb_core.storage.shared import (
    build_highlight,
    compute_confidence,
    compute_temporal_score,
    content_token_set,
    describes_distinct_events,
    escape_fts5,
    has_recency_intent,
    jaccard_similarity,
    passes_memory_query_gate,
)

logger = logging.getLogger(__name__)

# The only labels the storage layer accepts. `memory_type` used to be free
# text, and it silently flips storage semantics 180 degrees — 'episodic' skips
# conflict detection entirely while everything else is a supersede candidate —
# so a typo from the model meant either duplicates or destroyed facts.
VALID_MEMORY_TYPES = frozenset({"episodic", "semantic", "procedural"})

# Down-weight the derived `expansion` column so a camelCase-split match can
# surface an otherwise unreachable memory without outranking a literal hit.
_MEM_BM25 = "bm25(memories_fts, 1.0, 0.35)"

# Bounds on the re-ranking candidate window.
_MIN_CANDIDATES = 60
_MAX_CANDIDATES = 2000

# Jaccard overlap required before one memory supersedes another.
# See _find_conflicting_memory for why these differ by so much.
_SIGNALED_THRESHOLD = 0.15    # content says it is a correction
_UNSIGNALED_THRESHOLD = 0.65  # near-duplicate only


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
        conn=None,
    ) -> tuple[int | None, str | None]:
        """Find an existing memory that overlaps significantly with *content*.

        Returns (integer rowid, memory_id string) of the best match whose
        Jaccard token similarity >= *threshold*, or (None, None).

        Pinned memories and low-confidence memories are excluded.

        Runs on *conn* (the write transaction) when given, so the read that
        selects the supersede candidate and the write that supersedes it are
        one atomic unit — previously the SELECT ran outside the transaction and
        two concurrent stores of the same fact could both decide to supersede
        the same row, losing one of the updates.
        """
        from opendb_core.utils.tokenizer import tokenize_for_fts

        db = conn if conn is not None else self._db

        new_tokens = content_token_set(content)
        if len(new_tokens) < 2:
            return None, None

        # Two thresholds, because the evidence for "this replaces that" is very
        # different in the two cases.
        #
        # With an explicit update phrase ("switched to", "no longer", "migrated")
        # the author is telling us it is a correction, so modest overlap is
        # enough. Without one, only a near-duplicate should supersede: distinct
        # facts about a shared subject routinely reach 0.4-0.6 Jaccard, and at
        # the old flat 0.3 threshold three separate bug-fix records about the
        # same component collapsed into one.
        if self._UPDATE_SIGNALS.search(content):
            effective_threshold = min(threshold, _SIGNALED_THRESHOLD)
        else:
            effective_threshold = max(threshold, _UNSIGNALED_THRESHOLD)

        fts_query = escape_fts5(tokenize_for_fts(content), use_or=True)
        if not fts_query.strip():
            return None, None

        # Order by relevance and widen the window: with LIMIT 10 and no
        # ORDER BY, SQLite returned an arbitrary 10 of the matches, so on a
        # large store the true best candidate was usually not even considered.
        sql = f"""
            SELECT m.id, m.memory_id, m.content
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND m.memory_type = ?
                  AND m.pinned = 0 AND m.confidence >= 0.3
            ORDER BY {_MEM_BM25}
            LIMIT 50
        """
        try:
            async with db.execute(sql, (fts_query, memory_type)) as cur:
                rows = await cur.fetchall()
        except aiosqlite.DatabaseError:
            rows = []

        best_id: int | None = None
        best_mid: str | None = None
        best_sim = 0.0
        for r in rows:
            # Dated records of different events are never versions of one fact.
            if describes_distinct_events(content, r["content"]):
                continue
            old_tokens = content_token_set(r["content"])
            sim = jaccard_similarity(new_tokens, old_tokens)
            if sim >= effective_threshold and sim > best_sim:
                best_sim = sim
                best_id = r["id"]
                best_mid = r["memory_id"]

        # NOTE: there is deliberately no "recent same-type memory" fallback.
        #
        # A previous branch, when the content contained an update-signal phrase
        # and FTS matched nothing, scanned the 5 most-recently-updated
        # same-type memories and destructively overwrote any with Jaccard
        # >= 0.05 — i.e. a single shared token was enough to destroy an
        # unrelated fact. It was also near-unreachable by construction (zero
        # FTS overlap almost always implies zero token overlap), so it bought
        # nothing to offset that risk.

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
        from opendb_core.utils.tokenizer import expand_identifiers, tokenize_for_fts

        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"memory_type must be one of {sorted(VALID_MEMORY_TYPES)}, got {memory_type!r}"
            )

        async with self.write_txn() as conn:
            # Check for conflicting existing memory (knowledge-update detection).
            # Skip for episodic memories — they are event records that should
            # never overwrite each other.
            conflict_id = None
            superseded_mid = None
            if memory_type != "episodic":
                conflict_id, superseded_mid = await self._find_conflicting_memory(
                    content, memory_type, threshold=0.3, conn=conn,
                )

            if conflict_id is not None:
                # Supersede. The previous version is archived to
                # memory_revisions *before* the row is rewritten, so the old
                # fact stays retrievable via memory_history(). Previously this
                # was a bare UPDATE that destroyed the prior content outright
                # and left superseded_id pointing at a row that no longer
                # existed — a dangling pointer, not provenance.
                async with conn.execute(
                    "SELECT memory_id, content, memory_type, tags, metadata, "
                    "source, updated_at, created_at, valid_from "
                    "FROM memories WHERE id = ?",
                    (conflict_id,),
                ) as cur:
                    old = await cur.fetchone()
                if old is not None:
                    # The outgoing fact stopped being true now; the incoming
                    # one starts now. Closing the interval rather than deleting
                    # is what makes "what was true in March" a range query.
                    await conn.execute(
                        """
                        INSERT INTO memory_revisions
                            (memory_id, content, memory_type, tags, metadata,
                             source, superseded_by, valid_from, valid_to)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                                strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                        """,
                        (
                            old["memory_id"], old["content"], old["memory_type"],
                            old["tags"], old["metadata"], old["source"],
                            memory_id, old["valid_from"] or old["created_at"],
                        ),
                    )
                await conn.execute(
                    "UPDATE memories SET content = ?, memory_id = ?, tags = ?, "
                    "metadata = ?, pinned = ?, source = ?, superseded_id = ?, "
                    "confidence = 1.0, last_accessed = NULL, access_count = 0, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
                    "valid_from = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
                    "valid_to = NULL, "
                    "tx_from = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                    "WHERE id = ?",
                    (content, memory_id, json.dumps(tags), json.dumps(metadata),
                     int(pinned), source, superseded_mid, conflict_id),
                )
                await conn.execute(
                    "UPDATE memories_fts SET content = ?, expansion = ? WHERE rowid = ?",
                    (tokenize_for_fts(content), expand_identifiers(content), conflict_id),
                )
            else:
                # Normal insert
                await conn.execute(
                    """
                    INSERT INTO memories (memory_id, content, memory_type, pinned,
                                          source, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, content, memory_type, int(pinned),
                     source, json.dumps(tags), json.dumps(metadata)),
                )
                async with conn.execute(
                    "SELECT id FROM memories WHERE memory_id = ?", (memory_id,)
                ) as cur:
                    row = await cur.fetchone()
                rowid = row["id"]
                await conn.execute(
                    "INSERT INTO memories_fts(rowid, content, expansion) VALUES (?, ?, ?)",
                    (rowid, tokenize_for_fts(content), expand_identifiers(content)),
                )

        # Return the stored record
        return await self.get_memory(memory_id)  # type: ignore[return-value]

    async def memory_history(self, memory_id: str) -> list[dict]:
        """Every archived prior version of *memory_id*, newest first.

        A supersede no longer loses the fact it replaced, so an agent (or an
        operator cleaning up after a bad supersede) can see what a memory used
        to say and when it stopped being current.
        """
        async with self._db.execute(
            """
            SELECT memory_id, content, memory_type, tags, metadata, source,
                   superseded_by, valid_from, valid_to
            FROM memory_revisions
            WHERE memory_id = ? OR superseded_by = ?
            ORDER BY valid_to DESC, id DESC
            """,
            (memory_id, memory_id),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "memory_id": r["memory_id"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "source": r["source"],
                "superseded_by": r["superseded_by"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Recall reinforcement (fire-and-forget update after recall)
    # ------------------------------------------------------------------

    async def recall_as_of(
        self,
        query: str,
        as_of: str,
        *,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> dict:
        """What this store believed to be true at *as_of* (ISO-8601 UTC).

        The payoff of keeping validity intervals instead of overwriting. A
        destructive supersede cannot answer "what was the rate limit in March?"
        at all -- the March value is gone. Here it is a range predicate over two
        sources: rows still current as of that instant, and revisions whose
        validity interval covered it.

        Ranking is deliberately plain relevance: time-decay would be incoherent
        when the caller has already pinned the instant they care about.
        """
        from opendb_core.utils.tokenizer import tokenize_for_fts

        fts_query = escape_fts5(tokenize_for_fts(query), use_or=True)
        if not fts_query.strip():
            return {"as_of": as_of, "total": 0, "results": []}

        type_clause = " AND m.memory_type = ?" if memory_type else ""
        params: list = [fts_query]
        if memory_type:
            params.append(memory_type)

        # Live rows whose validity interval contains as_of.
        live_sql = f"""
            SELECT m.memory_id, m.content, m.memory_type, m.tags, m.metadata,
                   m.source, m.valid_from, m.valid_to, {_MEM_BM25} AS fts_rank
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ?{type_clause}
              AND m.valid_from <= ?
              AND (m.valid_to IS NULL OR m.valid_to > ?)
            ORDER BY {_MEM_BM25}
            LIMIT ?
        """
        async with self._db.execute(
            live_sql, [*params, as_of, as_of, limit * 3]
        ) as cur:
            live = await cur.fetchall()

        # Superseded versions that were current at as_of. These are not in the
        # FTS index (it tracks current content), so match in SQL.
        rev_sql = f"""
            SELECT memory_id, content, memory_type, tags, metadata, source,
                   valid_from, valid_to
            FROM memory_revisions
            WHERE valid_from <= ? AND valid_to > ?
            {"AND memory_type = ?" if memory_type else ""}
            ORDER BY valid_to DESC
            LIMIT ?
        """
        rev_params: list = [as_of, as_of]
        if memory_type:
            rev_params.append(memory_type)
        async with self._db.execute(rev_sql, [*rev_params, limit * 3]) as cur:
            revisions = await cur.fetchall()

        terms = content_token_set(query)
        results = []
        for r in live:
            results.append({
                "memory_id": r["memory_id"], "content": r["content"],
                "memory_type": r["memory_type"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "source": r["source"], "valid_from": r["valid_from"],
                "valid_to": r["valid_to"], "still_current": r["valid_to"] is None,
                "score": abs(float(r["fts_rank"])),
            })
        for r in revisions:
            overlap = len(terms & content_token_set(r["content"]))
            if not overlap:
                continue
            results.append({
                "memory_id": r["memory_id"], "content": r["content"],
                "memory_type": r["memory_type"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "source": r["source"], "valid_from": r["valid_from"],
                "valid_to": r["valid_to"], "still_current": False,
                "score": float(overlap),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return {"as_of": as_of, "total": len(results), "results": results[:limit]}

    async def reinforce_memories(self, memory_ids: list[str]) -> None:
        """Record an explicit review of *memory_ids* (FSRS-style reinforcement).

        This is **not** called from ``recall_memories`` any more, for two
        reasons:

        1. It set ``confidence = 1.0`` unconditionally, which made the FSRS
           decay model a no-op for every memory that had ever been retrieved —
           the curve could only ever demote things nobody read.
        2. Because the ``memories_updated`` trigger fired on any UPDATE that
           did not name ``updated_at``, it silently reset the column that the
           time-decay ranking reads. Reading a memory set its age to zero, so a
           stale fact that had been read once outranked a fresh one.

        Reinforcement is now an explicit operation a caller opts into, it
        preserves ``updated_at``, and it grows stability through
        ``access_count`` (which ``compute_confidence`` already folds in via
        ``S_eff``) rather than by pinning confidence to its maximum.
        """
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        async with self.write_txn() as conn:
            await conn.execute(
                f"UPDATE memories SET "
                f"last_accessed = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), "
                f"access_count = access_count + 1, "
                # Explicit no-op assignment: the decay clock must not move
                # just because a memory was reviewed.
                f"updated_at = updated_at "
                f"WHERE memory_id IN ({placeholders})",
                memory_ids,
            )

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
        explain: bool = False,
    ) -> dict:
        # Fast path: return all pinned memories without FTS search
        if pinned_only:
            return await self._list_pinned(memory_type, tags, limit, offset)

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

        # Candidate window for the Python-side time-decay/confidence re-rank.
        #
        # This used to be max(limit * 3, 60) regardless of `offset`, so any
        # offset at or beyond 60 returned nothing even when hundreds of
        # memories matched — the caller could not page past the first window.
        #
        # The window takes one of two sizes rather than growing with `offset`.
        # Re-ranking happens in Python over the window, so a window that grew
        # per page would re-order results between pages and the same memory
        # could appear on two pages (or on none). Two fixed sizes give every
        # page inside a regime an identical candidate set, and there is exactly
        # one boundary — leaving the shallow fast path.
        #
        # Fully stable deep pagination needs a cursor over a snapshot; that is
        # the real fix, and this keeps the failure mode to a single, documented
        # seam instead of every page boundary.
        if offset + limit > _MIN_CANDIDATES:
            fetch_limit = _MAX_CANDIDATES
        else:
            fetch_limit = _MIN_CANDIDATES

        search_sql = f"""
            SELECT m.memory_id, m.content, m.memory_type, m.pinned,
                   m.source, m.superseded_id, m.confidence,
                   m.last_accessed, m.access_count,
                   m.tags, m.metadata, m.created_at, m.updated_at,
                   {_MEM_BM25} AS fts_rank,
                   julianday('now') - julianday(m.updated_at) AS age_days,
                   julianday('now') - julianday(COALESCE(m.last_accessed, m.created_at)) AS days_since_access
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ?{filter_clause}
            ORDER BY {_MEM_BM25}
            LIMIT ?
        """
        count_sql = f"""
            SELECT COUNT(*)
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ?{filter_clause}
        """

        search_params = [fts_query, *params, fetch_limit]
        count_params = [fts_query, *params]

        async with self._db.execute(search_sql, search_params) as cur:
            rows = await cur.fetchall()
        async with self._db.execute(count_sql, count_params) as cur:
            total_row = await cur.fetchone()
        # How many memories actually match the query, independent of the
        # candidate window. This is the number a caller can trust.
        matched_total = total_row[0] if total_row else 0

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

            # Compute live confidence (may have decayed since last access)
            live_conf = compute_confidence(
                base_confidence=float(r["confidence"]),
                days_since_last_access=days_since,
                access_count=int(r["access_count"]),
                pinned=bool(r["pinned"]),
                stability=stability,
            )

            # Skip faded memories
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
            })

        # Rank-aware query gate. The token-overlap gate is a precision filter
        # for weak tail matches, but it must never discard a strong lexical
        # hit: different phrasing between a question and the stored answer can
        # yield <2 overlapping tokens while the answer is still the #1 FTS
        # match. So keep any result close to the best FTS score and apply the
        # overlap gate only to the weaker tail.
        if scored:
            best_fts = max(s["_fts"] for s in scored)
            keep_floor = best_fts * 0.15
            scored = [
                s for s in scored
                if s["_fts"] >= keep_floor
                or passes_memory_query_gate(query, str(s["content"]))
            ]

        # Fuse the signals. In rrf mode this replaces the score-space product
        # (BM25 x decay x pin x confidence) and the 0.7-cluster/1-day recency
        # bonus that followed it — see storage/ranking.py for why multiplying a
        # raw BM25 value by a decay factor is not a defensible combination.
        if settings.ranking_mode == "rrf":
            fuse(scored, weights=RankingWeights.from_settings(settings),
                 recency_intent=recency)
        elif len(scored) >= 2:
            max_score = max(s["score"] for s in scored)
            if max_score > 0:
                for s in scored:
                    if s["score"] / max_score > 0.7:
                        age = s["_age_days"]
                        recency_bonus = 1.0 + 0.3 * (0.5 ** (age / 1.0))
                        s["score"] = s["score"] * recency_bonus

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Strip internal fields and round final scores before returning
        for s in scored:
            s.pop("_age_days", None)
            s.pop("_fts", None)
            if not explain:
                s.pop("_explain", None)
            elif "_explain" in s:
                s["explain"] = s.pop("_explain")
            s["score"] = float(f"{s['score']:.6g}")
        results = scored[offset : offset + limit]

        # `total` used to be overwritten with len(scored) — the size of the
        # candidate window after filtering, not the number of matches. It
        # reported 60 on a 200-memory store and swung to 1 as the corpus grew,
        # because the rank-aware gate prunes differently as BM25 scores shift.
        #
        # `total` is now the honest match count. `ranked` is how many survived
        # scoring and gating inside the candidate window, and `truncated` says
        # whether the window itself cut the match set short — so a caller can
        # tell "there are no more" apart from "we stopped looking".
        return {
            "total": matched_total,
            "ranked": len(scored),
            "truncated": matched_total > len(rows),
            "results": results,
        }

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
        async with self.write_txn() as conn:
            async with conn.execute(
                "SELECT id FROM memories WHERE memory_id = ?", (memory_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return False
            rowid = row["id"]
            # memories_fts is a standalone FTS5 table with no cascade, so both
            # deletes must land together or a search can still match a memory
            # that no longer exists.
            await conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?", (rowid,)
            )
            await conn.execute(
                "DELETE FROM memories WHERE id = ?", (rowid,)
            )
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
