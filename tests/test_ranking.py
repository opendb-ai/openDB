"""Ranking fusion: the properties the old score-space product could not hold.

The previous scorer multiplied a raw BM25 value by a decay factor, a pin
multiplier and a confidence factor, then applied a 0.7-cluster / 1-day recency
bonus. Its constants were fitted against a benchmark that could not fail, and a
single float came out the far end with no way to decompose it.
"""

from __future__ import annotations

import pytest

from opendb_core.storage.ranking import (
    RankingWeights, fuse, recency_score, recompute_from_explain, _ranks,
)


def _c(fts: float, age: float, conf: float = 1.0, pinned: bool = False) -> dict:
    return {"_fts": fts, "_age_days": age, "confidence": conf, "pinned": pinned}


class TestSignalMappings:
    def test_recency_is_bounded_and_monotone(self) -> None:
        prev = 2.0
        for age in (0, 1, 10, 30, 100, 1000, 100_000):
            s = recency_score(age, 30.0)
            assert 0.0 < s <= 1.0
            assert s < prev
            prev = s
        assert recency_score(0, 30.0) == 1.0
        assert recency_score(30.0, 30.0) == pytest.approx(0.5)

    def test_negative_age_does_not_exceed_one(self) -> None:
        assert recency_score(-5.0, 30.0) == 1.0

    def test_ties_share_a_rank(self) -> None:
        """Otherwise two identical candidates are separated by whichever the
        storage engine happened to return first."""
        assert _ranks([5.0, 5.0, 1.0]) == [1, 1, 3]


class TestFusion:
    def test_explain_reproduces_the_score(self) -> None:
        """A scoring path that cannot reproduce its own number is one nobody
        can debug."""
        cands = [_c(9.0, 1), _c(4.0, 90, 0.6), _c(1.0, 400, 0.4, pinned=True)]
        fuse(cands, weights=RankingWeights())
        for c in cands:
            assert recompute_from_explain(c["_explain"]) == pytest.approx(
                c["score"], rel=1e-9
            )

    def test_pinned_outranks_everything_unpinned(self) -> None:
        """Pinning is a tier, not a magnitude that can be out-competed."""
        cands = [_c(100.0, 0, 1.0), _c(0.01, 5000, 0.31, pinned=True)]
        fuse(cands, weights=RankingWeights())
        assert cands[1]["score"] > cands[0]["score"]

    def test_recency_breaks_a_lexical_near_tie(self) -> None:
        cands = [_c(5.0, 400), _c(4.99, 2)]
        fuse(cands, weights=RankingWeights())
        assert cands[1]["score"] > cands[0]["score"]

    def test_the_newer_fact_wins_when_the_stale_row_matches_one_more_token(
        self,
    ) -> None:
        """Supersession must survive a moderate lexical deficit.

        2.0.0 shipped `rank_weight_recency = 0.5`, and at that weight this
        ordering inverted: a superseded fact outranked the one that replaced it
        whenever the stale row happened to match one extra query token. These
        are the real numbers from the `conflicting_dates` stress case — a React
        memory 947 days old whose BM25 is ~2.1x the Svelte memory that
        superseded it 152 days later.
        """
        stale, current = _c(1.0, 946.8), _c(0.4735, 794.8)
        fuse([stale, current], weights=RankingWeights())
        assert current["score"] > stale["score"], (
            f"superseded row won: {stale['score']:.4f} >= {current['score']:.4f}"
        )

    def test_supersession_has_margin_rather_than_sitting_on_the_threshold(
        self,
    ) -> None:
        """The default must not be one rounding error from flipping back.

        The inversion above happens below a recency weight of ~0.63. Pinning the
        boundary here means a future weight change that reintroduces the bug
        fails loudly instead of silently degrading recall for exactly the
        knowledge-update case this project sells.
        """
        def current_wins(recency: float) -> bool:
            stale, current = _c(1.0, 946.8), _c(0.4735, 794.8)
            fuse([stale, current], weights=RankingWeights(recency=recency))
            return current["score"] > stale["score"]

        assert not current_wins(0.60), "expected the documented failure below 0.63"
        assert current_wins(0.65), "expected the documented fix above 0.63"
        assert current_wins(RankingWeights().recency), "the shipped default must pass"

    def test_dataclass_defaults_match_the_shipped_settings(self) -> None:
        """Two independent defaults exist and nothing else forces them to agree.

        `from_settings` reads config, but a bare `RankingWeights()` uses the
        dataclass default. When 2.0.0's recency weight was raised, changing only
        config left every bare construction scoring against the old weights.
        """
        from opendb_core.config import settings

        defaults = RankingWeights()
        assert defaults.lexical == settings.rank_weight_lexical
        assert defaults.recency == settings.rank_weight_recency
        assert defaults.confidence == settings.rank_weight_confidence
        assert defaults.k == settings.rank_rrf_k

    def test_relevance_still_beats_recency_when_the_gap_is_large(self) -> None:
        """A scorer that always prefers recent is as broken as one that ignores
        time.

        "Large" means several rank positions, not a large BM25 ratio: lexical is
        mapped through rank, which deliberately discards magnitude because BM25
        values are not comparable across queries and a big ratio is often just
        document-length normalisation. A top hit therefore outranks the *second*
        rank by only ~1.6%, and recency can take that; it cannot take four
        positions.
        """
        cands = [_c(50.0, 400)] + [_c(float(9 - i), 1) for i in range(6)]
        fuse(cands, weights=RankingWeights())
        assert cands[0]["score"] == max(c["score"] for c in cands)

    def test_lexical_magnitude_is_preserved(self) -> None:
        """Rank alone reports a 50x BM25 gap and a 1.01x gap as the same one
        position, so it cannot express "this is much the better answer". That is
        what separates "a stale but far better match should win" from "these two
        match equally and recency should decide"."""
        big = [_c(50.0, 10), _c(1.0, 10)]
        small = [_c(1.01, 10), _c(1.0, 10)]
        fuse(big, weights=RankingWeights())
        fuse(small, weights=RankingWeights())
        assert (big[0]["score"] - big[1]["score"]) > 10 * (
            small[0]["score"] - small[1]["score"]
        )

    def test_near_tie_lets_recency_decide(self) -> None:
        """The case the 0.7-cluster-threshold hack was groping towards."""
        cands = [_c(1.0, 1160), _c(0.91, 915)]
        fuse(cands, weights=RankingWeights())
        assert cands[1]["score"] > cands[0]["score"]

    def test_recency_stays_discriminative_on_an_old_corpus(self) -> None:
        """With a fixed 30-day tau, 915 and 1160 days both map to ~0.03 and a
        26% real age difference cannot influence anything."""
        cands = [_c(5.0, 1160), _c(5.0, 915)]
        fuse(cands, weights=RankingWeights())
        gap = cands[1]["score"] - cands[0]["score"]
        assert gap > 0.01, f"recency saturated on an old corpus: gap {gap}"

    def test_a_tiny_age_spread_is_not_as_decisive_as_a_large_one(self) -> None:
        """Min-max alone would score the newest row 1.0 even if it were only an
        hour fresher, making an hour worth as much as a year."""
        hours = [_c(5.0, 0.0), _c(5.0, 0.05)]
        years = [_c(5.0, 0.0), _c(5.0, 700.0)]
        fuse(hours, weights=RankingWeights())
        fuse(years, weights=RankingWeights())
        assert (years[0]["score"] - years[1]["score"]) > (
            hours[0]["score"] - hours[1]["score"]
        )

    def test_one_day_gap_does_not_override_relevance(self) -> None:
        """The adaptive tau must not turn into 'newest always wins'."""
        cands = [_c(9.0, 4), _c(3.0, 3)]
        fuse(cands, weights=RankingWeights())
        assert cands[0]["score"] > cands[1]["score"]

    def test_confidence_separates_otherwise_equal_candidates(self) -> None:
        cands = [_c(5.0, 30, 0.35), _c(5.0, 30, 1.0)]
        fuse(cands, weights=RankingWeights())
        assert cands[1]["score"] > cands[0]["score"]

    def test_recency_intent_shortens_the_window(self) -> None:
        base = [_c(5.0, 60), _c(5.0, 5)]
        intent = [_c(5.0, 60), _c(5.0, 5)]
        fuse(base, weights=RankingWeights())
        fuse(intent, weights=RankingWeights(), recency_intent=True)
        assert (intent[1]["score"] - intent[0]["score"]) > (
            base[1]["score"] - base[0]["score"]
        )

    def test_empty_candidate_set_is_safe(self) -> None:
        fuse([], weights=RankingWeights())


class TestModeIsConfigurable:
    def test_default_is_rrf(self) -> None:
        from opendb_core.config import Settings
        assert Settings().ranking_mode == "rrf"

    @pytest.mark.asyncio
    async def test_legacy_mode_still_runs(self, tmp_path) -> None:
        """Kept so the two can be compared on one corpus."""
        import uuid
        from opendb_core.config import settings
        from opendb_core.storage.sqlite import SQLiteBackend

        db = SQLiteBackend(db_path=tmp_path / "legacy.db")
        await db.init()
        prev = settings.ranking_mode
        settings.ranking_mode = "legacy"
        try:
            await db.store_memory(
                memory_id=str(uuid.uuid4()), content="a note about deployments",
                memory_type="episodic", tags=[], metadata={}, pinned=False,
                source="user_explicit",
            )
            res = await db.recall_memories("deployments", None, None, 5, 0)
            assert len(res["results"]) == 1
            assert "explain" not in res["results"][0]
        finally:
            settings.ranking_mode = prev
            await db.close()

    @pytest.mark.asyncio
    async def test_explain_is_opt_in(self, tmp_path) -> None:
        import uuid
        from opendb_core.storage.sqlite import SQLiteBackend

        db = SQLiteBackend(db_path=tmp_path / "explain.db")
        await db.init()
        try:
            await db.store_memory(
                memory_id=str(uuid.uuid4()), content="a note about deployments",
                memory_type="episodic", tags=[], metadata={}, pinned=False,
                source="user_explicit",
            )
            off = await db.recall_memories("deployments", None, None, 5, 0)
            assert "explain" not in off["results"][0]
            assert "_explain" not in off["results"][0]

            on = await db.recall_memories("deployments", None, None, 5, 0, explain=True)
            e = on["results"][0]["explain"]
            assert {"lexical", "recency", "confidence", "total"} <= set(e)
            assert recompute_from_explain(e) == pytest.approx(
                on["results"][0]["score"], rel=1e-5
            )
        finally:
            await db.close()
