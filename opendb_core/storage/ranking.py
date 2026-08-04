"""Rank-space fusion for memory recall.

The previous scorer multiplied incommensurable quantities together::

    score = abs(bm25) * 0.5**(age/halflife) * pin_boost * confidence

then applied a 0.7-cluster / 1.0-day recency bonus on top. Three problems:

1. **BM25 is not a probability.** Its magnitude is corpus- and query-dependent
   and unbounded. Multiplying it by a decay factor means the *relative* weight
   of recency versus relevance silently changes with query length and corpus
   composition — at a 30-day half-life, a perfect match from a year ago scored
   ~5000x below a weak match from today, but nobody chose that ratio.
2. **The constants were unfalsifiable.** 0.15, 0.7, 1.0, 10.0, 30.0 were fitted
   against a benchmark that could not fail (see benchmark/REPORT.md Part 3), so
   there was no signal distinguishing a good value from a bad one.
3. **Ties are resolved by float noise.** A tiny change in age could flip an
   ordering, because the score is a product of continuous factors with no
   stable scale.

The replacement is a weighted sum of three signals, each mapped into [0, 1]
first so the weights mean what they look like they mean::

    score = w_lex * lexical + w_rec * recency + w_conf * confidence

Which mapping each signal gets depends on whether its magnitude carries
information:

* **Lexical -> rank space.** A BM25 value is uncalibrated: 8.2 means nothing on
  its own and is not comparable across queries. Only the ordering is
  trustworthy, so the score is Reciprocal Rank Fusion normalised to 1.0 at
  rank 1: ``(k+1)/(k+rank)``. `k` controls how sharply the head is favoured.

* **Recency -> calibrated magnitude.** Days *are* a real unit, and pure rank
  space would throw that away: it cannot tell "400 days versus 3" from "4 days
  versus 3", which is exactly the distinction that decides whether a superseded
  fact outranks the one that replaced it. The mapping is hyperbolic,
  ``1/(1 + age/tau)`` — bounded, monotone, and without the exponential's habit
  of collapsing to zero and taking the ordering with it. At the default
  tau of 30 days: 3 days -> 0.91, 90 days -> 0.25, 400 days -> 0.07.

* **Confidence -> used directly**, since `compute_confidence` already returns a
  retrievability in [0, 1].

Pinning stays a hard tier rather than a multiplier: a pinned memory outranks an
unpinned one, and pinned memories are ordered among themselves by the same sum.

What this removes: the 0.15 keep-floor's sibling constants — the 0.7 cluster
threshold, the 1.0-day recency-bonus half-life, the 10.0 pin multiplier — and
the multiplication of a raw BM25 value by a decay factor. What remains tunable
is three weights and one time constant, all interpretable, all cross-validated
by `benchmark/tune_ranking.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard RRF constant. Larger k flattens the contribution of top ranks.
DEFAULT_K = 60.0


@dataclass(frozen=True)
class RankingWeights:
    """Pull of each signal. Every signal is in [0, 1], so these are comparable."""

    lexical: float = 1.0
    recency: float = 0.5
    confidence: float = 0.3
    k: float = DEFAULT_K
    # Age at which the recency term is halved. Interpretable in days, unlike
    # the exponential half-life it replaces, which interacted multiplicatively
    # with an uncalibrated BM25 value.
    tau_days: float = 30.0

    @classmethod
    def from_settings(cls, settings) -> "RankingWeights":
        return cls(
            lexical=getattr(settings, "rank_weight_lexical", 1.0),
            recency=getattr(settings, "rank_weight_recency", 0.5),
            confidence=getattr(settings, "rank_weight_confidence", 0.3),
            k=getattr(settings, "rank_rrf_k", DEFAULT_K),
            tau_days=getattr(settings, "memory_decay_halflife_days", 30.0),
        )


def recency_score(age_days: float, tau_days: float) -> float:
    """Map an age in days to [0, 1]. 0 days -> 1.0, tau days -> 0.5."""
    return 1.0 / (1.0 + max(age_days, 0.0) / max(tau_days, 1e-9))


def _ranks(values: list[float], *, descending: bool = True) -> list[int]:
    """Dense 1-based ranks of *values*, ties sharing the best rank.

    Ties must share a rank: otherwise two memories with identical relevance and
    identical age would be separated by whichever the storage engine happened to
    return first.
    """
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0] * len(values)
    prev_value = None
    prev_rank = 0
    for position, idx in enumerate(order, start=1):
        if prev_value is not None and values[idx] == prev_value:
            ranks[idx] = prev_rank
        else:
            ranks[idx] = position
            prev_rank = position
            prev_value = values[idx]
    return ranks


def fuse(
    candidates: list[dict],
    *,
    weights: RankingWeights,
    recency_intent: bool = False,
) -> None:
    """Assign a fused ``score`` to each candidate, in place.

    Each candidate must carry ``_fts`` (higher is better), ``_age_days`` (lower
    is better) and ``confidence``. ``pinned`` promotes to a separate tier.

    Also sets ``_explain`` so ``recall(explain=True)`` can show why something
    ranked where it did — the old scorer produced a single float with no way to
    decompose it.
    """
    if not candidates:
        return

    lex_ranks = _ranks([c["_fts"] for c in candidates])

    # Both signals are normalised against the *candidate set*, not against an
    # absolute scale. Neither BM25 nor "days old" means anything on its own;
    # both mean something relative to the other rows competing for this slot.
    #
    # Lexical: ratio to the best hit. Rank alone cannot work here — it reports a
    # 50x BM25 gap and a 1.01x gap as the same one position, so it can express
    # "this is much the better answer" only by accident. The ratio keeps that
    # distinction, which is what separates "a stale but far better match should
    # win" from "these two match equally and recency should decide".
    #
    # Recency: min-max, so the newest row present scores 1.0 whatever the
    # corpus's absolute age. A fixed time constant saturates — at 915 and 1160
    # days a 30-day tau maps both to ~0.03 and a real 26% difference stops
    # mattering, which is how a 2023 memory outranked a 2024 one.
    best_fts = max(c["_fts"] for c in candidates)
    ages = [max(c["_age_days"], 0.0) for c in candidates]
    youngest, oldest = min(ages), max(ages)
    age_span = oldest - youngest

    k = weights.k
    tau = weights.tau_days / 2.0 if recency_intent else weights.tau_days

    # Ceiling on the unpinned sum, so the pinned tier cannot be out-competed.
    tier = weights.lexical + weights.recency + weights.confidence

    # Ceiling on the unpinned sum, so the pinned tier cannot be out-competed.
    tier = weights.lexical + weights.recency + weights.confidence

    for i, c in enumerate(candidates):
        if best_fts > 0:
            lex_norm = max(0.0, min(1.0, c["_fts"] / best_fts))
        else:
            lex_norm = (k + 1.0) / (k + lex_ranks[i])

        if age_span > 0:
            relative = 1.0 - (ages[i] - youngest) / age_span
            # Blend in the absolute mapping so a one-day spread cannot be as
            # decisive as a one-year spread: min-max alone would make the
            # newest row score 1.0 even if it is only an hour fresher.
            confidence_in_span = age_span / (age_span + tau)
            rec_norm = (
                relative * confidence_in_span
                + recency_score(ages[i], tau) * (1.0 - confidence_in_span)
            )
        else:
            rec_norm = recency_score(ages[i], tau)

        conf_norm = float(c.get("confidence", 1.0))

        lex = weights.lexical * lex_norm
        rec = weights.recency * rec_norm
        conf = weights.confidence * conf_norm
        fused = lex + rec + conf
        if c.get("pinned"):
            fused += tier
        c["score"] = fused
        c["_explain"] = {
            "lexical": {"rank": lex_ranks[i], "raw": c["_fts"], "best": best_fts,
                        "normalized": lex_norm,
                        "weight": weights.lexical, "contribution": lex},
            "recency": {"age_days": c["_age_days"], "tau_days": tau,
                        "age_span_days": age_span,
                        "normalized": rec_norm, "weight": weights.recency,
                        "contribution": rec},
            "confidence": {"normalized": conf_norm,
                           "weight": weights.confidence, "contribution": conf},
            "pinned": bool(c.get("pinned")),
            "pinned_tier": tier if c.get("pinned") else 0.0,
            "total": fused,
        }


def recompute_from_explain(explain: dict) -> float:
    """Rebuild a score from its decomposition.

    A scoring path that cannot reproduce its own number is a scoring path
    nobody can debug; `tests/test_ranking.py` asserts this round-trips.
    """
    return (
        explain["lexical"]["contribution"]
        + explain["recency"]["contribution"]
        + explain["confidence"]["contribution"]
        + explain.get("pinned_tier", 0.0)
    )
