"""Temporal ranking suite: an evaluation that can actually distinguish scorers.

Running the pooled LongMemEval corpus against `ranking_mode=legacy` and
`ranking_mode=rrf` produces *identical* numbers — R@1/3/5/10 and every
per-category figure. That is not evidence the two scorers are equivalent; it is
evidence the benchmark cannot see the difference. Every memory in that corpus is
stored in one pass, so all ages are ~0 and all confidences are 1.0. Only the
lexical term varies, so any scorer that respects lexical order scores the same.

That is the same failure as the original oracle harness: a measurement that
cannot fail carries no information about the thing it claims to measure.

This suite varies the signals the ranking function is *for*:

* **supersession** — an old fact and the newer fact that replaced it, both
  lexically plausible. The current one must win.
* **age spread** — several equally-relevant records months apart.
* **relevance dominance** — a much better lexical match that happens to be
  older. Relevance must still win; a scorer that always prefers recent is as
  broken as one that ignores time.
* **confidence** — a faded memory competing with a fresh one.
* **pinning** — a pinned memory must surface regardless of age.

Usage::

    python benchmark/tune_ranking.py                  # compare modes
    python benchmark/tune_ranking.py --tune           # cross-validated weights
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opendb_core.storage.sqlite import SQLiteBackend  # noqa: E402

# Each case: memories as (id, content, age_days, kind), the query, and the id
# that must come first. `kind` drives per-category reporting.
CASES: list[dict] = [
    {
        "name": "supersede/region",
        "category": "supersession",
        "query": "which region does the staging cluster run in",
        "gold": "region-new",
        "memories": [
            ("region-old", "The staging cluster runs in region us-east-1.", 400),
            ("region-new", "The staging cluster was moved to region eu-west-2.", 3),
        ],
    },
    {
        "name": "supersede/ratelimit",
        "category": "supersession",
        "query": "what is the API rate limit",
        "gold": "rl-new",
        "memories": [
            ("rl-old", "The API rate limit is 100 requests per minute.", 220),
            ("rl-new", "The API rate limit is now 200 requests per minute.", 5),
        ],
    },
    {
        "name": "supersede/framework",
        "category": "supersession",
        "query": "which frontend framework does the team use",
        "gold": "fw-new",
        "memories": [
            ("fw-old", "The team builds the frontend with React.", 500),
            ("fw-new", "The team migrated the frontend from React to Svelte.", 10),
        ],
    },
    {
        "name": "age-spread/deploys",
        "category": "age_spread",
        "query": "when did we last deploy the billing service",
        "gold": "dep-3",
        "memories": [
            ("dep-1", "Deployed the billing service to production.", 300),
            ("dep-2", "Deployed the billing service to production.", 120),
            ("dep-3", "Deployed the billing service to production.", 4),
        ],
    },
    {
        "name": "age-spread/car",
        "category": "age_spread",
        "query": "what car did I buy",
        "gold": "car-new",
        "memories": [
            ("car-old", "Bought a Honda Civic.", 1160),
            ("car-new", "Bought a Tesla Model Y.", 915),
        ],
    },
    {
        "name": "relevance-wins/auth",
        "category": "relevance_dominance",
        "query": "where does the JWT signing key live",
        "gold": "auth-old",
        "memories": [
            ("auth-old",
             "The JWT signing key lives in vault at secret/jwt/signing-key.", 400),
            ("auth-new", "We deployed a new authentication service yesterday.", 1),
        ],
    },
    {
        "name": "relevance-wins/port",
        "category": "relevance_dominance",
        "query": "what port does the metrics endpoint listen on",
        "gold": "port-old",
        "memories": [
            ("port-old", "The metrics endpoint listens on port 9090.", 250),
            ("port-new", "We reviewed the metrics dashboard layout today.", 2),
        ],
    },
    {
        "name": "confidence/faded-vs-fresh",
        "category": "confidence",
        "query": "which proxy does the gateway use",
        "gold": "prox-fresh",
        "memories": [
            ("prox-faded", "The gateway uses the legacy proxy for egress.", 30,
             {"confidence": 0.35}),
            ("prox-fresh", "The gateway uses the Envoy proxy for egress.", 30,
             {"confidence": 1.0}),
        ],
    },
    {
        "name": "confidence/faded-loses-despite-lexical-edge",
        "category": "confidence",
        "query": "retry budget for the payment worker",
        "gold": "retry-fresh",
        "memories": [
            ("retry-faded", "Retry budget for the payment worker is 3 attempts.", 30,
             {"confidence": 0.32}),
            ("retry-fresh", "The payment worker retry budget was raised.", 30,
             {"confidence": 1.0}),
        ],
    },
    {
        "name": "pinned/invariant",
        "category": "pinning",
        "query": "connection pool invariant",
        "gold": "pin-1",
        "memories": [
            ("pin-1", "INVARIANT: never hold the connection pool lock across an await.",
             900, {"pinned": True}),
            ("pin-2", "Looked at the connection pool metrics.", 1),
        ],
    },
]


async def _build(case: dict, tmp: Path) -> SQLiteBackend:
    db = SQLiteBackend(db_path=tmp / f"{uuid.uuid4().hex}.db")
    await db.init()
    for entry in case["memories"]:
        mid, content, age = entry[0], entry[1], entry[2]
        opts = entry[3] if len(entry) > 3 else {}
        await db.store_memory(
            memory_id=mid, content=content, memory_type="episodic",
            tags=[], metadata={}, pinned=opts.get("pinned", False),
            source="user_explicit",
        )
        # Backdate. episodic memories never supersede, so each row survives and
        # the case tests ranking rather than write-path conflict handling.
        await db._wdb.execute(
            "UPDATE memories SET updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now', ?), "
            "created_at = strftime('%Y-%m-%dT%H:%M:%SZ','now', ?) "
            "WHERE memory_id = ?",
            (f"-{age} days", f"-{age} days", mid),
        )
        if "confidence" in opts:
            # last_accessed pins days_since_access to 0 so compute_confidence
            # returns the stored value rather than decaying it further; the case
            # is about ranking, not about the decay curve.
            await db._wdb.execute(
                "UPDATE memories SET confidence = ?, "
                "last_accessed = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE memory_id = ?",
                (opts["confidence"], mid),
            )
    await db._wdb.commit()
    return db


async def evaluate(mode: str, weights: dict | None = None) -> dict:
    """Return per-case correctness and MRR for *mode*."""
    from opendb_core.config import settings

    prev = settings.ranking_mode
    settings.ranking_mode = mode
    saved = {}
    if weights:
        for key, value in weights.items():
            saved[key] = getattr(settings, key)
            setattr(settings, key, value)

    hits: list[tuple[str, str, bool, float]] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for case in CASES:
                db = await _build(case, tmp)
                try:
                    res = await db.recall_memories(
                        case["query"], None, None, 10, 0,
                    )
                    ids = [r["memory_id"] for r in res["results"]]
                    top1 = bool(ids and ids[0] == case["gold"])
                    rr = 1.0 / (ids.index(case["gold"]) + 1) if case["gold"] in ids else 0.0
                    hits.append((case["name"], case["category"], top1, rr))
                finally:
                    await db.close()
    finally:
        settings.ranking_mode = prev
        for key, value in saved.items():
            setattr(settings, key, value)

    by_cat: dict[str, list[bool]] = {}
    for _, cat, ok, _ in hits:
        by_cat.setdefault(cat, []).append(ok)

    return {
        "mode": mode,
        "weights": weights or {},
        "top1": sum(1 for _, _, ok, _ in hits if ok) / len(hits),
        "mrr": statistics.mean(rr for _, _, _, rr in hits),
        "by_category": {c: sum(v) / len(v) for c, v in by_cat.items()},
        "failures": [n for n, _, ok, _ in hits if not ok],
    }


def _print(result: dict) -> None:
    print(f"  top-1 accuracy : {result['top1'] * 100:5.1f}%")
    print(f"  MRR            : {result['mrr']:.3f}")
    for cat, acc in sorted(result["by_category"].items()):
        print(f"    {cat:22} {acc * 100:5.1f}%")
    if result["failures"]:
        print(f"    failures: {', '.join(result['failures'])}")


async def _tune() -> dict:
    """Grid search with leave-one-category-out cross-validation.

    Reporting the best score over a grid searched on the same data is how the
    original constants were arrived at. Held-out folds are the whole point: a
    weight set that only wins on the cases it was fitted to has told you
    nothing.
    """
    grid = [
        {"rank_weight_lexical": lex, "rank_weight_recency": rec,
         "rank_weight_confidence": conf}
        for lex, rec, conf in itertools.product(
            (1.0,), (0.25, 0.5, 0.75, 1.0, 1.5), (0.0, 0.3, 0.6)
        )
    ]
    global CASES
    categories = sorted({c["category"] for c in CASES})
    all_cases = CASES

    fold_results = []
    for held_out in categories:
        train = [c for c in all_cases if c["category"] != held_out]
        test = [c for c in all_cases if c["category"] == held_out]

        best, best_score = None, -1.0
        for w in grid:
            CASES = train
            r = await evaluate("rrf", w)
            if r["mrr"] > best_score:
                best_score, best = r["mrr"], w
        CASES = test
        held = await evaluate("rrf", best)
        fold_results.append({"held_out": held_out, "weights": best,
                             "test_mrr": held["mrr"], "test_top1": held["top1"]})
    CASES = all_cases
    return {
        "folds": fold_results,
        "mean_test_mrr": statistics.mean(f["test_mrr"] for f in fold_results),
        "mean_test_top1": statistics.mean(f["test_top1"] for f in fold_results),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true",
                    help="cross-validate ranking weights instead of comparing modes")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.tune:
        result = await _tune()
        print("Leave-one-category-out cross-validation")
        print("=" * 60)
        for f in result["folds"]:
            print(f"  held out {f['held_out']:22} test MRR {f['test_mrr']:.3f} "
                  f"top1 {f['test_top1'] * 100:5.1f}%  weights={f['weights']}")
        print()
        print(f"  mean held-out MRR : {result['mean_test_mrr']:.3f}")
        print(f"  mean held-out top1: {result['mean_test_top1'] * 100:.1f}%")
    else:
        print(f"Temporal ranking suite — {len(CASES)} cases")
        print("=" * 60)
        result = {}
        for mode in ("legacy", "rrf"):
            print(f"\n{mode}:")
            r = await evaluate(mode)
            _print(r)
            result[mode] = r
        print()
        d = (result["rrf"]["top1"] - result["legacy"]["top1"]) * 100
        print(f"rrf - legacy: {d:+.1f} points top-1")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
