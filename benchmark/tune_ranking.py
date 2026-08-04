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

**This suite failed its own thesis once.** As shipped in 2.0.0 it returned an
identical 80.0% top-1, with identical per-category results, for every recency
weight in [0.25, 1.5] — so it could not select the constant it exists to
select, and 2.0.0 shipped a supersession regression it scored as 100%. Two
defects caused that:

* Every supersession case gave the *newer* memory a lexical score at least as
  good as the stale one, so they passed at any weight. Real superseded facts
  usually look the opposite way — the stale row repeats the query's wording
  while its replacement describes the change. That shape is now covered, and
  it is the shape that broke.
* `age-spread/car` asked "what car did I buy" against "Bought a Honda Civic."
  Not one token matched, FTS returned nothing, and a case with no results was
  scored as a ranking failure. It held its category at 50% for every weight
  while measuring nothing at all. `evaluate` now reports an empty result set as
  a broken fixture rather than a scorer failure.

Repaired, it separates the weights (75.0% at 0.5, 91.7% at 0.75) and its
cross-validation independently selects 0.75 in four of five folds — the value
2.0.1 shipped after measuring it on other suites entirely.

Before trusting a tuned constant from any suite, run ``--sweep`` and confirm
the score actually moves. A number that does not move cannot choose anything.

Usage::

    python benchmark/tune_ranking.py                       # compare modes
    python benchmark/tune_ranking.py --explain             # why each failure lost
    python benchmark/tune_ranking.py --sweep 0.25,0.5,0.75 # can it discriminate?
    python benchmark/tune_ranking.py --tune                # cross-validated weights
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
        # The query used to be "what car did I buy" against "Bought a Honda
        # Civic." — no shared token ("car" appears in neither memory, and "buy"
        # is not "Bought"), so FTS returned nothing and no scorer could ever
        # pass. It counted as a ranking failure for every weight in the grid,
        # holding age_spread at 50% and hiding whatever the real answer was.
        # Both memories now match the query equally, which is what
        # "several equally-relevant records months apart" was supposed to mean.
        "query": "car I bought",
        "gold": "car-new",
        "memories": [
            ("car-old", "Bought a Honda Civic car.", 1160),
            ("car-new", "Bought a Tesla Model Y car.", 915),
        ],
    },
    {
        # The shape that let 2.0.0 ship a supersession regression. Every other
        # supersession case gives the newer memory a lexical score at least as
        # good as the stale one, so they pass at any recency weight and the
        # suite reported 100% while recall was broken. Here the stale row wins
        # on lexical -- it repeats the query's own words, while its replacement
        # describes the change -- which is the ordinary way a superseded fact
        # looks. memory_stress_bench.py caught this; this suite did not.
        "name": "supersede/stale-has-the-lexical-edge",
        "category": "supersession",
        "query": "frontend framework team",
        "gold": "sfe-new",
        "memories": [
            ("sfe-old", "The team uses React for the frontend.", 947),
            ("sfe-new", "We migrated the frontend from React to Svelte.", 795),
        ],
    },
    {
        # Same shape, both rows recent, so it is sensitive to the weight ratio
        # rather than to the age-span blend.
        "name": "supersede/stale-lexical-edge-recent",
        "category": "supersession",
        "query": "database billing service uses",
        "gold": "sdb-new",
        "memories": [
            ("sdb-old", "The billing service uses the database PostgreSQL 14.", 60),
            ("sdb-new", "Billing was moved onto PostgreSQL 15.", 3),
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
        # Known defect, not a regression. The row that literally answers the
        # question loses to a fresh row that does not:
        #
        #   port-old  lex=1.0000  rec=0.0116  conf=0.5818  -> 1.1832
        #   port-new  lex=0.5000  rec=0.9933  conf=0.9923  -> 1.5426
        #
        # Age is counted twice. It suppresses recency directly, and it also
        # decays confidence (0.58 against 0.99), so a 2x lexical advantage
        # cannot recover. It fails identically at recency weights 0.5 and 0.75,
        # so it predates and survives the 2.0.1 fix. Recorded rather than
        # silently averaged into the headline: a suite that reports 80% without
        # saying which 20% is broken is the thing this file exists to argue
        # against.
        "known_failing": "age suppresses recency and decays confidence, so a 2x "
                         "lexical edge cannot win",
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


async def evaluate(mode: str, weights: dict | None = None,
                   explain: bool = False) -> dict:
    """Return per-case correctness and MRR for *mode*."""
    from opendb_core.config import settings

    prev = settings.ranking_mode
    settings.ranking_mode = mode
    saved = {}
    if weights:
        for key, value in weights.items():
            saved[key] = getattr(settings, key)
            setattr(settings, key, value)

    hits: list[dict] = []
    dead: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for case in CASES:
                db = await _build(case, tmp)
                try:
                    res = await db.recall_memories(
                        case["query"], None, None, 10, 0, explain=explain,
                    )
                    rows = res["results"]
                    ids = [r["memory_id"] for r in rows]
                    # A case whose query retrieves nothing is not a scorer
                    # failure -- there is no ordering to get wrong. Scoring it
                    # as one is how `age-spread/car` sat at 0 for every weight
                    # in the grid while appearing to say something about
                    # ranking. Report it as a broken fixture instead.
                    if not ids:
                        dead.append(case["name"])
                        continue
                    hits.append({
                        "name": case["name"],
                        "category": case["category"],
                        "top1": ids[0] == case["gold"],
                        "rr": 1.0 / (ids.index(case["gold"]) + 1)
                              if case["gold"] in ids else 0.0,
                        "known_failing": case.get("known_failing"),
                        "rows": [
                            {"id": r["memory_id"], "score": r.get("score"),
                             "explain": r.get("explain")}
                            for r in rows
                        ] if explain else [],
                    })
                finally:
                    await db.close()
    finally:
        settings.ranking_mode = prev
        for key, value in saved.items():
            setattr(settings, key, value)

    by_cat: dict[str, list[bool]] = {}
    for h in hits:
        by_cat.setdefault(h["category"], []).append(h["top1"])

    # A known defect is not a regression. Keeping the two apart means the
    # headline moves when something breaks, instead of being permanently
    # discounted by failures nobody is acting on.
    regressions = [h["name"] for h in hits if not h["top1"] and not h["known_failing"]]
    known = [h["name"] for h in hits if not h["top1"] and h["known_failing"]]
    fixed = [h["name"] for h in hits if h["top1"] and h["known_failing"]]

    return {
        "mode": mode,
        "weights": weights or {},
        "top1": sum(1 for h in hits if h["top1"]) / len(hits) if hits else 0.0,
        "mrr": statistics.mean(h["rr"] for h in hits) if hits else 0.0,
        "by_category": {c: sum(v) / len(v) for c, v in by_cat.items()},
        "failures": regressions,
        "known_failures": known,
        "unexpectedly_fixed": fixed,
        "dead_fixtures": dead,
        "cases": hits if explain else [],
    }


def _print(result: dict) -> None:
    print(f"  top-1 accuracy : {result['top1'] * 100:5.1f}%")
    print(f"  MRR            : {result['mrr']:.3f}")
    for cat, acc in sorted(result["by_category"].items()):
        print(f"    {cat:22} {acc * 100:5.1f}%")
    if result["failures"]:
        print(f"    REGRESSIONS   : {', '.join(result['failures'])}")
    if result.get("known_failures"):
        print(f"    known defects : {', '.join(result['known_failures'])}")
    if result.get("unexpectedly_fixed"):
        print(f"    now passing (drop known_failing): "
              f"{', '.join(result['unexpectedly_fixed'])}")
    if result.get("dead_fixtures"):
        print(f"    BROKEN FIXTURES (query retrieved nothing, so the case "
              f"measures no ordering): {', '.join(result['dead_fixtures'])}")


def _print_explain(result: dict) -> None:
    """Show the decomposition for every case that did not come first.

    Without this the suite reports a name and leaves you to rebuild the score
    by hand to find out whether lexical, recency or confidence decided it.
    """
    failing = [c for c in result.get("cases", []) if not c["top1"]]
    if not failing:
        return
    print()
    print("  Why each failure ranked the way it did")
    print("  " + "-" * 68)
    for c in failing:
        tag = "known defect" if c["known_failing"] else "REGRESSION"
        print(f"  [{tag}] {c['name']}")
        if c["known_failing"]:
            print(f"      {c['known_failing']}")
        for r in c["rows"]:
            ex = r.get("explain") or {}
            if not ex:
                # `legacy` produces a single float with no decomposition --
                # that opacity is one of the reasons it was replaced. Say so
                # instead of printing a row of zeros, which would read as
                # "every signal contributed nothing".
                print(f"      {r['id']:<12} score={r['score']:.4f}"
                      f"  (no decomposition: this scorer does not emit one)")
                continue
            lx, rc, cf = (ex.get("lexical", {}), ex.get("recency", {}),
                          ex.get("confidence", {}))
            print(f"      {r['id']:<12} score={r['score']:.4f}"
                  f"  lex={lx.get('normalized', 0):.4f}"
                  f"  rec={rc.get('normalized', 0):.4f}"
                  f"  conf={cf.get('normalized', 0):.4f}"
                  f"  age={rc.get('age_days', 0):.0f}d")
        print()


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
    ap.add_argument("--explain", action="store_true",
                    help="show the score decomposition for every failing case")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated recency weights to score, to check the "
                         "suite can actually separate them")
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
    elif args.sweep:
        # A suite that returns the same score for every weight cannot select
        # one. Before trusting any number here, check that it moves.
        print(f"Recency-weight sweep — {len(CASES)} cases")
        print("=" * 60)
        print(f"  {'recency':>8}  {'top-1':>7}  {'MRR':>6}   regressions")
        print(f"  {'-'*8}  {'-'*7}  {'-'*6}   {'-'*34}")
        result = {}
        for w in [float(x) for x in args.sweep.split(",")]:
            r = await evaluate("rrf", {"rank_weight_recency": w})
            result[str(w)] = r
            print(f"  {w:8.2f}  {r['top1']*100:6.1f}%  {r['mrr']:.3f}   "
                  f"{', '.join(r['failures']) or '—'}")
        scores = {r["top1"] for r in result.values()}
        print()
        if len(scores) == 1:
            print("  The suite returns an identical score at every weight, so it")
            print("  cannot be used to choose one. Add cases that vary the signal")
            print("  being weighted before quoting a tuned value.")
        else:
            print(f"  The suite separates these weights ({len(scores)} distinct "
                  f"scores), so it can discriminate between them.")
    else:
        print(f"Temporal ranking suite — {len(CASES)} cases")
        print("=" * 60)
        result = {}
        for mode in ("legacy", "rrf"):
            print(f"\n{mode}:")
            r = await evaluate(mode, explain=args.explain)
            _print(r)
            if args.explain:
                _print_explain(r)
            result[mode] = r
        print()
        d = (result["rrf"]["top1"] - result["legacy"]["top1"]) * 100
        print(f"rrf - legacy: {d:+.1f} points top-1")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
