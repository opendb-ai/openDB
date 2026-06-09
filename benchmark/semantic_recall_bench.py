#!/usr/bin/env python3
"""
Semantic-recall benchmark — pure FTS vs hybrid (FTS + local vectors)
=============================================================================
LongMemEval and CodeMemEval already hit ~100% R@5 with pure FTS because their
questions share keywords with the answer. This benchmark isolates the case FTS
*cannot* solve on its own: **paraphrased queries with near-zero lexical overlap**
with the stored memory (synonyms, conceptual rephrasings). It is the direct
evidence for OpenDB's optional hybrid mode.

Deterministic and local: hand-authored (memory, paraphrase-query) pairs plus
distractor memories. No LLM, no network at query time. Runs both retrieval
modes in one process and prints a side-by-side R@K comparison.

Usage:
    python semantic_recall_bench.py            # both modes, R@1/3/5/10
    python semantic_recall_bench.py --k 5
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# (memory_text, paraphrase_query)  — query deliberately avoids the memory's words.
PAIRS: list[tuple[str, str]] = [
    ("We migrated the frontend from React to Svelte.", "which UI framework does the team use now"),
    ("The billing service stores invoices in PostgreSQL.", "where do we persist payment records"),
    ("Deployments go through ArgoCD watching the gitops repository.", "how does code reach production"),
    ("Container images must be signed with cosign before release.", "what security gate blocks unsigned artifacts"),
    ("The on-call engineer is paged through PagerDuty.", "how does someone get alerted during an outage"),
    ("Inter-service events flow over NATS JetStream.", "what carries messages between microservices"),
    ("Routes are cached in Redis with a six hour expiry.", "how long until the path lookup data goes stale"),
    ("Authentication uses short-lived JWTs validated at the gateway.", "how do we verify a caller's identity"),
    ("The user prefers a dark colour theme in the dashboard.", "what visual style does this person like"),
    ("Quarterly revenue grew to fifteen million dollars.", "how much money came in over three months"),
    ("Integration tests spin up ephemeral databases with testcontainers.", "how are real datastores provided to test suites"),
    ("The team stand-up happens every weekday at 9:30am.", "when do engineers sync each morning"),
    ("Errors are returned as RFC 7807 problem documents.", "what shape do failure responses take"),
    ("The search index is rebuilt nightly by a cron job.", "how does the lookup table stay fresh"),
    ("Customer data is encrypted at rest using AES-256.", "how is stored personal information protected"),
    ("The mobile app talks to the backend over GraphQL.", "what query language does the phone client use"),
    ("Feature flags are managed in LaunchDarkly.", "where do we toggle experimental functionality"),
    ("The CFO approved a budget increase for cloud spend.", "who signed off on more infrastructure money"),
    ("Logs are shipped to an OpenSearch cluster for analysis.", "where can engineers investigate runtime output"),
    ("The release train ships every second Tuesday.", "how often do new versions go out"),
    ("Background jobs run on a Celery worker pool.", "what processes asynchronous tasks"),
    ("The recommendation model is retrained weekly on fresh clicks.", "how often does the suggestion engine update"),
    ("Secrets are injected from HashiCorp Vault at boot.", "where do credentials come from at startup"),
    ("The API enforces a rate limit of 100 requests per minute.", "how many calls are callers allowed before throttling"),
    ("Our primary datacentre is in Frankfurt.", "where is the main server location"),
    ("The designer favours generous whitespace in layouts.", "what aesthetic does the design lead prefer"),
    ("Checkout abandonment dropped after we added Apple Pay.", "what change reduced cart drop-off"),
    ("The data pipeline is orchestrated with Apache Airflow.", "what schedules our batch ETL"),
    ("New hires shadow a mentor for their first two weeks.", "how are people onboarded when they join"),
    ("The website is served from a Cloudflare CDN.", "how do static assets get delivered fast"),
    ("We rolled back the pricing change after churn spiked.", "what did we undo when customers started leaving"),
    ("Photos are resized into thumbnails by a Lambda function.", "what makes the small preview images"),
    ("The contract renews automatically every January.", "when does the agreement extend itself"),
    ("Push notifications are delivered through Firebase.", "what sends alerts to phones"),
    ("The legacy monolith is being split into bounded contexts.", "what is happening to the old single codebase"),
    ("Staging mirrors production with anonymised data.", "what does the pre-release environment contain"),
    ("The founder wants to expand into the Japanese market next.", "which country is the growth target"),
    ("Database migrations must be reversible.", "what property is required of schema changes"),
    ("The support team uses Zendesk for ticketing.", "where are customer issues tracked"),
    ("Our uptime target is 99.95 percent.", "what availability are we committed to"),
]

# Distractor memories (no paraphrase query targets these) to make recall non-trivial.
DISTRACTORS: list[str] = [
    "The kitchen restocks coffee on Mondays.",
    "Parking permits are renewed at the front desk.",
    "The company picnic is scheduled for July.",
    "Meeting rooms are booked through the calendar app.",
    "The printer on the third floor is out of toner.",
    "Annual reviews happen in December.",
    "The gym membership discount expires soon.",
    "Visitor badges must be returned at reception.",
    "The fire drill is next Thursday.",
    "Standing desks can be requested from facilities.",
]


async def run_mode(mode: str, ks: list[int]) -> dict:
    import opendb_core.config as cfg
    importlib.reload(cfg)
    cfg.settings.memory_retrieval_mode = mode
    import opendb_core.storage.sqlite as sq
    importlib.reload(sq)

    db = Path(tempfile.mkdtemp()) / "sem.db"
    backend = sq.SQLiteBackend(db_path=db)
    await backend.init()

    target_ids = []
    for mem, _q in PAIRS:
        tid = str(uuid.uuid4())
        target_ids.append(tid)
        await backend.store_memory(memory_id=tid, content=mem, memory_type="semantic",
                                   tags=[], metadata={})
    for d in DISTRACTORS:
        await backend.store_memory(memory_id=str(uuid.uuid4()), content=d,
                                   memory_type="semantic", tags=[], metadata={})

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    latencies = []
    for (mem, q), tid in zip(PAIRS, target_ids):
        t0 = time.perf_counter()
        res = await backend.recall_memories(query=q, memory_type=None, tags=None,
                                            limit=max_k, offset=0)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [r["memory_id"] for r in res["results"]]
        for k in ks:
            if tid in ranked[:k]:
                hits[k] += 1
    await backend.close()

    import statistics
    n = len(PAIRS)
    return {
        "mode": mode,
        "hybrid_active": getattr(backend, "_hybrid", False),
        "n": n,
        "recall": {f"R@{k}": round(hits[k] / n * 100, 1) for k in ks},
        "median_ms": round(statistics.median(latencies), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--output", default=str(Path(__file__).parent / "semantic_recall_results.json"))
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))

    print(f"Semantic-recall benchmark — {len(PAIRS)} paraphrase pairs + "
          f"{len(DISTRACTORS)} distractors\n")
    rows = []
    for mode in ("fts", "hybrid"):
        rows.append(asyncio.run(run_mode(mode, ks)))

    header = f"{'mode':<10}{'active':<8}" + "".join(f"R@{k:<6}" for k in ks) + "median"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['mode']:<10}{str(r['hybrid_active']):<8}"
        line += "".join(f"{r['recall'][f'R@{k}']:<7}" for k in ks)
        line += f"{r['median_ms']}ms"
        print(line)

    fts = next(r for r in rows if r["mode"] == "fts")
    hyb = next(r for r in rows if r["mode"] == "hybrid")
    top = max(ks)
    print(f"\nHybrid lifts R@{top}: {fts['recall'][f'R@{top}']}% -> "
          f"{hyb['recall'][f'R@{top}']}%  "
          f"(+{round(hyb['recall'][f'R@{top}'] - fts['recall'][f'R@{top}'], 1)} pts)")

    import json
    Path(args.output).write_text(json.dumps(
        {"pairs": len(PAIRS), "distractors": len(DISTRACTORS), "ks": ks, "results": rows},
        ensure_ascii=False, indent=1))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
