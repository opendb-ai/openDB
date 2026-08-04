#!/usr/bin/env python3
"""
LongMemEval Benchmark for OpenDB Memory Pipeline
=====================================================

Evaluates OpenDB's memory_store + memory_recall against the LongMemEval
benchmark (500 questions, 6 types). Measures session-level Recall@K —
the fraction of questions where at least one evidence session appears
in the top-K recalled memories.

Comparison target: MemPalace reports 96.6% R@5 (pure local, zero API).

Usage:
    python longmemeval_bench.py [--data longmemeval_oracle.json] [--ks 1,3,5,10] [--limit N]

Requirements:
    pip install opendb   (this project)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add project root to path so we can import opendb_core directly
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from opendb_core.storage import init_backend, get_backend, close_backend

# ============================================================
# Config
# ============================================================

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BENCHMARK_DIR / "longmemeval_oracle.json"


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    answer_session_ids: list[str]
    recalled_session_ids: list[str]
    hits: dict[int, bool]  # k -> whether any answer session in top-k
    recall_time_ms: float
    store_time_ms: float = 0.0


# ============================================================
# Helpers
# ============================================================


def flatten_session(session: list[dict]) -> str:
    """Flatten a conversation session into a single text block."""
    parts = []
    for turn in session:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def load_dataset(path: str | Path) -> list[dict]:
    """Load the LongMemEval JSON dataset."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {Path(path).name}")
    return data


# ============================================================
# Core benchmark
# ============================================================


async def run_single_question(
    question: dict,
    ks: list[int],
) -> QuestionResult:
    """Store sessions as memories, recall by question, measure retrieval."""

    backend = get_backend()
    max_k = max(ks)

    session_ids = question["haystack_session_ids"]
    sessions = question["haystack_sessions"]
    dates = question.get("haystack_dates", [])
    answer_ids = set(question["answer_session_ids"])

    # --- Store phase: each session → one memory ---
    import uuid

    t0 = time.perf_counter()
    for i, (sid, session) in enumerate(zip(session_ids, sessions)):
        text = flatten_session(session)
        meta = {"session_id": sid}
        if i < len(dates):
            meta["date"] = dates[i]
        await backend.store_memory(
            memory_id=str(uuid.uuid4()),
            content=text,
            memory_type="episodic",
            tags=[sid],
            metadata=meta,
        )
    store_ms = (time.perf_counter() - t0) * 1000

    # --- Recall phase ---
    t1 = time.perf_counter()
    result = await backend.recall_memories(
        query=question["question"],
        memory_type=None,
        tags=None,
        limit=max_k,
        offset=0,
    )
    recall_ms = (time.perf_counter() - t1) * 1000

    # Map recalled memories back to session_ids via tags
    recalled_sids = []
    for mem in result.get("results", []):
        tags = mem.get("tags", [])
        for tag in tags:
            if tag in session_ids:
                recalled_sids.append(tag)
                break

    # Compute hits at each K
    hits = {}
    for k in ks:
        top_k_sids = set(recalled_sids[:k])
        hits[k] = bool(top_k_sids & answer_ids)

    return QuestionResult(
        question_id=question["question_id"],
        question_type=question["question_type"],
        answer_session_ids=list(answer_ids),
        recalled_session_ids=recalled_sids[:max_k],
        hits=hits,
        recall_time_ms=recall_ms,
        store_time_ms=store_ms,
    )


async def run_pooled_benchmark(
    data: list[dict],
    ks: list[int],
    limit: int | None = None,
) -> list[QuestionResult]:
    """Measure retrieval against ONE corpus built from every question's sessions.

    This is the mode that actually measures retrieval, and it should be the
    default one to quote.

    The isolated-DB mode below gives each question its own empty database
    containing only that question's haystack. In ``longmemeval_oracle.json``
    every haystack *is* the gold set — all 500 questions have
    ``set(haystack_session_ids) == set(answer_session_ids)``, mean 1.896
    sessions — so the retriever picks from ~2 candidates to fill 5 slots and
    R@5 is guaranteed by arithmetic for 497/500 questions. It cannot fail, and
    a measurement that cannot fail carries no information.

    Pooling every question's sessions into one store turns the other 499
    questions' sessions into distractors, which is the task a real deployment
    performs. Expect a materially lower number here; that is the point.
    """
    import uuid

    questions = [q for q in data if not q["question_id"].endswith("_abs")]
    if limit:
        questions = questions[:limit]

    max_k = max(ks)
    tmp_root = Path(tempfile.mkdtemp(prefix="opendb_longmemeval_pooled_"))
    db_path = tmp_root / "pooled.db"
    results: list[QuestionResult] = []

    try:
        await init_backend("sqlite", db_path=str(db_path))
        try:
            # --- Build one corpus from every question's sessions ---
            stored: set[str] = set()
            t0 = time.perf_counter()
            for q in questions:
                dates = q.get("haystack_dates", [])
                for i, (sid, session) in enumerate(
                    zip(q["haystack_session_ids"], q["haystack_sessions"])
                ):
                    if sid in stored:
                        continue  # sessions are shared between questions
                    stored.add(sid)
                    meta = {"session_id": sid}
                    if i < len(dates):
                        meta["date"] = dates[i]
                    await backend_store(sid, flatten_session(session), meta)
            store_ms = (time.perf_counter() - t0) * 1000
            print(
                f"Pooled corpus: {len(stored)} sessions from {len(questions)} questions "
                f"({store_ms / 1000:.1f}s to index)"
            )
            print(
                f"Distractors per question: "
                f"{len(stored) - (sum(len(q['haystack_session_ids']) for q in questions) / len(questions)):.0f} "
                f"on average"
            )
            print()

            backend = get_backend()
            for idx, q in enumerate(questions):
                answer_ids = set(q["answer_session_ids"])
                t1 = time.perf_counter()
                result = await backend.recall_memories(
                    query=q["question"], memory_type=None, tags=None,
                    limit=max_k, offset=0,
                )
                recall_ms = (time.perf_counter() - t1) * 1000

                recalled_sids = []
                for mem in result.get("results", []):
                    sid = (mem.get("metadata") or {}).get("session_id")
                    if sid:
                        recalled_sids.append(sid)

                results.append(QuestionResult(
                    question_id=q["question_id"],
                    question_type=q["question_type"],
                    answer_session_ids=list(answer_ids),
                    recalled_session_ids=recalled_sids[:max_k],
                    hits={k: bool(set(recalled_sids[:k]) & answer_ids) for k in ks},
                    recall_time_ms=recall_ms,
                    store_time_ms=0.0,
                ))

                if (idx + 1) % 50 == 0 or idx == len(questions) - 1:
                    r5 = sum(1 for r in results if r.hits.get(5, False)) / len(results) * 100
                    print(f"  [{idx + 1}/{len(questions)}] Running R@5: {r5:.1f}%")
        finally:
            await close_backend(str(db_path))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return results


async def backend_store(sid: str, text: str, meta: dict) -> None:
    import uuid
    await get_backend().store_memory(
        memory_id=str(uuid.uuid4()),
        content=text,
        memory_type="episodic",
        tags=[sid],
        metadata=meta,
    )


async def run_benchmark(
    data: list[dict],
    ks: list[int],
    limit: int | None = None,
) -> list[QuestionResult]:
    """Run the isolated-DB benchmark: one empty database per question.

    WARNING: on ``longmemeval_oracle.json`` this does not measure retrieval.
    Each question's haystack contains only its own gold sessions (mean 1.896,
    zero distractors), so R@5 is arithmetically guaranteed. Use
    ``--pooled`` for a number that can distinguish a good retriever from a bad
    one; this mode is kept for latency measurement and for datasets that carry
    real distractors (CodeMemEval does: ~17 per question).
    """

    # Skip abstention questions (no ground-truth answer location)
    questions = [q for q in data if not q["question_id"].endswith("_abs")]
    if limit:
        questions = questions[:limit]

    n_distractors = sum(
        len(set(q["haystack_session_ids"]) - set(q["answer_session_ids"]))
        for q in questions
    )
    if n_distractors == 0 and questions:
        print("!" * 72)
        print("! WARNING: this dataset has ZERO distractor sessions.")
        print("! Every haystack session is a gold session, so Recall@K is")
        print("! guaranteed whenever the haystack is smaller than K.")
        print("! The resulting number measures the harness, not the retriever.")
        print("! Re-run with --pooled for a discriminative measurement.")
        print("!" * 72)
        print()

    print(f"Running {len(questions)} questions (skipped {len(data) - len(questions)} abstention)")
    print(f"Recall@K levels: {ks}")
    print()

    results: list[QuestionResult] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="opendb_longmemeval_"))

    try:
        for idx, q in enumerate(questions):
            # Each question gets its own isolated SQLite DB
            db_path = tmp_root / f"q{idx}.db"
            await init_backend("sqlite", db_path=str(db_path))

            try:
                qr = await run_single_question(q, ks)
                results.append(qr)
            finally:
                await close_backend(str(db_path))

            # Progress
            if (idx + 1) % 50 == 0 or idx == len(questions) - 1:
                running_r5 = (
                    sum(1 for r in results if r.hits.get(5, False)) / len(results) * 100
                )
                print(
                    f"  [{idx + 1}/{len(questions)}] "
                    f"Running R@5: {running_r5:.1f}%  "
                    f"Last recall: {qr.recall_time_ms:.1f}ms"
                )
    finally:
        # Cleanup temp databases
        shutil.rmtree(tmp_root, ignore_errors=True)

    return results


# ============================================================
# Reporting
# ============================================================


def print_report(results: list[QuestionResult], ks: list[int]) -> dict:
    """Print a formatted report and return summary dict."""
    n = len(results)

    # Overall Recall@K
    print("\n" + "=" * 60)
    print("LongMemEval Benchmark Results — OpenDB Memory Pipeline")
    print("=" * 60)
    print(f"Questions evaluated: {n}")
    print()

    summary = {"total": n, "overall": {}, "by_type": {}}

    print("Overall Recall@K:")
    print("-" * 40)
    for k in ks:
        hit_count = sum(1 for r in results if r.hits.get(k, False))
        pct = hit_count / n * 100
        summary["overall"][f"R@{k}"] = round(pct, 1)
        marker = ""
        if k == 5:
            marker = "  ← compare vs MemPalace 96.6%"
        print(f"  R@{k:>2}: {pct:5.1f}% ({hit_count}/{n}){marker}")

    # By question type
    print()
    print("Recall@5 by Question Type:")
    print("-" * 50)
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)

    for qtype in sorted(by_type.keys()):
        type_results = by_type[qtype]
        hit5 = sum(1 for r in type_results if r.hits.get(5, False))
        pct = hit5 / len(type_results) * 100
        summary["by_type"][qtype] = {
            "count": len(type_results),
            "R@5": round(pct, 1),
        }
        print(f"  {qtype:<30} {pct:5.1f}% ({hit5}/{len(type_results)})")

    # Timing
    recall_times = [r.recall_time_ms for r in results]
    store_times = [r.store_time_ms for r in results]
    print()
    print("Timing:")
    print("-" * 40)
    print(f"  Recall  — median: {statistics.median(recall_times):.1f}ms, "
          f"mean: {statistics.mean(recall_times):.1f}ms, "
          f"p95: {sorted(recall_times)[int(len(recall_times) * 0.95)]:.1f}ms")
    print(f"  Store   — median: {statistics.median(store_times):.1f}ms, "
          f"mean: {statistics.mean(store_times):.1f}ms")

    summary["timing"] = {
        "recall_median_ms": round(statistics.median(recall_times), 1),
        "recall_mean_ms": round(statistics.mean(recall_times), 1),
        "store_median_ms": round(statistics.median(store_times), 1),
    }

    # Failure analysis (R@5 misses)
    misses = [r for r in results if not r.hits.get(5, False)]
    if misses:
        print()
        print(f"R@5 Misses ({len(misses)}):")
        print("-" * 50)
        miss_by_type = defaultdict(int)
        for m in misses:
            miss_by_type[m.question_type] += 1
        for qtype, count in sorted(miss_by_type.items(), key=lambda x: -x[1]):
            print(f"  {qtype:<30} {count}")
        # Show first few misses
        print()
        print("  Sample misses (first 5):")
        for m in misses[:5]:
            print(f"    {m.question_id} [{m.question_type}]")
            print(f"      expected: {m.answer_session_ids[:3]}")
            print(f"      recalled: {m.recalled_session_ids[:5]}")

    summary["misses"] = len(misses)

    print()
    print("=" * 60)
    return summary


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark for OpenDB memory pipeline"
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Path to longmemeval_oracle.json",
    )
    parser.add_argument(
        "--ks", default="1,3,5,10",
        help="Comma-separated K values for Recall@K (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (for quick testing)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: longmemeval_results.json)",
    )
    parser.add_argument(
        "--pooled", action="store_true",
        help=(
            "Measure retrieval against ONE corpus built from every question's "
            "sessions, so the other questions' sessions act as distractors. "
            "This is the mode that actually measures retrieval — the default "
            "per-question mode gives each question an empty database holding "
            "only its own gold sessions."
        ),
    )
    args = parser.parse_args()

    ks = sorted(int(k) for k in args.ks.split(","))
    data = load_dataset(args.data)

    t_start = time.time()
    runner = run_pooled_benchmark if args.pooled else run_benchmark
    results = asyncio.run(runner(data, ks, limit=args.limit))
    elapsed = time.time() - t_start

    summary = print_report(results, ks)
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Save detailed results
    output_path = args.output or str(BENCHMARK_DIR / "longmemeval_results.json")
    detail = {
        "summary": summary,
        "questions": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "hits": {str(k): v for k, v in r.hits.items()},
                "recall_time_ms": round(r.recall_time_ms, 2),
                "store_time_ms": round(r.store_time_ms, 2),
                "answer_session_ids": r.answer_session_ids,
                "recalled_session_ids": r.recalled_session_ids,
            }
            for r in results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {output_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
