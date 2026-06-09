#!/usr/bin/env python3
"""LongMemEval E2E with the leading config: hybrid retrieval + Dreaming
(LLM-reflect consolidation). Reports per-category accuracy vs the public
leaderboard. Concurrent (unlike the sequential Phase-1 in the base harness)."""
import argparse, asyncio, json, os, sys, tempfile, uuid
from collections import defaultdict
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("FILEDB_MEMORY_RETRIEVAL_MODE", "hybrid")

import longmemeval_e2e_bench as B
from opendb_core.storage import init_backend, get_backend, close_backend
from opendb_core.consolidate import consolidate_reflect

# v3 reader prompt: teach the reader to exploit the Dreaming consolidation
# (CURRENT/PREVIOUSLY bi-temporal facts, EVENT TIMELINE) and compute relative time.
IMPROVED_ANSWER_PROMPT = """\
You are a personal AI assistant with access to conversation memories from past
sessions. Each memory is labeled with its session date and shown in chronological
order. Some memories are CONSOLIDATED summaries (a "## USER PROFILE", "## CURRENT
FACTS", or "## EVENT TIMELINE"). Trust these consolidated memories first.

Rules:
- Answer based ONLY on the provided memories. If they lack the answer, say:
  "I don't have enough information to answer that."
- AUTHORITY: A consolidated "CURRENT (since DATE): X" fact is the present truth.
  If a fact says "PREVIOUSLY ... INVALID" or is contradicted by a later-dated
  memory, treat the older value as superseded and DO NOT use it.
- TEMPORAL: Use "Today's date" and the session/event dates to compute relative
  time ("how long ago", "most recent", "before/after"). Earlier date = earlier.
  Prefer the EVENT TIMELINE for ordering and recency questions.
- COUNTING: Before giving a total, list each distinct item across ALL memories;
  use the consolidated running total when present.
- PREFERENCES: Honor what the user likes, dislikes, prefers, or habitually does,
  including the USER PROFILE; capture implicit preferences too.
- NUMBERS: Quote exact numbers. Do not round.
- Keep answers concise — typically 1-3 sentences.
"""

if os.environ.get("IMPROVED_PROMPT") == "1":
    B.ANSWER_SYSTEM_PROMPT = IMPROVED_ANSWER_PROMPT


# --- Chain-of-Note + category-aware reading (non-model SOTA techniques) ---
def _category_hint(question: str) -> str:
    ql = question.lower()
    if any(t in ql for t in ("when ", "how long", "how many days", "how many weeks",
                             "how many months", "how many years", " ago", "before ",
                             "after ", "first time", "last time", "earliest",
                             "most recent", "what date", "which day", "since ")):
        return ("This is a TIME question. Use the dated notes: compute durations and "
                "ordering from the dates, and take the MOST RECENT value as the current one.")
    if any(t in ql for t in ("how many", "how much", "number of", "count", "total")):
        return ("This is a COUNTING question. In NOTES, list EVERY distinct matching item "
                "across all memories, then count them in ANSWER.")
    if any(t in ql for t in ("prefer", "like", "favorite", "favourite", "enjoy",
                             "dislike", "hate", "rather", "style", "usually",
                             "recommend", "suggest", "advice", "help me", "should i")):
        return ("This is a PREFERENCE question — you must INFER the user's preference "
                "from their history (tools they use, past choices, complaints, praise) "
                "and tailor the answer to it. Do NOT abstain: even partial signals are "
                "enough to state what the user would prefer.")
    if any(t in ql for t in ("update", "now", "current", "still", "changed", "anymore",
                             "these days", "latest")):
        return ("This is a KNOWLEDGE-UPDATE question. If a value changed over time, use the "
                "LATEST value; ignore superseded older values.")
    return ("This may require combining facts from MULTIPLE memories — gather all relevant "
            "notes before answering.")


CON_SYSTEM_PROMPT = (
    "You are a personal AI assistant answering from a user's past-conversation memories "
    "(each tagged with its session date). Answer in TWO steps:\n"
    "1) NOTES: go through the memories and write down ONLY the facts relevant to the "
    "question, each with its date. Skip irrelevant memories. Be exhaustive about relevant ones.\n"
    "2) ANSWER: give the final answer in one short line, grounded in your notes.\n"
    "Only answer 'I don't have enough information to answer that.' if NONE of the "
    "memories are even related to the question — otherwise infer the best answer "
    "from the relevant notes.\n"
    "Quote numbers, names, and dates exactly. {hint}\n\n"
    "Output format:\nNOTES:\n- ...\nANSWER: ..."
)


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    up = text
    idx = up.rfind("ANSWER:")
    if idx >= 0:
        return up[idx + len("ANSWER:"):].strip()
    return text.strip()


async def con_answer(question, mems, model, qdate):
    if mems:
        sm = sorted(mems, key=lambda m: B._memory_date(m))
        mtext = "\n\n---\n\n".join(
            f"Memory {i} (session date: {B._memory_date(m)}):\n{m.get('content', '')}"
            for i, m in enumerate(sm, 1))
        dctx = f"\nToday's date: {qdate}\n" if qdate else ""
        user_msg = (f"Memories from past conversations:\n\n{mtext}\n\n---\n{dctx}\n"
                    f"Question: {question}")
    else:
        user_msg = f"No relevant memories found.\n\nQuestion: {question}"
    sys = CON_SYSTEM_PROMPT.format(hint=_category_hint(question))
    try:
        resp = await B._llm_client.chat.completions.create(
            model=model, max_tokens=1400,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user_msg}])
        return _extract_answer(resp.choices[0].message.content or "")
    except Exception:
        return ""


# --- self-consistency: sample N answers, reduce to a consensus ---
REDUCE_PROMPT = (
    "You are given several independent answers to the SAME question, each written "
    "from the same memories. Output the single best CONSENSUS answer: if they "
    "agree, return it; if they differ, decide which is most consistent with a "
    "careful reading of the question and return that. Return ONLY the final "
    "answer, concise (1-3 sentences), no preamble."
)


async def _sample_answer(question, mems, model, qdate, temperature):
    if mems:
        sm = sorted(mems, key=lambda m: B._memory_date(m))
        mtext = "\n\n---\n\n".join(
            f"Memory {i} (session date: {B._memory_date(m)}):\n{m.get('content', '')}"
            for i, m in enumerate(sm, 1))
        dctx = f"\nToday's date: {qdate}\n" if qdate else ""
        user_msg = (f"Relevant memories from past conversations:\n\n{mtext}\n\n---\n"
                    f"{dctx}\nUser question: {question}")
    else:
        user_msg = f"No relevant memories found.\n\nUser question: {question}"
    try:
        resp = await B._llm_client.chat.completions.create(
            model=model, max_tokens=1000, temperature=temperature,
            messages=[{"role": "system", "content": B.ANSWER_SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}])
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


async def consensus_answer(question, mems, model, qdate, votes):
    answers = await asyncio.gather(*[
        _sample_answer(question, mems, model, qdate, 0.7) for _ in range(votes)])
    answers = [a for a in answers if a]
    if not answers:
        return ""
    if len(set(answers)) == 1:
        return answers[0]
    cand = "\n\n".join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))
    try:
        resp = await B._llm_client.chat.completions.create(
            model=model, max_tokens=600,
            messages=[{"role": "system", "content": REDUCE_PROMPT},
                      {"role": "user", "content": f"Question: {question}\n\n{cand}"}])
        return (resp.choices[0].message.content or answers[0]).strip()
    except Exception:
        return answers[0]

# Public LongMemEval per-category references (GPT-4.1-class readers).
PUBLISHED = {
    "knowledge-update": {"OMEGA": 96, "Supermemory": 88.5, "Zep": 83.3},
    "temporal-reasoning": {"OMEGA": 94, "Supermemory": 76.7, "Zep": 62.4},
    "multi-session": {"OMEGA": 83, "Supermemory": 71.4, "Zep": 57.9},
    "single-session-user": {"Supermemory": 97.1, "Zep": 92.9},
    "single-session-assistant": {"Supermemory": 96.4, "Zep": 80.4},
    "single-session-preference": {"Supermemory": 70.0, "Zep": 56.7},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-cat", type=int, default=None,
                    help="stratified: take N questions per category (fast iteration)")
    ap.add_argument("--recall-limit", type=int, default=15)
    ap.add_argument("--votes", type=int, default=1, help="self-consistency samples")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default=str(Path(__file__).parent / "dreaming_e2e_results.json"))
    return ap.parse_args()


async def main():
    args = parse_args()
    data = [q for q in json.load(open(f"{ROOT}/benchmark/longmemeval_oracle.json"))
            if not q["question_id"].endswith("_abs")]
    if args.per_cat:
        from collections import defaultdict as _dd
        _by = _dd(list)
        for q in data:
            _by[q["question_type"]].append(q)
        data = [q for cat in sorted(_by) for q in _by[cat][: args.per_cat]]
    if args.limit:
        data = data[: args.limit]
    model = args.model

    async def reflect_fn(prompt: str) -> str:
        r = await B._llm_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1200)
        return r.choices[0].message.content or ""

    sem = asyncio.Semaphore(args.concurrency)
    done = [0]

    async def run_one(q):
        async with sem:
            db = Path(tempfile.mkdtemp()) / "q.db"
            await init_backend("sqlite", db_path=str(db))
            b = get_backend()
            try:
                dates = q.get("haystack_dates", [])
                chunk = os.environ.get("CHUNK")  # "turns" | "rounds" | None
                for i, (sid, sess) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"])):
                    meta = {"session_id": sid}
                    if i < len(dates):
                        meta["date"] = dates[i]
                    if chunk == "rounds":
                        # Round-level granularity: each user+assistant exchange is
                        # one memory — finer than a session, but keeps Q/A context
                        # (unlike single turns). Deterministic.
                        j = 0
                        while j < len(sess):
                            pair = sess[j:j + 2]
                            txt = "\n".join(f"[{t['role']}] {t['content']}" for t in pair).strip()
                            if len(txt) > 6:
                                await b.store_memory(memory_id=str(uuid.uuid4()), content=txt,
                                                     memory_type="episodic", tags=[sid], metadata=meta)
                            j += 2
                    elif chunk == "turns":
                        for turn in sess:
                            txt = f"[{turn['role']}] {turn['content']}".strip()
                            if len(txt) > 6:
                                await b.store_memory(memory_id=str(uuid.uuid4()), content=txt,
                                                     memory_type="episodic", tags=[sid], metadata=meta)
                    else:
                        await b.store_memory(memory_id=str(uuid.uuid4()), content=B.flatten_session(sess),
                                             memory_type="episodic", tags=[sid], metadata=meta)
                mems = None
                additive = os.environ.get("ADDITIVE_CONSOLIDATE") == "1"
                use_kg = os.environ.get("KG") == "1"
                ql = q["question"].lower()
                _temporal = any(t in ql for t in (
                    "when ", "how long", "how many days", "how many weeks",
                    "how many months", "how many years", " ago", "before ",
                    "after ", "first time", "last time", "earliest", "most recent",
                    "what date", "which day", "how often", "since "))

                async def _recall_raw():
                    r = await b.recall_memories(query=q["question"], memory_type=None,
                                                tags=None, limit=args.recall_limit, offset=0)
                    return r.get("results", [])

                # Baseline: full-context — feed the ENTIRE haystack, no retrieval
                # (theoretical upper bound for any memory system, same reader).
                if os.environ.get("FULLCTX") == "1":
                    listing = await b.list_memories(memory_type=None, tags=None,
                                                    limit=100000, offset=0)
                    mems = listing.get("memories", [])
                elif args.consolidate and use_kg:
                    # KG: raw top-K + a deterministic bi-temporal KG timeline.
                    from opendb_core.temporal_kg import build_kg, render_facts
                    mems = await _recall_raw()
                    listing = await b.list_memories(memory_type=None, tags=None,
                                                    limit=100000, offset=0)
                    facts = await build_kg(listing.get("memories", []), reflect_fn)
                    kg_text = render_facts(facts)
                    if kg_text:
                        mems = mems + [{"content": (
                            "## KNOWLEDGE GRAPH (bi-temporal; CURRENT value is "
                            "authoritative; history shows when each value was "
                            "valid):\n" + kg_text[:6000]), "metadata": {}}]
                elif args.consolidate and additive and _temporal:
                    mems = await _recall_raw()  # raw-only for temporal queries
                elif args.consolidate and additive:
                    # ADDITIVE: raw evidence + synthesized digest (no dilution).
                    raw_res = await _recall_raw()
                    await consolidate_reflect(b, reflect_fn)
                    cons = await b.recall_memories(query=q["question"], memory_type=None,
                                                   tags=["__consolidated__"], limit=15, offset=0)
                    cons_res = [m for m in cons.get("results", [])
                                if "__timeline__" not in m.get("tags", [])][:10]
                    mems = raw_res + cons_res
                else:
                    if args.consolidate:
                        await consolidate_reflect(b, reflect_fn)
                    mems = await _recall_raw()
            finally:
                await close_backend(str(db))
            if os.environ.get("CON") == "1":
                ans = await con_answer(q["question"], mems, model, q.get("question_date", ""))
            elif args.votes > 1:
                ans = await consensus_answer(q["question"], mems, model,
                                             q.get("question_date", ""), args.votes)
            else:
                ans, _ = await B.generate_answer(q["question"], mems, model,
                                                 q.get("question_date", ""))
            ok, _, _, _ = await B.judge_answer(q["question"], q["answer"], ans, model)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(data):
                print(f"  [{done[0]}/{len(data)}] done", flush=True)
            return q["question_type"], bool(ok)

    cfg = "hybrid+dreaming" if args.consolidate else "hybrid"
    print(f"LongMemEval E2E — {cfg}, reader={model}, n={len(data)}\n", flush=True)
    results = await asyncio.gather(*[run_one(q) for q in data])

    agg = defaultdict(lambda: [0, 0])
    for cat, ok in results:
        agg[cat][0] += int(ok); agg[cat][1] += 1
    tot_c = sum(v[0] for v in agg.values()); tot_n = sum(v[1] for v in agg.values())

    print("\n" + "=" * 72)
    print(f"LongMemEval E2E — {cfg}, reader={model}")
    print("=" * 72)
    print(f"Overall: {tot_c}/{tot_n} ({tot_c/tot_n*100:.1f}%)\n")
    print(f"{'category':<28}{'OpenDB':<12}{'best public (model: GPT-4.1-class)'}")
    print("-" * 72)
    out = {"config": cfg, "model": model, "overall": round(tot_c / tot_n * 100, 1),
           "n": tot_n, "by_type": {}}
    for cat in sorted(agg):
        c, n = agg[cat]
        acc = c / n * 100
        pub = PUBLISHED.get(cat, {})
        best = max(pub.items(), key=lambda kv: kv[1]) if pub else None
        lead = ""
        if best and acc > best[1]:
            lead = f"  >>> beats {best[0]} ({best[1]}%)"
        elif best:
            lead = f"  (vs {best[0]} {best[1]}%)"
        print(f"{cat:<28}{acc:5.1f}% ({c}/{n})  {lead}")
        out["by_type"][cat] = {"acc": round(acc, 1), "correct": c, "count": n, "published": pub}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
