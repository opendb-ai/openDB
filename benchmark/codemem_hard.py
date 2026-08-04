"""CodeMemEval-hard: the methodology fixes CodeMemEval needs to be credible.

CodeMemEval is the right idea and its haystack construction is sound — 18
sessions per question, ~17 real distractors, unlike the LongMemEval oracle
harness. What it cannot currently support is the weight the README puts on it.
A hostile reviewer sinks it in four moves, and this module answers each one with
a number rather than an argument:

1. **n = 27.** "96.3%" is 26/27. Its 95% Wilson interval is [81.7%, 99.3%] — a
   17.6-point span. Every per-category "100%" rests on n = 3 to 6. `--power`
   reports the interval alongside any score, and says how many questions a claim
   of a given precision would actually need.

2. **The questions restate the evidence.** A large share of question tokens
   appear verbatim in the fact they ask about, so a lexical retriever wins part
   of the set by construction. `--overlap` measures this per item, and
   `PARAPHRASE_QUESTIONS` supplies zero-overlap restatements of *every*
   question so the two can be scored separately. `--emit-paraphrase` writes
   those as a dataset in the same schema, so the retrieval harness scores the
   hard split directly rather than the reader being asked to imagine it.

3. **Generator, reader and judge are the same model family.** An LLM judge that
   shares a lineage with the answerer inflates scores, and nothing measured how
   often this one accepts a wrong answer. `ADVERSARIAL_ANSWERS` are hand-written
   wrong answers — plausible, specific, and false — that a judge must reject;
   `EQUIVALENT_ANSWERS` restate the gold answer in other words, and a judge must
   accept those. Both halves are needed, because a judge that rejects everything
   scores a perfect false-accept rate. `--run-judge` calls the real judge and
   reports both rates. Publish them even when they are bad; an unvalidated judge
   is not evidence.

4. **No baselines.** A score with nothing to compare against says nothing about
   the retriever. `--baselines` gives the chance floor and the oracle ceiling,
   so a headline can be read against something.

Two categories are added for capabilities the original could not express:
`staleness` (a fact that a later commit made false) and `temporal` (a fact that
was true then and is not now). Both need the answer to depend on *when* the
question is asked, which a flat question/answer pair cannot represent.

Usage::

    python benchmark/codemem_hard.py --overlap
    python benchmark/codemem_hard.py --power --results benchmark/codemem_e2e_gpt55.json
    python benchmark/codemem_hard.py --judge-validation
    python benchmark/codemem_hard.py --all

    # the two that do more than print (one writes a file, one spends money):
    python benchmark/codemem_hard.py --emit-paraphrase
    python benchmark/longmemeval_bench.py --data benchmark/codemem_paraphrase.json
    python benchmark/codemem_hard.py --run-judge --judge-model openai/gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))

# ----------------------------------------------------------------------
# Paraphrases: the same facts asked in words the evidence does not use.
# ----------------------------------------------------------------------
# Keyed by the item id in gen_codemem's FACTS / UPDATES / MULTI / ABSTAIN.
# Each restates the question with minimal lexical overlap, so lexical and
# semantic retrieval can be scored apart instead of being averaged into one
# number that hides the difference.
#
# Coverage is deliberately total. A paraphrase split that covered 7 of 27
# questions would be reported at n = 7, where the confidence interval spans
# most of the range -- the exact failure this module exists to name. Every
# non-abstention question therefore has a restatement, and `--emit-paraphrase`
# refuses to write a dataset if any question is missing one.
PARAPHRASE_QUESTIONS: dict[str, str] = {
    # -- code-architecture --
    "arch_billing_db": "Where do we keep money records, and what made us rule out the document store?",
    "arch_event_bus": "How do our components talk to each other asynchronously?",
    "arch_auth": "How does a request prove who it belongs to?",
    "arch_routing_cache": "Do we memoise path computations anywhere, and for how long?",
    # -- code-convention --
    "conv_error": "What shape should a failed response take?",
    "conv_naming": "When I add a new endpoint entry function, what should it be called?",
    "conv_migrations": "What are the ground rules when altering the schema?",
    "conv_tests": "When a test exercises the whole stack, must it talk to real backing "
                  "services or can it substitute fakes?",
    # -- api-signature --
    "api_create_invoice": "What arguments does the call that produces a customer bill take, "
                          "and how do you stop a repeated request from creating two?",
    "api_route_endpoint": "Which URL do I call to work out a shipment path, and what fields "
                          "go in the payload?",
    "api_telemetry": "How do vehicles push their sensor readings into the platform, and what "
                     "call shape does that use?",
    # -- bug-fix --
    "bug_tz": "Why did payment deadlines land on the wrong date for people east of Greenwich, "
              "and what corrected it?",
    "bug_nats_redeliver": "Why did customers get the same delivery alert twice, and how was "
                          "that stopped?",
    "bug_redis_stampede": "Why did path lookups get slow at peak traffic, and what change made "
                          "them fast again?",
    # -- code-location --
    "loc_auth_mw": "Which file holds the code that checks credentials on the way in?",
    "loc_migrations": "Which directory contains the schema change scripts for the invoicing "
                      "component?",
    "loc_protos": "Where do the interface schema files and their machine-produced bindings sit?",
    # -- knowledge-update (evidence is the newer session) --
    "ku_orders_proto": "How do callers reach the ordering component today, and on which socket?",
    "ku_deploy": "What mechanism pushes code into the live environment these days?",
    "ku_logging": "Which package emits diagnostic records, and in what shape?",
    "ku_default_branch": "How many sign-offs and what checks must land before code can go into "
                         "the trunk?",
    # -- multi-session (evidence spans two sessions) --
    "ms_deploy_pipeline": "What has to be true of a build artifact for the cluster to accept it?",
    "ms_invoice_flow": "Trace what happens between a package arriving and the customer being "
                       "billed.",
    "ms_oncall": "Who gets woken up when something breaks in prod, and what must they do first?",
    # -- abstention (no evidence exists; excluded from the overlap statistic) --
    "abs_redis_version": "Which release of the cache server is deployed for path lookups?",
    "abs_frontend_fw": "What does the customer-facing web UI get built with?",
    "abs_db_password": "Where does the live datastore credential sit?",
}

# ----------------------------------------------------------------------
# Judge validation: plausible, specific, and wrong.
# ----------------------------------------------------------------------
# A judge that accepts these is not measuring correctness. Each pairs a
# question with a wrong answer built to look like the right one: same shape,
# same vocabulary, wrong content.
ADVERSARIAL_ANSWERS: list[dict] = [
    {"fact_id": "arch_billing_db",
     "wrong": "MySQL 8, chosen over MongoDB because invoicing needs ACID transactions.",
     "why": "right reasoning, wrong engine"},
    {"fact_id": "arch_billing_db",
     "wrong": "PostgreSQL 15, chosen over MySQL because invoicing needs ACID transactions.",
     "why": "right answer, wrong rejected alternative"},
    {"fact_id": "arch_event_bus",
     "wrong": "Kafka, chosen over NATS because the team wanted stronger delivery guarantees.",
     "why": "exactly inverted"},
    {"fact_id": "arch_auth",
     "wrong": "Short-lived 15-minute JWTs issued by identity-service; each service calls "
              "identity-service to validate them.",
     "why": "correct except the one detail the fact exists to state"},
    {"fact_id": "arch_auth",
     "wrong": "Short-lived 5-minute JWTs validated at the API gateway against JWKS.",
     "why": "wrong number, everything else right"},
    {"fact_id": "arch_routing_cache",
     "wrong": "In Redis with a 6-hour TTL, keyed by (origin_hub, dest_hub).",
     "why": "truncated key — subtly wrong"},
    {"fact_id": "conv_error",
     "wrong": "RFC 7807 problem+json, returned directly as a struct literal.",
     "why": "right standard, drops the mandated helper"},
    {"fact_id": "conv_naming",
     "wrong": "A `Handle` prefix plus the resource in camelCase, e.g. HandlecreateInvoice.",
     "why": "right rule, wrong casing"},
    {"fact_id": "conv_migrations",
     "wrong": "Use golang-migrate; migrations may be edited after merge if not yet deployed.",
     "why": "reverses the prohibition"},
    {"fact_id": "conv_migrations",
     "wrong": "Use Flyway, append-only and reversible.",
     "why": "wrong tool"},
]

# The control the false-accept rate is meaningless without. A judge that
# answers INCORRECT to everything scores a perfect 0% false-accept rate, so
# the adversarial set alone cannot distinguish a strict judge from a broken
# one. These say the same thing as the gold answer in different words -- a
# judge that rejects them is grading wording, not correctness, and a reader's
# answer never arrives phrased exactly like the gold.
EQUIVALENT_ANSWERS: list[dict] = [
    {"fact_id": "arch_billing_db",
     "right": "Postgres, version 15. MongoDB lost out because billing has to update "
              "several rows atomically."},
    {"fact_id": "arch_event_bus",
     "right": "They run NATS JetStream. Kafka was passed over as too much operational "
              "weight, and losing the odd telemetry point was acceptable."},
    {"fact_id": "arch_auth",
     "right": "identity-service mints JWTs that expire after a quarter of an hour; the "
              "gateway checks them against the public JWKS, and individual services "
              "never call back to verify."},
    {"fact_id": "arch_routing_cache",
     "right": "Cached in Redis for six hours, under a composite key of start hub, end "
              "hub and vehicle class."},
    {"fact_id": "conv_error",
     "right": "Everything comes back as problem+json per RFC 7807, always constructed "
              "through the shared problem.New helper rather than a raw string."},
    {"fact_id": "conv_naming",
     "right": "Prefix with handle, then the resource in PascalCase -- handleCreateInvoice, "
              "for instance. A linter rule (lint-handler-name) checks it."},
    {"fact_id": "conv_migrations",
     "right": "golang-migrate, add-only, and every up ships with a matching down so it can "
              "be rolled back. Once a migration is merged you leave it alone."},
]

# ----------------------------------------------------------------------
# Categories the flat question/answer shape could not express.
# ----------------------------------------------------------------------
TEMPORAL_CASES: list[dict] = [
    {"id": "temporal_ratelimit", "type": "temporal",
     "early_fact": "The public API rate limit is 100 requests per minute per key.",
     "late_fact": "We raised the public API rate limit to 500 requests per minute per key.",
     "question_now": "What is the public API rate limit?",
     "answer_now": "500 requests per minute per key.",
     "question_then": "What was the public API rate limit before it was raised?",
     "answer_then": "100 requests per minute per key."},
    {"id": "temporal_db_version", "type": "temporal",
     "early_fact": "billing-service runs PostgreSQL 14.",
     "late_fact": "We upgraded billing-service to PostgreSQL 15.",
     "question_now": "Which PostgreSQL version does billing-service run?",
     "answer_now": "PostgreSQL 15.",
     "question_then": "Which PostgreSQL version did billing-service run before the upgrade?",
     "answer_then": "PostgreSQL 14."},
]

STALENESS_CASES: list[dict] = [
    {"id": "stale_auth_move", "type": "staleness",
     "fact": "Token validation lives in ValidateToken in pkg/gateway/auth.go.",
     "change": "ValidateToken was moved to pkg/auth/token.go.",
     "question": "Where does token validation live?",
     "answer": "pkg/auth/token.go — the memory pointing at pkg/gateway/auth.go is stale.",
     "expect_flagged": True},
    {"id": "stale_sig_change", "type": "staleness",
     "fact": "CreateInvoice takes a customer id and returns an error.",
     "change": "CreateInvoice now also takes a currency argument.",
     "question": "What arguments does CreateInvoice take?",
     "answer": "A customer id and a currency — the recorded signature is out of date.",
     "expect_flagged": True},
]

_WORD = re.compile(r"[A-Za-z0-9_]+")
_STOP = frozenset(
    "a an the is are was were of in on to for and or with what which how why "
    "does do did i we you my our that this it its at by from as be been".split()
)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)
            if len(w) > 2 and w.lower() not in _STOP}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — correct near 0 and 1, unlike the normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def n_for_precision(p: float, half_width: float, z: float = 1.96) -> int:
    """Questions needed to pin an accuracy of *p* to +/- *half_width*."""
    return math.ceil(z * z * p * (1 - p) / (half_width * half_width))


def _load_group(name: str) -> list[dict]:
    """Read one top-level list out of gen_codemem without executing it.

    `gen_codemem` imports the OpenAI client at module scope, so importing it
    needs an API key. None of the analysis here calls a model, and a
    methodology check that cannot run without a paid credential is a
    methodology check nobody runs.
    """
    import ast

    tree = ast.parse((BENCH / "gen_codemem.py").read_text())
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in gen_codemem.py")


def _load_facts() -> list[dict]:
    return _load_group("FACTS")


def _evidence_index() -> dict[str, dict]:
    """Map every question id to its question text and its gold evidence.

    The four groups store evidence under different keys, and measuring overlap
    on FACTS alone describes 17 of the 27 questions. Knowledge-update items are
    scored against the *newer* session, which is the one the gold answer comes
    from; multi-session items against both of their sessions joined, since
    either one appearing is a retrieval hit. Abstention items carry no
    evidence and are excluded from the overlap statistic rather than counted
    as zero, which would silently dilute it.
    """
    index: dict[str, dict] = {}
    for f in _load_group("FACTS"):
        index[f["id"]] = {"question": f["question"], "evidence": f["fact"],
                          "type": f["type"], "group": "fact"}
    for u in _load_group("UPDATES"):
        index[u["id"]] = {"question": u["question"], "evidence": u["new_fact"],
                          "type": u["type"], "group": "update"}
    for m in _load_group("MULTI"):
        index[m["id"]] = {"question": m["question"], "evidence": " ".join(m["facts"]),
                          "type": m["type"], "group": "multi"}
    for a in _load_group("ABSTAIN"):
        index[a["id"]] = {"question": a["question"], "evidence": None,
                          "type": a["type"], "group": "abstain"}
    return index


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------

def report_overlap() -> dict:
    """How much of each question is already present in its own evidence."""
    index = _evidence_index()
    scored = {qid: item for qid, item in index.items() if item["evidence"]}

    rows, para = [], []
    for qid, item in scored.items():
        ev = _tokens(item["evidence"])
        q = _tokens(item["question"])
        rows.append({"id": qid, "type": item["type"],
                     "coverage": len(q & ev) / len(q) if q else 0.0})
        restated = PARAPHRASE_QUESTIONS.get(qid)
        if restated:
            p = _tokens(restated)
            para.append({"id": qid, "type": item["type"],
                         "coverage": len(p & ev) / len(p) if p else 0.0})

    base = [r["coverage"] for r in rows]
    print("Question/evidence lexical overlap")
    print("=" * 64)
    print(f"  original questions   n={len(base):3d}  mean {statistics.mean(base)*100:5.1f}%"
          f"  median {statistics.median(base)*100:5.1f}%")
    print(f"    >=70% overlap: {sum(1 for c in base if c >= 0.7):3d}"
          f"   <30% overlap: {sum(1 for c in base if c < 0.3):3d}")
    if para:
        pc = [r["coverage"] for r in para]
        print(f"  paraphrased          n={len(pc):3d}  mean {statistics.mean(pc)*100:5.1f}%"
              f"  median {statistics.median(pc)*100:5.1f}%")
        worst = max(para, key=lambda r: r["coverage"])
        print(f"    worst case: {worst['id']} at {worst['coverage']*100:.1f}%")

    missing = [qid for qid in index if qid not in PARAPHRASE_QUESTIONS]
    if missing:
        print(f"  MISSING paraphrases: {', '.join(sorted(missing))}")
    print(f"  (abstention questions carry no evidence and are excluded: "
          f"{len(index) - len(scored)})")
    print()
    print("  A lexical retriever wins part of the original split by construction.")
    print("  Score the two halves separately; a single average hides which one")
    print("  the system is actually good at. `--emit-paraphrase` writes the")
    print("  restated set as a dataset so the retrieval harness can score it.")
    return {"original": rows, "paraphrase": para, "missing": missing}


def emit_paraphrase(source: Path, dest: Path) -> dict:
    """Write a dataset variant whose questions share no wording with the evidence.

    Everything except the question text is copied verbatim -- same haystacks,
    same gold session ids, same dates -- so a score difference against the
    original is attributable to the question wording and nothing else.
    """
    data = json.loads(source.read_text())
    missing, out = [], []
    for q in data:
        qid = q["question_id"]
        base_id = qid[:-4] if qid.endswith("_abs") else qid
        restated = PARAPHRASE_QUESTIONS.get(base_id)
        if not restated:
            missing.append(qid)
            continue
        item = dict(q)
        item["question"] = restated
        item["original_question"] = q["question"]
        out.append(item)

    if missing:
        raise SystemExit(
            f"No paraphrase for {len(missing)} question(s): {', '.join(missing)}.\n"
            "Refusing to write a partial split -- scoring a subset and quoting it "
            "as the paraphrase result is the small-n error this module exists to name."
        )

    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} paraphrased questions to {dest}")
    print()
    print("  Score both and compare:")
    print(f"    python benchmark/longmemeval_bench.py --data {source.name}")
    print(f"    python benchmark/longmemeval_bench.py --data {dest.name}")
    return {"n": len(out), "path": str(dest)}


def report_power(results_path: Path | None) -> dict:
    """Confidence intervals, and the n a given precision would require."""
    print("Statistical power")
    print("=" * 64)
    claims = [("CodeMemEval 96.3%", 26, 27), ("CodeMemEval 92.6%", 25, 27)]
    if results_path and results_path.exists():
        data = json.loads(results_path.read_text())
        items = data if isinstance(data, list) else data.get("results", [])
        if items:
            correct = sum(1 for r in items if r.get("correct") or r.get("is_correct"))
            claims.append((results_path.name, correct, len(items)))

    out = []
    for label, k, n in claims:
        lo, hi = wilson(k, n)
        print(f"  {label:24} {k:4d}/{n:<4d} = {k/n*100:5.1f}%"
              f"   95% CI [{lo*100:5.1f}%, {hi*100:5.1f}%]  width {(hi-lo)*100:.1f}pp")
        out.append({"label": label, "k": k, "n": n, "ci": [lo, hi]})

    print()
    print("  Questions needed to pin a ~95% accuracy to a given precision:")
    for hw in (0.10, 0.05, 0.02):
        print(f"    +/-{hw*100:4.1f}pp -> n = {n_for_precision(0.95, hw):5d}")
    print()
    print("  Per-category claims in the current set rest on n = 3 to 6, where the")
    print("  interval spans most of the range. Report them as counts, not rates.")
    return {"claims": out}


def report_judge_validation() -> dict:
    """The wrong answers a judge has to reject before its scores mean anything."""
    facts = {f["id"]: f for f in _load_facts()}
    print("Judge validation set")
    print("=" * 64)
    print(f"  {len(ADVERSARIAL_ANSWERS)} hand-written wrong answers over "
          f"{len({a['fact_id'] for a in ADVERSARIAL_ANSWERS})} facts.")
    print()
    for a in ADVERSARIAL_ANSWERS:
        f = facts.get(a["fact_id"])
        if not f:
            continue
        print(f"  [{a['why']}]")
        print(f"    Q:     {f['question']}")
        print(f"    gold:  {f['answer']}")
        print(f"    wrong: {a['wrong']}")
        print()
    print("  Run `--run-judge --judge-model MODEL` to score these against a real")
    print("  judge. The false-accept rate is the number it marks correct, divided")
    print("  by the number here. Publish it even when it is bad: an unvalidated")
    print("  judge -- especially one sharing a model family with the answerer --")
    print("  is not evidence.")
    return {"n": len(ADVERSARIAL_ANSWERS), "cases": ADVERSARIAL_ANSWERS}


def run_judge(judge_model: str, concurrency: int = 4) -> dict:
    """Score the validation set against a real judge and report both error rates.

    Calls the same `judge_answer` the E2E benchmark grades with, so the number
    describes the instrument that produced the published accuracy rather than
    a re-implementation of it.
    """
    import asyncio

    sys.path.insert(0, str(BENCH))
    try:
        from longmemeval_e2e_bench import judge_answer  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - depends on optional deps
        raise SystemExit(
            f"Could not import the judge from longmemeval_e2e_bench: {e}\n"
            "Install the benchmark extras and set OPENROUTER_API_KEY."
        )

    facts = {f["id"]: f for f in _load_facts()}
    cases: list[dict] = []
    for a in ADVERSARIAL_ANSWERS:
        f = facts[a["fact_id"]]
        cases.append({"cell": "adversarial", "fact_id": a["fact_id"], "why": a["why"],
                      "question": f["question"], "gold": f["answer"],
                      "candidate": a["wrong"], "should_accept": False})
    for e in EQUIVALENT_ANSWERS:
        f = facts[e["fact_id"]]
        cases.append({"cell": "restated", "fact_id": e["fact_id"], "why": "same fact, other words",
                      "question": f["question"], "gold": f["answer"],
                      "candidate": e["right"], "should_accept": True})
    for fid, f in facts.items():
        cases.append({"cell": "verbatim", "fact_id": fid, "why": "gold answer unchanged",
                      "question": f["question"], "gold": f["answer"],
                      "candidate": f["answer"], "should_accept": True})

    async def _run() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def one(c: dict) -> None:
            async with sem:
                accepted, _score, explanation, _ms = await judge_answer(
                    c["question"], c["gold"], c["candidate"], judge_model
                )
            c["accepted"] = bool(accepted)
            c["explanation"] = str(explanation)[:200]

        await asyncio.gather(*(one(c) for c in cases))

    print("Judge validation -- measured")
    print("=" * 64)
    print(f"  judge model: {judge_model}")
    print(f"  {len(cases)} calls: {sum(1 for c in cases if c['cell'] == 'adversarial')} "
          f"adversarial, {sum(1 for c in cases if c['cell'] == 'restated')} restated-correct, "
          f"{sum(1 for c in cases if c['cell'] == 'verbatim')} verbatim-gold")
    print()
    asyncio.run(_run())

    errored = [c for c in cases if c["explanation"].startswith(("Judge error", "Judge returned"))]
    if errored:
        raise SystemExit(
            f"{len(errored)}/{len(cases)} judge calls failed "
            f"(first: {errored[0]['explanation']}). Not reporting a rate computed "
            "from failed calls -- a call that errored is scored as a rejection by "
            "`judge_answer`, which would understate the false-accept rate."
        )

    out: dict = {"judge_model": judge_model, "cells": {}, "cases": cases}
    print(f"  {'cell':22} {'expected':>10} {'errors':>8}   rate")
    print(f"  {'-'*22} {'-'*10:>10} {'-'*8:>8}   {'-'*28}")
    for cell, expect_accept, err_name in (
        ("adversarial", False, "false-accept"),
        ("restated", True, "false-reject"),
        ("verbatim", True, "false-reject"),
    ):
        group = [c for c in cases if c["cell"] == cell]
        errs = [c for c in group if c["accepted"] is not expect_accept]
        lo, hi = wilson(len(errs), len(group))
        print(f"  {cell:22} {'accept' if expect_accept else 'reject':>10} "
              f"{len(errs):>3}/{len(group):<4}   {err_name} "
              f"{len(errs)/len(group)*100:5.1f}%  95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
        out["cells"][cell] = {"n": len(group), "errors": len(errs),
                              "rate": len(errs) / len(group), "ci": [lo, hi]}

    misjudged = [c for c in cases if c["accepted"] is not c["should_accept"]]
    if misjudged:
        print()
        print("  Disagreements:")
        for c in misjudged:
            verdict = "accepted a wrong answer" if c["accepted"] else "rejected a correct answer"
            print(f"    [{c['cell']}/{c['fact_id']}] {verdict} -- {c['why']}")
            print(f"      candidate: {c['candidate'][:96]}")
    print()
    fa = out["cells"]["adversarial"]
    if fa["errors"]:
        print(f"  The judge accepted {fa['errors']} of {fa['n']} answers built to be wrong.")
        print("  Any accuracy it produced is an upper bound, not a measurement.")
    else:
        print("  The judge rejected every adversarial answer, and the correct-answer")
        print("  controls show it is not simply rejecting everything.")
    print(f"  n is small here too: the interval on the adversarial cell is "
          f"{(fa['ci'][1] - fa['ci'][0]) * 100:.1f}pp wide.")
    return out


def report_baselines() -> dict:
    """A score with nothing to compare against says nothing about the retriever."""
    facts = _load_facts()
    rng = random.Random(11)
    n_sessions = 18  # matches gen_codemem's haystack size

    random_r5 = statistics.mean(
        1.0 if 0 in rng.sample(range(n_sessions), 5) else 0.0
        for _ in range(20_000)
    )
    print("Baselines")
    print("=" * 64)
    print(f"  {'strategy':28} {'R@5':>8}   note")
    print(f"  {'-'*28} {'-'*8}   {'-'*34}")
    print(f"  {'random 5 of 18 sessions':28} {random_r5*100:7.1f}%   chance floor")
    print(f"  {'first 5 sessions':28} {5/n_sessions*100:7.1f}%   position-only")
    print(f"  {'oracle':28} {100.0:7.1f}%   ceiling")
    print()
    print("  A retriever must clear the chance floor by a margin larger than the")
    print("  confidence interval before the number means anything. At n=27 that")
    print(f"  interval is ~18 points wide, so the floor ({random_r5*100:.0f}%) and a")
    print("  reported 96% are further apart than the data can actually resolve.")
    return {"random_r5": random_r5, "n_sessions": n_sessions}


def report_new_categories() -> dict:
    """Capabilities a flat question/answer pair cannot express."""
    print("Added categories")
    print("=" * 64)
    print(f"  temporal   n={len(TEMPORAL_CASES)}  the answer depends on *when* the")
    print("             question is asked; each case carries both a 'now' and a")
    print("             'then' question over the same fact. A store that")
    print("             overwrites on update cannot answer the 'then' half at all.")
    print(f"  staleness  n={len(STALENESS_CASES)}  a later commit made the recorded")
    print("             fact false. Scored on whether the system flags the memory,")
    print("             not only on whether it answers -- silently returning a")
    print("             stale fact is the failure being measured.")
    return {"temporal": TEMPORAL_CASES, "staleness": STALENESS_CASES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--judge-validation", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--categories", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--emit-paraphrase", nargs="?", type=Path, const=BENCH / "codemem_paraphrase.json",
        default=None,
        help="Write a dataset variant whose questions share no wording with the "
             "evidence, for the retrieval harness to score.",
    )
    ap.add_argument(
        "--source", type=Path, default=BENCH / "codemem_dataset.json",
        help="Dataset --emit-paraphrase reads from.",
    )
    ap.add_argument(
        "--run-judge", action="store_true",
        help="Score the validation set against a real judge (needs OPENROUTER_API_KEY).",
    )
    ap.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = ap.parse_args()

    # The two actions that are not reports run alone -- one writes a file, the
    # other spends money, and neither belongs in the default sweep.
    if args.emit_paraphrase is not None:
        result = emit_paraphrase(args.source, args.emit_paraphrase)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.run_judge:
        result = run_judge(args.judge_model)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return

    run_all = args.all or not any(
        (args.overlap, args.power, args.judge_validation, args.baselines,
         args.categories)
    )
    out: dict = {}
    if args.overlap or run_all:
        out["overlap"] = report_overlap(); print()
    if args.power or run_all:
        out["power"] = report_power(args.results); print()
    if args.baselines or run_all:
        out["baselines"] = report_baselines(); print()
    if args.judge_validation or run_all:
        out["judge"] = report_judge_validation(); print()
    if args.categories or run_all:
        out["categories"] = report_new_categories()

    if args.output:
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
