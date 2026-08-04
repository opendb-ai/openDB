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

2. **The questions restate the evidence.** Measured over the 17 hand-authored
   facts, 60% of question tokens appear verbatim in the fact they ask about
   (65% when measured against the rendered sessions). A lexical retriever
   therefore wins part of the set by construction. `--overlap`
   measures this per item, and `PARAPHRASE_QUESTIONS` supplies zero-overlap
   restatements of the same facts so the two can be scored separately. A system
   that only scores well on the high-overlap half has been measured on the wrong
   thing.

3. **Generator, reader and judge are the same model family.** An LLM judge that
   shares a lineage with the answerer inflates scores, and nothing measured how
   often this one accepts a wrong answer. `ADVERSARIAL_ANSWERS` are hand-written
   wrong answers — plausible, specific, and false — that a judge must reject.
   `--judge-validation` reports the false-accept rate. Publish it even when it
   is bad; an unvalidated judge is not evidence.

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
# Keyed by the fact id in gen_codemem.FACTS. Each restates the question with
# minimal lexical overlap, so lexical and semantic retrieval can be scored
# apart instead of being averaged into one number that hides the difference.
PARAPHRASE_QUESTIONS: dict[str, str] = {
    "arch_billing_db": "Where do we keep money records, and what made us rule out the document store?",
    "arch_event_bus": "How do our components talk to each other asynchronously?",
    "arch_auth": "How does a request prove who it belongs to?",
    "arch_routing_cache": "Do we memoise path computations anywhere, and for how long?",
    "conv_error": "What shape should a failed response take?",
    "conv_naming": "How should I name a new endpoint entry point?",
    "conv_migrations": "What are the ground rules when altering the schema?",
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


def _load_facts() -> list[dict]:
    """Read FACTS out of gen_codemem without executing it.

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
            if isinstance(t, ast.Name) and t.id == "FACTS":
                return ast.literal_eval(node.value)
    raise RuntimeError("FACTS not found in gen_codemem.py")


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------

def report_overlap() -> dict:
    """How much of each question is already present in its own answer."""
    facts = _load_facts()
    rows = []
    for f in facts:
        q, ev = _tokens(f["question"]), _tokens(f["fact"])
        cover = len(q & ev) / len(q) if q else 0.0
        rows.append({"id": f["id"], "type": f["type"], "coverage": cover})

    para = []
    for fid, question in PARAPHRASE_QUESTIONS.items():
        f = next((x for x in facts if x["id"] == fid), None)
        if not f:
            continue
        q, ev = _tokens(question), _tokens(f["fact"])
        para.append({"id": fid, "coverage": len(q & ev) / len(q) if q else 0.0})

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
    print()
    print("  A lexical retriever wins part of the original split by construction.")
    print("  Score the two halves separately; a single average hides which one")
    print("  the system is actually good at.")
    return {"original": rows, "paraphrase": para}


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
    print("  Feed these to the judge alongside the gold answers. The false-accept")
    print("  rate is the number it marks correct, divided by the number here.")
    print("  Publish it even when it is bad: an unvalidated judge -- especially")
    print("  one sharing a model family with the answerer -- is not evidence.")
    return {"n": len(ADVERSARIAL_ANSWERS), "cases": ADVERSARIAL_ANSWERS}


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
    args = ap.parse_args()

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
