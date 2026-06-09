"""Local bi-temporal knowledge graph for agent memory.

Zep/Graphiti showed that a *temporal* knowledge graph — facts with explicit
validity windows (when a fact became true, when it was superseded) — is the
state of the art for temporal-reasoning and knowledge-update. Those systems are
cloud-hosted and graph-DB-backed.

This is OpenDB's version, and the thing the closed/cloud frontier can't offer:
**local, deterministic, and fully auditable.** Facts live in a plain SQLite-able
table; the invalidation and query logic is pure Python (deterministic, testable);
every fact keeps the source memory ids it was derived from. Only the *extraction*
step uses an LLM (or can be swapped for rules) — everything after it is reproducible.

Model:
  A fact is ``(subject, attribute, value, valid_from, valid_to, sources)``.
  * ``valid_from``  — when the fact became true (event date).
  * ``valid_to``    — when it was superseded by a newer value (None = still current).
  Bi-temporal invalidation: for each ``(subject, attribute)`` we order facts by
  ``valid_from``; each older value's ``valid_to`` is set to the next value's
  ``valid_from``; the newest value stays current (``valid_to = None``).

This makes three queries first-class and exact:
  * "current value of X"      → the fact with ``valid_to is None``
  * "value of X at time T"    → the fact with ``valid_from <= T < valid_to``
  * "when did X change"       → the ordered list of ``valid_from`` boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field

EXTRACT_PROMPT = (
    "You convert a user's dated conversation history into structured KNOWLEDGE "
    "GRAPH facts for temporal reasoning. Read the dated sessions below and emit "
    "one fact per line in EXACTLY this pipe format:\n"
    "  subject | attribute | value | YYYY-MM-DD\n\n"
    "Rules:\n"
    "- subject: who/what the fact is about (e.g. 'user', 'user's car', a named "
    "person/project).\n"
    "- attribute: the changing property (e.g. '5k_personal_best', 'city', "
    "'job_title', 'car', 'restaurants_tried_count').\n"
    "- value: the value at that time (keep numbers/names exact).\n"
    "- date: the event date (YYYY-MM-DD) from the session this fact came from.\n"
    "- Emit a SEPARATE line for every time a value was stated or changed, so "
    "evolution over time is captured (do NOT collapse history).\n"
    "- For running totals (things tried/owned/visited), emit the count AS OF each "
    "date.\n"
    "- Capture EVERY concrete fact, preference, event, number, and name. Aim for "
    "many lines — do not be terse.\n"
    "- No commentary, no headers, no markdown. ONLY pipe lines.\n\n"
    "Example:\n"
    "user | 5k_personal_best | 27:12 | 2023-05-25\n"
    "user | 5k_personal_best | 25:50 | 2023-05-27\n"
    "user | city | Boston | 2023-01-10\n"
    "user | korean_restaurants_tried_count | 4 | 2023-06-02\n\n"
    "SESSIONS:\n{sessions}\n\nFACTS:"
)


@dataclass
class Fact:
    subject: str
    attribute: str
    value: str
    valid_from: str  # YYYY-MM-DD
    valid_to: str | None = None
    sources: list[str] = field(default_factory=list)


def parse_facts(text: str, max_facts: int = 200) -> list[Fact]:
    """Parse 'subject | attribute | value | date' lines into Facts."""
    out: list[Fact] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line.count("|") < 3:
            continue
        parts = [p.strip() for p in line.split("|")]
        subj, attr, val, date = parts[0], parts[1], parts[2], parts[3]
        date = date[:10]
        if not subj or not attr or not val:
            continue
        out.append(Fact(subject=subj.lower(), attribute=attr.lower(), value=val,
                        valid_from=date))
        if len(out) >= max_facts:
            break
    return out


def bitemporalize(facts: list[Fact]) -> list[Fact]:
    """Deterministically assign validity windows per (subject, attribute).

    Older values are closed off at the next value's valid_from; the newest value
    stays current (valid_to=None). Duplicate consecutive values are merged.
    """
    from collections import defaultdict
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for f in facts:
        groups[(f.subject, f.attribute)].append(f)

    result: list[Fact] = []
    for (subj, attr), fs in groups.items():
        fs.sort(key=lambda x: x.valid_from)
        # merge consecutive identical values (keep earliest date)
        merged: list[Fact] = []
        for f in fs:
            if merged and _norm(merged[-1].value) == _norm(f.value):
                merged[-1].sources.extend(f.sources)
                continue
            merged.append(f)
        for i, f in enumerate(merged):
            f.valid_to = merged[i + 1].valid_from if i + 1 < len(merged) else None
            result.append(f)
    return result


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def render_facts(facts: list[Fact], *, only_subjects: set[str] | None = None) -> str:
    """Render facts as a compact, reader-friendly structured timeline."""
    from collections import defaultdict
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for f in facts:
        if only_subjects and f.subject not in only_subjects:
            continue
        groups[(f.subject, f.attribute)].append(f)

    lines: list[str] = []
    for (subj, attr), fs in sorted(groups.items()):
        fs.sort(key=lambda x: x.valid_from)
        current = next((f for f in fs if f.valid_to is None), fs[-1])
        head = f"{subj} · {attr}: CURRENT = {current.value} (since {current.valid_from})"
        if len(fs) > 1:
            hist = "; ".join(
                f"{f.value} [{f.valid_from}–{f.valid_to or 'now'}]" for f in fs
            )
            head += f" | history: {hist}"
        lines.append(head)
    return "\n".join(lines)


async def build_kg(memories: list[dict], extract_fn, *, max_chars: int = 48000) -> list[Fact]:
    """Extract + bi-temporalize a knowledge graph from dated memories.

    *extract_fn* is ``async (prompt:str) -> str``. Returns validity-windowed Facts.
    """
    if len(memories) < 2:
        return []

    def _date(m: dict) -> str:
        meta = m.get("metadata") or {}
        return str(meta.get("date") or m.get("updated_at") or m.get("created_at") or "")

    blocks = []
    for m in sorted(memories, key=_date):
        blocks.append(f"[{_date(m)[:10]}] {m.get('content', '').strip()}")
    text = "\n\n".join(blocks)[:max_chars]
    try:
        raw = await extract_fn(EXTRACT_PROMPT.format(sessions=text))
    except Exception:
        return []
    return bitemporalize(parse_facts(raw))
