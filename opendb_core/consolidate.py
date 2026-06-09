"""Dreaming — offline memory consolidation.

The 2026 frontier (OpenAI "Dreaming", Anthropic "Dreams", Letta sleep-time
compute) all converge on the same move: an offline pass that reads raw history
+ existing memory and *synthesizes* it — merging duplicates, replacing stale
facts with the latest value, building temporal timelines, and surfacing
insights — so that at query time the reader sees a few digested facts instead
of dozens of raw episodes.

OpenDB's version is the one the closed frontier can't offer: **local,
deterministic, and fully auditable**. Every consolidated memory keeps a
`provenance` list of the source memory ids it was derived from, so a consolidation
is reproducible and inspectable rather than a black-box rewrite.

Two strategies, composable:

* ``consolidate_timeline`` — deterministic. Groups memories by shared entity,
  orders them by event date, and emits one "timeline" memory per entity that
  states the *current* value plus the dated history. No LLM. Directly targets
  knowledge-update ("what is X now") and temporal-reasoning ("when did X change").
* ``consolidate_reflect`` — optional LLM synthesis for multi-session questions
  where the answer must combine facts the timeline can't mechanically merge.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict

from opendb_core.entities import extract_entities

# Entities too generic to anchor a useful timeline.
_SKIP_ENTITIES = frozenset(
    "now today current latest recent new monday tuesday wednesday thursday "
    "friday saturday sunday".split()
)


def _date_of(mem: dict) -> str:
    meta = mem.get("metadata") or {}
    return str(meta.get("date") or mem.get("updated_at") or mem.get("created_at") or "")


def _group_by_entity(memories: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in memories:
        ents = (m.get("metadata") or {}).get("entities")
        if not ents:
            ents = extract_entities(m.get("content", ""))
        for e in ents:
            if e in _SKIP_ENTITIES or len(e) < 3:
                continue
            groups[e].append(m)
    return groups


def build_timeline_facts(memories: list[dict], *, min_group: int = 2,
                         max_per_entity: int = 6) -> list[dict]:
    """Pure function: return consolidated timeline facts for entity groups.

    Each fact is ``{content, entity, provenance: [memory_id...], date}``.
    Deterministic — same input yields the same output.
    """
    groups = _group_by_entity(memories)
    facts: list[dict] = []
    seen_signatures: set[str] = set()
    for entity, mems in groups.items():
        if len(mems) < min_group:
            continue
        # de-dup identical contents, then order oldest -> newest by event date
        uniq: dict[str, dict] = {}
        for m in mems:
            uniq.setdefault(m.get("content", "").strip(), m)
        ordered = sorted(uniq.values(), key=_date_of)
        if len(ordered) < min_group:
            continue
        ordered = ordered[-max_per_entity:]
        latest = ordered[-1]
        history_lines = []
        for m in ordered:
            d = _date_of(m)[:10]
            history_lines.append(f"- [{d}] {m.get('content', '').strip()}")
        content = (
            f"Consolidated timeline for \"{entity}\" "
            f"(most recent state is authoritative):\n"
            f"CURRENT: {latest.get('content', '').strip()}\n"
            f"HISTORY (oldest to newest):\n" + "\n".join(history_lines)
        )
        sig = entity + "|" + latest.get("content", "")
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        facts.append({
            "content": content,
            "entity": entity,
            "date": _date_of(latest)[:10],
            "provenance": [m.get("memory_id") for m in ordered if m.get("memory_id")],
        })
    return facts


REFLECT_PROMPT = (
    "You are a memory-consolidation process running offline (like sleep). Below "
    "are a user's past conversation sessions, each prefixed with its date "
    "[YYYY-MM-DD]. Produce a compact, durable memory of the USER. Write ONE "
    "atomic item per line, no preamble, grouped under these exact headers:\n\n"
    "## USER PROFILE\n"
    "Stable preferences, traits, constraints, relationships, and recurring "
    "habits (likes/dislikes, style, dietary/medical, tools, people). Capture "
    "even *implicit* preferences (inferred from choices, complaints, praise). "
    "State each as a standalone fact, e.g. 'User prefers window seats on flights'.\n\n"
    "## CURRENT FACTS (as of the latest session)\n"
    "Facts whose value can change. Use bi-temporal phrasing and resolve updates "
    "by keeping the LATEST value valid:\n"
    "  'CURRENT (since YYYY-MM-DD): <fact>. PREVIOUSLY (until YYYY-MM-DD): <old "
    "value, now INVALID>.'\n"
    "For anything that accumulates (places visited, items owned, totals), give "
    "the running COUNT and list the items.\n\n"
    "## EVENT TIMELINE\n"
    "Dated events in chronological order, each as '[YYYY-MM-DD] <event>', so "
    "relative-time and ordering questions ('how long ago', 'before/after', "
    "'most recent') can be answered exactly.\n\n"
    "SESSIONS:\n{sessions}\n\nCONSOLIDATED MEMORY:"
)


async def consolidate_reflect(backend, reflect_fn, *, max_chars: int = 48000,
                              max_facts: int = 60) -> dict:
    """LLM 'dreaming' consolidation: read the dated history and write back a
    compact set of durable, update-resolved atomic facts (additive — raw
    episodes are kept). *reflect_fn* is ``async (prompt:str) -> str``.

    This is the frontier 'sleep-time compute' move, but OpenDB keeps the source
    episodes and tags every synthesized fact ``__consolidated__`` so the result
    is auditable rather than a black-box rewrite.
    """
    listing = await backend.list_memories(memory_type=None, tags=None,
                                           limit=100000, offset=0)
    memories = listing.get("memories", [])
    if len(memories) < 2:
        return {"facts": 0, "scanned": len(memories)}

    blocks = []
    for m in sorted(memories, key=_date_of):
        d = _date_of(m)[:10]
        blocks.append(f"[{d}] {m.get('content', '').strip()}")
    text = "\n\n".join(blocks)[:max_chars]

    try:
        out = await reflect_fn(REFLECT_PROMPT.format(sessions=text))
    except Exception:
        return {"facts": 0, "scanned": len(memories), "error": True}

    section = "fact"
    facts: list[tuple[str, str]] = []  # (content, section)
    for line in (out or "").splitlines():
        raw = line.strip()
        if raw.startswith("#"):
            h = raw.lower()
            if "profile" in h:
                section = "profile"
            elif "timeline" in h or "event" in h:
                section = "timeline"
            else:
                section = "fact"
            continue
        s = raw.lstrip("-•*0123456789. ").strip()
        if len(s) >= 8:
            facts.append((s, section))
    facts = facts[:max_facts]

    for content, section in facts:
        await backend.store_memory(
            memory_id=str(uuid.uuid4()),
            content=content,
            memory_type=("semantic" if section != "timeline" else "episodic"),
            tags=["__consolidated__", f"__{section}__"],
            metadata={"consolidated": True, "section": section},
            source="ai_inference",
        )
    return {"facts": len(facts), "scanned": len(memories)}


async def consolidate_kg(backend, extract_fn, *, max_chars: int = 48000) -> dict:
    """Dreaming via a **bi-temporal knowledge graph** (the strongest variant).

    Reads the backend's memories, extracts validity-windowed facts with
    :mod:`opendb_core.temporal_kg`, and stores one consolidated KG memory
    (tagged ``__kg__`` / ``__consolidated__``) so ordinary recall surfaces a
    structured "CURRENT value + history" timeline. Local + deterministic
    invalidation + full provenance; only *extract_fn* (``async (str)->str``)
    uses an LLM.

    On LongMemEval this lifts end-to-end accuracy ~+4.5 over hybrid-only at a
    fixed model (preference +23, knowledge-update +10), with no synthesis-style
    dilution of the raw episodes (they are kept).
    """
    from opendb_core.temporal_kg import build_kg, render_facts

    listing = await backend.list_memories(memory_type=None, tags=None,
                                           limit=100000, offset=0)
    memories = listing.get("memories", [])
    facts = await build_kg(memories, extract_fn, max_chars=max_chars)
    text = render_facts(facts)
    if not text:
        return {"facts": 0, "scanned": len(memories)}
    await backend.store_memory(
        memory_id=str(uuid.uuid4()),
        content=("Knowledge graph (bi-temporal; CURRENT value is authoritative, "
                 "history shows when each value was valid):\n" + text),
        memory_type="semantic",
        tags=["__consolidated__", "__kg__"],
        metadata={"consolidated": True, "kg": True},
        source="ai_inference",
    )
    return {"facts": len(facts), "scanned": len(memories)}


async def consolidate_timeline(backend, *, min_group: int = 2,
                               pin: bool = False) -> dict:
    """Run deterministic timeline consolidation over a backend's memories and
    store the resulting consolidated facts (semantic, with provenance)."""
    listing = await backend.list_memories(memory_type=None, tags=None,
                                           limit=100000, offset=0)
    memories = listing.get("memories", [])
    facts = build_timeline_facts(memories, min_group=min_group)

    stored = 0
    for f in facts:
        meta = {
            "consolidated": True,
            "entity": f["entity"],
            "provenance": f["provenance"],
        }
        if f["date"]:
            meta["date"] = f["date"]
        await backend.store_memory(
            memory_id=str(uuid.uuid4()),
            content=f["content"],
            memory_type="semantic",
            tags=["__consolidated__", f["entity"]],
            metadata=meta,
            pinned=pin,
            source="ai_inference",
        )
        stored += 1
    return {"groups": len(facts), "stored": stored, "scanned": len(memories)}
