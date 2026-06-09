"""Deterministic, local entity extraction for the third retrieval signal.

The 2026 SOTA memory algorithms (Mem0 et al.) fuse three signals — lexical
(BM25/FTS), dense (vectors), and **entity** matching — and report that the
entity leg drives the largest gains on *temporal* and *multi-hop* questions,
because it links facts about the same real-world thing across many memories.

To keep OpenDB's zero-API / local / deterministic contract, entities are
extracted with rules, not an LLM or cloud NER: proper-noun phrases, code
identifiers (CamelCase, snake_case, dotted paths), versions, URLs/hosts,
numbers-with-units, dates, and a curated tech vocabulary. The goal is not
perfect NER — it is *consistent* extraction so the same entity normalizes
identically everywhere (which is what linking and boosting need).
"""

from __future__ import annotations

import re

# --- patterns -------------------------------------------------------------
_CODE_IDENT = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*\b")  # dotted.path
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")  # CamelCase
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")  # snake_case
_CONST = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*\b")  # CONSTANT / ACRONYM
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")  # 1.2, v3.4.5
_HOSTPORT = re.compile(r"\b(?:[\w.-]+)?:\d{2,5}\b")  # :9090, host:5432
_URLISH = re.compile(r"\b(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[\w./-]*)?\b", re.I)
_PATH = re.compile(r"\b(?:[\w-]+/){1,}[\w.-]+\b")  # pkg/gateway/auth.go
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|k|m|bn))?\b", re.I)
_NUM_UNIT = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:ms|s|gb|mb|kb|tb|%|percent|rps|qps|req(?:uests)?/(?:s|min))\b", re.I)
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?"
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")  # Proper Noun phrase

# Curated lowercase tech/business vocabulary worth linking even when not Capitalized.
_VOCAB = frozenset(
    "react svelte vue angular postgres postgresql mysql sqlite redis mongodb "
    "kafka rabbitmq nats grpc graphql rest websocket oauth jwt saml "
    "docker kubernetes k8s terraform ansible argocd jenkins github gitlab "
    "aws gcp azure lambda s3 dynamodb cloudflare vercel netlify "
    "python javascript typescript golang rust java kotlin swift ruby php "
    "pytorch tensorflow onnx cuda numpy pandas fastapi django flask celery "
    "prometheus grafana opensearch elasticsearch datadog sentry pagerduty "
    "stripe twilio sendgrid firebase auth0 okta vault cosign "
    "svelte nextjs nuxt remix vite webpack".split()
)

# Words to never treat as a proper-noun entity on their own.
_PROPER_STOP = frozenset(
    "I The A An This That These Those It We You They He She My Our Your Their "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
    "When What Where Why How Who Which If And But Or So Then Now".split()
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_entities(text: str, *, max_entities: int = 40) -> list[str]:
    """Return a de-duplicated, normalized list of entity strings from *text*."""
    if not text:
        return []
    found: dict[str, None] = {}  # ordered set

    def add(s: str) -> None:
        n = _norm(s)
        if len(n) >= 2 and n not in found:
            found[n] = None

    for pat in (_CODE_IDENT, _CAMEL, _SNAKE, _CONST, _HOSTPORT, _PATH,
                _URLISH, _MONEY, _NUM_UNIT, _DATE, _VERSION):
        for m in pat.findall(text):
            add(m if isinstance(m, str) else m[0])

    # curated vocabulary (case-insensitive, whole word)
    low = text.lower()
    for term in _VOCAB:
        if re.search(rf"\b{re.escape(term)}\b", low):
            add(term)

    # proper-noun phrases, minus stopword-only matches
    for m in _PROPER.findall(text):
        head = m.split()[0]
        if head in _PROPER_STOP and len(m.split()) == 1:
            continue
        add(m)

    return list(found.keys())[:max_entities]


def query_entities(query: str) -> list[str]:
    """Entities to match a recall query against the stored entity index."""
    return extract_entities(query, max_entities=16)
