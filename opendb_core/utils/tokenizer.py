"""Tokenizer utilities for FTS5 indexing — handles CJK segmentation via jieba.

Custom tokenizers can be registered via ``register_tokenizer(name, fn)``
where *fn* accepts and returns a ``str``.  The active tokenizer is chosen
via ``FILEDB_TOKENIZER`` env var (default: ``"jieba"``).
"""

from __future__ import annotations

import os
import re
from typing import Callable

# CJK Unicode ranges: Chinese, Hiragana, Katakana, Hangul
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)

# Thai Unicode range
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")

# Hyphenated compound words: split "gardening-related" → "gardening-related gardening related"
_HYPHEN_RE = re.compile(r"\b(\w+(?:-\w+)+)\b")

# camelCase / PascalCase boundaries.
#   1. a lowercase letter or digit followed by an uppercase letter
#      ("createInvoice" → create|Invoice)
#   2. an uppercase run followed by an uppercase+lowercase pair, which is the
#      acronym-then-word case ("parseHTTPResponse" → parse|HTTP|Response)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# A candidate identifier: an alphanumeric run containing at least one uppercase
# letter and at least one lowercase letter. Pure-lowercase words and
# SCREAMING_CASE need no camel splitting (unicode61 already splits on "_").
_IDENT_CANDIDATE_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

# Registry of tokenizer functions: name → callable(text) → tokenized_text
_tokenizers: dict[str, Callable[[str], str]] = {}


# Japanese kana and Korean Hangul. jieba is a *Chinese* segmenter trained on a
# Chinese dictionary; handed a kana or Hangul run it returns it essentially
# whole, so a Japanese sentence was indexed as one enormous token and only an
# exact-substring query could ever match it.
_JA_KO_RUN_RE = re.compile(r"[぀-ゟ゠-ヿ가-힯]{2,}")

# Han characters, segmented by jieba.
_HAN_RE = re.compile(r"[一-鿿]")


def _character_bigrams(run: str) -> str:
    """Overlapping character bigrams, plus the unigrams.

    This is the standard language-agnostic fallback for scripts with no word
    delimiter and no available segmenter: 東京都 -> "東京 京都 東 京 都". Any
    substring query of length >= 1 can then match, at the cost of a larger
    index. It is what SQLite's own trigram tokenizer does, one size down.
    """
    if len(run) < 2:
        return run
    bigrams = [run[i : i + 2] for i in range(len(run) - 1)]
    return " ".join(bigrams) + " " + " ".join(run)


def _segment_ja_ko(text: str) -> str:
    """Replace kana/Hangul runs with their bigram expansion."""
    return _JA_KO_RUN_RE.sub(lambda m: _character_bigrams(m.group(0)), text)


def _jieba_tokenize(text: str) -> str:
    """CJK tokenizer: jieba for Chinese, character bigrams for Japanese/Korean."""
    import jieba
    segmented = " ".join(jieba.cut_for_search(text))
    return _segment_ja_ko(segmented)


def _pythainlp_tokenize(text: str) -> str:
    """Thai tokenizer using PyThaiNLP (must be installed separately)."""
    try:
        from pythainlp.tokenize import word_tokenize
        return " ".join(word_tokenize(text, engine="newmm"))
    except ImportError:
        raise RuntimeError(
            "PyThaiNLP is required for Thai tokenization. "
            "Install it with: pip install pythainlp"
        )


# Register built-in tokenizers
_tokenizers["jieba"] = _jieba_tokenize
_tokenizers["pythainlp"] = _pythainlp_tokenize


def register_tokenizer(name: str, fn: Callable[[str], str]) -> None:
    """Register a custom tokenizer function.

    Args:
        name: Tokenizer name (used in FILEDB_TOKENIZER env var).
        fn: Callable that takes raw text and returns space-separated tokens.
    """
    _tokenizers[name] = fn


def _expand_hyphens(text: str) -> str:
    """Expand hyphenated words while keeping the original form.

    "gardening-related tips" → "gardening-related gardening related tips"
    """
    def _replace(m: re.Match) -> str:
        original = m.group(0)
        parts = original.split("-")
        return original + " " + " ".join(parts)

    return _HYPHEN_RE.sub(_replace, text)


def split_identifier(token: str) -> list[str]:
    """Split a camelCase/PascalCase identifier into its parts.

    Returns ``[]`` when *token* is not a camel-cased identifier, so callers can
    cheaply skip ordinary words::

        split_identifier("CreateInvoice")      -> ["Create", "Invoice"]
        split_identifier("parseHTTPResponse")  -> ["parse", "HTTP", "Response"]
        split_identifier("invoice")            -> []
        split_identifier("DEFAULT_RETRY")      -> []   (unicode61 splits on "_")
    """
    if not (any(c.isupper() for c in token) and any(c.islower() for c in token)):
        return []
    parts = _CAMEL_BOUNDARY_RE.split(token)
    return parts if len(parts) > 1 else []


def expand_identifiers(text: str) -> str:
    """Derive the space-separated form of every camelCase identifier in *text*.

    FTS5's unicode61 tokenizer already splits on ``_``, ``/``, ``.`` and ``-``,
    so ``DEFAULT_RETRY_BUDGET`` and ``pkg/gateway/auth.go`` are searchable by
    their parts already. What it cannot split is a camel hump, so
    ``CreateInvoice`` was reachable only by reciting it verbatim — an agent
    asking "where do we create invoices" got nothing.

    The result is stored in a *separate* FTS column that is down-weighted at
    query time, so a derived match can never outrank a literal one.
    """
    out: list[str] = []
    for match in _IDENT_CANDIDATE_RE.finditer(text):
        parts = split_identifier(match.group(0))
        if parts:
            out.extend(p.lower() for p in parts)
    # Dedupe while preserving order — repeated identifiers otherwise inflate
    # term frequency in the expansion column.
    seen: set[str] = set()
    unique = [p for p in out if not (p in seen or seen.add(p))]
    return " ".join(unique)


def _get_tokenizer_name() -> str:
    return os.environ.get("FILEDB_TOKENIZER", "jieba")


# Bump whenever tokenize_for_fts / expand_identifiers change what they emit.
# Text is tokenized at *index* time and again at *query* time; if the two
# disagree, queries silently stop matching documents that were indexed under
# the old rules. There was no way to detect that and no reindex path — the only
# symptom was recall quietly degrading.
TOKENIZER_RULES_VERSION = 2


def tokenizer_fingerprint() -> str:
    """Identity of the tokenization currently in effect.

    Stored alongside the index. `opendb doctor` compares it against the stored
    value and tells the operator to reindex when they diverge.
    """
    name = _get_tokenizer_name()
    version = "unknown"
    if name == "jieba":
        try:
            import jieba
            version = getattr(jieba, "__version__", "unknown")
        except ImportError:
            version = "missing"
    elif name == "pythainlp":
        try:
            import pythainlp
            version = getattr(pythainlp, "__version__", "unknown")
        except ImportError:
            version = "missing"
    return f"{name}/{version}/rules{TOKENIZER_RULES_VERSION}"


def tokenize_for_fts(text: str) -> str:
    """Tokenize text for FTS5 indexing.

    For text containing CJK characters, applies the configured tokenizer
    (default: jieba) so that FTS5 can index individual words.
    For Thai text, uses PyThaiNLP if configured.
    For pure Latin/ASCII text, returns as-is (FTS5 unicode61 handles it fine).

    Hyphenated compound words are always expanded so both the whole form
    and individual parts are indexed.
    """
    text = _expand_hyphens(text)

    tokenizer_name = _get_tokenizer_name()

    # Check if text needs non-Latin tokenization
    has_cjk = bool(_CJK_RE.search(text))
    has_thai = bool(_THAI_RE.search(text))

    if not has_cjk and not has_thai:
        return text

    if has_thai and tokenizer_name == "pythainlp":
        fn = _tokenizers.get("pythainlp")
        if fn:
            return fn(text)

    # Default: use jieba for CJK (also handles Thai-CJK mixed text)
    fn = _tokenizers.get(tokenizer_name, _jieba_tokenize)
    return fn(text)
