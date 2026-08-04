"""Tree-sitter symbol extraction — real parsing instead of regex heuristics.

The previous extractor used Python's `ast` for `.py` (accurate) and a handful of
line-anchored regexes for the JS/TS family. Everything else — Go, Rust, Java,
C#, Ruby, PHP, C/C++ — produced *nothing at all*, so a coding-agent memory
product had no symbol index for most of the languages coding agents work in.
Even within JS the regexes only saw top-level `class`/`function`/`const fn =`
declarations: methods, object literals, decorated declarations and anything
indented under a namespace were invisible.

Tree-sitter parses, so it sees the same structure the compiler does, and it
reports real byte and line spans. Those spans are what makes staleness
detection possible: a memory anchored to a symbol can be revalidated by
comparing the symbol's hash across commits, which needs an exact span rather
than a regex's guess at where a definition ends.

This is an optional tier (`pip install open-db[code]`). Without it the previous
extractor still runs, so the default install is unchanged.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Extension -> tree-sitter language name.
LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}

# Node types that introduce a named symbol, and what kind to record it as.
# Keyed by language where the grammars disagree; `_DEFAULT` covers the rest.
_KINDS: dict[str, dict[str, str]] = {
    "_DEFAULT": {
        "function_declaration": "function",
        "function_definition": "function",
        "method_definition": "method",
        "method_declaration": "method",
        "class_declaration": "class",
        "class_definition": "class",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "struct_item": "struct",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "enum_item": "enum",
        "type_alias_declaration": "type",
        "impl_item": "impl",
        "trait_item": "trait",
        "function_item": "function",
        "module": "module",
        "program": None,
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        # Go wraps the name one level down in a type_spec, so match that
        # directly rather than the type_declaration that contains it.
        "type_spec": "type",
    },
    # Ruby's grammar names the nodes after the keyword.
    "ruby": {
        "class": "class",
        "module": "module",
        "method": "method",
        "singleton_method": "method",
    },
}

# An arrow function or function expression bound to a name. Extremely common in
# TS/JS, and invisible to a grammar walk that only looks at declarations.
_VALUE_BINDING_TYPES = frozenset({"variable_declarator", "public_field_definition"})
_FUNCTION_VALUE_TYPES = frozenset({
    "arrow_function", "function_expression", "function",
})

# Node types that introduce a naming scope, for qualified names.
_SCOPE_TYPES = frozenset({
    "class_declaration", "class_definition", "class_specifier",
    "impl_item", "trait_item", "interface_declaration", "struct_item",
    "namespace_definition", "module",
})

_MAX_SOURCE_BYTES = 4 * 1024 * 1024


def is_available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401
        return True
    except ImportError:
        return False


def language_for(filename: str, source_path: str | None = None) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(Path(source_path or filename).suffix.lower())


def _node_name(node, source: bytes) -> str | None:
    """The declared identifier of *node*, if it has one."""
    for field in ("name", "declarator"):
        child = node.child_by_field_name(field)
        while child is not None and child.type not in (
            "identifier", "type_identifier", "field_identifier",
            "property_identifier", "constant", "word",
        ):
            # C-style declarators nest: `int *foo(void)` -> pointer_declarator
            # -> function_declarator -> identifier.
            nxt = child.child_by_field_name("declarator") or child.child_by_field_name("name")
            if nxt is None:
                break
            child = nxt
        if child is not None and child.type.endswith(("identifier", "constant", "word")):
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "constant"):
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return None


def _signature(node, source: bytes) -> str:
    """The declaration line(s) up to the body, collapsed to one line."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else min(node.end_byte, node.start_byte + 400)
    text = source[node.start_byte:end].decode("utf-8", "replace")
    return " ".join(text.split())[:1000]


def extract_symbols(
    content: str,
    *,
    filename: str,
    source_path: str | None = None,
) -> list[dict] | None:
    """Symbols in *content*, or None when tree-sitter cannot handle this file.

    Returning None (rather than an empty list) lets the caller distinguish
    "no parser for this language" from "parsed, found nothing" and fall back.
    """
    lang_name = language_for(filename, source_path)
    if lang_name is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None

    source = content.encode("utf-8", "replace")
    if len(source) > _MAX_SOURCE_BYTES:
        logger.debug("tree-sitter: %s is too large to parse", filename)
        return None

    try:
        parser = get_parser(lang_name)
        tree = parser.parse(source)
    except Exception:  # noqa: BLE001 - a missing grammar must not break ingest
        logger.debug("tree-sitter: no usable grammar for %s", lang_name, exc_info=True)
        return None

    kinds = {**_KINDS["_DEFAULT"], **_KINDS.get(lang_name, {})}
    symbols: list[dict] = []

    def walk(node, scope: list[str]) -> None:
        kind = kinds.get(node.type)
        if kind is None and node.type in _VALUE_BINDING_TYPES:
            value = node.child_by_field_name("value")
            if value is not None and value.type in _FUNCTION_VALUE_TYPES:
                kind = "function"
        name = _node_name(node, source) if kind else None
        child_scope = scope

        if kind and name:
            qualified = ".".join([*scope, name])
            span = source[node.start_byte:node.end_byte]
            symbols.append({
                "name": name,
                "kind": kind,
                "qualified_name": qualified,
                # tree-sitter rows are 0-based; the rest of the codebase is 1-based.
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": _signature(node, source),
                "docstring": "",
                # Hashes for staleness detection: sig_hash changes when the
                # declaration changes, span_hash when anything in the body does.
                "sig_hash": hashlib.sha256(
                    _signature(node, source).encode("utf-8")
                ).hexdigest()[:16],
                "span_hash": hashlib.sha256(span).hexdigest()[:16],
            })
            if node.type in _SCOPE_TYPES:
                child_scope = [*scope, name]

        for child in node.children:
            walk(child, child_scope)

    walk(tree.root_node, [])
    return symbols
