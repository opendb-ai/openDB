from __future__ import annotations

from pathlib import Path
from typing import Any

from opendb_core.parsers.base import FileParser, ParseResult

_registry: dict[str, FileParser] = {}


def register(mime_type: str, parser: Any) -> None:
    _registry[mime_type] = parser


def get_parser(mime_type: str) -> FileParser:
    # Exact match
    if mime_type in _registry:
        return _registry[mime_type]
    # Prefix match (e.g. "text/*" catches "text/plain")
    prefix = mime_type.split("/")[0] + "/*"
    if prefix in _registry:
        return _registry[prefix]
    raise ValueError(f"No parser registered for MIME type: {mime_type}")


def parse_file(file_path: Path, mime_type: str) -> ParseResult:
    """Parse *file_path*, enforcing resource bounds first.

    Every parser here receives attacker-influenced bytes, and the size limit on
    the compressed input says nothing about what the parser will allocate. The
    checks live at this single dispatch point so no parser can be reached
    without them.
    """
    from opendb_core.parsers.limits import (
        check_page_count,
        check_zip_container,
        configure_pillow,
    )

    configure_pillow()
    check_zip_container(file_path)

    parser = get_parser(mime_type)
    result = parser.parse(file_path)
    check_page_count(len(result.pages), file_path)
    return result
