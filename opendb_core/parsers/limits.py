"""Resource bounds applied before untrusted documents reach a parser.

Every parser in this package is handed attacker-influenced bytes: an agent
indexes whatever is in the workspace, and the HTTP upload endpoint accepts
files directly. The size limit on the *compressed* input says nothing about
what a parser will allocate — a 40 KB zip can expand to gigabytes, and a 12 KB
PNG can declare a 60000x60000 canvas that Pillow will happily try to
materialise. These checks run first, so a malicious file is rejected instead of
taking the process down.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Pillow refuses images above this many pixels. 89 megapixels is well past any
# real scanned page and far below what exhausts memory (~256 MB decoded RGB).
MAX_IMAGE_PIXELS = 89_478_485

# Largest total uncompressed payload allowed from a zip container
# (DOCX/PPTX/XLSX are all zips).
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
# ...and the largest expansion ratio, which catches the classic bomb shape
# where a tiny archive declares an enormous payload.
MAX_COMPRESSION_RATIO = 200

# Cap on pages/sheets/slides extracted from a single document.
MAX_PAGES = 5_000


class DocumentTooComplexError(ValueError):
    """Raised when a document exceeds a resource bound before parsing."""


def configure_pillow() -> None:
    """Bound Pillow's decompression-bomb threshold.

    Pillow's own default raises a *warning* and keeps going. We want a hard
    error: the image never gets decoded.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def check_zip_container(file_path: Path) -> None:
    """Reject zip-based documents that would expand unreasonably.

    Raises DocumentTooComplexError. A file that is not a zip is left alone —
    the parser will produce its own error for that.
    """
    try:
        with zipfile.ZipFile(file_path) as zf:
            infos = zf.infolist()
            uncompressed = sum(i.file_size for i in infos)
            compressed = sum(i.compress_size for i in infos) or 1
    except (zipfile.BadZipFile, OSError):
        return

    if uncompressed > MAX_UNCOMPRESSED_BYTES:
        raise DocumentTooComplexError(
            f"{file_path.name}: archive expands to {uncompressed} bytes, "
            f"over the {MAX_UNCOMPRESSED_BYTES} byte limit"
        )
    ratio = uncompressed / compressed
    if ratio > MAX_COMPRESSION_RATIO and uncompressed > 32 * 1024 * 1024:
        raise DocumentTooComplexError(
            f"{file_path.name}: archive expands {ratio:.0f}x, over the "
            f"{MAX_COMPRESSION_RATIO}x limit — refusing to parse"
        )


def check_page_count(count: int, file_path: Path | str = "") -> None:
    """Reject documents with an implausible number of pages/sheets/slides."""
    if count > MAX_PAGES:
        raise DocumentTooComplexError(
            f"{file_path}: {count} pages exceeds the {MAX_PAGES} page limit"
        )


def truncate_pages(count: int) -> int:
    """Page count to actually parse, logging when the cap bites."""
    if count > MAX_PAGES:
        logger.warning("Document has %d pages; parsing the first %d", count, MAX_PAGES)
        return MAX_PAGES
    return count
