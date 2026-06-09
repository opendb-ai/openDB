"""Local, zero-API text embeddings for optional hybrid recall.

Hybrid recall pairs SQLite FTS5 (exact-match / lexical) with a dense vector
index so the memory layer also catches *semantic / paraphrased* queries that
share no keywords with the stored answer. To preserve OpenDB's zero-API,
local-first contract, embeddings are computed **locally** with a static model
(``model2vec``) — a distilled lookup table that needs no GPU, no torch/onnx
inference, and no network at query time. The default retrieval-tuned model is a
small on-disk download (tens of MB) and encodes in microseconds per token.

This module is import-safe even when the optional deps are absent: callers
should gate on :func:`hybrid_available` and fall back to pure FTS.

Install with::

    pip install "open-db[hybrid]"
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None
_lock = threading.Lock()


def hybrid_available() -> bool:
    """Return True if the optional hybrid deps (model2vec, sqlite-vec) import."""
    try:
        import model2vec  # noqa: F401
        import sqlite_vec  # noqa: F401
        return True
    except Exception:
        return False


def _require_deps() -> None:
    if not hybrid_available():
        raise RuntimeError(
            "Hybrid recall requires the optional dependencies. "
            'Install them with: pip install "open-db[hybrid]"'
        )


def get_embedder(model_name: str | None = None):
    """Lazily load and cache the static embedding model (thread-safe)."""
    global _model, _model_name
    from opendb_core.config import settings

    name = model_name or settings.memory_embed_model
    if _model is not None and _model_name == name:
        return _model
    with _lock:
        if _model is None or _model_name != name:
            _require_deps()
            from model2vec import StaticModel

            logger.info("Loading static embedding model: %s", name)
            _model = StaticModel.from_pretrained(name)
            _model_name = name
    return _model


def embed_dim(model_name: str | None = None) -> int:
    """Embedding dimensionality for the configured model."""
    return int(get_embedder(model_name).dim)


def embed_one(text: str, model_name: str | None = None) -> list[float]:
    """Embed a single string into a plain float list."""
    vec = get_embedder(model_name).encode([text or ""])[0]
    return [float(x) for x in vec.tolist()]


def embed_many(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed a batch of strings."""
    if not texts:
        return []
    arr = get_embedder(model_name).encode(list(texts))
    return [[float(x) for x in row] for row in arr.tolist()]
