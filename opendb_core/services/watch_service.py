"""Directory watch service.

Uses watchdog to monitor directories for file changes and automatically
ingests new or modified files. Each watcher runs an Observer in a background
thread; file-system events are debounced and dispatched to the async
ingestion pipeline via an asyncio queue.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from opendb_core.config import settings
from opendb_core.services.index_service import _has_parser, _is_excluded

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WatchEntry:
    """Tracks one active directory watcher."""

    id: str
    path: Path
    tags: list[str] | None
    metadata: dict | None
    observer: Observer
    created_at: float = field(default_factory=time.time)
    ingested: int = 0
    failed: int = 0
    skipped: int = 0


# In-memory registry of active watchers (lost on restart)
_watchers: dict[str, WatchEntry] = {}
_watchers_lock = Lock()

# Module-level event loop reference, set by start_watch()
_loop: asyncio.AbstractEventLoop | None = None

# Background consumer task per watcher
_consumer_tasks: dict[str, asyncio.Task] = {}

# Asyncio queues per watcher (watch_id -> queue)
_queues: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# Debounced event handler
# ---------------------------------------------------------------------------

# Minimum seconds between re-ingesting the same file path
_DEBOUNCE_SECONDS = 2.0

# A file must hold the same (size, mtime) for this long before it is read.
_QUIESCE_SECONDS = 0.4
# Give up waiting for a file that keeps changing (an actively appended log).
_QUIESCE_TIMEOUT = 30.0
# Bound on the debounce bookkeeping so a long-lived watcher cannot leak.
_MAX_TRACKED_PATHS = 10_000


class _IngestHandler(FileSystemEventHandler):
    """Watchdog handler that puts (action, path, dest) onto an asyncio queue."""

    def __init__(self, watch_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.watch_id = watch_id
        self.queue = queue
        self.loop = loop
        self._last_seen: dict[str, float] = {}
        self._lock = Lock()

    def _should_process(self, path_str: str) -> bool:
        """Rate-limit repeat events for one path.

        This is only a rate limit, not a correctness mechanism. It used to be
        the sole protection against reading a file mid-write: the *first* event
        was processed immediately and everything for the next two seconds was
        dropped, so a large file was ingested while still being written and the
        later events that would have corrected it were discarded. The consumer
        now waits for the file to quiesce before reading it.
        """
        now = time.time()
        with self._lock:
            last = self._last_seen.get(path_str, 0.0)
            if now - last < _DEBOUNCE_SECONDS:
                return False
            if len(self._last_seen) >= _MAX_TRACKED_PATHS:
                cutoff = now - _DEBOUNCE_SECONDS
                self._last_seen = {
                    k: v for k, v in self._last_seen.items() if v >= cutoff
                }
            self._last_seen[path_str] = now
            return True

    def _emit(self, action: str, path: Path, dest: Path | None = None) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (action, path, dest))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Remove a deleted file from the index.

        Nothing handled deletions before, so the index never converged with the
        filesystem: a removed file stayed searchable and readable forever.
        """
        if event.is_directory:
            return
        self._emit("delete", Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        """Treat a rename as delete-then-index, so it does not duplicate."""
        if event.is_directory:
            return
        dest = getattr(event, "dest_path", None)
        self._emit("move", Path(event.src_path), Path(dest) if dest else None)

    def _enqueue(self, event: FileSystemEvent) -> None:
        src = event.src_path
        path = Path(src)

        # Skip directories
        if path.is_dir():
            return

        # Get watch entry for exclusion check
        with _watchers_lock:
            entry = _watchers.get(self.watch_id)
        if entry is None:
            return

        # Skip excluded files
        try:
            rel = path.relative_to(entry.path)
        except ValueError:
            return
        if _is_excluded(rel, settings.index_exclude_patterns):
            return

        # Debounce
        if not self._should_process(src):
            return

        # Thread-safe put onto the asyncio queue
        self._emit("upsert", path)

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event)


# ---------------------------------------------------------------------------
# Background consumer: pulls paths from queue and ingests them
# ---------------------------------------------------------------------------

async def _wait_until_quiescent(path: Path) -> bool:
    """Block until *path* stops changing. False if it never settles or vanishes.

    Editors, downloads and build steps produce a create event long before the
    bytes are all there. Reading on the first event indexed truncated content,
    and because the debounce then swallowed the follow-up events, the truncated
    version was what the agent got — permanently.
    """
    deadline = time.monotonic() + _QUIESCE_TIMEOUT
    last: tuple[int, float] | None = None
    while time.monotonic() < deadline:
        try:
            st = path.stat()
        except OSError:
            return False
        current = (st.st_size, st.st_mtime)
        if current == last:
            return True
        last = current
        await asyncio.sleep(_QUIESCE_SECONDS)
    logger.warning("watch: %s kept changing for %.0fs; skipping", path, _QUIESCE_TIMEOUT)
    return False


async def _remove_from_index(path: Path) -> None:
    """Drop the indexed record for *path*, if there is one."""
    from opendb_core.storage import get_backend

    backend = get_backend()
    source = str(path.resolve()).replace("\\", "/")
    try:
        file_id = await backend.find_by_source_path(source)
        if file_id:
            await backend.delete_file(file_id)
            logger.info("watch: removed %s from the index", source)
    except Exception:  # noqa: BLE001 - a watcher must survive a bad event
        logger.exception("watch: failed to remove %s from the index", source)


async def _consume_queue(watch_id: str, queue: asyncio.Queue) -> None:
    """Long-running task that applies filesystem events to the index."""
    import magic as _magic
    from opendb_core.services.ingest_service import ingest_local_file

    while True:
        action, path, dest = await queue.get()
        try:
            if action == "delete":
                await _remove_from_index(path)
                continue
            if action == "move":
                # The old path is gone regardless; index the new one if it is
                # still inside this watch.
                await _remove_from_index(path)
                if dest is None:
                    continue
                path = dest

            if not path.exists() or not path.is_file():
                continue

            # Only read once the writer has finished.
            if not await _wait_until_quiescent(path):
                continue

            # Check MIME / parser support
            try:
                mime = _magic.from_file(str(path), mime=True)
            except OSError:
                logger.debug("watch %s: cannot detect MIME for %s", watch_id, path)
                continue
            if not _has_parser(mime):
                logger.debug("watch %s: unsupported MIME %s for %s", watch_id, mime, path)
                continue

            with _watchers_lock:
                entry = _watchers.get(watch_id)
            if entry is None:
                break  # watcher was removed

            result = await ingest_local_file(
                source_path=path,
                tags=entry.tags,
                metadata=entry.metadata,
            )
            status = result.get("status", "")
            with _watchers_lock:
                entry = _watchers.get(watch_id)
                if entry is not None:
                    if status == "ready":
                        entry.ingested += 1
                    elif status == "duplicate":
                        entry.skipped += 1
                    else:
                        entry.failed += 1

            logger.info(
                "watch %s: ingested %s -> %s", watch_id, path.name, status
            )

        except Exception as e:
            logger.error("watch %s: failed to ingest %s: %s", watch_id, path, e, exc_info=True)
            with _watchers_lock:
                entry = _watchers.get(watch_id)
                if entry is not None:
                    entry.failed += 1
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_watch(
    dir_path: Path,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> str:
    """Start watching *dir_path* for file changes.

    Returns a watch_id. Must be called from an async context (or pass *loop*).
    """
    global _loop

    if loop is None:
        loop = asyncio.get_running_loop()
    _loop = loop

    with _watchers_lock:
        if len(_watchers) >= settings.watch_max_watchers:
            raise ValueError(
                f"Maximum number of watchers ({settings.watch_max_watchers}) reached"
            )

        # Check if already watching this directory
        for entry in _watchers.values():
            if entry.path == dir_path:
                return entry.id

    watch_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()

    handler = _IngestHandler(watch_id, queue, loop)
    observer = Observer()
    observer.schedule(handler, str(dir_path), recursive=True)
    observer.daemon = True
    observer.start()

    entry = WatchEntry(
        id=watch_id,
        path=dir_path,
        tags=tags,
        metadata=metadata,
        observer=observer,
    )

    with _watchers_lock:
        _watchers[watch_id] = entry

    _queues[watch_id] = queue
    task = loop.create_task(_consume_queue(watch_id, queue))
    _consumer_tasks[watch_id] = task

    logger.info("Started watching %s (id=%s)", dir_path, watch_id)
    return watch_id


def stop_watch(watch_id: str) -> bool:
    """Stop a watcher by ID. Returns True if it existed."""
    with _watchers_lock:
        entry = _watchers.pop(watch_id, None)
    if entry is None:
        return False

    entry.observer.stop()
    entry.observer.join(timeout=5)

    task = _consumer_tasks.pop(watch_id, None)
    if task is not None:
        if not task.done():
            task.cancel()
        # If the consumer task was scheduled on a loop that never ran (e.g.
        # in unit tests that create a fresh loop and close it without ever
        # running it), the wrapped coroutine never reaches its first await
        # and Python emits a "coroutine '_consume_queue' was never awaited"
        # warning when the task is garbage-collected. Explicitly closing the
        # coroutine here marks it as cleanly finished and silences the warning.
        try:
            coro = task.get_coro()
            if coro is not None:
                coro.close()
        except Exception:
            pass

    _queues.pop(watch_id, None)

    logger.info("Stopped watching %s (id=%s)", entry.path, watch_id)
    return True


def stop_all() -> None:
    """Stop all active watchers. Called during shutdown."""
    with _watchers_lock:
        ids = list(_watchers.keys())
    for wid in ids:
        stop_watch(wid)


def list_watches() -> list[dict]:
    """Return info about all active watchers."""
    with _watchers_lock:
        entries = list(_watchers.values())
    return [
        {
            "id": e.id,
            "path": str(e.path),
            "tags": e.tags,
            "created_at": e.created_at,
            "ingested": e.ingested,
            "failed": e.failed,
            "skipped": e.skipped,
        }
        for e in entries
    ]


def get_watch(watch_id: str) -> dict | None:
    """Return info about a single watcher, or None."""
    with _watchers_lock:
        entry = _watchers.get(watch_id)
    if entry is None:
        return None
    return {
        "id": entry.id,
        "path": str(entry.path),
        "tags": entry.tags,
        "created_at": entry.created_at,
        "ingested": entry.ingested,
        "failed": entry.failed,
        "skipped": entry.skipped,
    }
