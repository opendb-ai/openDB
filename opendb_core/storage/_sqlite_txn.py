"""The single write-transaction primitive for the SQLite backend.

Before this module, writes were issued ad hoc: some paths took ``_write_lock``
and ran an explicit ``BEGIN``/``COMMIT``, others called ``commit()`` on the
shared connection with no lock at all.  Because every coroutine shared one
connection, an unlocked ``commit()`` would commit *another* coroutine's
in-flight transaction, and that coroutine's later ``rollback()`` became a
no-op.  Separately, ``persist_ingestion`` only rolled back on
``IntegrityError``, so any other exception left an open transaction on the
shared connection and every subsequent write in the process failed.

``write_txn()`` is now the only sanctioned way to write.  It guarantees:

* **One writer at a time in-process** — serialized by ``_write_lock``.
* **A dedicated writer connection** — readers use ``self._db`` and therefore
  never observe another coroutine's uncommitted rows (WAL gives them a stable
  snapshot).
* **BEGIN IMMEDIATE** — the write lock is taken up front, so cross-process
  contention surfaces at BEGIN where it can be retried, rather than midway
  through a transaction where it cannot.
* **Rollback on any BaseException** — including ``CancelledError`` and
  ``KeyboardInterrupt``.  An exception can no longer leak an open transaction.
* **Bounded retry with jitter** on SQLITE_BUSY / SQLITE_LOCKED at BEGIN.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Applied to every connection. SQLite's compiled-in default is 0 (fail
# immediately); Python's sqlite3 driver happens to set 5000 today, but relying
# on a driver default for a durability-relevant setting is not acceptable.
BUSY_TIMEOUT_MS = 10_000

_MAX_BEGIN_RETRIES = 6
_BASE_BACKOFF_S = 0.01
_MAX_BACKOFF_S = 0.4

_BUSY_MARKERS = ("database is locked", "database table is locked", "busy", "locked")


def is_busy_error(exc: BaseException) -> bool:
    """Whether *exc* is a transient SQLITE_BUSY/LOCKED that is worth retrying."""
    return any(marker in str(exc).lower() for marker in _BUSY_MARKERS)


async def apply_connection_pragmas(conn) -> None:
    """Set the durability/concurrency pragmas we depend on.

    WAL lets readers proceed while a writer holds the write lock, which is what
    makes the reader/writer connection split useful. ``busy_timeout`` is set
    explicitly rather than inherited from the driver.
    """
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    await conn.execute("PRAGMA foreign_keys=ON")
    # NORMAL is the correct pairing with WAL: durable across process crash,
    # and only at risk from OS/power loss, which is the tradeoff an embedded
    # metadata store should make.
    await conn.execute("PRAGMA synchronous=NORMAL")


class SQLiteTxnMixin:
    """Supplies ``write_txn()``.  Expects ``self._wdb`` and ``self._write_lock``."""

    async def _begin_immediate(self) -> None:
        """BEGIN IMMEDIATE, retrying transient lock contention with jitter."""
        last: BaseException | None = None
        for attempt in range(_MAX_BEGIN_RETRIES):
            try:
                await self._wdb.execute("BEGIN IMMEDIATE")
                return
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not is_busy_error(exc):
                    raise
                last = exc
                backoff = min(_BASE_BACKOFF_S * (2**attempt), _MAX_BACKOFF_S)
                await asyncio.sleep(backoff * (0.5 + random.random()))
        raise TimeoutError(
            f"could not acquire the SQLite write lock after {_MAX_BEGIN_RETRIES} "
            f"attempts ({BUSY_TIMEOUT_MS}ms busy_timeout each)"
        ) from last

    async def _safe_rollback(self) -> None:
        """Roll back, tolerating a transaction SQLite already aborted itself."""
        try:
            await self._wdb.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            logger.debug("ROLLBACK on an already-closed transaction", exc_info=True)

    @asynccontextmanager
    async def write_txn(self):
        """Run a write transaction on the dedicated writer connection.

        Usage::

            async with self.write_txn() as conn:
                await conn.execute(...)
                await conn.execute(...)
            # COMMIT on clean exit, ROLLBACK on any exception

        Not re-entrant: ``_write_lock`` is not reentrant, so a nested
        ``write_txn()`` would deadlock. Helpers meant to run inside a
        transaction take the connection as an argument and are suffixed
        ``_unlocked``.
        """
        async with self._write_lock:
            await self._begin_immediate()
            try:
                yield self._wdb
            except BaseException:
                await self._safe_rollback()
                raise
            await self._wdb.execute("COMMIT")
