import asyncio
import logging

import asyncpg

from opendb_core.config import settings

logger = logging.getLogger(__name__)

pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global pool
    last_err = None
    for attempt in range(3):
        try:
            pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                command_timeout=settings.db_command_timeout,
                # Bound how long a caller waits for a free connection. Without
                # it, pool exhaustion presents as an unbounded hang rather than
                # an error the caller can act on.
                timeout=settings.db_acquire_timeout,
                # Server-side backstop. command_timeout only cancels client
                # side; statement_timeout makes PostgreSQL itself abort a
                # runaway query so it stops burning CPU and holding locks.
                server_settings={
                    "statement_timeout": str(int(settings.db_statement_timeout_ms)),
                    "idle_in_transaction_session_timeout": "60000",
                    "application_name": "opendb",
                },
            )
            return
        except (OSError, asyncpg.PostgresError) as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning(
                "DB connection attempt %d failed: %s. Retrying in %ds...",
                attempt + 1, e, wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed to connect to database after 3 attempts: {last_err}")


async def close_pool() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


async def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return pool
