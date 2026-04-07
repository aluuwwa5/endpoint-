"""Chat history storage — PostgreSQL in production, SQLite locally."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# ── Detect backend ────────────────────────────────────────────

_use_postgres = bool(settings.database_url and settings.database_url.startswith("postgresql"))

# ── PostgreSQL (asyncpg) ──────────────────────────────────────

_pool = None


async def _pg_pool():
    global _pool
    if _pool is None:
        import asyncpg
        _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
        async with _pool.acquire() as con:
            await con.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id      BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    role    TEXT NOT NULL,
                    text    TEXT NOT NULL,
                    action  TEXT,
                    ts      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await con.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id, id)"
            )
        logger.info("chat_messages table ready (PostgreSQL)")
    return _pool


# ── SQLite (fallback) ─────────────────────────────────────────

_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"


def _sqlite_conn() -> sqlite3.Connection:
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_SQLITE_PATH))
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role    TEXT NOT NULL,
            text    TEXT NOT NULL,
            action  TEXT,
            ts      TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id, id)")
    con.commit()
    return con


# ── Public API ────────────────────────────────────────────────

async def save_message(user_id: str, role: str, text: str, action: str | None = None):
    if _use_postgres:
        pool = await _pg_pool()
        async with pool.acquire() as con:
            await con.execute(
                "INSERT INTO chat_messages (user_id, role, text, action) VALUES ($1,$2,$3,$4)",
                user_id, role, text, action,
            )
    else:
        with _sqlite_conn() as con:
            con.execute(
                "INSERT INTO chat_messages (user_id, role, text, action, ts) VALUES (?,?,?,?,?)",
                (user_id, role, text, action, datetime.utcnow().isoformat()),
            )


async def get_history(user_id: str, limit: int = 100) -> list[dict]:
    if _use_postgres:
        pool = await _pg_pool()
        async with pool.acquire() as con:
            rows = await con.fetch(
                """SELECT role, text, action, ts FROM chat_messages
                   WHERE user_id=$1 ORDER BY id DESC LIMIT $2""",
                user_id, limit,
            )
        return [
            {"role": r["role"], "text": r["text"], "action": r["action"],
             "ts": r["ts"].isoformat()}
            for r in reversed(rows)
        ]
    else:
        with _sqlite_conn() as con:
            rows = con.execute(
                """SELECT role, text, action, ts FROM chat_messages
                   WHERE user_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]


async def clear_history(user_id: str):
    if _use_postgres:
        pool = await _pg_pool()
        async with pool.acquire() as con:
            await con.execute("DELETE FROM chat_messages WHERE user_id=$1", user_id)
    else:
        with _sqlite_conn() as con:
            con.execute("DELETE FROM chat_messages WHERE user_id=?", (user_id,))
