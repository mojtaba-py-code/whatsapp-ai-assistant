"""SQLite storage: one connection shared by every thread, guarded by a lock, in WAL mode."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from wa_assistant.text import LANGUAGES, Lang

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id  TEXT PRIMARY KEY,
    received_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_processed_received ON processed_messages (received_at);

CREATE TABLE IF NOT EXISTS conversations (
    wa_id         TEXT PRIMARY KEY,
    lang          TEXT,
    handoff_until TEXT,
    flow_state    TEXT,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    wa_id      TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_wa_id ON messages (wa_id, id);

CREATE TABLE IF NOT EXISTS bookings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    wa_id        TEXT NOT NULL,
    service_id   TEXT NOT NULL,
    service_name TEXT NOT NULL,
    date         TEXT NOT NULL,
    time         TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    lang         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'requested',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS handoffs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    wa_id          TEXT NOT NULL,
    lang           TEXT NOT NULL,
    after_hours    INTEGER NOT NULL,
    owner_notified INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);
"""

PROCESSED_RETENTION = timedelta(days=8)  # Meta retries a failed webhook for up to 7 days
MESSAGES_KEPT_PER_USER = 50


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Conversation:
    wa_id: str
    lang: Lang | None = None
    handoff_until: datetime | None = None
    flow_state: str | None = None  # JSON of the booking flow, when one is in progress


@dataclass(frozen=True, slots=True)
class BookingRecord:
    wa_id: str
    service_id: str
    service_name: str
    date: str  # ISO date, business-local
    time: str  # HH:MM, business-local
    patient_name: str
    lang: Lang


class Repository:
    """Thread-safe repository. ``path`` may be ``":memory:"`` (simulator, some tests)."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _write(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def _read(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- idempotency -----------------------------------------------------------------------
    def claim_message(self, message_id: str, now: datetime) -> bool:
        """Record ``message_id``; ``False`` when it was already processed (a duplicate/retry)."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, received_at) VALUES (?, ?)",
                (message_id, _iso(now)),
            )
            self._conn.execute(
                "DELETE FROM processed_messages WHERE received_at < ?",
                (_iso(now - PROCESSED_RETENTION),),
            )
            return cursor.rowcount == 1

    # --- conversations -----------------------------------------------------------------------
    def get_conversation(self, wa_id: str) -> Conversation:
        rows = self._read(
            "SELECT lang, handoff_until, flow_state FROM conversations WHERE wa_id = ?", (wa_id,)
        )
        if not rows:
            return Conversation(wa_id)
        row = rows[0]
        lang = row["lang"] if row["lang"] in LANGUAGES else None
        until = datetime.fromisoformat(row["handoff_until"]) if row["handoff_until"] else None
        return Conversation(wa_id, cast("Lang | None", lang), until, row["flow_state"])

    def _upsert(self, wa_id: str, column: str, value: str | None, now: datetime) -> None:
        if column not in {"lang", "handoff_until", "flow_state"}:  # pragma: no cover - internal
            msg = f"unknown column {column}"
            raise ValueError(msg)
        self._write(
            f"INSERT INTO conversations (wa_id, {column}, updated_at) VALUES (?, ?, ?) "  # noqa: S608  # nosec B608 - column is whitelisted above
            f"ON CONFLICT (wa_id) DO UPDATE SET {column} = excluded.{column}, updated_at = excluded.updated_at",
            (wa_id, value, _iso(now)),
        )

    def set_language(self, wa_id: str, lang: Lang, now: datetime) -> None:
        self._upsert(wa_id, "lang", lang, now)

    def set_flow_state(self, wa_id: str, state: str | None, now: datetime) -> None:
        self._upsert(wa_id, "flow_state", state, now)

    def set_handoff(self, wa_id: str, until: datetime | None, now: datetime) -> None:
        self._upsert(wa_id, "handoff_until", _iso(until) if until else None, now)

    # --- conversation history (for the LLM) ------------------------------------------------------
    def add_message(
        self, wa_id: str, role: Literal["user", "assistant"], body: str, now: datetime
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (wa_id, role, body, created_at) VALUES (?, ?, ?, ?)",
                (wa_id, role, body, _iso(now)),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE wa_id = ? AND id NOT IN "
                "(SELECT id FROM messages WHERE wa_id = ? ORDER BY id DESC LIMIT ?)",
                (wa_id, wa_id, MESSAGES_KEPT_PER_USER),
            )

    def recent_messages(self, wa_id: str, limit: int) -> list[tuple[str, str]]:
        """The last ``limit`` messages (oldest first) as ``(role, body)`` pairs."""
        if limit <= 0:
            return []
        rows = self._read(
            "SELECT role, body FROM messages WHERE wa_id = ? ORDER BY id DESC LIMIT ?",
            (wa_id, limit),
        )
        return [(row["role"], row["body"]) for row in reversed(rows)]

    # --- bookings and handoffs ----------------------------------------------------------------------
    def add_booking(self, booking: BookingRecord, now: datetime) -> int:
        cursor = self._write(
            "INSERT INTO bookings (wa_id, service_id, service_name, date, time, patient_name, lang,"
            " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'requested', ?)",
            (
                booking.wa_id,
                booking.service_id,
                booking.service_name,
                booking.date,
                booking.time,
                booking.patient_name,
                booking.lang,
                _iso(now),
            ),
        )
        return int(cursor.lastrowid or 0)

    def list_bookings(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM bookings ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def add_handoff(
        self, wa_id: str, lang: Lang, *, after_hours: bool, owner_notified: bool, now: datetime
    ) -> int:
        cursor = self._write(
            "INSERT INTO handoffs (wa_id, lang, after_hours, owner_notified, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (wa_id, lang, int(after_hours), int(owner_notified), _iso(now)),
        )
        return int(cursor.lastrowid or 0)

    def list_handoffs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._read("SELECT * FROM handoffs ORDER BY id DESC LIMIT ?", (limit,))
        return [
            dict(row)
            | {
                "after_hours": bool(row["after_hours"]),
                "owner_notified": bool(row["owner_notified"]),
            }
            for row in rows
        ]
