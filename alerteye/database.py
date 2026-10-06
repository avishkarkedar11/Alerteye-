"""
AlertEye AI - Smart Event Logging
====================================

SQLite-backed session and event logging (spec section 7).

    10:42:15 -> Drowsiness -> Score 38 -> Critical
    10:47:22 -> Yawning    -> Score 62 -> Caution
    10:51:03 -> Distraction -> Score 45 -> Drowsy
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    driver_name  TEXT,
    start_time   TEXT NOT NULL,
    end_time     TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    event_type       TEXT NOT NULL,   -- Drowsiness | Yawning | Distraction | FaceAbsent | Recovery
    risk_score       REAL NOT NULL,
    duration         REAL DEFAULT 0,  -- seconds the condition lasted
    alert_level      TEXT NOT NULL,   -- NORMAL | CAUTION | DROWSY | CRITICAL
    driver_response  TEXT,            -- Acknowledged | NoResponse | Recovered | NULL
    response_time    REAL,            -- seconds until driver responded, if known
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


@dataclass
class EventRecord:
    event_type: str
    risk_score: float
    alert_level: str
    duration: float = 0.0
    driver_response: Optional[str] = None
    response_time: Optional[float] = None


class EventLogger:
    def __init__(self, db_path: str = "alerteye.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.session_id: Optional[str] = None

    # -- session lifecycle --------------------------------------------------

    def start_session(self, driver_name: str = "Unknown") -> str:
        self.session_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO sessions (session_id, driver_name, start_time) VALUES (?, ?, ?)",
            (self.session_id, driver_name, datetime.now().isoformat(timespec="seconds")),
        )
        self._conn.commit()
        return self.session_id

    def end_session(self) -> None:
        if not self.session_id:
            return
        self._conn.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (datetime.now().isoformat(timespec="seconds"), self.session_id),
        )
        self._conn.commit()

    # -- event logging --------------------------------------------------------

    def log_event(self, record: EventRecord) -> None:
        if not self.session_id:
            self.start_session()
        self._conn.execute(
            """INSERT INTO events
               (session_id, timestamp, event_type, risk_score, duration,
                alert_level, driver_response, response_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id,
                datetime.now().isoformat(timespec="seconds"),
                record.event_type,
                record.risk_score,
                record.duration,
                record.alert_level,
                record.driver_response,
                record.response_time,
            ),
        )
        self._conn.commit()

    # -- queries --------------------------------------------------------------

    def get_session_events(self, session_id: Optional[str] = None) -> list[sqlite3.Row]:
        sid = session_id or self.session_id
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp", (sid,)
        )
        return cur.fetchall()

    def get_session_info(self, session_id: Optional[str] = None) -> Optional[sqlite3.Row]:
        sid = session_id or self.session_id
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,))
        return cur.fetchone()

    def list_sessions(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM sessions ORDER BY start_time DESC")
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
