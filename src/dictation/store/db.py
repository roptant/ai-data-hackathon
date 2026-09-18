"""Local state database (plan sections 6 and 9).

SQLite holds only operational metadata - identifiers, states, timestamps,
counts, reason codes, digests - never transcript text or audio.  Payloads live
as encrypted files; this table set records where they are and what may happen
to them next.

Two properties are load-bearing:

* **Guarded transitions.**  Every state change is ``UPDATE ... WHERE state IN
  (allowed)``.  A crash or a concurrent worker cannot promote a partially built
  artifact to upload-ready, and a lost race raises
  :class:`~dictation.errors.StateTransitionConflict` instead of overwriting.
* **Payload rows outlive nothing.**  Each artifact row carries an expiry, and
  :mod:`dictation.store.retention` deletes by it.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dictation.errors import StateTransitionConflict

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   REAL NOT NULL,
    finished_at  REAL,
    language     TEXT NOT NULL DEFAULT 'en',
    contribute   INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    word_count   INTEGER NOT NULL DEFAULT 0,
    state        TEXT NOT NULL DEFAULT 'recording',
    expires_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL DEFAULT '',
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_by_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS artifacts_by_expiry ON artifacts(expires_at);

CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    state           TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    sample_id       TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL DEFAULT '',
    consent_id      TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL DEFAULT '',
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    expires_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs(state);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_by_session ON jobs(session_id);

CREATE TABLE IF NOT EXISTS consent (
    consent_id     TEXT PRIMARY KEY,
    version        TEXT NOT NULL,
    purposes       TEXT NOT NULL,
    granted_at     REAL NOT NULL,
    expires_at     REAL NOT NULL,
    revoked_at     REAL,
    paused         INTEGER NOT NULL DEFAULT 0,
    policy_version TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS receipts (
    job_id      TEXT PRIMARY KEY,
    receipt_id  TEXT NOT NULL,
    received_at REAL NOT NULL,
    accepted    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS deletions (
    request_id   TEXT PRIMARY KEY,
    scope        TEXT NOT NULL,
    created_at   REAL NOT NULL,
    completed_at REAL,
    detail       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS contributed_hashes (
    content_hash TEXT PRIMARY KEY,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS volume (
    day          TEXT PRIMARY KEY,
    packages     INTEGER NOT NULL DEFAULT 0,
    seconds      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS api_tokens (
    token_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    token_hash  TEXT NOT NULL,
    scopes      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    revoked_at  REAL,
    last_used   REAL
);
"""


@dataclass(slots=True)
class Database:
    """Thin SQLite wrapper with guarded state transitions."""

    path: Path
    connection: sqlite3.Connection = None  # type: ignore[assignment]

    @classmethod
    def open(cls, path: Path) -> Database:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        database = cls(path=path, connection=connection)
        database._migrate()
        return database

    @classmethod
    def in_memory(cls) -> Database:
        connection = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        database = cls(path=Path(":memory:"), connection=connection)
        database._migrate()
        return database

    def _migrate(self) -> None:
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def close(self) -> None:
        self.connection.close()

    # -- generic helpers -----------------------------------------------------

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, tuple(parameters))

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, tuple(parameters)).fetchall())

    def one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, tuple(parameters)).fetchone()

    def set_meta(self, key: str, value: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self.one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    # -- sessions ------------------------------------------------------------

    def insert_session(
        self,
        session_id: str,
        *,
        started_at: float,
        expires_at: float,
        language: str = "en",
        contribute: bool = False,
    ) -> None:
        self.execute(
            "INSERT INTO sessions(session_id, started_at, language, contribute, expires_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (session_id, started_at, language, int(contribute), expires_at),
        )

    def finish_session(
        self,
        session_id: str,
        *,
        finished_at: float,
        duration_ms: int,
        word_count: int,
        state: str = "finished",
    ) -> None:
        self.execute(
            "UPDATE sessions SET finished_at = ?, duration_ms = ?, word_count = ?, state = ? "
            "WHERE session_id = ?",
            (finished_at, duration_ms, word_count, state, session_id),
        )

    def session(self, session_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))

    def delete_session(self, session_id: str) -> None:
        self.execute("DELETE FROM artifacts WHERE session_id = ?", (session_id,))
        self.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))
        self.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # -- artifacts -----------------------------------------------------------

    def insert_artifact(
        self,
        artifact_id: str,
        session_id: str,
        *,
        kind: str,
        path: str,
        sha256: str,
        size_bytes: int,
        created_at: float,
        expires_at: float,
    ) -> None:
        self.execute(
            "INSERT INTO artifacts(artifact_id, session_id, kind, path, sha256, size_bytes, "
            "created_at, expires_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, session_id, kind, path, sha256, size_bytes, created_at, expires_at),
        )

    def artifacts_of(self, session_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        if kind is None:
            return self.query("SELECT * FROM artifacts WHERE session_id = ?", (session_id,))
        return self.query(
            "SELECT * FROM artifacts WHERE session_id = ? AND kind = ?", (session_id, kind)
        )

    def artifact(self, artifact_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))

    def expired_artifacts(self, now: float) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM artifacts WHERE expires_at <= ?", (now,))

    def delete_artifact(self, artifact_id: str) -> None:
        self.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))

    # -- jobs ----------------------------------------------------------------

    def insert_job(
        self,
        job_id: str,
        session_id: str,
        *,
        state: str,
        consent_id: str,
        expires_at: float,
        now: float,
        duration_ms: int = 0,
        content_hash: str = "",
    ) -> None:
        self.execute(
            "INSERT INTO jobs(job_id, session_id, state, consent_id, duration_ms, content_hash, "
            "created_at, updated_at, expires_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, session_id, state, consent_id, duration_ms, content_hash, now, now, expires_at),
        )

    def job(self, job_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))

    def job_for_session(self, session_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM jobs WHERE session_id = ?", (session_id,))

    def jobs_in_state(self, states: Iterable[str]) -> list[sqlite3.Row]:
        values = tuple(states)
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        return self.query(
            f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY created_at", values
        )

    def transition_job(
        self,
        job_id: str,
        *,
        allowed_from: Iterable[str],
        to_state: str,
        now: float | None = None,
        **fields: Any,
    ) -> sqlite3.Row:
        """Guarded state change.

        Raises :class:`StateTransitionConflict` when the row is not in one of
        ``allowed_from``, which is how a withdrawal racing an upload is
        resolved: exactly one of them wins, and the loser sees the conflict.
        """
        allowed = tuple(allowed_from)
        stamp = time.time() if now is None else now
        assignments = ["state = ?", "updated_at = ?"]
        values: list[Any] = [to_state, stamp]
        for key, value in fields.items():
            assignments.append(f"{key} = ?")
            values.append(value)
        placeholders = ",".join("?" for _ in allowed)
        values.extend([job_id, *allowed])
        cursor = self.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ? AND state IN ({placeholders})",
            values,
        )
        if cursor.rowcount != 1:
            current = self.job(job_id)
            state = current["state"] if current else "missing"
            raise StateTransitionConflict(
                f"job {job_id} is in state {state}, expected one of {allowed}"
            )
        row = self.job(job_id)
        assert row is not None
        return row

    def delete_job(self, job_id: str) -> None:
        self.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def expired_jobs(self, now: float) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM jobs WHERE expires_at <= ?", (now,))

    # -- consent -------------------------------------------------------------

    def insert_consent(
        self,
        consent_id: str,
        *,
        version: str,
        purposes: str,
        granted_at: float,
        expires_at: float,
        policy_version: str,
    ) -> None:
        self.execute(
            "INSERT INTO consent(consent_id, version, purposes, granted_at, expires_at, "
            "policy_version) VALUES(?, ?, ?, ?, ?, ?)",
            (consent_id, version, purposes, granted_at, expires_at, policy_version),
        )

    def active_consent(self, now: float) -> sqlite3.Row | None:
        return self.one(
            "SELECT * FROM consent WHERE revoked_at IS NULL AND expires_at > ? "
            "ORDER BY granted_at DESC LIMIT 1",
            (now,),
        )

    def consent(self, consent_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM consent WHERE consent_id = ?", (consent_id,))

    def revoke_consent(self, consent_id: str, *, now: float) -> None:
        self.execute("UPDATE consent SET revoked_at = ? WHERE consent_id = ?", (now, consent_id))

    def set_consent_paused(self, consent_id: str, paused: bool) -> None:
        self.execute(
            "UPDATE consent SET paused = ? WHERE consent_id = ?", (int(paused), consent_id)
        )

    def consent_history(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM consent ORDER BY granted_at DESC")

    # -- receipts, deletions, volume, hashes ---------------------------------

    def insert_receipt(self, job_id: str, receipt_id: str, *, now: float, accepted: bool) -> None:
        self.execute(
            "INSERT OR REPLACE INTO receipts(job_id, receipt_id, received_at, accepted) "
            "VALUES(?, ?, ?, ?)",
            (job_id, receipt_id, now, int(accepted)),
        )

    def receipt(self, job_id: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM receipts WHERE job_id = ?", (job_id,))

    def insert_deletion(self, request_id: str, scope: str, *, now: float, detail: str = "") -> None:
        self.execute(
            "INSERT INTO deletions(request_id, scope, created_at, detail) VALUES(?, ?, ?, ?)",
            (request_id, scope, now, detail),
        )

    def complete_deletion(self, request_id: str, *, now: float, detail: str = "") -> None:
        self.execute(
            "UPDATE deletions SET completed_at = ?, detail = ? WHERE request_id = ?",
            (now, detail, request_id),
        )

    def pending_deletions(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM deletions WHERE completed_at IS NULL")

    def record_contributed_hash(self, content_hash: str, *, now: float) -> None:
        self.execute(
            "INSERT OR IGNORE INTO contributed_hashes(content_hash, created_at) VALUES(?, ?)",
            (content_hash, now),
        )

    def contributed_hashes(self) -> set[str]:
        return {row["content_hash"] for row in self.query("SELECT content_hash FROM contributed_hashes")}

    def clear_contributed_hashes(self) -> None:
        self.execute("DELETE FROM contributed_hashes")

    def add_volume(self, day: str, *, packages: int, seconds: int) -> None:
        self.execute(
            "INSERT INTO volume(day, packages, seconds) VALUES(?, ?, ?) "
            "ON CONFLICT(day) DO UPDATE SET packages = packages + ?, seconds = seconds + ?",
            (day, packages, seconds, packages, seconds),
        )

    def volume(self, day: str) -> tuple[int, int]:
        row = self.one("SELECT packages, seconds FROM volume WHERE day = ?", (day,))
        if row is None:
            return 0, 0
        return int(row["packages"]), int(row["seconds"])

    # -- api tokens ----------------------------------------------------------

    def insert_token(
        self,
        token_id: str,
        *,
        name: str,
        token_hash: str,
        scopes: str,
        now: float,
    ) -> None:
        self.execute(
            "INSERT INTO api_tokens(token_id, name, token_hash, scopes, created_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (token_id, name, token_hash, scopes, now),
        )

    def token_by_hash(self, token_hash: str) -> sqlite3.Row | None:
        return self.one(
            "SELECT * FROM api_tokens WHERE token_hash = ? AND revoked_at IS NULL", (token_hash,)
        )

    def tokens(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM api_tokens ORDER BY created_at")

    def revoke_token(self, token_id: str, *, now: float) -> bool:
        cursor = self.execute(
            "UPDATE api_tokens SET revoked_at = ? WHERE token_id = ? AND revoked_at IS NULL",
            (now, token_id),
        )
        return cursor.rowcount == 1

    def touch_token(self, token_id: str, *, now: float) -> None:
        self.execute("UPDATE api_tokens SET last_used = ? WHERE token_id = ?", (now, token_id))
