"""Retention and cleanup (plan section 6).

The defaults are the plan's table, restated as code:

============================================  ==============================
Raw audio, raw transcript, sensitive spans     delete after dataset
                                               construction or rejection;
                                               hard expiry 24 hours even if
                                               processing failed
Eligible local upload package                  delete after confirmed upload;
                                               expire after 7 days offline
Rejected or uncertain examples                 delete at the decision
Operational metadata                           bounded, no transcript content
============================================  ==============================

These are engineering choices, not statutory deadlines.  Cleanup runs at
startup and periodically, and it is driven by the ``expires_at`` column rather
than by whoever remembered to delete something.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from dictation.config import RetentionSettings
from dictation.logging_ import events
from dictation.store.db import Database
from dictation.store.session_store import ArtifactKind, SessionStore

HOUR = 3600.0
DAY = 24 * HOUR


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    settings: RetentionSettings = field(default_factory=RetentionSettings)

    def expiry_for(self, kind: ArtifactKind, *, now: float) -> float:
        match kind:
            case ArtifactKind.RAW_AUDIO | ArtifactKind.RAW_TRANSCRIPT | ArtifactKind.SENSITIVE_SPANS:
                return now + self.settings.raw_session_hours * HOUR
            case ArtifactKind.SPLICED_EXPORT:
                return now + self.settings.raw_session_hours * HOUR
            case ArtifactKind.PACKAGE:
                return now + self.settings.upload_package_days * DAY
        return now + self.settings.raw_session_hours * HOUR

    def job_expiry(self, *, now: float) -> float:
        return now + self.settings.upload_package_days * DAY

    def session_expiry(self, *, now: float) -> float:
        return now + self.settings.raw_session_hours * HOUR

    def metadata_expiry(self, *, now: float) -> float:
        return now + self.settings.metadata_days * DAY


@dataclass(slots=True)
class CleanupReport:
    artifacts_deleted: int = 0
    jobs_deleted: int = 0
    sessions_deleted: int = 0
    orphan_files_deleted: int = 0

    @property
    def total(self) -> int:
        return (
            self.artifacts_deleted
            + self.jobs_deleted
            + self.sessions_deleted
            + self.orphan_files_deleted
        )


def run_cleanup(
    store: SessionStore,
    database: Database,
    *,
    now: float | None = None,
) -> CleanupReport:
    """Delete everything past its expiry, then sweep orphaned files."""
    stamp = time.time() if now is None else now
    report = CleanupReport()

    for row in database.expired_artifacts(stamp):
        store.delete(row["artifact_id"])
        report.artifacts_deleted += 1

    for row in database.expired_jobs(stamp):
        database.delete_job(row["job_id"])
        report.jobs_deleted += 1

    for row in database.query("SELECT session_id FROM sessions WHERE expires_at <= ?", (stamp,)):
        store.delete_session(row["session_id"])
        report.sessions_deleted += 1

    report.orphan_files_deleted = sweep_orphans(store, database)

    if report.total:
        events.emit(
            "retention.cleanup",
            artifacts=report.artifacts_deleted,
            jobs=report.jobs_deleted,
            sessions=report.sessions_deleted,
            orphans=report.orphan_files_deleted,
        )
    return report


def sweep_orphans(store: SessionStore, database: Database) -> int:
    """Remove payload files with no database row.

    A crash between writing a payload and committing its row would otherwise
    leave an encrypted file nothing tracks - and therefore nothing expires.
    """
    known = {
        Path(row["path"]).resolve()
        for row in database.query("SELECT path FROM artifacts")
    }
    removed = 0
    for root in (store.paths.sessions, store.paths.queue):
        if not root.exists():
            continue
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            for candidate in directory.iterdir():
                if candidate.suffix not in {".enc", ".tmp"}:
                    continue
                if candidate.resolve() in known:
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                except OSError:
                    pass
            try:
                next(directory.iterdir())
            except StopIteration:
                directory.rmdir()
            except OSError:
                pass
    return removed


def delete_after_decision(
    store: SessionStore,
    session_id: str,
    *,
    keep_package: bool,
) -> int:
    """Drop raw working data once the training copy has been decided.

    ``keep_package`` retains the eligible upload package and nothing else.  A
    rejected session keeps nothing: there is no review bucket for uncertain
    examples (plan section 6).
    """
    deleted = 0
    for row in store.db.artifacts_of(session_id):
        kind = ArtifactKind(row["kind"])
        if keep_package and kind is ArtifactKind.PACKAGE:
            continue
        store.delete(row["artifact_id"])
        deleted += 1
    if not keep_package:
        store.keys.destroy_session_key(session_id)
    events.emit(
        "retention.post_decision",
        session=session_id,
        deleted=deleted,
        kept_package=keep_package,
    )
    return deleted
