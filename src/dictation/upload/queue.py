"""Upload queue with consent rechecks at every boundary (plan section 9).

Consent is rechecked at enqueue, before the package is marked eligible, and
before transfer.  The server rechecks again at admission.  Each recheck reads
current state, so a withdrawal always wins over a decision made earlier.

Races are handled explicitly rather than hoped away:

* withdrawal cancels queued and in-flight jobs and deletes their payloads,
* a receipt that arrives after withdrawal is recorded as *not accepted* and
  triggers a server-side deletion request - it never makes the sample eligible
  again,
* every state change is a guarded SQL update, so two workers cannot both claim
  the same job.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from dictation.config import UploadSettings
from dictation.consent.consent import ConsentManager
from dictation.dataset.package import UploadPackage
from dictation.errors import StateTransitionConflict
from dictation.logging_ import events
from dictation.store.db import Database
from dictation.store.retention import RetentionPolicy, delete_after_decision
from dictation.store.session_store import ArtifactKind, SessionStore
from dictation.types import new_id
from dictation.upload.states import CANCELLABLE_STATES, JobState, predecessors


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    session_id: str
    state: JobState
    reason: str = ""
    sample_id: str = ""
    idempotency_key: str = ""
    consent_id: str = ""
    content_hash: str = ""
    duration_ms: int = 0
    attempts: int = 0
    next_attempt_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.state in {JobState.ACKNOWLEDGED, JobState.REJECTED, JobState.DELETED}


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    job: Job | None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.job is not None


class UploadQueue:
    """Owns job rows, the stored packages and the consent rechecks."""

    def __init__(
        self,
        database: Database,
        store: SessionStore,
        consent: ConsentManager,
        *,
        settings: UploadSettings | None = None,
        retention: RetentionPolicy | None = None,
    ) -> None:
        self.db = database
        self.store = store
        self.consent = consent
        self.settings = settings or UploadSettings()
        self.retention = retention or RetentionPolicy()

    # -- reading -------------------------------------------------------------

    def job(self, job_id: str) -> Job | None:
        row = self.db.job(job_id)
        return _job(row) if row else None

    def job_for_session(self, session_id: str) -> Job | None:
        row = self.db.job_for_session(session_id)
        return _job(row) if row else None

    def jobs(self, *states: JobState) -> tuple[Job, ...]:
        wanted = states or tuple(JobState)
        return tuple(_job(row) for row in self.db.jobs_in_state(str(state) for state in wanted))

    def ready_for_upload(self, *, now: float | None = None) -> tuple[Job, ...]:
        stamp = time.time() if now is None else now
        return tuple(
            job
            for job in self.jobs(JobState.ELIGIBLE)
            if job.next_attempt_at <= stamp
        )

    # -- enqueue and progress ------------------------------------------------

    def enqueue(
        self,
        session_id: str,
        *,
        session_started_at: float,
        session_opted_out: bool = False,
        language: str = "en",
        now: float | None = None,
    ) -> EnqueueResult:
        """Create a job if, and only if, this session may be contributed."""
        stamp = time.time() if now is None else now
        decision = self.consent.check_session(
            session_started_at=session_started_at,
            session_opted_out=session_opted_out,
            language=language,
            now=stamp,
        )
        if not decision:
            events.emit("queue.enqueue_refused", session=session_id, reason=decision.reason)
            return EnqueueResult(None, decision.reason)

        cap = self._volume_gate(stamp)
        if cap:
            events.emit("queue.enqueue_refused", session=session_id, reason=cap)
            return EnqueueResult(None, cap)

        job_id = new_id("job")
        self.db.insert_job(
            job_id,
            session_id,
            state=str(JobState.LOCAL_PENDING),
            consent_id=decision.consent_id,
            expires_at=self.retention.job_expiry(now=stamp),
            now=stamp,
        )
        events.emit("queue.enqueued", session=session_id, job=job_id, consent=decision.consent_id)
        job = self.job(job_id)
        assert job is not None
        return EnqueueResult(job)

    def start_analysis(self, job_id: str, *, now: float | None = None) -> Job:
        return self._transition(job_id, JobState.ANALYZING, now=now)

    def start_build(self, job_id: str, *, now: float | None = None) -> Job:
        return self._transition(job_id, JobState.BUILDING, now=now)

    def mark_eligible(
        self,
        job_id: str,
        package: UploadPackage,
        *,
        content_hash: str,
        now: float | None = None,
    ) -> Job:
        """Store the encrypted package and mark the job ready to upload.

        The consent recheck happens *before* the package is written, so a
        withdrawal between building and eligibility leaves nothing on disk.
        """
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        recheck = self.consent.recheck(current.consent_id, now=stamp)
        if not recheck:
            return self.reject(job_id, recheck.reason, now=stamp)

        artifact = self.store.put(
            current.session_id,
            ArtifactKind.PACKAGE,
            package.archive_bytes,
            expires_at=self.retention.expiry_for(ArtifactKind.PACKAGE, now=stamp),
            now=stamp,
        )
        job = self._transition(
            job_id,
            JobState.ELIGIBLE,
            now=stamp,
            sample_id=package.sample_id,
            idempotency_key=_idempotency_key(package),
            content_hash=content_hash,
            duration_ms=package.duration_ms,
            reason="",
        )
        # Raw working data goes now that the package exists (plan section 6).
        delete_after_decision(self.store, current.session_id, keep_package=True)
        events.emit(
            "queue.eligible",
            job=job_id,
            sample=package.sample_id,
            bytes=artifact.size_bytes,
            duration_ms=package.duration_ms,
        )
        return job

    def reject(self, job_id: str, reason: str, *, now: float | None = None) -> Job:
        """Reject a job and delete everything it produced.

        There is no review bucket: an uncertain or rejected example is deleted
        at the decision (plan section 6).
        """
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        job = self._transition(job_id, JobState.REJECTED, now=stamp, reason=reason)
        delete_after_decision(self.store, current.session_id, keep_package=False)
        events.emit("queue.rejected", job=job_id, session=current.session_id, reason=reason)
        return job

    # -- transfer ------------------------------------------------------------

    def claim_for_upload(self, job_id: str, *, now: float | None = None) -> Job:
        """Move ELIGIBLE -> UPLOADING, rechecking consent immediately before."""
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        recheck = self.consent.recheck(current.consent_id, now=stamp)
        if not recheck:
            return self.cancel(job_id, recheck.reason, now=stamp)
        cap = self._volume_gate(stamp)
        if cap:
            return self._transition(
                job_id,
                JobState.ELIGIBLE,
                now=stamp,
                reason=cap,
                next_attempt_at=stamp + 3600,
            )
        return self._transition(
            job_id,
            JobState.UPLOADING,
            now=stamp,
            attempts=current.attempts + 1,
        )

    def acknowledge(self, job_id: str, receipt_id: str, *, now: float | None = None) -> Job:
        """Record a server receipt and delete the local package."""
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        if current.state is not JobState.UPLOADING:
            # A receipt for a job that is no longer uploading - typically one
            # cancelled by withdrawal mid-transfer.
            return self.handle_late_receipt(job_id, receipt_id, now=stamp)
        job = self._transition(job_id, JobState.ACKNOWLEDGED, now=stamp, reason="")
        self.db.insert_receipt(job_id, receipt_id, now=stamp, accepted=True)
        if current.content_hash:
            self.db.record_contributed_hash(current.content_hash, now=stamp)
        self.db.add_volume(
            _day(stamp), packages=1, seconds=int(round(current.duration_ms / 1000))
        )
        package = self.store.find(current.session_id, ArtifactKind.PACKAGE)
        if package is not None:
            self.store.delete(package.artifact_id)
        events.emit("queue.acknowledged", job=job_id, receipt=receipt_id)
        return job

    def fail_upload(
        self,
        job_id: str,
        reason: str,
        *,
        retriable: bool,
        now: float | None = None,
    ) -> Job:
        """Handle a transport or server failure.

        A failure never falls back to uploading the original recording: the
        only artifact this queue can send is the package it already built.
        """
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        if not retriable or current.attempts >= self.settings.max_attempts:
            return self.reject(job_id, reason, now=stamp)
        delay = min(
            self.settings.max_backoff_s,
            self.settings.initial_backoff_s * (2 ** max(0, current.attempts - 1)),
        )
        job = self._transition(
            job_id,
            JobState.ELIGIBLE,
            now=stamp,
            reason=reason,
            next_attempt_at=stamp + delay,
        )
        events.emit(
            "queue.upload_retry",
            job=job_id,
            reason=reason,
            attempts=current.attempts,
            delay_s=int(delay),
        )
        return job

    # -- withdrawal and deletion --------------------------------------------

    def cancel(self, job_id: str, reason: str, *, now: float | None = None) -> Job:
        """Cancel one job and delete its payloads."""
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        job = self._transition(job_id, JobState.DELETED, now=stamp, reason=reason)
        # Payloads go; the job row stays in DELETED so a receipt that arrives
        # afterwards is recognised as late rather than unknown.
        self.store.delete_payloads(current.session_id)
        events.emit("queue.cancelled", job=job_id, reason=reason)
        return job

    def on_withdrawal(self, *, now: float | None = None) -> tuple[Job, ...]:
        """Cancel every cancellable job after a withdrawal or pause."""
        stamp = time.time() if now is None else now
        cancelled: list[Job] = []
        for job in self.jobs(*CANCELLABLE_STATES):
            try:
                cancelled.append(self.cancel(job.job_id, "consent_withdrawn", now=stamp))
            except StateTransitionConflict:
                # Another worker finished it first; the next recheck catches it.
                continue
        events.emit("queue.withdrawal_processed", cancelled=len(cancelled))
        return tuple(cancelled)

    def handle_late_receipt(self, job_id: str, receipt_id: str, *, now: float | None = None) -> Job:
        """A receipt arriving after local cancellation.

        The sample does not become eligible again.  The receipt is recorded as
        not accepted and a deletion request is opened for the server copy.
        """
        stamp = time.time() if now is None else now
        current = self._require(job_id)
        self.db.insert_receipt(job_id, receipt_id, now=stamp, accepted=False)
        request_id = self.consent.request_deletion(f"late_receipt:{receipt_id}", now=stamp)
        events.emit(
            "queue.late_receipt",
            job=job_id,
            receipt=receipt_id,
            state=current.state,
            request=request_id,
        )
        return current

    def purge(self, job_id: str) -> None:
        """Remove a terminal job row once its payloads are gone."""
        job = self.job(job_id)
        if job is None:
            return
        if not job.terminal:
            raise StateTransitionConflict(f"job {job_id} is still active ({job.state})")
        self.db.delete_job(job_id)

    # -- internals -----------------------------------------------------------

    def _transition(
        self,
        job_id: str,
        target: JobState,
        *,
        now: float | None = None,
        **fields: object,
    ) -> Job:
        row = self.db.transition_job(
            job_id,
            allowed_from=[str(state) for state in predecessors(target)],
            to_state=str(target),
            now=now,
            **fields,
        )
        return _job(row)

    def _require(self, job_id: str) -> Job:
        job = self.job(job_id)
        if job is None:
            raise StateTransitionConflict(f"unknown job {job_id}")
        return job

    def _volume_gate(self, now: float) -> str:
        packages, seconds = self.db.volume(_day(now))
        if packages >= self.settings.max_packages_per_day:
            return "daily_package_cap_reached"
        if seconds >= self.settings.max_seconds_per_day:
            return "daily_duration_cap_reached"
        return ""


def _idempotency_key(package: UploadPackage) -> str:
    """Stable per-package key so a retry cannot create a second example."""
    return hashlib.sha256(
        f"{package.sample_id}|{package.archive_sha256}".encode("utf-8")
    ).hexdigest()


def _day(now: float) -> str:
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")


def _job(row: object) -> Job:
    return Job(
        job_id=row["job_id"],  # type: ignore[index]
        session_id=row["session_id"],  # type: ignore[index]
        state=JobState(row["state"]),  # type: ignore[index]
        reason=row["reason"],  # type: ignore[index]
        sample_id=row["sample_id"],  # type: ignore[index]
        idempotency_key=row["idempotency_key"],  # type: ignore[index]
        consent_id=row["consent_id"],  # type: ignore[index]
        content_hash=row["content_hash"],  # type: ignore[index]
        duration_ms=row["duration_ms"],  # type: ignore[index]
        attempts=row["attempts"],  # type: ignore[index]
        next_attempt_at=row["next_attempt_at"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
        updated_at=row["updated_at"],  # type: ignore[index]
        expires_at=row["expires_at"],  # type: ignore[index]
    )
