"""Consent, the upload queue and the withdrawal races (plan section 9)."""

from __future__ import annotations

import time

import pytest

from dictation.config import UploadSettings
from dictation.consent.consent import ConsentManager, Purpose
from dictation.dataset.builder import build_dataset
from dictation.dataset.package import build_package
from dictation.errors import ConsentError
from dictation.paths import DataPaths
from dictation.privacy.classifier.mock import RuleEchoClassifier
from dictation.privacy.pipeline import analyze
from dictation.server.app import IngestServer
from dictation.server.tenants import TenantRegistry
from dictation.store.db import Database
from dictation.store.keys import EphemeralKeyStore
from dictation.store.session_store import ArtifactKind, SessionStore
from dictation.upload.queue import UploadQueue
from dictation.upload.states import JobState, may_transition, predecessors
from dictation.upload.transport import (
    FlakyTransport,
    LoopbackTransport,
    Receipt,
    RejectingTransport,
)
from dictation.upload.worker import UploadWorker
from dictation.version import CONSENT_VERSION

from tests.conftest import Session


@pytest.fixture
def stack(tmp_path):
    """Store, consent and queue over a temporary payload root."""
    paths = DataPaths.create(tmp_path / "data")
    database = Database.open(paths.database)
    store = SessionStore(paths, database, EphemeralKeyStore())
    # allow_unvalidated_upload mirrors the isolated non-production server of
    # plan milestone 4; production refuses while the gates are unmet.
    consent = ConsentManager(database, allow_unvalidated_upload=True)
    queue = UploadQueue(
        database, store, consent, settings=UploadSettings(max_packages_per_day=5, max_seconds_per_day=600)
    )
    return store, consent, queue


def eligible_package(session: Session):
    analysis = analyze(session.transcript, RuleEchoClassifier())
    result = build_dataset(analysis, session.pcm)
    assert result.eligible, result.reason
    package = build_package(
        result, consent_version=CONSENT_VERSION, consent_reference="ref"
    )
    return result, package


# -- consent -----------------------------------------------------------------


def test_contribution_is_off_by_default(stack) -> None:
    _, consent, _ = stack
    assert consent.active() is None
    decision = consent.check_session(session_started_at=time.time())
    assert not decision
    assert decision.reason == "no_active_consent"


def test_disclosure_states_what_is_uploaded_and_that_voice_is_identifiable(stack) -> None:
    _, consent, _ = stack
    disclosure = consent.disclosure()
    assert "identifiable" in disclosure["identifiability"]
    assert disclosure["automatic"]
    assert disclosure["withdrawal"]
    assert disclosure["local_only_alternative"]


def test_grant_then_session_is_eligible(stack) -> None:
    _, consent, _ = stack
    consent.grant()
    assert consent.check_session(session_started_at=time.time())


def test_shared_model_training_cannot_be_granted(stack) -> None:
    _, consent, _ = stack
    with pytest.raises(ConsentError):
        consent.grant((Purpose.SHARED_MODEL_TRAINING,))


def test_earlier_sessions_are_not_swept_up_retroactively(stack) -> None:
    _, consent, _ = stack
    record = consent.grant()
    decision = consent.check_session(session_started_at=record.granted_at - 60)
    assert not decision
    assert decision.reason == "session_predates_consent"


def test_per_session_opt_out_is_honoured(stack) -> None:
    _, consent, _ = stack
    consent.grant()
    decision = consent.check_session(session_started_at=time.time(), session_opted_out=True)
    assert decision.reason == "session_opted_out"


def test_pause_blocks_eligibility(stack) -> None:
    _, consent, _ = stack
    consent.grant()
    consent.pause()
    assert consent.check_session(session_started_at=time.time()).reason == "consent_paused"
    consent.resume()
    assert consent.check_session(session_started_at=time.time())


def test_expired_consent_is_not_usable(stack) -> None:
    _, consent, _ = stack
    record = consent.grant(days=1)
    later = record.granted_at + 2 * 86400
    assert consent.recheck(record.consent_id, now=later).reason == "consent_expired"


def test_withdrawal_opens_a_deletion_request(stack) -> None:
    _, consent, _ = stack
    consent.grant()
    request = consent.withdraw()
    assert request.startswith("deletion-")
    assert consent.active() is None


def test_unvalidated_gates_refuse_eligibility_in_production_mode(tmp_path) -> None:
    database = Database.in_memory()
    consent = ConsentManager(database)  # gates not overridden
    consent.grant()
    decision = consent.check_session(session_started_at=time.time())
    assert not decision
    assert decision.reason == "upload_gates_not_met"


# -- state table -------------------------------------------------------------


def test_transition_table_matches_the_plan_diagram() -> None:
    assert may_transition(JobState.LOCAL_PENDING, JobState.ANALYZING)
    assert may_transition(JobState.ANALYZING, JobState.BUILDING)
    assert may_transition(JobState.BUILDING, JobState.ELIGIBLE)
    assert may_transition(JobState.ELIGIBLE, JobState.UPLOADING)
    assert may_transition(JobState.UPLOADING, JobState.ACKNOWLEDGED)
    assert may_transition(JobState.UPLOADING, JobState.ELIGIBLE)  # retry
    for state in JobState:
        if state is not JobState.DELETED:
            assert may_transition(state, JobState.DELETED)
    assert not may_transition(JobState.ACKNOWLEDGED, JobState.UPLOADING)
    assert not may_transition(JobState.LOCAL_PENDING, JobState.ELIGIBLE)
    assert JobState.BUILDING in predecessors(JobState.ELIGIBLE)


# -- queue -------------------------------------------------------------------


def test_enqueue_is_refused_without_consent(stack, parcel_session: Session) -> None:
    _, _, queue = stack
    result = queue.enqueue("s1", session_started_at=time.time())
    assert not result.accepted
    assert result.reason == "no_active_consent"


def test_full_happy_path(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(parcel_session)
    job = queue.enqueue(parcel_session.transcript.session_id, session_started_at=time.time()).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    current = queue.job(job.job_id)
    assert current is not None and current.state is JobState.ELIGIBLE
    assert current.idempotency_key
    assert store.find(parcel_session.transcript.session_id, ArtifactKind.PACKAGE) is not None


def test_raw_audio_is_deleted_once_the_package_exists(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    session_id = parcel_session.transcript.session_id
    result, package = eligible_package(parcel_session)
    store.put(session_id, ArtifactKind.RAW_AUDIO, parcel_session.pcm, expires_at=time.time() + 60)
    job = queue.enqueue(session_id, session_started_at=time.time()).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    kinds = {row["kind"] for row in store.db.artifacts_of(session_id)}
    assert kinds == {str(ArtifactKind.PACKAGE)}


def test_rejection_deletes_everything(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    session_id = parcel_session.transcript.session_id
    store.put(session_id, ArtifactKind.RAW_AUDIO, b"audio", expires_at=time.time() + 60)
    job = queue.enqueue(session_id, session_started_at=time.time()).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.reject(job.job_id, "uncertain_analysis")
    assert store.db.artifacts_of(session_id) == []
    assert queue.job(job.job_id).state is JobState.REJECTED


def test_withdrawal_before_eligibility_leaves_no_package(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(parcel_session)
    session_id = parcel_session.transcript.session_id
    job = queue.enqueue(session_id, session_started_at=time.time()).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    consent.withdraw()
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    assert queue.job(job.job_id).state is JobState.REJECTED
    assert store.db.artifacts_of(session_id) == []


def test_withdrawal_cancels_queued_and_in_flight_jobs(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(parcel_session)
    job = queue.enqueue(
        parcel_session.transcript.session_id, session_started_at=time.time()
    ).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    queue.claim_for_upload(job.job_id)
    consent.withdraw()
    cancelled = queue.on_withdrawal()
    assert [c.job_id for c in cancelled] == [job.job_id]
    assert queue.job(job.job_id).state is JobState.DELETED


def test_late_receipt_does_not_make_a_cancelled_sample_eligible(stack, parcel_session) -> None:
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(parcel_session)
    job = queue.enqueue(
        parcel_session.transcript.session_id, session_started_at=time.time()
    ).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    queue.claim_for_upload(job.job_id)
    consent.withdraw()
    queue.on_withdrawal()
    queue.acknowledge(job.job_id, "receipt-late")
    current = queue.job(job.job_id)
    assert current.state is JobState.DELETED
    receipt = store.db.receipt(job.job_id)
    assert receipt is not None and receipt["accepted"] == 0
    assert store.db.pending_deletions()


def test_retry_backs_off_and_eventually_rejects(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(parcel_session)
    job = queue.enqueue(
        parcel_session.transcript.session_id, session_started_at=time.time()
    ).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    now = time.time()
    for _ in range(queue.settings.max_attempts):
        queue.claim_for_upload(job.job_id, now=now)
        queue.fail_upload(job.job_id, "transport_error", retriable=True, now=now)
    assert queue.job(job.job_id).state is JobState.REJECTED


def test_daily_cap_refuses_new_jobs(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    consent.grant()
    from dictation.upload.queue import _day

    store.db.add_volume(_day(time.time()), packages=5, seconds=0)
    result = queue.enqueue("s-capped", session_started_at=time.time())
    assert result.reason == "daily_package_cap_reached"


# -- worker ------------------------------------------------------------------


@pytest.fixture
def server(tmp_path) -> tuple[IngestServer, str]:
    root = tmp_path / "server"
    registry = TenantRegistry.load(root)
    _, token = registry.create("Test customer", tenant_id="tenant-test")
    return IngestServer(root, registry), token


def prepared_job(stack, session: Session):
    store, consent, queue = stack
    consent.grant()
    result, package = eligible_package(session)
    job = queue.enqueue(session.transcript.session_id, session_started_at=time.time()).job
    assert job is not None
    queue.start_analysis(job.job_id)
    queue.start_build(job.job_id)
    queue.mark_eligible(job.job_id, package, content_hash=result.content_hash)
    return job.job_id


def test_worker_uploads_and_acknowledges(stack, parcel_session: Session, server) -> None:
    store, consent, queue = stack
    ingest, token = server
    job_id = prepared_job(stack, parcel_session)
    worker = UploadWorker(
        queue, LoopbackTransport(ingest, token), store.package_reader(), consent
    )
    report = worker.run_once()
    assert report.acknowledged == 1
    assert queue.job(job_id).state is JobState.ACKNOWLEDGED
    assert store.db.artifacts_of(parcel_session.transcript.session_id) == []


def test_worker_retries_a_transport_failure(stack, parcel_session: Session, server) -> None:
    store, consent, queue = stack
    ingest, token = server
    job_id = prepared_job(stack, parcel_session)
    transport = FlakyTransport(LoopbackTransport(ingest, token), failures=1)
    worker = UploadWorker(queue, transport, store.package_reader(), consent)
    first = worker.run_once(now=time.time())
    assert first.retried == 1
    job = queue.job(job_id)
    assert job.state is JobState.ELIGIBLE
    second = worker.run_once(now=job.next_attempt_at)
    assert second.acknowledged == 1


def test_server_rejection_deletes_and_never_falls_back_to_raw(stack, parcel_session) -> None:
    store, consent, queue = stack
    job_id = prepared_job(stack, parcel_session)
    worker = UploadWorker(queue, RejectingTransport(), store.package_reader(), consent)
    report = worker.run_once()
    assert report.rejected == 1
    assert queue.job(job_id).state is JobState.REJECTED
    assert store.db.artifacts_of(parcel_session.transcript.session_id) == []


def test_worker_refuses_when_consent_vanishes_before_transfer(stack, parcel_session, server) -> None:
    store, consent, queue = stack
    ingest, token = server
    job_id = prepared_job(stack, parcel_session)
    consent.withdraw()
    worker = UploadWorker(queue, LoopbackTransport(ingest, token), store.package_reader(), consent)
    report = worker.run_once()
    assert report.acknowledged == 0
    assert queue.job(job_id).state is JobState.DELETED


def test_idempotency_key_is_stable_for_the_same_package(stack, parcel_session: Session) -> None:
    from dictation.upload.queue import _idempotency_key

    _, package = eligible_package(parcel_session)
    assert _idempotency_key(package) == _idempotency_key(package)


def test_worker_reports_a_refused_receipt_as_rejection(stack, parcel_session: Session) -> None:
    store, consent, queue = stack
    job_id = prepared_job(stack, parcel_session)

    class Refusing:
        def upload(self, archive: bytes, **kwargs: object) -> Receipt:
            return Receipt(receipt_id="r", accepted=False, reason="policy")

    worker = UploadWorker(queue, Refusing(), store.package_reader(), consent)
    report = worker.run_once()
    assert report.rejected == 1
    assert queue.job(job_id).state is JobState.REJECTED
