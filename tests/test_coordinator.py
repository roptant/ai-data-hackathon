"""End-to-end sessions through the coordinator (plan sections 2, 5, 6, 9).

The property these tests exist for is the plan's central separation: the user
always gets the full dictation, and the training copy is a separate, optional,
fail-closed path whose outcome never changes what was delivered.
"""

from __future__ import annotations

import time

import pytest

from dictation.api.events import EventBus, EventName
from dictation.api.tokens import Scope, TokenManager
from dictation.asr.mock import FailingAsr, ScriptedAsr, script_from_text
from dictation.capture.coordinator import CaptureCoordinator
from dictation.capture.devices import MemorySource, SilenceSource, synthetic_speech
from dictation.config import ApiSettings, CaptureSettings, Settings
from dictation.consent.consent import ConsentManager
from dictation.errors import ApiError
from dictation.paths import DataPaths
from dictation.platform_.base import FocusTarget
from dictation.platform_.insertion import DeliveryMethod, DeliveryReason, TextDelivery
from dictation.privacy.classifier.mock import FailingClassifier, RuleEchoClassifier
from dictation.store.db import Database
from dictation.store.keys import EphemeralKeyStore
from dictation.store.session_store import ArtifactKind, SessionStore
from dictation.types import Decision
from dictation.upload.queue import UploadQueue
from dictation.upload.states import JobState

from tests.test_platform_and_models import RecordingAdapter, editor

TEXT = (
    "Send the parcel to Jane at 14 Oak Street. "
    "The meeting starts tomorrow and I will bring the printed agenda. "
    "Remind me to water the plants before we leave the office."
)


def build(tmp_path, *, contribute: bool, classifier=None, asr=None, text: str = TEXT):
    paths = DataPaths.create(tmp_path / "data")
    database = Database.open(paths.database)
    store = SessionStore(paths, database, EphemeralKeyStore())
    consent = ConsentManager(database, allow_unvalidated_upload=True)
    queue = UploadQueue(database, store, consent)
    if contribute:
        consent.grant()

    script = script_from_text(text)
    total_ms = script[-1].end_ms + 400
    pcm = synthetic_speech(
        [(word.start_ms / 1000, word.end_ms / 1000) for word in script],
        total_seconds=total_ms / 1000,
    )
    target = editor()
    adapter = RecordingAdapter(target)
    settings = Settings(capture=CaptureSettings(max_session_seconds=60), api=ApiSettings())
    coordinator = CaptureCoordinator(
        settings=settings,
        asr=asr or ScriptedAsr(script),
        delivery=TextDelivery(adapter, settings.insertion),
        adapter=adapter,
        classifier=classifier if classifier is not None else RuleEchoClassifier(),
        bus=EventBus(),
        store=store,
        queue=queue if contribute else None,
        consent=consent if contribute else None,
    )
    return coordinator, adapter, store, queue, consent, pcm


# -- local dictation ---------------------------------------------------------


def test_full_transcript_is_delivered_unredacted(tmp_path) -> None:
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False)
    outcome = coordinator.run_session(MemorySource(pcm))
    assert outcome.error == ""
    assert outcome.delivery is not None
    assert outcome.delivery.method is DeliveryMethod.NATIVE
    delivered = adapter.inserted[-1]
    # Redaction applies to training data, never to the user's own text.
    assert "Jane" in delivered
    assert "Oak Street" in delivered
    assert "water the plants" in delivered


def test_session_returns_to_idle(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.run_session(MemorySource(pcm))
    assert coordinator.status()["state"] == "idle"
    assert coordinator.status()["recording"] is False


def test_status_never_contains_transcript_text(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.run_session(MemorySource(pcm))
    status = coordinator.status()
    assert "text" not in status
    assert "transcript" not in status


def test_a_silent_session_produces_no_text(tmp_path) -> None:
    coordinator, adapter, *_rest = build(tmp_path, contribute=False)
    outcome = coordinator.run_session(SilenceSource(2.0))
    assert outcome.transcript is None
    assert adapter.inserted == []
    assert outcome.training_reason == "silent_session"


def test_recognition_failure_is_reported_not_faked(tmp_path) -> None:
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False, asr=FailingAsr())
    outcome = coordinator.run_session(MemorySource(pcm))
    assert outcome.error
    assert adapter.inserted == []
    assert coordinator.status()["state"] == "idle"


def test_focus_change_defers_to_the_result_panel(tmp_path) -> None:
    """The user switched to a terminal while dictating; nothing is typed there."""
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False)
    original = adapter.current_focus

    def moving_focus() -> FocusTarget:
        # The first call happens at recording start and records the target;
        # afterwards focus has moved to a terminal.
        if not getattr(moving_focus, "called", False):
            moving_focus.called = True  # type: ignore[attr-defined]
            return original()
        return FocusTarget(handle="win-9", application="terminal", editable=True)

    adapter.current_focus = moving_focus  # type: ignore[method-assign]
    outcome = coordinator.run_session(MemorySource(pcm))
    assert outcome.delivery is not None
    assert outcome.delivery.method is DeliveryMethod.RESULT_PANEL
    assert outcome.delivery.reason is DeliveryReason.FOCUS_CHANGED
    assert adapter.inserted == []
    assert coordinator.delivery.panel.entries


def test_cancellation_delivers_nothing(tmp_path) -> None:
    coordinator, adapter, store, *_rest, pcm = build(tmp_path, contribute=False)
    session_id = coordinator.start_session(source="test")
    assert coordinator.cancel_session(session_id)
    assert adapter.inserted == []
    assert coordinator.status()["state"] == "idle"


def test_a_second_session_is_refused_while_one_is_active(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.source_factory = lambda: MemorySource(pcm)
    session_id = coordinator.start_session(source="test")
    with pytest.raises(ApiError):
        coordinator.start_session(source="test")
    coordinator.stop_session(session_id)


def test_stop_for_an_unknown_session_is_refused(tmp_path) -> None:
    coordinator, *_rest = build(tmp_path, contribute=False)
    assert coordinator.stop_session("session-unknown") is False
    assert coordinator.cancel_session("session-unknown") is False


def test_events_are_published_for_the_session_lifecycle(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    tokens = TokenManager(Database.in_memory())
    record, _ = tokens.issue(
        "client", frozenset({Scope.STATUS_READ, Scope.TRANSCRIPT_FINAL})
    )
    subscriber = coordinator.bus.subscribe("sub", record)
    coordinator.run_session(MemorySource(pcm))
    names = [event.event for event in subscriber.drain()]
    assert EventName.SESSION_STARTED in names
    assert EventName.TRANSCRIPT_FINAL in names
    assert EventName.SESSION_STOPPED in names


def test_caption_scope_does_not_receive_the_final_transcript(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    tokens = TokenManager(Database.in_memory())
    record, _ = tokens.issue("captions", frozenset({Scope.TRANSCRIPT_LIVE}))
    subscriber = coordinator.bus.subscribe("sub", record)
    coordinator.run_session(MemorySource(pcm))
    assert all(
        event.event is not EventName.TRANSCRIPT_FINAL for event in subscriber.drain()
    )


# -- contribution off --------------------------------------------------------


def test_with_contribution_off_nothing_is_stored_or_queued(tmp_path) -> None:
    coordinator, _adapter, store, queue, _consent, pcm = build(tmp_path, contribute=False)
    outcome = coordinator.run_session(MemorySource(pcm))
    assert outcome.training is None
    assert outcome.training_reason == "contribution_off"
    assert queue.jobs() == ()
    assert store.db.query("SELECT * FROM artifacts") == []


# -- contribution on ---------------------------------------------------------


def test_opted_in_session_builds_a_package(tmp_path) -> None:
    coordinator, adapter, store, queue, consent, pcm = build(tmp_path, contribute=True)
    outcome = coordinator.run_session(MemorySource(pcm))
    assert outcome.training is not None
    assert outcome.training.decision is Decision.ELIGIBLE, outcome.training.reason

    # The user still received everything.
    assert "Jane" in adapter.inserted[-1]

    # The training copy did not.
    retained = outcome.training.spliced_text().lower()
    assert "jane" not in retained
    assert "oak" not in retained

    jobs = queue.jobs()
    assert len(jobs) == 1
    assert jobs[0].state is JobState.ELIGIBLE
    kinds = {row["kind"] for row in store.db.artifacts_of(outcome.session_id)}
    assert kinds == {str(ArtifactKind.PACKAGE)}


def test_privacy_worker_failure_leaves_dictation_intact(tmp_path) -> None:
    coordinator, adapter, store, queue, _consent, pcm = build(
        tmp_path, contribute=True, classifier=FailingClassifier()
    )
    outcome = coordinator.run_session(MemorySource(pcm))
    # Delivery happened.
    assert "Jane" in adapter.inserted[-1]
    # Nothing is upload-ready.
    jobs = queue.jobs()
    assert len(jobs) == 1
    assert jobs[0].state is JobState.REJECTED
    assert store.db.artifacts_of(outcome.session_id) == []


def test_per_session_opt_out_skips_the_training_copy(tmp_path) -> None:
    coordinator, _adapter, store, queue, _consent, pcm = build(tmp_path, contribute=True)
    outcome = coordinator.run_session(MemorySource(pcm), opted_out=True)
    assert outcome.training is None
    assert queue.jobs() == ()


def test_withdrawal_before_the_session_prevents_any_queueing(tmp_path) -> None:
    coordinator, _adapter, store, queue, consent, pcm = build(tmp_path, contribute=True)
    consent.withdraw()
    outcome = coordinator.run_session(MemorySource(pcm))
    assert queue.jobs() == ()
    assert outcome.training is None


def test_duplicate_session_is_not_queued_twice(tmp_path) -> None:
    coordinator, _adapter, store, queue, _consent, pcm = build(tmp_path, contribute=True)
    first = coordinator.run_session(MemorySource(pcm))
    assert first.training is not None and first.training.eligible
    store.db.record_contributed_hash(first.training.content_hash, now=time.time())
    second = coordinator.run_session(MemorySource(pcm))
    assert second.training is not None
    assert second.training.decision is Decision.REJECTED
    assert second.training.reason == "duplicate_example"


# -- shortcut entry points ---------------------------------------------------


def test_hold_to_record_through_the_shortcut_path(tmp_path) -> None:
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.source_factory = lambda: MemorySource(pcm)
    session_id = coordinator.on_hold_down()
    assert session_id
    assert coordinator.status()["state"] == "recording_held"
    coordinator.on_hold_up()
    assert coordinator.status()["state"] == "idle"
    assert adapter.inserted


def test_lock_keeps_recording_after_the_key_is_released(tmp_path) -> None:
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.source_factory = lambda: MemorySource(pcm)
    coordinator.on_hold_down()
    coordinator.on_lock()
    assert coordinator.status()["locked"] is True
    coordinator.on_hold_up()
    assert coordinator.status()["recording"] is True
    coordinator.on_toggle()
    assert coordinator.status()["state"] == "idle"
    assert adapter.inserted


def test_auto_repeat_does_not_start_a_second_session(tmp_path) -> None:
    coordinator, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.source_factory = lambda: MemorySource(pcm)
    first = coordinator.on_hold_down()
    assert coordinator.on_hold_down() is None
    assert coordinator.status()["session_id"] == first
    coordinator.on_hold_up()


def test_cancel_through_the_shortcut_discards_the_session(tmp_path) -> None:
    coordinator, adapter, store, *_rest, pcm = build(tmp_path, contribute=True)
    coordinator.source_factory = lambda: MemorySource(pcm)
    coordinator.on_hold_down()
    coordinator.on_cancel()
    assert coordinator.status()["state"] == "idle"
    assert adapter.inserted == []
    assert store.db.query("SELECT * FROM artifacts") == []


def test_duration_limit_finalizes_a_forgotten_recording(tmp_path) -> None:
    coordinator, adapter, *_rest, pcm = build(tmp_path, contribute=False)
    coordinator.source_factory = lambda: MemorySource(pcm)
    coordinator.machine.max_session_seconds = 0  # the limit is already past
    coordinator.machine.warn_before_limit_seconds = 0
    coordinator.on_hold_down()
    deadline = time.monotonic() + 5
    while coordinator.status()["state"] != "idle" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert coordinator.status()["state"] == "idle"
    # The audio captured before the limit is still delivered, not dropped.
    assert adapter.inserted
