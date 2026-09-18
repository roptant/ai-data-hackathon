"""Recording state machine: the constraints of plan section 5."""

from __future__ import annotations

import pytest

from dictation.capture.state_machine import (
    Effect,
    Event,
    RecordingStateMachine,
    State,
)
from dictation.errors import InvalidTransition


def started(**kwargs) -> RecordingStateMachine:
    machine = RecordingStateMachine(**kwargs)
    machine.handle(Event.HOLD_DOWN, session_id="s1", now=0.0)
    machine.handle(Event.CAPTURE_STARTED, now=0.1)
    return machine


def test_hold_to_record_round_trip() -> None:
    machine = started()
    assert machine.state is State.RECORDING_HELD
    outcome = machine.handle(Event.HOLD_UP)
    assert machine.state is State.FINALIZING
    assert Effect.FINALIZE_TRANSCRIPT in outcome.effects
    machine.handle(Event.FINALIZED)
    assert machine.state is State.DELIVERING
    machine.handle(Event.DELIVERED)
    assert machine.state is State.IDLE


def test_toggle_starts_locked() -> None:
    machine = RecordingStateMachine()
    machine.handle(Event.TOGGLE, session_id="s1", now=0.0)
    machine.handle(Event.CAPTURE_STARTED)
    assert machine.state is State.RECORDING_LOCKED
    machine.handle(Event.TOGGLE)
    assert machine.state is State.FINALIZING


def test_key_auto_repeat_is_ignored() -> None:
    machine = started()
    outcome = machine.handle(Event.HOLD_DOWN, session_id="s2")
    assert outcome.ignored
    assert outcome.ignored_reason == "auto_repeat"
    assert machine.session_id == "s1"


def test_lock_survives_key_release() -> None:
    machine = started()
    machine.handle(Event.LOCK)
    assert machine.state is State.RECORDING_LOCKED
    outcome = machine.handle(Event.HOLD_UP)
    assert outcome.ignored
    assert outcome.ignored_reason == "locked"
    assert machine.state is State.RECORDING_LOCKED
    machine.handle(Event.TOGGLE)
    assert machine.state is State.FINALIZING


def test_lock_before_capture_confirms_applies_on_start() -> None:
    machine = RecordingStateMachine()
    machine.handle(Event.HOLD_DOWN, session_id="s1")
    machine.handle(Event.LOCK)
    machine.handle(Event.CAPTURE_STARTED)
    assert machine.state is State.RECORDING_LOCKED


def test_release_before_capture_confirms_still_finalizes() -> None:
    machine = RecordingStateMachine()
    machine.handle(Event.HOLD_DOWN, session_id="s1")
    machine.handle(Event.HOLD_UP)
    machine.handle(Event.CAPTURE_STARTED)
    assert machine.state is State.FINALIZING


def test_stop_is_idempotent() -> None:
    machine = started()
    first = machine.handle(Event.STOP)
    second = machine.handle(Event.STOP)
    third = machine.handle(Event.STOP)
    assert first.accepted
    assert second.ignored and third.ignored
    assert machine.state is State.FINALIZING


def test_cancel_is_idempotent_and_blocks_delivery() -> None:
    machine = started()
    machine.handle(Event.CANCEL)
    assert machine.state is State.CANCELLED
    assert machine.handle(Event.CANCEL).ignored
    assert not machine.insertion_permitted


def test_cancel_during_finalization_prevents_insertion() -> None:
    machine = started()
    machine.handle(Event.HOLD_UP)
    cancelled = machine.handle(Event.CANCEL)
    assert machine.state is State.CANCELLED
    assert Effect.DISCARD_SESSION in cancelled.effects
    # The finalisation that was already running completes afterwards; it must
    # not deliver anything.
    late = machine.handle(Event.FINALIZED)
    assert late.ignored
    assert Effect.DELIVER_TEXT not in late.effects
    assert machine.state is State.CANCELLED


def test_finalized_after_cancel_never_delivers() -> None:
    machine = started()
    machine.handle(Event.CANCEL)
    machine.handle(Event.RESET)
    outcome = machine.handle(Event.FINALIZED)
    assert outcome.ignored
    assert machine.state is State.IDLE


def test_new_session_is_refused_while_finalizing() -> None:
    machine = started()
    machine.handle(Event.HOLD_UP)
    outcome = machine.handle(Event.HOLD_DOWN, session_id="s2")
    assert outcome.ignored
    assert outcome.ignored_reason == "busy_finalizing"


def test_duration_limit_finalizes_rather_than_recording_forever() -> None:
    machine = started(max_session_seconds=600, warn_before_limit_seconds=30)
    warning = machine.tick(now=575.0)
    assert Effect.WARN_DURATION_LIMIT in warning.effects
    limit = machine.tick(now=601.0)
    assert Effect.FINALIZE_TRANSCRIPT in limit.effects
    assert machine.state is State.FINALIZING


def test_tick_before_the_limit_does_nothing() -> None:
    machine = started(max_session_seconds=600)
    assert machine.tick(now=10.0).ignored
    assert machine.state is State.RECORDING_HELD


def test_failure_releases_resources_and_resets() -> None:
    machine = started()
    outcome = machine.handle(Event.FAILED)
    assert machine.state is State.ERROR
    assert Effect.DISCARD_SESSION in outcome.effects
    assert Effect.RELEASE_RESOURCES in outcome.effects
    machine.handle(Event.RESET)
    assert machine.state is State.IDLE


def test_reset_from_recording_is_invalid() -> None:
    machine = started()
    with pytest.raises(InvalidTransition):
        machine.handle(Event.RESET)


def test_snapshot_has_no_transcript_fields() -> None:
    machine = started()
    snapshot = machine.snapshot()
    assert set(snapshot) == {
        "state",
        "recording",
        "locked",
        "elapsed_ms",
        "session_id",
        "max_session_seconds",
    }
    assert snapshot["recording"] is True


def test_history_records_accepted_transitions_only() -> None:
    machine = started()
    machine.handle(Event.HOLD_DOWN)  # ignored auto-repeat
    transitions = [(before, event, after) for _, before, event, after in machine.history]
    assert (State.IDLE, Event.HOLD_DOWN, State.STARTING) in transitions
    assert sum(1 for _, event, _ in transitions if event is Event.HOLD_DOWN) == 1
