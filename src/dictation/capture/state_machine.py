"""Recording state machine (plan section 5).

The plan's diagram plus its prose constraints, in one auditable place:

* key auto-repeat is ignored,
* releasing the held key after ``lock`` does not stop recording,
* ``stop`` and ``cancel`` are idempotent,
* text is never inserted after a cancellation or an ambiguous recovery,
* a missed key-up cannot record forever: the duration limit finalises.

The machine is pure: it owns no audio device and performs no I/O.  It returns
the effects the coordinator must carry out, which is what makes the "never
insert after cancel" rule testable rather than a review comment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from dictation.errors import InvalidTransition


class State(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING_HELD = "recording_held"
    RECORDING_LOCKED = "recording_locked"
    FINALIZING = "finalizing"
    DELIVERING = "delivering"
    CANCELLED = "cancelled"
    ERROR = "error"


ACTIVE_STATES = frozenset(
    {State.STARTING, State.RECORDING_HELD, State.RECORDING_LOCKED, State.FINALIZING, State.DELIVERING}
)
RECORDING_STATES = frozenset({State.RECORDING_HELD, State.RECORDING_LOCKED})


class Event(StrEnum):
    """Inputs.  ``HOLD_DOWN``/``HOLD_UP`` come from the shortcut adapter,
    ``CAPTURE_STARTED``/``FINALIZED``/``DELIVERED`` from the coordinator."""

    HOLD_DOWN = "hold_down"
    HOLD_UP = "hold_up"
    TOGGLE = "toggle"
    LOCK = "lock"
    STOP = "stop"
    CANCEL = "cancel"
    CAPTURE_STARTED = "capture_started"
    FINALIZED = "finalized"
    DELIVERED = "delivered"
    FAILED = "failed"
    DURATION_LIMIT = "duration_limit"
    RESET = "reset"


class Effect(StrEnum):
    """What the coordinator must do as a result of a transition."""

    START_CAPTURE = "start_capture"
    STOP_CAPTURE = "stop_capture"
    FINALIZE_TRANSCRIPT = "finalize_transcript"
    DELIVER_TEXT = "deliver_text"
    DISCARD_SESSION = "discard_session"
    ANNOUNCE_STATE = "announce_state"
    WARN_DURATION_LIMIT = "warn_duration_limit"
    RELEASE_RESOURCES = "release_resources"


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of feeding one event."""

    accepted: bool
    state: State
    effects: tuple[Effect, ...] = ()
    ignored_reason: str = ""

    @property
    def ignored(self) -> bool:
        return not self.accepted


@dataclass(slots=True)
class RecordingStateMachine:
    """Serialised recording state.

    ``max_session_seconds`` is provisional (plan section 5).  The coordinator is
    responsible for calling :meth:`tick`; the machine decides when the warning
    and the forced finalisation happen.
    """

    max_session_seconds: int = 600
    warn_before_limit_seconds: int = 30
    state: State = State.IDLE
    session_id: str = ""
    started_at: float | None = None
    #: Set when the user cancelled anywhere in the session.  Once true, no
    #: delivery effect is ever produced for this session again.
    cancel_requested: bool = False
    #: Recording mode chosen before capture confirmed it started.
    _pending_locked: bool = False
    _pending_stop: bool = False
    _warned: bool = False
    history: list[tuple[float, State, Event, State]] = field(default_factory=list)

    # -- queries -------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self.state in RECORDING_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def insertion_permitted(self) -> bool:
        """Delivery is allowed only for a session that was never cancelled."""
        return not self.cancel_requested

    def snapshot(self) -> dict[str, object]:
        """Coarse state for the local API and the indicator.  No transcript."""
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = max(0.0, time.monotonic() - self.started_at)
        return {
            "state": str(self.state),
            "recording": self.is_recording,
            "locked": self.state is State.RECORDING_LOCKED,
            "elapsed_ms": int(elapsed * 1000),
            "session_id": self.session_id,
            "max_session_seconds": self.max_session_seconds,
        }

    # -- driving -------------------------------------------------------------

    def handle(self, event: Event, *, session_id: str = "", now: float | None = None) -> Outcome:
        """Feed one event.

        Returns an :class:`Outcome`.  Events that are deliberately ignored
        (auto-repeat, redundant stop, key-up while locked) return
        ``accepted=False`` with a reason code instead of raising; genuinely
        invalid commands raise :class:`~dictation.errors.InvalidTransition`.
        """
        before = self.state
        outcome = self._dispatch(event, session_id=session_id, now=now)
        if outcome.accepted:
            self.history.append((now or time.monotonic(), before, event, self.state))
        return outcome

    def _dispatch(self, event: Event, *, session_id: str, now: float | None) -> Outcome:
        clock = now if now is not None else time.monotonic()
        match event:
            case Event.HOLD_DOWN:
                return self._start(locked=False, session_id=session_id, clock=clock, source=event)
            case Event.TOGGLE:
                return self._toggle(session_id=session_id, clock=clock)
            case Event.HOLD_UP:
                return self._hold_up()
            case Event.LOCK:
                return self._lock()
            case Event.STOP:
                return self._stop()
            case Event.CANCEL:
                return self._cancel()
            case Event.CAPTURE_STARTED:
                return self._capture_started()
            case Event.FINALIZED:
                return self._finalized()
            case Event.DELIVERED:
                return self._delivered()
            case Event.FAILED:
                return self._failed()
            case Event.DURATION_LIMIT:
                return self._duration_limit()
            case Event.RESET:
                return self._reset()
        raise InvalidTransition(self.state, event)  # pragma: no cover - exhaustive match

    # -- individual events ---------------------------------------------------

    def _start(self, *, locked: bool, session_id: str, clock: float, source: Event) -> Outcome:
        if self.state in (State.STARTING, State.RECORDING_HELD, State.RECORDING_LOCKED):
            # Key auto-repeat, or a second activation while already recording.
            return Outcome(False, self.state, ignored_reason="auto_repeat")
        if self.state in (State.FINALIZING, State.DELIVERING):
            # Serialise conflicting commands: the previous session must finish
            # before a new one starts, otherwise two sessions share a target.
            return Outcome(False, self.state, ignored_reason="busy_finalizing")
        if self.state is not State.IDLE:
            raise InvalidTransition(self.state, source)
        self.state = State.STARTING
        self.session_id = session_id
        self.started_at = clock
        self.cancel_requested = False
        self._pending_locked = locked
        self._pending_stop = False
        self._warned = False
        return Outcome(True, self.state, (Effect.START_CAPTURE, Effect.ANNOUNCE_STATE))

    def _toggle(self, *, session_id: str, clock: float) -> Outcome:
        if self.state is State.IDLE:
            return self._start(locked=True, session_id=session_id, clock=clock, source=Event.TOGGLE)
        if self.state is State.STARTING:
            # Stop requested before capture confirmed; honour it once started.
            self._pending_stop = True
            return Outcome(True, self.state, ignored_reason="")
        if self.state in RECORDING_STATES:
            return self._begin_finalize()
        return Outcome(False, self.state, ignored_reason="not_recording")

    def _hold_up(self) -> Outcome:
        if self.state is State.RECORDING_HELD:
            return self._begin_finalize()
        if self.state is State.RECORDING_LOCKED:
            # Locked: releasing the held key must not stop the recording.
            return Outcome(False, self.state, ignored_reason="locked")
        if self.state is State.STARTING and not self._pending_locked:
            self._pending_stop = True
            return Outcome(True, self.state)
        return Outcome(False, self.state, ignored_reason="not_held")

    def _lock(self) -> Outcome:
        if self.state is State.RECORDING_HELD:
            self.state = State.RECORDING_LOCKED
            return Outcome(True, self.state, (Effect.ANNOUNCE_STATE,))
        if self.state is State.STARTING:
            self._pending_locked = True
            return Outcome(True, self.state)
        if self.state is State.RECORDING_LOCKED:
            return Outcome(False, self.state, ignored_reason="already_locked")
        return Outcome(False, self.state, ignored_reason="not_recording")

    def _stop(self) -> Outcome:
        if self.state in RECORDING_STATES:
            return self._begin_finalize()
        if self.state is State.STARTING:
            self._pending_stop = True
            return Outcome(True, self.state)
        # Idempotent: stopping something already stopping or stopped is a no-op.
        return Outcome(False, self.state, ignored_reason="already_stopping")

    def _cancel(self) -> Outcome:
        if self.state is State.IDLE or self.state is State.CANCELLED:
            return Outcome(False, self.state, ignored_reason="nothing_to_cancel")
        self.cancel_requested = True
        self.state = State.CANCELLED
        return Outcome(
            True,
            self.state,
            (Effect.STOP_CAPTURE, Effect.DISCARD_SESSION, Effect.ANNOUNCE_STATE, Effect.RELEASE_RESOURCES),
        )

    def _capture_started(self) -> Outcome:
        if self.state is not State.STARTING:
            return Outcome(False, self.state, ignored_reason="not_starting")
        self.state = State.RECORDING_LOCKED if self._pending_locked else State.RECORDING_HELD
        if self._pending_stop:
            self._pending_stop = False
            return self._begin_finalize()
        return Outcome(True, self.state, (Effect.ANNOUNCE_STATE,))

    def _begin_finalize(self) -> Outcome:
        self.state = State.FINALIZING
        return Outcome(
            True,
            self.state,
            (Effect.STOP_CAPTURE, Effect.FINALIZE_TRANSCRIPT, Effect.ANNOUNCE_STATE),
        )

    def _finalized(self) -> Outcome:
        if self.state is not State.FINALIZING:
            # A finalisation completing after cancellation must not deliver.
            return Outcome(False, self.state, ignored_reason="not_finalizing")
        if self.cancel_requested:
            self.state = State.CANCELLED
            return Outcome(True, self.state, (Effect.DISCARD_SESSION,))
        self.state = State.DELIVERING
        return Outcome(True, self.state, (Effect.DELIVER_TEXT,))

    def _delivered(self) -> Outcome:
        if self.state is not State.DELIVERING:
            return Outcome(False, self.state, ignored_reason="not_delivering")
        self.state = State.IDLE
        self.session_id = ""
        self.started_at = None
        return Outcome(True, self.state, (Effect.ANNOUNCE_STATE, Effect.RELEASE_RESOURCES))

    def _failed(self) -> Outcome:
        if self.state is State.IDLE:
            return Outcome(False, self.state, ignored_reason="idle")
        self.state = State.ERROR
        return Outcome(
            True,
            self.state,
            (Effect.STOP_CAPTURE, Effect.DISCARD_SESSION, Effect.ANNOUNCE_STATE, Effect.RELEASE_RESOURCES),
        )

    def _duration_limit(self) -> Outcome:
        if self.state not in RECORDING_STATES:
            return Outcome(False, self.state, ignored_reason="not_recording")
        return self._begin_finalize()

    def _reset(self) -> Outcome:
        if self.state in (State.CANCELLED, State.ERROR):
            self.state = State.IDLE
            self.session_id = ""
            self.started_at = None
            self.cancel_requested = False
            self._pending_locked = False
            self._pending_stop = False
            self._warned = False
            return Outcome(True, self.state, (Effect.ANNOUNCE_STATE,))
        if self.state is State.IDLE:
            return Outcome(False, self.state, ignored_reason="already_idle")
        raise InvalidTransition(self.state, Event.RESET)

    # -- time ----------------------------------------------------------------

    def tick(self, now: float | None = None) -> Outcome:
        """Enforce the session duration limit.

        A missed key-up must not record indefinitely: the machine warns first
        and then finalises (never silently drops) the session.
        """
        if self.state not in RECORDING_STATES or self.started_at is None:
            return Outcome(False, self.state, ignored_reason="not_recording")
        clock = now if now is not None else time.monotonic()
        elapsed = clock - self.started_at
        if elapsed >= self.max_session_seconds:
            return self.handle(Event.DURATION_LIMIT, now=clock)
        if not self._warned and elapsed >= self.max_session_seconds - self.warn_before_limit_seconds:
            self._warned = True
            return Outcome(True, self.state, (Effect.WARN_DURATION_LIMIT,))
        return Outcome(False, self.state, ignored_reason="within_limit")
